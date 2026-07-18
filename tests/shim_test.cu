// Validates ds4_rocm_wmma_gfx906.cuh on real gfx906 hardware.
// Mimics the ds4 MoE tile kernel pattern: blockDim=256, wave=tid>>5 (8 software
// wave32 groups on wave64 hardware), rocwmma fragment load/mma/store.
// Verifies: launch safety (no HSA exception) + numerics vs CPU reference.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "rocm/ds4_rocm_wmma_gfx906.cuh"

#define MTILES 8
#define NB 4  // 4 B fragments per wave like the MoE gate_up_mid kernel (bg0,bu0,bg1,bu1)

static bool hip_ok(hipError_t err, const char *what) {
    if (err == hipSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    return false;
}

__global__ void shim_mma_kernel(const half *A, const half *B, float *C, int n_waves) {
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    if (wave >= (uint32_t)n_waves) return;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major> a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::row_major> b0, b1, b2, b3;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float, rocwmma::row_major> c0, c1, c2, c3;

    rocwmma::fill_fragment(c0, 0.0f);
    rocwmma::fill_fragment(c1, 0.0f);
    rocwmma::fill_fragment(c2, 0.0f);
    rocwmma::fill_fragment(c3, 0.0f);

    // 2 K-step iterations to exercise accumulation (like k0 loop in ds4).
    for (uint32_t k0 = 0; k0 < 2u; k0++) {
        rocwmma::load_matrix_sync(a, A + (size_t)wave * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b0, B + ((size_t)wave * NB + 0) * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b1, B + ((size_t)wave * NB + 1) * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b2, B + ((size_t)wave * NB + 2) * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b3, B + ((size_t)wave * NB + 3) * 512 + k0 * 256, 16);
        rocwmma::mma_sync(c0, a, b0, c0);
        rocwmma::mma_sync(c1, a, b1, c1);
        rocwmma::mma_sync(c2, a, b2, c2);
        rocwmma::mma_sync(c3, a, b3, c3);
    }

    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 0) * 256, c0, 16, rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 1) * 256, c1, 16, rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 2) * 256, c2, 16, rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 3) * 256, c3, 16, rocwmma::mem_row_major);
}

// CPU reference: C[i][j] = sum over 2 K-tiles of sum_k A[i][k]*B[k][j]
static void ref_gemm(const std::vector<float> &A, const std::vector<float> &B,
                     std::vector<float> &C, int wave, int nb) {
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float acc = 0.0f;
            for (int kt = 0; kt < 2; kt++)
                for (int k = 0; k < 16; k++)
                    acc += A[(size_t)wave * 512 + kt * 256 + i * 16 + k] *
                           B[((size_t)wave * NB + nb) * 512 + kt * 256 + k * 16 + j];
            C[((size_t)wave * NB + nb) * 256 + i * 16 + j] = acc;
        }
}

int main() {
    const int waves = MTILES;
    std::vector<float> hA(waves * 512), hB(waves * NB * 512);
    std::vector<float> hC_ref(waves * NB * 256, 0.0f);
    std::vector<float> hC(waves * NB * 256, -1.0f);

    srand(42);
    for (auto &v : hA) v = ((rand() % 17) - 8) / 8.0f;
    for (auto &v : hB) v = ((rand() % 13) - 6) / 8.0f;

    for (int w = 0; w < waves; w++)
        for (int nb = 0; nb < NB; nb++) ref_gemm(hA, hB, hC_ref, w, nb);

    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    std::vector<half> hAh(hA.size()), hBh(hB.size());
    for (size_t i = 0; i < hA.size(); i++) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < hB.size(); i++) hBh[i] = __float2half(hB[i]);

    if (!hip_ok(hipMalloc(&dA, hAh.size() * sizeof(half)), "malloc A") ||
        !hip_ok(hipMalloc(&dB, hBh.size() * sizeof(half)), "malloc B") ||
        !hip_ok(hipMalloc(&dC, hC.size() * sizeof(float)), "malloc C") ||
        !hip_ok(hipMemcpy(dA, hAh.data(), hAh.size() * sizeof(half), hipMemcpyHostToDevice), "copy A") ||
        !hip_ok(hipMemcpy(dB, hBh.data(), hBh.size() * sizeof(half), hipMemcpyHostToDevice), "copy B") ||
        !hip_ok(hipMemset(dC, 0, hC.size() * sizeof(float)), "clear C")) {
        (void)hipFree(dC);
        (void)hipFree(dB);
        (void)hipFree(dA);
        return 2;
    }

    shim_mma_kernel<<<1, 256>>>(dA, dB, dC, waves);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("SHIM TEST LAUNCH FAILURE: %s\n", hipGetErrorString(err));
        return 3;
    }
    if (!hip_ok(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost), "copy C")) {
        (void)hipFree(dC);
        (void)hipFree(dB);
        (void)hipFree(dA);
        return 3;
    }

    double max_err = 0.0;
    for (size_t i = 0; i < hC.size(); i++) {
        double e = fabs((double)hC[i] - (double)hC_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("max_err=%g (256 tiles checked, %zu values)\n", max_err, hC.size());
    (void)hipFree(dC);
    (void)hipFree(dB);
    (void)hipFree(dA);
    if (max_err > 1e-3) { printf("SHIM TEST NUMERIC MISMATCH\n"); return 4; }
    printf("SHIM TEST PASS\n");
    return 0;
}
