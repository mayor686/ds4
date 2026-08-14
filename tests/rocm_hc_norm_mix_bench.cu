#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ds4_gpu.h"

static constexpr uint32_t kInDim = 16384u;
static constexpr uint32_t kOutDim = 24u;
static constexpr float kEps = 1.0e-6f;
static constexpr uint32_t kExactPatterns = 8u;
static constexpr int kWarmup = 100;
static constexpr int kIterations = 2000;

static uint32_t ordered_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}

static uint32_t ulp_distance(float a, float b) {
    const uint32_t aa = ordered_bits(a);
    const uint32_t bb = ordered_bits(b);
    return aa > bb ? aa - bb : bb - aa;
}

static bool run_separate(ds4_gpu_tensor *out, ds4_gpu_tensor *normalized,
                         ds4_gpu_tensor *x, const void *model,
                         uint64_t model_bytes) {
    return ds4_gpu_rms_norm_plain_tensor(normalized, x, kInDim, kEps) &&
           ds4_gpu_matmul_f16_tensor(out, model, model_bytes, 0u,
                                     kInDim, kOutDim, normalized, 1u);
}

static bool run_fused(ds4_gpu_tensor *out, ds4_gpu_tensor *x,
                      const void *model, uint64_t model_bytes) {
    return ds4_gpu_hc_rms_norm_mix_f16_tensor(
        out, x, model, model_bytes, 0u, kInDim, kOutDim, kEps);
}

template <typename Fn>
static float time_path(Fn fn) {
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    if (hipEventCreate(&begin) != hipSuccess ||
        hipEventCreate(&end) != hipSuccess) return -1.0f;
    for (int i = 0; i < kWarmup; ++i) {
        if (!fn()) return -1.0f;
    }
    if (hipDeviceSynchronize() != hipSuccess ||
        hipEventRecord(begin, nullptr) != hipSuccess) return -1.0f;
    for (int i = 0; i < kIterations; ++i) {
        if (!fn()) return -1.0f;
    }
    if (hipEventRecord(end, nullptr) != hipSuccess ||
        hipEventSynchronize(end) != hipSuccess) return -1.0f;
    float elapsed = 0.0f;
    if (hipEventElapsedTime(&elapsed, begin, end) != hipSuccess) return -1.0f;
    (void)hipEventDestroy(end);
    (void)hipEventDestroy(begin);
    return elapsed / (float)kIterations;
}

int main(void) {
    std::vector<float> host_x(kInDim);
    std::vector<__half> host_w((uint64_t)kInDim * kOutDim);
    for (uint32_t row = 0; row < kOutDim; ++row) {
        for (uint32_t i = 0; i < kInDim; ++i) {
            const int32_t p = (int32_t)((i * 13u + row * 97u + 3u) % 257u) - 128;
            host_w[(uint64_t)row * kInDim + i] = __float2half((float)p / 1024.0f);
        }
    }
    const uint64_t model_bytes = host_w.size() * sizeof(host_w[0]);
    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_map(host_w.data(), model_bytes)) {
        std::fprintf(stderr, "HC norm/mix: backend initialization failed\n");
        return 1;
    }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)kInDim * sizeof(float));
    ds4_gpu_tensor *normalized = ds4_gpu_tensor_alloc((uint64_t)kInDim * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float));
    ds4_gpu_tensor *fused = ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float));
    if (!x || !normalized || !reference || !fused ||
        !ds4_gpu_hc_rms_norm_mix_f16_available()) {
        std::fprintf(stderr, "HC norm/mix: execution failed\n");
        return 2;
    }

    std::vector<float> host_reference(kOutDim);
    std::vector<float> host_fused(kOutDim);
    uint32_t mismatches = 0u;
    uint32_t max_ulp = 0u;
    for (uint32_t pattern = 0u; pattern < kExactPatterns; ++pattern) {
        const float magnitude = std::ldexp(1.0f, (int)pattern - 4);
        for (uint32_t i = 0; i < kInDim; ++i) {
            const int32_t p = (int32_t)((i * (37u + 2u * pattern) +
                                         11u + 101u * pattern) % 1009u) - 504;
            host_x[i] = (float)p * magnitude / 257.0f;
        }
        if (!ds4_gpu_tensor_write(x, 0u, host_x.data(),
                                  (uint64_t)kInDim * sizeof(float)) ||
            !run_separate(reference, normalized, x, host_w.data(), model_bytes) ||
            !run_fused(fused, x, host_w.data(), model_bytes) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(reference, 0u, host_reference.data(),
                                 (uint64_t)kOutDim * sizeof(float)) ||
            !ds4_gpu_tensor_read(fused, 0u, host_fused.data(),
                                 (uint64_t)kOutDim * sizeof(float))) {
            std::fprintf(stderr, "HC norm/mix: pattern %u failed\n", pattern);
            return 3;
        }
        for (uint32_t i = 0; i < kOutDim; ++i) {
            if (std::memcmp(&host_reference[i], &host_fused[i], sizeof(float)) != 0) {
                ++mismatches;
                const uint32_t ulp = ulp_distance(host_reference[i], host_fused[i]);
                if (ulp > max_ulp) max_ulp = ulp;
            }
        }
    }

    const float separate_ms = time_path([&] {
        return run_separate(reference, normalized, x, host_w.data(), model_bytes);
    });
    const float fused_ms = time_path([&] {
        return run_fused(fused, x, host_w.data(), model_bytes);
    });
    std::printf("HC RMSNorm+F16 mix: separate %.4f ms, fused %.4f ms, "
                "speedup %.2fx, saved %.4f ms\n",
                separate_ms, fused_ms, separate_ms / fused_ms,
                separate_ms - fused_ms);
    std::printf("bit-exact verification: %s (%u/%u mismatches, max ULP %u)\n",
                mismatches == 0u ? "PASS" : "FAIL",
                mismatches, kOutDim * kExactPatterns, max_ulp);

    ds4_gpu_tensor_free(fused);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(normalized);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    return mismatches == 0u && separate_ms > 0.0f && fused_ms > 0.0f ? 0 : 4;
}
