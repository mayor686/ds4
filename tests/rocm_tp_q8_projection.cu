/* Validate the production ROCm Q8 K-slice primitive at DeepSeek-V4 hidden
 * dimensions.  This is deliberately single-process: it isolates projection
 * arithmetic from the HIP-IPC collective tested by rocm_tp_ipc_star. */

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_gpu.h"

#define CHECK(expr, what) do {                                              \
    if (!(expr)) {                                                          \
        fprintf(stderr, "FAIL: %s\n", what);                              \
        return 1;                                                           \
    }                                                                       \
} while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void pack_q8_weights(unsigned char *model,
                            uint64_t in_dim,
                            uint64_t out_dim) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *q = model + (row * blocks + block) * 34u;
            q[0] = 0x00; /* IEEE fp16 1.0 */
            q[1] = 0x3c;
            for (uint64_t i = 0; i < 32u; i++) {
                const int value = (int)((row * 13u + block * 7u + i * 5u) % 15u) - 7;
                q[2u + i] = (unsigned char)(int8_t)value;
            }
        }
    }
}

int main(void) {
    enum { world = 6, warmup = 3, iterations = 20 };
    const uint64_t in_dim = 7168u;
    const uint64_t out_dim = 4096u;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t model_size = out_dim * blocks * 34u;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_x = (float *)malloc((size_t)in_dim * sizeof(float));
    float *host_full = (float *)malloc((size_t)out_dim * sizeof(float));
    float *host_sum = (float *)calloc((size_t)out_dim, sizeof(float));
    float *host_partial = (float *)malloc((size_t)out_dim * sizeof(float));
    CHECK(model && host_x && host_full && host_sum && host_partial,
          "host allocation");
    pack_q8_weights(model, in_dim, out_dim);
    for (uint64_t i = 0; i < in_dim; i++) {
        host_x[i] = (float)((int)(i % 61u) - 30) * 0.015625f;
    }

    CHECK(ds4_gpu_init(), "ROCm init");
    CHECK(ds4_gpu_set_model_map(model, model_size), "model map");
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *full = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *partial[world] = {};
    CHECK(x && full, "device allocation");
    for (int rank = 0; rank < world; rank++) {
        partial[rank] = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
        CHECK(partial[rank], "partial allocation");
    }
    CHECK(ds4_gpu_tensor_write(x, 0, host_x, in_dim * sizeof(float)),
          "activation upload");

    for (int i = 0; i < warmup; i++) {
        CHECK(ds4_gpu_matmul_q8_0_tensor(full, model, model_size, 0,
                                         in_dim, out_dim, x, 1u),
              "full warmup");
    }
    CHECK(ds4_gpu_synchronize(), "full warmup sync");
    double begin = now_ms();
    for (int i = 0; i < iterations; i++) {
        CHECK(ds4_gpu_matmul_q8_0_tensor(full, model, model_size, 0,
                                         in_dim, out_dim, x, 1u),
              "full projection");
    }
    CHECK(ds4_gpu_synchronize(), "full sync");
    const double full_ms = (now_ms() - begin) / iterations;

    double shard_ms[world] = {};
    for (int rank = 0; rank < world; rank++) {
        const uint64_t block0 = blocks * (uint64_t)rank / world;
        const uint64_t block1 = blocks * (uint64_t)(rank + 1) / world;
        const uint64_t k_off = block0 * 32u;
        const uint64_t k_cnt = (block1 - block0) * 32u;
        for (int i = 0; i < warmup; i++) {
            CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(
                          partial[rank], model, model_size, 0,
                          in_dim, k_off, k_cnt, out_dim, x, k_off),
                  "K-slice warmup");
        }
        CHECK(ds4_gpu_synchronize(), "K-slice warmup sync");
        begin = now_ms();
        for (int i = 0; i < iterations; i++) {
            CHECK(ds4_gpu_matmul_q8_0_kslice_tensor(
                          partial[rank], model, model_size, 0,
                          in_dim, k_off, k_cnt, out_dim, x, k_off),
                  "K-slice projection");
        }
        CHECK(ds4_gpu_synchronize(), "K-slice sync");
        shard_ms[rank] = (now_ms() - begin) / iterations;
    }

    CHECK(ds4_gpu_tensor_read(full, 0, host_full,
                               out_dim * sizeof(float)), "full readback");
    double max_shard_ms = 0.0;
    for (int rank = 0; rank < world; rank++) {
        CHECK(ds4_gpu_tensor_read(partial[rank], 0, host_partial,
                                   out_dim * sizeof(float)),
              "partial readback");
        for (uint64_t row = 0; row < out_dim; row++) {
            host_sum[row] += host_partial[row];
        }
        if (shard_ms[rank] > max_shard_ms) max_shard_ms = shard_ms[rank];
    }
    double sq = 0.0;
    double max_abs = 0.0;
    for (uint64_t row = 0; row < out_dim; row++) {
        const double diff = (double)host_full[row] - (double)host_sum[row];
        sq += diff * diff;
        if (fabs(diff) > max_abs) max_abs = fabs(diff);
    }
    const double rms = sqrt(sq / out_dim);
    printf("ROCm Q8 TP6 projection %llu -> %llu\n",
           (unsigned long long)in_dim, (unsigned long long)out_dim);
    printf("  full: %.4f ms\n", full_ms);
    for (int rank = 0; rank < world; rank++) {
        printf("  shard %d: %.4f ms\n", rank, shard_ms[rank]);
    }
    printf("  ideal parallel compute: %.4f ms (%.2fx)\n",
           max_shard_ms, full_ms / max_shard_ms);
    printf("  numerical: rms=%g max_abs=%g\n", rms, max_abs);

    CHECK(max_abs <= 2.0e-3, "Q8 full versus six K-slices");
    for (int rank = 0; rank < world; rank++) ds4_gpu_tensor_free(partial[rank]);
    ds4_gpu_tensor_free(full);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(host_partial);
    free(host_sum);
    free(host_full);
    free(host_x);
    free(model);
    printf("  verification: PASS\n");
    return 0;
}
