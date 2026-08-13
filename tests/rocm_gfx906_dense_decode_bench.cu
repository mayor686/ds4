/* Microbenchmark the two one-token Q8 projections used by the fused
 * attention-output/HC decode path at the DeepSeek-V4 production shapes. */

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
        fprintf(stderr, "FAIL: %s\n", what);                               \
        return 1;                                                           \
    }                                                                       \
} while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint64_t hash_f32(const float *x, uint64_t n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, x + i, sizeof(bits));
        for (unsigned b = 0; b < 4; b++) {
            h ^= (uint8_t)(bits >> (8u * b));
            h *= UINT64_C(1099511628211);
        }
    }
    return h;
}

static void pack_q8_weights(unsigned char *dst,
                            uint64_t in_dim,
                            uint64_t out_dim,
                            uint64_t seed) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *q = dst + (row * blocks + block) * 34u;
            q[0] = 0x00; /* IEEE fp16 1.0 */
            q[1] = 0x3c;
            for (uint64_t i = 0; i < 32u; i++) {
                const int value = (int)((row * 13u + block * 7u +
                                         i * 5u + seed) % 31u) - 15;
                q[2u + i] = (unsigned char)(int8_t)value;
            }
        }
    }
}

int main(void) {
    enum { warmup = 20, iterations = 500 };
    const uint64_t group_dim = 4096u;
    const uint64_t rank = 1024u;
    const uint32_t n_groups = 8u;
    const uint64_t low_dim = rank * n_groups;
    const uint64_t n_embd = 7168u;
    const uint32_t n_hc = 4u;
    const uint64_t out_a_bytes = low_dim * (group_dim / 32u) * 34u;
    const uint64_t out_b_bytes = n_embd * (low_dim / 32u) * 34u;
    const uint64_t model_size = out_a_bytes + out_b_bytes;
    unsigned char *model = (unsigned char *)malloc((size_t)model_size);
    float *host_heads = (float *)malloc((size_t)(n_groups * group_dim) * sizeof(float));
    float *host_residual = (float *)malloc((size_t)(n_hc * n_embd) * sizeof(float));
    float host_split[2u * n_hc + n_hc * n_hc];
    float *host_low = (float *)malloc((size_t)low_dim * sizeof(float));
    float *host_hc = (float *)malloc((size_t)(n_hc * n_embd) * sizeof(float));
    CHECK(model && host_heads && host_residual && host_low && host_hc,
          "host allocation");

    pack_q8_weights(model, group_dim, low_dim, 3u);
    pack_q8_weights(model + out_a_bytes, low_dim, n_embd, 11u);
    for (uint64_t i = 0; i < n_groups * group_dim; i++)
        host_heads[i] = (float)((int)(i % 127u) - 63) * 0.00390625f;
    for (uint64_t i = 0; i < n_hc * n_embd; i++)
        host_residual[i] = (float)((int)(i % 67u) - 33) * 0.001953125f;
    for (uint32_t i = 0; i < 2u * n_hc + n_hc * n_hc; i++)
        host_split[i] = (float)((int)(i % 9u) - 4) * 0.0625f;

    CHECK(ds4_gpu_init(), "ROCm init");
    CHECK(ds4_gpu_set_model_map(model, model_size), "model map");
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(n_groups * group_dim * sizeof(float));
    ds4_gpu_tensor *low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    ds4_gpu_tensor *block_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(n_hc * n_embd * sizeof(float));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(sizeof(host_split));
    ds4_gpu_tensor *out_hc = ds4_gpu_tensor_alloc(n_hc * n_embd * sizeof(float));
    CHECK(heads && low && block_out && residual && split && out_hc,
          "device allocation");
    CHECK(ds4_gpu_tensor_write(heads, 0, host_heads,
                               n_groups * group_dim * sizeof(float)),
          "heads upload");
    CHECK(ds4_gpu_tensor_write(residual, 0, host_residual,
                               n_hc * n_embd * sizeof(float)),
          "residual upload");
    CHECK(ds4_gpu_tensor_write(split, 0, host_split, sizeof(host_split)),
          "split upload");

    for (int i = 0; i < warmup; i++) {
        CHECK(ds4_gpu_attention_output_low_q8_tensor(
                      low, model, model_size, 0, group_dim, rank,
                      n_groups, heads), "attention output low warmup");
    }
    CHECK(ds4_gpu_synchronize(), "attention output low warmup sync");
    double begin = now_ms();
    for (int i = 0; i < iterations; i++) {
        CHECK(ds4_gpu_attention_output_low_q8_tensor(
                      low, model, model_size, 0, group_dim, rank,
                      n_groups, heads), "attention output low");
    }
    CHECK(ds4_gpu_synchronize(), "attention output low sync");
    const double low_ms = (now_ms() - begin) / iterations;

    for (int i = 0; i < warmup; i++) {
        CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
                      out_hc, block_out, model, model_size, out_a_bytes,
                      low_dim, n_embd, low, residual, split,
                      (uint32_t)n_embd, n_hc), "Q8 HC warmup");
    }
    CHECK(ds4_gpu_synchronize(), "Q8 HC warmup sync");
    begin = now_ms();
    for (int i = 0; i < iterations; i++) {
        CHECK(ds4_gpu_matmul_q8_0_hc_expand_tensor(
                      out_hc, block_out, model, model_size, out_a_bytes,
                      low_dim, n_embd, low, residual, split,
                      (uint32_t)n_embd, n_hc), "Q8 HC");
    }
    CHECK(ds4_gpu_synchronize(), "Q8 HC sync");
    const double hc_ms = (now_ms() - begin) / iterations;
    CHECK(ds4_gpu_tensor_read(low, 0, host_low, low_dim * sizeof(float)),
          "low readback");
    CHECK(ds4_gpu_tensor_read(out_hc, 0, host_hc,
                              n_hc * n_embd * sizeof(float)),
          "HC readback");
    for (uint64_t i = 0; i < low_dim; i++) CHECK(isfinite(host_low[i]), "finite low");
    for (uint64_t i = 0; i < n_hc * n_embd; i++) CHECK(isfinite(host_hc[i]), "finite HC");

    printf("ROCm gfx906 dense decode: low=%.4f ms hc=%.4f ms total=%.4f ms\n",
           low_ms, hc_ms, low_ms + hc_ms);
    printf("  hashes: low=%016llx hc=%016llx\n",
           (unsigned long long)hash_f32(host_low, low_dim),
           (unsigned long long)hash_f32(host_hc, n_hc * n_embd));

    ds4_gpu_tensor_free(out_hc);
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(block_out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_cleanup();
    free(host_hc);
    free(host_low);
    free(host_residual);
    free(host_heads);
    free(model);
    return 0;
}
