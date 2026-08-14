#include "ds4_gpu.h"

#include <inttypes.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(DS4_ROCM_BUILD)
extern int ds4_gpu_decode_attn_rope_fuse_available(void);
extern void ds4_gpu_set_decode_attn_rope_fuse(
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow);
#endif

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time GPU module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_GPU_TOPK_REGRESSION_SEC",
                getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0));
        fprintf(stderr, "gpu-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_decode_attention_ring_reference(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 257;
    const uint32_t raw_cap = 300;
    const uint32_t raw_start = 270;
    const uint32_t n_comp = 511;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    float *sinks = (float *)malloc(n_head * sizeof(float));
    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    float *comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    float *mask_host = (float *)malloc(n_comp * sizeof(float));
    float *heads_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *reference = (float *)malloc((size_t)q_count * sizeof(float));
    float *scores = (float *)malloc((n_raw + n_comp) * sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !mask_host ||
        !heads_host || !reference || !scores) return 1;

    for (uint32_t h = 0; h < n_head; h++) sinks[h] = -0.25f + (float)h * 0.01f;
    for (uint64_t i = 0; i < q_count; i++)
        q_host[i] = (float)((int)((i * 13u + 5u) % 101u) - 50) / 700.0f;
    for (uint64_t i = 0; i < raw_count; i++)
        raw_host[i] = (float)((int)((i * 17u + 3u) % 127u) - 63) / 500.0f;
    for (uint64_t i = 0; i < comp_count; i++)
        comp_host[i] = (float)((int)((i * 19u + 7u) % 131u) - 65) / 550.0f;
    for (uint32_t c = 0; c < n_comp; c++)
        mask_host[c] = (c % 17u == 0u) ? -1.0e30f : -(float)(c % 7u) * 0.015f;

    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q_host + (uint64_t)h * head_dim;
        float max_score = sinks[h];
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            const float *kv = raw_host + (uint64_t)row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            scores[r] = dot * scale;
            if (scores[r] > max_score) max_score = scores[r];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            float s = -3.4e38f;
            if (mask_host[c] > -5.0e29f) {
                const float *kv = comp_host + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                s = dot * scale + mask_host[c];
            }
            scores[n_raw + c] = s;
            if (s > max_score) max_score = s;
        }
        float denom = expf(sinks[h] - max_score);
        for (uint32_t r = 0; r < n_raw + n_comp; r++) {
            scores[r] = expf(scores[r] - max_score);
            denom += scores[r];
        }
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t row = (raw_start + r) % raw_cap;
                acc += scores[r] * raw_host[(uint64_t)row * head_dim + d];
            }
            for (uint32_t c = 0; c < n_comp; c++)
                acc += scores[n_raw + c] * comp_host[(uint64_t)c * head_dim + d];
            reference[(uint64_t)h * head_dim + d] = acc / denom;
        }
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *mask = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp && mask &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(mask, 0, mask_host, (uint64_t)n_comp * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads, sinks, n_head * sizeof(float), 0,
                                              q, raw, n_raw, raw_cap, raw_start,
                                              comp, 0, n_comp, mask, 1,
                                              n_head, head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (uint64_t i = 0; i < q_count; i++) {
            const float abs_err = fabsf(heads_host[i] - reference[i]);
            const float rel_err = abs_err / fmaxf(1.0e-4f, fabsf(reference[i]));
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
        }
        fprintf(stderr, "gpu-regression: attention ring reference max_abs=%g max_rel=%g\n",
                (double)max_abs, (double)max_rel);
        rc = (max_abs <= 2.0e-5f && max_rel <= 2.0e-3f) ? 0 : 1;
    }

    if (rc == 0 && getenv("DS4_TEST_ROCM_F16_CACHE") != NULL) {
        const uint32_t n_rot = 64;
        const uint64_t compact_row_bytes =
            (uint64_t)(head_dim - n_rot) * sizeof(uint16_t) +
            (uint64_t)n_rot * sizeof(float);
        ds4_gpu_tensor *comp_f16 = ds4_gpu_tensor_alloc(
            (uint64_t)n_comp * compact_row_bytes);
        float *heads_f16 = (float *)malloc((size_t)q_count * sizeof(float));
        if (!comp_f16 || !heads_f16 ||
            !ds4_gpu_tensor_pack_comp_kv_f16_rope_f32(
                comp_f16, 0, comp, 0, n_comp, head_dim, n_rot) ||
            !ds4_gpu_attention_decode_heads_tensor(
                heads, sinks, n_head * sizeof(float), 0,
                q, raw, n_raw, raw_cap, raw_start,
                comp_f16, 1, n_comp, mask, 1, n_head, head_dim) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(heads, 0, heads_f16,
                                 q_count * sizeof(float))) {
            rc = 1;
        } else {
            float max_abs = 0.0f;
            for (uint64_t i = 0; i < q_count; i++) {
                const float err = fabsf(heads_f16[i] - heads_host[i]);
                if (err > max_abs) max_abs = err;
            }
            fprintf(stderr,
                    "gpu-regression: ROCm F16+RoPE-F32 compressed-cache max_abs=%g\n",
                    (double)max_abs);
            if (max_abs > 2.0e-4f) rc = 1;
        }
        ds4_gpu_tensor_free(comp_f16);
        free(heads_f16);
    }

    ds4_gpu_tensor_free(mask);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(scores); free(reference); free(heads_host); free(mask_host);
    free(comp_host); free(raw_host); free(q_host); free(sinks);
    return rc;
}

static int check_decode_attention_indexed_reference(void) {
    enum {
        n_head = 64,
        head_dim = 512,
        n_raw = 256,
        raw_cap = 300,
        raw_start = 271,
        n_comp = 4096,
        top_k = 512,
        pos0 = 16383,
        ratio = 4,
    };
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    float *sinks = (float *)malloc(n_head * sizeof(float));
    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    float *comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    int32_t *topk_host = (int32_t *)malloc(top_k * sizeof(int32_t));
    float *heads_host = (float *)malloc((size_t)q_count * sizeof(float));
    const int compare_paths = getenv("DS4_TEST_ATTENTION_AB") != NULL;
    float *stable_heads = compare_paths
        ? (float *)malloc((size_t)q_count * sizeof(float)) : NULL;
    float *reference = (float *)malloc((size_t)q_count * sizeof(float));
    float *scores = (float *)malloc((n_raw + top_k) * sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !topk_host ||
        !heads_host || (compare_paths && !stable_heads) ||
        !reference || !scores) return 1;

    for (uint32_t h = 0; h < n_head; h++)
        sinks[h] = -0.35f + (float)(h % 11u) * 0.017f;
    for (uint64_t i = 0; i < q_count; i++)
        q_host[i] = (float)((int)((i * 13u + 7u) % 127u) - 63) / 600.0f;
    for (uint64_t i = 0; i < raw_count; i++)
        raw_host[i] = (float)((int)((i * 17u + 5u) % 131u) - 65) / 500.0f;
    for (uint64_t i = 0; i < comp_count; i++)
        comp_host[i] = (float)((int)((i * 19u + 3u) % 137u) - 68) / 550.0f;
    for (uint32_t i = 0; i < top_k; i++)
        topk_host[i] = (int32_t)((i * 7u + 11u) % n_comp);

    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q_host + (uint64_t)h * head_dim;
        float max_score = sinks[h];
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            const float *kv = raw_host + (uint64_t)row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            scores[r] = dot * scale;
            if (scores[r] > max_score) max_score = scores[r];
        }
        for (uint32_t i = 0; i < top_k; i++) {
            const float *kv = comp_host + (uint64_t)topk_host[i] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            scores[n_raw + i] = dot * scale;
            if (scores[n_raw + i] > max_score) max_score = scores[n_raw + i];
        }
        float denom = expf(sinks[h] - max_score);
        for (uint32_t r = 0; r < n_raw + top_k; r++) {
            scores[r] = expf(scores[r] - max_score);
            denom += scores[r];
        }
        for (uint32_t d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < n_raw; r++) {
                const uint32_t row = (raw_start + r) % raw_cap;
                acc += scores[r] * raw_host[(uint64_t)row * head_dim + d];
            }
            for (uint32_t i = 0; i < top_k; i++) {
                acc += scores[n_raw + i] *
                    comp_host[(uint64_t)topk_host[i] * head_dim + d];
            }
            reference[(uint64_t)h * head_dim + d] = acc / denom;
        }
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *topk_t = ds4_gpu_tensor_alloc(top_k * sizeof(int32_t));
    int rc = 1;
    if (heads && q && raw && comp && topk_t &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(topk_t, 0, topk_host,
                             top_k * sizeof(int32_t))) {
        const int warm = 5;
        const int iters = 50;
        int ok = 1;
        if (stable_heads) {
            setenv("DS4_ROCM_DISABLE_ATTENTION_INDEXED_TRANSPOSE", "1", 1);
            for (int i = 0; ok && i < warm; i++) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    heads, sinks, n_head * sizeof(float), 0, q, raw, comp, 0,
                    topk_t, 1, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                    0, ratio, n_head, head_dim);
            }
            ok = ok && ds4_gpu_synchronize() && ds4_gpu_tensor_read(
                heads, 0, stable_heads, q_count * sizeof(float));
            if (ok) unsetenv("DS4_ROCM_DISABLE_ATTENTION_INDEXED_TRANSPOSE");
        }
        for (int i = 0; ok && i < warm; i++) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, sinks, n_head * sizeof(float), 0, q, raw, comp, 0,
                topk_t, 1, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                0, ratio, n_head, head_dim);
        }
        ok = ok && ds4_gpu_synchronize();
        const double t0 = monotonic_seconds();
        for (int i = 0; ok && i < iters; i++) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                heads, sinks, n_head * sizeof(float), 0, q, raw, comp, 0,
                topk_t, 1, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                0, ratio, n_head, head_dim);
        }
        ok = ok && ds4_gpu_synchronize();
        const double elapsed_ms =
            (monotonic_seconds() - t0) * 1000.0 / (double)iters;
        int rope_regression_failed = 0;
        if (ok && ds4_gpu_tensor_read(
                heads, 0, heads_host, q_count * sizeof(float))) {
            float max_abs = 0.0f;
            float max_rel = 0.0f;
            uint64_t output_hash = UINT64_C(1469598103934665603);
            for (uint64_t i = 0; i < q_count; i++) {
                const float abs_err = fabsf(heads_host[i] - reference[i]);
                const float rel_err =
                    abs_err / fmaxf(1.0e-4f, fabsf(reference[i]));
                if (abs_err > max_abs) max_abs = abs_err;
                if (rel_err > max_rel) max_rel = rel_err;
                uint32_t bits;
                memcpy(&bits, &heads_host[i], sizeof(bits));
                for (uint32_t byte = 0; byte < sizeof(bits); byte++) {
                    output_hash ^= (bits >> (8u * byte)) & 0xffu;
                    output_hash *= UINT64_C(1099511628211);
                }
            }
            fprintf(stderr,
                    "gpu-regression: indexed attention 16K avg=%.3f ms "
                    "max_abs=%g max_rel=%g hash=%016" PRIx64 "\n",
                    elapsed_ms, (double)max_abs, (double)max_rel, output_hash);
            if (stable_heads) {
                float path_max_abs = 0.0f;
                double path_sum_sq = 0.0;
                uint64_t path_exact = 0;
                for (uint64_t i = 0; i < q_count; i++) {
                    const float d = fabsf(heads_host[i] - stable_heads[i]);
                    if (d > path_max_abs) path_max_abs = d;
                    path_sum_sq += (double)d * (double)d;
                    if (d == 0.0f) path_exact++;
                }
                fprintf(stderr,
                        "gpu-regression: indexed attention stable/transpose "
                        "max_abs=%g rms=%g exact=%" PRIu64 "/%" PRIu64 "\n",
                        (double)path_max_abs,
                        sqrt(path_sum_sq / (double)q_count),
                        path_exact, q_count);
            }
#if defined(DS4_ROCM_BUILD)
            if (stable_heads && ds4_gpu_decode_attn_rope_fuse_available()) {
                float *rope_reference = (float *)malloc(
                    (size_t)q_count * sizeof(float));
                int rope_ok = rope_reference != NULL &&
                    ds4_gpu_rope_tail_tensor(
                        heads, 1, n_head, head_dim, 64, pos0, 65536, true,
                        160000.0f, 1.0f / 16.0f, 1.0f,
                        1.0f / (1.0f + 0.1f * logf(16.0f)),
                        32.0f, 1.0f) &&
                    ds4_gpu_synchronize() &&
                    ds4_gpu_tensor_read(heads, 0, rope_reference,
                                         q_count * sizeof(float));
                if (rope_ok) {
                    ds4_gpu_set_decode_attn_rope_fuse(
                        head_dim, 64, pos0, 65536, true,
                        160000.0f, 1.0f / 16.0f, 1.0f,
                        1.0f / (1.0f + 0.1f * logf(16.0f)),
                        32.0f, 1.0f);
                    rope_ok =
                        ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                            heads, sinks, n_head * sizeof(float), 0,
                            q, raw, comp, 0, topk_t, 1, pos0,
                            n_raw, raw_cap, raw_start, n_comp, top_k,
                            0, ratio, n_head, head_dim) &&
                        ds4_gpu_synchronize() &&
                        ds4_gpu_tensor_read(heads, 0, heads_host,
                                             q_count * sizeof(float));
                }
                float rope_max_abs = 0.0f;
                double rope_sum_sq = 0.0;
                uint64_t rope_exact = 0;
                if (rope_ok) {
                    for (uint64_t i = 0; i < q_count; i++) {
                        const float d = fabsf(
                            heads_host[i] - rope_reference[i]);
                        if (d > rope_max_abs) rope_max_abs = d;
                        rope_sum_sq += (double)d * (double)d;
                        if (d == 0.0f) rope_exact++;
                    }
                    fprintf(stderr,
                            "gpu-regression: indexed attention fused-RoPE "
                            "max_abs=%g rms=%g exact=%" PRIu64 "/%" PRIu64 "\n",
                            (double)rope_max_abs,
                            sqrt(rope_sum_sq / (double)q_count),
                            rope_exact, q_count);
                }
                if (!rope_ok || rope_max_abs > 1.0e-7f)
                    rope_regression_failed = 1;
                free(rope_reference);
            }
#endif
            rc = (max_abs <= 2.0e-5f && max_rel <= 2.0e-3f &&
                  !rope_regression_failed) ? 0 : 1;
        }
    }
    ds4_gpu_tensor_free(topk_t);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(scores); free(reference); free(stable_heads); free(heads_host);
    free(topk_host);
    free(comp_host); free(raw_host); free(q_host); free(sinks);
    return rc;
}

static float ordered_add(float a, float b) {
    volatile float lhs = a;
    volatile float rhs = b;
    return lhs + rhs;
}

static int check_owned_moe_combine(void) {
    enum { rows = 66, out_dim = 17, slots = 6 };
    const uint32_t split = 128u;
    int32_t selected[rows * slots];
    float home[rows * slots * out_dim];
    float peer[rows * slots * out_dim];
    float expected[rows * out_dim];
    float got[rows * out_dim];
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t s = 0; s < slots; s++) {
            if (r < 64u) {
                selected[r * slots + s] = (r & (1u << s))
                    ? (int32_t)(split + s) : (int32_t)s;
            } else if (r == 64u) {
                selected[r * slots + s] = -1;
            } else {
                selected[r * slots + s] = s == 3u
                    ? (int32_t)(2u * split) : (int32_t)s;
            }
            for (uint32_t c = 0; c < out_dim; c++) {
                const float sign = (s & 1u) ? -1.0f : 1.0f;
                home[((r * slots + s) * out_dim) + c] = sign *
                    (10.0f * (float)(s + 1u) + 0.125f * (float)c);
                peer[((r * slots + s) * out_dim) + c] = sign *
                    (100.0f + 10.0f * (float)(s + 1u) +
                     0.25f * (float)c);
            }
        }
        for (uint32_t c = 0; c < out_dim; c++) {
            float part0 = 0.0f;
            float part1 = 0.0f;
            for (uint32_t s = 0; s < 3u; s++) {
                const int32_t e = selected[r * slots + s];
                if (e >= 0 && (uint32_t)e < 2u * split) {
                    const float *src = (uint32_t)e < split ? home : peer;
                    part0 = ordered_add(
                        part0, src[((r * slots + s) * out_dim) + c]);
                }
            }
            for (uint32_t s = 3u; s < slots; s++) {
                const int32_t e = selected[r * slots + s];
                if (e >= 0 && (uint32_t)e < 2u * split) {
                    const float *src = (uint32_t)e < split ? home : peer;
                    part1 = ordered_add(
                        part1, src[((r * slots + s) * out_dim) + c]);
                }
            }
            expected[r * out_dim + c] = ordered_add(part0, part1);
        }
    }

    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(sizeof(got));
    ds4_gpu_tensor *home_t = ds4_gpu_tensor_alloc(sizeof(home));
    ds4_gpu_tensor *peer_t = ds4_gpu_tensor_alloc(sizeof(peer));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc(sizeof(selected));
    int rc = 1;
    if (out_t && home_t && peer_t && selected_t &&
        ds4_gpu_tensor_write(home_t, 0, home, sizeof(home)) &&
        ds4_gpu_tensor_write(peer_t, 0, peer, sizeof(peer)) &&
        ds4_gpu_tensor_write(selected_t, 0, selected, sizeof(selected)) &&
        ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
            out_t, home_t, peer_t, selected_t, out_dim, split, rows) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(out_t, 0, got, sizeof(got))) {
        rc = 0;
        for (uint32_t i = 0; i < rows * out_dim; i++) {
            if (got[i] != expected[i]) {
                fprintf(stderr,
                        "owned MoE combine mismatch i=%u got=%g expected=%g\n",
                        i, (double)got[i], (double)expected[i]);
                rc = 1;
                break;
            }
        }
    }
    ds4_gpu_tensor_free(selected_t);
    ds4_gpu_tensor_free(peer_t);
    ds4_gpu_tensor_free(home_t);
    ds4_gpu_tensor_free(out_t);
    if (rc == 0) fprintf(stderr, "gpu-regression: owned MoE combine OK\n");
    return rc;
}

#if defined(DS4_ROCM_BUILD)
static int check_f16_compressor_quad(void) {
    if (!ds4_gpu_f16_compressor_quad_available()) {
        fprintf(stderr, "gpu-regression: F16 compressor quad unavailable; skipped\n");
        return 0;
    }
    enum { in_dim = 7168, out_dim0 = 1024, out_dim1 = 256 };
    const uint64_t bytes0 =
        (uint64_t)in_dim * out_dim0 * sizeof(uint16_t);
    const uint64_t bytes1 =
        (uint64_t)in_dim * out_dim1 * sizeof(uint16_t);
    const uint64_t model_bytes = 2u * bytes0 + 2u * bytes1;
    uint16_t *model = (uint16_t *)malloc((size_t)model_bytes);
    float *x_host = (float *)malloc((size_t)in_dim * sizeof(float));
    float *ref[4] = {
        (float *)malloc((size_t)out_dim0 * sizeof(float)),
        (float *)malloc((size_t)out_dim0 * sizeof(float)),
        (float *)malloc((size_t)out_dim1 * sizeof(float)),
        (float *)malloc((size_t)out_dim1 * sizeof(float)),
    };
    float *got[4] = {
        (float *)malloc((size_t)out_dim0 * sizeof(float)),
        (float *)malloc((size_t)out_dim0 * sizeof(float)),
        (float *)malloc((size_t)out_dim1 * sizeof(float)),
        (float *)malloc((size_t)out_dim1 * sizeof(float)),
    };
    if (!model || !x_host || !ref[0] || !ref[1] || !ref[2] || !ref[3] ||
        !got[0] || !got[1] || !got[2] || !got[3]) return 1;

    const uint16_t values[] = {
        0x0000u, 0x3c00u, 0xbc00u, 0x3800u,
        0xb800u, 0x3400u, 0xb400u,
    };
    for (uint64_t i = 0; i < model_bytes / sizeof(uint16_t); i++)
        model[i] = values[(i * 17u + i / in_dim * 11u + 3u) %
                          (sizeof(values) / sizeof(values[0]))];
    for (uint32_t i = 0; i < in_dim; i++)
        x_host[i] = (float)((int)((i * 29u + 7u) % 97u) - 48) / 512.0f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *ref_t[4] = {
        ds4_gpu_tensor_alloc((uint64_t)out_dim0 * sizeof(float)),
        ds4_gpu_tensor_alloc((uint64_t)out_dim0 * sizeof(float)),
        ds4_gpu_tensor_alloc((uint64_t)out_dim1 * sizeof(float)),
        ds4_gpu_tensor_alloc((uint64_t)out_dim1 * sizeof(float)),
    };
    ds4_gpu_tensor *got_t[4] = {
        ds4_gpu_tensor_alloc((uint64_t)out_dim0 * sizeof(float)),
        ds4_gpu_tensor_alloc((uint64_t)out_dim0 * sizeof(float)),
        ds4_gpu_tensor_alloc((uint64_t)out_dim1 * sizeof(float)),
        ds4_gpu_tensor_alloc((uint64_t)out_dim1 * sizeof(float)),
    };
    int rc = 1;
    if (x && ref_t[0] && ref_t[1] && ref_t[2] && ref_t[3] &&
        got_t[0] && got_t[1] && got_t[2] && got_t[3] &&
        ds4_gpu_set_model_map(model, model_bytes) &&
        ds4_gpu_tensor_write(x, 0, x_host,
                             (uint64_t)in_dim * sizeof(float)) &&
        ds4_gpu_matmul_f16_pair_tensor(
            ref_t[0], ref_t[1], model, model_bytes, 0, bytes0,
            in_dim, out_dim0, x, 1) &&
        ds4_gpu_matmul_f16_pair_tensor(
            ref_t[2], ref_t[3], model, model_bytes,
            2u * bytes0, 2u * bytes0 + bytes1,
            in_dim, out_dim1, x, 1) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_matmul_f16_quad_tensor(
            got_t[0], got_t[1], got_t[2], got_t[3], model, model_bytes,
            0, bytes0, 2u * bytes0, 2u * bytes0 + bytes1,
            in_dim, out_dim0, out_dim1, x) == 1 &&
        ds4_gpu_synchronize()) {
        rc = 0;
        uint64_t nonexact = 0;
        float max_abs = 0.0f;
        for (uint32_t output = 0; output < 4u && rc == 0; output++) {
            const uint32_t count = output < 2u ? out_dim0 : out_dim1;
            if (!ds4_gpu_tensor_read(
                    ref_t[output], 0, ref[output],
                    (uint64_t)count * sizeof(float)) ||
                !ds4_gpu_tensor_read(
                    got_t[output], 0, got[output],
                    (uint64_t)count * sizeof(float))) {
                rc = 1;
                break;
            }
            for (uint32_t i = 0; i < count; i++) {
                const float delta = fabsf(ref[output][i] - got[output][i]);
                if (delta != 0.0f) nonexact++;
                if (delta > max_abs) max_abs = delta;
            }
        }
        fprintf(stderr,
                "gpu-regression: F16 compressor quad max_abs=%g "
                "nonexact=%" PRIu64 "\n",
                (double)max_abs, nonexact);
        if (nonexact != 0u) rc = 1;
    }

    for (uint32_t i = 0; i < 4u; i++) {
        ds4_gpu_tensor_free(got_t[i]);
        ds4_gpu_tensor_free(ref_t[i]);
        free(got[i]);
        free(ref[i]);
    }
    ds4_gpu_tensor_free(x);
    free(x_host);
    free(model);
    return rc;
}
#endif

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_decode_attention_ring_reference() != 0) rc = 1;
    if (check_decode_attention_indexed_reference() != 0) rc = 1;
    if (check_owned_moe_combine() != 0) rc = 1;
#if defined(DS4_ROCM_BUILD)
    if (check_f16_compressor_quad() != 0) rc = 1;
#endif
    ds4_gpu_cleanup();
    if (rc == 0) puts("GPU long-context regression: OK");
    return rc;
}
