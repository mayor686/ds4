#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

#include "rocm/ds4_rocm_q8.cuh"

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

#include "rocm/ds4_rocm_embedding_launch.cuh"

#include "rocm/ds4_rocm_matmul.cuh"

#include "rocm/ds4_rocm_fp8_kv_launch.cuh"

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

#include "rocm/ds4_rocm_misc_launch.cuh"
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#include "rocm/ds4_rocm_moe_launch.cuh"

#include "rocm/ds4_rocm_glm.cuh"

#include "rocm/ds4_rocm_hc_output_launch.cuh"

#include "rocm/ds4_rocm_current_api_compat.cuh"

/* Tensor-parallel gates are Metal-only; stubs keep shared graph code
 * linkable (TP option validation rejects non-Metal backends). */
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    (void)layer; (void)gate;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    (void)fn;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    (void)fn;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) {
        return 0;
    }
    ds4_gpu_tensor x_slice = *x;
    x_slice.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    x_slice.bytes = k_cnt * sizeof(float);
    x_slice.owner = 0;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, out_dim, k_off, k_cnt, &x_slice, 1u);
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t in_start,
        uint64_t in_count,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 ||
        in_count == 0 || n_tok == 0 || n_tok > 65535u) return 0;
    if ((in_start % 32u) != 0 || (in_count % 32u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    const uint64_t full_blocks = (in_dim + 31u) / 32u;
    const uint64_t block_start = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    if (weight_offset > model_size || full_blocks > UINT64_MAX / 34u ||
        out_dim > UINT64_MAX / (full_blocks * 34u) ||
        in_count > UINT64_MAX / n_tok || out_dim > UINT64_MAX / n_tok) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * full_blocks * 34u;
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_count * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                             weight_bytes, "q8_0_kslice");
    if (!wptr) return 0;

    const uint64_t xq_bytes = n_tok * slice_blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + n_tok * slice_blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 kslice prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const dim3 qgrid((unsigned)slice_blocks, (unsigned)n_tok, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_count, slice_blocks);
    if (!cuda_ok(cudaGetLastError(),
                 "matmul_q8_0_kslice quantize launch")) return 0;
    const dim3 grid(((unsigned)out_dim + 7u) / 8u,
                    (unsigned)n_tok, 1u);
    matmul_q8_0_kslice_preq_warp8_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            (const unsigned char *)wptr,
            xq,
            xscale,
            in_count,
            out_dim,
            full_blocks,
            block_start,
            slice_blocks,
            1);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice launch");
}

extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t weight_type,
        uint64_t full_in_dim,
        uint64_t k_off,
        uint64_t k_cnt,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_matmul_q8_0_kslice_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, k_off, k_cnt, out_dim, x, x_elem_off);
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    (void)out; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups_total; (void)group0; (void)group_cnt; (void)out_dim;
    (void)heads;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add; (void)residual_hc;
    (void)post; (void)comb; (void)n_embd; (void)n_hc;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}
