/* Process-per-GPU gate for row-sharding the DeepSeek output projection.
 *
 * The output head is different from the K-sliced projections used inside a
 * transformer layer: every rank owns complete, disjoint vocabulary rows, so
 * greedy decode needs only one (token, value) candidate per rank and no FP32
 * all-reduce.  This benchmark includes the same-host IPC broadcast, process
 * scheduling, local Q8 projection, local top-1 and candidate readback.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <hip/hip_runtime.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "ds4_gpu.h"
#include "ds4_rocm_tp.h"

static constexpr uint64_t kInDim = 7168u;
static constexpr uint64_t kVocab = 129280u;
static constexpr uint64_t kRowBytes = (kInDim / 32u) * 34u;
static constexpr uint64_t kModelBytes = kVocab * kRowBytes;
static constexpr uint32_t kMaxWorld = 6u;
static constexpr int kWarmup = 5;
static constexpr int kIterations = 100;

struct candidate {
    uint32_t id;
    float value;
};

struct rank_handles {
    ds4_rocm_tp_ipc_handle dummy_input;
    ds4_rocm_tp_ipc_handle activation;
    char name[64];
};

struct rank_sample {
    candidate best;
    double compute_ms;
};

struct output_buffers {
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *best;
    ds4_gpu_tensor *dummy;
    float *logits_ptr;
    candidate *best_ptr;
};

__device__ static bool candidate_better(float value, uint32_t id,
                                         float best_value,
                                         uint32_t best_id) {
    return value > best_value || (value == best_value && id < best_id);
}

__global__ static void output_top1_kernel(const float *scores, uint32_t count,
                                          uint32_t index_offset,
                                          candidate *result) {
    __shared__ float values[256];
    __shared__ uint32_t ids[256];
    float best_value = -INFINITY;
    uint32_t best_id = UINT32_MAX;
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = scores[i];
        const uint32_t id = index_offset + i;
        if (candidate_better(value, id, best_value, best_id)) {
            best_value = value;
            best_id = id;
        }
    }
    values[threadIdx.x] = best_value;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride; stride /= 2u) {
        if (threadIdx.x < stride &&
            candidate_better(values[threadIdx.x + stride],
                             ids[threadIdx.x + stride],
                             values[threadIdx.x], ids[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            ids[threadIdx.x] = ids[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        result->id = ids[0];
        result->value = values[0];
    }
}

static bool write_all(int fd, const void *data, size_t bytes) {
    const char *p = static_cast<const char *>(data);
    while (bytes) {
        const ssize_t n = write(fd, p, bytes);
        if (n <= 0) return false;
        p += n;
        bytes -= (size_t)n;
    }
    return true;
}

static bool read_all(int fd, void *data, size_t bytes) {
    char *p = static_cast<char *>(data);
    while (bytes) {
        const ssize_t n = read(fd, p, bytes);
        if (n <= 0) return false;
        p += n;
        bytes -= (size_t)n;
    }
    return true;
}

static double elapsed_ms(std::chrono::steady_clock::time_point begin,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count() * 1.0e3;
}

static candidate merge_candidate(candidate a, candidate b) {
    if (b.value > a.value || (b.value == a.value && b.id < a.id)) return b;
    return a;
}

static bool allocate_buffers(output_buffers *b, uint64_t rows,
                             const float *host_x) {
    std::memset(b, 0, sizeof(*b));
    b->x = ds4_gpu_tensor_alloc(kInDim * sizeof(float));
    b->logits = ds4_gpu_tensor_alloc(rows * sizeof(float));
    b->best = ds4_gpu_tensor_alloc(sizeof(candidate));
    b->dummy = ds4_gpu_tensor_alloc(kInDim * sizeof(float));
    if (!b->x || !b->logits || !b->best || !b->dummy ||
        !ds4_gpu_tensor_write(b->x, 0, host_x,
                              kInDim * sizeof(float))) return false;
    /* Cache the raw pointers once: ds4_gpu_tensor_contents deliberately
     * synchronizes, which must not contaminate the timed path. */
    b->logits_ptr = static_cast<float *>(
        ds4_gpu_tensor_contents(b->logits));
    b->best_ptr = static_cast<candidate *>(
        ds4_gpu_tensor_contents(b->best));
    return b->logits_ptr && b->best_ptr;
}

static void free_buffers(output_buffers *b) {
    ds4_gpu_tensor_free(b->dummy);
    ds4_gpu_tensor_free(b->best);
    ds4_gpu_tensor_free(b->logits);
    ds4_gpu_tensor_free(b->x);
}

static bool run_output(output_buffers *b, const unsigned char *model,
                       uint64_t row0, uint64_t rows) {
    if (!ds4_gpu_matmul_q8_0_tensor(
            b->logits, model, kModelBytes, row0 * kRowBytes,
            kInDim, rows, b->x, 1u)) return false;
    output_top1_kernel<<<1, 256>>>(b->logits_ptr, (uint32_t)rows,
                                  (uint32_t)row0, b->best_ptr);
    return hipGetLastError() == hipSuccess;
}

static bool read_candidate(output_buffers *b, candidate *best) {
    return ds4_gpu_tensor_read(b->best, 0, best, sizeof(*best)) != 0;
}

static bool init_backend(const unsigned char *model, int model_fd) {
    return ds4_gpu_init() && ds4_gpu_set_model_fd(model_fd) &&
           ds4_gpu_set_model_map(model, kModelBytes);
}

static int worker_main(int physical, uint32_t rank, uint32_t world,
                       const unsigned char *model, int model_fd,
                       const float *host_x, int ready_fd, int go_fd) {
    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", physical);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!init_backend(model, model_fd)) return 20;

    const uint64_t row0 = kVocab * rank / world;
    const uint64_t row1 = kVocab * (rank + 1u) / world;
    output_buffers buffers{};
    if (!allocate_buffers(&buffers, row1 - row0, host_x)) return 21;
    for (int i = 0; i < kWarmup; i++)
        if (!run_output(&buffers, model, row0, row1 - row0)) return 22;
    if (!ds4_gpu_synchronize()) return 23;

    rank_handles handles{};
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 24;
    std::snprintf(handles.name, sizeof(handles.name), "%s", properties.name);
    if (!ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.dummy),
                                &handles.dummy_input) ||
        !ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.x),
                                &handles.activation) ||
        !write_all(ready_fd, &handles, sizeof(handles))) return 25;

    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        char go = 0;
        if (!read_all(go_fd, &go, sizeof(go))) return 26;
        const auto begin = std::chrono::steady_clock::now();
        candidate best{};
        if (!run_output(&buffers, model, row0, row1 - row0) ||
            !read_candidate(&buffers, &best)) return 27;
        const auto end = std::chrono::steady_clock::now();
        const rank_sample sample{best, elapsed_ms(begin, end)};
        if (!write_all(ready_fd, &sample, sizeof(sample))) return 28;
    }
    char done = 0;
    if (!read_all(go_fd, &done, sizeof(done))) return 29;
    free_buffers(&buffers);
    ds4_gpu_cleanup();
    return 0;
}

static int parse_devices(const char *text, int devices[kMaxWorld]) {
    const char *p = text;
    for (uint32_t i = 0; i < kMaxWorld; i++) {
        char *end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end == p || value < 0 || value > 255) return 0;
        devices[i] = (int)value;
        if (*end == '\0') return (int)i + 1;
        if (*end != ',') return 0;
        p = end + 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    int devices[kMaxWorld]{};
    const int parsed = argc == 3 && !std::strcmp(argv[1], "--devices") ?
        parse_devices(argv[2], devices) : 0;
    const uint32_t world = parsed > 0 ? (uint32_t)parsed : 0u;
    if (argc != 3 || std::strcmp(argv[1], "--devices") ||
        world < 2u || world > kMaxWorld) {
        std::fprintf(stderr,
                     "usage: %s --devices ROOT,PEER1[,PEER2,...,PEER5]\n",
                     argv[0]);
        return 1;
    }
    for (uint32_t i = 0; i < world; i++)
        for (uint32_t j = 0; j < i; j++)
            if (devices[i] == devices[j]) {
                std::fprintf(stderr, "devices must be unique\n");
                return 1;
            }

    const int model_fd = memfd_create("ds4-q8-output-tp", MFD_CLOEXEC);
    if (model_fd < 0 || ftruncate(model_fd, (off_t)kModelBytes)) return 2;
    unsigned char *model = static_cast<unsigned char *>(
        mmap(nullptr, (size_t)kModelBytes, PROT_READ | PROT_WRITE,
             MAP_SHARED, model_fd, 0));
    float *host_x = static_cast<float *>(
        std::malloc(kInDim * sizeof(float)));
    if (model == MAP_FAILED || !host_x) return 3;

    /* All rows intentionally produce the same score.  Besides minimizing
     * setup time, this exercises the global lower-index tie break across
     * shard boundaries. */
    std::memset(model, 1, (size_t)kModelBytes);
    for (uint64_t row = 0; row < kVocab; row++) {
        unsigned char *q = model + row * kRowBytes;
        for (uint64_t block = 0; block < kInDim / 32u; block++) {
            q[block * 34u] = 0x00;
            q[block * 34u + 1u] = 0x3c; /* fp16 1.0 */
        }
    }
    for (uint64_t i = 0; i < kInDim; i++)
        host_x[i] = (float)((int)(i % 31u) - 15) * 0.0078125f;
    setenv("DS4_ROCM_WEIGHT_ARENA_CHUNK_MB", "256", 1);

    const uint32_t peers = world - 1u;
    int ready[kMaxWorld - 1u][2]{};
    int go[kMaxWorld - 1u][2]{};
    pid_t children[kMaxWorld - 1u]{};
    for (uint32_t i = 0; i < peers; i++)
        if (pipe(ready[i]) || pipe(go[i])) return 4;
    for (uint32_t i = 0; i < peers; i++) {
        children[i] = fork();
        if (children[i] < 0) return 5;
        if (children[i] == 0) {
            for (uint32_t j = 0; j < peers; j++) {
                close(ready[j][0]);
                close(go[j][1]);
                if (j != i) {
                    close(ready[j][1]);
                    close(go[j][0]);
                }
            }
            const int rc = worker_main(devices[i + 1u], i + 1u, world,
                                       model, model_fd, host_x,
                                       ready[i][1], go[i][0]);
            _exit(rc);
        }
    }
    for (uint32_t i = 0; i < peers; i++) {
        close(ready[i][1]);
        close(go[i][0]);
    }

    rank_handles handles[kMaxWorld - 1u]{};
    for (uint32_t i = 0; i < peers; i++)
        if (!read_all(ready[i][0], &handles[i], sizeof(handles[i]))) return 6;

    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", devices[0]);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!init_backend(model, model_fd)) return 7;

    output_buffers full{};
    output_buffers shard{};
    const uint64_t root_row1 = kVocab / world;
    if (!allocate_buffers(&full, kVocab, host_x) ||
        !allocate_buffers(&shard, root_row1, host_x)) return 8;

    for (int i = 0; i < kWarmup; i++)
        if (!run_output(&full, model, 0u, kVocab)) return 9;
    if (!ds4_gpu_synchronize()) return 10;
    double full_ms = 0.0;
    candidate full_best{};
    for (int i = 0; i < kIterations; i++) {
        const auto begin = std::chrono::steady_clock::now();
        if (!run_output(&full, model, 0u, kVocab) ||
            !read_candidate(&full, &full_best)) return 11;
        full_ms += elapsed_ms(begin, std::chrono::steady_clock::now());
    }
    full_ms /= kIterations;

    ds4_rocm_tp_ipc_handle peer_inputs[kMaxWorld - 1u]{};
    ds4_rocm_tp_ipc_handle peer_outputs[kMaxWorld - 1u]{};
    for (uint32_t i = 0; i < peers; i++) {
        peer_inputs[i] = handles[i].dummy_input;
        peer_outputs[i] = handles[i].activation;
    }
    ds4_rocm_tp_star *star = ds4_rocm_tp_star_create(
        ds4_gpu_tensor_contents(shard.x),
        ds4_gpu_tensor_contents(shard.dummy),
        peer_inputs, peer_outputs, peers);
    if (!star) return 12;

    double total_ms = 0.0;
    double broadcast_ms = 0.0;
    double trigger_ms = 0.0;
    double root_ms = 0.0;
    double wait_ms = 0.0;
    double worker_ms[kMaxWorld - 1u]{};
    candidate distributed_best{};
    bool candidates_ok = true;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        const auto begin = std::chrono::steady_clock::now();
        if (!ds4_rocm_tp_star_broadcast_f32(star, kInDim, 1)) return 13;
        const auto broadcast_end = std::chrono::steady_clock::now();
        const char signal = 1;
        for (uint32_t i = 0; i < peers; i++)
            if (!write_all(go[i][1], &signal, sizeof(signal))) return 14;
        const auto trigger_end = std::chrono::steady_clock::now();
        candidate root_best{};
        if (!run_output(&shard, model, 0u, root_row1) ||
            !read_candidate(&shard, &root_best)) return 15;
        const auto root_end = std::chrono::steady_clock::now();
        distributed_best = root_best;
        for (uint32_t i = 0; i < peers; i++) {
            rank_sample sample{};
            if (!read_all(ready[i][0], &sample, sizeof(sample))) return 16;
            worker_ms[i] += iter >= kWarmup ? sample.compute_ms : 0.0;
            distributed_best = merge_candidate(distributed_best, sample.best);
            candidates_ok = candidates_ok && std::isfinite(sample.best.value);
        }
        const auto end = std::chrono::steady_clock::now();
        if (iter >= kWarmup) {
            total_ms += elapsed_ms(begin, end);
            broadcast_ms += elapsed_ms(begin, broadcast_end);
            trigger_ms += elapsed_ms(broadcast_end, trigger_end);
            root_ms += elapsed_ms(trigger_end, root_end);
            wait_ms += elapsed_ms(root_end, end);
        }
    }
    total_ms /= kIterations;
    broadcast_ms /= kIterations;
    trigger_ms /= kIterations;
    root_ms /= kIterations;
    wait_ms /= kIterations;
    for (uint32_t i = 0; i < peers; i++) worker_ms[i] /= kIterations;

    const char done = 1;
    for (uint32_t i = 0; i < peers; i++)
        if (!write_all(go[i][1], &done, sizeof(done))) return 17;
    bool children_ok = true;
    for (uint32_t i = 0; i < peers; i++) {
        int status = 0;
        children_ok = children_ok && waitpid(children[i], &status, 0) >= 0 &&
                      WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

    hipDeviceProp_t root_properties{};
    if (hipGetDeviceProperties(&root_properties, 0) != hipSuccess) return 18;
    const bool same_value =
        std::fabs(full_best.value - distributed_best.value) <=
        1.0e-6f * std::fmax(1.0f, std::fabs(full_best.value));
    const bool pass = children_ok && candidates_ok && same_value &&
                      full_best.id == distributed_best.id;
    const double saved_ms = full_ms - total_ms;

    std::printf("ROCm DeepSeek Q8 output TP%u %llu -> %llu\n", world,
                (unsigned long long)kInDim,
                (unsigned long long)kVocab);
    std::printf("  full single GPU + top1: %.4f ms\n", full_ms);
    std::printf("  TP%u synchronized + top1: %.4f ms "
                "(%.2fx, saved %.4f ms)\n",
                world, total_ms, full_ms / total_ms, saved_ms);
    std::printf("  root physical=%d %-24s compute=%.4f ms\n",
                devices[0], root_properties.name, root_ms);
    for (uint32_t i = 0; i < peers; i++)
        std::printf("  peer physical=%d %-24s compute=%.4f ms\n",
                    devices[i + 1u], handles[i].name, worker_ms[i]);
    std::printf("  stages: broadcast=%.4f ms trigger=%.4f ms "
                "root_compute=%.4f ms residual_wait=%.4f ms\n",
                broadcast_ms, trigger_ms, root_ms, wait_ms);
    std::printf("  candidate: full=(%u, %.7g) tp=(%u, %.7g)\n",
                full_best.id, full_best.value,
                distributed_best.id, distributed_best.value);
    std::printf("  projected decode gain at 63.17 ms/token: %.2f%%\n",
                100.0 * saved_ms / 63.17);
    std::printf("  verification: %s\n", pass ? "PASS" : "FAIL");

    ds4_rocm_tp_star_destroy(star);
    free_buffers(&shard);
    free_buffers(&full);
    ds4_gpu_cleanup();
    std::free(host_x);
    munmap(model, (size_t)kModelBytes);
    close(model_fd);
    return pass ? 0 : 19;
}
