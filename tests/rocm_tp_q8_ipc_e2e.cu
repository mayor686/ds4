/* End-to-end process-per-GPU tensor parallelism for the complete Q8 shared
 * expert used by DeepSeek-V4-Flash: 4096 -> 2048 gate/up, SwiGLU, then
 * 2048 -> 4096 down.  Column-sliced gate/up and row-sliced down require one
 * final FP32 reduction rather than one collective per projection. */

#include <hip/hip_runtime.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ds4_gpu.h"
#include "ds4_rocm_tp.h"

static constexpr uint64_t kInDim = 4096u;
static constexpr uint64_t kSharedDim = 2048u;
static constexpr uint64_t kOutDim = 4096u;
static constexpr uint64_t kGateRowBytes = (kInDim / 32u) * 34u;
static constexpr uint64_t kDownRowBytes = (kSharedDim / 32u) * 34u;
static constexpr uint64_t kGateBytes = kSharedDim * kGateRowBytes;
static constexpr uint64_t kDownBytes = kOutDim * kDownRowBytes;
static constexpr uint64_t kGateOffset = 0u;
static constexpr uint64_t kUpOffset = kGateOffset + kGateBytes;
static constexpr uint64_t kDownOffset = kUpOffset + kGateBytes;
static constexpr uint64_t kModelBytes = kDownOffset + kDownBytes;
static constexpr int kWarmup = 5;
static constexpr int kIterations = 100;

typedef struct {
    ds4_rocm_tp_ipc_handle partial;
    ds4_rocm_tp_ipc_handle reduced;
    char name[64];
} rank_handles;

typedef struct {
    double compute_ms;
} rank_sample;

typedef struct {
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *partial;
    ds4_gpu_tensor *reduced;
} shared_buffers;

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

static void pack_q8_tensor(unsigned char *dst, uint64_t in_dim,
                           uint64_t out_dim, uint32_t seed) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *q = dst + (row * blocks + block) * 34u;
            q[0] = 0x00;
            q[1] = 0x24; /* IEEE fp16 0.015625 */
            for (uint64_t i = 0; i < 32u; i++) {
                const int value =
                    (int)((row * 13u + block * 7u + i * 5u + seed) % 15u) - 7;
                q[2u + i] = (unsigned char)(int8_t)value;
            }
        }
    }
}

static bool allocate_buffers(shared_buffers *b, uint64_t lane_dim,
                             const float *host_x) {
    memset(b, 0, sizeof(*b));
    b->x = ds4_gpu_tensor_alloc(kInDim * sizeof(float));
    b->gate = ds4_gpu_tensor_alloc(lane_dim * sizeof(float));
    b->up = ds4_gpu_tensor_alloc(lane_dim * sizeof(float));
    b->mid = ds4_gpu_tensor_alloc(lane_dim * sizeof(float));
    b->partial = ds4_gpu_tensor_alloc(kOutDim * sizeof(float));
    b->reduced = ds4_gpu_tensor_alloc(kOutDim * sizeof(float));
    return b->x && b->gate && b->up && b->mid && b->partial && b->reduced &&
           ds4_gpu_tensor_write(b->x, 0, host_x,
                                kInDim * sizeof(float));
}

static void free_buffers(shared_buffers *b) {
    ds4_gpu_tensor_free(b->reduced);
    ds4_gpu_tensor_free(b->partial);
    ds4_gpu_tensor_free(b->mid);
    ds4_gpu_tensor_free(b->up);
    ds4_gpu_tensor_free(b->gate);
    ds4_gpu_tensor_free(b->x);
}

static bool run_shard(shared_buffers *b, const unsigned char *model,
                      uint32_t rank, uint32_t world) {
    if (world == 0u || kSharedDim % world != 0u) return false;
    const uint64_t lane_dim = kSharedDim / world;
    const uint64_t lane_off = (uint64_t)rank * lane_dim;
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               b->gate, b->up, b->mid, model, kModelBytes,
               kGateOffset + lane_off * kGateRowBytes,
               kUpOffset + lane_off * kGateRowBytes,
               kInDim, lane_dim, b->x, 7.0f) &&
           ds4_gpu_matmul_q8_0_kslice_tensor(
               b->partial, model, kModelBytes, kDownOffset,
               kSharedDim, lane_off, lane_dim, kOutDim, b->mid, 0u);
}

static bool run_full(shared_buffers *b, ds4_gpu_tensor *full,
                     const unsigned char *model) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               b->gate, b->up, b->mid, model, kModelBytes,
               kGateOffset, kUpOffset, kInDim, kSharedDim, b->x, 7.0f) &&
           ds4_gpu_matmul_q8_0_tensor(
               full, model, kModelBytes, kDownOffset,
               kSharedDim, kOutDim, b->mid, 1u);
}

static bool init_backend(const unsigned char *model, int model_fd) {
    return ds4_gpu_init() && ds4_gpu_set_model_fd(model_fd) &&
           ds4_gpu_set_model_map(model, kModelBytes) &&
           ds4_gpu_cache_model_range(model, kModelBytes, 0, kModelBytes,
                                     "tp_shared_q8");
}

static int worker_main(int physical, const unsigned char *model, int model_fd,
                       const float *host_x, int ready_fd, int go_fd) {
    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", physical);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!init_backend(model, model_fd)) return 20;
    shared_buffers buffers{};
    if (!allocate_buffers(&buffers, kSharedDim / 2u, host_x) ||
        !run_shard(&buffers, model, 1u, 2u) || !ds4_gpu_synchronize())
        return 21;
    rank_handles handles{};
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 22;
    std::snprintf(handles.name, sizeof(handles.name), "%s", properties.name);
    if (!ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.partial),
                                &handles.partial) ||
        !ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.reduced),
                                &handles.reduced) ||
        !write_all(ready_fd, &handles, sizeof(handles))) return 23;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        char go = 0;
        if (!read_all(go_fd, &go, 1u)) return 24;
        const auto begin = std::chrono::steady_clock::now();
        if (!run_shard(&buffers, model, 1u, 2u) || !ds4_gpu_synchronize())
            return 25;
        const auto end = std::chrono::steady_clock::now();
        const rank_sample sample{
            std::chrono::duration<double>(end - begin).count() * 1.0e3};
        if (!write_all(ready_fd, &sample, sizeof(sample))) return 26;
    }
    char done = 0;
    if (!read_all(go_fd, &done, 1u)) return 27;
    free_buffers(&buffers);
    ds4_gpu_cleanup();
    return 0;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    int root_device = -1, peer_device = -1;
    if (argc != 3 || std::strcmp(argv[1], "--devices") ||
        std::sscanf(argv[2], "%d,%d", &root_device, &peer_device) != 2 ||
        root_device < 0 || peer_device < 0 || root_device == peer_device) {
        std::fprintf(stderr, "usage: %s --devices ROOT,PEER\n", argv[0]);
        return 1;
    }
    unsigned char *model = (unsigned char *)malloc((size_t)kModelBytes);
    float *host_x = (float *)malloc(kInDim * sizeof(float));
    if (!model || !host_x) return 2;
    pack_q8_tensor(model + kGateOffset, kInDim, kSharedDim, 3u);
    pack_q8_tensor(model + kUpOffset, kInDim, kSharedDim, 7u);
    pack_q8_tensor(model + kDownOffset, kSharedDim, kOutDim, 11u);
    for (uint64_t i = 0; i < kInDim; i++)
        host_x[i] = (float)((int)(i % 61u) - 30) * 0.00390625f;
    FILE *model_file = tmpfile();
    if (!model_file ||
        fwrite(model, 1u, (size_t)kModelBytes, model_file) != kModelBytes ||
        fflush(model_file)) return 3;
    const int model_fd = fileno(model_file);
    setenv("DS4_ROCM_WEIGHT_ARENA_CHUNK_MB", "256", 1);

    int ready[2]{}, go[2]{};
    if (pipe(ready) || pipe(go)) return 4;
    const pid_t child = fork();
    if (child < 0) return 5;
    if (child == 0) {
        close(ready[0]);
        close(go[1]);
        const int rc = worker_main(peer_device, model, model_fd, host_x,
                                   ready[1], go[0]);
        _exit(rc);
    }
    close(ready[1]);
    close(go[0]);
    rank_handles handles{};
    if (!read_all(ready[0], &handles, sizeof(handles))) return 6;

    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", root_device);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!init_backend(model, model_fd)) return 7;
    shared_buffers shard{};
    shared_buffers full_buffers{};
    if (!allocate_buffers(&shard, kSharedDim / 2u, host_x) ||
        !allocate_buffers(&full_buffers, kSharedDim, host_x)) return 8;
    ds4_gpu_tensor *full =
        ds4_gpu_tensor_alloc(kOutDim * sizeof(float));
    if (!full) return 9;
    for (int i = 0; i < kWarmup; i++)
        if (!run_full(&full_buffers, full, model)) return 10;
    if (!ds4_gpu_synchronize()) return 11;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; i++)
        if (!run_full(&full_buffers, full, model)) return 12;
    if (!ds4_gpu_synchronize()) return 13;
    auto end = std::chrono::steady_clock::now();
    const double full_ms =
        std::chrono::duration<double>(end - begin).count() * 1.0e3 /
        kIterations;

    ds4_rocm_tp_star *star = ds4_rocm_tp_star_create(
        ds4_gpu_tensor_contents(shard.partial),
        ds4_gpu_tensor_contents(shard.reduced),
        &handles.partial, &handles.reduced, 1u);
    if (!star) return 14;
    double total_ms = 0.0, trigger_ms = 0.0, root_ms = 0.0;
    double peer_ms = 0.0, wait_ms = 0.0, collective_ms = 0.0;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        const auto iter_begin = std::chrono::steady_clock::now();
        const char signal = 1;
        if (!write_all(go[1], &signal, 1u)) return 15;
        const auto trigger_end = std::chrono::steady_clock::now();
        if (!run_shard(&shard, model, 0u, 2u) || !ds4_gpu_synchronize())
            return 16;
        const auto root_end = std::chrono::steady_clock::now();
        rank_sample sample{};
        if (!read_all(ready[0], &sample, sizeof(sample))) return 17;
        const auto ready_end = std::chrono::steady_clock::now();
        if (!ds4_rocm_tp_star_allreduce_f32(star, kOutDim, 1)) return 18;
        const auto iter_end = std::chrono::steady_clock::now();
        if (iter >= kWarmup) {
            total_ms += std::chrono::duration<double>(iter_end - iter_begin).count() * 1.0e3;
            trigger_ms += std::chrono::duration<double>(trigger_end - iter_begin).count() * 1.0e3;
            root_ms += std::chrono::duration<double>(root_end - trigger_end).count() * 1.0e3;
            wait_ms += std::chrono::duration<double>(ready_end - root_end).count() * 1.0e3;
            collective_ms += std::chrono::duration<double>(iter_end - ready_end).count() * 1.0e3;
            peer_ms += sample.compute_ms;
        }
    }
    total_ms /= kIterations;
    trigger_ms /= kIterations;
    root_ms /= kIterations;
    peer_ms /= kIterations;
    wait_ms /= kIterations;
    collective_ms /= kIterations;

    std::vector<float> host_full(kOutDim), host_reduced(kOutDim);
    if (!ds4_gpu_tensor_read(full, 0, host_full.data(),
                             kOutDim * sizeof(float)) ||
        !ds4_gpu_tensor_read(shard.reduced, 0, host_reduced.data(),
                             kOutDim * sizeof(float))) return 19;
    double sq = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (uint64_t i = 0; i < kOutDim; i++) {
        const double diff = (double)host_full[i] - host_reduced[i];
        sq += diff * diff;
        if (std::fabs(diff) > max_abs) max_abs = std::fabs(diff);
        if (std::fabs(host_full[i]) > max_ref) max_ref = std::fabs(host_full[i]);
    }
    const double rms = std::sqrt(sq / kOutDim);
    const double rel = max_ref ? max_abs / max_ref : max_abs;
    const char done = 1;
    if (!write_all(go[1], &done, 1u)) return 20;
    int status = 0;
    const bool child_ok = waitpid(child, &status, 0) >= 0 &&
                          WIFEXITED(status) && WEXITSTATUS(status) == 0;
    const bool pass = child_ok && std::isfinite(rms) && rel <= 5.0e-4;
    hipDeviceProp_t root_properties{};
    if (hipGetDeviceProperties(&root_properties, 0) != hipSuccess) return 21;
    std::printf("ROCm Flash Q8 shared-expert TP2 4096 -> 2048 -> 4096\n");
    std::printf("  full single GPU: %.4f ms\n", full_ms);
    std::printf("  TP2 synchronized end-to-end: %.4f ms (%.2fx)\n",
                total_ms, full_ms / total_ms);
    std::printf("  root physical=%d %-24s compute=%.4f ms\n",
                root_device, root_properties.name, root_ms);
    std::printf("  peer physical=%d %-24s compute=%.4f ms\n",
                peer_device, handles.name, peer_ms);
    std::printf("  stages: trigger=%.4f ms root_compute=%.4f ms "
                "residual_wait=%.4f ms collective=%.4f ms\n",
                trigger_ms, root_ms, wait_ms, collective_ms);
    std::printf("  numerical: rms=%g max_abs=%g max_ref=%g rel=%g\n",
                rms, max_abs, max_ref, rel);
    std::printf("  verification: %s\n", pass ? "PASS" : "FAIL");

    ds4_rocm_tp_star_destroy(star);
    ds4_gpu_tensor_free(full);
    free_buffers(&full_buffers);
    free_buffers(&shard);
    ds4_gpu_cleanup();
    fclose(model_file);
    free(host_x);
    free(model);
    return pass ? 0 : 22;
}
