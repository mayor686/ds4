/* Decode-sized expert-parallel gate for DeepSeek-V4-Flash 0731:
 * IQ2_XXS gate/up, Q2_K down, six routed experts and one process per GPU. */

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

static constexpr uint32_t kTotalExperts = 8u;
static constexpr uint32_t kActiveExperts = 6u;
static constexpr uint32_t kInDim = 4096u;
static constexpr uint32_t kMidDim = 2048u;
static constexpr uint32_t kOutDim = 4096u;
static constexpr uint64_t kIq2BlockBytes = 66u;
static constexpr uint64_t kQ2BlockBytes = 84u;
static constexpr uint64_t kGateRowBytes = (kInDim / 256u) * kIq2BlockBytes;
static constexpr uint64_t kGateExpertBytes = kMidDim * kGateRowBytes;
static constexpr uint64_t kDownRowBytes = (kMidDim / 256u) * kQ2BlockBytes;
static constexpr uint64_t kDownExpertBytes = kOutDim * kDownRowBytes;
static constexpr uint64_t kGateOffset = 0u;
static constexpr uint64_t kUpOffset =
    kGateOffset + kTotalExperts * kGateExpertBytes;
static constexpr uint64_t kDownOffset =
    kUpOffset + kTotalExperts * kGateExpertBytes;
static constexpr uint64_t kModelBytes =
    kDownOffset + kTotalExperts * kDownExpertBytes;
static constexpr int kWarmup = 5;
static constexpr int kIterations = 100;

typedef struct {
    ds4_rocm_tp_ipc_handle partial;
    ds4_rocm_tp_ipc_handle reduced;
    char name[64];
    uint64_t memory_bytes;
} rank_handles;

typedef struct {
    double compute_ms;
} rank_sample;

typedef struct {
    ds4_gpu_tensor *out;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *down;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *weights;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *reduced;
} moe_buffers;

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

static void fill_iq2(unsigned char *base, uint64_t bytes, uint32_t seed) {
    for (uint64_t off = 0; off < bytes; off += kIq2BlockBytes) {
        base[off + 0u] = 0x00;
        base[off + 1u] = 0x24; /* fp16 scale 0.015625 */
        uint16_t *qs = reinterpret_cast<uint16_t *>(base + off + 2u);
        for (uint32_t i = 0; i < 32u; i += 4u) {
            const uint32_t word = seed + (uint32_t)(off / kIq2BlockBytes) + i;
            qs[i + 0u] = (uint16_t)(word * 13u);
            qs[i + 1u] = (uint16_t)(word * 29u);
            qs[i + 2u] = (uint16_t)(word * 7u);
            qs[i + 3u] = (uint16_t)((word * 11u) & 0x0fffffffu);
        }
    }
}

static void fill_q2(unsigned char *base, uint64_t bytes, uint32_t seed) {
    for (uint64_t off = 0; off < bytes; off += kQ2BlockBytes) {
        unsigned char *block = base + off;
        for (uint32_t i = 0; i < 16u; i++) block[i] = 0x01u;
        for (uint32_t i = 0; i < 64u; i++)
            block[16u + i] = (unsigned char)(0x55u ^ (seed + i));
        block[80u] = 0x00;
        block[81u] = 0x18; /* fp16 scale 0.001953125 */
        block[82u] = 0x00;
        block[83u] = 0x00;
    }
}

static void build_model(unsigned char *model) {
    for (uint32_t expert = 0; expert < kTotalExperts; expert++) {
        fill_iq2(model + kGateOffset + (uint64_t)expert * kGateExpertBytes,
                 kGateExpertBytes, 101u + expert);
        fill_iq2(model + kUpOffset + (uint64_t)expert * kGateExpertBytes,
                 kGateExpertBytes, 211u + expert);
        fill_q2(model + kDownOffset + (uint64_t)expert * kDownExpertBytes,
                kDownExpertBytes, 17u + expert);
    }
}

static bool cache_owner_ranges(const unsigned char *model, uint64_t model_size,
                               uint32_t base, uint32_t count) {
    return ds4_gpu_cache_model_range(
               model, model_size,
               kGateOffset + (uint64_t)base * kGateExpertBytes,
               (uint64_t)count * kGateExpertBytes, "ep_gate") &&
           ds4_gpu_cache_model_range(
               model, model_size,
               kUpOffset + (uint64_t)base * kGateExpertBytes,
               (uint64_t)count * kGateExpertBytes, "ep_up") &&
           ds4_gpu_cache_model_range(
               model, model_size,
               kDownOffset + (uint64_t)base * kDownExpertBytes,
               (uint64_t)count * kDownExpertBytes, "ep_down");
}

static bool allocate_buffers(moe_buffers *b, const float *host_x,
                             const int32_t *host_selected,
                             const float *host_weights) {
    memset(b, 0, sizeof(*b));
    b->out = ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float));
    b->gate = ds4_gpu_tensor_alloc(
        (uint64_t)kActiveExperts * kMidDim * sizeof(float));
    b->up = ds4_gpu_tensor_alloc(
        (uint64_t)kActiveExperts * kMidDim * sizeof(float));
    b->mid = ds4_gpu_tensor_alloc(
        (uint64_t)kActiveExperts * kMidDim * sizeof(float));
    b->down = ds4_gpu_tensor_alloc(
        (uint64_t)kActiveExperts * kOutDim * sizeof(float));
    b->selected = ds4_gpu_tensor_alloc(
        (uint64_t)kActiveExperts * sizeof(int32_t));
    b->weights = ds4_gpu_tensor_alloc(
        (uint64_t)kActiveExperts * sizeof(float));
    b->x = ds4_gpu_tensor_alloc((uint64_t)kInDim * sizeof(float));
    b->reduced = ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float));
    return b->out && b->gate && b->up && b->mid && b->down && b->selected &&
           b->weights && b->x && b->reduced &&
           ds4_gpu_tensor_write(b->x, 0, host_x,
                                (uint64_t)kInDim * sizeof(float)) &&
           ds4_gpu_tensor_write(b->selected, 0, host_selected,
                                (uint64_t)kActiveExperts * sizeof(int32_t)) &&
           ds4_gpu_tensor_write(b->weights, 0, host_weights,
                                (uint64_t)kActiveExperts * sizeof(float));
}

static void free_buffers(moe_buffers *b) {
    ds4_gpu_tensor_free(b->reduced);
    ds4_gpu_tensor_free(b->x);
    ds4_gpu_tensor_free(b->weights);
    ds4_gpu_tensor_free(b->selected);
    ds4_gpu_tensor_free(b->down);
    ds4_gpu_tensor_free(b->mid);
    ds4_gpu_tensor_free(b->up);
    ds4_gpu_tensor_free(b->gate);
    ds4_gpu_tensor_free(b->out);
}

static bool run_owned(moe_buffers *b, const unsigned char *model,
                      uint32_t expert_base, uint32_t expert_count) {
    return ds4_gpu_routed_moe_one_owned_tensor(
        b->out, b->gate, b->up, b->mid, b->down,
        model, kModelBytes, kGateOffset, kUpOffset, kDownOffset,
        16u, 10u, kGateExpertBytes, kGateRowBytes,
        kDownExpertBytes, kDownRowBytes, kInDim, kMidDim, kOutDim,
        b->selected, b->weights, kTotalExperts, kActiveExperts,
        expert_base, expert_count, 7.0f, b->x, nullptr, false, nullptr) != 0;
}

static bool run_full(moe_buffers *b, ds4_gpu_tensor *out,
                     const unsigned char *model) {
    return ds4_gpu_routed_moe_one_tensor(
        out, b->gate, b->up, b->mid, b->down,
        model, kModelBytes, kGateOffset, kUpOffset, kDownOffset,
        16u, 10u, kGateExpertBytes, kGateRowBytes,
        kDownExpertBytes, kDownRowBytes, kInDim, kMidDim, kOutDim,
        b->selected, b->weights, kTotalExperts, kActiveExperts,
        7.0f, b->x, nullptr, UINT32_MAX, true) != 0;
}

static int worker_main(int physical, const unsigned char *model, int model_fd,
                       const float *host_x, const int32_t *host_selected,
                       const float *host_weights, int ready_fd, int go_fd) {
    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", physical);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!ds4_gpu_init() || !ds4_gpu_set_model_fd(model_fd) ||
        !ds4_gpu_set_model_map(model, kModelBytes) ||
        !cache_owner_ranges(model, kModelBytes, 4u, 4u)) return 20;
    moe_buffers buffers{};
    if (!allocate_buffers(&buffers, host_x, host_selected, host_weights) ||
        !run_owned(&buffers, model, 4u, 4u) || !ds4_gpu_synchronize())
        return 21;
    rank_handles handles{};
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 22;
    std::snprintf(handles.name, sizeof(handles.name), "%s", properties.name);
    handles.memory_bytes = (uint64_t)properties.totalGlobalMem;
    if (!ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.out),
                                &handles.partial) ||
        !ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.reduced),
                                &handles.reduced) ||
        !write_all(ready_fd, &handles, sizeof(handles))) return 23;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        char go = 0;
        if (!read_all(go_fd, &go, 1u)) return 24;
        const auto begin = std::chrono::steady_clock::now();
        if (!run_owned(&buffers, model, 4u, 4u) || !ds4_gpu_synchronize())
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
    if (argc != 3 || std::strcmp(argv[1], "--devices") ||
        std::strchr(argv[2], ',') == nullptr) {
        std::fprintf(stderr, "usage: %s --devices ROOT,PEER\n", argv[0]);
        return 1;
    }
    int root_device = -1, peer_device = -1;
    if (std::sscanf(argv[2], "%d,%d", &root_device, &peer_device) != 2 ||
        root_device < 0 || peer_device < 0 || root_device == peer_device)
        return 1;
    unsigned char *model = (unsigned char *)malloc((size_t)kModelBytes);
    float *host_x = (float *)malloc((size_t)kInDim * sizeof(float));
    if (!model || !host_x) return 2;
    build_model(model);
    for (uint32_t i = 0; i < kInDim; i++)
        host_x[i] = (float)((int)(i % 67u) - 33) * 0.00390625f;
    const int32_t host_selected[kActiveExperts] = {0, 1, 2, 4, 5, 6};
    const float host_weights[kActiveExperts] = {
        0.11f, 0.17f, 0.19f, 0.13f, 0.23f, 0.17f};
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
                                   host_selected, host_weights,
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
    if (!ds4_gpu_init() || !ds4_gpu_set_model_fd(model_fd) ||
        !ds4_gpu_set_model_map(model, kModelBytes) ||
        !cache_owner_ranges(model, kModelBytes, 0u, kTotalExperts)) return 7;
    moe_buffers buffers{};
    if (!allocate_buffers(&buffers, host_x, host_selected, host_weights))
        return 8;
    ds4_gpu_tensor *reference =
        ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float));
    if (!reference) return 9;
    for (int i = 0; i < kWarmup; i++)
        if (!run_full(&buffers, reference, model)) return 10;
    if (!ds4_gpu_synchronize()) return 11;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; i++)
        if (!run_full(&buffers, reference, model)) return 12;
    if (!ds4_gpu_synchronize()) return 13;
    auto end = std::chrono::steady_clock::now();
    const double full_ms =
        std::chrono::duration<double>(end - begin).count() * 1.0e3 /
        kIterations;

    ds4_rocm_tp_star *star = ds4_rocm_tp_star_create(
        ds4_gpu_tensor_contents(buffers.out),
        ds4_gpu_tensor_contents(buffers.reduced),
        &handles.partial, &handles.reduced, 1u);
    if (!star) return 14;
    double total_ms = 0.0, root_compute_ms = 0.0, peer_compute_ms = 0.0;
    double wait_ms = 0.0, collective_ms = 0.0, trigger_ms = 0.0;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        const auto iter_begin = std::chrono::steady_clock::now();
        const char signal = 1;
        if (!write_all(go[1], &signal, 1u)) return 15;
        const auto trigger_end = std::chrono::steady_clock::now();
        if (!run_owned(&buffers, model, 0u, 4u) || !ds4_gpu_synchronize())
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
            root_compute_ms += std::chrono::duration<double>(root_end - trigger_end).count() * 1.0e3;
            wait_ms += std::chrono::duration<double>(ready_end - root_end).count() * 1.0e3;
            collective_ms += std::chrono::duration<double>(iter_end - ready_end).count() * 1.0e3;
            peer_compute_ms += sample.compute_ms;
        }
    }
    total_ms /= kIterations;
    trigger_ms /= kIterations;
    root_compute_ms /= kIterations;
    wait_ms /= kIterations;
    collective_ms /= kIterations;
    peer_compute_ms /= kIterations;

    std::vector<float> host_reference(kOutDim), host_reduced(kOutDim);
    if (!ds4_gpu_tensor_read(reference, 0, host_reference.data(),
                             (uint64_t)kOutDim * sizeof(float)) ||
        !ds4_gpu_tensor_read(buffers.reduced, 0, host_reduced.data(),
                             (uint64_t)kOutDim * sizeof(float))) return 19;
    double sq = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (uint32_t i = 0; i < kOutDim; i++) {
        const double diff = (double)host_reference[i] - host_reduced[i];
        sq += diff * diff;
        if (std::fabs(diff) > max_abs) max_abs = std::fabs(diff);
        if (std::fabs(host_reference[i]) > max_ref)
            max_ref = std::fabs(host_reference[i]);
    }
    const double rms = std::sqrt(sq / kOutDim);
    const double rel = max_ref ? max_abs / max_ref : max_abs;
    const char done = 1;
    if (!write_all(go[1], &done, 1u)) return 20;
    int status = 0;
    const bool child_ok = waitpid(child, &status, 0) >= 0 &&
                          WIFEXITED(status) && WEXITSTATUS(status) == 0;
    hipDeviceProp_t root_properties{};
    if (hipGetDeviceProperties(&root_properties, 0) != hipSuccess) return 21;
    const bool pass = child_ok && std::isfinite(rms) && rel <= 5.0e-4;
    std::printf("ROCm Flash-0731 IQ2/Q2 EP2 decode, six active experts\n");
    std::printf("  full single GPU: %.4f ms\n", full_ms);
    std::printf("  EP2 synchronized end-to-end: %.4f ms (%.2fx)\n",
                total_ms, full_ms / total_ms);
    std::printf("  root physical=%d %-24s compute=%.4f ms\n",
                root_device, root_properties.name, root_compute_ms);
    std::printf("  peer physical=%d %-24s compute=%.4f ms\n",
                peer_device, handles.name, peer_compute_ms);
    std::printf("  stages: trigger=%.4f ms root_compute=%.4f ms "
                "residual_wait=%.4f ms collective=%.4f ms\n",
                trigger_ms, root_compute_ms, wait_ms, collective_ms);
    std::printf("  numerical: rms=%g max_abs=%g max_ref=%g rel=%g\n",
                rms, max_abs, max_ref, rel);
    std::printf("  verification: %s\n", pass ? "PASS" : "FAIL");

    ds4_rocm_tp_star_destroy(star);
    ds4_gpu_tensor_free(reference);
    free_buffers(&buffers);
    ds4_gpu_cleanup();
    fclose(model_file);
    free(host_x);
    free(model);
    return pass ? 0 : 22;
}
