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

static constexpr uint32_t kTotalExperts = 256u;
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
static constexpr uint64_t kRoutedBytes =
    kDownOffset + kTotalExperts * kDownExpertBytes;
static constexpr uint64_t kSharedRowBytes = (kInDim / 32u) * 34u;
static constexpr uint64_t kSharedDownRowBytes = (kMidDim / 32u) * 34u;
static constexpr uint64_t kSharedGateBytes = kMidDim * kSharedRowBytes;
static constexpr uint64_t kSharedDownBytes = kOutDim * kSharedDownRowBytes;
static constexpr uint64_t kSharedGateOffset = kRoutedBytes;
static constexpr uint64_t kSharedUpOffset =
    kSharedGateOffset + kSharedGateBytes;
static constexpr uint64_t kSharedDownOffset =
    kSharedUpOffset + kSharedGateBytes;
static constexpr uint64_t kModelBytes =
    kSharedDownOffset + kSharedDownBytes;
static constexpr int kWarmup = 50;
static constexpr int kIterations = 100;
static constexpr int kMaxRanks = 6;

typedef struct {
    ds4_rocm_tp_ipc_handle partial;
    ds4_rocm_tp_ipc_handle reduced;
    ds4_rocm_tp_ipc_handle x;
    char name[64];
    uint64_t memory_bytes;
    double remap_ms;
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

static void fill_q8(unsigned char *base, uint64_t in_dim, uint64_t out_dim,
                    uint32_t seed) {
    const uint64_t blocks = in_dim / 32u;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            unsigned char *q = base + (row * blocks + block) * 34u;
            q[0] = 0x00;
            q[1] = 0x24; /* fp16 scale 0.015625 */
            for (uint64_t i = 0; i < 32u; i++) {
                const int value =
                    (int)((row * 13u + block * 7u + i * 5u + seed) % 15u) - 7;
                q[2u + i] = (unsigned char)(int8_t)value;
            }
        }
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
    fill_q8(model + kSharedGateOffset, kInDim, kMidDim, 3u);
    fill_q8(model + kSharedUpOffset, kInDim, kMidDim, 7u);
    fill_q8(model + kSharedDownOffset, kMidDim, kOutDim, 11u);
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

static bool cache_shared_ranges(const unsigned char *model,
                                uint64_t model_size) {
    return ds4_gpu_cache_model_range(model, model_size, kSharedGateOffset,
                                     kSharedGateBytes, "ep_shared_gate") &&
           ds4_gpu_cache_model_range(model, model_size, kSharedUpOffset,
                                     kSharedGateBytes, "ep_shared_up") &&
           ds4_gpu_cache_model_range(model, model_size, kSharedDownOffset,
                                     kSharedDownBytes, "ep_shared_down");
}

static bool remap_owner_ranges(const unsigned char *model,
                               uint64_t model_size,
                               uint32_t base,
                               uint32_t count,
                               bool include_shared) {
    uint64_t offsets[6]{};
    uint64_t sizes[6]{};
    uint32_t n = 0u;
    uint64_t max_bytes = 0u;
    if (count != 0u) {
        const uint64_t gate_bytes = (uint64_t)count * kGateExpertBytes;
        const uint64_t down_bytes = (uint64_t)count * kDownExpertBytes;
        offsets[n] = kGateOffset + (uint64_t)base * kGateExpertBytes;
        sizes[n++] = gate_bytes;
        offsets[n] = kUpOffset + (uint64_t)base * kGateExpertBytes;
        sizes[n++] = gate_bytes;
        offsets[n] = kDownOffset + (uint64_t)base * kDownExpertBytes;
        sizes[n++] = down_bytes;
        max_bytes = gate_bytes > down_bytes ? gate_bytes : down_bytes;
    }
    if (include_shared) {
        offsets[n] = kSharedGateOffset;
        sizes[n++] = kSharedGateBytes;
        offsets[n] = kSharedUpOffset;
        sizes[n++] = kSharedGateBytes;
        offsets[n] = kSharedDownOffset;
        sizes[n++] = kSharedDownBytes;
        if (kSharedGateBytes > max_bytes) max_bytes = kSharedGateBytes;
        if (kSharedDownBytes > max_bytes) max_bytes = kSharedDownBytes;
    }
    return n != 0u && ds4_gpu_rocm_replace_model_map_spans(
            model, model_size, offsets, sizes, n, max_bytes) != 0;
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

static bool run_shared(moe_buffers *b, ds4_gpu_tensor *out,
                       const unsigned char *model) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
               b->gate, b->up, b->mid, model, kModelBytes,
               kSharedGateOffset, kSharedUpOffset,
               kInDim, kMidDim, b->x, 7.0f) &&
           ds4_gpu_matmul_q8_0_tensor(
               out, model, kModelBytes, kSharedDownOffset,
               kMidDim, kOutDim, b->mid, 1u);
}

static int worker_main(int physical, uint32_t rank, uint32_t world,
                       const unsigned char *model, int model_fd,
                       const float *host_x, const int32_t *host_selected,
                       const float *host_weights, bool require_broadcast,
                       bool remap_residency,
                       int ready_fd, int go_fd) {
    const uint32_t expert_base = kTotalExperts * rank / world;
    const uint32_t expert_end = kTotalExperts * (rank + 1u) / world;
    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", physical);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!ds4_gpu_init() || !ds4_gpu_set_model_fd(model_fd) ||
        !ds4_gpu_set_model_map(model, kModelBytes) ||
        !(remap_residency ?
              cache_owner_ranges(model, kModelBytes, 0u, kTotalExperts) :
              cache_owner_ranges(model, kModelBytes, expert_base,
                                  expert_end - expert_base))) return 20;
    moe_buffers buffers{};
    if (!allocate_buffers(&buffers, host_x, host_selected, host_weights))
        return 21;
    double remap_ms = 0.0;
    if (remap_residency) {
        if (!run_full(&buffers, buffers.out, model) || !ds4_gpu_synchronize())
            return 21;
        const auto remap_begin = std::chrono::steady_clock::now();
        if (!remap_owner_ranges(model, kModelBytes, expert_base,
                                expert_end - expert_base, false)) return 21;
        const auto remap_end = std::chrono::steady_clock::now();
        remap_ms = std::chrono::duration<double>(
                remap_end - remap_begin).count() * 1.0e3;
    }
    if (!run_owned(&buffers, model, expert_base,
                   expert_end - expert_base) || !ds4_gpu_synchronize())
        return 21;
    /* Make the overlap result depend on the root broadcast instead of the
     * identical host-side initialization performed by every process. */
    if (require_broadcast &&
        (!ds4_gpu_tensor_fill_f32(buffers.x, -1.0f, kInDim) ||
         !ds4_gpu_synchronize())) return 21;
    rank_handles handles{};
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 22;
    std::snprintf(handles.name, sizeof(handles.name), "%s", properties.name);
    handles.memory_bytes = (uint64_t)properties.totalGlobalMem;
    handles.remap_ms = remap_ms;
    if (!ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.out),
                                &handles.partial) ||
        !ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.reduced),
                                &handles.reduced) ||
        !ds4_rocm_tp_ipc_export(ds4_gpu_tensor_contents(buffers.x),
                                &handles.x) ||
        !write_all(ready_fd, &handles, sizeof(handles))) return 23;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        char go = 0;
        if (!read_all(go_fd, &go, 1u)) return 24;
        const auto begin = std::chrono::steady_clock::now();
        if (!run_owned(&buffers, model, expert_base,
                       expert_end - expert_base) || !ds4_gpu_synchronize())
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

static int parse_devices(const char *arg, int *devices, int cap) {
    if (!arg || !*arg) return 0;
    char *copy = strdup(arg);
    if (!copy) return 0;
    int count = 0;
    char *save = nullptr;
    for (char *token = strtok_r(copy, ",", &save); token;
         token = strtok_r(nullptr, ",", &save)) {
        char *end = nullptr;
        const long value = strtol(token, &end, 10);
        if (count == cap || !end || *end || value < 0 || value > 255) {
            free(copy);
            return 0;
        }
        for (int i = 0; i < count; i++) {
            if (devices[i] == value) {
                free(copy);
                return 0;
            }
        }
        devices[count++] = (int)value;
    }
    free(copy);
    return count;
}

static bool parse_owner_counts(const char *arg, uint32_t *counts, int world) {
    if (!arg || !*arg) return false;
    char *copy = strdup(arg);
    if (!copy) return false;
    int count = 0;
    uint32_t sum = 0u;
    char *save = nullptr;
    for (char *token = strtok_r(copy, ",", &save); token;
         token = strtok_r(nullptr, ",", &save)) {
        char *end = nullptr;
        const long value = strtol(token, &end, 10);
        if (count == world || !end || *end || value < 0 ||
            value > (long)kActiveExperts) {
            free(copy);
            return false;
        }
        counts[count++] = (uint32_t)value;
        sum += (uint32_t)value;
    }
    free(copy);
    return count == world && sum == kActiveExperts;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3 || std::strcmp(argv[1], "--devices")) {
        std::fprintf(stderr,
                     "usage: %s --devices ROOT,PEER[,PEER...] "
                     "[--owner-counts N,N,...] [--overlap-shared] [--remap]\n",
                     argv[0]);
        return 1;
    }
    int devices[kMaxRanks]{};
    const int world = parse_devices(argv[2], devices, kMaxRanks);
    if (world < 2)
        return 1;
    const int peers = world - 1;
    unsigned char *model = (unsigned char *)malloc((size_t)kModelBytes);
    float *host_x = (float *)malloc((size_t)kInDim * sizeof(float));
    if (!model || !host_x) return 2;
    build_model(model);
    for (uint32_t i = 0; i < kInDim; i++)
        host_x[i] = (float)((int)(i % 67u) - 33) * 0.00390625f;
    uint32_t owner_counts[kMaxRanks]{};
    const char *counts_arg = nullptr;
    bool overlap_shared = false;
    bool remap_residency = false;
    for (int argi = 3; argi < argc; argi++) {
        if (!std::strcmp(argv[argi], "--owner-counts") &&
            argi + 1 < argc && !counts_arg) {
            counts_arg = argv[++argi];
        } else if (!std::strcmp(argv[argi], "--overlap-shared") &&
                   !overlap_shared) {
            overlap_shared = true;
        } else if (!std::strcmp(argv[argi], "--remap") &&
                   !remap_residency) {
            remap_residency = true;
        } else {
            std::fprintf(stderr, "invalid or duplicate argument: %s\n",
                         argv[argi]);
            return 1;
        }
    }
    if (counts_arg) {
        if (!parse_owner_counts(counts_arg, owner_counts, world)) {
            std::fprintf(stderr,
                         "--owner-counts must contain %d counts whose sum "
                         "is %u\n", world, kActiveExperts);
            return 2;
        }
    } else {
        for (uint32_t slot = 0; slot < kActiveExperts; slot++)
            owner_counts[slot % (uint32_t)world]++;
    }
    if (overlap_shared && owner_counts[0] != 0u) {
        std::fprintf(stderr,
                     "--overlap-shared requires root owner count 0\n");
        return 2;
    }
    int32_t host_selected[kActiveExperts]{};
    uint32_t selected_slot = 0u;
    for (uint32_t owner = 0; owner < (uint32_t)world; owner++) {
        const uint32_t expert_base = kTotalExperts * owner / (uint32_t)world;
        for (uint32_t local = 0; local < owner_counts[owner]; local++)
            host_selected[selected_slot++] = (int32_t)(expert_base + local);
    }
    const float host_weights[kActiveExperts] = {
        0.11f, 0.17f, 0.19f, 0.13f, 0.23f, 0.17f};
    FILE *model_file = tmpfile();
    if (!model_file ||
        fwrite(model, 1u, (size_t)kModelBytes, model_file) != kModelBytes ||
        fflush(model_file)) return 3;
    const int model_fd = fileno(model_file);
    setenv("DS4_ROCM_WEIGHT_ARENA_CHUNK_MB", "256", 1);

    int ready[kMaxRanks - 1][2]{}, go[kMaxRanks - 1][2]{};
    pid_t children[kMaxRanks - 1]{};
    for (int i = 0; i < peers; i++) {
        if (pipe(ready[i]) || pipe(go[i])) return 4;
        const pid_t child = fork();
        if (child < 0) return 5;
        if (child == 0) {
            close(ready[i][0]);
            close(go[i][1]);
            const int rc = worker_main(devices[i + 1], (uint32_t)i + 1u,
                                       (uint32_t)world, model, model_fd,
                                       host_x, host_selected, host_weights,
                                       overlap_shared,
                                       remap_residency,
                                       ready[i][1], go[i][0]);
            _exit(rc);
        }
        children[i] = child;
        close(ready[i][1]);
        close(go[i][0]);
    }
    std::vector<rank_handles> handles((size_t)peers);
    for (int i = 0; i < peers; i++)
        if (!read_all(ready[i][0], &handles[(size_t)i],
                      sizeof(rank_handles))) return 6;

    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", devices[0]);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (!ds4_gpu_init() || !ds4_gpu_set_model_fd(model_fd) ||
        !ds4_gpu_set_model_map(model, kModelBytes) ||
        !cache_owner_ranges(model, kModelBytes, 0u, kTotalExperts) ||
        (overlap_shared && !cache_shared_ranges(model, kModelBytes))) return 7;
    moe_buffers buffers{};
    if (!allocate_buffers(&buffers, host_x, host_selected, host_weights))
        return 8;
    ds4_gpu_tensor *reference =
        ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float));
    ds4_gpu_tensor *reference_routed = overlap_shared ?
        ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float)) : nullptr;
    ds4_gpu_tensor *reference_shared = overlap_shared ?
        ds4_gpu_tensor_alloc((uint64_t)kOutDim * sizeof(float)) : nullptr;
    ds4_gpu_tensor *broadcast_copy = overlap_shared ?
        ds4_gpu_tensor_alloc((uint64_t)kInDim * sizeof(float)) : nullptr;
    if (!reference || (overlap_shared &&
        (!reference_routed || !reference_shared || !broadcast_copy))) return 9;
    for (int i = 0; i < kWarmup; i++)
        if ((!overlap_shared && !run_full(&buffers, reference, model)) ||
            (overlap_shared &&
             (!run_full(&buffers, reference_routed, model) ||
              !run_shared(&buffers, reference_shared, model) ||
              !ds4_gpu_add_tensor(reference, reference_routed,
                                  reference_shared, kOutDim)))) return 10;
    if (!ds4_gpu_synchronize()) return 11;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; i++)
        if ((!overlap_shared && !run_full(&buffers, reference, model)) ||
            (overlap_shared &&
             (!run_full(&buffers, reference_routed, model) ||
              !run_shared(&buffers, reference_shared, model) ||
              !ds4_gpu_add_tensor(reference, reference_routed,
                                  reference_shared, kOutDim)))) return 12;
    if (!ds4_gpu_synchronize()) return 13;
    auto end = std::chrono::steady_clock::now();
    const double full_ms =
        std::chrono::duration<double>(end - begin).count() * 1.0e3 /
        kIterations;

    double root_remap_ms = 0.0;
    if (remap_residency) {
        const uint32_t root_count = overlap_shared ?
            0u : kTotalExperts / (uint32_t)world;
        const auto remap_begin = std::chrono::steady_clock::now();
        if (!remap_owner_ranges(model, kModelBytes, 0u, root_count,
                                overlap_shared)) return 13;
        const auto remap_end = std::chrono::steady_clock::now();
        root_remap_ms = std::chrono::duration<double>(
                remap_end - remap_begin).count() * 1.0e3;
    }

    std::vector<ds4_rocm_tp_ipc_handle> peer_partials((size_t)peers);
    std::vector<ds4_rocm_tp_ipc_handle> peer_reduced((size_t)peers);
    std::vector<ds4_rocm_tp_ipc_handle> peer_x((size_t)peers);
    for (int i = 0; i < peers; i++) {
        peer_partials[(size_t)i] = handles[(size_t)i].partial;
        peer_reduced[(size_t)i] = handles[(size_t)i].reduced;
        peer_x[(size_t)i] = handles[(size_t)i].x;
    }
    ds4_rocm_tp_star *star = ds4_rocm_tp_star_create(
        ds4_gpu_tensor_contents(buffers.out),
        ds4_gpu_tensor_contents(buffers.reduced),
        peer_partials.data(), peer_reduced.data(), (uint32_t)peers);
    if (!star) return 14;
    ds4_rocm_tp_star *broadcast_star = overlap_shared ?
        ds4_rocm_tp_star_create(
            ds4_gpu_tensor_contents(buffers.x),
            ds4_gpu_tensor_contents(broadcast_copy),
            peer_partials.data(), peer_x.data(), (uint32_t)peers) : nullptr;
    if (overlap_shared && !broadcast_star) return 14;
    double total_ms = 0.0, root_compute_ms = 0.0;
    std::vector<double> peer_compute_ms((size_t)peers, 0.0);
    double broadcast_ms = 0.0, wait_ms = 0.0, collective_ms = 0.0;
    double trigger_ms = 0.0;
    for (int iter = 0; iter < kWarmup + kIterations; iter++) {
        const auto iter_begin = std::chrono::steady_clock::now();
        if (overlap_shared &&
            !ds4_rocm_tp_star_broadcast_f32(broadcast_star, kInDim, 1))
            return 15;
        const auto broadcast_end = std::chrono::steady_clock::now();
        const char signal = 1;
        for (int i = 0; i < peers; i++)
            if (!write_all(go[i][1], &signal, 1u)) return 15;
        const auto trigger_end = std::chrono::steady_clock::now();
        const uint32_t root_end_expert = kTotalExperts / (uint32_t)world;
        if ((!overlap_shared &&
             !run_owned(&buffers, model, 0u, root_end_expert)) ||
            (overlap_shared && !run_shared(&buffers, buffers.out, model)) ||
            !ds4_gpu_synchronize())
            return 16;
        const auto root_end = std::chrono::steady_clock::now();
        std::vector<rank_sample> samples((size_t)peers);
        for (int i = 0; i < peers; i++)
            if (!read_all(ready[i][0], &samples[(size_t)i],
                          sizeof(rank_sample))) return 17;
        const auto ready_end = std::chrono::steady_clock::now();
        if (!ds4_rocm_tp_star_reduce_f32(star, kOutDim, 1)) return 18;
        const auto iter_end = std::chrono::steady_clock::now();
        if (iter >= kWarmup) {
            total_ms += std::chrono::duration<double>(iter_end - iter_begin).count() * 1.0e3;
            broadcast_ms += std::chrono::duration<double>(broadcast_end - iter_begin).count() * 1.0e3;
            trigger_ms += std::chrono::duration<double>(trigger_end - broadcast_end).count() * 1.0e3;
            root_compute_ms += std::chrono::duration<double>(root_end - trigger_end).count() * 1.0e3;
            wait_ms += std::chrono::duration<double>(ready_end - root_end).count() * 1.0e3;
            collective_ms += std::chrono::duration<double>(iter_end - ready_end).count() * 1.0e3;
            for (int i = 0; i < peers; i++)
                peer_compute_ms[(size_t)i] += samples[(size_t)i].compute_ms;
        }
    }
    total_ms /= kIterations;
    broadcast_ms /= kIterations;
    trigger_ms /= kIterations;
    root_compute_ms /= kIterations;
    wait_ms /= kIterations;
    collective_ms /= kIterations;
    for (double &ms : peer_compute_ms) ms /= kIterations;

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
    for (int i = 0; i < peers; i++)
        if (!write_all(go[i][1], &done, 1u)) return 20;
    bool children_ok = true;
    for (int i = 0; i < peers; i++) {
        int status = 0;
        const bool child_ok = waitpid(children[i], &status, 0) >= 0 &&
                              WIFEXITED(status) && WEXITSTATUS(status) == 0;
        children_ok = children_ok && child_ok;
    }
    hipDeviceProp_t root_properties{};
    if (hipGetDeviceProperties(&root_properties, 0) != hipSuccess) return 21;
    const bool pass = children_ok && std::isfinite(rms) && rel <= 5.0e-4;
    std::printf("ROCm Flash-0731 IQ2/Q2 EP%d decode, six active experts%s\n",
                world, overlap_shared ? " + root shared overlap" : "");
    std::printf("  full single GPU%s: %.4f ms\n",
                overlap_shared ? " routed+shared" : "", full_ms);
    std::printf("  EP%d synchronized end-to-end: %.4f ms (%.2fx)\n",
                world, total_ms, full_ms / total_ms);
    std::printf("  root physical=%d %-24s compute=%.4f ms\n",
                devices[0], root_properties.name, root_compute_ms);
    for (int i = 0; i < peers; i++)
        std::printf("  peer physical=%d %-24s compute=%.4f ms\n",
                    devices[i + 1], handles[(size_t)i].name,
                    peer_compute_ms[(size_t)i]);
    std::printf("  stages: broadcast=%.4f ms trigger=%.4f ms root_compute=%.4f ms "
                "residual_wait=%.4f ms collective=%.4f ms\n",
                broadcast_ms, trigger_ms, root_compute_ms, wait_ms,
                collective_ms);
    std::printf("  numerical: rms=%g max_abs=%g max_ref=%g rel=%g\n",
                rms, max_abs, max_ref, rel);
    std::printf("  verification: %s\n", pass ? "PASS" : "FAIL");
    if (remap_residency) {
        std::printf("  PP->EP residency remap: root=%.2f ms", root_remap_ms);
        for (int i = 0; i < peers; i++)
            std::printf(" peer%d=%.2f ms", i + 1,
                        handles[(size_t)i].remap_ms);
        std::printf("\n");
    }

    ds4_rocm_tp_star_destroy(broadcast_star);
    ds4_rocm_tp_star_destroy(star);
    ds4_gpu_tensor_free(broadcast_copy);
    ds4_gpu_tensor_free(reference_shared);
    ds4_gpu_tensor_free(reference_routed);
    ds4_gpu_tensor_free(reference);
    free_buffers(&buffers);
    ds4_gpu_cleanup();
    fclose(model_file);
    free(host_x);
    free(model);
    return pass ? 0 : 22;
}
