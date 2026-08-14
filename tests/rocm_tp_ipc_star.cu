/* Same-host, process-per-GPU validation for the ROCm tensor-parallel star.
 * Example for five gfx906 GPUs while excluding physical device 2:
 *   ./tests/rocm_tp_ipc_star --devices 0,1,3,4,5
 */

#include <hip/hip_runtime.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ds4_rocm_tp.h"

static constexpr size_t kVerifyBytes = 8ull << 20;
static constexpr size_t kHiddenBytes = 28ull << 10;
static constexpr int kMaxRanks = 16;

__global__ static void fill_f32(float *data, float value, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}

__global__ static void consume_f32(float *data, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] += 0.0f;
}

typedef struct {
    ds4_rocm_tp_ipc_handle input;
    ds4_rocm_tp_ipc_handle output;
} rank_handles;

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

static int worker_main(int physical_device, int rank, int ready_fd, int go_fd,
                       float expected) {
    char visible[24];
    std::snprintf(visible, sizeof(visible), "%d", physical_device);
    setenv("ROCR_VISIBLE_DEVICES", visible, 1);
    if (hipSetDevice(0) != hipSuccess) return 20;
    float *input = nullptr;
    float *output = nullptr;
    if (hipMalloc(&input, kVerifyBytes) != hipSuccess ||
        hipMalloc(&output, kVerifyBytes) != hipSuccess) return 21;
    const size_t count = kVerifyBytes / sizeof(float);
    fill_f32<<<(count + 255u) / 256u, 256>>>(input, (float)(rank + 1), count);
    if (hipGetLastError() != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) return 22;
    rank_handles handles{};
    if (!ds4_rocm_tp_ipc_export(input, &handles.input) ||
        !ds4_rocm_tp_ipc_export(output, &handles.output) ||
        !write_all(ready_fd, &handles, sizeof(handles))) return 23;

    char go = 0;
    if (!read_all(go_fd, &go, 1u)) return 24;
    consume_f32<<<(count + 255u) / 256u, 256>>>(output, count);
    if (hipGetLastError() != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) return 25;
    std::vector<float> host(count);
    if (hipMemcpy(host.data(), output, kVerifyBytes,
                  hipMemcpyDeviceToHost) != hipSuccess) return 26;
    size_t wrong = 0;
    for (float value : host) wrong += value != expected;
    std::printf("rank=%d physical=%d checked=%zu wrong=%zu\n",
                rank, physical_device, count, wrong);
    std::fflush(stdout);
    (void)hipFree(output);
    (void)hipFree(input);
    return wrong ? 27 : 0;
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
        devices[count++] = (int)value;
    }
    free(copy);
    return count;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 3 || std::strcmp(argv[1], "--devices") != 0) {
        std::fprintf(stderr, "usage: %s --devices 0,1,3,4,5\n", argv[0]);
        return 1;
    }
    int devices[kMaxRanks]{};
    const int world = parse_devices(argv[2], devices, kMaxRanks);
    if (world < 2) {
        std::fprintf(stderr, "at least two valid devices are required\n");
        return 1;
    }
    const int peers = world - 1;
    const float expected = (float)(world * (world + 1) / 2);
    int ready[kMaxRanks - 1][2]{};
    int go[kMaxRanks - 1][2]{};
    pid_t children[kMaxRanks - 1]{};
    for (int i = 0; i < peers; i++) {
        if (pipe(ready[i]) || pipe(go[i])) return 2;
        const pid_t pid = fork();
        if (pid < 0) return 3;
        if (pid == 0) {
            close(ready[i][0]);
            close(go[i][1]);
            const int rc = worker_main(devices[i + 1], i + 1,
                                       ready[i][1], go[i][0], expected);
            _exit(rc);
        }
        children[i] = pid;
        close(ready[i][1]);
        close(go[i][0]);
    }

    std::vector<rank_handles> handles((size_t)peers);
    for (int i = 0; i < peers; i++) {
        if (!read_all(ready[i][0], &handles[(size_t)i], sizeof(rank_handles)))
            return 4;
    }
    char root_visible[24];
    std::snprintf(root_visible, sizeof(root_visible), "%d", devices[0]);
    setenv("ROCR_VISIBLE_DEVICES", root_visible, 1);
    if (hipSetDevice(0) != hipSuccess) return 5;
    float *root_input = nullptr;
    float *root_output = nullptr;
    if (hipMalloc(&root_input, kVerifyBytes) != hipSuccess ||
        hipMalloc(&root_output, kVerifyBytes) != hipSuccess) return 6;
    const size_t verify_count = kVerifyBytes / sizeof(float);
    fill_f32<<<(verify_count + 255u) / 256u, 256>>>(root_input, 1.0f,
                                                    verify_count);
    if (hipGetLastError() != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) return 7;

    std::vector<ds4_rocm_tp_ipc_handle> peer_inputs((size_t)peers);
    std::vector<ds4_rocm_tp_ipc_handle> peer_outputs((size_t)peers);
    for (int i = 0; i < peers; i++) {
        peer_inputs[(size_t)i] = handles[(size_t)i].input;
        peer_outputs[(size_t)i] = handles[(size_t)i].output;
    }
    ds4_rocm_tp_star *star = ds4_rocm_tp_star_create(
            root_input, root_output, peer_inputs.data(), peer_outputs.data(),
            (uint32_t)peers);
    if (!star) return 8;

    const uint64_t hidden_count = kHiddenBytes / sizeof(float);
    for (int i = 0; i < 20; i++) {
        if (!ds4_rocm_tp_star_allreduce_f32(star, hidden_count, 0)) return 9;
    }
    if (!ds4_rocm_tp_star_synchronize(star)) return 10;
    constexpr int queued_iters = 2000;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < queued_iters; i++) {
        if (!ds4_rocm_tp_star_allreduce_f32(star, hidden_count, 0)) return 11;
    }
    if (!ds4_rocm_tp_star_synchronize(star)) return 12;
    auto end = std::chrono::steady_clock::now();
    const double queued_us =
        std::chrono::duration<double>(end - begin).count() * 1.0e6 /
        queued_iters;

    constexpr int serial_iters = 300;
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < serial_iters; i++) {
        if (!ds4_rocm_tp_star_allreduce_f32(star, hidden_count, 1)) return 13;
    }
    end = std::chrono::steady_clock::now();
    const double serial_us =
        std::chrono::duration<double>(end - begin).count() * 1.0e6 /
        serial_iters;

    for (int i = 0; i < 20; i++) {
        if (!ds4_rocm_tp_star_broadcast_f32(star, hidden_count, 1)) return 14;
    }
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < serial_iters; i++) {
        if (!ds4_rocm_tp_star_broadcast_f32(star, hidden_count, 1)) return 14;
    }
    end = std::chrono::steady_clock::now();
    const double broadcast_us =
        std::chrono::duration<double>(end - begin).count() * 1.0e6 /
        serial_iters;

    if (!ds4_rocm_tp_star_allreduce_f32(star, verify_count, 1)) return 14;
    std::vector<float> root_host(verify_count);
    if (hipMemcpy(root_host.data(), root_output, kVerifyBytes,
                  hipMemcpyDeviceToHost) != hipSuccess) return 15;
    size_t root_wrong = 0;
    for (float value : root_host) root_wrong += value != expected;
    std::printf("HIP IPC star world=%d root=%d payload=%zuKiB "
                "queued=%.2fus serialized=%.2fus broadcast=%.2fus\n",
                world, devices[0], kHiddenBytes >> 10, queued_us, serial_us,
                broadcast_us);
    std::printf("root checked=%zu wrong=%zu expected=%.1f\n",
                verify_count, root_wrong, expected);

    const char signal = 1;
    for (int i = 0; i < peers; i++) {
        if (!write_all(go[i][1], &signal, 1u)) return 16;
    }
    bool pass = root_wrong == 0;
    for (int i = 0; i < peers; i++) {
        int status = 0;
        if (waitpid(children[i], &status, 0) < 0 || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) {
            std::printf("rank=%d physical=%d failed status=%d\n",
                        i + 1, devices[i + 1], status);
            pass = false;
        }
    }
    ds4_rocm_tp_star_destroy(star);
    (void)hipFree(root_output);
    (void)hipFree(root_input);
    std::printf("multi-process verification: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 17;
}
