#include "ds4_rocm_tp.h"

#include <hip/hip_runtime.h>

#include <new>
#include <vector>

static_assert(sizeof(hipIpcMemHandle_t) == DS4_ROCM_TP_IPC_HANDLE_BYTES,
              "unexpected HIP IPC memory handle size");

struct ds4_rocm_tp_star {
    std::vector<float *> inputs;
    std::vector<float *> outputs;
    std::vector<void *> imported;
    float **device_inputs;
    float **device_outputs;
    hipStream_t stream;
};

__global__ static void ds4_rocm_tp_star_allreduce_kernel(
        float *const *inputs,
        float *const *outputs,
        uint32_t world,
        uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float sum = 0.0f;
    for (uint32_t rank = 0; rank < world; rank++) sum += inputs[rank][i];
    for (uint32_t rank = 0; rank < world; rank++) outputs[rank][i] = sum;
}

extern "C" int ds4_rocm_tp_ipc_export(
        void *device_ptr,
        ds4_rocm_tp_ipc_handle *handle) {
    if (!device_ptr || !handle) return 0;
    return hipIpcGetMemHandle(
                   reinterpret_cast<hipIpcMemHandle_t *>(handle),
                   device_ptr) == hipSuccess;
}

extern "C" int ds4_rocm_tp_ipc_open(
        const ds4_rocm_tp_ipc_handle *handle,
        void **device_ptr) {
    if (!handle || !device_ptr) return 0;
    *device_ptr = nullptr;
    return hipIpcOpenMemHandle(
                   device_ptr,
                   *reinterpret_cast<const hipIpcMemHandle_t *>(handle),
                   hipIpcMemLazyEnablePeerAccess) == hipSuccess;
}

extern "C" int ds4_rocm_tp_ipc_close(void *device_ptr) {
    return device_ptr && hipIpcCloseMemHandle(device_ptr) == hipSuccess;
}

extern "C" ds4_rocm_tp_star *ds4_rocm_tp_star_create(
        void *root_input,
        void *root_output,
        const ds4_rocm_tp_ipc_handle *peer_inputs,
        const ds4_rocm_tp_ipc_handle *peer_outputs,
        uint32_t peer_count) {
    if (!root_input || !root_output ||
        (peer_count && (!peer_inputs || !peer_outputs))) return nullptr;
    ds4_rocm_tp_star *star = new (std::nothrow) ds4_rocm_tp_star{};
    if (!star) return nullptr;
    star->device_inputs = nullptr;
    star->device_outputs = nullptr;
    star->stream = nullptr;
    star->inputs.reserve((size_t)peer_count + 1u);
    star->outputs.reserve((size_t)peer_count + 1u);
    star->imported.reserve((size_t)peer_count * 2u);
    star->inputs.push_back(static_cast<float *>(root_input));
    star->outputs.push_back(static_cast<float *>(root_output));
    for (uint32_t i = 0; i < peer_count; i++) {
        void *input = nullptr;
        void *output = nullptr;
        if (!ds4_rocm_tp_ipc_open(&peer_inputs[i], &input) ||
            !ds4_rocm_tp_ipc_open(&peer_outputs[i], &output)) {
            if (input) ds4_rocm_tp_ipc_close(input);
            ds4_rocm_tp_star_destroy(star);
            return nullptr;
        }
        star->imported.push_back(input);
        star->imported.push_back(output);
        star->inputs.push_back(static_cast<float *>(input));
        star->outputs.push_back(static_cast<float *>(output));
    }
    const size_t pointer_bytes = star->inputs.size() * sizeof(float *);
    if (hipMalloc(&star->device_inputs, pointer_bytes) != hipSuccess ||
        hipMalloc(&star->device_outputs, pointer_bytes) != hipSuccess ||
        hipMemcpy(star->device_inputs, star->inputs.data(), pointer_bytes,
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(star->device_outputs, star->outputs.data(), pointer_bytes,
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipStreamCreateWithFlags(&star->stream, hipStreamNonBlocking) !=
                  hipSuccess) {
        ds4_rocm_tp_star_destroy(star);
        return nullptr;
    }
    return star;
}

extern "C" int ds4_rocm_tp_star_allreduce_f32(
        ds4_rocm_tp_star *star,
        uint64_t count,
        int synchronize) {
    if (!star || !count || !star->stream) return 0;
    const uint32_t threads = 256u;
    const uint64_t grid64 = (count + threads - 1u) / threads;
    if (grid64 > UINT32_MAX) return 0;
    hipLaunchKernelGGL(ds4_rocm_tp_star_allreduce_kernel,
                       dim3((uint32_t)grid64), dim3(threads), 0, star->stream,
                       star->device_inputs, star->device_outputs,
                       (uint32_t)star->inputs.size(), count);
    if (hipGetLastError() != hipSuccess) return 0;
    return !synchronize || hipStreamSynchronize(star->stream) == hipSuccess;
}

extern "C" int ds4_rocm_tp_star_synchronize(ds4_rocm_tp_star *star) {
    return star && star->stream &&
           hipStreamSynchronize(star->stream) == hipSuccess;
}

extern "C" void ds4_rocm_tp_star_destroy(ds4_rocm_tp_star *star) {
    if (!star) return;
    if (star->stream) (void)hipStreamDestroy(star->stream);
    if (star->device_outputs) (void)hipFree(star->device_outputs);
    if (star->device_inputs) (void)hipFree(star->device_inputs);
    for (void *ptr : star->imported) ds4_rocm_tp_ipc_close(ptr);
    delete star;
}
