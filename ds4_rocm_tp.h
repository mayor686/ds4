#ifndef DS4_ROCM_TP_H
#define DS4_ROCM_TP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* HIP memory handles are intentionally kept opaque here so that the C engine
 * and control protocol do not need to include HIP headers. */
#define DS4_ROCM_TP_IPC_HANDLE_BYTES 64u

typedef struct {
    unsigned char bytes[DS4_ROCM_TP_IPC_HANDLE_BYTES];
} ds4_rocm_tp_ipc_handle;

typedef struct ds4_rocm_tp_star ds4_rocm_tp_star;

/* Export/import device allocations owned by one process per GPU.  The owner
 * must keep the allocation alive until all importers have closed the handle. */
int ds4_rocm_tp_ipc_export(void *device_ptr,
                           ds4_rocm_tp_ipc_handle *handle);
int ds4_rocm_tp_ipc_open(const ds4_rocm_tp_ipc_handle *handle,
                         void **device_ptr);
int ds4_rocm_tp_ipc_close(void *device_ptr);

/* Build a same-host star collective on the root GPU.  peer_* contains one
 * exported input/output pair for every non-root rank.  allreduce queues a
 * FP32 sum followed by a broadcast into every rank's output allocation.
 * When synchronize is non-zero the call returns only after remote outputs
 * are visible to the owning GPUs. */
ds4_rocm_tp_star *ds4_rocm_tp_star_create(
        void *root_input,
        void *root_output,
        const ds4_rocm_tp_ipc_handle *peer_inputs,
        const ds4_rocm_tp_ipc_handle *peer_outputs,
        uint32_t peer_count);
int ds4_rocm_tp_star_allreduce_f32(ds4_rocm_tp_star *star,
                                   uint64_t count,
                                   int synchronize);
/* Sum every rank's input into the root output only.  This is the preferred
 * operation when the layer-owning rank alone consumes the reduced tensor. */
int ds4_rocm_tp_star_reduce_f32(ds4_rocm_tp_star *star,
                               uint64_t count,
                               int synchronize);
/* Copy the root input to every rank's output allocation. */
int ds4_rocm_tp_star_broadcast_f32(ds4_rocm_tp_star *star,
                                  uint64_t count,
                                  int synchronize);
int ds4_rocm_tp_star_synchronize(ds4_rocm_tp_star *star);
void ds4_rocm_tp_star_destroy(ds4_rocm_tp_star *star);

#ifdef __cplusplus
}
#endif

#endif
