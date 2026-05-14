#pragma once

#include "ds4_rocm.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HIP_CHECK
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error %d: %s at %s:%d\n", err, hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Struttura che contiene il contesto Multi-GPU per l'inferenza
typedef struct {
    int num_devices;
    int* device_ids;
    hipStream_t* streams;
    hipEvent_t* sync_events;
} ds4_rocm_mgpu_context_t;

// Inizializza il contesto Multi-GPU
static inline ds4_rocm_mgpu_context_t* ds4_rocm_mgpu_init() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
        fprintf(stderr, "[ROCm] Nessun device compatibile trovato.\n");
        return NULL;
    }

    fprintf(stdout, "[ROCm] Trovati %d device. Inizializzazione contesto Multi-GPU...\n", count);

    ds4_rocm_mgpu_context_t* ctx = (ds4_rocm_mgpu_context_t*)malloc(sizeof(ds4_rocm_mgpu_context_t));
    if (!ctx) return NULL;

    ctx->num_devices = count;
    ctx->device_ids = (int*)malloc(sizeof(int) * count);
    ctx->streams = (hipStream_t*)malloc(sizeof(hipStream_t) * count);
    ctx->sync_events = (hipEvent_t*)malloc(sizeof(hipEvent_t) * count);

    for (int i = 0; i < count; ++i) {
        ctx->device_ids[i] = i;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&ctx->streams[i]));
        HIP_CHECK(hipEventCreate(&ctx->sync_events[i]));
    }

    // Abilita il Peer-to-Peer (P2P) access tra tutte le GPU disponibili
    // Questo è fondamentale per spostare rapidamente le attivazioni tra un layer 
    // e l'altro se si usa il Pipeline Parallelism (divisione dei layer su GPU diverse).
    for (int i = 0; i < count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        for (int j = 0; j < count; ++j) {
            if (i == j) continue;
            int can_access = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access, i, j));
            if (can_access) {
                hipError_t err = hipDeviceEnablePeerAccess(j, 0);
                // Ignoriamo l'errore se il P2P era già abilitato precedentemente
                if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "[ROCm] Attenzione: Impossibile abilitare P2P da GPU %d a GPU %d\n", i, j);
                }
            } else {
                fprintf(stderr, "[ROCm] Avviso: P2P non supportato da GPU %d a GPU %d (sarà usato il fallback via host)\n", i, j);
            }
        }
    }

    // Reset del device attivo al default (0)
    HIP_CHECK(hipSetDevice(0));
    fprintf(stdout, "[ROCm] Contesto Multi-GPU inizializzato con successo.\n");
    return ctx;
}

// Libera le risorse
static inline void ds4_rocm_mgpu_free(ds4_rocm_mgpu_context_t* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->num_devices; ++i) {
        hipSetDevice(ctx->device_ids[i]);
        hipStreamDestroy(ctx->streams[i]);
        hipEventDestroy(ctx->sync_events[i]);
    }
    free(ctx->device_ids);
    free(ctx->streams);
    free(ctx->sync_events);
    free(ctx);
}

// Helper per trasferire tensori/attivazioni tra una GPU e l'altra (P2P) in modo asincrono.
// Ideale per inferenza dove il device src ha finito di processare un layer e passa 
// l'input al device dst per il layer successivo (Pipeline Parallelism).
static inline void ds4_rocm_mgpu_copy_peer_async(void* dst, int dst_device, const void* src, int src_device, size_t size, ds4_rocm_mgpu_context_t* ctx) {
    // Effettua la copia asincrona sfruttando lo stream del device sorgente
    HIP_CHECK(hipSetDevice(src_device));
    HIP_CHECK(hipMemcpyPeerAsync(dst, dst_device, src, src_device, size, ctx->streams[src_device]));
    
    // Registra un evento quando la copia sul src_stream è terminata
    HIP_CHECK(hipEventRecord(ctx->sync_events[src_device], ctx->streams[src_device]));
    
    // Fa in modo che lo stream di destinazione attenda che la copia sia conclusa prima di usare i dati
    HIP_CHECK(hipSetDevice(dst_device));
    HIP_CHECK(hipStreamWaitEvent(ctx->streams[dst_device], ctx->sync_events[src_device], 0));
}

// Sincronizza tutti i device (utile a fine iterazione / batch)
static inline void ds4_rocm_mgpu_sync_all(ds4_rocm_mgpu_context_t* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->num_devices; ++i) {
        HIP_CHECK(hipSetDevice(ctx->device_ids[i]));
        HIP_CHECK(hipDeviceSynchronize());
    }
}

// Helper per determinare su quale GPU caricare e computare un dato layer
// basato su una suddivisione uniforme (Pipeline Parallelism).
static inline int ds4_rocm_mgpu_get_layer_device(int layer_idx, int total_layers, ds4_rocm_mgpu_context_t* ctx) {
    if (!ctx || ctx->num_devices <= 1) return 0;
    
    // Distribuzione uniforme dei layer sulle GPU disponibili
    // Esempio: 32 layer, 2 GPU -> GPU 0: layer 0-15, GPU 1: layer 16-31
    int layers_per_gpu = (total_layers + ctx->num_devices - 1) / ctx->num_devices; // arrotondamento per eccesso
    int target_device = layer_idx / layers_per_gpu;
    
    // Fallback di sicurezza per non sforare l'indice dei device
    if (target_device >= ctx->num_devices) {
        target_device = ctx->num_devices - 1;
    }
    
    return ctx->device_ids[target_device];
}

// Helper per il trasferimento dati in pipeline.
// Verifica se il layer successivo risiede su un'altra GPU.
// Se sì, avvia il trasferimento P2P in modo trasparente.
static inline int ds4_rocm_mgpu_pipeline_transfer(
    void* current_data, int current_layer_idx, 
    void* next_data_buffer, int total_layers, 
    size_t data_size, ds4_rocm_mgpu_context_t* ctx) 
{
    if (!ctx || ctx->num_devices <= 1) return 0;

    int current_device = ds4_rocm_mgpu_get_layer_device(current_layer_idx, total_layers, ctx);
    int next_device = ds4_rocm_mgpu_get_layer_device(current_layer_idx + 1, total_layers, ctx);

    if (current_device != next_device) {
        // Limite raggiunto: i dati devono passare all'altra GPU
        ds4_rocm_mgpu_copy_peer_async(next_data_buffer, next_device, current_data, current_device, data_size, ctx);
        return next_device;
    }
    
    // Nessun trasferimento necessario per il prossimo layer
    return current_device;
}

#ifdef __cplusplus
}
#endif
