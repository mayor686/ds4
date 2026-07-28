#!/usr/bin/env bash
#
# run.sh — configurable distributed DeepSeek-V4-Flash launcher for AMD gfx906.
# The supplied run-speed.sh and run-context.sh are validated six-GPU examples.
#
# Adapted from the 4x RTX 3090 recipe:
#   https://github.com/Forge-the-Kingdom/inference-serving-recipes/blob/main/recipes/dwarfstar/deepseek-v4-iq2-4x3090-gpu-resident.md
#
# Key changes vs the CUDA recipe:
#   - Backend flag is --rocm (not --cuda) in a DS4_ROCM_BUILD.
#   - Per-process GPU pinning uses ROCR_VISIBLE_DEVICES (not CUDA_VISIBLE_DEVICES).
#   - Coordinator and worker device indexes are explicit below because ROCR
#     indexes need not match the rocm-smi card numbers.
#
# Prerequisites:
#   - ds4 and ds4-server built for your ROCm arch:  make rocm ROCM_ARCH=<arch>
#     gfx906 builds use the in-tree wave64-safe rocWMMA compatibility shim.
#   - Enough free VRAM on every device selected by the chosen profile.
#   - The Flash IQ2XXS GGUF downloaded, e.g.:
#       ./download_model.sh q2-imatrix
#     which links ./ds4flash.gguf to the file below.
#
# Usage:
#   1. Set MODEL_PATH and the device/layer map below for your system.
#   2. ./run.sh            # starts workers then the coordinator
#   3. In another shell:   curl -s http://127.0.0.1:8080/v1/models
#   Ctrl+C the coordinator; workers exit when the route drops.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Override MODEL_PATH when the GGUF is not available through ./ds4flash.gguf.
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/ds4flash.gguf}"
# =============================================================================

if [ -z "${MODEL_PATH}" ]; then
    echo "run.sh: MODEL_PATH is empty. Edit this script and set MODEL_PATH to the" >&2
    echo "        DeepSeek-V4-Flash IQ2XXS GGUF, e.g. /path/to/model.gguf" >&2
    exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
    echo "run.sh: model file not found: ${MODEL_PATH}" >&2
    exit 1
fi

# --- binaries (built with: make rocm ROCM_ARCH=<arch>) ---
DS4_SERVER="${SCRIPT_DIR}/ds4-server"   # coordinator
DS4_WORKER="${SCRIPT_DIR}/ds4"          # worker
for b in "${DS4_SERVER}" "${DS4_WORKER}"; do
    if [ ! -x "${b}" ]; then
        echo "run.sh: missing executable: ${b}" >&2
        echo "        build with: make rocm ROCM_ARCH=<arch>" >&2
        exit 1
    fi
done

# --- distributed pipeline tuning (from the 4x3090 recipe) ---
DIST_HOST="${DIST_HOST:-127.0.0.1}"
DIST_PORT="${DIST_PORT:-19000}"
CTX="${CTX:-28672}"                       # tested safe ceiling with the GUI active
MAX_TOKENS="${MAX_TOKENS:-${CTX}}"
PREFILL_CHUNK="${PREFILL_CHUNK:-64}"      # small chunks keep all 6 cards busy
DIST_WINDOW="${DIST_WINDOW:-5}"           # larger windows can OOM 16 GB workers
WORKER_START_DELAY="${WORKER_START_DELAY:-20}"
DS4_PROFILE="${DS4_PROFILE:-0}"
DS4_COORD_SERIALIZE="${DS4_COORD_SERIALIZE:-0}"
ROCPROF_COORD_OUTPUT_DIR="${ROCPROF_COORD_OUTPUT_DIR:-}"
ROCPROF_WORKER_DEVICE="${ROCPROF_WORKER_DEVICE:-}"
ROCPROF_WORKER_OUTPUT_DIR="${ROCPROF_WORKER_OUTPUT_DIR:-}"
TRACE_FILE="${TRACE_FILE:-}"
TRACE_ARGS=()
if [ -n "${TRACE_FILE}" ]; then
    TRACE_ARGS+=(--trace "${TRACE_FILE}")
fi

# --- optional SSD-backed routed-expert cache ---
# Keep disabled for the fastest fully resident path.  Enabling it leaves
# non-routed weights resident and loads routed MoE experts into a bounded cache
# from the GGUF on demand.  The cache budget is per process/GPU.
SSD_STREAMING="${SSD_STREAMING:-0}"
SSD_STREAMING_CACHE_EXPERTS="${SSD_STREAMING_CACHE_EXPERTS:-}"
SSD_STREAMING_PRELOAD_EXPERTS="${SSD_STREAMING_PRELOAD_EXPERTS:-}"
SSD_STREAMING_COLD="${SSD_STREAMING_COLD:-0}"
SSD_ARGS=()
if [ "${SSD_STREAMING}" != "0" ]; then
    SSD_ARGS+=(--ssd-streaming)
    if [ -n "${SSD_STREAMING_CACHE_EXPERTS}" ]; then
        SSD_ARGS+=(--ssd-streaming-cache-experts "${SSD_STREAMING_CACHE_EXPERTS}")
    fi
    if [ -n "${SSD_STREAMING_PRELOAD_EXPERTS}" ]; then
        SSD_ARGS+=(--ssd-streaming-preload-experts "${SSD_STREAMING_PRELOAD_EXPERTS}")
    fi
    if [ "${SSD_STREAMING_COLD}" != "0" ]; then
        SSD_ARGS+=(--ssd-streaming-cold)
    fi
fi

# --- HTTP API ---
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8080}"

# --- MTP speculative decoding (draft model, P1 speedup) ---
# Disabled: the draft model does not fit reliably beside the resident profile
# and did not improve quality-adjusted throughput on this PP6 topology.
MTP_PATH="${MTP_PATH:-${SCRIPT_DIR}/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
MTP_DRAFT="${MTP_DRAFT:-2}"
MTP_ARGS=()

# --- layer split: all 43 layers (0..42), final worker owns output head ---
# The context profile keeps the conservative 13 + 6x5 split.  The speed
# profile uses 8 + 7x5 at a smaller context: equal-size GPU stages remove the
# coordinator prefill bottleneck while still fitting the 16GB workers.
#
# A worker spec is DEVICE,LAYER_START:LAYER_END; use LAYER_START:output for
# the final worker so it also owns the output head.  Keeping the route in data
# instead of hard-coding five workers makes this shared launcher usable for
# other gfx906 counts and memory layouts.  The two example profiles below set
# the exact six-GPU maps validated on the development machine.
COORD_DEVICE="${COORD_DEVICE:-2}"
COORD_LAYERS="${COORD_LAYERS:-0:12}"
WORKER_SPECS="${WORKER_SPECS:-0,13:18 1,19:24 3,25:30 4,31:36 5,37:output}"

WORKER_PIDS=()

cleanup() {
    local pid
    for pid in "${WORKER_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    pkill -f "ds4.*--role worker.*${DIST_PORT}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
cleanup

start_worker() { # <rocr_device> <layers> [extra ds4 arguments...]
    local dev="$1" layers="$2"
    local arg ssd_worker=0
    local -a worker_env=(env
        -u DS4_DIST_DECODE_PROFILE
        DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256
        DS4_LOCK_FILE="/tmp/ds4-worker-${dev}.lock"
        ROCR_VISIBLE_DEVICES="${dev}")
    local -a worker_prefix=()
    shift 2
    for arg in "$@"; do
        if [ "${arg}" = "--ssd-streaming" ]; then
            ssd_worker=1
            break
        fi
    done
    if [ "${DS4_PROFILE}" != "0" ] && [ "${ssd_worker}" != "0" ]; then
        worker_env+=(DS4_ROCM_STREAM_CACHE_STATS=1)
    fi
    if [ -n "${ROCPROF_WORKER_DEVICE}" ] &&
       [ "${ROCPROF_WORKER_DEVICE}" = "${dev}" ]; then
        if [ -z "${ROCPROF_WORKER_OUTPUT_DIR}" ]; then
            echo "run.sh: ROCPROF_WORKER_OUTPUT_DIR is required for profiling" >&2
            exit 1
        fi
        worker_prefix=(rocprofv3 --kernel-trace --stats --output-format csv
            --output-directory "${ROCPROF_WORKER_OUTPUT_DIR}" --)
        echo "run.sh: profiling worker ROCR ${dev} into ${ROCPROF_WORKER_OUTPUT_DIR}"
    fi
    "${worker_env[@]}" \
    "${worker_prefix[@]}" \
    "${DS4_WORKER}" -m "${MODEL_PATH}" --rocm \
        --prefill-chunk "${PREFILL_CHUNK}" \
        --ctx "${CTX}" \
        --role worker --layers "${layers}" \
        --coordinator "${DIST_HOST}" "${DIST_PORT}" \
        "$@" &
    WORKER_PIDS+=("$!")
    echo "run.sh: worker started on ROCR device ${dev}, layers ${layers} (pid $!)"
}

check_last_worker() {
    local pid="${WORKER_PIDS[${#WORKER_PIDS[@]}-1]}"
    sleep "${WORKER_START_DELAY}"
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "run.sh: worker ${pid} exited during startup" >&2
        wait "${pid}" || true
        exit 1
    fi
}

if [ "${SSD_STREAMING}" != "0" ]; then
    echo "run.sh: starting SSD expert-streaming profile, ctx=${CTX}, chunk=${PREFILL_CHUNK}, window=${DIST_WINDOW}"
else
    echo "run.sh: starting GPU-resident profile, ctx=${CTX}, chunk=${PREFILL_CHUNK}, window=${DIST_WINDOW}"
fi
for worker_spec in ${WORKER_SPECS}; do
    IFS=, read -r worker_dev worker_layers worker_extra <<< "${worker_spec}"
    if [[ ! "${worker_dev}" =~ ^[0-9]+$ ]] ||
       [[ ! "${worker_layers}" =~ ^[0-9]+:([0-9]+|output)$ ]] ||
       [ -n "${worker_extra}" ]; then
        echo "run.sh: invalid worker spec '${worker_spec}'; expected DEVICE,START:END or DEVICE,START:output" >&2
        exit 1
    fi
    start_worker "${worker_dev}" "${worker_layers}" "${SSD_ARGS[@]}"
    check_last_worker
done

# Let workers fail visibly before the coordinator commits its allocation.
sleep 2

echo "run.sh: starting coordinator on ROCR device ${COORD_DEVICE}, layers ${COORD_LAYERS}; final worker owns output head"
COORD_ENV=(env
    -u DS4_DIST_DECODE_PROFILE
    DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256
    DS4_LOCK_FILE="/tmp/ds4-coordinator-${COORD_DEVICE}.lock"
    ROCR_VISIBLE_DEVICES="${COORD_DEVICE}")
if [ "${DS4_COORD_SERIALIZE}" != "0" ]; then
    COORD_ENV+=(AMD_SERIALIZE_KERNEL="${DS4_COORD_SERIALIZE}")
fi
COORD_PREFIX=()
if [ -n "${ROCPROF_COORD_OUTPUT_DIR}" ]; then
    COORD_PREFIX=(rocprofv3 --kernel-trace --stats --output-format csv
        --output-directory "${ROCPROF_COORD_OUTPUT_DIR}" --)
    echo "run.sh: profiling coordinator into ${ROCPROF_COORD_OUTPUT_DIR}"
fi

"${COORD_ENV[@]}" \
"${COORD_PREFIX[@]}" \
"${DS4_SERVER}" -m "${MODEL_PATH}" --rocm \
    --prefill-chunk "${PREFILL_CHUNK}" \
    --ctx "${CTX}" \
    --tokens "${MAX_TOKENS}" \
    --host "${HTTP_HOST}" --port "${HTTP_PORT}" \
    "${TRACE_ARGS[@]}" \
    --role coordinator --layers "${COORD_LAYERS}" \
    "${MTP_ARGS[@]}" \
    "${SSD_ARGS[@]}" \
    --listen "${DIST_HOST}" "${DIST_PORT}" \
    --dist-prefill-chunk "${PREFILL_CHUNK}" \
    --dist-prefill-window "${DIST_WINDOW}" \
    --dist-activation-bits 16 &
COORD_PID=$!
wait "${COORD_PID}"
