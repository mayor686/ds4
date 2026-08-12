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
DIST_ACTIVATION_BITS="${DIST_ACTIVATION_BITS:-16}"
DIST_REQUIRE_WORKER_OUTPUT="${DIST_REQUIRE_WORKER_OUTPUT:-auto}"
WORKER_START_DELAY="${WORKER_START_DELAY:-20}"
DS4_PROFILE="${DS4_PROFILE:-0}"
DS4_COORD_SERIALIZE="${DS4_COORD_SERIALIZE:-0}"
DS4_ROCM_ATTN_COMP_CACHE_F16="${DS4_ROCM_ATTN_COMP_CACHE_F16:-0}"
ROCPROF_COORD_OUTPUT_DIR="${ROCPROF_COORD_OUTPUT_DIR:-}"
ROCPROF_WORKER_DEVICE="${ROCPROF_WORKER_DEVICE:-}"
ROCPROF_WORKER_OUTPUT_DIR="${ROCPROF_WORKER_OUTPUT_DIR:-}"
DS4_LAUNCH_DRY_RUN="${DS4_LAUNCH_DRY_RUN:-0}"
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

if [ "${DS4_ROCM_ATTN_COMP_CACHE_F16}" != "0" ] &&
   [ "${DS4_ROCM_ATTN_COMP_CACHE_F16}" != "1" ]; then
    echo "run.sh: DS4_ROCM_ATTN_COMP_CACHE_F16 must be 0 (F32) or 1 (experimental compact cache)" >&2
    exit 1
fi
case "${DIST_ACTIVATION_BITS}" in
    8|16|32) ;;
    *)
        echo "run.sh: DIST_ACTIVATION_BITS must be 8, 16, or 32" >&2
        exit 1
        ;;
esac

# --- HTTP API ---
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8080}"
MONITOR_HOST="${MONITOR_HOST:-127.0.0.1}"
MONITOR_PORT="${MONITOR_PORT:-0}"
MONITOR_ARGS=()
if [ "${MONITOR_PORT}" != "0" ]; then
    MONITOR_ARGS+=(--monitor-host "${MONITOR_HOST}"
                   --monitor-port "${MONITOR_PORT}")
fi

# --- MTP speculative decoding (draft model, P1 speedup) ---
# Disabled: the draft model does not fit reliably beside the resident profile
# and did not improve quality-adjusted throughput on this PP6 topology.
MTP_PATH="${MTP_PATH:-${SCRIPT_DIR}/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
MTP_DRAFT="${MTP_DRAFT:-2}"
MTP_ARGS=()

# --- layer split: all 43 layers (0..42), final worker owns output head ---
# PIPELINE_DEVICES and PIPELINE_LAYER_COUNTS are the portable form.  Device
# references may be ROCR indexes or stable PCI addresses such as
# pci:0000:43:00.0.  The explicit COORD_*/WORKER_SPECS form remains supported
# for existing launchers and unusual routes.
#
# A worker spec is DEVICE,LAYER_START:LAYER_END; use LAYER_START:output for
# the final worker so it also owns the output head.  Keeping the route in data
# instead of hard-coding five workers makes this shared launcher usable for
# other gfx906 counts and memory layouts.  The two example profiles below set
# the exact six-GPU maps validated on the development machine.
MODEL_LAYER_COUNT="${MODEL_LAYER_COUNT:-43}"
PIPELINE_DEVICES="${PIPELINE_DEVICES:-}"
PIPELINE_LAYER_COUNTS="${PIPELINE_LAYER_COUNTS:-}"
COORD_DEVICE="${COORD_DEVICE:-2}"
COORD_LAYERS="${COORD_LAYERS:-0:12}"
WORKER_SPECS="${WORKER_SPECS:-0,13:18 1,19:24 3,25:30 4,31:36 5,37:output}"

configure_pipeline_from_counts() {
    local i count start=0 end
    local -a devices counts specs=()

    if [ -z "${PIPELINE_DEVICES}" ] && [ -z "${PIPELINE_LAYER_COUNTS}" ]; then
        return
    fi
    if [ -z "${PIPELINE_DEVICES}" ] || [ -z "${PIPELINE_LAYER_COUNTS}" ]; then
        echo "run.sh: set both PIPELINE_DEVICES and PIPELINE_LAYER_COUNTS" >&2
        exit 1
    fi

    read -r -a devices <<< "${PIPELINE_DEVICES}"
    read -r -a counts <<< "${PIPELINE_LAYER_COUNTS}"
    if [ "${#devices[@]}" -lt 2 ] || [ "${#devices[@]}" -ne "${#counts[@]}" ]; then
        echo "run.sh: pipeline device and layer-count lists must have the same length (at least two)" >&2
        exit 1
    fi

    for i in "${!counts[@]}"; do
        count="${counts[i]}"
        if [[ ! "${count}" =~ ^[1-9][0-9]*$ ]]; then
            echo "run.sh: invalid pipeline layer count '${count}'" >&2
            exit 1
        fi
        end=$((start + count - 1))
        if [ "${i}" -eq 0 ]; then
            COORD_DEVICE="${devices[i]}"
            COORD_LAYERS="0:${end}"
        elif [ "${i}" -eq $((${#counts[@]} - 1)) ]; then
            specs+=("${devices[i]},${start}:output")
        else
            specs+=("${devices[i]},${start}:${end}")
        fi
        start=$((end + 1))
    done
    if [ "${start}" -ne "${MODEL_LAYER_COUNT}" ]; then
        echo "run.sh: pipeline counts cover ${start} layers, model requires ${MODEL_LAYER_COUNT}" >&2
        exit 1
    fi
    WORKER_SPECS="${specs[*]}"
}

ROCR_INVENTORY_LOADED=0
ROCR_BDFIDS=()

load_rocr_inventory() {
    if [ "${ROCR_INVENTORY_LOADED}" != "0" ]; then
        return
    fi
    if ! command -v rocminfo >/dev/null 2>&1; then
        echo "run.sh: rocminfo is required when using pci: device references" >&2
        exit 1
    fi
    mapfile -t ROCR_BDFIDS < <(
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
            rocminfo 2>/dev/null |
        awk '
            /^[[:space:]]*Device Type:[[:space:]]*GPU[[:space:]]*$/ { gpu = 1; next }
            gpu && /^[[:space:]]*BDFID:/ { print $2; gpu = 0 }
        '
    )
    if [ "${#ROCR_BDFIDS[@]}" -eq 0 ]; then
        echo "run.sh: rocminfo did not report any GPU BDF identifiers" >&2
        exit 1
    fi
    ROCR_INVENTORY_LOADED=1
}

resolve_device_ref() { # <ROCR index | pci:DOMAIN:BUS:DEVICE.FUNCTION>
    local ref="$1" domain bus slot function bdf i
    if [[ "${ref}" =~ ^[0-9]+$ ]]; then
        RESOLVED_DEVICE="${ref}"
        return
    fi
    if [[ ! "${ref}" =~ ^pci:(([[:xdigit:]]{4}):)?([[:xdigit:]]{2}):([[:xdigit:]]{2})\.([0-7])$ ]]; then
        echo "run.sh: invalid device '${ref}'; use a ROCR index or pci:0000:BB:DD.F" >&2
        exit 1
    fi
    domain="${BASH_REMATCH[2]:-0000}"
    bus="${BASH_REMATCH[3]}"
    slot="${BASH_REMATCH[4]}"
    function="${BASH_REMATCH[5]}"
    if [ "${domain,,}" != "0000" ]; then
        echo "run.sh: rocminfo BDFID cannot disambiguate non-zero PCI domain ${domain}" >&2
        exit 1
    fi
    bdf=$((16#${bus} * 256 + 16#${slot} * 8 + function))
    load_rocr_inventory
    for i in "${!ROCR_BDFIDS[@]}"; do
        if [ "${ROCR_BDFIDS[i]}" -eq "${bdf}" ]; then
            RESOLVED_DEVICE="${i}"
            return
        fi
    done
    echo "run.sh: PCI device '${ref}' was not found in the ROCr inventory" >&2
    exit 1
}

resolve_route_devices() {
    local coord_ref worker_spec worker_ref worker_layers worker_extra worker_dev
    local -a resolved_specs=()
    local -A used=()

    coord_ref="${COORD_DEVICE}"
    resolve_device_ref "${coord_ref}"
    COORD_DEVICE="${RESOLVED_DEVICE}"
    used["${COORD_DEVICE}"]="coordinator (${coord_ref})"

    for worker_spec in ${WORKER_SPECS}; do
        IFS=, read -r worker_ref worker_layers worker_extra <<< "${worker_spec}"
        if [ -n "${worker_extra}" ]; then
            echo "run.sh: invalid worker spec '${worker_spec}'" >&2
            exit 1
        fi
        resolve_device_ref "${worker_ref}"
        worker_dev="${RESOLVED_DEVICE}"
        if [ -n "${used[${worker_dev}]:-}" ]; then
            echo "run.sh: device ${worker_ref} resolves to ROCR ${worker_dev}, already used by ${used[${worker_dev}]}" >&2
            exit 1
        fi
        used["${worker_dev}"]="worker ${worker_layers} (${worker_ref})"
        resolved_specs+=("${worker_dev},${worker_layers}")
        if [ "${worker_ref}" != "${worker_dev}" ]; then
            echo "run.sh: resolved ${worker_ref} to ROCR device ${worker_dev}"
        fi
    done
    if [ "${coord_ref}" != "${COORD_DEVICE}" ]; then
        echo "run.sh: resolved coordinator ${coord_ref} to ROCR device ${COORD_DEVICE}"
    fi
    WORKER_SPECS="${resolved_specs[*]}"
}

configure_pipeline_from_counts
resolve_route_devices

case "${DIST_REQUIRE_WORKER_OUTPUT}" in
    auto)
        DIST_REQUIRE_WORKER_OUTPUT=0
        for route_spec in ${WORKER_SPECS}; do
            if [[ "${route_spec}" == *:output ]]; then
                DIST_REQUIRE_WORKER_OUTPUT=1
            fi
        done
        ;;
    0|1) ;;
    *)
        echo "run.sh: DIST_REQUIRE_WORKER_OUTPUT must be auto, 0, or 1" >&2
        exit 1
        ;;
esac

echo "run.sh: route coordinator ROCR ${COORD_DEVICE} layers ${COORD_LAYERS}"
for route_spec in ${WORKER_SPECS}; do
    echo "run.sh: route worker ${route_spec%%,*} layers ${route_spec#*,}"
done
if [ "${DS4_LAUNCH_DRY_RUN}" != "0" ]; then
    echo "run.sh: dry run complete"
    exit 0
fi

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
        DS4_ROCM_ATTN_COMP_CACHE_F16="${DS4_ROCM_ATTN_COMP_CACHE_F16}"
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
    DS4_ROCM_ATTN_COMP_CACHE_F16="${DS4_ROCM_ATTN_COMP_CACHE_F16}"
    DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256
    DS4_LOCK_FILE="/tmp/ds4-coordinator-${COORD_DEVICE}.lock"
    ROCR_VISIBLE_DEVICES="${COORD_DEVICE}")
if [ "${DS4_COORD_SERIALIZE}" != "0" ]; then
    COORD_ENV+=(AMD_SERIALIZE_KERNEL="${DS4_COORD_SERIALIZE}")
fi
COORD_PREFIX=()
OUTPUT_ROUTE_ARGS=()
if [ "${DIST_REQUIRE_WORKER_OUTPUT}" != "0" ]; then
    OUTPUT_ROUTE_ARGS+=(--dist-require-worker-output)
fi
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
    "${MONITOR_ARGS[@]}" \
    "${TRACE_ARGS[@]}" \
    --role coordinator --layers "${COORD_LAYERS}" \
    "${MTP_ARGS[@]}" \
    "${SSD_ARGS[@]}" \
    --listen "${DIST_HOST}" "${DIST_PORT}" \
    --dist-prefill-chunk "${PREFILL_CHUNK}" \
    --dist-prefill-window "${DIST_WINDOW}" \
    --dist-activation-bits "${DIST_ACTIVATION_BITS}" \
    "${OUTPUT_ROUTE_ARGS[@]}" &
COORD_PID=$!
wait "${COORD_PID}"
