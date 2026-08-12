#!/usr/bin/env bash
# Reproducible six-GPU gfx906 benchmark for DeepSeek-V4-Flash 0731.
#
# Usage:
#   ./benchmark-gfx906-0731.sh all       # baseline, then production DSpark
#   ./benchmark-gfx906-0731.sh baseline  # baseline only
#   ./benchmark-gfx906-0731.sh dspark    # production DSpark only
#   ./benchmark-gfx906-0731.sh tuned     # DSpark at the measured 0.5 threshold
#   ./benchmark-gfx906-0731.sh forced    # DSpark every cycle, no pruning
#   ./benchmark-gfx906-0731.sh verify    # short baseline/forced correctness run
#   ./benchmark-gfx906-0731.sh verify-tuned # short baseline/0.5 correctness run
#   ./benchmark-gfx906-0731.sh tune      # short 0.9-versus-0.5 threshold run
#   ./benchmark-gfx906-0731.sh graph     # intrusive layer-20 GPU graph profile
#   ./benchmark-gfx906-0731.sh chunk128  # baseline with 128-token prefill chunks
#   ./benchmark-gfx906-0731.sh chunk256  # baseline with 256-token prefill chunks
#   ./benchmark-gfx906-0731.sh swap14    # swap ROCR devices 1 and 4
#   ./benchmark-gfx906-0731.sh optimized # 256-token chunks plus the device swap
#   ./benchmark-gfx906-0731.sh windowN   # chunk 256, window N (3, 6, or 8)
#   ./benchmark-gfx906-0731.sh moe-rpbN  # chunk 256, MoE decode rows/block N
#   ./benchmark-gfx906-0731.sh q8-rpbN   # chunk 256, Q8 decode rows/block N
#   ./benchmark-gfx906-0731.sh decode-base # short-frontier decode reference
#   ./benchmark-gfx906-0731.sh decode-graph # gfx906 HIP graph capture + logging
#   ./benchmark-gfx906-0731.sh decode-eager # disable HIP graphs for A/B comparison
#   ./benchmark-gfx906-0731.sh decode-rpb8 # combined gfx906 rows/block candidate
#   ./benchmark-gfx906-0731.sh moe-profile # intrusive Routed-MoE decode profile
#   ./benchmark-gfx906-0731.sh no-moe-wmma # compare scalar expert tiles in prefill
#   ./benchmark-gfx906-0731.sh legacy-moe-wmma # opt into emulated rocWMMA
#   ./benchmark-gfx906-0731.sh logits      # full-logit WMMA/scalar comparison
#   ./benchmark-gfx906-0731.sh logits-porting # same comparison, second prompt
#   ./benchmark-gfx906-0731.sh logits-speedup # same comparison, third prompt
#   ./benchmark-gfx906-0731.sh logits-f16 # one current-backend full-logit dump
#   ./benchmark-gfx906-0731.sh dspark256 # production DSpark with chunk 256
#   ./benchmark-gfx906-0731.sh dspark-long # paired baseline/DSpark, 1024 decode tokens
#   ./benchmark-gfx906-0731.sh dspark-context16k # paired run at a 16K frontier
#   ./benchmark-gfx906-0731.sh dspark-context16k-only # DSpark half after a saved baseline
#   ./benchmark-gfx906-0731.sh dspark-context32k # paired run at a 32K frontier
#   ./benchmark-gfx906-0731.sh dspark-context32k-only # DSpark half after a saved baseline
#   ./benchmark-gfx906-0731.sh dspark-context64k # paired run at a 64K frontier
#   ./benchmark-gfx906-0731.sh dspark-context64k-only # DSpark half after a saved baseline
#   ./benchmark-gfx906-0731.sh long64k   # 64K prefill frontier, short decode
#   ./benchmark-gfx906-0731.sh long64k-700 # same frontier with production 700K allocation
#   ./benchmark-gfx906-0731.sh long16k   # 16K prefill frontier, short decode
#   ./benchmark-gfx906-0731.sh long300k  # 300K prefill frontier, short decode
set -euo pipefail

ROOT=/home/mayor86/App/ds4
MODEL=/home/mayor86/llama/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
DSPARK=/home/mayor86/llama/models/DeepSeek-V4-Flash-DSpark-support.gguf
PROMPT="${ROOT}/tests/long_context_security_prompt.txt"
LOGITS_PROMPT="${ROOT}/PULL_REQUEST-GFX906.md"
RESULT_ROOT="${ROOT}/.ds4-benchmarks/gfx906-0731"

# Fixed benchmark definition. Edit these values here, not in the environment,
# when deliberately creating a new benchmark series.
CTX_ALLOC=300000
FRONTIER=8192
GEN_TOKENS=256
PREFILL_CHUNK=64
DIST_WINDOW="${DIST_WINDOW:-6}"
ACTIVATION_BITS="${ACTIVATION_BITS:-16}"
DSPARK_CONFIDENCE=0.9
PROFILE_LAYER=
RUNTIME_ENV=()
BENCH_EXTRA_ARGS=()

# ROCR indexes on the validation host. The coordinator is a 16 GiB Pro VII;
# device 2 is the 32 GiB card and owns layers 35:42 plus the output head.
# This is also the resident FP32 700K production split; window 6 keeps one
# prefill chunk available for each of the six balanced stages.
COORD_DEVICE=3
COORD_LAYERS=0:6
FINAL_DEVICE=2
FINAL_LAYERS=35:output
WORKER_SPECS=("0 7:13" "1 14:20" "4 21:27" "5 28:34")

MODE="${1:-all}"
case "${MODE}" in
    all|baseline|dspark|tuned|forced) ;;
    verify|verify-tuned|tune)
        FRONTIER=1024
        GEN_TOKENS=16
        ;;
    graph)
        FRONTIER=1024
        GEN_TOKENS=16
        PROFILE_LAYER=20
        ;;
    chunk128)
        PREFILL_CHUNK=128
        ;;
    chunk256)
        PREFILL_CHUNK=256
        ;;
    swap14)
        WORKER_SPECS=("0 7:13" "4 14:20" "1 21:27" "5 28:34")
        ;;
    optimized)
        PREFILL_CHUNK=256
        WORKER_SPECS=("0 7:13" "4 14:20" "1 21:27" "5 28:34")
        ;;
    window3|window6|window8)
        PREFILL_CHUNK=256
        DIST_WINDOW="${MODE#window}"
        ;;
    moe-rpb1|moe-rpb2|moe-rpb4|moe-rpb8)
        PREFILL_CHUNK=256
        FRONTIER=1024
        RUNTIME_ENV+=("DS4_ROCM_MOE_DECODE_RPB=${MODE#moe-rpb}")
        ;;
    q8-rpb1|q8-rpb2|q8-rpb4|q8-rpb8)
        PREFILL_CHUNK=256
        FRONTIER=1024
        RUNTIME_ENV+=("DS4_ROCM_Q8_DECODE_RPB=${MODE#q8-rpb}")
        ;;
    decode-base|decode-f16|decode-f32)
        PREFILL_CHUNK=256
        FRONTIER=1024
        if [ "${MODE}" = decode-f16 ]; then
            RUNTIME_ENV+=("DS4_ROCM_ATTN_COMP_CACHE_F16=1")
        elif [ "${MODE}" = decode-f32 ]; then
            RUNTIME_ENV+=("DS4_ROCM_ATTN_COMP_CACHE_F16=0")
        fi
        ;;
    decode-graph)
        PREFILL_CHUNK=256
        FRONTIER=1024
        RUNTIME_ENV+=("DS4_ROCM_DECODE_GRAPHS=1")
        RUNTIME_ENV+=("DS4_ROCM_DECODE_GRAPH_LOG=1")
        ;;
    decode-eager)
        PREFILL_CHUNK=256
        FRONTIER=1024
        RUNTIME_ENV+=("DS4_ROCM_DECODE_GRAPHS=0")
        ;;
    decode-rpb8)
        PREFILL_CHUNK=256
        FRONTIER=1024
        RUNTIME_ENV+=("DS4_ROCM_MOE_DECODE_RPB=8")
        RUNTIME_ENV+=("DS4_ROCM_Q8_DECODE_RPB=8")
        ;;
    moe-profile)
        PREFILL_CHUNK=256
        FRONTIER=1024
        GEN_TOKENS=64
        RUNTIME_ENV+=("DS4_ROCM_MOE_DECODE_PROFILE=1")
        RUNTIME_ENV+=("DS4_ROCM_DECODE_GRAPHS=0")
        ;;
    no-moe-wmma)
        PREFILL_CHUNK=256
        RUNTIME_ENV+=("DS4_ROCM_DISABLE_MOE_WMMA_HOT=1")
        ;;
    legacy-moe-wmma)
        PREFILL_CHUNK=256
        RUNTIME_ENV+=("DS4_ROCM_ENABLE_EMULATED_MOE_WMMA=1")
        ;;
    logits|logits-f16|logits-f32)
        PREFILL_CHUNK=256
        ;;
    logits-porting)
        PREFILL_CHUNK=256
        LOGITS_PROMPT="${ROOT}/PORTING-GFX906.md"
        ;;
    logits-speedup)
        PREFILL_CHUNK=256
        LOGITS_PROMPT="${ROOT}/SPEEDUP-GFX906.md"
        ;;
    dspark256)
        PREFILL_CHUNK=256
        ;;
    dspark-long)
        PREFILL_CHUNK=256
        GEN_TOKENS=1024
        ;;
    dspark-context32k|dspark-context32k-only)
        PREFILL_CHUNK=256
        FRONTIER=32768
        GEN_TOKENS=256
        ;;
    dspark-context16k|dspark-context16k-only)
        PREFILL_CHUNK=256
        FRONTIER=16384
        GEN_TOKENS=256
        ;;
    dspark-context64k|dspark-context64k-only)
        PREFILL_CHUNK=256
        FRONTIER=65536
        GEN_TOKENS=256
        BENCH_EXTRA_ARGS+=("--repeat-prompt")
        ;;
    long16k|long16k-f16|long16k-f32)
        PREFILL_CHUNK=256
        FRONTIER=16384
        GEN_TOKENS=64
        BENCH_EXTRA_ARGS+=("--repeat-prompt")
        if [ "${MODE}" = long16k-f16 ]; then
            RUNTIME_ENV+=("DS4_ROCM_ATTN_COMP_CACHE_F16=1")
        elif [ "${MODE}" = long16k-f32 ]; then
            RUNTIME_ENV+=("DS4_ROCM_ATTN_COMP_CACHE_F16=0")
        fi
        ;;
    long64k)
        PREFILL_CHUNK=256
        FRONTIER=65536
        GEN_TOKENS=32
        BENCH_EXTRA_ARGS+=("--repeat-prompt")
        ;;
    long64k-700)
        CTX_ALLOC=700000
        PREFILL_CHUNK=256
        FRONTIER=65536
        GEN_TOKENS=32
        BENCH_EXTRA_ARGS+=("--repeat-prompt")
        ;;
    long300k)
        CTX_ALLOC=700000
        PREFILL_CHUNK=256
        FRONTIER=300000
        GEN_TOKENS=16
        BENCH_EXTRA_ARGS+=("--repeat-prompt")
        ;;
    *)
        echo "usage: $0 [all|baseline|dspark|tuned|forced|verify|verify-tuned|tune|graph|chunk128|chunk256|swap14|optimized|window{3,6,8}|decode-base|decode-f16|decode-f32|decode-graph|decode-eager|decode-rpb8|moe-rpb{1,2,4,8}|q8-rpb{1,2,4,8}|moe-profile|no-moe-wmma|legacy-moe-wmma|logits|logits-f16|logits-f32|logits-porting|logits-speedup|dspark256|dspark-long|dspark-context{16k,16k-only,32k,32k-only,64k,64k-only}|long16k|long16k-f16|long16k-f32|long64k|long64k-700|long300k]" >&2
        exit 2
        ;;
esac

for path in "${MODEL}" "${DSPARK}" "${PROMPT}" \
            "${ROOT}/ds4" "${ROOT}/ds4-bench"; do
    if [ ! -e "${path}" ]; then
        echo "benchmark: missing ${path}" >&2
        exit 1
    fi
done
if pgrep -f '[/]ds4(-bench|-server)? .*--role (worker|coordinator)' >/dev/null; then
    echo "benchmark: a distributed ds4 process is already active" >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)-${MODE}"
OUT="${RESULT_ROOT}/${RUN_ID}"
mkdir -p "${OUT}"
PIDS=()

cleanup() {
    if [ "${#PIDS[@]}" -ne 0 ]; then
        kill -9 "${PIDS[@]}" 2>/dev/null || true
        for pid in "${PIDS[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
    fi
    PIDS=()
}
trap cleanup EXIT INT TERM

gpu_max_edge_temp_millideg() {
    local file value max=0
    for file in /sys/class/drm/card[0-9]/device/hwmon/hwmon*/temp1_input; do
        if [ ! -r "${file}" ]; then
            continue
        fi
        read -r value <"${file}"
        if [[ "${value}" =~ ^[0-9]+$ ]] && [ "${value}" -gt "${max}" ]; then
            max="${value}"
        fi
    done
    echo "${max}"
}

wait_for_gpu_cooldown() {
    local target="${1:-75000}" timeout="${2:-900}"
    local start="${SECONDS}" max_temp
    while true; do
        max_temp="$(gpu_max_edge_temp_millideg)"
        if [ "${max_temp}" -eq 0 ]; then
            echo "benchmark: GPU temperature sensors unavailable; continuing"
            return 0
        fi
        echo "benchmark: cooldown max GPU edge temp $((max_temp / 1000)) C, target $((target / 1000)) C"
        if [ "${max_temp}" -le "${target}" ]; then
            return 0
        fi
        if [ $((SECONDS - start)) -ge "${timeout}" ]; then
            echo "benchmark: cooldown timed out after ${timeout}s" >&2
            return 1
        fi
        sleep 15
    done
}

start_worker() {
    local case_name="$1" port="$2" device="$3" layers="$4"
    shift 4
    local -a profile_env=()
    if [ -n "${PROFILE_LAYER}" ]; then
        profile_env=(
            "DS4_ROCM_LAYER_STAGE_PROFILE=${PROFILE_LAYER}"
            "DS4_ROCM_DECODE_STAGE_PROFILE=${PROFILE_LAYER}"
        )
    fi
    env "${profile_env[@]}" "${RUNTIME_ENV[@]}" ROCR_VISIBLE_DEVICES="${device}" \
        DS4_LOCK_FILE="/tmp/ds4-bench-${RUN_ID}-${case_name}-${device}.lock" \
        DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256 \
        "${ROOT}/ds4" -m "${MODEL}" --rocm \
        --prefill-chunk "${PREFILL_CHUNK}" --ctx "${CTX_ALLOC}" \
        --role worker --layers "${layers}" \
        --coordinator 127.0.0.1 "${port}" "$@" \
        >"${OUT}/${case_name}-worker-${device}.stdout" \
        2>"${OUT}/${case_name}-worker-${device}.log" &
    PIDS+=("$!")
}

run_case() {
    local case_name="$1" port="$2" confidence="$3" scheduler="$4"
    local spec device layers
    local -a final_args=() bench_args=()

    echo "benchmark: starting ${case_name} (ctx=${CTX_ALLOC}, frontier=${FRONTIER}, gen=${GEN_TOKENS})"
    for spec in "${WORKER_SPECS[@]}"; do
        read -r device layers <<<"${spec}"
        start_worker "${case_name}" "${port}" "${device}" "${layers}"
    done
    case "${case_name}" in
        baseline|graph|chunk128|chunk256|swap14|optimized|window*|decode-base|decode-f16|decode-f32|decode-graph|decode-eager|decode-rpb8|moe-rpb*|q8-rpb*|moe-profile|no-moe-wmma|legacy-moe-wmma|long*) ;;
        *)
            final_args=(--mtp "${DSPARK}" --dspark --dspark-confidence "${confidence}")
            bench_args=(--mtp "${DSPARK}" --dspark --dspark-confidence "${confidence}")
            ;;
    esac

    env "${RUNTIME_ENV[@]}" ROCR_VISIBLE_DEVICES="${FINAL_DEVICE}" \
        DS4_LOCK_FILE="/tmp/ds4-bench-${RUN_ID}-${case_name}-${FINAL_DEVICE}.lock" \
        DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256 \
        DS4_DSPARK_SCHEDULER="${scheduler}" \
        DS4_DSPARK_STATS=1 \
        "${ROOT}/ds4" -m "${MODEL}" --rocm \
        --prefill-chunk "${PREFILL_CHUNK}" --ctx "${CTX_ALLOC}" \
        --role worker --layers "${FINAL_LAYERS}" \
        --coordinator 127.0.0.1 "${port}" "${final_args[@]}" \
        >"${OUT}/${case_name}-worker-${FINAL_DEVICE}.stdout" \
        2>"${OUT}/${case_name}-worker-${FINAL_DEVICE}.log" &
    PIDS+=("$!")

    local -a profile_env=()
    if [ -n "${PROFILE_LAYER}" ]; then
        profile_env=(
            "DS4_ROCM_LAYER_STAGE_PROFILE=${PROFILE_LAYER}"
            "DS4_ROCM_DECODE_STAGE_PROFILE=${PROFILE_LAYER}"
        )
    fi
    local bench_status=0
    env "${profile_env[@]}" "${RUNTIME_ENV[@]}" ROCR_VISIBLE_DEVICES="${COORD_DEVICE}" \
        DS4_LOCK_FILE="/tmp/ds4-bench-${RUN_ID}-${case_name}-coordinator.lock" \
        DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256 \
        DS4_DSPARK_SCHEDULER="${scheduler}" \
        DS4_DSPARK_STATS=1 \
        "${ROOT}/ds4-bench" -m "${MODEL}" --rocm \
        --prompt-file "${PROMPT}" \
        --ctx-start "${FRONTIER}" --ctx-max "${FRONTIER}" \
        --ctx-alloc "${CTX_ALLOC}" --gen-tokens "${GEN_TOKENS}" \
        --prefill-chunk "${PREFILL_CHUNK}" --show-output \
        --csv "${OUT}/${case_name}.csv" \
        --role coordinator --layers "${COORD_LAYERS}" \
        --listen 127.0.0.1 "${port}" \
        --dist-prefill-chunk "${PREFILL_CHUNK}" \
        --dist-prefill-window "${DIST_WINDOW}" \
        --dist-activation-bits "${ACTIVATION_BITS}" \
        --dist-require-worker-output \
        "${BENCH_EXTRA_ARGS[@]}" \
        "${bench_args[@]}" \
        >"${OUT}/${case_name}.stdout" 2>"${OUT}/${case_name}.log" || bench_status=$?

    # Give workers time to observe the closed route and print final counters.
    sleep 2
    cleanup
    if [ "${bench_status}" -ne 0 ]; then
        echo "benchmark: ${case_name} failed with ds4-bench status ${bench_status}; see ${OUT}/${case_name}.log" >&2
        return "${bench_status}"
    fi
    tail -n 1 "${OUT}/${case_name}.csv"
}

run_logits_case() {
    local case_name="$1" port="$2" wmma_mode="$3" quality="$4" cache_format="${5:-f32}"
    local spec device layers
    local -a quality_args=()
    RUNTIME_ENV=()
    if [ "${cache_format}" = f16 ]; then
        RUNTIME_ENV+=("DS4_ROCM_ATTN_COMP_CACHE_F16=1")
    else
        RUNTIME_ENV+=("DS4_ROCM_ATTN_COMP_CACHE_F16=0")
    fi
    if [ "${wmma_mode}" = disable ]; then
        RUNTIME_ENV+=("DS4_ROCM_DISABLE_MOE_WMMA_HOT=1")
    elif [ "${wmma_mode}" = enable ]; then
        RUNTIME_ENV+=("DS4_ROCM_ENABLE_EMULATED_MOE_WMMA=1")
    fi
    if [ "${quality}" = 1 ]; then
        quality_args+=("--quality")
    fi

    echo "benchmark: starting ${case_name} full-logit comparison"
    for spec in "${WORKER_SPECS[@]}"; do
        read -r device layers <<<"${spec}"
        start_worker "${case_name}" "${port}" "${device}" "${layers}" \
            "${quality_args[@]}"
    done
    env "${RUNTIME_ENV[@]}" ROCR_VISIBLE_DEVICES="${FINAL_DEVICE}" \
        DS4_LOCK_FILE="/tmp/ds4-bench-${RUN_ID}-${case_name}-${FINAL_DEVICE}.lock" \
        DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256 \
        "${ROOT}/ds4" -m "${MODEL}" --rocm \
        --prefill-chunk "${PREFILL_CHUNK}" --ctx "${CTX_ALLOC}" \
        --role worker --layers "${FINAL_LAYERS}" \
        --coordinator 127.0.0.1 "${port}" "${quality_args[@]}" \
        >"${OUT}/${case_name}-worker-${FINAL_DEVICE}.stdout" \
        2>"${OUT}/${case_name}-worker-${FINAL_DEVICE}.log" &
    PIDS+=("$!")

    env "${RUNTIME_ENV[@]}" ROCR_VISIBLE_DEVICES="${COORD_DEVICE}" \
        DS4_LOCK_FILE="/tmp/ds4-bench-${RUN_ID}-${case_name}-coordinator.lock" \
        DS4_ROCM_WEIGHT_ARENA_CHUNK_MB=256 \
        "${ROOT}/ds4" -m "${MODEL}" --rocm \
        --prompt-file "${LOGITS_PROMPT}" --nothink --temp 0 --tokens 1 \
        --ctx "${CTX_ALLOC}" --prefill-chunk "${PREFILL_CHUNK}" \
        --dump-logits "${OUT}/${case_name}.json" \
        --role coordinator --layers "${COORD_LAYERS}" \
        --listen 127.0.0.1 "${port}" \
        --dist-prefill-chunk "${PREFILL_CHUNK}" \
        --dist-prefill-window "${DIST_WINDOW}" \
        --dist-activation-bits "${ACTIVATION_BITS}" \
        --dist-require-worker-output "${quality_args[@]}" \
        >"${OUT}/${case_name}.stdout" 2>"${OUT}/${case_name}.log"
    sleep 2
    cleanup
}

case "${MODE}" in
    all)
        run_case baseline 19231 0 0
        run_case dspark 19232 "${DSPARK_CONFIDENCE}" 1
        ;;
    baseline)
        run_case baseline 19231 0 0
        ;;
    dspark)
        run_case dspark 19232 "${DSPARK_CONFIDENCE}" 1
        ;;
    tuned)
        run_case tuned 19236 0.5 1
        ;;
    forced)
        run_case forced 19233 0 0
        ;;
    verify)
        run_case baseline 19234 0 0
        run_case forced 19235 0 0
        ;;
    verify-tuned)
        run_case baseline 19239 0 0
        run_case tuned 19240 0.5 1
        ;;
    tune)
        run_case dspark 19237 "${DSPARK_CONFIDENCE}" 1
        run_case tuned 19238 0.5 1
        ;;
    graph)
        run_case graph 19241 0 0
        ;;
    chunk128)
        run_case chunk128 19242 0 0
        ;;
    chunk256)
        run_case chunk256 19243 0 0
        ;;
    swap14)
        run_case swap14 19244 0 0
        ;;
    optimized)
        run_case optimized 19245 0 0
        ;;
    window3|window6|window8)
        run_case "${MODE}" 19246 0 0
        ;;
    moe-rpb1|moe-rpb2|moe-rpb4|moe-rpb8)
        run_case "${MODE}" 19247 0 0
        ;;
    q8-rpb1|q8-rpb2|q8-rpb4|q8-rpb8)
        run_case "${MODE}" 19248 0 0
        ;;
    decode-base|decode-f16|decode-f32|decode-graph|decode-eager|decode-rpb8|moe-profile)
        run_case "${MODE}" 19251 0 0
        ;;
    no-moe-wmma)
        run_case "${MODE}" 19252 0 0
        ;;
    legacy-moe-wmma)
        run_case "${MODE}" 19252 0 0
        ;;
    logits|logits-porting|logits-speedup)
        run_logits_case default 19253 default 0
        run_logits_case legacy-moe-wmma 19254 enable 0
        run_logits_case quality 19255 default 1
        "${ROOT}/speed-bench/compare_logits.py" \
            "${OUT}/legacy-moe-wmma.json" "${OUT}/default.json" \
            | tee "${OUT}/legacy-vs-default.txt"
        "${ROOT}/speed-bench/compare_logits.py" \
            "${OUT}/quality.json" "${OUT}/default.json" \
            | tee "${OUT}/quality-vs-default.txt"
        "${ROOT}/speed-bench/compare_logits.py" \
            "${OUT}/quality.json" "${OUT}/legacy-moe-wmma.json" \
            | tee "${OUT}/quality-vs-legacy.txt"
        ;;
    logits-f16)
        run_logits_case f16 19253 default 0 f16
        ;;
    logits-f32)
        run_logits_case f32 19253 default 0 f32
        ;;
    dspark256)
        run_case dspark256 19249 "${DSPARK_CONFIDENCE}" 1
        ;;
    dspark-long)
        run_case baseline 19256 0 0
        run_case dspark 19257 "${DSPARK_CONFIDENCE}" 1
        ;;
    dspark-context16k|dspark-context32k|dspark-context64k)
        run_case baseline 19258 0 0
        wait_for_gpu_cooldown 75000 900
        run_case dspark 19259 "${DSPARK_CONFIDENCE}" 1
        ;;
    dspark-context16k-only|dspark-context32k-only|dspark-context64k-only)
        wait_for_gpu_cooldown 75000 900
        run_case dspark 19259 "${DSPARK_CONFIDENCE}" 1
        ;;
    long16k|long16k-f16|long16k-f32|long64k|long64k-700|long300k)
        run_case "${MODE}" 19250 0 0
        ;;
esac

if [ -f "${OUT}/baseline.log" ] && [ -f "${OUT}/dspark.log" ]; then
    sed -n 's/^ds4-bench: gen\[ctx=.*\] token ids: //p' \
        "${OUT}/baseline.log" >"${OUT}/baseline.tokens"
    sed -n 's/^ds4-bench: gen\[ctx=.*\] token ids: //p' \
        "${OUT}/dspark.log" >"${OUT}/dspark.tokens"
    if cmp -s "${OUT}/baseline.tokens" "${OUT}/dspark.tokens"; then
        echo "token_identity=PASS" | tee "${OUT}/correctness.txt"
    else
        echo "token_identity=FAIL" | tee "${OUT}/correctness.txt"
        diff -u "${OUT}/baseline.tokens" "${OUT}/dspark.tokens" \
            >>"${OUT}/correctness.txt" || true
    fi
elif [ -f "${OUT}/baseline.log" ] && [ -f "${OUT}/forced.log" ]; then
    sed -n 's/^ds4-bench: gen\[ctx=.*\] token ids: //p' \
        "${OUT}/baseline.log" >"${OUT}/baseline.tokens"
    sed -n 's/^ds4-bench: gen\[ctx=.*\] token ids: //p' \
        "${OUT}/forced.log" >"${OUT}/forced.tokens"
    if cmp -s "${OUT}/baseline.tokens" "${OUT}/forced.tokens"; then
        echo "token_identity=PASS" | tee "${OUT}/correctness.txt"
    else
        echo "token_identity=FAIL" | tee "${OUT}/correctness.txt"
        diff -u "${OUT}/baseline.tokens" "${OUT}/forced.tokens" \
            >>"${OUT}/correctness.txt" || true
    fi
elif [ -f "${OUT}/baseline.log" ] && [ -f "${OUT}/tuned.log" ]; then
    sed -n 's/^ds4-bench: gen\[ctx=.*\] token ids: //p' \
        "${OUT}/baseline.log" >"${OUT}/baseline.tokens"
    sed -n 's/^ds4-bench: gen\[ctx=.*\] token ids: //p' \
        "${OUT}/tuned.log" >"${OUT}/tuned.tokens"
    if cmp -s "${OUT}/baseline.tokens" "${OUT}/tuned.tokens"; then
        echo "token_identity=PASS" | tee "${OUT}/correctness.txt"
    else
        echo "token_identity=FAIL" | tee "${OUT}/correctness.txt"
        diff -u "${OUT}/baseline.tokens" "${OUT}/tuned.tokens" \
            >>"${OUT}/correctness.txt" || true
    fi
fi

echo "benchmark: results in ${OUT}"
