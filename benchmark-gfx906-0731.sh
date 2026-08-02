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
set -euo pipefail

ROOT=/home/mayor86/App/ds4
MODEL=/home/mayor86/llama/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
DSPARK=/home/mayor86/llama/models/DeepSeek-V4-Flash-DSpark-support.gguf
PROMPT="${ROOT}/tests/long_context_security_prompt.txt"
RESULT_ROOT="${ROOT}/.ds4-benchmarks/gfx906-0731"

# Fixed benchmark definition. Edit these values here, not in the environment,
# when deliberately creating a new benchmark series.
CTX_ALLOC=300000
FRONTIER=8192
GEN_TOKENS=256
PREFILL_CHUNK=64
DIST_WINDOW=5
ACTIVATION_BITS=16
DSPARK_CONFIDENCE=0.9

# ROCR indexes on the validation host. The 32 GiB card is device 2 and owns
# layers 34:42, the output head, and the 5.58 GiB DSpark support model.
COORD_DEVICE=3
COORD_LAYERS=0:5
FINAL_DEVICE=2
FINAL_LAYERS=34:output
WORKER_SPECS=("0 6:12" "1 13:19" "4 20:26" "5 27:33")

MODE="${1:-all}"
case "${MODE}" in
    all|baseline|dspark|tuned|forced) ;;
    verify|verify-tuned|tune)
        FRONTIER=1024
        GEN_TOKENS=16
        ;;
    *)
        echo "usage: $0 [all|baseline|dspark|tuned|forced|verify|verify-tuned|tune]" >&2
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

start_worker() {
    local case_name="$1" port="$2" device="$3" layers="$4"
    shift 4
    env ROCR_VISIBLE_DEVICES="${device}" \
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
    if [ "${case_name}" != baseline ]; then
        final_args=(--mtp "${DSPARK}" --dspark --dspark-confidence "${confidence}")
        bench_args=(--mtp "${DSPARK}" --dspark --dspark-confidence "${confidence}")
    fi

    env ROCR_VISIBLE_DEVICES="${FINAL_DEVICE}" \
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

    env ROCR_VISIBLE_DEVICES="${COORD_DEVICE}" \
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
        "${bench_args[@]}" \
        >"${OUT}/${case_name}.stdout" 2>"${OUT}/${case_name}.log"

    # Give workers time to observe the closed route and print final counters.
    sleep 2
    cleanup
    tail -n 1 "${OUT}/${case_name}.csv"
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
