#!/usr/bin/env bash
# Six-gfx906 profile for maximum throughput: all model weights stay resident.
# Edit MODEL_PATH here; no launch-time environment variables are required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="/home/mayor86/llama/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
# The coordinator runs on a 16 GiB Pro VII. Four seven-layer workers use the
# other 16 GiB cards, while the 32 GiB device owns the final nine layers and
# output head. This preserves the validated activation boundaries and keeps a
# 700K FP32 KV cache resident without SSD streaming. A 256-token chunk is the
# measured long-prefill choice; keep output budget and context separate.
export CTX=700000
export MAX_TOKENS=32768
export PREFILL_CHUNK=256
export DIST_WINDOW=5
export WORKER_START_DELAY=5
export HTTP_HOST=0.0.0.0
export HTTP_PORT=8080
export MONITOR_HOST=127.0.0.1
export MONITOR_PORT=9091
export DIST_HOST=127.0.0.1
export DIST_PORT=19000
export DS4_PROFILE=0
export DS4_COORD_SERIALIZE=0
# Quality-first persistent cache. The experimental compact cache changes greedy
# output and is intentionally not used by production profiles.
export DS4_ROCM_ATTN_COMP_CACHE_F16=0
export SSD_STREAMING=0
export SSD_STREAMING_CACHE_EXPERTS=
export SSD_STREAMING_PRELOAD_EXPERTS=
export SSD_STREAMING_COLD=0
export COORD_DEVICE=3
export COORD_LAYERS=0:5
export WORKER_SPECS="0,6:12 1,13:19 4,20:26 5,27:33 2,34:output"

exec "${SCRIPT_DIR}/run.sh"
