#!/usr/bin/env bash
# Six-gfx906 profile for maximum throughput: all model weights stay resident.
# Edit MODEL_PATH here; no launch-time environment variables are required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${SCRIPT_DIR}/ds4flash.gguf"
# Seven resident layers fit the 16GB workers at 32K context.  This balanced
# split is about 50% faster in prefill than the capacity-oriented 13 + 6x5.
export CTX=32768
export MAX_TOKENS=32768
export PREFILL_CHUNK=64
export DIST_WINDOW=5
export WORKER_START_DELAY=5
export HTTP_HOST=0.0.0.0
export HTTP_PORT=8080
export DIST_HOST=127.0.0.1
export DIST_PORT=19000
export DS4_PROFILE=0
export DS4_COORD_SERIALIZE=0
export SSD_STREAMING=0
export SSD_STREAMING_CACHE_EXPERTS=
export SSD_STREAMING_PRELOAD_EXPERTS=
export SSD_STREAMING_COLD=0
export COORD_DEVICE=2
export COORD_LAYERS=0:7
export WORKER_SPECS="0,8:14 1,15:21 3,22:28 4,29:35 5,36:42"

exec "${SCRIPT_DIR}/run.sh"
