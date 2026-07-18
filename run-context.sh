#!/usr/bin/env bash
# Six-gfx906 capacity profile: bounded SSD expert cache and native 1M context.
# Edit MODEL_PATH here; no launch-time environment variables are required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${SCRIPT_DIR}/ds4flash.gguf"
export CTX=1000000
export MAX_TOKENS=1000000
export PREFILL_CHUNK=64
export DIST_WINDOW=5
export WORKER_START_DELAY=5
export HTTP_HOST=0.0.0.0
export HTTP_PORT=8080
export DIST_HOST=127.0.0.1
export DIST_PORT=19000
export DS4_PROFILE=0
export DS4_COORD_SERIALIZE=0
export SSD_STREAMING=1
export SSD_STREAMING_CACHE_EXPERTS=2GB
export SSD_STREAMING_PRELOAD_EXPERTS=
export SSD_STREAMING_COLD=0

exec "${SCRIPT_DIR}/run.sh"
