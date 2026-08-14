#!/usr/bin/env bash
# Six-gfx906 profile for maximum throughput: all model weights stay resident.
# Edit MODEL_PATH here; no launch-time environment variables are required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="/home/mayor86/llama/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
# The 16 GiB coordinator owns six layers so the FP32 compressed cache retains
# runtime scratch headroom at the 700K frontier.  The four middle workers own
# seven layers each; the 32 GiB final GPU owns nine layers plus the output head.
# PCI references remain stable when ROCr renumbers devices after a reboot.  On
# another six-gfx906 host, edit only this ordered device list: put the
# largest-VRAM card last.
export CTX=700000
export MAX_TOKENS=32768
export PREFILL_CHUNK=256
export DIST_WINDOW=6
export DIST_ACTIVATION_BITS=16
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
export PIPELINE_DEVICES="pci:0000:46:00.0 pci:0000:63:00.0 pci:0000:66:00.0 pci:0000:30:00.0 pci:0000:03:00.0 pci:0000:43:00.0"
export PIPELINE_LAYER_COUNTS="6 7 7 7 7 9"

exec "${SCRIPT_DIR}/run.sh"
