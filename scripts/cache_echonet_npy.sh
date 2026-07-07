#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="${INPUT_ROOT:-/root/autodl-fs/datasets/EchoNet-Dynamic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python tools/cache_echonet_npy.py \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --num-workers "$NUM_WORKERS" \
  "$@"
