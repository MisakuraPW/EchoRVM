#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="${1:-/root/autodl-fs/datasets/EchoNet-Dynamic}"
OUTPUT_ROOT="${2:-/root/autodl-fs/augmented/EchoNet-Dynamic}"
VARIANTS="${VARIANTS:-1}"

python tools/augment_dataset.py \
  --dataset echonet \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --variants "${VARIANTS}" \
  --img-size 112
