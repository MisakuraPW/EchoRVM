#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA=()
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA+=(--output_dir "$OUTPUT_DIR")
fi

python trainers/train_hiera_mae.py \
  --config configs/pretrain/echonet_hiera_t_mae.yaml \
  --data_root "$ECHO_DATA_ROOT" \
  "${EXTRA[@]}" \
  "$@"
