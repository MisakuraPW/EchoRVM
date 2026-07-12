#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA=()
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA+=(--output_dir "$OUTPUT_DIR")
fi
if [[ -n "${HIERA_REPO:-}" ]]; then
  EXTRA+=(--hiera_repo "$HIERA_REPO")
fi

python trainers/train_hiera_mae.py \
  --config configs/pretrain/camus_hiera_t_mae.yaml \
  --data_root "$CAMUS_DATA_ROOT" \
  "${EXTRA[@]}" \
  "$@"
