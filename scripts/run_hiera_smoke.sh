#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python trainers/train_hiera_mae.py \
  --config configs/pretrain/hiera_smoke_t_128.yaml \
  --max_steps 2 \
  "$@"
