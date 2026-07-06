#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mae_debug.yaml}"

if [[ ! -f "trainers/train_mae.py" ]]; then
  echo "[ERROR] trainers/train_mae.py not found."
  echo "The AutoDL scaffold is ready, but the MAE training entry has not been implemented yet."
  echo "After adding trainers/train_mae.py, rerun: bash scripts/run_debug.sh"
  exit 2
fi

python trainers/train_mae.py \
  --config "${CONFIG}" \
  --debug \
  --max_steps 20
