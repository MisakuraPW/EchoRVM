#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mae_train.yaml}"
EXP_NAME="mae_train"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "${LOG_ROOT}"

LOG_FILE="${LOG_ROOT}/${EXP_NAME}_$(date +%F_%H-%M).log"

if [[ ! -f "trainers/train_mae.py" ]]; then
  echo "[ERROR] trainers/train_mae.py not found."
  echo "The AutoDL scaffold is ready, but the MAE training entry has not been implemented yet."
  echo "Expected command after implementation:"
  echo "  python trainers/train_mae.py --config ${CONFIG}"
  exit 2
fi

python trainers/train_mae.py \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
