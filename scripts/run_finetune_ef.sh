#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/finetune_ef.yaml}"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/finetune_ef_$(date +%F_%H-%M).log"

if [[ ! -f "trainers/train_finetune.py" ]]; then
  echo "[ERROR] trainers/train_finetune.py not found."
  echo "Add the downstream training entry before running EF fine-tuning."
  exit 2
fi

python trainers/train_finetune.py \
  --task ef \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
