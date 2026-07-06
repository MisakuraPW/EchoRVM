#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/finetune_seg.yaml}"
LOG_ROOT="/root/autodl-tmp/logs"
mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/finetune_seg_$(date +%F_%H-%M).log"

if [[ ! -f "trainers/train_finetune.py" ]]; then
  echo "[ERROR] trainers/train_finetune.py not found."
  echo "Add the downstream training entry before running segmentation fine-tuning."
  exit 2
fi

python trainers/train_finetune.py \
  --task seg \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
