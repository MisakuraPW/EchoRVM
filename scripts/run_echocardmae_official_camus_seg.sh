#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
CKPT="${ECHOCARDMAE_CKPT:-ckpt/mae/EchoCardMAE.pt}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/outputs_downstream/echocardmae_official_camus_seg}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/$RUN_TAG}"

python trainers/train_finetune.py \
  --task camus_seg \
  --config configs/finetune_camus_seg_echocardmae_official.yaml \
  --pretrained "$CKPT" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
