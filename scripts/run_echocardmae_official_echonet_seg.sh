#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CKPT="${ECHOCARDMAE_CKPT:-ckpt/mae/EchoCardMAE.pt}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/outputs_downstream/echocardmae_official_echonet_seg}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/$RUN_TAG}"

python trainers/train_finetune.py \
  --task echonet_seg \
  --config configs/finetune_echonet_seg_echocardmae_official.yaml \
  --pretrained "$CKPT" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
