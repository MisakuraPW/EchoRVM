#!/usr/bin/env bash
set -euo pipefail

export ECHOCARDMAE_CKPT="${ECHOCARDMAE_CKPT:-ckpt/mae/EchoCardMAE.pt}"
export RUN_TAG="${RUN_TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"

bash scripts/run_echocardmae_official_echonet_seg.sh \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --max_steps 2
