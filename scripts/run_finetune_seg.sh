#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-echonet_seg}"
CONFIG="${1:-configs/finetune_echonet_seg.yaml}"
PRETRAINED="${PRETRAINED:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -z "$PRETRAINED" ]]; then
  echo "[ERROR] PRETRAINED=/path/to/best.pt is required for this thin wrapper."
  echo "For the planned four-way runs use: bash scripts/run_downstream_echonet_seg.sh or bash scripts/run_downstream_camus_seg.sh"
  exit 2
fi

python trainers/train_finetune.py --task "$TASK" --config "$CONFIG" --pretrained "$PRETRAINED" "$@"
