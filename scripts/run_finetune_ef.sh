#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/finetune_echonet_ef.yaml}"
PRETRAINED="${PRETRAINED:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ -z "$PRETRAINED" ]]; then
  echo "[ERROR] PRETRAINED=/path/to/best.pt is required for this thin wrapper."
  echo "For the planned four-way run use: bash scripts/run_downstream_echonet_ef.sh"
  exit 2
fi

python trainers/train_finetune.py --task echonet_ef --config "$CONFIG" --pretrained "$PRETRAINED" "$@"
