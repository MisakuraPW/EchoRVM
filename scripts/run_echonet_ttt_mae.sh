#!/usr/bin/env bash
set -euo pipefail

python trainers/train_rmae.py --config configs/pretrain/echonet_ttt_mae.yaml "$@"
