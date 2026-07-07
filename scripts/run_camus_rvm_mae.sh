#!/usr/bin/env bash
set -euo pipefail

python trainers/train_rmae.py --config configs/pretrain/camus_rvm_mae.yaml "$@"
