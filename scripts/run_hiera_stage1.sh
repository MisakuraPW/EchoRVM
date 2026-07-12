#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Stage 1 baseline only: EchoNet Hiera-T MAE, then CAMUS Hiera-T MAE.
# Override with e.g.
#   ECHO_DATA_ROOT=/root/autodl-tmp/datasets/EchoNet-Dynamic \
#   CAMUS_DATA_ROOT=/root/autodl-tmp/datasets/CAMUS \
#   bash scripts/run_hiera_stage1.sh --batch_size 96 --num_workers 8

bash scripts/run_hiera_pretrain_echonet.sh "$@"
bash scripts/run_hiera_pretrain_camus.sh "$@"
