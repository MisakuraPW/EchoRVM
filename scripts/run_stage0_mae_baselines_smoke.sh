#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs_smoke}"
export RUN_TAG="${RUN_TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"
export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-2}"
export CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export DEBUG="${DEBUG:-1}"

bash scripts/run_stage0_mae_baselines_400.sh

echo "========== smoke done =========="
echo "run_tag=${RUN_TAG}"
