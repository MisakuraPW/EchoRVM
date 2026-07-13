#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN_TAG="${RUN_TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs_smoke}"
export DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-/root/autodl-tmp/outputs_downstream_smoke}"
export REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_baseline_compare_smoke}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-1}"
export PRETRAIN_MAX_STEPS="${PRETRAIN_MAX_STEPS:-2}"
export FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:-2}"
export HIERA_PRETRAIN_BATCH_SIZE="${HIERA_PRETRAIN_BATCH_SIZE:-2}"
export VIDEO_PRETRAIN_BATCH_SIZE="${VIDEO_PRETRAIN_BATCH_SIZE:-2}"
export IMAGE_PRETRAIN_BATCH_SIZE="${IMAGE_PRETRAIN_BATCH_SIZE:-2}"
export FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-2}"
export FINETUNE_GRAD_ACCUM_STEPS="${FINETUNE_GRAD_ACCUM_STEPS:-1}"
export PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-0}"
export FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-0}"

bash scripts/run_mae_baseline_compare_full.sh "$@"
