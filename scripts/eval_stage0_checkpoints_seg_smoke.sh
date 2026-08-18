#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs_smoke}"
export EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/outputs_stage0_eval_smoke}"
export REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_stage0_eval_reports_smoke}"
export RUN_TAG="${RUN_TAG:?set RUN_TAG from scripts/run_stage0_mae_baselines_smoke.sh}"
export CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
export EVAL_MODE="${EVAL_MODE:-frozen}"
export EVAL_EPOCHS="${EVAL_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

bash scripts/eval_stage0_checkpoints_seg.sh
