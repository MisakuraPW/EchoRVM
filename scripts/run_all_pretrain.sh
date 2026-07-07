#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_bs${BATCH_SIZE}}"

COMMON_ARGS=(
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
)

if [[ -n "${EPOCHS:-}" ]]; then
  COMMON_ARGS+=(--epochs "$EPOCHS")
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  COMMON_ARGS+=(--max_steps "$MAX_STEPS")
fi

run_one() {
  local name="$1"
  local config="$2"
  shift 2
  local out_dir="${OUTPUT_ROOT}/${name}/${RUN_TAG}"
  echo "========== ${name} =========="
  echo "config=${config}"
  echo "output_dir=${out_dir}"
  python trainers/train_rmae.py \
    --config "$config" \
    --output_dir "$out_dir" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_one "echonet_rvm_mae" "configs/pretrain/echonet_rvm_mae.yaml" "$@"
run_one "echonet_ttt_mae" "configs/pretrain/echonet_ttt_mae.yaml" "$@"
run_one "camus_rvm_mae" "configs/pretrain/camus_rvm_mae.yaml" "$@"
run_one "camus_ttt_mae" "configs/pretrain/camus_ttt_mae.yaml" "$@"
