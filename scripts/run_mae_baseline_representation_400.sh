#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
RUN_TAG="${RUN_TAG:-baseline400_$(date +%Y%m%d_%H%M%S)}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpt/mae/videomae_vit_s.pth}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
START_STAGE="${START_STAGE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_FULL_FINETUNE_400="${RUN_FULL_FINETUNE_400:-1}"

run_pretrain() {
  local stage="$1"
  local method="$2"
  local config="$3"
  if (( stage < START_STAGE )); then
    echo "========== stage ${stage} skipped: ${method} =========="
    return
  fi
  local output_dir="${PRETRAIN_ROOT}/${method}/${RUN_TAG}"
  local args=(
    --config "$config"
    --data_root "$ECHO_DATA_ROOT"
    --output_dir "$output_dir"
    --init_checkpoint "$INIT_CHECKPOINT"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
  )
  if [[ -n "${BATCH_SIZE:-}" ]]; then args+=(--batch_size "$BATCH_SIZE"); fi
  if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then args+=(--grad_accum_steps "$GRAD_ACCUM_STEPS"); fi
  if [[ -n "${EPOCHS:-}" ]]; then args+=(--epochs "$EPOCHS"); fi
  if [[ -n "${MAX_STEPS:-}" ]]; then args+=(--max_steps "$MAX_STEPS"); fi
  if [[ -n "${CHECKPOINT_EPOCHS:-}" ]]; then args+=(--checkpoint_epochs "$CHECKPOINT_EPOCHS"); fi
  echo "========== stage ${stage}: ${method} =========="
  echo "output_dir=$output_dir"
  python trainers/train_rmae.py "${args[@]}"
}

run_pretrain 1 echonet_echocardmae_repro configs/pretrain/stage0_echonet_echocardmae_400.yaml
run_pretrain 2 echonet_videomae_clean configs/pretrain/stage0_echonet_videomae_clean_400.yaml

if [[ "$RUN_AUDIT" == "1" ]]; then
  RUN_TAG="$RUN_TAG"   PRETRAIN_ROOT="$PRETRAIN_ROOT"   ECHO_DATA_ROOT="$ECHO_DATA_ROOT"   CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-50 100 150 200 250 300 350 400}"   NUM_WORKERS="$NUM_WORKERS"   bash scripts/eval_representation_checkpoints.sh
fi

if [[ "$RUN_FULL_FINETUNE_400" == "1" ]]; then
  RUN_TAG="$RUN_TAG" \
  PRETRAIN_ROOT="$PRETRAIN_ROOT" \
  ECHO_DATA_ROOT="$ECHO_DATA_ROOT" \
  NUM_WORKERS="$NUM_WORKERS" \
  PREFETCH_FACTOR="$PREFETCH_FACTOR" \
  bash scripts/eval_epoch400_full_downstream.sh
fi

echo "========== baseline representation pipeline done =========="
echo "run_tag=$RUN_TAG"
