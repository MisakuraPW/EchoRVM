#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
RUN_TAG="${RUN_TAG:-baseline400_$(date +%Y%m%d_%H%M%S)}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpt/mae/videomae_vit_s.pth}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
ECHOCARD_BATCH_SIZE="${ECHOCARD_BATCH_SIZE:-16}"
ECHOCARD_GRAD_ACCUM_STEPS="${ECHOCARD_GRAD_ACCUM_STEPS:-8}"
MATCHED_BATCH_SIZE="${MATCHED_BATCH_SIZE:-32}"
MATCHED_GRAD_ACCUM_STEPS="${MATCHED_GRAD_ACCUM_STEPS:-4}"
VIDEOMAE_BATCH_SIZE="${VIDEOMAE_BATCH_SIZE:-32}"
VIDEOMAE_GRAD_ACCUM_STEPS="${VIDEOMAE_GRAD_ACCUM_STEPS:-4}"
START_STAGE="${START_STAGE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_FULL_FINETUNE_400="${RUN_FULL_FINETUNE_400:-1}"

run_pretrain() {
  local stage="$1"
  local method="$2"
  local config="$3"
  local default_batch_size="$4"
  local default_grad_accum="$5"
  if (( stage < START_STAGE )); then
    echo "========== stage ${stage} skipped: ${method} =========="
    return
  fi
  local micro_batch="${BATCH_SIZE:-$default_batch_size}"
  local grad_accum="${GRAD_ACCUM_STEPS:-$default_grad_accum}"
  local output_dir="${PRETRAIN_ROOT}/${method}/${RUN_TAG}"
  local args=(
    --config "$config"
    --data_root "$ECHO_DATA_ROOT"
    --output_dir "$output_dir"
    --init_checkpoint "$INIT_CHECKPOINT"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
    --batch_size "$micro_batch"
    --grad_accum_steps "$grad_accum"
  )
  if [[ -n "${EPOCHS:-}" ]]; then args+=(--epochs "$EPOCHS"); fi
  if [[ -n "${MAX_STEPS:-}" ]]; then args+=(--max_steps "$MAX_STEPS"); fi
  if [[ -n "${CHECKPOINT_EPOCHS:-}" ]]; then args+=(--checkpoint_epochs "$CHECKPOINT_EPOCHS"); fi
  echo "========== stage ${stage}: ${method} =========="
  echo "output_dir=$output_dir micro_batch=$micro_batch grad_accum=$grad_accum effective_batch=$((micro_batch * grad_accum)) workers=$NUM_WORKERS"
  python trainers/train_rmae.py "${args[@]}"
}

run_pretrain 1 echonet_echocardmae_official_video configs/pretrain/stage0_echonet_echocardmae_400.yaml "$ECHOCARD_BATCH_SIZE" "$ECHOCARD_GRAD_ACCUM_STEPS"
run_pretrain 2 echonet_videomae_matched configs/pretrain/stage0_echonet_videomae_matched_400.yaml "$MATCHED_BATCH_SIZE" "$MATCHED_GRAD_ACCUM_STEPS"
run_pretrain 3 echonet_videomae_clean configs/pretrain/stage0_echonet_videomae_clean_400.yaml "$VIDEOMAE_BATCH_SIZE" "$VIDEOMAE_GRAD_ACCUM_STEPS"

if [[ "$RUN_AUDIT" == "1" ]]; then
  METHODS="echonet_echocardmae_official_video echonet_videomae_matched echonet_videomae_clean" RUN_TAG="$RUN_TAG" PRETRAIN_ROOT="$PRETRAIN_ROOT" ECHO_DATA_ROOT="$ECHO_DATA_ROOT" CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-50 100 150 200 250 300 350 400}" NUM_WORKERS="$NUM_WORKERS" bash scripts/eval_representation_checkpoints.sh
fi

if [[ "$RUN_FULL_FINETUNE_400" == "1" ]]; then
  METHODS="echonet_echocardmae_official_video echonet_videomae_matched echonet_videomae_clean" RUN_TAG="$RUN_TAG"   PRETRAIN_ROOT="$PRETRAIN_ROOT"   ECHO_DATA_ROOT="$ECHO_DATA_ROOT"   NUM_WORKERS="$NUM_WORKERS"   PREFETCH_FACTOR="$PREFETCH_FACTOR"   bash scripts/eval_epoch400_full_downstream.sh
fi

echo "========== baseline representation pipeline done =========="
echo "run_tag=$RUN_TAG"
