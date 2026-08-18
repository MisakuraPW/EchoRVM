#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpt/mae/videomae_vit_s.pth}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
START_STAGE="${START_STAGE:-1}"

append_args() {
  local -n arr="$1"
  arr+=(--num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" --init_checkpoint "$INIT_CHECKPOINT")
  if [[ -n "${BATCH_SIZE:-}" ]]; then arr+=(--batch_size "$BATCH_SIZE"); fi
  if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then arr+=(--grad_accum_steps "$GRAD_ACCUM_STEPS"); fi
  if [[ -n "${EPOCHS:-}" ]]; then arr+=(--epochs "$EPOCHS"); fi
  if [[ -n "${LR:-}" ]]; then arr+=(--lr "$LR"); fi
  if [[ -n "${MAX_STEPS:-}" ]]; then arr+=(--max_steps "$MAX_STEPS"); fi
  if [[ -n "${CHECKPOINT_EPOCHS:-}" ]]; then arr+=(--checkpoint_epochs "$CHECKPOINT_EPOCHS"); fi
  if [[ -n "${SAVE_EVERY_N_EPOCHS:-}" ]]; then arr+=(--save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"); fi
  if [[ "${DEBUG:-0}" == "1" ]]; then arr+=(--debug); fi
}

run_one() {
  local stage="$1"
  local name="$2"
  local config="$3"
  local data_root="$4"
  local out_dir="${PRETRAIN_ROOT}/${name}/${RUN_TAG}"
  if (( stage < START_STAGE )); then
    echo "========== stage ${stage} skipped: ${name} =========="
    return
  fi
  local args=(--config "$config" --data_root "$data_root" --output_dir "$out_dir")
  append_args args
  echo "========== stage ${stage}: pretrain ${name} =========="
  echo "output_dir=${out_dir}"
  python trainers/train_rmae.py "${args[@]}"
}

run_one 1 "echonet_rvm_mae" "configs/pretrain/echonet_rvm_mae.yaml" "$ECHO_DATA_ROOT"
run_one 2 "echonet_ttt_mae" "configs/pretrain/echonet_ttt_mae.yaml" "$ECHO_DATA_ROOT"
run_one 3 "camus_rvm_mae" "configs/pretrain/camus_rvm_mae.yaml" "$CAMUS_DATA_ROOT"
run_one 4 "camus_ttt_mae" "configs/pretrain/camus_ttt_mae.yaml" "$CAMUS_DATA_ROOT"

echo "========== stage2 recurrent pretrain done =========="
echo "run_tag=${RUN_TAG}"
echo "pretrain_root=${PRETRAIN_ROOT}"
