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

append_args() {
  local -n arr="$1"
  arr+=(--num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR" --init_checkpoint "$INIT_CHECKPOINT")
  if [[ -n "${BATCH_SIZE:-}" ]]; then arr+=(--batch_size "$BATCH_SIZE"); fi
  if [[ -n "${EPOCHS:-}" ]]; then arr+=(--epochs "$EPOCHS"); fi
  if [[ -n "${LR:-}" ]]; then arr+=(--lr "$LR"); fi
  if [[ -n "${MAX_STEPS:-}" ]]; then arr+=(--max_steps "$MAX_STEPS"); fi
}

run_one() {
  local name="$1"
  local config="$2"
  local data_root="$3"
  local out_dir="${PRETRAIN_ROOT}/${name}/${RUN_TAG}"
  local args=(--config "$config" --data_root "$data_root" --output_dir "$out_dir")
  append_args args
  echo "========== pretrain ${name} =========="
  echo "output_dir=${out_dir}"
  python trainers/train_rmae.py "${args[@]}"
}

run_one "echonet_videomae_single_frame" "configs/pretrain/echonet_videomae_single_frame.yaml" "$ECHO_DATA_ROOT"
run_one "camus_videomae_single_frame" "configs/pretrain/camus_videomae_single_frame.yaml" "$CAMUS_DATA_ROOT"

echo "========== done =========="
echo "run_tag=${RUN_TAG}"
