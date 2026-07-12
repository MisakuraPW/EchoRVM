#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs_downstream}"
DOWNSTREAM_TAG="${DOWNSTREAM_TAG:-$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MAX_STEPS="${MAX_STEPS:-}"
FRAMES="${FRAMES:-}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
PRETRAIN_TAG="${PRETRAIN_TAG:-}"
START_INDEX="${START_INDEX:-1}"

find_best_ckpt() {
  local name="$1"
  local run_dir=""
  if [[ -n "$PRETRAIN_TAG" ]]; then
    run_dir="${PRETRAIN_ROOT}/${name}/${PRETRAIN_TAG}"
  else
    run_dir="$(find "${PRETRAIN_ROOT}/${name}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  fi
  local ckpt="${run_dir}/checkpoints/best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] best checkpoint not found for ${name}: ${ckpt}" >&2
    echo "[HINT] set PRETRAIN_ROOT=/root/autodl-tmp/outputs and optionally PRETRAIN_TAG=<your_run_tag>" >&2
    exit 2
  fi
  echo "$ckpt"
}

append_common_args() {
  local -n arr_ref="$1"
  arr_ref+=(--num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR")
  if [[ -n "$BATCH_SIZE" ]]; then arr_ref+=(--batch_size "$BATCH_SIZE"); fi
  if [[ -n "$GRAD_ACCUM_STEPS" ]]; then arr_ref+=(--grad_accum_steps "$GRAD_ACCUM_STEPS"); fi
  if [[ -n "$EPOCHS" ]]; then arr_ref+=(--epochs "$EPOCHS"); fi
  if [[ -n "$LR" ]]; then arr_ref+=(--lr "$LR"); fi
  if [[ -n "$WEIGHT_DECAY" ]]; then arr_ref+=(--weight_decay "$WEIGHT_DECAY"); fi
  if [[ -n "$MAX_STEPS" ]]; then arr_ref+=(--max_steps "$MAX_STEPS"); fi
  if [[ -n "$FRAMES" ]]; then arr_ref+=(--frames "$FRAMES"); fi
  if [[ "${FREEZE_BACKBONE:-0}" == "1" ]]; then arr_ref+=(--freeze_backbone); fi
}

run_downstream_one() {
  local task="$1"
  local config="$2"
  local data_root="$3"
  local pretrain_name="$4"
  local label="$5"
  local ckpt
  ckpt="$(find_best_ckpt "$pretrain_name")"
  local out_dir="${OUTPUT_ROOT}/${task}/${DOWNSTREAM_TAG}/${label}"
  local args=(--task "$task" --config "$config" --pretrained "$ckpt" --data_root "$data_root" --output_dir "$out_dir")
  append_common_args args
  echo "========== ${task} | ${label} =========="
  echo "pretrained=${ckpt}"
  echo "output_dir=${out_dir}"
  python trainers/train_finetune.py "${args[@]}"
}

run_four_for_task() {
  local task="$1"
  local config="$2"
  local data_root="$3"
  local names=("echonet_rvm_mae" "echonet_ttt_mae" "camus_rvm_mae" "camus_ttt_mae")
  local labels=("init_echonet_rvm" "init_echonet_ttt" "init_camus_rvm" "init_camus_ttt")
  local idx
  for idx in "${!names[@]}"; do
    local human_idx=$((idx + 1))
    if (( human_idx < START_INDEX )); then
      echo "========== ${task} | ${labels[$idx]} skipped by START_INDEX=${START_INDEX} =========="
      continue
    fi
    run_downstream_one "$task" "$config" "$data_root" "${names[$idx]}" "${labels[$idx]}"
  done
}
