#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-/root/autodl-tmp/outputs_downstream}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpt/mae/videomae_vit_s.pth}"

PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-4}"
PRETRAIN_PREFETCH_FACTOR="${PRETRAIN_PREFETCH_FACTOR:-4}"
FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-4}"
FINETUNE_PREFETCH_FACTOR="${FINETUNE_PREFETCH_FACTOR:-4}"

START_STAGE="${START_STAGE:-1}"

append_optional_pretrain_args() {
  local -n args_ref="$1"
  args_ref+=(--num_workers "$PRETRAIN_NUM_WORKERS" --prefetch_factor "$PRETRAIN_PREFETCH_FACTOR")
  args_ref+=(--init_checkpoint "$INIT_CHECKPOINT")
  if [[ -n "${PRETRAIN_BATCH_SIZE:-}" ]]; then args_ref+=(--batch_size "$PRETRAIN_BATCH_SIZE"); fi
  if [[ -n "${PRETRAIN_GRAD_ACCUM_STEPS:-}" ]]; then args_ref+=(--grad_accum_steps "$PRETRAIN_GRAD_ACCUM_STEPS"); fi
  if [[ -n "${PRETRAIN_EPOCHS:-}" ]]; then args_ref+=(--epochs "$PRETRAIN_EPOCHS"); fi
  if [[ -n "${PRETRAIN_LR:-}" ]]; then args_ref+=(--lr "$PRETRAIN_LR"); fi
  if [[ -n "${PRETRAIN_MAX_STEPS:-}" ]]; then args_ref+=(--max_steps "$PRETRAIN_MAX_STEPS"); fi
  if [[ -n "${PRETRAIN_FRAMES:-}" ]]; then args_ref+=(--frames "$PRETRAIN_FRAMES"); fi
}

append_optional_finetune_args() {
  local -n args_ref="$1"
  args_ref+=(--num_workers "$FINETUNE_NUM_WORKERS" --prefetch_factor "$FINETUNE_PREFETCH_FACTOR")
  if [[ -n "${FINETUNE_EPOCHS:-}" ]]; then args_ref+=(--epochs "$FINETUNE_EPOCHS"); fi
  if [[ -n "${FINETUNE_LR:-}" ]]; then args_ref+=(--lr "$FINETUNE_LR"); fi
  if [[ -n "${FINETUNE_MAX_STEPS:-}" ]]; then args_ref+=(--max_steps "$FINETUNE_MAX_STEPS"); fi
}

run_pretrain() {
  local name="$1"
  local config="$2"
  local data_root="$3"
  local out_dir="${PRETRAIN_ROOT}/${name}/${RUN_TAG}"
  local args=(--config "$config" --data_root "$data_root" --output_dir "$out_dir")
  append_optional_pretrain_args args
  echo "========== pretrain ${name} =========="
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "output_dir=${out_dir}"
  python trainers/train_rmae.py "${args[@]}"
}

run_finetune() {
  local task="$1"
  local config="$2"
  local data_root="$3"
  local pretrain_name="$4"
  local label="$5"
  local batch_size="$6"
  local grad_accum="$7"
  local frames="${8:-}"
  local ckpt="${PRETRAIN_ROOT}/${pretrain_name}/${RUN_TAG}/checkpoints/best.pt"
  local out_dir="${DOWNSTREAM_ROOT}/${task}/${RUN_TAG}/${label}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] checkpoint not found: ${ckpt}" >&2
    exit 2
  fi
  local args=(--task "$task" --config "$config" --pretrained "$ckpt" --data_root "$data_root" --output_dir "$out_dir")
  args+=(--batch_size "$batch_size" --grad_accum_steps "$grad_accum")
  if [[ -n "$frames" ]]; then args+=(--frames "$frames"); fi
  append_optional_finetune_args args
  echo "========== finetune ${task} | ${label} =========="
  echo "pretrained=${ckpt}"
  echo "output_dir=${out_dir}"
  python trainers/train_finetune.py "${args[@]}"
}

stage() {
  local idx="$1"
  local title="$2"
  shift 2
  if (( idx < START_STAGE )); then
    echo "========== stage ${idx} skipped: ${title} =========="
    return
  fi
  echo "========== stage ${idx}: ${title} =========="
  "$@"
}

stage 1 "EchoNet TTT-MAE pretrain" run_pretrain "echonet_ttt_mae" "configs/pretrain/echonet_ttt_mae.yaml" "$ECHO_DATA_ROOT"
stage 2 "EchoNet RVM-MAE pretrain" run_pretrain "echonet_rvm_mae" "configs/pretrain/echonet_rvm_mae.yaml" "$ECHO_DATA_ROOT"

stage 3 "EchoNet segmentation fine-tune from EchoNet checkpoints" run_finetune "echonet_seg" "configs/finetune_echonet_seg.yaml" "$ECHO_DATA_ROOT" "echonet_ttt_mae" "init_echonet_ttt" "16" "4" "8"
stage 4 "EchoNet segmentation fine-tune from EchoNet checkpoints" run_finetune "echonet_seg" "configs/finetune_echonet_seg.yaml" "$ECHO_DATA_ROOT" "echonet_rvm_mae" "init_echonet_rvm" "16" "4" "8"
stage 5 "EchoNet EF fine-tune from EchoNet checkpoints" run_finetune "echonet_ef" "configs/finetune_echonet_ef.yaml" "$ECHO_DATA_ROOT" "echonet_ttt_mae" "init_echonet_ttt" "4" "3" "32"
stage 6 "EchoNet EF fine-tune from EchoNet checkpoints" run_finetune "echonet_ef" "configs/finetune_echonet_ef.yaml" "$ECHO_DATA_ROOT" "echonet_rvm_mae" "init_echonet_rvm" "4" "3" "32"

stage 7 "CAMUS TTT-MAE pretrain" run_pretrain "camus_ttt_mae" "configs/pretrain/camus_ttt_mae.yaml" "$CAMUS_DATA_ROOT"
stage 8 "CAMUS RVM-MAE pretrain" run_pretrain "camus_rvm_mae" "configs/pretrain/camus_rvm_mae.yaml" "$CAMUS_DATA_ROOT"
stage 9 "CAMUS segmentation fine-tune from CAMUS checkpoints" run_finetune "camus_seg" "configs/finetune_camus_seg.yaml" "$CAMUS_DATA_ROOT" "camus_ttt_mae" "init_camus_ttt" "64" "1" ""
stage 10 "CAMUS segmentation fine-tune from CAMUS checkpoints" run_finetune "camus_seg" "configs/finetune_camus_seg.yaml" "$CAMUS_DATA_ROOT" "camus_rvm_mae" "init_camus_rvm" "64" "1" ""

echo "========== done =========="
echo "run_tag=${RUN_TAG}"
