#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-/root/autodl-tmp/outputs_downstream}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_baseline_compare}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
VIDEO_INIT_CHECKPOINT="${VIDEO_INIT_CHECKPOINT:-ckpt/mae/videomae_vit_s.pth}"
HIERA_INIT_CHECKPOINT="${HIERA_INIT_CHECKPOINT:-ckpt/mae/mae_hiera_tiny_224.pth}"
HIERA_REPO="${HIERA_REPO:-}"

PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-4}"
PRETRAIN_PREFETCH_FACTOR="${PRETRAIN_PREFETCH_FACTOR:-4}"
FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-4}"
FINETUNE_PREFETCH_FACTOR="${FINETUNE_PREFETCH_FACTOR:-4}"
START_STAGE="${START_STAGE:-1}"

mkdir -p "${REPORT_ROOT}/${RUN_TAG}"
TIMES_CSV="${REPORT_ROOT}/${RUN_TAG}/stage_times.csv"
echo "stage,label,seconds,run_dir" > "$TIMES_CSV"

append_hiera_pretrain_args() {
  local -n arr="$1"
  arr+=(--num_workers "$PRETRAIN_NUM_WORKERS" --prefetch_factor "$PRETRAIN_PREFETCH_FACTOR")
  arr+=(--init_checkpoint "$HIERA_INIT_CHECKPOINT")
  if [[ -n "$HIERA_REPO" ]]; then arr+=(--hiera_repo "$HIERA_REPO"); fi
  if [[ -n "${HIERA_PRETRAIN_BATCH_SIZE:-${PRETRAIN_BATCH_SIZE:-}}" ]]; then arr+=(--batch_size "${HIERA_PRETRAIN_BATCH_SIZE:-${PRETRAIN_BATCH_SIZE}}"); fi
  if [[ -n "${PRETRAIN_EPOCHS:-}" ]]; then arr+=(--epochs "$PRETRAIN_EPOCHS"); fi
  if [[ -n "${HIERA_PRETRAIN_LR:-${PRETRAIN_LR:-}}" ]]; then arr+=(--lr "${HIERA_PRETRAIN_LR:-${PRETRAIN_LR}}"); fi
  if [[ -n "${PRETRAIN_MAX_STEPS:-}" ]]; then arr+=(--max_steps "$PRETRAIN_MAX_STEPS"); fi
}

append_video_pretrain_args() {
  local -n arr="$1"
  arr+=(--num_workers "$PRETRAIN_NUM_WORKERS" --prefetch_factor "$PRETRAIN_PREFETCH_FACTOR")
  arr+=(--init_checkpoint "$VIDEO_INIT_CHECKPOINT")
  if [[ -n "${VIDEO_PRETRAIN_BATCH_SIZE:-${PRETRAIN_BATCH_SIZE:-}}" ]]; then arr+=(--batch_size "${VIDEO_PRETRAIN_BATCH_SIZE:-${PRETRAIN_BATCH_SIZE}}"); fi
  if [[ -n "${PRETRAIN_EPOCHS:-}" ]]; then arr+=(--epochs "$PRETRAIN_EPOCHS"); fi
  if [[ -n "${VIDEO_PRETRAIN_LR:-${PRETRAIN_LR:-}}" ]]; then arr+=(--lr "${VIDEO_PRETRAIN_LR:-${PRETRAIN_LR}}"); fi
  if [[ -n "${PRETRAIN_MAX_STEPS:-}" ]]; then arr+=(--max_steps "$PRETRAIN_MAX_STEPS"); fi
}

append_finetune_args() {
  local -n arr="$1"
  local default_batch="$2"
  local default_accum="$3"
  arr+=(--num_workers "$FINETUNE_NUM_WORKERS" --prefetch_factor "$FINETUNE_PREFETCH_FACTOR")
  arr+=(--batch_size "${FINETUNE_BATCH_SIZE:-$default_batch}" --grad_accum_steps "${FINETUNE_GRAD_ACCUM_STEPS:-$default_accum}")
  if [[ -n "${FINETUNE_EPOCHS:-}" ]]; then arr+=(--epochs "$FINETUNE_EPOCHS"); fi
  if [[ -n "${FINETUNE_LR:-}" ]]; then arr+=(--lr "$FINETUNE_LR"); fi
  if [[ -n "${FINETUNE_MAX_STEPS:-}" ]]; then arr+=(--max_steps "$FINETUNE_MAX_STEPS"); fi
}

timed_stage() {
  local idx="$1"
  local label="$2"
  local run_dir="$3"
  shift 3
  if (( idx < START_STAGE )); then
    echo "========== stage ${idx} skipped: ${label} =========="
    return
  fi
  echo "========== stage ${idx}: ${label} =========="
  echo "output_dir=${run_dir}"
  local start
  start="$(date +%s)"
  "$@"
  local end
  end="$(date +%s)"
  echo "${idx},${label},$((end - start)),${run_dir}" >> "$TIMES_CSV"
}

run_hiera_pretrain() {
  local dataset="$1"
  local config="$2"
  local data_root="$3"
  local name="${dataset}_hiera_t_mae"
  local out_dir="${PRETRAIN_ROOT}/${name}/${RUN_TAG}"
  local args=(--config "$config" --data_root "$data_root" --output_dir "$out_dir")
  append_hiera_pretrain_args args
  timed_stage "$4" "pretrain_${name}" "$out_dir" python trainers/train_hiera_mae.py "${args[@]}"
}

run_video_pretrain() {
  local dataset="$1"
  local config="$2"
  local data_root="$3"
  local name="${dataset}_videomae_single_frame"
  local out_dir="${PRETRAIN_ROOT}/${name}/${RUN_TAG}"
  local args=(--config "$config" --data_root "$data_root" --output_dir "$out_dir")
  append_video_pretrain_args args
  timed_stage "$4" "pretrain_${name}" "$out_dir" python trainers/train_rmae.py "${args[@]}"
}

run_seg_finetune() {
  local dataset="$1"
  local backbone="$2"
  local config="$3"
  local data_root="$4"
  local pretrain_name="$5"
  local stage_idx="$6"
  local default_batch="$7"
  local default_accum="$8"
  local task="${dataset}_seg"
  if [[ "$dataset" == "echo" ]]; then task="echonet_seg"; fi
  local ckpt="${PRETRAIN_ROOT}/${pretrain_name}/${RUN_TAG}/checkpoints/best.pt"
  local out_dir="${DOWNSTREAM_ROOT}/${task}/${RUN_TAG}/${backbone}"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] checkpoint not found: ${ckpt}" >&2
    exit 2
  fi
  local args=(--task "$task" --config "$config" --pretrained "$ckpt" --data_root "$data_root" --output_dir "$out_dir")
  append_finetune_args args "$default_batch" "$default_accum"
  timed_stage "$stage_idx" "finetune_${task}_${backbone}" "$out_dir" python trainers/train_finetune.py "${args[@]}"
}

run_hiera_pretrain "echonet" "configs/pretrain/echonet_hiera_t_mae.yaml" "$ECHO_DATA_ROOT" 1
run_seg_finetune "echo" "hiera_t" "configs/finetune_echonet_seg_hiera.yaml" "$ECHO_DATA_ROOT" "echonet_hiera_t_mae" 2 32 2
run_video_pretrain "echonet" "configs/pretrain/echonet_videomae_single_frame.yaml" "$ECHO_DATA_ROOT" 3
run_seg_finetune "echo" "videomae_single_frame" "configs/finetune_echonet_seg_single_frame.yaml" "$ECHO_DATA_ROOT" "echonet_videomae_single_frame" 4 32 2

run_hiera_pretrain "camus" "configs/pretrain/camus_hiera_t_mae.yaml" "$CAMUS_DATA_ROOT" 5
run_seg_finetune "camus" "hiera_t" "configs/finetune_camus_seg_hiera.yaml" "$CAMUS_DATA_ROOT" "camus_hiera_t_mae" 6 32 2
run_video_pretrain "camus" "configs/pretrain/camus_videomae_single_frame.yaml" "$CAMUS_DATA_ROOT" 7
run_seg_finetune "camus" "videomae_single_frame" "configs/finetune_camus_seg_single_frame.yaml" "$CAMUS_DATA_ROOT" "camus_videomae_single_frame" 8 32 2

python tools/summarize_mae_baseline_compare.py \
  --run_tag "$RUN_TAG" \
  --pretrain_root "$PRETRAIN_ROOT" \
  --downstream_root "$DOWNSTREAM_ROOT" \
  --report_dir "${REPORT_ROOT}/${RUN_TAG}" \
  --stage_times "$TIMES_CSV"

echo "========== done =========="
echo "run_tag=${RUN_TAG}"
echo "report_dir=${REPORT_ROOT}/${RUN_TAG}"
