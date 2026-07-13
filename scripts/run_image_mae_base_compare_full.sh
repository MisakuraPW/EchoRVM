#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
DOWNSTREAM_ROOT="${DOWNSTREAM_ROOT:-/root/autodl-tmp/outputs_downstream}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_baseline_compare}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
IMAGE_INIT_CHECKPOINT="${IMAGE_INIT_CHECKPOINT:-ckpt/mae/mae_pretrain_vit_base.pth}"

PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-4}"
PRETRAIN_PREFETCH_FACTOR="${PRETRAIN_PREFETCH_FACTOR:-4}"
FINETUNE_NUM_WORKERS="${FINETUNE_NUM_WORKERS:-4}"
FINETUNE_PREFETCH_FACTOR="${FINETUNE_PREFETCH_FACTOR:-4}"
START_STAGE="${START_STAGE:-1}"

mkdir -p "${REPORT_ROOT}/${RUN_TAG}"
TIMES_CSV="${REPORT_ROOT}/${RUN_TAG}/stage_times.csv"
if [[ ! -f "$TIMES_CSV" ]]; then
  echo "stage,label,seconds,run_dir" > "$TIMES_CSV"
fi

append_image_pretrain_args() {
  local -n arr="$1"
  arr+=(--num_workers "$PRETRAIN_NUM_WORKERS" --prefetch_factor "$PRETRAIN_PREFETCH_FACTOR")
  arr+=(--init_checkpoint "$IMAGE_INIT_CHECKPOINT")
  if [[ -n "${IMAGE_PRETRAIN_BATCH_SIZE:-${PRETRAIN_BATCH_SIZE:-}}" ]]; then arr+=(--batch_size "${IMAGE_PRETRAIN_BATCH_SIZE:-${PRETRAIN_BATCH_SIZE}}"); fi
  if [[ -n "${PRETRAIN_EPOCHS:-}" ]]; then arr+=(--epochs "$PRETRAIN_EPOCHS"); fi
  if [[ -n "${IMAGE_PRETRAIN_LR:-${PRETRAIN_LR:-}}" ]]; then arr+=(--lr "${IMAGE_PRETRAIN_LR:-${PRETRAIN_LR}}"); fi
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
  echo "========== image-mae stage ${idx}: ${label} =========="
  echo "output_dir=${run_dir}"
  local start
  start="$(date +%s)"
  "$@"
  local end
  end="$(date +%s)"
  echo "${idx},${label},$((end - start)),${run_dir}" >> "$TIMES_CSV"
}

run_image_pretrain() {
  local dataset="$1"
  local config="$2"
  local data_root="$3"
  local stage_idx="$4"
  local name="${dataset}_image_mae_base"
  local out_dir="${PRETRAIN_ROOT}/${name}/${RUN_TAG}"
  local args=(--config "$config" --data_root "$data_root" --output_dir "$out_dir")
  append_image_pretrain_args args
  timed_stage "$stage_idx" "pretrain_${name}" "$out_dir" python trainers/train_rmae.py "${args[@]}"
}

run_seg_finetune() {
  local dataset="$1"
  local config="$2"
  local data_root="$3"
  local pretrain_name="$4"
  local stage_idx="$5"
  local task="${dataset}_seg"
  if [[ "$dataset" == "echo" ]]; then task="echonet_seg"; fi
  local ckpt="${PRETRAIN_ROOT}/${pretrain_name}/${RUN_TAG}/checkpoints/best.pt"
  local out_dir="${DOWNSTREAM_ROOT}/${task}/${RUN_TAG}/image_mae_base"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] checkpoint not found: ${ckpt}" >&2
    exit 2
  fi
  local args=(--task "$task" --config "$config" --pretrained "$ckpt" --data_root "$data_root" --output_dir "$out_dir")
  append_finetune_args args 32 2
  timed_stage "$stage_idx" "finetune_${task}_image_mae_base" "$out_dir" python trainers/train_finetune.py "${args[@]}"
}

run_image_pretrain "echonet" "configs/pretrain/echonet_image_mae_base.yaml" "$ECHO_DATA_ROOT" 1
run_seg_finetune "echo" "configs/finetune_echonet_seg_image_mae_base.yaml" "$ECHO_DATA_ROOT" "echonet_image_mae_base" 2
run_image_pretrain "camus" "configs/pretrain/camus_image_mae_base.yaml" "$CAMUS_DATA_ROOT" 3
run_seg_finetune "camus" "configs/finetune_camus_seg_image_mae_base.yaml" "$CAMUS_DATA_ROOT" "camus_image_mae_base" 4

python tools/summarize_mae_baseline_compare.py \
  --run_tag "$RUN_TAG" \
  --pretrain_root "$PRETRAIN_ROOT" \
  --downstream_root "$DOWNSTREAM_ROOT" \
  --report_dir "${REPORT_ROOT}/${RUN_TAG}" \
  --stage_times "$TIMES_CSV"

echo "========== done =========="
echo "run_tag=${RUN_TAG}"
echo "report_dir=${REPORT_ROOT}/${RUN_TAG}"
