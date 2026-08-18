#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/outputs_stage0_eval}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_stage0_eval_reports}"
RUN_TAG="${RUN_TAG:?set RUN_TAG to the pretraining run tag}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-50 100 150 200 250 300 350 400}"
METHODS="${METHODS:-echonet:echonet_echocardmae_repro echonet:echonet_videomae_clean camus:camus_echocardmae_repro camus:camus_videomae_clean}"
EVAL_MODE="${EVAL_MODE:-frozen}"
EVAL_EPOCHS="${EVAL_EPOCHS:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
START_INDEX="${START_INDEX:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"

if [[ -z "$EVAL_EPOCHS" ]]; then
  if [[ "$EVAL_MODE" == "full" ]]; then
    EVAL_EPOCHS=80
  else
    EVAL_EPOCHS=20
  fi
fi

checkpoint_path() {
  local method="$1"
  local epoch="$2"
  local ckpt="${PRETRAIN_ROOT}/${method}/${RUN_TAG}/checkpoints/epoch_$(printf "%04d" "$epoch").pt"
  if [[ -f "$ckpt" ]]; then
    echo "$ckpt"
    return
  fi
  ckpt="${PRETRAIN_ROOT}/${method}/${RUN_TAG}/checkpoints/epoch_$(printf "%03d" "$epoch").pt"
  if [[ -f "$ckpt" ]]; then
    echo "$ckpt"
    return
  fi
  echo ""
}

append_eval_args() {
  local -n arr="$1"
  arr+=(--num_workers "$NUM_WORKERS" --prefetch_factor "$PREFETCH_FACTOR")
  arr+=(--epochs "$EVAL_EPOCHS")
  if [[ -n "${BATCH_SIZE:-}" ]]; then arr+=(--batch_size "$BATCH_SIZE"); fi
  if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then arr+=(--grad_accum_steps "$GRAD_ACCUM_STEPS"); fi
  if [[ -n "${LR:-}" ]]; then arr+=(--lr "$LR"); fi
  if [[ -n "${MAX_STEPS:-}" ]]; then arr+=(--max_steps "$MAX_STEPS"); fi
  if [[ "$EVAL_MODE" == "frozen" ]]; then arr+=(--freeze_backbone); fi
}

run_eval_one() {
  local index="$1"
  local dataset="$2"
  local method="$3"
  local epoch="$4"
  local task config data_root
  if (( index < START_INDEX )); then
    echo "========== eval ${index} skipped: ${method} epoch ${epoch} =========="
    return
  fi
  if [[ "$dataset" == "echonet" ]]; then
    task="echonet_seg"
    config="configs/finetune_echonet_seg_stage0_112.yaml"
    data_root="$ECHO_DATA_ROOT"
  else
    task="camus_seg"
    config="configs/finetune_camus_seg_stage0_112.yaml"
    data_root="$CAMUS_DATA_ROOT"
  fi
  local ckpt
  ckpt="$(checkpoint_path "$method" "$epoch")"
  if [[ -z "$ckpt" ]]; then
    echo "[WARN] missing checkpoint: ${PRETRAIN_ROOT}/${method}/${RUN_TAG}/checkpoints/epoch_${epoch}.pt"
    if [[ "$SKIP_MISSING" == "1" ]]; then
      return
    fi
    exit 2
  fi
  local out_dir="${EVAL_ROOT}/${RUN_TAG}/${EVAL_MODE}/${method}/epoch_$(printf "%04d" "$epoch")"
  local args=(--task "$task" --config "$config" --pretrained "$ckpt" --data_root "$data_root" --output_dir "$out_dir")
  append_eval_args args
  echo "========== eval ${index}: ${EVAL_MODE} ${task} ${method} epoch ${epoch} =========="
  echo "pretrained=${ckpt}"
  echo "output_dir=${out_dir}"
  python trainers/train_finetune.py "${args[@]}"
}

idx=1
for spec in $METHODS; do
  dataset="${spec%%:*}"
  method="${spec#*:}"
  for epoch in $CHECKPOINT_EPOCHS; do
    run_eval_one "$idx" "$dataset" "$method" "$epoch"
    idx=$((idx + 1))
  done
done

python tools/summarize_stage0_checkpoint_eval.py \
  --run_tag "$RUN_TAG" \
  --pretrain_root "$PRETRAIN_ROOT" \
  --eval_root "$EVAL_ROOT" \
  --report_dir "${REPORT_ROOT}/${RUN_TAG}/${EVAL_MODE}" \
  --eval_mode "$EVAL_MODE" \
  --checkpoint_epochs "$CHECKPOINT_EPOCHS" \
  --methods "$METHODS"

echo "========== stage0 checkpoint eval done =========="
echo "report_dir=${REPORT_ROOT}/${RUN_TAG}/${EVAL_MODE}"
