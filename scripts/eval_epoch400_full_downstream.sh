#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
FULL_ROOT="${FULL_ROOT:-/root/autodl-tmp/outputs_representation_full}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_representation_reports}"
RUN_TAG="${RUN_TAG:?set RUN_TAG to the baseline pretraining tag}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
CAMUS_DATA_ROOT="${CAMUS_DATA_ROOT:-/root/autodl-fs/datasets/CAMUS}"
METHODS="${METHODS:-echonet_echocardmae_repro echonet_videomae_clean}"
EPOCH="${EPOCH:-400}"
START_INDEX="${START_INDEX:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

checkpoint_path() {
  local method="$1"
  local path="${PRETRAIN_ROOT}/${method}/${RUN_TAG}/checkpoints/epoch_$(printf "%04d" "$EPOCH").pt"
  [[ -f "$path" ]] && echo "$path" || echo ""
}

run_task() {
  local index="$1"
  local method="$2"
  local task="$3"
  local config="$4"
  local root="$5"
  if (( index < START_INDEX )); then
    echo "========== full anchor ${index} skipped: ${method} ${task} =========="
    return
  fi
  local checkpoint
  checkpoint="$(checkpoint_path "$method")"
  if [[ -z "$checkpoint" ]]; then
    echo "[ERROR] missing epoch ${EPOCH} checkpoint for ${method}"
    exit 2
  fi
  local output_dir="${FULL_ROOT}/${RUN_TAG}/${method}/${task}"
  local args=(
    --task "$task"
    --config "$config"
    --pretrained "$checkpoint"
    --data_root "$root"
    --output_dir "$output_dir"
    --num_workers "$NUM_WORKERS"
    --prefetch_factor "$PREFETCH_FACTOR"
  )
  if [[ -n "${FULL_EPOCHS:-}" ]]; then args+=(--epochs "$FULL_EPOCHS"); fi
  if [[ -n "${MAX_STEPS:-}" ]]; then args+=(--max_steps "$MAX_STEPS"); fi
  echo "========== full anchor ${index}: ${method} ${task} =========="
  python trainers/train_finetune.py "${args[@]}"
}

index=1
for method in $METHODS; do
  run_task "$index" "$method" echonet_seg configs/finetune_echonet_seg_stage0_112.yaml "$ECHO_DATA_ROOT"
  index=$((index + 1))
  run_task "$index" "$method" echonet_ef configs/finetune_echonet_ef.yaml "$ECHO_DATA_ROOT"
  index=$((index + 1))
  if [[ -d "$CAMUS_DATA_ROOT" ]]; then
    run_task "$index" "$method" camus_seg configs/finetune_camus_seg_stage0_112.yaml "$CAMUS_DATA_ROOT"
  else
    echo "[WARN] CAMUS not found at $CAMUS_DATA_ROOT; CAMUS full anchor skipped"
  fi
  index=$((index + 1))
done

python tools/summarize_epoch400_downstream.py   --run_tag "$RUN_TAG"   --root "$FULL_ROOT"   --report_dir "${REPORT_ROOT}/${RUN_TAG}"   --methods "$METHODS"
