#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/autodl-tmp/outputs_representation_audit}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_representation_reports}"
RUN_TAG="${RUN_TAG:?set RUN_TAG to the baseline pretraining tag}"
ECHO_DATA_ROOT="${ECHO_DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
METHODS="${METHODS:-echonet_echocardmae_repro echonet_videomae_clean}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-50 100 150 200 250 300 350 400}"
START_INDEX="${START_INDEX:-1}"
SKIP_MISSING="${SKIP_MISSING:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

checkpoint_path() {
  local method="$1"
  local epoch="$2"
  local path="${PRETRAIN_ROOT}/${method}/${RUN_TAG}/checkpoints/epoch_$(printf "%04d" "$epoch").pt"
  if [[ -f "$path" ]]; then
    echo "$path"
    return
  fi
  path="${PRETRAIN_ROOT}/${method}/${RUN_TAG}/checkpoints/epoch_$(printf "%03d" "$epoch").pt"
  [[ -f "$path" ]] && echo "$path" || echo ""
}

run_one() {
  local index="$1"
  local method="$2"
  local epoch="$3"
  if (( index < START_INDEX )); then
    echo "========== audit ${index} skipped: ${method} epoch ${epoch} =========="
    return
  fi
  local checkpoint
  checkpoint="$(checkpoint_path "$method" "$epoch")"
  if [[ -z "$checkpoint" ]]; then
    echo "[WARN] missing checkpoint: ${method} epoch ${epoch}"
    [[ "$SKIP_MISSING" == "1" ]] && return
    exit 2
  fi
  local output_dir="${AUDIT_ROOT}/${RUN_TAG}/${method}/epoch_$(printf "%04d" "$epoch")"
  local args=(
    --checkpoint "$checkpoint"
    --data_root "$ECHO_DATA_ROOT"
    --output_dir "$output_dir"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
  )
  if [[ -n "${RECON_SAMPLES:-}" ]]; then args+=(--recon_samples "$RECON_SAMPLES"); fi
  if [[ -n "${FEATURE_SAMPLES:-}" ]]; then args+=(--feature_samples "$FEATURE_SAMPLES"); fi
  if [[ -n "${EF_TRAIN_SAMPLES:-}" ]]; then args+=(--ef_train_samples "$EF_TRAIN_SAMPLES"); fi
  if [[ -n "${EF_VAL_SAMPLES:-}" ]]; then args+=(--ef_val_samples "$EF_VAL_SAMPLES"); fi
  if [[ -n "${SEG_TRAIN_SAMPLES:-}" ]]; then args+=(--seg_train_samples "$SEG_TRAIN_SAMPLES"); fi
  if [[ -n "${SEG_VAL_SAMPLES:-}" ]]; then args+=(--seg_val_samples "$SEG_VAL_SAMPLES"); fi
  if [[ -n "${SEG_STEPS:-}" ]]; then args+=(--seg_steps "$SEG_STEPS"); fi
  if [[ "${SMOKE:-0}" == "1" ]]; then args+=(--smoke); fi
  echo "========== audit ${index}: ${method} epoch ${epoch} =========="
  python tools/evaluate_representation_quality.py "${args[@]}"
}

index=1
for method in $METHODS; do
  for epoch in $CHECKPOINT_EPOCHS; do
    run_one "$index" "$method" "$epoch"
    index=$((index + 1))
  done
done

python tools/summarize_representation_quality.py   --run_tag "$RUN_TAG"   --audit_root "$AUDIT_ROOT"   --pretrain_root "$PRETRAIN_ROOT"   --report_dir "${REPORT_ROOT}/${RUN_TAG}"   --methods "$METHODS"   --checkpoint_epochs "$CHECKPOINT_EPOCHS"

echo "========== representation audit done =========="
echo "report=${REPORT_ROOT}/${RUN_TAG}/representation_quality.md"
