#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ACTION="${1:-all}"
if (($#)); then shift; fi
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-fs/outputs_cr_spatial}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
case "$PRETRAIN_EPOCHS" in
  400|1600) ;;
  *) printf 'PRETRAIN_EPOCHS must be 400 or 1600\n' >&2; exit 2 ;;
esac
RUN_TAG="${RUN_TAG:-cr_patch8_w64_${PRETRAIN_EPOCHS}_v1}"
if [[ "$PRETRAIN_EPOCHS" == 1600 && "$RUN_TAG" == cr_patch8_w64_400_v1 ]]; then
  printf 'Use a separate RUN_TAG for the 1600-epoch run\n' >&2; exit 2
fi
INIT_CHECKPOINT="${INIT_CHECKPOINT:-ckpt/mae/videomae_vit_s.pth}"
INIT_CHECKPOINT="$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$INIT_CHECKPOINT")"
CONFIG="configs/pipeline/clip_recurrent_patch8.yaml"
SSL_MANIFEST="$OUTPUT_ROOT/manifests/dynamic_w64_h0_ssl.json"
TASK_MANIFEST="$OUTPUT_ROOT/manifests/dynamic_w64_h0_tasks.json"
COMMON=(--config "$CONFIG" --root "$OUTPUT_ROOT" --tag "$RUN_TAG"
  --set "data.ssl_manifest=$SSL_MANIFEST" --set "data.manifest=$TASK_MANIFEST"
  --set "model.init_checkpoint=$INIT_CHECKPOINT")
if [[ "$PRETRAIN_EPOCHS" == 1600 ]]; then
  COMMON+=(--set stages.pretrain.training.epochs=1600
    --set 'stages.pretrain.training.save_epochs=[0,100,200,400,800,1200,1600]')
  for stage in repr_0400 ef_linear_0400 seg_linear_0400; do
    COMMON+=(--set "stages.$stage.source='@pretrain/epoch_1600'"
      --set "stages.$stage.id=${stage%0400}1600")
  done
  for stage in seg_decoder_transfer_ft ef_l1_ft; do
    COMMON+=(--set "stages.$stage.source='@pretrain/epoch_1600'")
  done
fi
prepare() {
  if [[ ! -f "$INIT_CHECKPOINT" ]]; then
    printf 'Missing initial VideoMAE weights: %s\n' "$INIT_CHECKPOINT" >&2; exit 2
  fi
  if [[ ! -f "$SSL_MANIFEST" ]]; then
    "$PYTHON" -m research_pipeline manifest --scope all_ssl --data-root "$DATA_ROOT" \
      --recent-frames 64 --local-frames 16 --max-prefix 0 --output "$SSL_MANIFEST"
  fi
  if [[ ! -f "$TASK_MANIFEST" ]]; then
    "$PYTHON" -m research_pipeline manifest --scope train_val --data-root "$DATA_ROOT" \
      --recent-frames 64 --local-frames 16 --max-prefix 0 --output "$TASK_MANIFEST"
  fi
}
case "$ACTION" in
  prepare) prepare ;;
  plan) "$PYTHON" -m research_pipeline plan "${COMMON[@]}" "$@" ;;
  smoke) prepare; "$PYTHON" -m research_pipeline smoke "${COMMON[@]}" "$@" ;;
  all)
    prepare
    "$PYTHON" -m research_pipeline smoke "${COMMON[@]}" "$@"
    "$PYTHON" -u -m research_pipeline run "${COMMON[@]}" "$@"
    ;;
  run) prepare; "$PYTHON" -u -m research_pipeline run "${COMMON[@]}" "$@" ;;
  pretrain) prepare; "$PYTHON" -u -m research_pipeline run "${COMMON[@]}" --only pretrain "$@" ;;
  status) "$PYTHON" -m research_pipeline status --root "$OUTPUT_ROOT" --tag "$RUN_TAG" ;;
  export) "$PYTHON" -m research_pipeline export --root "$OUTPUT_ROOT" --tag "$RUN_TAG" \
    --destination "$OUTPUT_ROOT/${RUN_TAG}_results.zip" "$@" ;;
  *) printf 'Usage: %s [all|prepare|plan|smoke|run|pretrain|status|export] [pipeline flags]\n' "$0" >&2; exit 2 ;;
esac
