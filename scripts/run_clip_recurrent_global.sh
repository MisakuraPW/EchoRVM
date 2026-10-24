#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-all}"
if (($#)); then shift; fi
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/EchoNet-Dynamic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs_cr_clean}"
RUN_TAG="${RUN_TAG:-clip_recurrent_global_v1}"
CONFIG="configs/pipeline/clip_recurrent_global.yaml"
SSL_MANIFEST="$OUTPUT_ROOT/manifests/dynamic_all_ssl.json"
TASK_MANIFEST="$OUTPUT_ROOT/manifests/dynamic_train_val.json"
COMMON=(--config "$CONFIG" --root "$OUTPUT_ROOT" --tag "$RUN_TAG"
        --set "data.ssl_manifest=$SSL_MANIFEST" --set "data.manifest=$TASK_MANIFEST")

prepare() {
  if [[ ! -f "$SSL_MANIFEST" ]]; then
    "$PYTHON" -m research_pipeline manifest --scope all_ssl --data-root "$DATA_ROOT" --output "$SSL_MANIFEST"
  fi
  if [[ ! -f "$TASK_MANIFEST" ]]; then
    "$PYTHON" -m research_pipeline manifest --scope train_val --data-root "$DATA_ROOT" --output "$TASK_MANIFEST"
  fi
}

case "$ACTION" in
  prepare) prepare ;;
  plan) "$PYTHON" -m research_pipeline plan "${COMMON[@]}" "$@" ;;
  smoke) prepare; "$PYTHON" -m research_pipeline smoke "${COMMON[@]}" "$@" ;;
  all)
    prepare
    "$PYTHON" -m research_pipeline smoke "${COMMON[@]}" "$@"
    "$PYTHON" -m research_pipeline run "${COMMON[@]}" "$@"
    ;;
  run) prepare; "$PYTHON" -m research_pipeline run "${COMMON[@]}" "$@" ;;
  pretrain) prepare; "$PYTHON" -m research_pipeline run "${COMMON[@]}" --only pretrain "$@" ;;
  status) "$PYTHON" -m research_pipeline status --root "$OUTPUT_ROOT" --tag "$RUN_TAG" ;;
  export)
    "$PYTHON" -m research_pipeline export --root "$OUTPUT_ROOT" --tag "$RUN_TAG" \
      --destination "$OUTPUT_ROOT/${RUN_TAG}_results.zip" "$@"
    ;;
  *) printf 'Usage: bash %s [all|prepare|plan|smoke|run|pretrain|status|export] [pipeline flags]\n' "$0" >&2; exit 2 ;;
esac
