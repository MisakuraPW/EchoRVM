#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-clip_recurrent_forward_v1}"
ACTION="${1:-all}"
if (($#)); then shift; fi
COMMON=(--tag "$RUN_TAG")
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  COMMON+=(--root "$OUTPUT_ROOT" --set "data.manifest=$OUTPUT_ROOT/manifests/dynamic_train_val.json"
           --set "data.ssl_manifest=$OUTPUT_ROOT/manifests/dynamic_all_ssl.json")
fi
case "$ACTION" in
  plan) "$PYTHON" -m research_pipeline.forward_suite "${COMMON[@]}" --plan-only "$@" ;;
  smoke)
    bash scripts/run_clip_recurrent_global.sh prepare
    "$PYTHON" -m research_pipeline.forward_suite "${COMMON[@]}" --smoke "$@"
    ;;
  all)
    bash scripts/run_clip_recurrent_global.sh prepare
    "$PYTHON" -m research_pipeline.forward_suite "${COMMON[@]}" --smoke "$@"
    "$PYTHON" -m research_pipeline.forward_suite "${COMMON[@]}" "$@"
    ;;
  run) "$PYTHON" -m research_pipeline.forward_suite "${COMMON[@]}" "$@" ;;
  *) printf 'Usage: %s [plan|smoke|all|run] [--with-w] [--endpoint EPOCH] [pipeline arguments]\n' "$0" >&2; exit 2 ;;
esac
