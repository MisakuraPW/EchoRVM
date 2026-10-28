#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/pipeline/clip_recurrent_global.yaml}"
RUN_TAG="${RUN_TAG:-cr_forward_fast_v3}"
ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs_cr_clean}"
SEARCH="$ROOT/result/$RUN_TAG/batch_search"
ACTION="${1:-all}"
if (($#)); then shift; fi
if [[ "$ACTION" != all && "$ACTION" != tune && "$ACTION" != run ]]; then
  printf 'Usage: %s [all|tune|run] [forward-suite arguments]\n' "$0" >&2
  exit 2
fi
if [[ "$ACTION" == tune && -f "$ROOT/result/$RUN_TAG/plan.json" ]]; then
  printf 'Cannot change effective batch under an existing trajectory; use a new RUN_TAG.\n' >&2
  exit 2
fi
if [[ "$ACTION" != run && ! -f "$SEARCH/selection.json" ]]; then
  PYTHON="$PYTHON" bash scripts/run_clip_recurrent_global.sh prepare
  "$PYTHON" -u tools/tune_mae_batch.py --config "$CONFIG" --output "$SEARCH" \
    --batches 32 64 128 --workers 2 --threads 4
fi
if [[ "$ACTION" == tune ]]; then exit 0; fi
if [[ ! -f "$SEARCH/selection.json" ]]; then
  printf 'Missing authorized batch calibration: %s\n' "$SEARCH/selection.json" >&2
  exit 2
fi
BATCH="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["micro_batch"])' "$SEARCH/selection.json")"
HALF=$((BATCH/2))
if ((HALF<1)); then HALF=1; fi
printf 'SSL effective batch=%s; LR unchanged; B0 and A share this recipe.\n' "$BATCH"
PYTHON="$PYTHON" RUN_TAG="$RUN_TAG" OUTPUT_ROOT="$ROOT" \
  bash scripts/run_clip_recurrent_forward.sh run --config "$CONFIG" \
    --set "stages.pretrain.training.effective_batch=$BATCH" \
    --set "stages.pretrain.runtime.micro_batch=$BATCH" \
    --set "stages.pretrain.runtime.batch_candidates=[$HALF,$BATCH]" \
    --set 'stages.pretrain.runtime.checkpoint_candidates=[false]' \
    --set 'stages.pretrain.runtime.threads=4' --set 'stages.pretrain.runtime.thread_candidates=[4]' \
    --set 'stages.pretrain.runtime.worker_candidates=[0,2]' "$@"
