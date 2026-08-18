#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG

RUN_STAGE0_BASELINES="${RUN_STAGE0_BASELINES:-1}"
RUN_STAGE2_RECURRENT="${RUN_STAGE2_RECURRENT:-1}"
RUN_CHECKPOINT_EVAL="${RUN_CHECKPOINT_EVAL:-1}"

if [[ "$RUN_STAGE0_BASELINES" == "1" ]]; then
  bash scripts/run_stage0_mae_baselines_400.sh
fi

if [[ "$RUN_STAGE2_RECURRENT" == "1" ]]; then
  bash scripts/run_stage2_recurrent_mae_400.sh
fi

if [[ "$RUN_CHECKPOINT_EVAL" == "1" ]]; then
  export METHODS="${METHODS:-echonet:echonet_echocardmae_repro echonet:echonet_videomae_clean camus:camus_echocardmae_repro camus:camus_videomae_clean echonet:echonet_rvm_mae echonet:echonet_ttt_mae camus:camus_rvm_mae camus:camus_ttt_mae}"
  bash scripts/eval_stage0_checkpoints_seg.sh
fi

echo "========== mae research 400 pipeline done =========="
echo "run_tag=${RUN_TAG}"
