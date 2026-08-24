#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PRETRAIN_ROOT="${PRETRAIN_ROOT:-/root/autodl-tmp/outputs_representation_smoke}"
export AUDIT_ROOT="${AUDIT_ROOT:-/root/autodl-tmp/outputs_representation_audit_smoke}"
export REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/outputs_representation_reports_smoke}"
export RUN_TAG="${RUN_TAG:-rep_smoke_$(date +%Y%m%d_%H%M%S)}"
export EPOCHS=1
export MAX_STEPS=2
export CHECKPOINT_EPOCHS=1
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=1
export NUM_WORKERS=0
export FULL_EPOCHS=1
export SMOKE=1

bash scripts/run_mae_baseline_representation_400.sh
