#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python tools/run_temporal_research.py \
  --run_tag "${RUN_TAG:-screen_$(date +%Y%m%d_%H%M%S)}" \
  --only clip_mae_pool64 hier_global hier_spatial hier_dual \
  --audit_profile quick "$@"
