#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python tools/run_temporal_research.py \
  --suite tsf --run_tag "${RUN_TAG:-tsf_pilot_$(date +%Y%m%d_%H%M%S)}" \
  --epochs 100 --audit_epochs 0 100 --audit_profile quick --screen_seg_final \
  --input_protocol gray_repeat3 --batch_size 8 --grad_accum_steps 4 \
  --num_workers 8 --prefetch_factor 4 --audit_batch_size 8 \
  --save_last_every 10 --autotune "$@"
