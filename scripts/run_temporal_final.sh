#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS; do
  value="${!name:-4}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then value=4; fi
  export "$name=$value"
done
export PYTHONUNBUFFERED=1
python tools/run_temporal_final.py --run_tag "${RUN_TAG:-temporal_final_20261010}" "$@"
