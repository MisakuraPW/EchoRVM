#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_TAG="${RUN_TAG:-tsf_pilot_smoke_$(date +%Y%m%d_%H%M%S)}" \
  bash scripts/run_tsf_pilot.sh --smoke "$@"
