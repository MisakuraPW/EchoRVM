#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_TAG="${RUN_TAG:-temporal_final_smoke_20261010}" bash scripts/run_temporal_final.sh --smoke "$@"
