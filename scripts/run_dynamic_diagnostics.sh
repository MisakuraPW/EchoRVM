#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/run_dynamic_refinement.sh --phase diagnose "$@"
