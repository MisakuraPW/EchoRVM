#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/run_downstream_common.sh"
run_four_for_task "camus_seg" "configs/finetune_camus_seg.yaml" "$CAMUS_DATA_ROOT"

