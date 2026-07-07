#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/run_downstream_common.sh"
run_four_for_task "echonet_seg" "configs/finetune_echonet_seg.yaml" "$ECHO_DATA_ROOT"

