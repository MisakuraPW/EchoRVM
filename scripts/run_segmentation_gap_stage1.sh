#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
ACTION="${1:-plan}"
if (($#)); then shift; fi
case "$ACTION" in
  plan|smoke|run)
    "$PYTHON" -u tools/run_segmentation_gap_stage1.py --action "$ACTION" "$@"
    ;;
  all)
    "$PYTHON" -u tools/run_segmentation_gap_stage1.py --action smoke "$@"
    "$PYTHON" -u tools/run_segmentation_gap_stage1.py --action run "$@"
    ;;
  *) printf 'Usage: %s [plan|smoke|run|all] [options]\n' "$0" >&2; exit 2 ;;
esac
