#!/usr/bin/env bash
set -euo pipefail

START_STAGE="${START_STAGE:-1}"

if (( START_STAGE <= 1 )); then
  echo "===== Stage 1/3: EchoNet segmentation fine-tuning, four pretrained initializations ====="
  bash scripts/run_downstream_echonet_seg.sh
fi

if (( START_STAGE <= 2 )); then
  echo "===== Stage 2/3: EchoNet EF fine-tuning, four pretrained initializations ====="
  bash scripts/run_downstream_echonet_ef.sh
fi

if (( START_STAGE <= 3 )); then
  echo "===== Stage 3/3: CAMUS segmentation fine-tuning, four pretrained initializations ====="
  bash scripts/run_downstream_camus_seg.sh
fi
