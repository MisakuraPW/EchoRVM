#!/usr/bin/env bash
set -euo pipefail

echo "===== Stage 1/3: EchoNet segmentation fine-tuning, four pretrained initializations ====="
bash scripts/run_downstream_echonet_seg.sh

echo "===== Stage 2/3: EchoNet EF fine-tuning, four pretrained initializations ====="
bash scripts/run_downstream_echonet_ef.sh

echo "===== Stage 3/3: CAMUS segmentation fine-tuning, four pretrained initializations ====="
bash scripts/run_downstream_camus_seg.sh
