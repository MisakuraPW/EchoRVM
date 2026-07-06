#!/usr/bin/env bash
set -euo pipefail

AUG="${1:-configs/aug_validation/augment/A7_clip_consistent.yaml}"
OUT="${OUT:-/root/autodl-tmp/outputs/aug_validation_debug}"

python -m echo_aug_validation.prepare_data --limit 5

python -m echo_aug_validation.train_small_mae \
  --config configs/aug_validation/small_mae.yaml \
  --augmentation_config "${AUG}" \
  --output_dir "${OUT}" \
  --debug \
  --max_steps 2

python -m echo_aug_validation.train_unet_seg_proxy \
  --config configs/aug_validation/unet_seg_proxy.yaml \
  --augmentation_config "${AUG}" \
  --output_dir "${OUT}" \
  --debug \
  --max_steps 2

python -m echo_aug_validation.train_ef_temporal_proxy \
  --config configs/aug_validation/ef_temporal_proxy.yaml \
  --augmentation_config "${AUG}" \
  --output_dir "${OUT}" \
  --debug \
  --max_steps 2

python -m echo_aug_validation.aggregate_results \
  --root "${OUT}" \
  --output-dir "${OUT}/results"
