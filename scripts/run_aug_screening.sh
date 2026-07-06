#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/root/autodl-tmp/outputs/aug_validation}"
SEEDS="${SEEDS:-0 1 2}"
AUGS="${AUGS:-A0_no_aug A1_basic_photometric A2_basic_tgc A4_tgc_zoom_speckle A6_per_frame_random A7_clip_consistent}"

python -m echo_aug_validation.prepare_data

for seed in ${SEEDS}; do
  for aug in ${AUGS}; do
    AUG_CFG="configs/aug_validation/augment/${aug}.yaml"
    echo "===== small-MAE ${aug} seed=${seed} ====="
    python -m echo_aug_validation.train_small_mae \
      --config configs/aug_validation/small_mae.yaml \
      --augmentation_config "${AUG_CFG}" \
      --seed "${seed}" \
      --output_dir "${OUT}"

    echo "===== U-Net SEG ${aug} seed=${seed} ====="
    python -m echo_aug_validation.train_unet_seg_proxy \
      --config configs/aug_validation/unet_seg_proxy.yaml \
      --augmentation_config "${AUG_CFG}" \
      --seed "${seed}" \
      --output_dir "${OUT}"

    echo "===== EF proxy ${aug} seed=${seed} ====="
    python -m echo_aug_validation.train_ef_temporal_proxy \
      --config configs/aug_validation/ef_temporal_proxy.yaml \
      --augmentation_config "${AUG_CFG}" \
      --seed "${seed}" \
      --output_dir "${OUT}"
  done
done

python -m echo_aug_validation.aggregate_results --root "${OUT}" --output-dir "${OUT}/results"
python -m echo_aug_validation.select_best_recipe --summary "${OUT}/results/aug_screening_summary.csv" --output "${OUT}/results/best_recipe.yaml"
