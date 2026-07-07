"""Save EchoNet segmentation image/mask overlays for sanity checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from utils.downstream_datasets import EchoNetSegmentationDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview EchoNet VolumeTracings masks.")
    parser.add_argument("--data_root", default="/root/autodl-tmp/datasets/EchoNet-Dynamic")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/outputs_downstream/echonet_seg_preview")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=112)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = EchoNetSegmentationDataset(args.data_root, args.split, img_size=args.img_size, aug_cfg=None, limit=args.num_samples)
    ratios = []
    for idx in range(min(args.num_samples, len(dataset))):
        item = dataset[idx]
        image = item["image"][0].numpy()
        mask = item["mask"].numpy().astype(np.uint8)
        ratios.append(float(mask.mean()))
        img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        overlay = rgb.copy()
        overlay[mask > 0] = (0, 0, 255)
        blended = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
        stem = str(item["id"]).replace(":", "_").replace("/", "_").replace("\\", "_")
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_image.png"), img_u8)
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_mask.png"), mask * 255)
        cv2.imwrite(str(out_dir / f"{idx:03d}_{stem}_overlay.png"), blended)
    if ratios:
        print(f"saved={out_dir} samples={len(ratios)} mask_ratio_mean={np.mean(ratios):.4f} min={np.min(ratios):.4f} max={np.max(ratios):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
