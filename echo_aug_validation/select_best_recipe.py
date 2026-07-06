"""Apply simple rule-based selection for augmentation recipes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Select best augmentation recipe from proxy summary.")
    parser.add_argument("--summary", default="/root/autodl-tmp/outputs/aug_validation/results/aug_screening_summary.csv")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/aug_validation/results/best_recipe.yaml")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    score_rows = []
    for aug, sub in df.groupby("augmentation_id"):
        score = 0.0
        notes = []
        for _, row in sub.iterrows():
            model = row["model_type"]
            if model == "ef_temporal_proxy" and "mae" in row and pd.notna(row.get("mae")):
                score -= float(row["mae"])
                notes.append(f"ef_mae={row['mae']:.4f}")
            elif model == "unet_seg_proxy" and "dice_mean" in row and pd.notna(row.get("dice_mean")):
                score += 100.0 * float(row["dice_mean"])
                notes.append(f"dice={row['dice_mean']:.4f}")
            elif model == "small_mae" and pd.notna(row.get("val_loss")):
                score -= 10.0 * float(row["val_loss"])
                notes.append(f"mae_val_loss={row['val_loss']:.4f}")
        if "shadow" in str(aug).lower():
            score -= 5.0
        if "per_frame" in str(aug).lower():
            score -= 2.0
        score_rows.append({"augmentation_id": aug, "score": score, "notes": "; ".join(notes)})
    ranked = sorted(score_rows, key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        yaml.safe_dump({"selected": best, "ranked": ranked}, f, allow_unicode=True, sort_keys=False)
    print(f"selected={best['augmentation_id']} score={best['score']:.4f}")
    print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
