"""Aggregate augmentation screening metrics into comparison CSV/Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["augmentation_id", "model_type", "seed"]
    if not set(keys).issubset(df.columns):
        return df
    idx = df.groupby(keys)["epoch"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate augmentation validation metrics.")
    parser.add_argument("--root", default="/root/autodl-tmp/outputs/aug_validation")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/outputs/aug_validation/results")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("metrics.csv"))
    if not files:
        print(f"No metrics.csv files found under {root}")
        return 2
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["run_dir"] = str(path.parent.parent)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    latest = latest_rows(all_df)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "aug_screening_all_metrics.csv", index=False)
    latest.to_csv(out / "aug_screening_latest_metrics.csv", index=False)

    agg_spec = {"train_loss": ("train_loss", "mean"), "val_loss": ("val_loss", "mean"), "seeds": ("seed", "nunique")}
    if "dice_mean" in latest.columns:
        agg_spec["dice_mean"] = ("dice_mean", "mean")
    if "mae" in latest.columns:
        agg_spec["mae"] = ("mae", "mean")
    if "r2" in latest.columns:
        agg_spec["r2"] = ("r2", "mean")
    summary = latest.groupby(["augmentation_id", "model_type"]).agg(**agg_spec).reset_index()
    summary.to_csv(out / "aug_screening_summary.csv", index=False)
    with (out / "aug_screening_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Augmentation Screening Summary\n\n")
        headers = list(summary.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in summary.iterrows():
            f.write("| " + " | ".join(str(row.get(col, "")) for col in headers) + " |\n")
    print(f"Wrote aggregate results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
