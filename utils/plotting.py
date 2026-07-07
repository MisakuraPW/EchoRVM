"""Plot training metrics."""

from __future__ import annotations

from pathlib import Path


def plot_loss_curves(metrics_csv: str | Path, save_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_csv = Path(metrics_csv)
    save_path = Path(save_path)
    if not metrics_csv.exists():
        return
    df = pd.read_csv(metrics_csv)
    if df.empty or "train_loss" not in df:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss", marker="o", markersize=2)
    if "val_loss" in df and df["val_loss"].notna().any():
        plt.plot(df["epoch"], df["val_loss"], label="val_loss", marker="o", markersize=2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
