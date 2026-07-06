"""Metrics and CSV logging for proxy experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch


def dice_per_class(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    out: dict[str, float] = {}
    for cls in range(1, num_classes):
        p = pred == cls
        t = target == cls
        denom = p.sum().item() + t.sum().item()
        out[f"dice_class_{cls}"] = 1.0 if denom == 0 else (2.0 * (p & t).sum().item()) / denom
    values = list(out.values())
    out["dice_mean"] = float(np.mean(values)) if values else 0.0
    return out


def ef_metrics(preds: list[float], targets: list[float]) -> dict[str, float]:
    p = np.asarray(preds, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = float(np.corrcoef(p, y)[0, 1]) if len(p) > 1 and np.std(p) > 0 and np.std(y) > 0 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
        "within_5": float(np.mean(np.abs(err) <= 5.0)),
        "within_10": float(np.mean(np.abs(err) <= 10.0)),
    }


class MetricsCSV:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, object]) -> None:
        exists = self.path.exists()
        fields = list(row.keys())
        if exists:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                fields = next(reader)
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
