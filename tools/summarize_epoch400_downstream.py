"""Summarize the epoch-400 full downstream anchor experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


TASKS = {
    "echonet_seg": ("max", "val_dice_global_mean", "val_dice_mean"),
    "echonet_ef": ("min", "val_mae", "val_rmse", "val_corr"),
    "camus_seg": ("max", "val_dice_global_mean", "val_dice_mean"),
}


def number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--root", default="/root/autodl-tmp/outputs_representation_full")
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--methods", default="echonet_echocardmae_repro echonet_videomae_clean")
    args = parser.parse_args()
    rows = []
    for method in args.methods.split():
        for task, (mode, *preferred) in TASKS.items():
            path = Path(args.root) / args.run_tag / method / task / "logs" / "metrics.csv"
            row: dict[str, Any] = {"method": method, "task": task, "metrics_path": str(path), "exists": path.exists()}
            if path.exists():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    history = list(csv.DictReader(handle))
                best_key = next((key for key in preferred if any(number(item.get(key)) is not None for item in history)), None)
                if best_key:
                    candidates = [item for item in history if number(item.get(best_key)) is not None]
                    best = (max if mode == "max" else min)(candidates, key=lambda item: number(item[best_key]))
                    row["best_epoch"] = int(float(best.get("epoch", 0)))
                    for key, value in best.items():
                        parsed = number(value)
                        if parsed is not None:
                            row[key] = parsed
                    row["selection_metric"] = best_key
            rows.append(row)

    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    csv_path = report / "epoch0400_full_downstream.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    columns = ["method", "task", "best_epoch", "val_dice_mean", "val_dice_global_mean", "val_mae", "val_rmse", "val_corr"]
    lines = [
        "# Epoch 400 Full Downstream Anchor",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |")
    md_path = report / "epoch0400_full_downstream.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
