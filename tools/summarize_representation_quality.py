"""Aggregate checkpoint-wise representation audits into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DISPLAY_COLUMNS = [
    "method",
    "checkpoint_epoch",
    "reconstruction_loss",
    "pretrain_val_loss",
    "parameters",
    "checkpoint_mb",
    "encoder_samples_per_second",
    "inference_peak_memory_gb",
    "effective_rank",
    "collapsed_dim_fraction",
    "augmentation_cosine",
    "adjacent_frame_cosine",
    "temporal_reverse_delta",
    "pixel_feature_dynamics_corr",
    "ef_ridge_001pct_mae",
    "ef_ridge_010pct_mae",
    "ef_ridge_100pct_mae",
    "ef_knn5_mae",
    "seg_linear_patch_dice",
    "probe_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--audit_root", default="/root/autodl-tmp/outputs_representation_audit")
    parser.add_argument("--pretrain_root", default="/root/autodl-tmp/outputs")
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--methods", default="echonet_echocardmae_repro echonet_videomae_clean")
    parser.add_argument("--checkpoint_epochs", default="50 100 150 200 250 300 350 400")
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def zscores(rows: list[dict[str, Any]], key: str, higher: bool) -> list[float]:
    values = [finite(row.get(key)) for row in rows]
    valid = [value for value in values if value is not None]
    if len(valid) < 2:
        return [0.0] * len(rows)
    mean = sum(valid) / len(valid)
    std = (sum((value - mean) ** 2 for value in valid) / len(valid)) ** 0.5
    if std < 1e-12:
        return [0.0] * len(rows)
    sign = 1.0 if higher else -1.0
    return [sign * (value - mean) / std if value is not None else 0.0 for value in values]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6g}"
    return "" if value is None else str(value)


def main() -> int:
    args = parse_args()
    audit_root = Path(args.audit_root) / args.run_tag
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    methods = args.methods.replace(",", " ").split()
    epochs = [int(x) for x in args.checkpoint_epochs.replace(",", " ").split()]
    rows = []
    pretrain_root = Path(args.pretrain_root)
    for method in methods:
        for epoch in epochs:
            path = audit_root / method / f"epoch_{epoch:04d}" / "representation_metrics.json"
            row: dict[str, Any] = {
                "run_tag": args.run_tag,
                "method": method,
                "checkpoint_epoch": epoch,
                "audit_exists": path.exists(),
                "audit_path": str(path),
            }
            if path.exists():
                row.update(json.loads(path.read_text(encoding="utf-8")))
            pretrain_csv = pretrain_root / method / args.run_tag / "logs" / "metrics.csv"
            if pretrain_csv.exists():
                with pretrain_csv.open("r", encoding="utf-8", newline="") as handle:
                    for item in csv.DictReader(handle):
                        try:
                            matches = int(float(item.get("epoch", -1))) == epoch
                        except (TypeError, ValueError):
                            matches = False
                        if matches:
                            for source, target in (
                                ("train_loss", "pretrain_train_loss"),
                                ("val_loss", "pretrain_val_loss"),
                                ("step_time", "pretrain_step_time"),
                                ("forward_time", "pretrain_forward_time"),
                                ("backward_time", "pretrain_backward_time"),
                            ):
                                value = finite(item.get(source))
                                if value is not None:
                                    row[target] = value
                            break
            rows.append(row)

    components = [
        zscores(rows, "seg_linear_patch_dice", True),
        zscores(rows, "ef_ridge_010pct_mae", False),
        zscores(rows, "ef_knn5_mae", False),
    ]
    for index, row in enumerate(rows):
        row["probe_score"] = sum(component[index] for component in components) / len(components)
        collapsed = finite(row.get("collapsed_dim_fraction"))
        row["collapse_warning"] = bool(collapsed is not None and collapsed > 0.05)

    fields = sorted({key for row in rows for key in row})
    csv_path = report_dir / "representation_quality.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MAE Representation Quality Audit",
        "",
        "The composite probe score is a within-report standardized summary of frozen",
        "10% EF ridge MAE, EF kNN MAE and patch-linear segmentation Dice. It is a",
        "screening score, not a replacement for final full fine-tuning.",
        "",
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(DISPLAY_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key in DISPLAY_COLUMNS) + " |")

    existing = [row for row in rows if row.get("audit_exists")]
    if existing:
        best = max(existing, key=lambda row: float(row.get("probe_score", -math.inf)))
        lines.extend(
            [
                "",
                "## Screening Result",
                "",
                f"Highest frozen-probe score: {best['method']} at epoch {best['checkpoint_epoch']} "
                f"(score={float(best['probe_score']):.4f}).",
                "",
                "Promote a candidate only when the advantage is consistent across checkpoints",
                "and is not accompanied by collapse warnings or loss of temporal sensitivity.",
            ]
        )
    md_path = report_dir / "representation_quality.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
