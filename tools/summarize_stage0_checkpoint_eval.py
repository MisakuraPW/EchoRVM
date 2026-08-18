"""Summarize Stage-0 checkpoint-wise pretrain and downstream evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


METHODS = (
    ("echonet", "echonet_echocardmae_repro", "echonet_seg"),
    ("echonet", "echonet_videomae_clean", "echonet_seg"),
    ("camus", "camus_echocardmae_repro", "camus_seg"),
    ("camus", "camus_videomae_clean", "camus_seg"),
)


def parse_epochs(raw: str) -> list[int]:
    return [int(item) for item in raw.replace(",", " ").split() if item.strip()]


def parse_methods(raw: str | None) -> list[tuple[str, str, str]]:
    if not raw:
        return list(METHODS)
    out = []
    for item in raw.split():
        dataset, method = item.split(":", 1) if ":" in item else ("echonet", item)
        task = "echonet_seg" if dataset == "echonet" else "camus_seg"
        out.append((dataset, method, task))
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def pretrain_epoch_metrics(run_dir: Path, epoch: int) -> dict[str, Any]:
    rows = read_csv_rows(run_dir / "logs" / "metrics.csv")
    for row in rows:
        if int(float(row.get("epoch", -1))) == int(epoch):
            return {
                "pretrain_train_loss": float_or_none(row.get("train_loss")),
                "pretrain_val_loss": float_or_none(row.get("val_loss")),
                "pretrain_lr": float_or_none(row.get("lr")),
                "pretrain_step_time": float_or_none(row.get("step_time")),
            }
    return {}


def downstream_best(run_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(run_dir / "logs" / "metrics.csv")
    if not rows:
        return {"eval_exists": False}
    monitor = rows[-1].get("monitor", "val_loss")
    mode = "min" if monitor in {"loss", "val_loss", "mae", "rmse"} else "max"

    def key(row: dict[str, str]) -> float:
        value = float_or_none(row.get("monitor_value"))
        if value is not None:
            return value
        fallback = float_or_none(row.get("val_loss"))
        return float("inf") if fallback is None else fallback

    best = min(rows, key=key) if mode == "min" else max(rows, key=key)
    val_dice_mean_values = [v for v in (float_or_none(r.get("val_dice_mean")) for r in rows) if v is not None]
    val_dice_global_values = [v for v in (float_or_none(r.get("val_dice_global_mean")) for r in rows) if v is not None]
    return {
        "eval_exists": True,
        "eval_epochs_ran": int(float(rows[-1].get("epoch", len(rows)))),
        "eval_monitor": monitor,
        "eval_best_epoch": int(float(best.get("epoch", 0))),
        "eval_best_monitor": float_or_none(best.get("monitor_value")),
        "eval_best_val_dice_mean": max(val_dice_mean_values) if val_dice_mean_values else None,
        "eval_best_val_dice_global_mean": max(val_dice_global_values) if val_dice_global_values else None,
        "eval_last_val_dice_mean": float_or_none(rows[-1].get("val_dice_mean")),
        "eval_last_val_loss": float_or_none(rows[-1].get("val_loss")),
    }


def checkpoint_file(run_dir: Path, epoch: int) -> Path | None:
    for width in (4, 3):
        path = run_dir / "checkpoints" / f"epoch_{epoch:0{width}d}.pt"
        if path.exists():
            return path
    return None


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "dataset",
        "method",
        "checkpoint_epoch",
        "checkpoint_exists",
        "pretrain_train_loss",
        "pretrain_val_loss",
        "eval_mode",
        "eval_best_val_dice_mean",
        "eval_best_val_dice_global_mean",
        "eval_best_epoch",
        "eval_last_val_loss",
    ]
    lines = [
        "# Stage-0 Checkpoint Evaluation",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--pretrain_root", default="/root/autodl-tmp/outputs")
    parser.add_argument("--eval_root", default="/root/autodl-tmp/outputs_stage0_eval")
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--eval_mode", default="frozen")
    parser.add_argument("--checkpoint_epochs", default="50 100 150 200 250 300 350 400")
    parser.add_argument("--methods", default=None, help="Space-separated dataset:method specs.")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root)
    eval_root = Path(args.eval_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset, method, task in parse_methods(args.methods):
        pretrain_dir = pretrain_root / method / args.run_tag
        for epoch in parse_epochs(args.checkpoint_epochs):
            ckpt = checkpoint_file(pretrain_dir, epoch)
            eval_dir = eval_root / args.run_tag / args.eval_mode / method / f"epoch_{epoch:04d}"
            row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "task": task,
                "run_tag": args.run_tag,
                "checkpoint_epoch": epoch,
                "checkpoint_exists": ckpt is not None,
                "checkpoint_path": str(ckpt) if ckpt else "",
                "checkpoint_mb": round(ckpt.stat().st_size / (1024 * 1024), 2) if ckpt else None,
                "pretrain_run_dir": str(pretrain_dir),
                "eval_mode": args.eval_mode,
                "eval_run_dir": str(eval_dir),
            }
            row.update(pretrain_epoch_metrics(pretrain_dir, epoch))
            row.update(downstream_best(eval_dir))
            rows.append(row)

    csv_path = report_dir / "stage0_checkpoint_eval.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, report_dir / "stage0_checkpoint_eval.md")
    print(f"wrote {csv_path}")
    print(f"wrote {report_dir / 'stage0_checkpoint_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
