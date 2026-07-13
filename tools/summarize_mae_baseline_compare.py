"""Summarize Hiera-T vs VideoMAE single-frame baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from models.downstream import _checkpoint_model_state, is_hiera_checkpoint, load_pretrained_hiera_mae, load_pretrained_rmae
from trainers.train_finetune import build_model as build_finetune_model
from utils.config import load_config


class _PrintLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metrics_best(path: Path) -> dict[str, Any]:
    csv_path = path / "logs" / "metrics.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    out: dict[str, Any] = {
        "epochs_ran": int(df["epoch"].max()) if "epoch" in df else len(df),
        "last_train_loss": float(df.iloc[-1].get("train_loss", float("nan"))),
        "last_val_loss": float(df.iloc[-1].get("val_loss", float("nan"))),
    }
    if "monitor" in df.columns and "monitor_value" in df.columns:
        monitor = str(df.iloc[-1]["monitor"])
        mode = "min" if monitor in {"loss", "val_loss", "mae", "rmse"} else "max"
        idx = df["monitor_value"].idxmin() if mode == "min" else df["monitor_value"].idxmax()
        out.update({"monitor": monitor, "best_monitor": float(df.loc[idx, "monitor_value"]), "best_epoch": int(df.loc[idx, "epoch"])})
    elif "val_loss" in df.columns:
        idx = df["val_loss"].idxmin()
        out.update({"monitor": "val_loss", "best_monitor": float(df.loc[idx, "val_loss"]), "best_epoch": int(df.loc[idx, "epoch"])})
    for key in ("val_dice_mean", "val_dice_global_mean", "train_dice_mean", "train_dice_global_mean"):
        if key in df.columns:
            out[f"best_{key}"] = float(df[key].max())
            out[f"last_{key}"] = float(df.iloc[-1][key])
    return out


def _checkpoint_size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None


def _state_param_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = _checkpoint_model_state(ckpt)
        return int(sum(t.numel() for t in state.values() if torch.is_tensor(t)))
    except Exception:
        return None


@torch.no_grad()
def _bench(fn, device: torch.device, warmup: int = 3, iters: int = 10) -> float | None:
    try:
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return round((time.perf_counter() - start) * 1000.0 / iters, 3)
    except Exception:
        return None


def _benchmark_pretrain(ckpt: Path, device: torch.device) -> float | None:
    if not ckpt.exists():
        return None
    try:
        if is_hiera_checkpoint(ckpt):
            model, cfg, _ = load_pretrained_hiera_mae(ckpt, map_location="cpu")
            model = model.to(device).eval()
            img = torch.rand(1, 1, int(cfg.get("img_size", 192)), int(cfg.get("img_size", 192)), device=device)
            valid = torch.ones_like(img)
            return _bench(lambda: model(img, valid), device)
        model, cfg, _ = load_pretrained_rmae(ckpt, map_location="cpu")
        model = model.to(device).eval()
        frames = int(cfg.get("frames", 1))
        img_size = int(cfg.get("img_size", 112))
        video = torch.rand(1, frames, 1, img_size, img_size, device=device)
        return _bench(lambda: model(video), device)
    except Exception:
        return None


def _benchmark_finetune(run_dir: Path, task: str, device: torch.device) -> float | None:
    ckpt = run_dir / "checkpoints" / "best.pt"
    cfg_path = run_dir / "config.yaml"
    if not ckpt.exists() or not cfg_path.exists():
        return None
    try:
        cfg = load_config(cfg_path)
        model = build_finetune_model(cfg, task, device, _PrintLogger())
        state = _checkpoint_model_state(torch.load(ckpt, map_location=device, weights_only=False))
        model.load_state_dict(state, strict=False)
        model.eval()
        img_size = int(cfg.get("model", {}).get("img_size", 112))
        frames = int(cfg.get("model", {}).get("frames", 1))
        if task.endswith("seg") and frames > 1:
            video = torch.rand(1, frames, 1, img_size, img_size, device=device)
            target_index = torch.tensor([frames // 2], device=device)
            return _bench(lambda: model.forward_video(video, target_index=target_index), device)
        image = torch.rand(1, 1, img_size, img_size, device=device)
        return _bench(lambda: model(image), device)
    except Exception:
        return None


def _read_stage_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["label"]] = float(row["seconds"])
    return out


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Small dependency-free markdown table writer.

    ``pandas.DataFrame.to_markdown`` requires the optional ``tabulate`` package,
    which is not guaranteed to exist on the AutoDL image.  Keep the experiment
    summary self-contained instead of asking the training environment to install
    another dependency.
    """

    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines) + "\n"


def _row(kind: str, dataset: str, backbone: str, task: str, run_dir: Path, label: str, stage_times: dict[str, float], device: torch.device) -> dict[str, Any]:
    ckpt = run_dir / "checkpoints" / "best.pt"
    row = {
        "kind": kind,
        "dataset": dataset,
        "backbone": backbone,
        "task": task,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "best_ckpt_mb": _checkpoint_size_mb(ckpt),
        "param_count": _state_param_count(ckpt),
        "stage_seconds": stage_times.get(label),
    }
    row.update(_metrics_best(run_dir))
    if kind == "pretrain":
        row["inference_ms"] = _benchmark_pretrain(ckpt, device)
    else:
        row["inference_ms"] = _benchmark_finetune(run_dir, task, device)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--pretrain_root", default="/root/autodl-tmp/outputs")
    parser.add_argument("--downstream_root", default="/root/autodl-tmp/outputs_downstream")
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--stage_times", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pretrain_root = Path(args.pretrain_root)
    downstream_root = Path(args.downstream_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stage_times = _read_stage_times(Path(args.stage_times)) if args.stage_times else {}
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    specs = [
        ("pretrain", "echonet", "hiera_t", "mae", pretrain_root / "echonet_hiera_t_mae" / args.run_tag, "pretrain_echonet_hiera_t_mae"),
        ("downstream", "echonet", "hiera_t", "echonet_seg", downstream_root / "echonet_seg" / args.run_tag / "hiera_t", "finetune_echonet_seg_hiera_t"),
        ("pretrain", "echonet", "videomae_single_frame", "mae", pretrain_root / "echonet_videomae_single_frame" / args.run_tag, "pretrain_echonet_videomae_single_frame"),
        ("downstream", "echonet", "videomae_single_frame", "echonet_seg", downstream_root / "echonet_seg" / args.run_tag / "videomae_single_frame", "finetune_echonet_seg_videomae_single_frame"),
        ("pretrain", "echonet", "image_mae_base", "mae", pretrain_root / "echonet_image_mae_base" / args.run_tag, "pretrain_echonet_image_mae_base"),
        ("downstream", "echonet", "image_mae_base", "echonet_seg", downstream_root / "echonet_seg" / args.run_tag / "image_mae_base", "finetune_echonet_seg_image_mae_base"),
        ("pretrain", "camus", "hiera_t", "mae", pretrain_root / "camus_hiera_t_mae" / args.run_tag, "pretrain_camus_hiera_t_mae"),
        ("downstream", "camus", "hiera_t", "camus_seg", downstream_root / "camus_seg" / args.run_tag / "hiera_t", "finetune_camus_seg_hiera_t"),
        ("pretrain", "camus", "videomae_single_frame", "mae", pretrain_root / "camus_videomae_single_frame" / args.run_tag, "pretrain_camus_videomae_single_frame"),
        ("downstream", "camus", "videomae_single_frame", "camus_seg", downstream_root / "camus_seg" / args.run_tag / "videomae_single_frame", "finetune_camus_seg_videomae_single_frame"),
        ("pretrain", "camus", "image_mae_base", "mae", pretrain_root / "camus_image_mae_base" / args.run_tag, "pretrain_camus_image_mae_base"),
        ("downstream", "camus", "image_mae_base", "camus_seg", downstream_root / "camus_seg" / args.run_tag / "image_mae_base", "finetune_camus_seg_image_mae_base"),
    ]
    rows = [_row(*spec, stage_times=stage_times, device=device) for spec in specs]
    csv_path = report_dir / "comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md_path = report_dir / "comparison.md"
    df = pd.DataFrame(rows)
    cols = [
        "kind",
        "dataset",
        "backbone",
        "task",
        "epochs_ran",
        "best_monitor",
        "monitor",
        "last_val_loss",
        "best_val_dice_mean",
        "best_val_dice_global_mean",
        "stage_seconds",
        "inference_ms",
        "param_count",
        "best_ckpt_mb",
    ]
    cols = [c for c in cols if c in df.columns]
    md_path.write_text(_to_markdown_table(df[cols]), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
