"""Train the Hiera-T single-frame echo MAE baseline."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hiera_echo.datasets import build_hiera_frame_dataset
from hiera_echo.models import build_echo_hiera_mae
from optim import build_optimizer
from utils.augmentation import AugmentedVideoDataset, EchoVideoAugmenter
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config, resolve_output_root, save_config
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.metrics_logger import MetricsLogger
from utils.plotting import plot_loss_curves
from utils.runtime import AverageMeter, gpu_memory_gb
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Hiera-T echo MAE baseline.")
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--init_checkpoint", default=None)
    p.add_argument("--hiera_repo", default=None)
    return p.parse_args()


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    out: list[str] = []

    def setv(section: str, key: str, value: Any) -> None:
        if value is None:
            return
        cfg.setdefault(section, {})[key] = value
        out.append(f"{section}.{key}={value}")

    setv("train", "batch_size", args.batch_size)
    setv("train", "epochs", args.epochs)
    setv("data", "num_workers", args.num_workers)
    setv("data", "prefetch_factor", args.prefetch_factor)
    setv("data", "data_root", args.data_root)
    setv("model", "img_size", args.img_size)
    setv("model", "init_checkpoint", args.init_checkpoint)
    setv("model", "hiera_repo", args.hiera_repo)
    if args.lr is not None:
        cfg.setdefault("optimizer", {})["lr"] = args.lr
        cfg.setdefault("optimizer", {})["adamw_lr"] = args.lr
        cfg.setdefault("optimizer", {})["muon_lr"] = args.lr
        out.append(f"optimizer.lr={args.lr}")
    if args.debug:
        cfg.setdefault("debug", {})["enabled"] = True
        out.append("debug.enabled=True")
    return out


def make_run_dir(cfg: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        run_dir = Path(output_dir)
    else:
        exp = cfg.get("experiment", {})
        run_dir = resolve_output_root(exp.get("output_root", "outputs")) / exp.get("name", "hiera_echo_mae") / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    for sub in ("logs", "checkpoints", "plots", "tensorboard", "samples"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


class AugmentedFrameDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augmenter):
        self.dataset = dataset
        self.augmenter = augmenter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        video = sample["image"][None]
        sample["image"] = self.augmenter(video)[0]
        sample["valid_mask"] = torch.ones_like(sample["image"])
        return sample


def build_loader(cfg: dict[str, Any], split: str, max_steps: int | None) -> DataLoader:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    dataset = build_hiera_frame_dataset(data_cfg, model_cfg, split, seed=int(cfg.get("experiment", {}).get("seed", 42)))
    if split == "train" and bool(cfg.get("augment", {}).get("enabled", False)):
        dataset = AugmentedFrameDataset(
            dataset,
            EchoVideoAugmenter(cfg.get("augment", {}), img_size=int(model_cfg.get("img_size", 192)), channels=1),
        )
    workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 16)),
        "shuffle": split == "train",
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        "drop_last": bool(data_cfg.get("drop_last", True)) if split == "train" else False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


def build_scheduler(optimizer, cfg: dict[str, Any], steps_per_epoch: int):
    sched_cfg = cfg.get("scheduler", {})
    if str(sched_cfg.get("name", "cosine")).lower() in {"none", "null"}:
        return None
    epochs = int(cfg.get("train", {}).get("epochs", 100))
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = int(sched_cfg.get("warmup_epochs", 10)) * max(1, steps_per_epoch)
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        min_factor = min(min_lr / max(base_lrs), 1.0)
        return min_factor + (1 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch: int, global_step: int, train: bool, max_steps: int | None):
    model.train(train)
    amp = bool(cfg.get("train", {}).get("mixed_precision", True)) and device.type == "cuda"
    clip_grad = cfg.get("train", {}).get("clip_grad_norm", None)
    loss_meter, data_meter, step_meter = AverageMeter(), AverageMeter(), AverageMeter()
    last = time.perf_counter()
    desc = f"{'train' if train else 'val'} hiera epoch {epoch}"
    iterator = tqdm(loader, desc=desc, leave=False, disable=not cfg.get("logging", {}).get("use_tqdm", True))
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(iterator, 1):
        data_meter.update(time.perf_counter() - last)
        batch = move_batch(batch, device)
        t0 = time.perf_counter()
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(batch["image"], batch.get("valid_mask"))
                loss = out["loss"]
            if train:
                scaler.scale(loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
        loss_meter.update(float(loss.detach().cpu()), int(batch["image"].shape[0]))
        step_meter.update(time.perf_counter() - t0)
        lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
        iterator.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{lr:.2e}", data=f"{data_meter.avg:.3f}", step=f"{step_meter.avg:.3f}", mem=f"{gpu_memory_gb()[0]:.1f}G")
        if max_steps is not None and step >= max_steps:
            break
        last = time.perf_counter()
    return {"loss": loss_meter.avg, "data_time": data_meter.avg, "step_time": step_meter.avg}, global_step


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = apply_overrides(cfg, args)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    run_dir = make_run_dir(cfg, args.output_dir)
    save_config(cfg, run_dir / "config.yaml")
    logger = setup_logger(run_dir / "logs" / "train.log")
    logger.info("run_dir=%s", run_dir)
    logger.info("command=trainers/train_hiera_mae.py --config %s", args.config)
    if overrides:
        logger.info("cli_overrides=%s", ", ".join(overrides))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_echo_hiera_mae(cfg.get("model", {})).to(device)
    if getattr(model, "init_report", None) is not None:
        report = model.init_report or {}
        logger.info(
            "hiera_init exact=%d interpolated=%s skipped_shape=%d missing=%d unexpected=%d",
            len(report.get("copied_exact", [])),
            report.get("interpolated", []),
            len(report.get("skipped_shape", [])),
            len(report.get("missing", [])),
            len(report.get("unexpected", [])),
        )

    optimizer, opt_stats = build_optimizer(model, cfg.get("optimizer", {"name": "adamw"}))
    logger.info("optimizer=%s muon_params=%d adam_params=%d", cfg.get("optimizer", {}).get("name", "adamw"), opt_stats.muon_params, opt_stats.adam_params)
    train_loader = build_loader(cfg, "train", args.max_steps)
    val_loader = build_loader(cfg, "val", args.max_steps)
    logger.info("train_loader samples=%d batches=%d", len(train_loader.dataset), len(train_loader))
    logger.info("val_loader samples=%d batches=%d", len(val_loader.dataset), len(val_loader))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("train", {}).get("mixed_precision", True)) and device.type == "cuda")

    start_epoch, global_step, best = 1, 0, None
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best = ckpt.get("best_metric")
        logger.info("resumed=%s start_epoch=%d global_step=%d best=%s", args.resume, start_epoch, global_step, best)

    metrics = MetricsLogger(run_dir / "logs")
    early_cfg = cfg.get("early_stopping", {})
    stopper = EarlyStopping(
        monitor=str(early_cfg.get("monitor", "val_loss")),
        mode=str(early_cfg.get("mode", "min")),
        patience=int(early_cfg.get("patience", 20)),
        min_delta=float(early_cfg.get("min_delta", 1e-4)),
    ) if bool(early_cfg.get("enabled", True)) else None

    try:
        for epoch in range(start_epoch, int(cfg.get("train", {}).get("epochs", 100)) + 1):
            if not args.eval_only:
                tr, global_step = run_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, True, args.max_steps)
                metrics.write_jsonl("train_metrics.jsonl", {"epoch": epoch, "global_step": global_step, **tr})
            val, global_step = run_epoch(model, val_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, False, args.max_steps)
            metrics.write_jsonl("val_metrics.jsonl", {"epoch": epoch, "global_step": global_step, **val})
            metrics.update_csv(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": tr["loss"] if not args.eval_only else float("nan"),
                    "val_loss": val["loss"],
                    "lr": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0,
                }
            )
            metric = float(val["loss"])
            improved = best is None or metric < float(best) - 1e-12
            if improved:
                best = metric
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, epoch, global_step, best, cfg)
            save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, global_step, best, cfg)
            if epoch % int(cfg.get("checkpoint", {}).get("save_every", 10)) == 0:
                save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, best, cfg)
            logger.info("epoch=%d train_loss=%.6f val_loss=%.6f best=%s", epoch, tr["loss"] if not args.eval_only else float("nan"), val["loss"], best)
            try:
                plot_loss_curves(run_dir / "logs" / "metrics.csv", run_dir / "plots" / "loss_latest.png")
            except Exception as exc:
                logger.warning("plot failed: %s", exc)
            if stopper is not None and stopper.step({"val_loss": metric}):
                logger.info("early stopping at epoch=%d best=%s", epoch, best)
                break
    except KeyboardInterrupt:
        save_checkpoint(run_dir / "checkpoints" / "interrupt.pt", model, optimizer, scheduler, scaler, epoch, global_step, best, cfg)
        logger.warning("interrupted; saved interrupt checkpoint")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
