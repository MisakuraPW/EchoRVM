"""Train EchoRMAE / EchoTTT-MAE.

The current data contract is deliberately small:
    batch = {"video": Tensor[B, T, 1, 112, 112]}
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import build_echo_rmae
from optim import build_optimizer
from utils.augmentation import AugmentedVideoDataset, EchoVideoAugmenter, build_echo_augment_config
from utils.checkpoint import find_last_checkpoint, load_checkpoint, save_checkpoint
from utils.config import load_config, resolve_output_root, save_config
from utils.datasets import build_rmae_dataset
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.metrics_logger import MetricsLogger
from utils.plotting import plot_loss_curves
from utils.runtime import AverageMeter, gpu_memory_gb, now
from utils.seed import seed_everything


class SyntheticEchoVideoDataset(Dataset):
    """Small synthetic clip dataset for trainer smoke tests."""

    def __init__(self, length: int, frames: int, img_size: int, in_chans: int = 1):
        self.length = int(length)
        self.frames = int(frames)
        self.img_size = int(img_size)
        self.in_chans = int(in_chans)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator()
        gen.manual_seed(index)
        video = torch.rand(self.frames, self.in_chans, self.img_size, self.img_size, generator=gen)
        # Add a soft sector-like foreground so masking/ROI paths can be exercised later.
        return {"video": video}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recurrent echocardiography MAE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def make_run_dir(cfg: dict[str, Any], output_dir: str | None) -> Path:
    if output_dir:
        run_dir = Path(output_dir)
    else:
        exp = cfg.get("experiment", {})
        output_root = resolve_output_root(exp.get("output_root", "outputs"))
        stamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_dir = output_root / exp.get("name", "rmae") / stamp
    for sub in ["logs", "checkpoints", "plots", "tensorboard", "samples"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_loader(cfg: dict[str, Any], split: str, max_steps: int | None) -> DataLoader:
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    batch_size = int(train_cfg.get("batch_size", 2))
    frames = int(model_cfg.get("frames", 16))
    img_size = int(model_cfg.get("img_size", 112))
    in_chans = int(model_cfg.get("in_chans", 1))
    loader_name = str(data_cfg.get("loader", data_cfg.get("dataset_name", ""))).lower()
    debug_enabled = bool(cfg.get("debug", {}).get("enabled", False))
    use_synthetic = loader_name in {"synthetic", "debug", "smoke"} or debug_enabled
    if use_synthetic:
        length = int(data_cfg.get(f"synthetic_{split}_samples", max(8, batch_size * max(1, max_steps or 20))))
        dataset = SyntheticEchoVideoDataset(length=length, frames=frames, img_size=img_size, in_chans=in_chans)
    else:
        dataset = build_rmae_dataset(data_cfg, model_cfg, split, seed=int(cfg.get("experiment", {}).get("seed", 42)))
    augment_cfg = cfg.get("augment", {})
    if split == "train" and bool(augment_cfg.get("enabled", False)):
        dataset = AugmentedVideoDataset(dataset, EchoVideoAugmenter(augment_cfg, img_size=img_size, channels=in_chans))
    workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": split == "train",
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        "drop_last": bool(data_cfg.get("drop_last", False)) if split == "train" else False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], steps_per_epoch: int):
    sched_cfg = cfg.get("scheduler", {})
    train_cfg = cfg.get("train", {})
    name = str(sched_cfg.get("name", "cosine")).lower()
    if name in {"none", "null"}:
        return None
    epochs = int(train_cfg.get("epochs", 1))
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = max(0, int(sched_cfg.get("warmup_epochs", 0)) * max(1, steps_per_epoch))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        min_factor = min(min_lr / max(base_lrs), 1.0) if base_lrs else 0.0
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    cfg,
    epoch: int,
    global_step: int,
    train: bool,
    max_steps: int | None,
    logger,
):
    train_cfg = cfg.get("train", {})
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    clip_grad = train_cfg.get("clip_grad_norm", None)
    model.train(train)
    loss_meter = AverageMeter()
    data_meter = AverageMeter()
    step_meter = AverageMeter()
    fwd_meter = AverageMeter()
    bwd_meter = AverageMeter()
    last_time = time.perf_counter()
    iterator = tqdm(loader, desc=f"{'train' if train else 'val'} epoch {epoch}", leave=False, disable=not cfg.get("logging", {}).get("use_tqdm", True))
    if train:
        optimizer.zero_grad(set_to_none=True)
    expected_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    for step, batch in enumerate(iterator, start=1):
        if max_steps is not None and step > max_steps:
            break
        data_time = time.perf_counter() - last_time
        data_meter.update(data_time)
        batch = move_batch(batch, device)
        if step == 1:
            source = batch.get("dataset", "unknown")
            logger.info("%s first_batch video_shape=%s dataset=%s", "train" if train else "val", tuple(batch["video"].shape), source)
        start = now()
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(batch["video"])
                loss = out["loss"] / grad_accum
        if bool(train_cfg.get("stop_on_nan", True)) and not torch.isfinite(out["loss"]).all():
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}: {float(out['loss'].detach().cpu())}")
        fwd_time = now() - start
        fwd_meter.update(fwd_time)
        if train:
            bwd_start = now()
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == expected_steps:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
            bwd_meter.update(now() - bwd_start)
        true_loss = float(out["loss"].detach().cpu())
        loss_meter.update(true_loss, n=batch["video"].shape[0])
        mem_alloc, mem_reserved = gpu_memory_gb()
        step_time = time.perf_counter() - last_time
        step_meter.update(step_time)
        iterator.set_postfix(
            loss=f"{true_loss:.4f}",
            avg=f"{loss_meter.avg:.4f}",
            lr=f"{current_lr(optimizer):.2e}",
            mem=f"{mem_alloc:.1f}/{mem_reserved:.1f}G",
            data=f"{data_meter.avg:.3f}s",
            step=f"{step_meter.avg:.3f}s",
        )
        last_time = time.perf_counter()
    metrics = {
        "loss": loss_meter.avg,
        "data_time": data_meter.avg,
        "forward_time": fwd_meter.avg,
        "backward_time": bwd_meter.avg,
        "step_time": step_meter.avg,
        "lr": current_lr(optimizer),
    }
    logger.info(
        "%s epoch=%s loss=%.6f lr=%.3e data_time=%.4f forward_time=%.4f backward_time=%.4f step_time=%.4f",
        "train" if train else "val",
        epoch,
        metrics["loss"],
        metrics["lr"],
        metrics["data_time"],
        metrics["forward_time"],
        metrics["backward_time"],
        metrics["step_time"],
    )
    return metrics, global_step


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.debug:
        cfg.setdefault("debug", {})["enabled"] = True
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = args.max_steps
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    run_dir = make_run_dir(cfg, args.output_dir)
    shutil.copy2(args.config, run_dir / "config_source.yaml")
    save_config(cfg, run_dir / "config.yaml")
    logger = setup_logger(run_dir / "logs" / "train.log")
    logger.info("run_dir=%s", run_dir)
    logger.info("command=%s", " ".join(sys.argv))
    augment_cfg = cfg.get("augment", {})
    if bool(augment_cfg.get("enabled", False)):
        aug, per_frame_random, preset = build_echo_augment_config(augment_cfg, img_size=int(cfg.get("model", {}).get("img_size", 112)))
        logger.info(
            "augment enabled preset=%s per_frame_random=%s img_size=%d tgc=%.2f gamma_contrast=%.2f brightness=%.2f zoom=%.2f blur=%.2f speckle=%.2f shadow=%.2f",
            preset,
            per_frame_random,
            aug.img_size,
            aug.tgc_prob,
            aug.gamma_contrast_prob,
            aug.brightness_prob,
            aug.zoom_prob,
            aug.blur_prob,
            aug.speckle_prob,
            aug.shadow_prob,
        )
    else:
        logger.info("augment disabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(cfg.get("train", {}).get("matmul_precision", "high")))
    model = build_echo_rmae(cfg.get("model", {})).to(device)
    if bool(cfg.get("train", {}).get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer, opt_stats = build_optimizer(model, cfg.get("optimizer", {}))
    logger.info(
        "optimizer=%s muon_tensors=%d muon_params=%d adam_tensors=%d adam_params=%d",
        cfg.get("optimizer", {}).get("name", "muon_adamw_hybrid"),
        opt_stats.muon_tensors,
        opt_stats.muon_params,
        opt_stats.adam_tensors,
        opt_stats.adam_params,
    )
    max_steps = cfg.get("train", {}).get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)
    train_loader = build_loader(cfg, "train", max_steps)
    val_interval = cfg.get("train", {}).get("val_interval", None)
    val_loader = build_loader(cfg, "val", max_steps) if val_interval is not None else None
    logger.info("train_loader samples=%d batches=%d", len(train_loader.dataset), len(train_loader))
    if val_loader is not None:
        logger.info("val_loader samples=%d batches=%d", len(val_loader.dataset), len(val_loader))
    grad_accum = max(1, int(cfg.get("train", {}).get("grad_accum_steps", 1)))
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=optimizer_steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(cfg.get("train", {}).get("mixed_precision", True)))
    start_epoch = 1
    global_step = 0
    best_metric = None
    resume_path = args.resume or cfg.get("checkpoint", {}).get("resume")
    if not resume_path and cfg.get("checkpoint", {}).get("auto_resume", False):
        last = find_last_checkpoint(run_dir)
        resume_path = str(last) if last else None
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_metric = ckpt.get("best_metric")
        logger.info("resumed path=%s start_epoch=%d global_step=%d best_metric=%s", resume_path, start_epoch, global_step, best_metric)

    metrics_logger = MetricsLogger(run_dir / "logs")
    tb_writer = None
    if cfg.get("logging", {}).get("use_tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(run_dir / "tensorboard")
        except Exception as exc:
            logger.warning("TensorBoard disabled: %s", exc)
    ckpt_cfg = cfg.get("checkpoint", {})
    monitor = str(ckpt_cfg.get("monitor", "val_loss"))
    mode = str(ckpt_cfg.get("mode", "min"))
    early_cfg = cfg.get("early_stopping", {})
    stopper = EarlyStopping(
        patience=int(early_cfg.get("patience", 20)),
        min_delta=float(early_cfg.get("min_delta", 1e-4)),
        mode=mode,
        enabled=bool(early_cfg.get("enabled", val_loader is not None)),
    )
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    plot_interval = int(cfg.get("train", {}).get("plot_interval", 1))
    save_every = int(ckpt_cfg.get("save_every_n_epochs", 5))
    try:
        for epoch in range(start_epoch, epochs + 1):
            val_metrics = None
            if args.eval_only:
                if val_loader is None:
                    raise ValueError("--eval_only requires train.val_interval to create a validation loader")
                with torch.no_grad():
                    val_metrics, global_step = run_epoch(model, val_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, False, max_steps, logger)
                train_metrics = val_metrics
            else:
                train_metrics, global_step = run_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, True, max_steps, logger)
            if not args.eval_only and val_loader is not None and (epoch % int(val_interval) == 0):
                with torch.no_grad():
                    val_metrics, global_step = run_epoch(model, val_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, False, max_steps, logger)
            row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"] if val_metrics else None,
                "lr": train_metrics["lr"],
                "data_time": train_metrics["data_time"],
                "forward_time": train_metrics["forward_time"],
                "backward_time": train_metrics["backward_time"],
                "step_time": train_metrics["step_time"],
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            metrics_logger.write_jsonl("train_metrics.jsonl", row)
            if val_metrics:
                metrics_logger.write_jsonl("val_metrics.jsonl", {"epoch": epoch, **val_metrics})
            metrics_logger.update_csv(row)
            if tb_writer:
                tb_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                if val_metrics:
                    tb_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                tb_writer.add_scalar("LR", train_metrics["lr"], epoch)
                tb_writer.add_scalar("Runtime/data_time", train_metrics["data_time"], epoch)
                tb_writer.add_scalar("Runtime/step_time", train_metrics["step_time"], epoch)
            metric_value = val_metrics["loss"] if val_metrics else train_metrics["loss"]
            improved = False
            if best_metric is None:
                improved = True
            elif mode == "min":
                improved = metric_value < float(best_metric)
            else:
                improved = metric_value > float(best_metric)
            if improved:
                best_metric = metric_value
            if ckpt_cfg.get("save_last", True):
                save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
            if ckpt_cfg.get("save_best", True) and improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
            if cfg.get("logging", {}).get("save_plots", True) and epoch % plot_interval == 0:
                try:
                    plot_loss_curves(run_dir / "logs" / "metrics.csv", run_dir / "plots" / "loss_latest.png")
                    plot_loss_curves(run_dir / "logs" / "metrics.csv", run_dir / "plots" / f"loss_epoch_{epoch:03d}.png")
                except Exception as exc:
                    logger.warning("plot failed: %s", exc)
            if val_metrics and stopper.step(float(metric_value)):
                logger.info("early stopping at epoch=%d monitor=%s value=%.6f", epoch, monitor, metric_value)
                break
            if args.eval_only:
                break
    except KeyboardInterrupt:
        logger.warning("interrupted, saving interrupt checkpoint")
        save_checkpoint(run_dir / "checkpoints" / "interrupt.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
        raise
    finally:
        summary = {
            "run_dir": str(run_dir),
            "global_step": global_step,
            "best_metric": best_metric,
            "best_metric_name": monitor,
            "last_checkpoint": str(run_dir / "checkpoints" / "last.pt"),
            "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "end_time": datetime.now().isoformat(timespec="seconds"),
        }
        metrics_logger.write_summary(summary)
        if tb_writer:
            tb_writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
