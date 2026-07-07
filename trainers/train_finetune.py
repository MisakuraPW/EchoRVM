"""Fine-tune EchoRMAE checkpoints on EchoNet/CAMUS downstream tasks."""

from __future__ import annotations

import argparse
import math
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from echo_aug_validation.augment_recipes import load_recipe
from models.downstream import (
    EchoEFFineTuner,
    EchoSegFineTuner,
    ef_metrics,
    load_pretrained_rmae,
    segmentation_loss,
    segmentation_metrics,
)
from utils.checkpoint import find_last_checkpoint, load_checkpoint, save_checkpoint
from utils.config import load_config, resolve_output_root, save_config
from utils.downstream_datasets import CAMUSSegmentationDataset, EchoNetEFDataset, EchoNetSegmentationDataset
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger
from utils.metrics_logger import MetricsLogger
from utils.plotting import plot_loss_curves
from utils.runtime import AverageMeter, gpu_memory_gb, now
from utils.seed import seed_everything


TASK_ALIASES = {
    "echo_seg": "echonet_seg",
    "echonet-seg": "echonet_seg",
    "echonet_seg": "echonet_seg",
    "echo_ef": "echonet_ef",
    "echonet-ef": "echonet_ef",
    "echonet_ef": "echonet_ef",
    "camus-seg": "camus_seg",
    "camus_seg": "camus_seg",
    "seg": "seg",
    "ef": "echonet_ef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune EchoRMAE downstream heads.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_ALIASES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    return parser.parse_args()


def canonical_task(task: str, cfg: dict[str, Any]) -> str:
    task = TASK_ALIASES[task]
    if task != "seg":
        return task
    name = str(cfg.get("data", {}).get("dataset_name", "")).lower()
    if "camus" in name:
        return "camus_seg"
    return "echonet_seg"


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []

    def set_value(section: str, key: str, value: Any) -> None:
        if value is None:
            return
        cfg.setdefault(section, {})[key] = value
        overrides.append(f"{section}.{key}={value}")

    set_value("train", "batch_size", args.batch_size)
    set_value("train", "grad_accum_steps", args.grad_accum_steps)
    set_value("train", "epochs", args.epochs)
    set_value("data", "num_workers", args.num_workers)
    set_value("data", "prefetch_factor", args.prefetch_factor)
    set_value("model", "frames", args.frames)
    if args.pretrained is not None:
        cfg.setdefault("model", {})["backbone_checkpoint"] = args.pretrained
        overrides.append(f"model.backbone_checkpoint={args.pretrained}")
    if args.data_root is not None:
        cfg.setdefault("data", {})["data_root"] = args.data_root
        overrides.append(f"data.data_root={args.data_root}")
    if args.lr is not None:
        cfg.setdefault("optimizer", {})["lr"] = args.lr
        overrides.append(f"optimizer.lr={args.lr}")
    if args.weight_decay is not None:
        cfg.setdefault("optimizer", {})["weight_decay"] = args.weight_decay
        overrides.append(f"optimizer.weight_decay={args.weight_decay}")
    if args.freeze_backbone:
        cfg.setdefault("train", {})["freeze_backbone"] = True
        overrides.append("train.freeze_backbone=True")
    if args.debug:
        cfg.setdefault("data", {})["limit"] = int(cfg.get("debug", {}).get("limit", 24))
        cfg.setdefault("train", {})["epochs"] = min(int(cfg.get("train", {}).get("epochs", 2)), 2)
        overrides.append("debug.limit/epochs")
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = args.max_steps
        overrides.append(f"train.max_steps={args.max_steps}")
    return overrides


def make_run_dir(cfg: dict[str, Any], task: str, output_dir: str | None) -> Path:
    if output_dir:
        run_dir = Path(output_dir)
    else:
        exp = cfg.get("experiment", {})
        output_root = resolve_output_root(exp.get("output_root", "outputs_downstream"))
        stamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_dir = output_root / exp.get("name", task) / stamp
    for sub in ("logs", "checkpoints", "plots", "tensorboard"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def build_aug(cfg: dict[str, Any], train: bool):
    aug_cfg = cfg.get("augment", {})
    if not train or not bool(aug_cfg.get("enabled", False)):
        return None, False, "disabled"
    preset = aug_cfg.get("preset", "A4_tgc_zoom_speckle")
    recipe_id, aug, per_frame = load_recipe(preset, img_size=int(aug_cfg.get("img_size", cfg.get("model", {}).get("img_size", 112))))
    for key, value in aug_cfg.items():
        if hasattr(aug, key):
            object.__setattr__(aug, key, value)
    per_frame = bool(aug_cfg.get("per_frame_random", per_frame))
    return aug, per_frame, recipe_id


def build_dataset(cfg: dict[str, Any], task: str, split: str):
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    img_size = int(model_cfg.get("img_size", 112))
    limit = data_cfg.get("limit")
    limit = int(limit) if limit is not None else None
    aug, per_frame, _ = build_aug(cfg, train=split == "train")
    split_name = str(data_cfg.get(f"{split}_split", split))
    root = data_cfg.get("data_root")
    if not root:
        raise ValueError("data.data_root is required")
    if task == "echonet_ef":
        return EchoNetEFDataset(
            root,
            split_name,
            frames=int(model_cfg.get("frames", 32)),
            img_size=img_size,
            aug_cfg=aug,
            per_frame_random=per_frame,
            seed=seed,
            limit=limit,
        )
    if task == "echonet_seg":
        return EchoNetSegmentationDataset(root, split_name, img_size=img_size, aug_cfg=aug, seed=seed, limit=limit)
    if task == "camus_seg":
        return CAMUSSegmentationDataset(
            root,
            split_name,
            img_size=img_size,
            aug_cfg=aug,
            seed=seed,
            limit=limit,
            val_fraction=float(data_cfg.get("val_fraction", 0.15)),
        )
    raise ValueError(f"Unsupported task: {task}")


def build_loader(cfg: dict[str, Any], task: str, split: str) -> DataLoader:
    dataset = build_dataset(cfg, task, split)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    workers = int(data_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "shuffle": split == "train",
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", torch.cuda.is_available())),
        "drop_last": bool(data_cfg.get("drop_last", False)) if split == "train" else False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


def build_model(cfg: dict[str, Any], task: str, device: torch.device, logger) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    ckpt_path = model_cfg.get("backbone_checkpoint")
    if not ckpt_path:
        raise ValueError("model.backbone_checkpoint or --pretrained is required")
    rmae, loaded_model_cfg, report = load_pretrained_rmae(ckpt_path, fallback_model_cfg=model_cfg, map_location=device)
    logger.info(
        "loaded_pretrained=%s core_type=%s missing=%d unexpected=%d",
        ckpt_path,
        loaded_model_cfg.get("core_type", getattr(rmae, "core_type", "unknown")),
        len(report["missing"]),
        len(report["unexpected"]),
    )
    if task in {"echonet_seg", "camus_seg"}:
        model = EchoSegFineTuner(rmae, num_classes=int(model_cfg.get("num_classes", 2)), dropout=float(model_cfg.get("head_dropout", 0.1)))
    elif task == "echonet_ef":
        model = EchoEFFineTuner(
            rmae,
            hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
            dropout=float(model_cfg.get("head_dropout", 0.2)),
        )
    else:
        raise ValueError(task)
    if bool(cfg.get("train", {}).get("freeze_backbone", False)):
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
        logger.info("backbone frozen; training head only")
    return model.to(device)


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 5e-5))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    no_decay = []
    decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(key in name.lower() for key in ("bias", "norm", "pos_embed", "token")):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict[str, Any], steps_per_epoch: int):
    sched_cfg = cfg.get("scheduler", {})
    if str(sched_cfg.get("name", "cosine")).lower() in {"none", "null"}:
        return None
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, int(float(sched_cfg.get("warmup_epochs", 0)) * steps_per_epoch))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    base_lr = max(group["lr"] for group in optimizer.param_groups)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        min_factor = min(min_lr / base_lr, 1.0)
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_loss_and_metrics(model, batch, task: str, cfg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float], int, tuple[torch.Tensor, torch.Tensor] | None]:
    if task in {"echonet_seg", "camus_seg"}:
        num_classes = int(cfg.get("model", {}).get("num_classes", 2))
        logits = model(batch["image"])
        target = batch["mask"].clamp(0, num_classes - 1)
        loss = segmentation_loss(logits, target, num_classes, dice_weight=float(cfg.get("train", {}).get("dice_weight", 1.0)))
        metrics = segmentation_metrics(logits.detach(), target.detach(), num_classes)
        return loss, metrics, int(target.shape[0]), None
    pred = model(batch["video"])
    target = batch["target"].float()
    loss = F.smooth_l1_loss(pred, target, beta=float(cfg.get("train", {}).get("smooth_l1_beta", 1.0)))
    metrics = ef_metrics(pred.detach(), target.detach())
    return loss, metrics, int(target.shape[0]), (pred.detach().cpu(), target.detach().cpu())


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    cfg,
    task: str,
    epoch: int,
    global_step: int,
    train: bool,
    logger,
):
    train_cfg = cfg.get("train", {})
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    clip_grad = train_cfg.get("clip_grad_norm", 1.0)
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    model.train(train)
    loss_meter = AverageMeter()
    data_meter = AverageMeter()
    step_meter = AverageMeter()
    fwd_meter = AverageMeter()
    bwd_meter = AverageMeter()
    metric_sums: dict[str, float] = {}
    metric_weight = 0
    ef_preds = []
    ef_targets = []
    last_time = time.perf_counter()
    iterator = tqdm(loader, desc=f"{'train' if train else 'val'} {task} epoch {epoch}", leave=False, disable=not cfg.get("logging", {}).get("use_tqdm", True))
    if train:
        optimizer.zero_grad(set_to_none=True)
    expected = len(loader) if max_steps is None else min(len(loader), max_steps)
    for step, batch in enumerate(iterator, start=1):
        if max_steps is not None and step > max_steps:
            break
        data_meter.update(time.perf_counter() - last_time)
        batch = move_batch(batch, device)
        if step == 1:
            shape_key = "video" if "video" in batch else "image"
            logger.info("%s first_batch %s_shape=%s source=%s", "train" if train else "val", shape_key, tuple(batch[shape_key].shape), batch.get("source_path", ""))
        fwd_start = now()
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss_raw, metrics, batch_n, ef_pair = compute_loss_and_metrics(model, batch, task, cfg)
                loss = loss_raw / grad_accum
        if not torch.isfinite(loss_raw).all():
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}: {float(loss_raw.detach().cpu())}")
        fwd_meter.update(now() - fwd_start)
        if train:
            bwd_start = now()
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == expected:
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
        loss_meter.update(float(loss_raw.detach().cpu()), n=batch_n)
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_n
        metric_weight += batch_n
        if ef_pair is not None:
            ef_preds.append(ef_pair[0])
            ef_targets.append(ef_pair[1])
        mem_alloc, mem_reserved = gpu_memory_gb()
        step_meter.update(time.perf_counter() - last_time)
        main_metric = metrics.get("dice_mean", metrics.get("mae", 0.0))
        iterator.set_postfix(
            loss=f"{float(loss_raw.detach().cpu()):.4f}",
            avg=f"{loss_meter.avg:.4f}",
            metric=f"{main_metric:.4f}",
            lr=f"{current_lr(optimizer):.2e}",
            mem=f"{mem_alloc:.1f}/{mem_reserved:.1f}G",
            data=f"{data_meter.avg:.3f}s",
            step=f"{step_meter.avg:.3f}s",
        )
        last_time = time.perf_counter()
    avg_metrics = {key: value / max(1, metric_weight) for key, value in metric_sums.items()}
    if ef_preds:
        avg_metrics.update(ef_metrics(torch.cat(ef_preds), torch.cat(ef_targets)))
    row = {
        "loss": loss_meter.avg,
        **avg_metrics,
        "lr": current_lr(optimizer),
        "data_time": data_meter.avg,
        "forward_time": fwd_meter.avg,
        "backward_time": bwd_meter.avg,
        "step_time": step_meter.avg,
    }
    logger.info("%s epoch=%d task=%s loss=%.6f metrics=%s lr=%.3e", "train" if train else "val", epoch, task, row["loss"], avg_metrics, row["lr"])
    return row, global_step


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    task = canonical_task(args.task, cfg)
    cli_overrides = apply_cli_overrides(cfg, args)
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))
    run_dir = make_run_dir(cfg, task, args.output_dir)
    shutil.copy2(args.config, run_dir / "config_source.yaml")
    save_config(cfg, run_dir / "config.yaml")
    logger = setup_logger(run_dir / "logs" / "train.log")
    logger.info("run_dir=%s", run_dir)
    logger.info("task=%s command=%s", task, " ".join(sys.argv))
    if cli_overrides:
        logger.info("cli_overrides=%s", ", ".join(cli_overrides))
    aug, per_frame, recipe_id = build_aug(cfg, train=True)
    logger.info("augment=%s per_frame_random=%s", recipe_id if aug is not None else "disabled", per_frame)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(cfg.get("train", {}).get("matmul_precision", "high")))
    model = build_model(cfg, task, device, logger)
    if bool(cfg.get("train", {}).get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg)
    train_loader = build_loader(cfg, task, "train")
    val_loader = build_loader(cfg, task, "val")
    logger.info("train_loader samples=%d batches=%d", len(train_loader.dataset), len(train_loader))
    logger.info("val_loader samples=%d batches=%d", len(val_loader.dataset), len(val_loader))
    grad_accum = max(1, int(cfg.get("train", {}).get("grad_accum_steps", 1)))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, math.ceil(len(train_loader) / grad_accum)))
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

    monitor = str(cfg.get("checkpoint", {}).get("monitor", "dice_mean" if task.endswith("seg") else "mae"))
    mode = str(cfg.get("checkpoint", {}).get("mode", "max" if task.endswith("seg") else "min"))
    early_cfg = cfg.get("early_stopping", {})
    stopper = EarlyStopping(
        patience=int(early_cfg.get("patience", 20)),
        min_delta=float(early_cfg.get("min_delta", 1e-4)),
        mode=mode,
        enabled=bool(early_cfg.get("enabled", True)),
    )
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    plot_interval = int(cfg.get("train", {}).get("plot_interval", 1))
    save_every = int(cfg.get("checkpoint", {}).get("save_every_n_epochs", 5))
    try:
        for epoch in range(start_epoch, epochs + 1):
            if args.eval_only:
                with torch.no_grad():
                    val_metrics, global_step = run_epoch(model, val_loader, optimizer, scheduler, scaler, device, cfg, task, epoch, global_step, False, logger)
                train_metrics = val_metrics
            else:
                train_metrics, global_step = run_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, task, epoch, global_step, True, logger)
                with torch.no_grad():
                    val_metrics, global_step = run_epoch(model, val_loader, optimizer, scheduler, scaler, device, cfg, task, epoch, global_step, False, logger)
            metric_value = float(val_metrics.get(monitor, val_metrics["loss"]))
            improved = best_metric is None or (metric_value < float(best_metric) if mode == "min" else metric_value > float(best_metric))
            if improved:
                best_metric = metric_value
            row = {
                "epoch": epoch,
                "step": global_step,
                "task": task,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "monitor": monitor,
                "monitor_value": metric_value,
                "lr": train_metrics["lr"],
                "time": datetime.now().isoformat(timespec="seconds"),
                **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
                **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
            }
            metrics_logger.write_jsonl("train_metrics.jsonl", row)
            metrics_logger.write_jsonl("val_metrics.jsonl", {"epoch": epoch, **val_metrics})
            metrics_logger.update_csv(row)
            if tb_writer:
                tb_writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                tb_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                tb_writer.add_scalar(f"Monitor/{monitor}", metric_value, epoch)
                tb_writer.add_scalar("LR", train_metrics["lr"], epoch)
            if cfg.get("checkpoint", {}).get("save_last", True):
                save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
            if cfg.get("checkpoint", {}).get("save_best", True) and improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg)
            if cfg.get("logging", {}).get("save_plots", True) and epoch % plot_interval == 0:
                try:
                    plot_loss_curves(run_dir / "logs" / "metrics.csv", run_dir / "plots" / "loss_latest.png")
                except Exception as exc:
                    logger.warning("plot failed: %s", exc)
            if stopper.step(metric_value):
                logger.info("early stopping epoch=%d monitor=%s value=%.6f", epoch, monitor, metric_value)
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
            "task": task,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_metric_name": monitor,
            "best_checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "last_checkpoint": str(run_dir / "checkpoints" / "last.pt"),
            "end_time": datetime.now().isoformat(timespec="seconds"),
        }
        metrics_logger.write_summary(summary)
        if tb_writer:
            tb_writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
