"""Checkpoint save and load helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

import torch

from .seed import get_rng_state, set_rng_state


def checkpoint_epoch_name(epoch: int, width: int = 3) -> str:
    return f"epoch_{int(epoch):0{int(width)}d}.pt"


def configured_checkpoint_epochs(ckpt_cfg: dict | None) -> set[int]:
    """Parse fixed scientific-evaluation checkpoint epochs from config."""

    ckpt_cfg = ckpt_cfg or {}
    raw = (
        ckpt_cfg.get("save_epochs")
        or ckpt_cfg.get("fixed_epochs")
        or ckpt_cfg.get("eval_epochs")
        or ckpt_cfg.get("scientific_eval_epochs")
        or []
    )
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    out = set()
    for item in raw:
        try:
            epoch = int(item)
        except (TypeError, ValueError):
            continue
        if epoch > 0:
            out.add(epoch)
    return out


def checkpoint_directory(run_dir: str | Path, config: dict | None = None) -> Path:
    configured = (config or {}).get('checkpoint', {}).get('dir')
    return Path(configured) if configured else Path(run_dir) / 'checkpoints'


def should_save_last(config: dict, epoch: int, epochs: int, *, stopping: bool = False) -> bool:
    ckpt = config.get('checkpoint', {})
    interval = int(ckpt.get('save_last_every_n_epochs', 1))
    if interval < 1:
        raise ValueError('checkpoint.save_last_every_n_epochs must be positive')
    return bool(ckpt.get('save_last', True)) and (
        epoch == 1 or epoch == epochs or stopping or epoch % interval == 0
        or epoch in configured_checkpoint_epochs(ckpt))


def tensor_payload_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_payload_bytes(item) for item in value)
    return 0


def atomic_torch_save(obj: dict, path: str | Path, min_free_gb: float = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if min_free_gb > 0:
        # Keep the previous file until the temporary checkpoint is fully written.
        required = int(tensor_payload_bytes(obj) * 1.1) + 32 * 1024**2 + int(min_free_gb * 1024**3)
        free = shutil.disk_usage(path.parent).free
        if free < required:
            raise OSError(f'Insufficient checkpoint space at {path.parent}: free={free/1024**3:.2f} GiB, '
                          f'required~{required/1024**3:.2f} GiB including reserve. Previous checkpoint kept.')
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    config: dict,
    extra: dict | None = None,
) -> None:
    ckpt_cfg = config.get("checkpoint", {}) if isinstance(config, dict) else {}
    save_optimizer = bool(ckpt_cfg.get("save_optimizer", True))
    save_scheduler = bool(ckpt_cfg.get("save_scheduler", save_optimizer))
    save_scaler = bool(ckpt_cfg.get("save_scaler", save_optimizer))
    save_rng_state = bool(ckpt_cfg.get("save_rng_state", save_optimizer))
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if save_optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if save_scheduler and scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if save_scaler and scaler is not None else None,
        "best_metric": best_metric,
        "config": config,
        "rng_state": get_rng_state() if save_rng_state else None,
    }
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path, min_free_gb=float(ckpt_cfg.get('min_free_gb', 0)))


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if ckpt.get('partial_epoch') and optimizer is not None:
        raise ValueError('interrupt.pt contains a partial epoch, not an exact resume boundary. '
                         'Resume from last.pt; interrupted batch position is not recoverable.')
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    set_rng_state(ckpt.get("rng_state"))
    return ckpt


def find_last_checkpoint(run_dir: str | Path, config: dict | None = None) -> Path | None:
    path = checkpoint_directory(run_dir, config) / "last.pt"
    return path if path.exists() else None
