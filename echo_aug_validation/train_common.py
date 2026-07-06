"""Shared training utilities."""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: str | Path, exp_name: str, augmentation_id: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / exp_name / augmentation_id / f"seed_{seed}_{stamp}"
    for sub in ["checkpoints", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, config_path: str | Path, cfg: dict[str, Any], augmentation_id: str, seed: int) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"augmentation_id": augmentation_id, "seed": seed, "config": cfg}, f, indent=2, ensure_ascii=False)


def device_from_config() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def build_dataloader(dataset, train_cfg: dict[str, Any], default_batch_size: int, shuffle: bool) -> DataLoader:
    workers = int(train_cfg.get("num_workers", 2))
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", default_batch_size)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metric": best_metric}, tmp)
    tmp.replace(path)
