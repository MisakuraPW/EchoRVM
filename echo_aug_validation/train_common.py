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


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_metric": best_metric}, tmp)
    tmp.replace(path)
