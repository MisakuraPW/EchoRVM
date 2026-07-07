"""Runtime helpers for training telemetry."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def gpu_memory_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def ensure_executable_hint(path: str | Path) -> None:
    if os.name != "nt":
        try:
            mode = Path(path).stat().st_mode
            Path(path).chmod(mode | 0o111)
        except OSError:
            pass
