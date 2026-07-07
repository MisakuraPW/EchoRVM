"""Early stopping helper."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "min", enabled: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.enabled = enabled
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if not self.enabled:
            return False
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
