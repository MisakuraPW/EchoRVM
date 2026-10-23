"""The only interface a new model/dataset adapter has to implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Experiment:
    model: torch.nn.Module
    train: Any
    val: Any
    task: str
    metadata: dict
    microbatch_safe: bool = False
    cacheable: bool = False
    last_metrics: dict = field(default_factory=dict)

    def train_mode(self):
        self.model.train()

    def loss(self, samples):
        """Return the mean per-sample scalar loss, not a sum across the batch."""
        raise NotImplementedError

    def predict(self, samples, boundaries=False):
        """Return one JSON-compatible observation per sample; EF is in percentage points."""
        raise NotImplementedError

    def evaluate_batch(self, samples, boundaries=False):
        return self.loss(samples), self.predict(samples, boundaries)

    def preview(self, samples, directory):
        return None

    def features(self, samples):
        """Frozen [B,...,D] features in stable sample order, independent of the task head."""
        raise NotImplementedError

    def set_checkpointing(self, enabled):
        if enabled:
            raise ValueError('Adapter does not implement gradient checkpointing')

    def calibration_samples(self, batch_size):
        return [self.train[0]] * batch_size

    def model_export(self):
        """Include native model config and weights for future adapters, not only this task head."""
        return {}

    def input_digests(self):
        """Content hashes of split manifests and scientific source files, excluding model weights."""
        return self.metadata


def collate_samples(samples):
    return samples


def worker_init(worker_id):
    import random
    import numpy as np
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
