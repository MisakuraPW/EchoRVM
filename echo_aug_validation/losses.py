"""Loss functions for proxy experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice[1:].mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.cross_entropy(logits, target) + dice_loss(logits, target, num_classes)


def mae_recon_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
