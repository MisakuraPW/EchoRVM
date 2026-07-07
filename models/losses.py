"""Loss functions for recurrent echo MAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patch_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on masked patches.

    pred/target: [B,T,N,D], mask: [B,T,N] bool where True means reconstruct.
    """

    loss = (pred - target).pow(2).mean(dim=-1)
    weight = mask.float()
    if patch_weight is not None:
        weight = weight * patch_weight.float()
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def state_smooth_loss(states: torch.Tensor) -> torch.Tensor:
    if states.shape[1] < 2:
        return states.new_tensor(0.0)
    return (states[:, 1:] - states[:, :-1]).pow(2).mean()


def temporal_infonce_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE for two clips/views from the same batch."""

    features_a = F.normalize(features_a, dim=-1)
    features_b = F.normalize(features_b, dim=-1)
    logits = features_a @ features_b.t() / temperature
    labels = torch.arange(features_a.shape[0], device=features_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
