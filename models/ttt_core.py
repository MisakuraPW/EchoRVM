"""Lightweight TTT-style temporal core for EchoRMAE.

This is intentionally small and dependency-free. It keeps the same public
interface as RVMCore while using a per-video fast state updated by a local
self-supervised prediction error.
"""

from __future__ import annotations

import torch
from torch import nn

from .vit_blocks import Block


class TTTCore(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        inner_lr: float = 0.25,
        inner_steps: int = 1,
    ):
        super().__init__()
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.fast_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def init_state(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tokens)

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.init_state(tokens)
        fast_state = state
        k = self.key(tokens)
        v = self.value(tokens)
        for _ in range(max(1, self.inner_steps)):
            pred = self.fast_proj(fast_state + k)
            error = v - pred
            fast_state = fast_state + self.inner_lr * error.detach()
        q = self.query(tokens)
        refined = q + fast_state
        for block in self.blocks:
            refined = block(refined)
        mix = torch.sigmoid(self.gate(torch.cat([tokens, fast_state], dim=-1)))
        new_state = (1.0 - mix) * state + mix * self.norm(refined)
        return new_state, new_state
