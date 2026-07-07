"""RVM-style gated transformer recurrent core."""

from __future__ import annotations

import torch
from torch import nn

from .vit_blocks import CrossBlock


class RVMCore(nn.Module):
    """GRU-style recurrent state updated by transformer cross/self attention."""

    def __init__(self, dim: int = 384, num_heads: int = 6, depth: int = 2, mlp_ratio: float = 4.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.update_x = nn.Linear(dim, dim)
        self.update_s = nn.Linear(dim, dim)
        self.reset_x = nn.Linear(dim, dim)
        self.reset_s = nn.Linear(dim, dim)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.integration = nn.ModuleList([CrossBlock(dim, num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def init_state(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tokens)

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.init_state(tokens)
        update = torch.sigmoid(self.update_x(tokens) + self.update_s(state))
        reset = torch.sigmoid(self.reset_x(tokens) + self.reset_s(state))
        context = reset * state
        candidate = tokens
        for block in self.integration:
            candidate = block(candidate, context)
        candidate = self.norm(candidate)
        new_state = (1.0 - update) * state + update * candidate
        return new_state, new_state
