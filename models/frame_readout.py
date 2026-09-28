"""Shared, visible-token-only frame expansion for stage-two MAE and downstreams."""

import torch
from torch import nn


class FrameExpansion(nn.Module):
    def __init__(self, dim, offsets=2):
        super().__init__()
        self.offsets = offsets
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(offsets)])
        for projection in self.projections:
            nn.init.eye_(projection.weight)
            nn.init.normal_(projection.bias, std=.001)

    def forward(self, tokens):
        # Input [B,G,P,D]; concatenate offsets inside each temporal group.
        expanded = torch.stack([layer(tokens) for layer in self.projections], dim=2)
        return expanded.flatten(1, 2)
