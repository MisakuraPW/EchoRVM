"""Shared, visible-token-only frame expansion for stage-two MAE and downstreams."""

import torch
from torch import nn
from torch.nn import functional as F


class FrameExpansion(nn.Module):
    def __init__(self, dim, offsets=2):
        super().__init__()
        self.offsets = offsets
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(offsets)])
        for projection in self.projections:
            nn.init.eye_(projection.weight)
            nn.init.normal_(projection.bias, std=.001)

    def forward(self, tokens, valid=None):
        # Input [B,G,P,D]; concatenate offsets inside each temporal group.
        expanded = torch.stack([layer(tokens) for layer in self.projections], dim=2)
        return expanded.flatten(1, 2)


class FactorizedFrameExpansion(nn.Module):
    """Full-dimensional clip structure plus a low-rank temporal residual.

    The reference is per clip/spatial patch, not a fitted video-level Functa code.
    Only visible tokens contribute during masked reconstruction.
    """

    def __init__(self, dim, offsets=2, rank=16):
        super().__init__()
        if not 1 <= rank <= dim:
            raise ValueError('dynamic_rank must be between 1 and embed_dim')
        self.base = FrameExpansion(dim, offsets)
        # Extra parameters must not shift initialization of the shared decoder.
        with torch.random.fork_rng(devices=[]):
            self.coefficients = nn.Linear(dim, rank, bias=False)
            self.basis = nn.Parameter(torch.empty(dim, rank))
            nn.init.orthogonal_(self.basis)
        with torch.no_grad():
            self.coefficients.weight.copy_(self.basis.T)
        self.offsets = offsets

    def components(self, tokens, valid=None):
        expanded = self.base(tokens)
        weights = (torch.ones_like(tokens[..., 0]) if valid is None else valid.to(tokens))
        weights = weights.repeat_interleave(self.offsets, 1)[..., None]
        denominator = weights.sum(1, keepdim=True).clamp_min(1)
        average = (expanded * weights).sum(1, keepdim=True) / denominator
        coefficients = self.coefficients(expanded)
        average_coefficients = (coefficients * weights).sum(1, keepdim=True) / denominator
        # Export absolute shared coordinates: resetting their mean each clip would
        # manufacture jumps and discard between-clip phase information.
        reference = average - F.linear(average_coefficients, self.basis)
        dynamic = F.linear(coefficients, self.basis)
        return dict(features=(reference + dynamic) * weights,
                    reference=reference, coefficients=coefficients * weights,
                    dynamic=dynamic * weights)

    def forward(self, tokens, valid=None):
        return self.components(tokens, valid)['features']

    def orthogonal_loss(self):
        with torch.autocast(self.basis.device.type, enabled=False):
            gram = self.basis.float().T @ self.basis.float()
            return (gram - torch.eye(gram.shape[0], device=gram.device)).square().mean()


class FrameQueryExpansion(nn.Module):
    """Optional per-spatial-location queries over a local clip's tubelet tokens."""

    def __init__(self, dim, offsets=2, heads=6):
        super().__init__()
        self.base = FrameExpansion(dim, offsets)
        with torch.random.fork_rng(devices=[]):
            self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, valid=None):
        b, groups, patches, dim = tokens.shape
        queries = self.base(tokens).permute(0, 2, 1, 3).reshape(b * patches, -1, dim)
        context = tokens.permute(0, 2, 1, 3).reshape(b * patches, groups, dim)
        padding = None
        if valid is not None:
            padding = ~valid.permute(0, 2, 1).reshape(b * patches, groups).bool()
            empty = padding.all(1)
            # Hidden spatial columns are never decoded; keep attention numerically defined.
            padding = padding.clone()
            padding[empty, 0] = False
        read, _ = self.attention(self.norm(queries), self.norm(context), context,
                                 key_padding_mask=padding, need_weights=False)
        return (queries + read).reshape(b, patches, -1, dim).permute(0, 2, 1, 3)
