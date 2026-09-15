"""Patch-local Haar analysis and lightweight scale-conditioned memory gates."""
from __future__ import annotations

import torch
from torch import nn


def haar2(x):
    """Orthonormal 2D Haar analysis; no support outside the supplied patch."""
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a+b+c+d)/2, (a+b-c-d)/2, (a-b+c-d)/2, (a-b-c+d)/2)


def patch_bands(patches, tubelet, size, channels):
    # Patchify stores time, row, column, channel in that order.
    x = patches.float().reshape(*patches.shape[:-1], tubelet, size, size, channels)
    x = x.movedim(-1, -3)
    low, *fine = haar2(x)
    coarse, *middle = haar2(low)
    return [coarse, *middle, *fine]


def band_descriptor(patches, tubelet, size, channels):
    bands = patch_bands(patches, tubelet, size, channels)
    energy = torch.stack([band.square().mean((-4, -3, -2, -1)) for band in bands], -1)
    return energy / energy.sum(-1, keepdim=True).clamp_min(1e-8)


def multiband_error(pred, target, tubelet, size, channels):
    predicted = patch_bands(pred, tubelet, size, channels)
    expected = patch_bands(target, tubelet, size, channels)
    # Equal subband means and L1 are deliberately not full-coefficient Parseval L2.
    # Normalize the Haar gain at each level using fixed, batch-independent scales.
    errors = [(a-b).abs().mean((-4, -3, -2, -1))/scale
              for a, b, scale in zip(predicted, expected, (4,4,4,4,2,2,2))]
    return torch.stack(errors, -1).mean(-1)


class FrequencyMemoryGates(nn.Module):
    """Same memory token count; frequency modulates read/write rather than adding an encoder."""
    def __init__(self, dim):
        super().__init__()
        self.history_norm = nn.LayerNorm(dim)
        self.read_band = nn.Linear(7, dim)
        self.read_history = nn.Linear(dim, dim, bias=False)
        self.write_band = nn.Linear(7, dim)
        self.write_history = nn.Linear(dim, dim, bias=False)
        for layer in (self.read_band, self.read_history, self.write_band, self.write_history):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.read_band.bias)
        nn.init.constant_(self.write_band.bias, 2.)

    def read(self, encoded, fused, descriptor, history):
        condition = self.read_band(descriptor.to(encoded))
        condition = condition + self.read_history(self.history_norm(history.to(encoded)).mean(1, keepdim=True))
        gate = 2 * condition.sigmoid()
        return encoded + gate * (fused - encoded), gate.detach().mean()

    def write(self, previous, proposed, descriptor):
        gate = (self.write_band(descriptor.to(proposed)) +
                self.write_history(self.history_norm(previous.to(proposed)))).sigmoid()
        return previous + gate * (proposed - previous), gate.detach().mean()
