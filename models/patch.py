"""Patch utilities and fixed 2D sinusoidal positional embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    """2D image to patch embedding using a strided convolution."""

    def __init__(self, img_size: int = 112, patch_size: int = 8, in_chans: int = 1, embed_dim: int = 384):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"Expected image size {self.img_size}x{self.img_size}, got {h}x{w}")
        return self.proj(x).flatten(2).transpose(1, 2)


def patchify(imgs: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """Convert images [B,C,H,W] into patch vectors [B,N,P*P*C]."""

    b, c, h, w = imgs.shape
    if h != w or h % patch_size != 0:
        raise ValueError("patchify expects square images divisible by patch_size")
    gh = gw = h // patch_size
    x = imgs.reshape(b, c, gh, patch_size, gw, patch_size)
    x = torch.einsum("bchpwq->bhwpqc", x)
    return x.reshape(b, gh * gw, patch_size * patch_size * c)


def unpatchify(patches: torch.Tensor, patch_size: int = 8, in_chans: int = 1) -> torch.Tensor:
    """Convert patch vectors [B,N,P*P*C] back to images [B,C,H,W]."""

    b, n, d = patches.shape
    grid = int(math.sqrt(n))
    if grid * grid != n:
        raise ValueError("Number of patches must be a square")
    expected = patch_size * patch_size * in_chans
    if d != expected:
        raise ValueError(f"Expected patch dim {expected}, got {d}")
    x = patches.reshape(b, grid, grid, patch_size, patch_size, in_chans)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(b, in_chans, grid * patch_size, grid * patch_size)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, device: torch.device | None = None) -> torch.Tensor:
    """Return [1, grid_size*grid_size, embed_dim] fixed sin-cos embedding."""

    if embed_dim % 4 != 0:
        raise ValueError("2D sin-cos embedding requires embed_dim divisible by 4")
    y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32, device=device),
        torch.arange(grid_size, dtype=torch.float32, device=device),
        indexing="ij",
    )
    omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 4)))
    out_x = x.reshape(-1, 1) * omega.reshape(1, -1)
    out_y = y.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
    return pos.unsqueeze(0)


def roi_to_patch_mask(roi_mask: torch.Tensor | None, img_size: int = 112, patch_size: int = 8) -> torch.Tensor | None:
    """Downsample pixel ROI mask [B,H,W] or [B,1,H,W] to patch-level bool [B,N]."""

    if roi_mask is None:
        return None
    if roi_mask.ndim == 4:
        roi_mask = roi_mask[:, 0]
    if roi_mask.ndim != 3:
        raise ValueError("roi_mask must have shape [B,H,W] or [B,1,H,W]")
    if roi_mask.shape[-2:] != (img_size, img_size):
        raise ValueError(f"roi_mask must be {img_size}x{img_size}")
    p = patch_size
    b = roi_mask.shape[0]
    grid = img_size // p
    x = roi_mask.float().reshape(b, grid, p, grid, p)
    x = x.mean(dim=(2, 4))
    return (x.reshape(b, grid * grid) > 0.05)
