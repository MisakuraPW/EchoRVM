"""VideoMAE baseline with 3D tubelet embedding and tube masking."""

from __future__ import annotations

import math

import torch
from torch import nn

from .vit_blocks import Block


def _sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("1D sin-cos embedding dimension must be even")
    omega = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 2)))
    phase = positions.float().reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((phase.sin(), phase.cos()), dim=1)


def get_3d_sincos_pos_embed(dim: int, temporal: int, height: int, width: int) -> torch.Tensor:
    """Return fixed temporal-spatial embeddings in Conv3d flatten order."""
    temporal_dim = dim // 4
    temporal_dim -= temporal_dim % 2
    spatial_dim = dim - temporal_dim
    y_dim = spatial_dim // 2
    y_dim -= y_dim % 2
    x_dim = spatial_dim - y_dim
    if x_dim % 2:
        y_dim -= 2
        x_dim += 2
    t = _sincos_1d(temporal_dim, torch.arange(temporal))
    y = _sincos_1d(y_dim, torch.arange(height))
    x = _sincos_1d(x_dim, torch.arange(width))
    pos = torch.cat(
        (
            t[:, None, None, :].expand(-1, height, width, -1),
            y[None, :, None, :].expand(temporal, -1, width, -1),
            x[None, None, :, :].expand(temporal, height, -1, -1),
        ),
        dim=-1,
    )
    return pos.reshape(1, temporal * height * width, dim)


def tubelet_patchify(video: torch.Tensor, tubelet_size: int, patch_size: int) -> torch.Tensor:
    """Convert [B,T,C,H,W] to [B,N,tubelet*patch*patch*C]."""
    b, t, c, h, w = video.shape
    if t % tubelet_size or h % patch_size or w % patch_size:
        raise ValueError("Video dimensions must be divisible by tubelet/patch size")
    gt, gh, gw = t // tubelet_size, h // patch_size, w // patch_size
    x = video.reshape(b, gt, tubelet_size, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    return x.reshape(b, gt * gh * gw, tubelet_size * patch_size * patch_size * c)


class TubeletEmbed3D(nn.Module):
    def __init__(self, img_size: int, frames: int, patch_size: int, tubelet_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if img_size % patch_size or frames % tubelet_size:
            raise ValueError("img_size/frames must be divisible by patch_size/tubelet_size")
        self.grid_size = (frames // tubelet_size, img_size // patch_size, img_size // patch_size)
        self.num_patches = math.prod(self.grid_size)
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.proj(video.transpose(1, 2)).flatten(2).transpose(1, 2)


class EchoVideoMAE(nn.Module):
    """VideoMAE-S style encoder/decoder adapted to grayscale echocardiograms."""

    def __init__(self, **cfg):
        super().__init__()
        self.core_type = "videomae"
        self.img_size = int(cfg.get("img_size", 112))
        self.frames = int(cfg.get("frames", 16))
        self.patch_size = int(cfg.get("patch_size", 16))
        self.tubelet_size = int(cfg.get("tubelet_size", 2))
        self.in_chans = int(cfg.get("in_chans", 1))
        self.embed_dim = int(cfg.get("embed_dim", 384))
        self.mask_ratio = float(cfg.get("mask_ratio", 0.9))
        self.mask_strategy = str(cfg.get("mask_strategy", "tube")).lower()
        self.norm_pix_loss = bool(cfg.get("norm_pix_loss", True))
        mean, std = cfg.get("input_mean"), cfg.get("input_std")
        self.register_buffer("input_mean", None if mean is None else torch.tensor(mean).view(1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("input_std", None if std is None else torch.tensor(std).view(1, 1, -1, 1, 1), persistent=False)
        depth = int(cfg.get("depth", 12))
        heads = int(cfg.get("num_heads", 6))
        decoder_dim = int(cfg.get("decoder_embed_dim", 192))
        decoder_depth = int(cfg.get("decoder_depth", 4))
        decoder_heads = int(cfg.get("decoder_num_heads", 3))
        mlp_ratio = float(cfg.get("mlp_ratio", 4.0))
        drop_path = float(cfg.get("drop_path_rate", 0.0))

        self.patch_embed = TubeletEmbed3D(
            self.img_size, self.frames, self.patch_size, self.tubelet_size, self.in_chans, self.embed_dim
        )
        gt, gh, gw = self.patch_embed.grid_size
        self.register_buffer("pos_embed", get_3d_sincos_pos_embed(self.embed_dim, gt, gh, gw), persistent=False)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [Block(self.embed_dim, heads, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.decoder_embed = nn.Linear(self.embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer(
            "decoder_pos_embed", get_3d_sincos_pos_embed(decoder_dim, gt, gh, gw), persistent=False
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads, mlp_ratio=mlp_ratio) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        patch_dim = self.tubelet_size * self.patch_size * self.patch_size * self.in_chans
        self.decoder_pred = nn.Linear(decoder_dim, patch_dim)
        self.initialize_weights()

    @property
    def token_grid(self) -> tuple[int, int, int]:
        return self.patch_embed.grid_size

    def initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.flatten(1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _tube_mask(self, batch: int, ratio: float, device: torch.device) -> torch.Tensor:
        """One spatial mask shared by all temporal tubelets."""
        gt, gh, gw = self.token_grid
        spatial = gh * gw
        keep = max(1, int(round(spatial * (1.0 - ratio))))
        ids = torch.rand(batch, spatial, device=device).argsort(dim=1)
        spatial_mask = torch.ones(batch, spatial, dtype=torch.bool, device=device)
        spatial_mask.scatter_(1, ids[:, :keep], False)
        return spatial_mask[:, None].expand(-1, gt, -1).reshape(batch, gt * spatial)

    def _normalize_input(self, video: torch.Tensor) -> torch.Tensor:
        if self.input_mean is None:
            return video
        return (video - self.input_mean.to(video)) / self.input_std.to(video)

    def _make_mask(self, batch: int, ratio: float, device: torch.device) -> torch.Tensor:
        if self.mask_strategy == "tube":
            return self._tube_mask(batch, ratio, device)
        if self.mask_strategy != "random":
            raise ValueError(f"Unknown mask_strategy={self.mask_strategy!r}")
        keep = max(1, int(round(self.patch_embed.num_patches * (1.0 - ratio))))
        ids = torch.rand(batch, self.patch_embed.num_patches, device=device).argsort(dim=1)
        mask = torch.ones(batch, self.patch_embed.num_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :keep], False)
        return mask

    def encode_video(
        self, video: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        video = self._normalize_input(video)
        tokens = self.patch_embed(video)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        if mask is not None:
            visible_count = int((~mask[0]).sum().item())
            tokens = tokens[~mask].reshape(video.shape[0], visible_count, self.embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens), mask

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.encode_video(video, mask=None)
        b = video.shape[0]
        gt, gh, gw = self.token_grid
        return tokens.reshape(b, gt, gh * gw, self.embed_dim)

    def forward(self, video: torch.Tensor, mask_ratio: float | None = None) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,T,C,H,W]")
        expected = (self.frames, self.img_size, self.img_size)
        if (video.shape[1], video.shape[-2], video.shape[-1]) != expected:
            raise ValueError(f"Expected [B,{self.frames},C,{self.img_size},{self.img_size}], got {tuple(video.shape)}")
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        mask = self._make_mask(video.shape[0], ratio, video.device)
        visible, _ = self.encode_video(video, mask)
        decoded_visible = self.decoder_embed(visible)
        full = self.mask_token.to(
            device=decoded_visible.device, dtype=decoded_visible.dtype
        ).expand(video.shape[0], self.patch_embed.num_patches, -1).clone()
        full[~mask] = decoded_visible.reshape(-1, decoded_visible.shape[-1])
        full = full + self.decoder_pos_embed.to(device=full.device, dtype=full.dtype)
        for block in self.decoder_blocks:
            full = block(full)
        pred = self.decoder_pred(self.decoder_norm(full))
        target = tubelet_patchify(self._normalize_input(video), self.tubelet_size, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        patch_loss = (pred - target).square().mean(dim=-1)
        loss = (patch_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return {
            "loss": loss,
            "loss_total": loss.detach(),
            "loss_recon": loss.detach(),
            "pred": pred,
            "target": target,
            "mask": mask,
            "tokens": visible,
        }


def build_echo_videomae(cfg: dict) -> EchoVideoMAE:
    return EchoVideoMAE(**cfg)
