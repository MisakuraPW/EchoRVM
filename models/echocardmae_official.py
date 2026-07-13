"""Official EchoCardMAE-style ViT segmentation fine-tuner.

This module mirrors the public EchoCardMAE ``ViTSeg2D`` head closely enough to
load its released checkpoints by key name.  It is intentionally separate from
the RMAE/Hiera wrappers so that official-vs-local comparisons do not pass
through a different backbone abstraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


class OfficialAttention(nn.Module):
    """Attention with q/v bias parameters matching EchoCardMAE keys."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_head_dim: int | None = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads if attn_head_dim is None else int(attn_head_dim)
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        # The public EchoCardMAE code defines q_bias/v_bias but its active
        # forward path ignores them.  Keep that behavior for checkpoint parity.
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, b * self.num_heads, n, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(b, self.num_heads, n, -1).permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.proj(x)


class OfficialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OfficialAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)
        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class OfficialPatchEmbed2D(nn.Module):
    def __init__(self, img_size: int = 112, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = (int(img_size), int(img_size))
        self.patch_size = (int(patch_size), int(patch_size))
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"Input image size ({h}*{w}) does not match model ({self.img_size[0]}*{self.img_size[1]}).")
        return self.proj(x).flatten(2).transpose(1, 2)


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    def get_position_angle_vec(position: int):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32, requires_grad=False).unsqueeze(0)


class OfficialViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch_embed = OfficialPatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                OfficialBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed.to(device=x.device, dtype=x.dtype))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class OfficialViTDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        patch_size: int = 8,
        grid_size: int = 14,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)
        self.grid_size = int(grid_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                OfficialBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes * patch_size * patch_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.head(self.norm(x))
        b = x.shape[0]
        g = self.grid_size
        p = self.patch_size
        x = x.reshape(b, g, g, p, p, self.num_classes)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.num_classes, g * p, g * p)


class EchoCardMAEOfficialSeg(nn.Module):
    """EchoCardMAE public ViTSeg2D with grayscale-to-RGB input adapter."""

    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 112,
        patch_size: int = 8,
        encoder_in_chans: int = 3,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_num_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
        repeat_gray_to_rgb: bool = True,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.repeat_gray_to_rgb = bool(repeat_gray_to_rgb)
        self.normalize_input = bool(normalize_input)
        self.encoder = OfficialViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )
        grid_size = img_size // patch_size
        self.decoder = OfficialViTDecoder(
            num_classes=num_classes,
            patch_size=patch_size,
            grid_size=grid_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim))
        self.register_buffer("input_mean", torch.tensor([0.1257, 0.1271, 0.1292]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor([0.1951, 0.1957, 0.1974]).view(1, 3, 1, 1), persistent=False)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeat_gray_to_rgb and x.shape[1] == 1 and self.encoder.patch_embed.proj.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        if self.normalize_input and x.shape[1] == 3:
            x = (x - self.input_mean.to(dtype=x.dtype, device=x.device)) / self.input_std.to(dtype=x.dtype, device=x.device)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        tokens = self.encoder(x)
        tokens = self.encoder_to_decoder(tokens)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        return self.decoder(tokens)

    def forward_video(self, video: torch.Tensor, target_index: torch.Tensor | None = None) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,T,C,H,W]")
        if target_index is None:
            image = video[:, video.shape[1] // 2]
        else:
            target_index = target_index.to(device=video.device, dtype=torch.long).clamp(0, video.shape[1] - 1)
            image = video[torch.arange(video.shape[0], device=video.device), target_index]
        return self.forward(image)


def _checkpoint_model_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break
    else:
        state = ckpt
    cleaned = {}
    for name, tensor in state.items():
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("_orig_mod."):
            name = name[10:]
        cleaned[name] = tensor
    return cleaned


def build_echocardmae_official_seg(cfg: dict[str, Any]) -> EchoCardMAEOfficialSeg:
    return EchoCardMAEOfficialSeg(
        num_classes=int(cfg.get("num_classes", 2)),
        img_size=int(cfg.get("img_size", 112)),
        patch_size=int(cfg.get("patch_size", 8)),
        encoder_in_chans=int(cfg.get("encoder_in_chans", cfg.get("in_chans", 3))),
        encoder_embed_dim=int(cfg.get("embed_dim", cfg.get("encoder_embed_dim", 384))),
        encoder_depth=int(cfg.get("depth", cfg.get("encoder_depth", 12))),
        encoder_num_heads=int(cfg.get("num_heads", cfg.get("encoder_num_heads", 6))),
        decoder_embed_dim=int(cfg.get("decoder_embed_dim", 192)),
        decoder_depth=int(cfg.get("decoder_depth", 4)),
        decoder_num_heads=int(cfg.get("decoder_num_heads", 3)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        qkv_bias=bool(cfg.get("qkv_bias", True)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
        init_values=float(cfg.get("init_values", 0.0)),
        repeat_gray_to_rgb=bool(cfg.get("repeat_gray_to_rgb", True)),
        normalize_input=bool(cfg.get("normalize_input", True)),
    )


def load_echocardmae_official_seg(
    checkpoint_path: str | Path,
    cfg: dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[EchoCardMAEOfficialSeg, dict[str, list[str]]]:
    model = build_echocardmae_official_seg(cfg)
    checkpoint_path = Path(checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = _checkpoint_model_state(ckpt)
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for name, tensor in state.items():
        if name not in model_state:
            continue
        if tuple(tensor.shape) != tuple(model_state[name].shape):
            skipped_shape.append(name)
            continue
        filtered[name] = tensor
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    report = {
        "loaded": sorted(filtered.keys()),
        "skipped_shape": skipped_shape,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
    return model, report
