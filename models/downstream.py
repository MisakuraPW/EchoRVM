"""Downstream heads on top of EchoRMAE backbones."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .echo_rmae import EchoRMAE, build_echo_rmae
from .echo_single_frame_mae import EchoSingleFrameMAE, build_echo_single_frame_mae
from hiera_echo.models import EchoHieraMAE
from .patch import get_2d_sincos_pos_embed
from .vit_blocks import Block


def _checkpoint_model_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        state = ckpt
    cleaned: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("_orig_mod."):
            name = name[10:]
        cleaned[name] = tensor
    return cleaned


def load_pretrained_rmae(
    checkpoint_path: str | Path,
    fallback_model_cfg: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any], dict[str, list[str]]]:
    """Build EchoRMAE and load a pretraining checkpoint.

    The pretraining trainer stores the full config in the checkpoint. That is
    preferred because it guarantees RVM/TTT depth and dimensions match the run.
    """

    checkpoint_path = Path(checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    cfg = dict(fallback_model_cfg or {})
    if isinstance(ckpt, dict):
        saved_cfg = ckpt.get("config", {})
        if isinstance(saved_cfg, dict) and isinstance(saved_cfg.get("model"), dict):
            cfg = dict(saved_cfg["model"])
    model_name = str(cfg.get("name", "echo_rmae")).lower()
    if model_name in {"echo_single_frame_mae", "single_frame_mae", "videomae_single_frame"}:
        model = build_echo_single_frame_mae(cfg)
    else:
        model = build_echo_rmae(cfg)
    missing, unexpected = model.load_state_dict(_checkpoint_model_state(ckpt), strict=False)
    report = {"missing": list(missing), "unexpected": list(unexpected)}
    return model, cfg, report


def is_hiera_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> bool:
    checkpoint_path = Path(checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(ckpt, dict):
        return False
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if isinstance(model_cfg, dict) and ("hiera_repo" in model_cfg or "loss_mode" in model_cfg):
        return True
    state = _checkpoint_model_state(ckpt)
    return any(key.startswith("model.patch_embed.") or key.startswith("model.blocks.") for key in state)


def load_pretrained_hiera_mae(
    checkpoint_path: str | Path,
    fallback_model_cfg: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[EchoHieraMAE, dict[str, Any], dict[str, list[str]]]:
    checkpoint_path = Path(checkpoint_path)
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    cfg = dict(fallback_model_cfg or {})
    if isinstance(ckpt, dict):
        saved_cfg = ckpt.get("config", {})
        if isinstance(saved_cfg, dict) and isinstance(saved_cfg.get("model"), dict):
            cfg = dict(saved_cfg["model"])
    model = EchoHieraMAE(
        img_size=int(cfg.get("img_size", 192)),
        mask_ratio=float(cfg.get("mask_ratio", 0.6)),
        loss_mode=str(cfg.get("loss_mode", "valid_weighted")),
        hiera_repo=cfg.get("hiera_repo"),
        init_checkpoint=None,
    )
    missing, unexpected = model.load_state_dict(_checkpoint_model_state(ckpt), strict=False)
    return model, cfg, {"missing": list(missing), "unexpected": list(unexpected)}


class EchoRMAEBackbone(nn.Module):
    """Feature extractor for downstream dense and sequence tasks."""

    def __init__(self, rmae: EchoRMAE | EchoSingleFrameMAE):
        super().__init__()
        self.rmae = rmae
        self.embed_dim = int(rmae.frame_mae.encoder.norm.normalized_shape[0])
        self.num_patches = int(rmae.frame_mae.num_patches)
        self.grid_size = int(self.num_patches**0.5)
        self.has_temporal_core = hasattr(rmae, "core")

    def forward_tokens(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,T,C,H,W]")
        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        encoded_flat, _ = self.rmae.frame_mae.encode_frames(flat, mask=None, valid_mask=None)
        encoded = encoded_flat.reshape(b, t, self.num_patches, self.embed_dim)
        if not self.has_temporal_core:
            return {"encoded": encoded, "outputs": encoded, "states": encoded}
        states = []
        outputs = []
        state = None
        for idx in range(t):
            out, state = self.rmae.core(encoded[:, idx], state)
            outputs.append(out)
            states.append(state)
        return {
            "encoded": encoded,
            "outputs": torch.stack(outputs, dim=1),
            "states": torch.stack(states, dim=1),
        }


class PatchSegDecoder(nn.Module):
    """Upsample 14x14 ViT tokens to a 112x112 segmentation map."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        b, n, d = tokens.shape
        if n != grid_size * grid_size:
            raise ValueError(f"Expected {grid_size * grid_size} tokens, got {n}")
        x = self.norm(tokens).transpose(1, 2).reshape(b, d, grid_size, grid_size)
        return self.decoder(x)


class ViTPatchSegDecoder(nn.Module):
    """EchoCardMAE-style transformer decoder for dense patch logits."""

    def __init__(
        self,
        encoder_dim: int,
        num_classes: int,
        grid_size: int,
        patch_size: int = 8,
        decoder_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=False)
        pos = get_2d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer("pos_embed", pos, persistent=False)
        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, num_classes * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        b, n, _ = tokens.shape
        if grid_size != self.grid_size or n != self.grid_size * self.grid_size:
            raise ValueError(f"Expected {self.grid_size * self.grid_size} tokens, got grid={grid_size} n={n}")
        x = self.encoder_to_decoder(tokens) + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.head(self.norm(x))
        p = self.patch_size
        g = self.grid_size
        x = x.reshape(b, g, g, p, p, self.num_classes)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.num_classes, g * p, g * p)
        return x


class EchoSegFineTuner(nn.Module):
    def __init__(
        self,
        rmae: EchoRMAE | EchoSingleFrameMAE,
        num_classes: int,
        dropout: float = 0.1,
        decoder_type: str = "vit_patch",
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
    ):
        super().__init__()
        self.backbone = EchoRMAEBackbone(rmae)
        decoder_type = decoder_type.lower()
        if decoder_type in {"vit", "vit_patch", "echocardmae"}:
            self.head = ViTPatchSegDecoder(
                self.backbone.embed_dim,
                num_classes,
                self.backbone.grid_size,
                patch_size=int(rmae.frame_mae.patch_size),
                decoder_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
            )
        elif decoder_type in {"conv", "convtranspose"}:
            self.head = PatchSegDecoder(self.backbone.embed_dim, num_classes, dropout=dropout)
        else:
            raise ValueError(f"Unknown segmentation decoder_type={decoder_type!r}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("image must have shape [B,C,H,W]")
        video = image.unsqueeze(1)
        return self.forward_video(video)

    def forward_video(self, video: torch.Tensor, target_index: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone.forward_tokens(video)
        outputs = features["outputs"]
        if target_index is None:
            tokens = outputs[:, outputs.shape[1] // 2]
        else:
            target_index = target_index.to(device=outputs.device, dtype=torch.long).clamp(0, outputs.shape[1] - 1)
            tokens = outputs[torch.arange(outputs.shape[0], device=outputs.device), target_index]
        return self.head(tokens, self.backbone.grid_size)


class HieraBackbone(nn.Module):
    def __init__(self, mae: EchoHieraMAE):
        super().__init__()
        self.mae = mae
        self.embed_dim = int(mae.model.blocks[-1].dim_out)
        self.grid_size = int(mae.img_size // mae.pred_stride)
        self.patch_size = int(mae.pred_stride)

    def forward_tokens(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,T,C,H,W]")
        from hiera.hiera import Hiera
        from hiera.hiera_utils import undo_windowing

        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        x = self.mae._to_rgb(flat)
        keep_mask_mu = torch.ones(
            x.shape[0],
            math.prod(self.mae.model.mask_spatial_shape),
            dtype=torch.bool,
            device=x.device,
        )
        _, raw_stages = Hiera.forward(self.mae.model, x, mask=keep_mask_mu, return_intermediates=True)
        stages = []
        for block_idx, feat in zip(self.mae.model.stage_ends, raw_stages):
            if feat.ndim == 5:
                _, size = self.mae.model.reroll.schedule[block_idx]
                feat = undo_windowing(feat, size, list(feat.shape[2:-1]))
            if feat.ndim == 4:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            stages.append(feat)
        fmap = stages[-1]
        tokens = fmap.flatten(2).transpose(1, 2).reshape(b, t, self.grid_size * self.grid_size, self.embed_dim)
        return {"outputs": tokens, "encoded": tokens, "states": tokens}


class HieraSegFineTuner(nn.Module):
    def __init__(
        self,
        mae: EchoHieraMAE,
        num_classes: int,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
    ):
        super().__init__()
        self.backbone = HieraBackbone(mae)
        self.head = ViTPatchSegDecoder(
            self.backbone.embed_dim,
            num_classes,
            self.backbone.grid_size,
            patch_size=self.backbone.patch_size,
            decoder_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward_video(image.unsqueeze(1))

    def forward_video(self, video: torch.Tensor, target_index: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone.forward_tokens(video)
        outputs = features["outputs"]
        if target_index is None:
            tokens = outputs[:, outputs.shape[1] // 2]
        else:
            target_index = target_index.to(device=outputs.device, dtype=torch.long).clamp(0, outputs.shape[1] - 1)
            tokens = outputs[torch.arange(outputs.shape[0], device=outputs.device), target_index]
        return self.head(tokens, self.backbone.grid_size)


class TemporalQueryEFHead(nn.Module):
    """Learn ED/ES-like query pooling over recurrent frame states."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.q_ed = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.q_es = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _query_pool(self, sequence: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, dim=0)
        seq = F.normalize(sequence, dim=-1)
        attn = torch.softmax(torch.einsum("btd,d->bt", seq, query), dim=1)
        return torch.einsum("bt,btd->bd", attn, sequence)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        pooled = self.norm(states.mean(dim=2))
        ed = self._query_pool(pooled, self.q_ed)
        es = self._query_pool(pooled, self.q_es)
        mean = pooled.mean(dim=1)
        x = torch.cat([ed, es, ed - es, mean], dim=-1)
        return self.mlp(x).squeeze(-1)


class EchoEFFineTuner(nn.Module):
    def __init__(self, rmae: EchoRMAE, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = EchoRMAEBackbone(rmae)
        self.head = TemporalQueryEFHead(self.backbone.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_tokens(video)
        return self.head(features["states"])


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice[1:].mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, dice_weight: float = 1.0) -> torch.Tensor:
    target = target.clamp(0, num_classes - 1)
    return F.cross_entropy(logits, target) + dice_weight * dice_loss(logits, target, num_classes)


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    target = target.clamp(0, num_classes - 1)
    out: dict[str, float] = {}
    dices = []
    for cls in range(1, num_classes):
        p = pred == cls
        t = target == cls
        denom = p.sum().item() + t.sum().item()
        tp = (p & t).sum().item()
        dice = 1.0 if denom == 0 else 2.0 * (p & t).sum().item() / denom
        out[f"dice_class_{cls}"] = float(dice)
        out[f"_seg_tp_class_{cls}"] = float(tp)
        out[f"_seg_pred_class_{cls}"] = float(p.sum().item())
        out[f"_seg_target_class_{cls}"] = float(t.sum().item())
        dices.append(float(dice))
    out["dice_mean"] = float(sum(dices) / max(1, len(dices)))
    return out


def ef_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = pred.detach().float().cpu()
    y = target.detach().float().cpu()
    err = p - y
    mae = torch.mean(torch.abs(err)).item()
    rmse = torch.sqrt(torch.mean(err.square())).item()
    within_5 = torch.mean((torch.abs(err) <= 5.0).float()).item()
    within_10 = torch.mean((torch.abs(err) <= 10.0).float()).item()
    if p.numel() > 1 and float(torch.std(p)) > 0 and float(torch.std(y)) > 0:
        corr = float(torch.corrcoef(torch.stack([p, y]))[0, 1])
    else:
        corr = 0.0
    return {"mae": mae, "rmse": rmse, "corr": corr, "within_5": within_5, "within_10": within_10}
