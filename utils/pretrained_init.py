"""Pretrained checkpoint adapters for EchoRMAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def _checkpoint_state(ckpt: Any) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model", "module", "state_dict", "model_state_dict"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "backbone.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _convert_patch_embed(weight: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Convert VideoMAE 3D RGB patch embedding to 2D grayscale patch embedding."""

    if weight.shape == target.shape:
        return weight
    if weight.ndim == 5 and target.ndim == 4:
        # [O, RGB, tubelet, P, P] -> [O, 1, P, P]
        if weight.shape[-2:] != target.shape[-2:]:
            weight = F.interpolate(
                weight,
                size=(weight.shape[2], target.shape[-2], target.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        weight = weight.mean(dim=2).mean(dim=1, keepdim=True)
        if weight.shape == target.shape:
            return weight
    if weight.ndim == 4 and target.ndim == 4 and weight.shape[1] != target.shape[1]:
        if weight.shape[-2:] != target.shape[-2:]:
            weight = F.interpolate(weight, size=target.shape[-2:], mode="bilinear", align_corners=False)
        weight = weight.mean(dim=1, keepdim=True)
        if weight.shape == target.shape:
            return weight
    return None


def load_videomae_init(model: torch.nn.Module, checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load compatible VideoMAE/EchoCardMAE ViT weights into EchoRMAE.

    The recurrent core is intentionally left randomly initialized. The adapter
    maps the shared token transformer body and decoder body while converting the
    3D RGB patch embedding into our 2D grayscale patch embedding when possible.
    """

    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    src = {_strip_prefix(k): v for k, v in _checkpoint_state(ckpt).items() if torch.is_tensor(v)}
    dst = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}

    aliases = [
        ("encoder.patch_embed.proj.weight", "frame_mae.encoder.patch_embed.proj.weight"),
        ("encoder.patch_embed.proj.bias", "frame_mae.encoder.patch_embed.proj.bias"),
        ("encoder.norm.weight", "frame_mae.encoder.norm.weight"),
        ("encoder.norm.bias", "frame_mae.encoder.norm.bias"),
        ("encoder_to_decoder.weight", "frame_mae.decoder.encoder_to_decoder.weight"),
        ("mask_token", "frame_mae.decoder.mask_token"),
        ("background_token", "frame_mae.decoder.background_token"),
        ("decoder.norm.weight", "frame_mae.decoder.norm.weight"),
        ("decoder.norm.bias", "frame_mae.decoder.norm.bias"),
    ]
    for i in range(len(getattr(model.frame_mae.encoder, "blocks", []))):
        aliases.extend(
            [
                (f"encoder.blocks.{i}.norm1.weight", f"frame_mae.encoder.blocks.{i}.norm1.weight"),
                (f"encoder.blocks.{i}.norm1.bias", f"frame_mae.encoder.blocks.{i}.norm1.bias"),
                (f"encoder.blocks.{i}.attn.qkv.weight", f"frame_mae.encoder.blocks.{i}.attn.qkv.weight"),
                (f"encoder.blocks.{i}.attn.qkv.bias", f"frame_mae.encoder.blocks.{i}.attn.qkv.bias"),
                (f"encoder.blocks.{i}.attn.proj.weight", f"frame_mae.encoder.blocks.{i}.attn.proj.weight"),
                (f"encoder.blocks.{i}.attn.proj.bias", f"frame_mae.encoder.blocks.{i}.attn.proj.bias"),
                (f"encoder.blocks.{i}.norm2.weight", f"frame_mae.encoder.blocks.{i}.norm2.weight"),
                (f"encoder.blocks.{i}.norm2.bias", f"frame_mae.encoder.blocks.{i}.norm2.bias"),
                (f"encoder.blocks.{i}.mlp.fc1.weight", f"frame_mae.encoder.blocks.{i}.mlp.fc1.weight"),
                (f"encoder.blocks.{i}.mlp.fc1.bias", f"frame_mae.encoder.blocks.{i}.mlp.fc1.bias"),
                (f"encoder.blocks.{i}.mlp.fc2.weight", f"frame_mae.encoder.blocks.{i}.mlp.fc2.weight"),
                (f"encoder.blocks.{i}.mlp.fc2.bias", f"frame_mae.encoder.blocks.{i}.mlp.fc2.bias"),
            ]
        )
    for i in range(len(getattr(model.frame_mae.decoder, "blocks", []))):
        aliases.extend(
            [
                (f"decoder.blocks.{i}.norm1.weight", f"frame_mae.decoder.blocks.{i}.norm1.weight"),
                (f"decoder.blocks.{i}.norm1.bias", f"frame_mae.decoder.blocks.{i}.norm1.bias"),
                (f"decoder.blocks.{i}.attn.qkv.weight", f"frame_mae.decoder.blocks.{i}.attn.qkv.weight"),
                (f"decoder.blocks.{i}.attn.qkv.bias", f"frame_mae.decoder.blocks.{i}.attn.qkv.bias"),
                (f"decoder.blocks.{i}.attn.proj.weight", f"frame_mae.decoder.blocks.{i}.attn.proj.weight"),
                (f"decoder.blocks.{i}.attn.proj.bias", f"frame_mae.decoder.blocks.{i}.attn.proj.bias"),
                (f"decoder.blocks.{i}.norm2.weight", f"frame_mae.decoder.blocks.{i}.norm2.weight"),
                (f"decoder.blocks.{i}.norm2.bias", f"frame_mae.decoder.blocks.{i}.norm2.bias"),
                (f"decoder.blocks.{i}.mlp.fc1.weight", f"frame_mae.decoder.blocks.{i}.mlp.fc1.weight"),
                (f"decoder.blocks.{i}.mlp.fc1.bias", f"frame_mae.decoder.blocks.{i}.mlp.fc1.bias"),
                (f"decoder.blocks.{i}.mlp.fc2.weight", f"frame_mae.decoder.blocks.{i}.mlp.fc2.weight"),
                (f"decoder.blocks.{i}.mlp.fc2.bias", f"frame_mae.decoder.blocks.{i}.mlp.fc2.bias"),
            ]
        )

    for src_key, dst_key in aliases:
        if src_key not in src or dst_key not in dst:
            continue
        src_tensor = src[src_key]
        dst_tensor = dst[dst_key]
        if src_key.endswith("patch_embed.proj.weight"):
            converted = _convert_patch_embed(src_tensor, dst_tensor)
            if converted is None:
                skipped[src_key] = f"shape {tuple(src_tensor.shape)} -> {tuple(dst_tensor.shape)}"
                continue
            src_tensor = converted
        if src_tensor.shape != dst_tensor.shape:
            skipped[src_key] = f"shape {tuple(src_tensor.shape)} != {tuple(dst_tensor.shape)}"
            continue
        mapped[dst_key] = src_tensor.to(dtype=dst_tensor.dtype)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    return {
        "path": str(checkpoint_path),
        "loaded_tensors": len(mapped),
        "loaded_params": int(sum(t.numel() for t in mapped.values())),
        "skipped": skipped,
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
    }
