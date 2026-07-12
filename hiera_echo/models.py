"""Thin wrappers around the official Meta Hiera MAE implementation.

The upstream code lives outside this package.  We keep all echo-specific
behavior here so the official ``hiera/`` package can be updated cleanly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIERA_REPO = ROOT / "third_party" / "hiera"


def ensure_hiera_importable(hiera_repo: str | Path | None = None) -> Path:
    repo = Path(hiera_repo or DEFAULT_HIERA_REPO)
    if not repo.exists():
        raise FileNotFoundError(
            f"Official Hiera repo not found at {repo}. Set HIERA_REPO or config model.hiera_repo."
        )
    repo_s = str(repo.resolve())
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    return repo


def _load_torch(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    return ckpt


def _interp_pos_embed(value: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if value.ndim != 3 or value.shape[1] == target_tokens:
        return value
    src = int(round(math.sqrt(value.shape[1])))
    dst = int(round(math.sqrt(target_tokens)))
    if src * src != value.shape[1] or dst * dst != target_tokens:
        raise ValueError(f"Cannot interpolate positional embedding {tuple(value.shape)} -> {target_tokens}")
    x = value.reshape(1, src, src, value.shape[-1]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(dst, dst), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(1, target_tokens, value.shape[-1])


def convert_hiera_state_for_model(
    source_state: dict[str, torch.Tensor],
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Adapt official Hiera MAE weights to the target input resolution.

    Only encoder and decoder positional embeddings are allowed to change shape.
    Any other mismatch is reported and skipped by this helper; the caller can
    decide whether that should be fatal.
    """

    target_state = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    exact: list[str] = []
    interpolated: list[str] = []
    skipped_shape: list[dict[str, Any]] = []
    unexpected: list[str] = []

    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        target = target_state[key]
        if tuple(value.shape) == tuple(target.shape):
            converted[key] = value
            exact.append(key)
            continue
        if key in {"pos_embed", "decoder_pos_embed"} and value.ndim == 3 and target.ndim == 3:
            converted[key] = _interp_pos_embed(value, target.shape[1])
            interpolated.append(key)
            continue
        skipped_shape.append({"key": key, "source": tuple(value.shape), "target": tuple(target.shape)})

    missing = [key for key in target_state if key not in converted]
    report = {
        "copied_exact": exact,
        "interpolated": interpolated,
        "skipped_shape": skipped_shape,
        "missing": missing,
        "unexpected": unexpected,
    }
    return converted, report


def load_hiera_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> dict[str, Any]:
    ckpt = _load_torch(checkpoint_path)
    state = _as_state_dict(ckpt)
    converted, report = convert_hiera_state_for_model(state, model)
    incompatible = model.load_state_dict(converted, strict=False)
    report["load_missing"] = list(incompatible.missing_keys)
    report["load_unexpected"] = list(incompatible.unexpected_keys)
    if strict and (report["skipped_shape"] or report["load_missing"] or report["load_unexpected"]):
        raise RuntimeError(f"Hiera checkpoint did not strictly load: {report}")
    return report


def patchify_valid_mask(valid_mask: torch.Tensor, pred_stride: int) -> torch.Tensor:
    """Return per-prediction-token validity weights from [B,1,H,W] masks."""

    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, None]
    patches = valid_mask.float().unfold(2, pred_stride, pred_stride).unfold(3, pred_stride, pred_stride)
    return patches.flatten(2, 3).flatten(3).mean(dim=-1).squeeze(1)


class EchoHieraMAE(nn.Module):
    """Single-frame echo MAE based on official ``mae_hiera_tiny_224``.

    Input contract:
        ``image``: ``[B,1,H,W]`` or ``[B,3,H,W]`` in ``[0,1]``.
    """

    def __init__(
        self,
        img_size: int = 192,
        mask_ratio: float = 0.6,
        loss_mode: str = "valid_weighted",
        hiera_repo: str | Path | None = None,
        init_checkpoint: str | Path | None = None,
    ):
        super().__init__()
        ensure_hiera_importable(hiera_repo)
        from hiera.hiera_mae import mae_hiera_tiny_224

        self.img_size = int(img_size)
        self.mask_ratio = float(mask_ratio)
        self.loss_mode = str(loss_mode)
        self.model = mae_hiera_tiny_224(input_size=(self.img_size, self.img_size), pretrained=False)
        self.init_report: dict[str, Any] | None = None
        if init_checkpoint:
            self.init_report = load_hiera_checkpoint(self.model, init_checkpoint, strict=False)

    @property
    def pred_stride(self) -> int:
        return int(self.model.pred_stride)

    def _to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[1] == 1:
            return image.repeat(1, 3, 1, 1)
        if image.shape[1] == 3:
            return image
        raise ValueError(f"Hiera baseline expects 1 or 3 channels, got {image.shape[1]}")

    def forward(self, image: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        image = self._to_rgb(image)
        if self.loss_mode == "official" or valid_mask is None:
            loss, pred, target, keep_mask_mu = self.model(image, mask_ratio=self.mask_ratio)
            return {"loss": loss, "pred": pred, "target": target, "keep_mask_mu": keep_mask_mu}

        latent, keep_mask_mu = self.model.forward_encoder(image, self.mask_ratio)
        pred, pred_keep_mask = self.model.forward_decoder(latent, keep_mask_mu)
        masked_mask_pred = ~pred_keep_mask
        label = self.model.get_pixel_label_2d(image, masked_mask_pred)
        pred_masked = pred[masked_mask_pred]
        per_patch = ((pred_masked - label) ** 2).mean(dim=-1)
        weights = patchify_valid_mask(valid_mask, self.pred_stride).to(per_patch.device)
        weights = weights[masked_mask_pred].clamp_min(0.0)
        denom = weights.sum().clamp_min(1.0)
        loss = (per_patch * weights).sum() / denom
        return {
            "loss": loss,
            "pred": pred_masked,
            "target": label,
            "keep_mask_mu": keep_mask_mu,
            "valid_weight": weights.detach(),
        }

    @torch.no_grad()
    def encode_frame(self, image: torch.Tensor, return_stages: bool = True) -> dict[str, Any]:
        """Return spatial stage features for later Recurrent-Hiera work."""

        x = self._to_rgb(image)
        from hiera.hiera import Hiera
        from hiera.hiera_utils import undo_windowing

        keep_mask_mu = torch.ones(
            x.shape[0],
            math.prod(self.model.mask_spatial_shape),
            dtype=torch.bool,
            device=x.device,
        )
        _, stages = Hiera.forward(self.model, x, mask=keep_mask_mu, return_intermediates=True)
        out = []
        stage_indices = self.model.stage_ends
        for block_idx, feat in zip(stage_indices, stages):
            if feat.ndim == 5:
                _, size = self.model.reroll.schedule[block_idx]
                feat = undo_windowing(feat, size, list(feat.shape[2:-1]))
            if feat.ndim == 4:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            out.append(feat)
        return {"stages": out, "img_size": self.img_size, "pred_stride": self.pred_stride}


def build_echo_hiera_mae(cfg: dict[str, Any]) -> EchoHieraMAE:
    return EchoHieraMAE(
        img_size=int(cfg.get("img_size", 192)),
        mask_ratio=float(cfg.get("mask_ratio", 0.6)),
        loss_mode=str(cfg.get("loss_mode", "valid_weighted")),
        hiera_repo=cfg.get("hiera_repo"),
        init_checkpoint=cfg.get("init_checkpoint"),
    )
