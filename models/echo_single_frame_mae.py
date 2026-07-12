"""Clean single-frame Echo/VideoMAE baseline without a temporal core."""

from __future__ import annotations

import torch
from torch import nn

from .echo_frame_mae import EchoFrameMAE
from .losses import masked_reconstruction_loss


class EchoSingleFrameMAE(nn.Module):
    """EchoCardMAE-style frame MAE used as a clean VideoMAE baseline.

    The input contract intentionally stays compatible with the existing trainer:
    ``video`` is ``[B,T,1,H,W]``.  Formal single-frame runs set ``T=1``; if a
    caller passes more frames, every frame is reconstructed independently.
    """

    def __init__(self, **cfg):
        super().__init__()
        self.core_type = "single_frame"
        self.mask_ratio = float(cfg.get("mask_ratio", 0.75))
        self.frame_mae = EchoFrameMAE(**cfg)

    def forward(self, video: torch.Tensor, roi_mask: torch.Tensor | None = None, mask_ratio: float | None = None) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("video must have shape [B,T,C,H,W]")
        b, t, c, h, w = video.shape
        flat = video.reshape(b * t, c, h, w)
        flat_roi = None
        if roi_mask is not None:
            if roi_mask.ndim == 5:
                roi_mask = roi_mask[:, :, 0]
            flat_roi = roi_mask.reshape(b * t, h, w)
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        mask_flat, valid_flat = self.frame_mae.make_masks(flat, ratio, flat_roi)
        encoded_flat, _ = self.frame_mae.encode_frames(flat, mask=mask_flat, valid_mask=valid_flat)
        pred_flat = self.frame_mae.decode_frames(encoded_flat, mask_flat, valid_flat)
        target_flat = self.frame_mae.target_patches(flat)
        n = self.frame_mae.num_patches
        pred = pred_flat.reshape(b, t, n, -1)
        target = target_flat.reshape_as(pred)
        mask = mask_flat.reshape(b, t, n)
        valid = valid_flat.reshape(b, t, n)
        recon = masked_reconstruction_loss(pred, target, mask, patch_weight=valid)
        return {
            "loss": recon,
            "loss_total": recon.detach(),
            "loss_recon": recon.detach(),
            "pred": pred,
            "target": target,
            "mask": mask,
            "valid_mask": valid,
            "tokens": encoded_flat.reshape(b, t, n, -1),
        }


def build_echo_single_frame_mae(cfg: dict) -> EchoSingleFrameMAE:
    return EchoSingleFrameMAE(**cfg)
