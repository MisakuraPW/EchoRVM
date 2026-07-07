"""Recurrent echocardiography masked autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from .echo_frame_mae import EchoFrameMAE
from .losses import masked_reconstruction_loss, state_smooth_loss, temporal_infonce_loss
from .rvm_core import RVMCore
from .ttt_core import TTTCore


class EchoRMAE(nn.Module):
    """EchoCardMAE-style frame MAE plus a recurrent temporal core."""

    def __init__(self, **cfg):
        super().__init__()
        self.core_type = str(cfg.get("core_type", cfg.get("recurrent_core", "rvm"))).lower()
        self.mask_ratio = float(cfg.get("mask_ratio", 0.75))
        self.align_weight = float(cfg.get("align_weight", 0.0))
        self.state_smooth_weight = float(cfg.get("state_smooth_weight", 0.0))
        self.frame_mae = EchoFrameMAE(**cfg)
        dim = int(cfg.get("embed_dim", 384))
        heads = int(cfg.get("num_heads", 6))
        core_depth = int(cfg.get("core_depth", 2))
        mlp_ratio = float(cfg.get("mlp_ratio", 4.0))
        if self.core_type == "rvm":
            self.core = RVMCore(dim=dim, num_heads=heads, depth=core_depth, mlp_ratio=mlp_ratio, drop_path_rate=float(cfg.get("core_drop_path_rate", 0.0)))
        elif self.core_type == "ttt":
            self.core = TTTCore(
                dim=dim,
                num_heads=heads,
                depth=core_depth,
                mlp_ratio=mlp_ratio,
                inner_lr=float(cfg.get("ttt_inner_lr", 0.25)),
                inner_steps=int(cfg.get("ttt_inner_steps", 1)),
            )
        else:
            raise ValueError(f"Unknown core_type: {self.core_type}")

    def forward(self, video: torch.Tensor, roi_mask: torch.Tensor | None = None, mask_ratio: float | None = None) -> dict[str, torch.Tensor]:
        """Forward current-frame masked reconstruction.

        video: [B,T,1,112,112]
        roi_mask: optional [B,T,112,112] or [B,T,1,112,112]
        """

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
        encoded_flat, _ = self.frame_mae.encode_frames(flat)
        encoded = encoded_flat.reshape(b, t, self.frame_mae.num_patches, -1)
        mask = mask_flat.reshape(b, t, self.frame_mae.num_patches)
        valid = valid_flat.reshape(b, t, self.frame_mae.num_patches)

        states = []
        outputs = []
        state = None
        for idx in range(t):
            out, state = self.core(encoded[:, idx], state)
            outputs.append(out)
            states.append(state)
        state_seq = torch.stack(states, dim=1)
        out_seq = torch.stack(outputs, dim=1)
        pred_flat = self.frame_mae.decode_frames(out_seq.reshape(b * t, self.frame_mae.num_patches, -1), mask_flat, valid_flat)
        pred = pred_flat.reshape(b, t, self.frame_mae.num_patches, -1)
        target = self.frame_mae.target_patches(flat).reshape_as(pred)
        recon = masked_reconstruction_loss(pred, target, mask, patch_weight=valid)
        smooth = state_smooth_loss(state_seq)
        align = pred.new_tensor(0.0)
        if self.align_weight > 0.0 and t >= 2:
            align = temporal_infonce_loss(out_seq[:, 0].mean(dim=1), out_seq[:, -1].mean(dim=1))
        total = recon + self.state_smooth_weight * smooth + self.align_weight * align
        return {
            "loss": total,
            "loss_total": total.detach(),
            "loss_recon": recon.detach(),
            "loss_state_smooth": smooth.detach(),
            "loss_align": align.detach(),
            "pred": pred,
            "target": target,
            "mask": mask,
            "valid_mask": valid,
            "tokens": encoded,
            "state_sequence": state_seq,
        }


def build_echo_rmae(cfg: dict) -> EchoRMAE:
    return EchoRMAE(**cfg)
