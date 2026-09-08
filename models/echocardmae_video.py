"""Video EchoCardMAE following the official 16-frame pretraining recipe."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .losses import temporal_infonce_loss
from .video_mae import EchoVideoMAE, tubelet_patchify


_OFFICIAL_ROI = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=torch.bool,
)


def _median_blur_video(video: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return video
    b, t, c, h, w = video.shape
    x = video.reshape(b * t, c, h, w)
    x = F.pad(x, (kernel // 2,) * 4, mode='replicate')
    x = x.unfold(2, kernel, 1).unfold(3, kernel, 1)
    return x.contiguous().view(b, t, c, h, w, kernel * kernel).median(dim=-1).values


class EchoCardMAEVideo(EchoVideoMAE):
    """Official-style EchoCardMAE: two video clips, ROI MVM, denoising and InfoNCE."""

    def __init__(self, **cfg):
        cfg = dict(cfg)
        cfg.setdefault('position_embedding', 'flat_sinusoid')
        cfg.setdefault('separate_qv_bias', True)
        cfg.setdefault('norm_eps', 1e-6)
        cfg.setdefault('decoder_embed_bias', False)
        super().__init__(**cfg)
        self.core_type = "echocardmae_video"
        self.align_loss_weight = float(cfg.get("align_loss_weight", 0.2))
        self.alignment_temperature = float(cfg.get("alignment_temperature", 0.1))
        self.median_blur_kernel = int(cfg.get("median_blur_kernel", 3))
        self.background_token = nn.Parameter(torch.zeros(1, 1, self.decoder_pred.in_features))

    def _normalize(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape[2] != self.input_mean.shape[2]:
            raise ValueError(f"Expected {self.input_mean.shape[2]} channels, got {video.shape[2]}")
        return self._normalize_input(video)

    def _foreground(self, batch: int, device: torch.device) -> torch.Tensor:
        gt, gh, gw = self.token_grid
        roi = _OFFICIAL_ROI.to(device=device)
        if (gh, gw) != tuple(roi.shape):
            roi = F.interpolate(roi[None, None].float(), size=(gh, gw), mode="nearest")[0, 0].bool()
        return roi.flatten()[None, None].expand(batch, gt, -1).reshape(batch, -1)

    def _roi_mask(self, foreground: torch.Tensor) -> torch.Tensor:
        b, _ = foreground.shape
        gt, gh, gw = self.token_grid
        fg = foreground.reshape(b, gt, gh * gw)
        mask = torch.zeros_like(fg)
        for bi in range(b):
            for ti in range(gt):
                ids = fg[bi, ti].nonzero(as_tuple=False).flatten()
                count = len(ids) - max(1, int(len(ids) * (1 - self.mask_ratio)))
                chosen = ids[torch.randperm(len(ids), device=ids.device)[:count]]
                mask[bi, ti, chosen] = True
        return mask.reshape(b, -1)

    def _encode_visible(self, video: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(video)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        count = int(visible[0].sum())
        tokens = tokens[visible].reshape(video.shape[0], count, self.embed_dim)
        tokens = self.run_blocks(tokens, self.blocks)
        return self.norm(tokens)

    def _forward_view(
        self, video: torch.Tensor, mask: torch.Tensor, foreground: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self._normalize(video)
        visible = ~mask & foreground
        encoded = self._encode_visible(normalized, visible)
        decoded = self.decoder_embed(encoded)
        b, _, d = decoded.shape
        pos = self.decoder_pos_embed.to(device=decoded.device, dtype=decoded.dtype).expand(b, -1, -1)
        pos_vis = pos[visible].reshape(b, -1, d)
        pos_bg = pos[~foreground].reshape(b, -1, d)
        pos_mask = pos[mask].reshape(b, -1, d)
        full = torch.cat(
            (
                decoded + pos_vis,
                self.background_token.to(decoded).expand(b, pos_bg.shape[1], -1) + pos_bg,
                self.mask_token.to(decoded).expand(b, pos_mask.shape[1], -1) + pos_mask,
            ),
            dim=1,
        )
        full = self.run_blocks(full, self.decoder_blocks)
        pred = self.decoder_pred(self.decoder_norm(full[:, -pos_mask.shape[1] :]))
        target_video = _median_blur_video(normalized, self.median_blur_kernel)
        target = tubelet_patchify(target_video, self.tubelet_size, self.patch_size)
        target = target[mask].reshape(b, -1, target.shape[-1])
        loss = F.mse_loss(pred, target)
        return loss, pred, encoded.mean(dim=1)

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        return super().forward_features(video)

    def forward(
        self,
        video: torch.Tensor,
        video_view2: torch.Tensor | None = None,
        mask_ratio: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if mask_ratio is not None and float(mask_ratio) != self.mask_ratio:
            raise ValueError("EchoCardMAE uses its configured ROI mask ratio")
        if self.training and video_view2 is None:
            raise ValueError('EchoCardMAE training requires two independently sampled clips.')
        foreground = self._foreground(video.shape[0], video.device)
        mask = self._roi_mask(foreground)
        loss1, pred1, feat1 = self._forward_view(video, mask, foreground)
        if video_view2 is None:
            loss_rec = loss1
            loss_align = loss1.new_zeros(())
        else:
            loss2, _, feat2 = self._forward_view(video_view2, mask, foreground)
            loss_rec = 0.5 * (loss1 + loss2)
            loss_align = 0.5 * (
                temporal_infonce_loss(feat1, feat2, self.alignment_temperature)
                + temporal_infonce_loss(feat2, feat1, self.alignment_temperature)
            )
        loss = (1.0 - self.align_loss_weight) * loss_rec + self.align_loss_weight * loss_align
        return {
            "loss": loss,
            "loss_total": loss.detach(),
            "loss_recon": loss_rec.detach(),
            "loss_align": loss_align.detach(),
            "pred": pred1,
            "mask": mask,
            "tokens": feat1,
        }


def build_echocardmae_video(cfg: dict) -> EchoCardMAEVideo:
    return EchoCardMAEVideo(**cfg)
