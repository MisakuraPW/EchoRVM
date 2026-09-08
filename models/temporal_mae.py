"""Local VideoMAE with causal memory between non-overlapping clips."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rvm_core import RVMCore
from .video_mae import EchoVideoMAE, tubelet_patchify
from .vit_blocks import CrossBlock


def temporal_mask(batch, grid, ratio, strategy, device, clip_index=0, spatial_order=None):
    """All samples have the same mask budget; True means hidden."""
    t, h, w = grid
    spatial, total = h * w, t * h * w
    keep_spatial = max(1, int(spatial * (1 - ratio)))
    keep = t * keep_spatial
    if keep >= total:
        raise ValueError('Masking needs both visible and hidden tokens.')
    if strategy in {'tube', 'complementary'}:
        if spatial_order is None:
            spatial_order = torch.rand(batch, spatial, device=device).argsort(-1)
        order = spatial_order[:, None].expand(-1, t, -1)
        if strategy == 'complementary':
            order = torch.stack([
                spatial_order.roll((clip_index * t + ti) * keep_spatial, dims=-1)
                for ti in range(t)
            ], dim=1)
        mask = torch.ones(batch, t, spatial, device=device, dtype=torch.bool)
        mask.scatter_(2, order[:, :, :keep_spatial], False)
        return mask.flatten(1)
    if strategy == 'random_frame':
        order = torch.rand(batch, t, spatial, device=device).argsort(-1)
        mask = torch.ones(batch, t, spatial, device=device, dtype=torch.bool)
        mask.scatter_(2, order[:, :, :keep_spatial], False)
        return mask.flatten(1)
    if strategy == 'temporal_block':
        # A contiguous temporal interval is hidden; a partial boundary tubelet
        # gives exactly the same budget as tube masking.
        starts = torch.randint(keep + 1, (batch, 1), device=device) // spatial * spatial
        hidden_order = (torch.arange(total, device=device)[None] + starts) % total
        mask = torch.zeros(batch, total, device=device, dtype=torch.bool)
        mask.scatter_(1, hidden_order[:, :total - keep], True)
        return mask
    raise ValueError(f'Unknown temporal mask: {strategy}')


class TemporalMAE(EchoVideoMAE):
    """Variants none/global/spatial/dual share one local MAE and loss."""

    def __init__(self, **cfg):
        local = dict(cfg)
        self.local_frames = int(cfg.get('local_frames', 16))
        self.clip_count = int(cfg.get('clip_count', 4))
        local['frames'] = self.local_frames
        super().__init__(**local)
        self.frames = self.local_frames * self.clip_count
        self.core_type = 'temporal_mae'
        self.memory_mode = str(cfg.get('memory_mode', 'global'))
        self.memory_grid = int(cfg.get('memory_grid', 4))
        self.research_mask = str(cfg.get('research_mask', 'tube'))
        if self.memory_mode not in {'none', 'global', 'spatial', 'dual'}:
            raise ValueError(self.memory_mode)
        if self.memory_mode != 'none':
            self.memory = RVMCore(self.embed_dim, int(cfg.get('num_heads', 6)),
                                  int(cfg.get('core_depth', 1)))
            self.memory_to_decoder = nn.Linear(self.embed_dim, self.decoder_pred.in_features)
            self.memory_fusion = CrossBlock(self.decoder_pred.in_features,
                                           int(cfg.get('decoder_num_heads', 3)))
            self.feature_fusion = CrossBlock(self.embed_dim, int(cfg.get('num_heads', 6)))
        if self.memory_mode == 'dual':
            self.short_gate = nn.Parameter(torch.zeros(()))

    def _pool(self, encoded, mask, valid):
        b, _, dim = encoded.shape
        gt, gh, gw = self.token_grid
        dense = encoded.new_zeros(b, gt * gh * gw, dim)
        if mask is None:
            dense = encoded
            weights = valid.to(encoded.dtype)
        else:
            dense[~mask] = encoded.reshape(-1, dim)
            weights = (~mask).to(encoded.dtype) * valid
        dense = dense * weights[..., None]
        if self.memory_mode in {'global', 'none'}:
            return dense.sum(1, keepdim=True) / weights.sum(1)[:, None, None].clamp_min(1)
        dense = dense.reshape(b, gt, gh, gw, dim).sum(1).permute(0, 3, 1, 2)
        weights = weights.reshape(b, gt, gh, gw).sum(1)[:, None]
        size = (self.memory_grid, self.memory_grid)
        pooled = F.adaptive_avg_pool2d(dense, size) / F.adaptive_avg_pool2d(weights, size).clamp_min(1e-6)
        return pooled.flatten(2).transpose(1, 2)

    def _intervene(self, state, index, intervention, reset_interval):
        if state is None:
            return None
        if intervention == 'reset' or (reset_interval and index % reset_interval == 0):
            return None
        if intervention == 'shuffle':
            if state.shape[0] < 2:
                raise ValueError('Cross-patient shuffle requires batch size >= 2.')
            return state.roll(1, dims=0)
        if intervention != 'normal':
            raise ValueError(intervention)
        return state

    def _unroll(self, video, frame_valid=None, reconstruct=False,
                intervention='normal', reset_interval=0, masks=None):
        if tuple(video.shape[1:]) != (self.frames, self.in_chans, self.img_size, self.img_size):
            raise ValueError(f'TemporalMAE expects T={self.frames}, C={self.in_chans}, H=W={self.img_size}.')
        b = video.shape[0]
        if frame_valid is None:
            frame_valid = torch.ones(b, self.frames, dtype=torch.bool, device=video.device)
        state, short = None, None
        features, states, predictions, targets, used_masks = [], [], [], [], []
        loss_sum, weight_sum = video.new_zeros(()), video.new_zeros(())
        gt, gh, gw = self.token_grid
        spatial_order = torch.rand(b, gh * gw, device=video.device).argsort(-1) if reconstruct else None
        for i, clip in enumerate(video.split(self.local_frames, dim=1)):
            fv = frame_valid[:, i * self.local_frames:(i + 1) * self.local_frames]
            tv = fv.reshape(b, gt, self.tubelet_size).all(-1)
            valid = tv[:, :, None].expand(-1, -1, gh * gw).reshape(b, -1)
            state = self._intervene(state, i, intervention, reset_interval)
            short = self._intervene(short, i, intervention, reset_interval)
            mask = None
            if reconstruct:
                mask = masks[:, i] if masks is not None else temporal_mask(
                    b, self.token_grid, self.mask_ratio, self.research_mask,
                    video.device, i, spatial_order)
            encoded, _ = self.encode_video(clip, mask)
            previous = state
            if previous is not None and short is not None:
                previous = previous + self.short_gate.sigmoid() * short
            if previous is not None:
                encoded = self.feature_fusion(encoded, previous.to(encoded))
            pooled = self._pool(encoded, mask, valid)
            if reconstruct:
                decoded = self.decoder_embed(encoded)
                full = self.mask_token.to(decoded).expand(b, self.patch_embed.num_patches, -1).clone()
                full[~mask] = decoded.reshape(-1, decoded.shape[-1])
                full = full + self.decoder_pos_embed.to(full)
                if previous is not None:
                    full = self.memory_fusion(full, self.memory_to_decoder(previous).to(full))
                full = self.run_blocks(full, self.decoder_blocks)
                pred = self.decoder_pred(self.decoder_norm(full))
                target = tubelet_patchify(self._normalize_input(clip), self.tubelet_size, self.patch_size)
                if self.norm_pix_loss:
                    target = (target - target.mean(-1, keepdim=True)) / (
                        target.var(-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
                weights = (mask & valid).float()
                loss_sum = loss_sum + ((pred.float() - target.float()).square().mean(-1) * weights).sum()
                weight_sum = weight_sum + weights.sum()
                predictions.append(pred)
                targets.append(target)
                used_masks.append(mask & valid)
            if self.memory_mode != 'none':
                _, proposed = self.memory(pooled, state)
                old = torch.zeros_like(proposed) if state is None else state
                active = valid.any(-1)[:, None, None]
                state = torch.where(active, proposed, old)
                states.append(state.mean(1))
                if self.memory_mode == 'dual':
                    short = torch.where(active, pooled, torch.zeros_like(pooled))
            else:
                states.append(pooled.mean(1))
            if not reconstruct:
                dense = encoded.reshape(b, gt, gh, gw, self.embed_dim)
                features.append(dense.reshape(b, gt, gh * gw, self.embed_dim) * tv[:, :, None, None])
        result = {'states': torch.stack(states, 1)}
        if reconstruct:
            if not bool(weight_sum > 0):
                raise ValueError('No valid masked tubelets: video is too short for this configuration.')
            loss = loss_sum / weight_sum
            result.update(loss=loss, loss_recon=loss.detach(), pred=torch.cat(predictions, 1),
                          target=torch.cat(targets, 1), mask=torch.cat(used_masks, 1))
        else:
            result['features'] = torch.cat(features, 1)
        return result

    def forward(self, video, frame_valid=None, masks=None):
        return self._unroll(video, frame_valid, reconstruct=True, masks=masks)

    def forward_features(self, video, frame_valid=None, intervention='normal', reset_interval=0):
        return self._unroll(video, frame_valid, intervention=intervention,
                            reset_interval=reset_interval)['features']

    def state_trajectory(self, video, frame_valid=None, intervention='normal', reset_interval=0):
        return self._unroll(video, frame_valid, intervention=intervention, reset_interval=reset_interval)
