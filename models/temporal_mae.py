"""Local VideoMAE with causal memory between non-overlapping clips."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rvm_core import RVMCore
from .video_mae import EchoVideoMAE, tubelet_patchify
from .vit_blocks import CrossBlock
from .frequency import FrequencyMemoryGates, band_descriptor, multiband_error


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
        self.memory_compression = str(cfg.get('memory_compression', 'mean'))
        self.memory_write_source = str(cfg.get('memory_write_source', 'fused'))
        if self.memory_compression not in {'mean', 'temporal_attention'}:
            raise ValueError('Unknown memory_compression')
        if self.memory_write_source not in {'fused', 'local'}:
            raise ValueError('Unknown memory_write_source')
        if self.memory_compression == 'temporal_attention':
            if self.memory_mode not in {'spatial', 'dual'}:
                raise ValueError('Temporal attention compression requires spatial state slots')
            from .video_mae import flat_sinusoid
            self.register_buffer('compression_time_embed', flat_sinusoid(self.embed_dim, self.token_grid[0]), persistent=False)
            self.compression_score = nn.Sequential(nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, max(8,self.embed_dim//4)), nn.GELU(),
                nn.Linear(max(8,self.embed_dim//4), 1))
            # Start at uniform masked averaging; change compression only as it learns.
            nn.init.zeros_(self.compression_score[-1].weight)
            nn.init.zeros_(self.compression_score[-1].bias)
        self.frequency_conditioned = bool(cfg.get('frequency_conditioned', False))
        self.frequency_loss_weight = float(cfg.get('frequency_loss_weight', 0.))
        if (self.frequency_conditioned or self.frequency_loss_weight) and (
                self.memory_compression != 'mean' or self.memory_write_source != 'fused'):
            raise ValueError('Stage-three mechanism candidates exclude frequency branches')
        if self.frequency_loss_weight < 0:
            raise ValueError('frequency_loss_weight must be nonnegative')
        if self.frequency_conditioned or self.frequency_loss_weight:
            if self.patch_size % 4 or self.norm_pix_loss:
                raise ValueError('Two-level local Haar requires patch_size divisible by 4 and norm_pix_loss=false')
        if self.frequency_conditioned:
            if self.memory_mode == 'none':
                raise ValueError('Frequency-conditioned memory requires an active memory core')
            self.frequency_gates = FrequencyMemoryGates(self.embed_dim)
        self.frame_readout = str(cfg.get('frame_readout', 'repeat'))
        self.patch_init_temporal_sum = bool(cfg.get('patch_init_temporal_sum', False))
        if self.frame_readout not in {'repeat', 'learned'}:
            raise ValueError('frame_readout must be repeat or learned')
        if self.frame_readout == 'learned':
            if self.tubelet_size != 2 or self.norm_pix_loss or self.frequency_loss_weight:
                raise ValueError('Learned frame expansion requires tubelet2, raw targets, no frequency loss')
            from .frame_readout import FrameExpansion
            self.frame_expansion = FrameExpansion(self.embed_dim, self.tubelet_size)
            self.decoder_pred = nn.Linear(self.decoder_pred.in_features,
                                          self.patch_size**2 * self.in_chans)
            nn.init.xavier_uniform_(self.decoder_pred.weight)
            nn.init.zeros_(self.decoder_pred.bias)
            from .video_mae import flat_sinusoid, get_3d_sincos_pos_embed
            _, gh, gw = self.token_grid
            dim = self.decoder_pred.in_features
            pos = (flat_sinusoid(dim, self.local_frames * gh * gw)
                   if cfg.get('position_embedding') == 'flat_sinusoid' else
                   get_3d_sincos_pos_embed(dim, self.local_frames, gh, gw))
            self.register_buffer('frame_decoder_pos_embed', pos, persistent=False)

    def frame_features(self, features):
        """[B,tubelets,patches,D] -> [B,frames,patches,D], shared with MAE."""
        if self.frame_readout == 'learned':
            return self.frame_expansion(features)
        return features.repeat_interleave(self.tubelet_size, dim=1)

    def _decode(self, encoded, mask, previous):
        b = encoded.shape[0]
        if self.frame_readout == 'learned':
            # Expand only visible encoder tokens. Hidden pixels have no shortcut.
            gt, gh, gw = self.token_grid
            dense = encoded.new_zeros(b, gt * gh * gw, self.embed_dim)
            dense[~mask] = encoded.reshape(-1, self.embed_dim)
            expanded = self.frame_features(dense.reshape(b, gt, gh * gw, -1)).flatten(1, 2)
            frame_mask = mask.reshape(b, gt, gh * gw).repeat_interleave(self.tubelet_size, 1).flatten(1)
            decoded = self.decoder_embed(expanded[~frame_mask].reshape(b, -1, self.embed_dim))
            full = self.mask_token.to(decoded).expand(b, frame_mask.shape[1], -1).clone()
            full[~frame_mask] = decoded.reshape(-1, decoded.shape[-1])
            pos = self.frame_decoder_pos_embed.to(full)
        else:
            decoded = self.decoder_embed(encoded)
            full = self.mask_token.to(decoded).expand(b, self.patch_embed.num_patches, -1).clone()
            full[~mask] = decoded.reshape(-1, decoded.shape[-1])
            pos = self.decoder_pos_embed.to(full)
        full = full + pos
        if previous is not None:
            full = self.memory_fusion(full, self.memory_to_decoder(previous).to(full))
        full = self.run_blocks(full, self.decoder_blocks)
        pred = self.decoder_pred(self.decoder_norm(full))
        if self.frame_readout == 'learned':
            # Restore original tubelet_patchify order: (time, spatial, offset, pixels).
            pred = pred.reshape(b, gt, self.tubelet_size, gh * gw, -1)
            pred = pred.permute(0, 1, 3, 2, 4).reshape(b, gt * gh * gw, -1)
        return pred

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
        dense = dense.reshape(b, gt, gh, gw, dim)
        if self.memory_compression == 'temporal_attention':
            validity = weights.reshape(b, gt, gh, gw)
            position = self.compression_time_embed.to(dense)[:, :, None, None, :]
            scores = self.compression_score(dense + position).squeeze(-1).float()
            scores = scores.masked_fill(validity == 0, -1e4)
            # Counts retain baseline spatial weighting, including empty cells.
            coefficients = scores.softmax(1) * validity.sum(1,keepdim=True)
            dense = dense * coefficients.to(dense)[...,None]
        dense = dense.sum(1).permute(0, 3, 1, 2)
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
                intervention='normal', reset_interval=0, masks=None, return_local=False,
                initial_state=None, initial_short=None, streaming=False):
        frames = video.shape[1]
        expected_frames = frames if streaming else self.frames
        if (frames < 1 or frames % self.local_frames or
                tuple(video.shape[1:]) != (expected_frames, self.in_chans, self.img_size, self.img_size)):
            raise ValueError(f'TemporalMAE expects T={self.frames}, C={self.in_chans}, H=W={self.img_size}.')
        b = video.shape[0]
        if frame_valid is None:
            frame_valid = torch.ones(b, frames, dtype=torch.bool, device=video.device)
        if frame_valid.shape != (b, frames):
            raise ValueError('frame_valid must match [B,T]')
        state, short = initial_state, initial_short
        features, states, predictions, targets, used_masks = [], [], [], [], []
        local_features = []
        loss_sum, weight_sum = video.new_zeros(()), video.new_zeros(())
        frequency_sum = video.new_zeros(())
        read_gates, write_gates = [], []
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
            local_encoded = encoded
            if return_local:
                if reconstruct:
                    raise ValueError('Local diagnostics require unmasked inference.')
                local_features.append(encoded.reshape(b, gt, gh * gw, self.embed_dim)
                                      * tv[:, :, None, None])
            descriptor = None
            if self.frequency_conditioned:
                with torch.no_grad():
                    raw_patches = tubelet_patchify(clip, self.tubelet_size, self.patch_size)
                    # Select first: hidden pixels never enter the conditioning path.
                    selected = raw_patches if mask is None else raw_patches[~mask].reshape(b, -1, raw_patches.shape[-1])
                    descriptor = band_descriptor(selected, self.tubelet_size, self.patch_size, self.in_chans)
            previous = state
            if previous is not None and short is not None:
                previous = previous + self.short_gate.sigmoid() * short
            if previous is not None:
                fused = self.feature_fusion(encoded, previous.to(encoded))
                if self.frequency_conditioned:
                    encoded, gate = self.frequency_gates.read(encoded, fused, descriptor, previous)
                    read_gates.append(gate)
                else:
                    encoded = fused
            # Local-write isolates new evidence from the recurrent read feedback.
            write_encoded = local_encoded if self.memory_write_source == 'local' else encoded
            pooled = self._pool(write_encoded, mask, valid)
            if reconstruct:
                pred = self._decode(encoded, mask, previous)
                target = tubelet_patchify(self._normalize_input(clip), self.tubelet_size, self.patch_size)
                if self.norm_pix_loss:
                    target = (target - target.mean(-1, keepdim=True)) / (
                        target.var(-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
                weights = (mask & valid).float()
                loss_sum = loss_sum + ((pred.float() - target.float()).square().mean(-1) * weights).sum()
                if self.frequency_loss_weight:
                    selected_mask = mask & valid
                    error = multiband_error(pred[selected_mask], target[selected_mask].detach(),
                                            self.tubelet_size, self.patch_size, self.in_chans)
                    frequency_sum = frequency_sum + error.sum()
                weight_sum = weight_sum + weights.sum()
                predictions.append(pred)
                targets.append(target)
                used_masks.append(mask & valid)
            if self.memory_mode != 'none':
                _, proposed = self.memory(pooled, state)
                old = torch.zeros_like(proposed) if state is None else state
                if self.frequency_conditioned:
                    pooled_descriptor = self._pool(descriptor, mask, valid)
                    proposed, gate = self.frequency_gates.write(old, proposed, pooled_descriptor)
                    write_gates.append(gate)
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
        if streaming:
            result.update(final_state=state, final_short=short)
        if reconstruct:
            if not bool(weight_sum > 0):
                raise ValueError('No valid masked tubelets: video is too short for this configuration.')
            pixel_loss = loss_sum / weight_sum
            frequency_loss = frequency_sum / weight_sum
            loss = pixel_loss + self.frequency_loss_weight * frequency_loss
            result.update(loss=loss, loss_recon=pixel_loss.detach(),
                          loss_frequency=frequency_loss.detach(),
                          loss_frequency_weighted=(self.frequency_loss_weight * frequency_loss).detach(),
                          pred=torch.cat(predictions, 1),
                          target=torch.cat(targets, 1), mask=torch.cat(used_masks, 1))
            if write_gates:
                result['frequency_write_gate'] = torch.stack(write_gates).mean()
                result['frequency_read_gate'] = torch.stack(read_gates).mean() if read_gates else loss.new_zeros(())
        else:
            result['features'] = torch.cat(features, 1)
            if return_local:
                result['local_features'] = torch.cat(local_features, 1)
        return result

    def forward(self, video, frame_valid=None, masks=None):
        return self._unroll(video, frame_valid, reconstruct=True, masks=masks)

    def forward_features(self, video, frame_valid=None, intervention='normal', reset_interval=0):
        return self._unroll(video, frame_valid, intervention=intervention,
                            reset_interval=reset_interval)['features']

    def state_trajectory(self, video, frame_valid=None, intervention='normal', reset_interval=0):
        return self._unroll(video, frame_valid, intervention=intervention, reset_interval=reset_interval)

    def diagnostic_features(self, video, frame_valid=None):
        """Paired local/fused exits from one normal unroll; no new parameters."""
        return self._unroll(video, frame_valid, return_local=True)

    def stream_clip(self, video, state=None, short_state=None, frame_valid=None):
        """Process ONE complete local clip; caller owns patient reset and detachment.

        Attention within the clip is bidirectional. Outputs become available at
        clip end, not at each incoming frame. This does not truncate BPTT.
        """
        if video.shape[1] != self.local_frames:
            raise ValueError('stream_clip requires exactly local_frames frames')
        if self.memory_mode == 'none' and (state is not None or short_state is not None):
            raise ValueError('A no-memory model cannot accept recurrent state')
        if short_state is not None and self.memory_mode != 'dual':
            raise ValueError('short_state is only valid for dual memory')
        for value in (state, short_state):
            if value is not None:
                slots = 1 if self.memory_mode == 'global' else self.memory_grid**2
                if value.shape != (len(video), slots, self.embed_dim) or value.device != video.device:
                    raise ValueError('State batch, slots, width or device mismatch')
        return self._unroll(video, frame_valid, return_local=True, streaming=True,
                            initial_state=state, initial_short=short_state)
