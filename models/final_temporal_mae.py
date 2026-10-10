"""Bounded final-study mechanisms, with backward-compatible legacy weights."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as recompute

from .temporal_mae import TemporalMAE, temporal_mask
from .rvm_core import RVMCore
from .video_mae import tubelet_patchify


class CandidateLowRankRVM(RVMCore):
    """Only the normalized candidate is restricted; the gated state is not."""

    def __init__(self, dim, heads, depth, rank):
        super().__init__(dim, heads, depth)
        if not 1 <= rank <= dim:
            raise ValueError('Candidate rank must lie in [1, embed_dim]')
        self.candidate_down = nn.Linear(dim, rank, bias=False)
        self.candidate_up = nn.Linear(rank, dim, bias=False)
        nn.init.orthogonal_(self.candidate_up.weight)
        with torch.no_grad():
            self.candidate_down.weight.copy_(self.candidate_up.weight.T)

    def forward(self, tokens, state=None):
        state = self.init_state(tokens) if state is None else state
        update = torch.sigmoid(self.update_x(tokens) + self.update_s(state))
        reset = torch.sigmoid(self.reset_x(tokens) + self.reset_s(state))
        candidate = tokens
        for block in self.integration:
            candidate = block(candidate, reset * state)
        candidate = self.candidate_up(self.candidate_down(self.norm(candidate)))
        result = (1 - update) * state + update * candidate
        return result, result


class FinalTemporalMAE(TemporalMAE):
    """L16 streaming with independently specified history and FIFO capacity.

    Native token exits are preserved. Explicit frame exits avoid expanding a
    late-read representation twice or pretending it is a tubelet tensor.
    """

    def __init__(self, **config):
        cfg = dict(config)
        if cfg.get('frequency_conditioned') or cfg.get('frequency_loss_weight', 0):
            raise ValueError('Final temporal study excludes frequency branches')
        mode = str(cfg.get('memory_mode', 'global'))
        readout = str(cfg.get('frame_readout', 'repeat'))
        if mode not in {'global', 'spatial', 'spatial_global', 'none'}:
            raise ValueError('Final study excludes dual memory')
        if readout not in {'repeat', 'learned', 'factorized', 'shrink', 'soft_factorized'}:
            raise ValueError('Unsupported final-study frame readout')
        # The parent supplies identical shared parameter names/initialization.
        parent = dict(cfg, memory_mode='spatial' if mode != 'none' else 'none',
                      frame_readout='factorized' if readout == 'soft_factorized' else
                      'learned' if readout == 'shrink' else readout)
        super().__init__(**parent)
        self.memory_mode, self.frame_readout = mode, readout
        self.memory_read_location = cfg.get('memory_read_location', 'tokens')
        self.allow_variable_context = bool(cfg.get('allow_variable_context', True))
        self.reconstruction_recent_frames = int(cfg.get('reconstruction_recent_frames', self.frames))
        self.soft_beta = float(cfg.get('soft_beta', .5))
        if self.memory_read_location not in {'tokens', 'frames'}:
            raise ValueError('Unknown memory read location')
        if self.memory_read_location == 'frames' and (
                readout == 'repeat' or self.memory_write_source != 'local'):
            raise ValueError('Late reading requires learned frames and local-only writing')
        if self.reconstruction_recent_frames < 1 or self.reconstruction_recent_frames % self.local_frames:
            raise ValueError('Reconstruction suffix must contain complete clips')
        if not 0 <= self.soft_beta <= 1:
            raise ValueError('soft_beta must lie in [0,1]')
        if mode == 'spatial_global':
            self.memory_type = nn.Parameter(torch.zeros(2, self.embed_dim))
        if readout == 'shrink':
            self.frame_gamma = nn.Parameter(torch.ones(self.embed_dim))
        rank = int(cfg.get('candidate_rank', 0))
        if rank:
            if mode == 'none':
                raise ValueError('Candidate restriction requires memory')
            with torch.random.fork_rng(devices=[]):
                core = CandidateLowRankRVM(self.embed_dim, int(cfg.get('num_heads', 6)),
                                          int(cfg.get('core_depth', 1)), rank)
            core.load_state_dict(self.memory.state_dict(), strict=False)
            self.memory = core

    @property
    def memory_slots(self):
        if self.memory_mode == 'none':
            return 0
        return 1 if self.memory_mode == 'global' else self.memory_grid ** 2 + int(self.memory_mode == 'spatial_global')

    def _call(self, module, *values):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return recompute(module, *values, use_reentrant=False)
        return module(*values)

    def frame_base_features(self, tokens, valid=None):
        if self.frame_readout == 'repeat':
            return tokens.repeat_interleave(self.tubelet_size, 1)
        layer = self.frame_expansion.base if hasattr(self.frame_expansion, 'base') else self.frame_expansion
        groups = self.token_grid[0]
        chunks = tokens.split(groups, 1)
        if any(x.shape[1] != groups for x in chunks):
            raise ValueError('Frame expansion requires complete local clips')
        weights = [None] * len(chunks) if valid is None else valid.split(groups, 1)
        return torch.cat([layer(x, w) for x, w in zip(chunks, weights)], 1)

    def frame_transform(self, expanded, valid=None):
        if self.frame_readout in {'repeat', 'learned'}:
            return expanded
        outputs = []
        chunks = expanded.split(self.local_frames, 1)
        masks = [None] * len(chunks) if valid is None else valid.split(self.local_frames, 1)
        for value, mask in zip(chunks, masks):
            weights = torch.ones_like(value[..., :1]) if mask is None else mask.to(value)[..., None]
            mean = (value * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1)
            if self.frame_readout == 'shrink':
                result = mean + self.frame_gamma.to(value) * (value - mean)
            else:
                layer = self.frame_expansion
                coefficients = layer.coefficients(value)
                average = (coefficients * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1)
                result = mean + F.linear(coefficients - average, layer.basis)
                if self.frame_readout == 'soft_factorized':
                    result = value + self.soft_beta * (result - value)
            outputs.append(result * weights)
        return torch.cat(outputs, 1)

    def frame_features(self, features, valid=None):
        base = self.frame_base_features(features, valid)
        weight = None if valid is None else valid.repeat_interleave(self.tubelet_size, 1)
        return self.frame_transform(base, weight)

    def _pool(self, encoded, mask, valid):
        b, _, d = encoded.shape
        gt, gh, gw = self.token_grid
        dense = encoded.new_zeros(b, gt * gh * gw, d)
        if mask is None:
            dense, weights = encoded, valid.to(encoded)
        else:
            dense[~mask] = encoded.reshape(-1, d)
            weights = (~mask).to(encoded) * valid
        dense = (dense * weights[..., None]).reshape(b, gt, gh, gw, d)
        if self.memory_compression == 'temporal_attention':
            validity = weights.reshape(b, gt, gh, gw)
            position = self.compression_time_embed.to(dense)[:, :, None, None]
            score = self.compression_score(dense + position).squeeze(-1).float()
            coefficient = score.masked_fill(validity == 0, -1e4).softmax(1) * validity.sum(1, keepdim=True)
            dense = dense * coefficient.to(dense)[..., None]
        dense = dense.sum(1).permute(0, 3, 1, 2)
        weights = weights.reshape(b, gt, gh, gw).sum(1)[:, None]
        def pool(size):
            value = F.adaptive_avg_pool2d(dense, size) / F.adaptive_avg_pool2d(weights, size).clamp_min(1e-6)
            return value.flatten(2).transpose(1, 2)
        if self.memory_mode in {'global', 'none'}:
            return pool((1, 1))
        spatial = pool((self.memory_grid, self.memory_grid))
        if self.memory_mode == 'spatial_global':
            values = torch.cat((pool((1, 1)), spatial), 1)
            types = torch.cat((self.memory_type[:1], self.memory_type[1:].expand(spatial.shape[1], -1)))
            return values + types.to(values)[None]
        return spatial

    def _frame_exits(self, tokens, previous=None, visible=None):
        base = self.frame_base_features(tokens, visible)
        weights = None if visible is None else visible.repeat_interleave(self.tubelet_size, 1)
        if self.memory_read_location == 'frames' and previous is not None:
            if weights is None:
                base = self._call(self.feature_fusion, base.flatten(1, 2), previous.to(base)).reshape_as(base)
            else:
                # Hidden locations must never participate in encoder-side reading.
                flat = base.flatten(1, 2)
                active = weights.flatten(1).bool()
                selected = flat[active].reshape(len(base), -1, self.embed_dim)
                selected = self._call(self.feature_fusion, selected, previous.to(selected))
                dense = torch.zeros_like(flat)
                dense[active] = selected.reshape(-1, self.embed_dim)
                base = dense.reshape_as(base)
        return base, self.frame_transform(base, weights)

    def _decode(self, encoded, mask, previous):
        if self.frame_readout == 'repeat':
            return super()._decode(encoded, mask, previous)
        b = len(encoded)
        gt, gh, gw = self.token_grid
        dense = encoded.new_zeros(b, gt * gh * gw, self.embed_dim)
        dense[~mask] = encoded.reshape(-1, self.embed_dim)
        _, expanded = self._frame_exits(dense.reshape(b, gt, gh * gw, -1), previous,
                                         (~mask).reshape(b, gt, gh * gw))
        frame_mask = mask.reshape(b, gt, gh * gw).repeat_interleave(self.tubelet_size, 1).flatten(1)
        decoded = self.decoder_embed(expanded.flatten(1, 2)[~frame_mask].reshape(b, -1, self.embed_dim))
        full = self.mask_token.to(decoded).expand(b, frame_mask.shape[1], -1).clone()
        full[~frame_mask] = decoded.reshape(-1, decoded.shape[-1])
        full = full + self.frame_decoder_pos_embed.to(full)
        if previous is not None:
            full = self._call(self.memory_fusion, full, self.memory_to_decoder(previous).to(full))
        full = self.run_blocks(full, self.decoder_blocks)
        pred = self.decoder_pred(self.decoder_norm(full)).reshape(b, gt, self.tubelet_size, gh * gw, -1)
        return pred.permute(0, 1, 3, 2, 4).reshape(b, gt * gh * gw, -1)

    def _unroll(self, video, frame_valid=None, reconstruct=False, intervention='normal', reset_interval=0,
                masks=None, return_local=False, initial_state=None, initial_short=None, streaming=False):
        b, frames, c, h, w = video.shape
        if frames < self.local_frames or frames % self.local_frames or (c, h, w) != (self.in_chans, self.img_size, self.img_size):
            raise ValueError('Expected real complete clips [B,T,C,H,W] at native resolution')
        if not (streaming or self.allow_variable_context) and frames != self.frames:
            raise ValueError('Unexpected native context length')
        if initial_short is not None:
            raise ValueError('Final study has no dual short state')
        if frame_valid is None:
            frame_valid = torch.ones(b, frames, dtype=torch.bool, device=video.device)
        if frame_valid.shape != (b, frames):
            raise ValueError('Invalid frame validity shape')
        if reconstruct and not bool(frame_valid.all()):
            raise ValueError('Final-study MAE never pads fake history')
        state = initial_state
        if state is not None and state.shape != (b, self.memory_slots, self.embed_dim):
            raise ValueError('State shape does not match registered memory organization')
        gt, gh, gw = self.token_grid
        order = torch.rand(b, gh * gw, device=video.device).argsort(-1) if reconstruct and masks is None else None
        suffix = min(frames, self.reconstruction_recent_frames)
        pred_all, target_all, mask_all, features, local, states = [], [], [], [], [], []
        frame_final, frame_base, local_base, local_final = [], [], [], []
        loss_sum, weight_sum = video.new_zeros(()), video.new_zeros(())
        for i, clip in enumerate(video.split(self.local_frames, 1)):
            fv = frame_valid[:, i * self.local_frames:(i + 1) * self.local_frames]
            tv = fv.reshape(b, gt, self.tubelet_size).all(-1)
            valid = tv[:, :, None].expand(-1, -1, gh * gw).flatten(1)
            state = self._intervene(state, i, intervention, reset_interval)
            previous = state
            mask = (masks[:, i] if masks is not None else temporal_mask(b, self.token_grid, self.mask_ratio,
                     self.research_mask, video.device, i, order)) if reconstruct else None
            encoded, _ = self.encode_video(clip, mask)
            raw = encoded
            if previous is not None and self.memory_read_location == 'tokens':
                encoded = self._call(self.feature_fusion, encoded, previous.to(encoded))
            pooled = self._pool(raw if self.memory_write_source == 'local' else encoded, mask, valid)
            if reconstruct and (i + 1) * self.local_frames > frames - suffix:
                pred = self._decode(encoded, mask, previous)
                target = tubelet_patchify(self._normalize_input(clip), self.tubelet_size, self.patch_size)
                if self.norm_pix_loss:
                    target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True, unbiased=False) + 1e-6).sqrt()
                weight = (mask & valid).float()
                loss_sum = loss_sum + ((pred.float() - target.float()).square().mean(-1) * weight).sum()
                weight_sum = weight_sum + weight.sum()
                pred_all.append(pred); target_all.append(target); mask_all.append(mask & valid)
            if self.memory_mode != 'none':
                _, proposed = self._call(self.memory, pooled, state)
                old = torch.zeros_like(proposed) if state is None else state
                state = torch.where(valid.any(-1)[:, None, None], proposed, old)
                states.append(state.mean(1))
            else:
                states.append(pooled.mean(1))
            if not reconstruct:
                native = encoded.reshape(b, gt, gh * gw, -1) * tv[:, :, None, None]
                features.append(native)
                if return_local:
                    raw = raw.reshape_as(native) * tv[:, :, None, None]
                    visible = valid.reshape(b, gt, gh * gw)
                    base, final = self._frame_exits(native, previous, visible)
                    lb = self.frame_base_features(raw, visible)
                    frame_base.append(base); frame_final.append(final)
                    local_base.append(lb); local_final.append(self.frame_features(raw, visible)); local.append(raw)
        result = dict(states=torch.stack(states, 1))
        if streaming:
            result.update(final_state=state, final_short=None)
        if reconstruct:
            if not bool(weight_sum > 0):
                raise ValueError('No masked reconstruction targets')
            pixel = loss_sum / weight_sum
            orth = self.frame_expansion.orthogonal_loss() if self.frame_readout in {'factorized', 'soft_factorized'} else pixel.new_zeros(())
            result.update(loss=pixel + self.dynamic_orthogonal_weight * orth, loss_recon=pixel.detach(),
                          loss_dynamic_orthogonal=orth.detach(), pred=torch.cat(pred_all, 1),
                          target=torch.cat(target_all, 1), mask=torch.cat(mask_all, 1))
        else:
            result['features'] = torch.cat(features, 1)
            if return_local:
                result.update(local_features=torch.cat(local, 1), frame_outputs=torch.cat(frame_final, 1),
                              frame_base_outputs=torch.cat(frame_base, 1), local_base_outputs=torch.cat(local_base, 1),
                              local_frame_outputs=torch.cat(local_final, 1))
        return result

    def stream_clip(self, video, state=None, short_state=None, frame_valid=None):
        if video.shape[1] != self.local_frames:
            raise ValueError('stream_clip needs one complete local clip')
        return self._unroll(video, frame_valid, return_local=True, streaming=True,
                            initial_state=state, initial_short=short_state)


def checkpoint_payload(path):
    value = torch.load(Path(path), map_location='cpu', weights_only=False)
    config = copy.deepcopy(value.get('config', {}))
    state = value.get('model_state_dict', value.get('model', value.get('state_dict')))
    if not isinstance(state, dict) or not isinstance(config.get('model'), dict):
        raise ValueError('A final-study source needs model weights and saved scientific config')
    for prefix in ('module.', '_orig_mod.'):
        while state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]:tensor for key,tensor in state.items()}
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise ValueError('Source state dictionary contains non-tensor entries')
    # Old saved configs omit these fields and rely on TemporalMAE defaults.
    # Resolve those same defaults, never substitute a different frame module.
    config['model'].setdefault('frame_readout', 'repeat')
    config['model'].setdefault('memory_mode', 'global')
    return value, config, state


def load_final_model(path, overrides=None, seed=42):
    value, config, state = checkpoint_payload(path)
    cfg = config['model']
    cfg.update(overrides or {})
    cfg.update(name='temporal_final', allow_variable_context=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FinalTemporalMAE(**cfg)
    target = model.state_dict()
    loaded, discarded = {}, []
    for key, tensor in state.items():
        alternate = key.replace('frame_expansion.projections.', 'frame_expansion.base.projections.')
        if key not in target and alternate in target:
            key = alternate
        if key not in target or target[key].shape != tensor.shape:
            discarded.append(key)
        else:
            loaded[key] = tensor
    missing = set(target) - set(loaded)
    permitted = ('memory_type', 'frame_gamma', 'frame_expansion.basis', 'frame_expansion.coefficients.',
                 'memory.candidate_down.', 'memory.candidate_up.')
    bad_missing = [k for k in missing if not k.startswith(permitted)]
    if discarded or bad_missing:
        raise ValueError(f'Unexpected warm-start mismatch: missing={bad_missing}, discarded={discarded}')
    model.load_state_dict(loaded, strict=False)
    return model, config, dict(source_epoch=value.get('epoch'), loaded_tensors=len(loaded),
                                initialized_keys=sorted(missing), discarded_keys=discarded)
