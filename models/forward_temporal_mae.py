"""Two single-change forward candidates; no combined variant or layer search."""

import torch
from torch import nn

from .final_temporal_mae import FinalTemporalMAE
from .video_mae import flat_sinusoid


class TemporalWriter(nn.Module):
    """Process ordered time groups before compressing to the original single S."""

    def __init__(self, dim, time, width=96):
        super().__init__()
        self.time = time
        self.project = nn.Linear(dim, width)
        self.norm = nn.LayerNorm(width)
        self.relation = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                                 nn.GELU(), nn.Linear(2*width, width))
        self.out = nn.Linear(width, dim, bias=False)
        nn.init.zeros_(self.out.weight)
        self.register_buffer('position', flat_sinusoid(width, time), persistent=False)

    def forward(self, encoded, mask, valid, group_order=None):
        b, n, d = encoded.shape
        if mask is not None:
            # Encoder emits visible indices in native flattened order. Tube
            # masking guarantees equal spatial counts for all time groups.
            valid = valid[~mask].reshape(b, n)
        groups = encoded.reshape(b, self.time, -1, d)
        if valid is None:
            base = encoded.mean(1, keepdim=True)
            group_mean = groups.float().mean(2)
            counts = torch.full((b,self.time),groups.shape[2],device=encoded.device)
        else:
            valid = valid.reshape(b, self.time, -1)
            safe = groups.masked_fill(~valid[..., None], 0)
            counts = valid.sum(-1)
            group_mean = safe.float().sum(2)/counts.clamp_min(1)[..., None]
            base = safe.reshape(b,n,d).float().sum(1,keepdim=True)/counts.sum(1).clamp_min(1)[:,None,None]
        total = counts.sum(1, keepdim=True).clamp_min(1)
        if group_order is not None:
            group_mean, counts = group_mean[:, group_order], counts[:, group_order]
        # Allow one zero key for all-invalid rows to avoid an all-masked softmax.
        missing = counts == 0
        missing = torch.cat((missing[:, :1] & ~missing.all(1, keepdim=True), missing[:, 1:]), 1)
        x = self.project(group_mean.to(encoded))
        z = self.norm(x + self.position.to(x))
        x = x + self.relation(z, z, z, key_padding_mask=missing, need_weights=False)[0]
        x = x + self.ffn(x)
        delta = (x.float()*counts[..., None]).sum(1, keepdim=True)/total[..., None]
        return base.to(encoded) + self.out(delta.to(encoded)).to(encoded)


class ContentWriter(nn.Module):
    def __init__(self, dim, width=64):
        super().__init__()
        self.feature_norm = nn.LayerNorm(dim)
        self.state_norm = nn.LayerNorm(dim)
        self.feature = nn.Linear(dim, width)
        self.state = nn.Linear(dim, width, bias=False)
        self.score = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, features, previous=None, visible=None):
        if visible is None:
            visible = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
        if not bool(visible.any(1).all()):
            raise ValueError('Writer requires at least one valid visible token per sample')
        # Mask BEFORE projection as well: invalid values cannot affect statistics or gradients.
        safe = features.masked_fill(~visible[..., None], 0)
        state = safe.new_zeros(len(safe), 1, safe.shape[-1]) if previous is None else previous
        hidden = self.feature(self.feature_norm(safe)) + self.state(self.state_norm(state))
        logits = self.score(hidden.tanh()).squeeze(-1).float().masked_fill(~visible, -torch.inf)
        alpha = logits.softmax(1).to(safe)
        return (safe*alpha[..., None]).sum(1, keepdim=True)


class ForwardTemporalMAE(FinalTemporalMAE):
    def __init__(self, **config):
        super().__init__(**config)
        self.forward_variant = config['forward_variant']
        self.read_after_block = int(config.get('read_after_block', 6))
        if self.forward_variant not in {'mid_read', 'content_write', 'mid_temporal_write'}:
            raise ValueError('Unknown fixed-length recurrence variant')
        if (self.memory_mode != 'global' or self.frame_readout != 'repeat' or
                self.memory_read_location != 'tokens' or self.memory_write_source != 'fused' or
                self.memory_compression != 'mean' or config.get('candidate_rank', 0)):
            raise ValueError('Forward candidates require the clean single-global-state baseline')
        if self.forward_variant != 'content_write' and not 0 < self.read_after_block < len(self.blocks):
            raise ValueError('Read insertion must precede at least one remaining encoder block')
        if self.forward_variant == 'content_write':
            with torch.random.fork_rng(devices=[]):
                self.content_writer = ContentWriter(self.embed_dim, int(config.get('writer_width', 64)))
        if self.forward_variant == 'mid_temporal_write':
            if self.research_mask != 'tube':
                raise ValueError('Temporal writer requires equal per-time tube masking')
            with torch.random.fork_rng(devices=[]):
                self.temporal_writer = TemporalWriter(self.embed_dim, self.token_grid[0],
                                                      int(config.get('temporal_writer_width', 96)))
        self.asymmetric_mask = bool(config.get('asymmetric_mask', False))
        if self.asymmetric_mask:
            keep = super().visible_spatial_tokens(0)
            delta = keep//2
            if (self.forward_variant != 'mid_read' or self.research_mask != 'tube' or
                    delta < 1 or keep+delta >= self.token_grid[1]*self.token_grid[2]):
                raise ValueError('Asymmetric masking requires an A-R tube mask with two valid budgets')

    def visible_spatial_tokens(self, clip_index, *, asymmetric=True):
        keep = super().visible_spatial_tokens(clip_index)
        if asymmetric and self.asymmetric_mask:
            return keep + (keep//2 if clip_index % 2 == 0 else -(keep//2))
        return keep

    def encode_current(self, clip, mask, previous, visible_indices=None):
        if self.forward_variant == 'content_write':
            return super().encode_current(clip, mask, previous, visible_indices=visible_indices)
        tokens = self.patch_embed(self._normalize_input(clip))
        tokens = tokens+self.pos_embed.to(tokens)
        if visible_indices is not None:
            tokens = tokens.gather(1, visible_indices[..., None].expand(-1, -1, self.embed_dim))
        elif mask is not None:
            tokens = tokens[~mask].reshape(len(clip), -1, self.embed_dim)
        tokens = self.run_blocks(tokens, self.blocks[:self.read_after_block])
        if previous is not None:
            tokens = self._call(self.feature_fusion, tokens, previous.to(tokens))
        tokens = self.norm(self.run_blocks(tokens, self.blocks[self.read_after_block:]))
        # There is no history-free layer-12 H here. Metadata explicitly identifies this exit.
        return tokens, tokens

    def write_summary(self, encoded, mask, valid, previous):
        if self.forward_variant == 'mid_temporal_write':
            return self._call(self.temporal_writer, encoded, mask, valid)
        if self.forward_variant != 'content_write':
            return super().write_summary(encoded, mask, valid, previous)
        visible = valid if mask is None else valid[~mask].reshape(len(encoded), -1)
        return self._call(self.content_writer, encoded, previous, visible)
