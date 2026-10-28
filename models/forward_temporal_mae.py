"""Two single-change forward candidates; no combined variant or layer search."""

import torch
from torch import nn

from .final_temporal_mae import FinalTemporalMAE


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
        if self.forward_variant not in {'mid_read', 'content_write'}:
            raise ValueError('Only A or W individually are supported')
        if (self.memory_mode != 'global' or self.frame_readout != 'repeat' or
                self.memory_read_location != 'tokens' or self.memory_write_source != 'fused' or
                self.memory_compression != 'mean' or config.get('candidate_rank', 0)):
            raise ValueError('Forward candidates require the clean single-global-state baseline')
        if self.forward_variant == 'mid_read' and not 0 < self.read_after_block < len(self.blocks):
            raise ValueError('Read insertion must precede at least one remaining encoder block')
        if self.forward_variant == 'content_write':
            with torch.random.fork_rng(devices=[]):
                self.content_writer = ContentWriter(self.embed_dim, int(config.get('writer_width', 64)))

    def encode_current(self, clip, mask, previous):
        if self.forward_variant != 'mid_read':
            return super().encode_current(clip, mask, previous)
        tokens = self.patch_embed(self._normalize_input(clip))
        tokens = tokens+self.pos_embed.to(tokens)
        if mask is not None:
            tokens = tokens[~mask].reshape(len(clip), -1, self.embed_dim)
        tokens = self.run_blocks(tokens, self.blocks[:self.read_after_block])
        if previous is not None:
            tokens = self._call(self.feature_fusion, tokens, previous.to(tokens))
        tokens = self.norm(self.run_blocks(tokens, self.blocks[self.read_after_block:]))
        # There is no history-free layer-12 H here. Metadata explicitly identifies this exit.
        return tokens, tokens

    def write_summary(self, encoded, mask, valid, previous):
        if self.forward_variant != 'content_write':
            return super().write_summary(encoded, mask, valid, previous)
        visible = valid if mask is None else valid[~mask].reshape(len(encoded), -1)
        return self._call(self.content_writer, encoded, previous, visible)
