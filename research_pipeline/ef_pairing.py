"""Matched EF readouts over one fixed, frozen A-R SSL window."""

from pathlib import Path

import torch
from torch import nn

from utils.final_temporal_execution import encode_window
from . import stage_a
from .artifacts import digest, layout
from .cache import prepare_cache
from .readout_calibration import CalibratedExperiment


VIEWS = {'I': 'independent', 'J': 'independent', 'R': 'recurrent', 'P': 'recurrent',
         'I_stride4': 'stride4'}


def clip_views(video, view):
    if video.shape[1] != 64:
        raise ValueError('Require exactly 64 source frames')
    if view == 'stride4':
        return [video[:, i::4] for i in range(4)]
    if view == 'independent':
        return list(video.split(16, dim=1))
    raise ValueError(f'Unknown independent grouping: {view}')


@torch.no_grad()
def describe(backbone, video, view):
    if video.shape[1] != 64:
        raise ValueError('Require a complete W64 window')
    groups = [video] if view == 'recurrent' else clip_views(video, view)
    # Every encode_window call starts with empty state, including overlapping stride4 tracks.
    seq = torch.cat([encode_window(backbone, x)['sequences']['final'] for x in groups], dim=1)
    if seq.shape[1] != 64 or backbone.frame_readout != 'repeat' or backbone.tubelet_size != 2:
        raise ValueError('Cache requires the original repeated tubelet2 frame exit')
    return seq[:, ::2].contiguous()


class PairReadout(nn.Module):
    """Ordered, video-only latent endpoint pairing with a trainable no-pair option."""

    def __init__(self, dim, width=32):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU())
        self.phase = nn.Linear(width, 2)
        self.null_score = nn.Parameter(torch.zeros(()))
        self.correction = nn.Sequential(nn.Linear(3*width+1, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        times = torch.arange(32)*2+.5
        self.register_buffer('gap', times[None, :] - times[:, None], persistent=False)

    def forward(self, seq, fps, *, perturb=False):
        if seq.shape[1] != 32 or fps.shape != (len(seq),):
            raise ValueError('Require one FPS and 32 ordered tubelets per window')
        z = self.project(seq)
        ed, es = self.phase(z).float().unbind(-1)
        seconds = self.gap[None, :, :] / fps.float()[:, None, None]
        valid = (seconds >= .12) & (seconds <= .65)
        scores = ed[:, :, None] + es[:, None, :]
        # Normalize candidate count against the null; extra possible pairs alone must not boost confidence.
        scores = scores.masked_fill(~valid, -torch.inf) - valid.sum((1, 2)).clamp_min(1).log()[:, None, None]
        weights = torch.cat((scores.flatten(1), self.null_score.float().expand(len(seq), 1)), dim=1)
        probabilities = weights.softmax(1)
        weights = probabilities[:, :-1].reshape_as(scores)
        if not self.training:
            # Fixed before fitting, not calibrated on VAL or TEST.
            weights = weights * (probabilities[:, -1] < .75)[:, None, None]
        b, i, j = valid.nonzero(as_tuple=True)
        if perturb:
            # VAL-only stress test: reverse eligible end features within each start position.
            # Keep the actual interval, membership and unperturbed pair scores fixed.
            _, counts = torch.unique_consecutive(b*32+i, return_counts=True)
            ends = counts.cumsum(0)
            reverse = torch.repeat_interleave(ends + ends-counts-1, counts, output_size=len(j))
            end = z[b, j[reverse-torch.arange(len(j),device=j.device)]]
        else:
            end = z[b, j]
        start = z[b, i]
        correction = self.correction(torch.cat((start, end, end-start, seconds[b,i,j,None].to(z)), dim=-1))
        return weights.new_zeros(len(seq)).index_add(0, b, correction[:,0].float()*weights[b,i,j])


class FrozenReadout(nn.Module):
    def __init__(self, head, arm, dim):
        super().__init__()
        self.c0, self.arm = head, arm
        self.pair = PairReadout(dim) if arm == 'P' else None

    def forward(self, tokens, fps, *, perturb=False):
        if tokens.ndim != 3 or tokens.shape[1] != 32:
            raise ValueError('Expected 32 distinct tubelet descriptors')
        expanded = tokens.repeat_interleave(2, dim=1)
        if self.arm in ('I', 'I_stride4'):
            return self.c0(expanded.reshape(-1, 16, expanded.size(-1))).reshape(-1,4).mean(1)
        prediction = self.c0(expanded)
        if self.pair is not None:
            prediction = prediction.float() + self.pair(tokens, fps, perturb=perturb)
        return prediction


class PairingExperiment(CalibratedExperiment):
    def __init__(self, reference, arm):
        from .spatial_ef import tensor_digest
        backbone = reference.backbone.requires_grad_(False).eval()
        model = FrozenReadout(reference.model.head, arm, backbone.embed_dim)
        super().__init__(model.to(next(backbone.parameters()).device), reference.train, reference.val,
                         'ef', reference.metadata, microbatch_safe=True, cacheable=True)
        self._backbone, self.arm = backbone, arm
        self.initial_digest = tensor_digest(backbone)

    @property
    def backbone(self):
        return self._backbone

    def set_checkpointing(self, enabled):
        if enabled:
            raise ValueError('Only a head is being optimized')

    def model_export(self):
        if self.backbone.training or any(p.requires_grad or p.grad is not None for p in self.backbone.parameters()):
            raise ValueError('Backbone must remain frozen and in eval mode')
        return dict(head_only=True, arm=self.arm)

    def verify_frozen(self):
        from .spatial_ef import tensor_digest
        self.model_export()
        final = tensor_digest(self.backbone)
        if final!=self.initial_digest:
            raise ValueError('Frozen backbone tensors were changed')
        return dict(backbone_digest=final, head_only=True, backbone_eval=True)

    def features(self, samples):
        if any(int(s['prefix_frames']) != 0 for s in samples):
            raise ValueError('Unexpected extra prefix')
        device = next(self.backbone.parameters()).device
        video = torch.stack([s['video'] for s in samples]).to(device, non_blocking=True)
        return describe(self.backbone, video, VIEWS[self.arm])

    def forward_samples(self, samples, *, perturb=False):
        device = next(self.model.parameters()).device
        if 'features' in samples[0]:
            values = torch.stack([s['features'] for s in samples]).to(device, non_blocking=True)
        else:
            values = self.features(samples)
        fps = torch.tensor([s['fps'] for s in samples], device=device)
        return self.model(values, fps, perturb=perturb)

    def loss(self, samples):
        return self._task_loss(self.forward_samples(samples), samples)

    def evaluate_batch(self, samples, boundaries=False):
        y = self.forward_samples(samples)
        rows = self._rows(y, samples)
        for row,sample in zip(rows,samples):
            row.update(recent_start=int(sample['recent_start']), fps=float(sample['fps']))
        return self._task_loss(y, samples), rows

    def prepare_feature_cache(self, cfg, directory, key, heartbeat):
        # Readout identity is intentionally excluded; encoding/view, data and weights are not.
        key['metadata'] = {k:v for k,v in key['metadata'].items() if k != 'ef_pairing_arm'}
        key.update(encoding_view=VIEWS[self.arm], descriptors='32 spatial means FP32')
        root = layout(cfg)['cache']
        runtime = cfg['runtime']
        result = prepare_cache(self, root/'frozen'/digest(key), key,
            int(runtime['cache_gb']*1024**3), int(runtime['min_free_gb']*1024**3), heartbeat,
            batch_size=runtime['micro_batch'], budget_root=root)
        self.backbone.cpu()
        return result


def build(cfg, stage, source, device):
    if stage.get('mode') != 'frozen_lite' or stage.get('kind') != 'ef':
        raise ValueError('EF pairing is intentionally head-only')
    reference = stage_a.build(cfg, stage, source, device)
    native, data = reference.native_config, cfg['data']
    if (reference.metadata['initialization']['source_epoch'] != 400 or Path(source).name != 'epoch_0400.pt'
        or native.get('forward_variant') != 'mid_read' or native.get('read_after_block') != 6
        or native.get('asymmetric_mask') or reference.model.append_boundary or reference.model.exit_name != 'final'
        or tuple(data[k] for k in ('recent_frames','local_frames','prefix','max_prefix')) != (64,16,0,0)):
        raise ValueError('Expected the original A-R SSL400 H0/W64/L16 final exit')
    arm = stage['ef_pairing_arm']
    if arm not in VIEWS:
        raise ValueError(f'Unknown EF arm: {arm}')
    reference.metadata.update(ef_pairing_arm=arm, checkpointing=False)
    return PairingExperiment(reference, arm)
