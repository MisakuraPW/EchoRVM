"""Dense vs interleaved L16 groups, with one real W64 and no group-to-group state."""

import torch
from torch import nn

from . import stage_a
from .readout_calibration import CalibratedExperiment


GROUPS = ('D','G')


def source_groups(arm):
    frames = torch.arange(64)
    if arm=='D':
        return frames.reshape(4,16)
    if arm=='G':
        return frames.reshape(16,4).T.contiguous()
    raise ValueError(arm)


def restore_chronology(sequence, inverse):
    """Expanded tubelet outputs map back to their two individual source positions."""
    return sequence.index_select(1,inverse)


class GapTask(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        self.backbone,self.head,self.arm = base.backbone,base.head,arm
        groups = source_groups(arm)
        self.register_buffer('frame_groups',groups,persistent=False)
        self.register_buffer('inverse_order',groups.flatten().argsort(),persistent=False)

    def descriptors(self, video):
        if video.shape[1]!=64:
            raise ValueError('D/G expect exactly 64 real source frames')
        b = self.backbone
        if video.shape[2]==1 and b.in_chans==3:
            video = video.expand(-1,-1,3,-1,-1)
        g,h,w = b.token_grid
        descriptors = []
        for indices in self.frame_groups:
            clip = video.index_select(1,indices)
            # A-R mid fusion receives no previous state. No recurrent update is made.
            encoded,_ = b.encode_current(clip,None,None)
            native = encoded.reshape(len(video),g,h*w,b.embed_dim)
            # Repeat first, then use the established spatial-mean frame exit. No fabricated
            # interval between the two sources of a stride4 tubelet is ever materialized.
            descriptors.append(b.frame_features(native).mean(2))
        return restore_chronology(torch.cat(descriptors,dim=1),self.inverse_order)

    def forward(self, samples, *, residual=True):
        if not residual or any(s['prefix_frames'] for s in samples):
            raise ValueError('Independent groups have neither residual intervention nor prefix')
        device = next(self.head.parameters()).device
        video = torch.stack([r['video'] for r in samples]).to(device,non_blocking=True)
        return self.head(self.descriptors(video))


class GapExperiment(CalibratedExperiment):
    @property
    def backbone(self):
        return self.model.backbone

    def model_export(self):
        return dict(variant=self.model.arm,scope='independently encoded four-group W64 EF adaptation')


def build(cfg, stage, source, device):
    if (stage['kind'],stage['mode'])!=('ef','finetune') or stage['gap_arm'] not in GROUPS:
        raise ValueError('Only D/G EF FT is supported')
    ref = stage_a.build(cfg,stage,source,device)
    n,d = ref.native_config,cfg['data']
    if (n.get('forward_variant')!='mid_read' or n.get('read_after_block')!=6
        or n.get('asymmetric_mask') or ref.metadata['initialization']['source_epoch']!=400
        or (d['recent_frames'],d['prefix'],d['max_prefix'])!=(64,0,0)
        or ref.model.exit_name!='final' or ref.model.append_boundary):
        raise ValueError('Expected the original independent-400 A-R source and matching W64 EF recipe')
    arm = stage['gap_arm']
    model = GapTask(ref.model,arm).to(device)
    exp = GapExperiment(model,ref.train,ref.val,'ef',ref.metadata,microbatch_safe=True)
    exp.native_config,exp.pretraining_protocol = ref.native_config,ref.pretraining_protocol
    exp.metadata.update(source_groups=source_groups(arm).tolist(),state='empty on every group',
                        head_input='native repeat tubelet exit restored to original frame order')
    return exp
