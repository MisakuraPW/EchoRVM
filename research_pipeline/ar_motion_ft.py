"""Supervised A-R EF variants; no label-conditioned inference or SSL changes."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import stage_a
from .readout_calibration import CalibratedExperiment


AUX_WEIGHT = .75
ARMS = ('M1','M1_aux','M2')


class MotionWrite(nn.Module):
    def __init__(self, dim, width=32):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,width),nn.GELU())
        self.area = nn.Linear(width,1)
        self.write = nn.Linear(width,dim,bias=False)
        nn.init.zeros_(self.write.weight)


class SparseRead(nn.Module):
    def __init__(self, dim, width=64):
        super().__init__()
        self.query = nn.Linear(dim,width)
        self.keys = nn.Linear(dim,width)
        self.values = nn.Linear(dim,width)
        self.time = nn.Linear(2,width,bias=False)
        self.out = nn.Linear(width,dim,bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, old, age):
        position = self.time(torch.stack((age,age.square()),dim=-1).to(x))
        attended = F.scaled_dot_product_attention(self.query(x)[:,None],
                       (self.keys(old)+position)[:,None],self.values(old)[:,None])
        return self.out(attended[:,0])


class MotionTask(nn.Module):
    def __init__(self, original, arm):
        super().__init__()
        self.backbone,self.head = original.backbone,original.head
        self.arm = arm
        if arm in ('M1','M1_aux'):
            self.motion = MotionWrite(self.backbone.embed_dim)
            if arm=='M1_aux':
                self.motion.write.requires_grad_(False)
        elif arm=='M2':
            self.sparse = SparseRead(self.backbone.embed_dim)
        else:
            raise ValueError(arm)

    def _encode(self, clip, state, old, age):
        b = self.backbone
        if self.arm!='M2' or old is None:
            return b.encode_current(clip,None,state)[0]
        x = b.patch_embed(b._normalize_input(clip))
        x = b.run_blocks(x+b.pos_embed.to(x),b.blocks[:b.read_after_block])
        x = b._call(b.feature_fusion,x,state.to(x))
        x = x+self.sparse(x,old,age)
        return b.norm(b.run_blocks(x,b.blocks[b.read_after_block:]))

    def forward(self, rows, *, with_aux=False, residual=True):
        b = self.backbone
        if any(r['prefix_frames'] or len(r['video'])!=64 for r in rows):
            raise ValueError('Require fixed H0/W64 real windows')
        device = next(self.head.parameters()).device
        video = torch.stack([r['video'] for r in rows]).to(device,non_blocking=True)
        if video.shape[2]==1 and b.in_chans==3:
            video = video.expand(-1,-1,3,-1,-1)
        g,h,w = b.token_grid
        visible = torch.ones((len(rows),g,h*w),device=device,dtype=torch.bool)
        history,state,times = None,None,None
        recent, q = [],[]
        fps = (torch.tensor([r['fps'] for r in rows],device=device)
               if self.arm=='M2' and residual else None)
        for i,clip in enumerate(video.split(16,dim=1)):
            age = (16*i-times)[None,:]/fps[:,None] if history is not None else None
            encoded = self._encode(clip,state,history,age)
            grid = encoded.reshape(len(rows),g,h*w,-1)
            recent.append(b._frame_exits(grid,state,visible)[1].mean(2))
            if (with_aux and self.arm in ('M1','M1_aux')) or (self.arm=='M1' and residual and i<3):
                m = self.motion.project(grid.mean(2))
                if with_aux:
                    q.append(self.motion.area(m).squeeze(-1))
            if i==3:
                break  # The terminal write has no later EF consumer.
            pooled = b.write_summary(encoded,None,visible.flatten(1),state)
            if self.arm=='M1' and residual:
                pooled = pooled + self.motion.write(m[:,-1]-m[:,0])[:,None].to(pooled)
            _,state = b._call(b.memory,pooled,state)
            if self.arm=='M2' and residual:
                new = grid[:,::2].mean(2)
                t = torch.arange(0,16,4,device=device)+16*i+.5
                history = new if history is None else torch.cat((history,new),dim=1)[:,-12:]
                times = t if times is None else torch.cat((times,t))[-12:]
        pred = self.head(torch.cat(recent,dim=1))
        return (pred,torch.cat(q,dim=1) if q else None) if with_aux else pred


def area_log_targets(cases, rasterize):
    result = {}
    for p,c in cases.items():
        if len(c['traces'])!=2:
            continue
        ed,es = sorted(c['traces'],key=lambda t:t['frame'])
        a,b = int(rasterize(ed).sum()),int(rasterize(es).sum())
        if ed['frame']<es['frame'] and a>b>0:
            result[p] = (ed['frame'],es['frame'],math.log(b/a))
    return result


def window_target(mark, start):
    if mark is None:
        return None
    ed,es,value = mark
    if not start<=ed<es<start+64 or (ed-start)//2==(es-start)//2:
        return None
    return ((ed-start)//2,(es-start)//2,value)


class TrainAreas(torch.utils.data.Dataset):
    def __init__(self, source):
        from utils.final_temporal_data import _trace_mask
        if source.split!='train' or not source.dataset.training:
            raise ValueError('Motion targets are only authorized on sampled TRAIN windows')
        self.source,self.dataset = source,source.dataset
        self.target = area_log_targets({r['patient']:self.dataset.cases[r['patient']]
                                        for r in self.dataset.records},_trace_mask)

    def __len__(self):
        return len(self.source)

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.source[index]
        return dict(sample,motion_target=window_target(self.target.get(sample['patient']),int(sample['recent_start'])))


class MotionExperiment(CalibratedExperiment):
    def calibration_samples(self, batch_size):
        if not isinstance(self.train,TrainAreas):
            return super().calibration_samples(batch_size)
        for i in range(len(self.train)):
            r = self.train.dataset.get_record(i)
            if window_target(self.train.target.get(r['patient']),r['recent_start']) is not None:
                return [self.train[i]]*batch_size
        return [self.train[0]]*batch_size

    @property
    def backbone(self):
        return self.model.backbone

    def loss(self, rows):
        if self.model.arm=='M2':
            return self._task_loss(self.model(rows),rows)
        y,q = self.model(rows,with_aux=True)
        loss = self._task_loss(y,rows)
        targets = [(i,*r['motion_target']) for i,r in enumerate(rows) if r.get('motion_target') is not None]
        device = q.device
        aux = loss.new_zeros((),dtype=torch.float32)
        if targets:
            t = torch.tensor(targets,device=device,dtype=torch.float32)
            i,ed,es = t[:,:3].long().unbind(1)
            # Per-patient normalization makes accumulation invariant to the microbatch grouping.
            aux = F.smooth_l1_loss(q[i,es].float()-q[i,ed].float(),t[:,3],beta=.1,reduction='sum')/len(rows)
        count = lambda x: torch.tensor(x/len(rows),device=device)
        self.last_metrics.update(motion_aux=aux.detach(),valid_pair_rate=count(len(targets)),
             low_ef_pair_rate=count(sum(float(rows[i]['target'])<40 for i,*_ in targets)),
             low_ef_rate=count(sum(float(r['target'])<40 for r in rows)))
        return loss + AUX_WEIGHT*aux

    def model_export(self):
        return dict(variant=self.model.arm,scope='A-R SSL400 supervised EF adaptation')


def build(cfg, stage, source, device):
    if (stage['kind'],stage['mode'])!=('ef','finetune') or stage['motion_arm'] not in ARMS:
        raise ValueError('Motion adapter only permits registered supervised EF arms')
    ref = stage_a.build(cfg,stage,source,device)
    n,d = ref.native_config,cfg['data']
    if (n.get('forward_variant')!='mid_read' or n.get('read_after_block')!=6
        or n.get('asymmetric_mask') or ref.metadata['initialization']['source_epoch']!=400
        or (d['recent_frames'],d['prefix'],d['max_prefix'])!=(64,0,0)):
        raise ValueError('Require exact A-R SSL400 H0/L16/W64')
    arm = stage['motion_arm']
    exp = MotionExperiment(MotionTask(ref.model,arm).to(device),
                           TrainAreas(ref.train) if arm!='M2' else ref.train,ref.val,'ef',
                           ref.metadata,microbatch_safe=True)
    exp.native_config,exp.pretraining_protocol = ref.native_config,ref.pretraining_protocol
    exp.metadata.update(motion_arm=arm,motion_aux_weight=AUX_WEIGHT if arm!='M2' else 0.,
                        motion_targets='TRAIN-only log ES/ED trace area ratio, no input or sampling use' if arm!='M2' else None)
    return exp
