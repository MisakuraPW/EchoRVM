"""Head-only EF readout over two views of one immutable A-R feature cache."""

import hashlib
import shutil

import torch
from torch import nn
from torch.nn import functional as F

from models.video_mae import flat_sinusoid
from utils.final_temporal_execution import encode_window
from . import stage_a
from .artifacts import layout
from .cache import prepare_cache
from .readout_calibration import CalibratedExperiment


FRAMES, GROUPS, GRID, DIM = 64, 32, 14, 384
BATCH_BYTES = GROUPS*GRID*GRID*DIM*2 + FRAMES*DIM*4 + 4


class QueryEF(nn.Module):
    def __init__(self, spatial):
        super().__init__()
        width = 192
        self.spatial = spatial
        self.embed = nn.Linear(DIM,width)
        self.queries = nn.Parameter(torch.randn(1,4,width)*.02)
        self.query_projection = nn.Linear(width,width)
        self.cross_output = nn.Linear(width,width)
        self.interaction = nn.Sequential(*(nn.TransformerEncoderLayer(
            width,4,384,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            for _ in range(2)))
        self.regress = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.regress[-1].weight)
        nn.init.constant_(self.regress[-1].bias,.556)
        self.register_buffer('time',flat_sinusoid(width,FRAMES)[:,:,None],persistent=False)
        axis = flat_sinusoid(width//2,GRID)[0]
        pos = torch.cat((axis[:,None].expand(-1,GRID,-1),axis[None].expand(GRID,-1,-1)),dim=-1)
        pos = pos.reshape(1,1,GRID*GRID,width)
        self.register_buffer('space',pos-pos.mean(2,keepdim=True),persistent=False)

    def forward(self, features):
        if self.spatial:
            if features.shape[1:] != (GROUPS,GRID*GRID,DIM):
                raise ValueError('C2 requires full A-R spatial token grids')
            # Project before replicating the two frame positions of each tubelet.
            values = self.embed(features).repeat_interleave(2,dim=1)
        else:
            if features.shape[1:] != (FRAMES,DIM):
                raise ValueError('C1 requires exactly the legacy 64 frame descriptors')
            values = self.embed(features)[:,:,None]
        keys = values+self.time.to(values)
        if self.spatial:
            keys = keys+self.space.to(values)
        def heads(x):
            return x.reshape(len(x),-1,4,48).transpose(1,2).contiguous()
        q = self.query_projection(self.queries.expand(len(values),-1,-1))
        cross = F.scaled_dot_product_attention(heads(q),heads(keys),heads(values))
        output = self.queries+self.cross_output(cross.transpose(1,2).reshape(len(q),4,192))
        return self.regress(self.interaction(output).mean(1)).squeeze(-1)


def spatial_mean(tokens):
    return tokens.mean(2).repeat_interleave(2,dim=1)


def tensor_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def allocation_guard(directory, count, reserve, allowance=0):
    need = count*BATCH_BYTES + 16*1024**2
    if shutil.disk_usage(directory).free+allowance < need+reserve:
        raise OSError(f'Need {need/1024**3:.2f} GiB for registered features plus {reserve/1024**3:.2f} GiB reserve')
    return need


class SpatialEF(CalibratedExperiment):
    def attach_backbone(self, backbone, initial_digest):
        self._backbone = backbone.requires_grad_(False).eval()
        self.initial_digest = initial_digest
        self.precision_error = 0.

    @property
    def backbone(self):
        return self._backbone

    def train_mode(self):
        self.model.train()

    def set_checkpointing(self, enabled):
        if enabled:
            raise ValueError('Checkpointing a frozen backbone is not a head runtime option')

    def model_export(self):
        return dict(frozen_backbone_digest=self.initial_digest, head_only=True)

    def verify_frozen(self):
        if (self.backbone.training or any(p.requires_grad or p.grad is not None for p in self.backbone.parameters())
                or tensor_digest(self.backbone)!=self.initial_digest):
            raise ValueError('Frozen encoder/state update was changed')
        return dict(backbone_digest=self.initial_digest, backbone_eval=True, backbone_has_grad=False,
                    trainable_only_query_head=True,
                    frozen_parameters=sum(p.numel() for p in self.backbone.parameters()))

    @torch.no_grad()
    def features(self, samples):
        if any(len(s['video'])!=FRAMES or int(s['prefix_frames'])!=0 for s in samples):
            raise ValueError('Expected exactly four chronological clips with independent initial state')
        device = next(self.backbone.parameters()).device
        video = torch.stack([s['video'] for s in samples]).to(device,non_blocking=True)
        exits = encode_window(self.backbone,video,keep_final_grid=True)
        full = exits['final_grid']
        if full.shape[1:] != (FRAMES,GRID*GRID,DIM) or not torch.equal(full[:,::2],full[:,1::2]):
            raise ValueError('A-R is no longer the original repeat-tubelet frame exit')
        native = full[:,::2].contiguous()
        torch.testing.assert_close(spatial_mean(native),exits['sequences']['final'],rtol=1e-5,atol=1e-6)
        if not bool((native[:,:,0]!=native[:,:,1]).any()):
            raise ValueError('C2 extraction collapsed the spatial grid')
        self.precision_error = max(self.precision_error,float((native-native.half().float()).abs().max()))
        return native

    def forward_head(self, head, values):
        device = next(head.parameters()).device
        values = values.to(device,non_blocking=True)
        if device.type=='cpu' or not torch.is_autocast_enabled():
            values = values.float()
        return head(values)

    def loss(self, samples):
        output = self.forward_head(self.model,torch.stack([s['features'] for s in samples]))
        return self._task_loss(output,samples)

    def evaluate_batch(self, samples, boundaries=False):
        output = self.forward_head(self.model,torch.stack([s['features'] for s in samples]))
        return self._task_loss(output,samples),self._rows(output,samples,boundaries)

    def _rows(self, prediction, samples, boundaries=False):
        rows = super()._rows(prediction,samples,boundaries)
        for row,sample in zip(rows,samples):
            row.update(recent_start=int(sample['recent_start']),frames=FRAMES,
                       prefix_frames=int(sample['prefix_frames']))
        return rows

    def prepare_feature_cache(self,cfg,directory,key,heartbeat):
        run=cfg['runtime']
        directory.mkdir(parents=True,exist_ok=True)
        budget=int(run['cache_gb']*1024**3)
        reserve=int(run['min_free_gb']*1024**3)
        allocated=0
        for p in directory.rglob('*.npy'):
            stat=p.stat()
            allocated += getattr(stat,'st_blocks',stat.st_size//512)*512
        allocation_guard(directory,len(self.train)+len(self.val),reserve,allocated)
        result=prepare_cache(self,directory,key,budget,reserve,heartbeat,batch_size=run['micro_batch'],
                             budget_root=layout(cfg)['cache'],feature_dtype='float16',
                             derived_views={'mean':spatial_mean},view='features' if self.model.spatial else 'mean')
        result.update(view='dense tubelet2' if self.model.spatial else 'legacy mean',
                      dense_shape=[GROUPS,GRID*GRID,DIM], mean_precision='float32',
                      new_extraction_max_fp16_absolute_error=self.precision_error,
                      zero_means_cache_reused=True, legacy_mean_checked_on_creation=True)
        # Once sealed, all optimizer trials and updates use only head and cached rows.
        self.backbone.cpu()
        return result


def build(cfg, stage, source, device):
    if (stage['kind'],stage.get('mode'),stage.get('spatial_ef_view')) not in {
            ('ef','frozen_lite','mean'),('ef','frozen_lite','dense')} or not source:
        raise ValueError('This adapter only fits the two frozen EF query readouts')
    reference=stage_a.build(cfg,stage,source,device)
    native=reference.native_config
    expected=dict(forward_variant='mid_read',read_after_block=6,patch_size=8,img_size=112,
                  tubelet_size=2,frame_readout='repeat',memory_mode='global',embed_dim=DIM)
    if (any(native.get(k)!=v for k,v in expected.items()) or native.get('asymmetric_mask') or
            reference.metadata['initialization']['source_epoch']!=400 or
            reference.model.exit_name!='final' or reference.model.append_boundary):
        raise ValueError('Require original A-R epoch-0400 SSL, not a downstream checkpoint or another exit')
    backbone=reference.backbone
    model=QueryEF(stage['spatial_ef_view']=='dense').to(device)
    reference.metadata.update(head='QueryEF four learned queries, width192, two query blocks',
                              head_initial_sha256=tensor_digest(model),
                              parameter_count_scope='head-only; immutable A-R not optimized or duplicated in checkpoints',
                              checkpointing=False)
    exp=SpatialEF(model,reference.train,reference.val,'ef',reference.metadata,microbatch_safe=True,cacheable=True)
    exp.attach_backbone(backbone,tensor_digest(backbone))
    return exp
