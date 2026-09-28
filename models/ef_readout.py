"""Compact supervised EF readout; not an additional MAE encoder."""

import torch
from torch import nn
from .video_mae import flat_sinusoid


class TemporalEFReadout(nn.Module):
    """Small learned attention pool with ordering and explicit history token type.

    Complexity is linear in descriptor count. Parameter count is independent of
    the number of clips or memory slots, unlike flatten-and-MLP readouts.
    """
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.project = nn.Linear(dim, hidden)
        self.history_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Linear(hidden, 1)
        self.regress = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, descriptors, history_slots=0):
        x = self.project(descriptors)
        pos = flat_sinusoid(x.shape[-1], x.shape[1]).to(x)
        x = x + pos
        if history_slots:
            if not 0 < history_slots < x.shape[1]:
                raise ValueError('history_slots must leave at least one recent descriptor')
            kind = torch.zeros_like(x)
            kind[:, -history_slots:] = self.history_type
            x = x + kind
        x = self.norm(x)
        weights = self.score(x).softmax(1)
        return self.regress((x * weights).sum(1)).squeeze(-1)


class Stage2EFFineTuner(nn.Module):
    """Differentiable counterpart of the frozen audit; no persistent patient state.

    No detach is used: full fine-tuning backpropagates through the requested
    prefix and recent window. Unlike the inference FIFO, memory scales with T.
    """
    MODES = ('last','cache','cache_empty','cache_history','local_empty','local_history','state','joint_recent')

    def __init__(self, backbone, mode='cache', recent_frames=64, prefix_frames=64, hidden=64):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(mode)
        if recent_frames % backbone.local_frames or prefix_frames % backbone.local_frames or prefix_frames<1:
            raise ValueError('Windows must contain complete clips and a real prefix')
        if mode=='joint_recent' and (backbone.memory_mode!='none' or backbone.local_frames!=recent_frames):
            raise ValueError('joint_recent requires a whole-window no-memory encoder')
        if mode!='joint_recent' and backbone.memory_mode=='none':
            raise ValueError('Recurrent readout requires memory')
        self.backbone = backbone
        self.mode,self.recent_frames,self.prefix_frames=mode,recent_frames,prefix_frames
        self.head=TemporalEFReadout(backbone.embed_dim,hidden)

    def forward(self, video):
        if video.shape[1]!=self.prefix_frames+self.recent_frames:
            raise ValueError('EF needs the declared real prefix + recent window')
        if video.shape[2]==1 and self.backbone.in_chans==3:
            video=video.expand(-1,-1,3,-1,-1)
        model=self.backbone
        if self.mode=='joint_recent':
            sequence=model.frame_features(model.forward_features(video[:,-self.recent_frames:])).mean(2)
            return self.head(sequence).sigmoid()*100
        history=self.mode.endswith(('_empty','_history'))
        start=0 if history else self.prefix_frames
        state=short=boundary=None
        recent=[]
        for i in range(start,video.shape[1],model.local_frames):
            if i==self.prefix_frames:
                boundary=state
            result=model.stream_clip(video[:,i:i+model.local_frames],state,short)
            state,short=result['final_state'],result['final_short']
            if i>=self.prefix_frames:
                key='local_features' if self.mode.startswith('local_') else 'features'
                recent.append(model.frame_features(result[key]).mean(2))
        sequence=torch.cat(recent,1)
        slots=0
        if self.mode=='last':
            sequence=recent[-1]
        elif self.mode=='state':
            sequence=state
        elif history:
            slots=boundary.shape[1]
            value=boundary if self.mode.endswith('_history') else torch.zeros_like(boundary)
            sequence=torch.cat((sequence,value),1)
        return self.head(sequence,slots).sigmoid()*100
