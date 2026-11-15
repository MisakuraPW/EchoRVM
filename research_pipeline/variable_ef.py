"""Stage B: causal boundary decisions over real, variable-length A-R clips."""

from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.video_mae import flat_sinusoid
from . import stage_a
from .artifacts import file_hash
from .readout_calibration import CalibratedExperiment


WINDOW, CLIPS, CELL = 64, 4, 4
MIN_CELLS, MAX_CELLS = 2, 8
CELLS = WINDOW//CELL


def stop_table():
    # Completion counts give a uniform distribution over complete legal partitions.
    counts = [[int(i==0) for i in range(CELLS+1)]]
    for _ in range(CLIPS):
        counts.append([sum(counts[-1][r-k] for k in range(MIN_CELLS,MAX_CELLS+1) if r>=k)
                       for r in range(CELLS+1)])
    table = torch.zeros(CLIPS,CELLS+1,MAX_CELLS+1)
    for future in range(CLIPS):
        for remaining in range(CELLS+1):
            for k in range(MIN_CELLS,MAX_CELLS+1):
                n = counts[future][remaining-k] if remaining>=k else 0
                tail = sum(counts[future][remaining-j] for j in range(k,MAX_CELLS+1) if remaining>=j)
                table[future,remaining,k] = n/tail if tail else 0.
    return table


def update_fifo(cache, descriptors, capacity=WINDOW):
    return (descriptors if cache is None else torch.cat((cache,descriptors),1))[:,-capacity:]


class Observer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(1,8,3,stride=2,padding=1),nn.GELU(),
                                   nn.AdaptiveAvgPool2d((2,2)),nn.Flatten())
        self.state = nn.Linear(dim,32)
        self.gru = nn.GRUCell(32+3,32)
        self.actor = nn.Linear(32,1)
        self.value = nn.Linear(32,1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        nn.init.constant_(self.value.bias,-4.)

    def cell_features(self, video):
        # Spatial pooling and CNN are framewise. Averaging never crosses a four-frame arrival.
        with torch.no_grad():
            b,frames,_,h,w=video.shape
            low=F.adaptive_avg_pool2d(video[:,:,0].reshape(b*frames,1,h,w),(16,16))
            low=low.reshape(b*CELLS,CELL,1,16,16)
            mean=low.mean(1)
            motion=(low[:,1:]-low[:,:-1]).abs().mean((1,2,3,4)).reshape(b,CELLS)
        return self.vision(mean).reshape(b,CELLS,32),motion


class VariableEF(nn.Module):
    def __init__(self,backbone,head,policy):
        super().__init__()
        self.backbone,self.head,self.policy=backbone,head,policy
        with torch.random.fork_rng(devices=[]):
            self.observer=Observer(backbone.embed_dim)
        self.observer.requires_grad_(policy=='observed')
        self.register_buffer('stop_prior',stop_table(),persistent=False)
        _,gh,gw=backbone.token_grid
        self.spatial=gh*gw
        self.register_buffer('extended_position',flat_sinusoid(backbone.embed_dim,16*self.spatial),persistent=False)
        if not torch.equal(self.extended_position[:,:backbone.pos_embed.shape[1]],backbone.pos_embed.cpu()):
            raise ValueError('Position rule disagrees with native L16 A-R')

    def encode_clip(self, video, previous):
        model=self.backbone
        frames=video.shape[1]
        if frames not in range(8,33,4):
            raise ValueError('Encode only real admissible clips')
        if frames==16:
            tokens,_=model.encode_current(video,None,previous)
        else:
            tokens=model.patch_embed(model._normalize_input(video))
            tokens=tokens+self.extended_position[:,:tokens.shape[1]].to(tokens)
            split=model.read_after_block
            tokens=model.run_blocks(tokens,model.blocks[:split])
            if previous is not None:
                tokens=model._call(model.feature_fusion,tokens,previous.to(tokens))
            tokens=model.norm(model.run_blocks(tokens,model.blocks[split:]))
        grid=tokens.reshape(len(video),frames//model.tubelet_size,self.spatial,-1)
        # Same unmasked global writer reduction as FinalTemporalMAE.
        dense=grid.reshape(len(video),grid.shape[1],*model.token_grid[1:],model.embed_dim)
        pooled=dense.sum(1).permute(0,3,1,2)
        pooled=(F.adaptive_avg_pool2d(pooled,(1,1))/grid.shape[1]).flatten(2).transpose(1,2)
        _,state=model._call(model.memory,pooled,previous)
        frames_out=grid.repeat_interleave(model.tubelet_size,1).mean(2)
        return frames_out,state

    def choose(self,start,future,draws,previous,vision,motion,policy):
        b=len(start)
        ids=torch.arange(b,device=start.device)
        remaining=CELLS-start
        chosen=torch.zeros_like(start)
        observed=policy=='observed'
        logps,entropies,values,free_masks,signal,actions=[],[],[],[],[],[]
        if observed:
            state=(vision.new_zeros(b,self.backbone.embed_dim) if previous is None else previous[:,0].detach())
            h=self.observer.state(state)
        for k in range(1,MAX_CELLS+1):
            active=chosen==0
            if observed:
                # Later iterations for stopped rows see no new pixel data.
                available=torch.where(active,k,chosen)
                cell=(start+available-1).clamp_max(CELLS-1)
                budget=vision.new_tensor([k/MAX_CELLS,future/CLIPS]).expand(b,2)
                obs=torch.cat((vision[ids,cell],budget,remaining[:,None].to(vision)/CELLS),1)
                h=torch.where(active[:,None],self.observer.gru(obs,h),h)
            if k<MIN_CELLS:
                continue
            p=self.stop_prior[future,remaining,k]
            free=active & (p>0) & (p<1)
            if observed:
                logits=torch.logit(p,eps=1e-5)+self.observer.actor(h).squeeze(-1).float()
                chance=torch.where(free,logits.sigmoid(),p)
            else:
                chance=p
            stop=active & ((chance>=1) | ((chance>0) & (draws[:,k-1]<chance)))
            if observed:
                logps.append(-F.binary_cross_entropy_with_logits(logits,stop.float(),reduction='none'))
                entropies.append(F.softplus(logits)-logits*logits.sigmoid())
                values.append(self.observer.value(h.detach()).squeeze(-1).float())
                free_masks.append(free)
                signal.append(motion[ids,cell])
                actions.append(stop)
            chosen=torch.where(stop,k,chosen)
        stats=None if not observed else tuple(torch.stack(x,1) for x in (logps,entropies,values,free_masks,signal,actions))
        return chosen,stats

    def forward(self,video,draws,policy=None,forced=None):
        if video.ndim!=5 or video.shape[1]!=WINDOW:
            raise ValueError('Stage B needs exactly one 64-real-frame window')
        if forced is not None and (forced.shape!=(len(video),CLIPS) or
                not bool(((forced%CELL==0).all()) & ((forced.sum(1)==WINDOW).all()))):
            raise ValueError('Forced execution must also partition all 64 frames on the registered boundary grid')
        policy=self.policy if policy is None else policy
        if video.shape[2]==1 and self.backbone.in_chans==3:
            video=video.expand(-1,-1,3,-1,-1)
        b=len(video)
        ids=torch.arange(b,device=video.device)
        start=torch.zeros(b,device=video.device,dtype=torch.long)
        cache=state=None
        features,all_lengths,stats=[],[],[]
        if policy=='observed' and forced is None:
            vision,motion=self.observer.cell_features(video)
        else:
            vision=motion=None
        for hop in range(CLIPS):
            if forced is not None:
                lengths=forced[:,hop]//CELL
            elif policy=='fixed':
                lengths=torch.full_like(start,16//CELL)
            else:
                lengths,info=self.choose(start,CLIPS-1-hop,draws[:,hop],state,vision,motion,policy)
                if info is not None:
                    stats.append(info)
            all_lengths.append(lengths)
            groups={}
            for i,length in enumerate(lengths.tolist()):
                if not MIN_CELLS<=length<=MAX_CELLS:
                    raise ValueError('Illegal or unclosed boundary')
                groups.setdefault(length,[]).append(i)
            new_state=[]
            for cells,indices in sorted(groups.items()):
                group=ids[indices]
                frame_index=start[group,None]*CELL+torch.arange(cells*CELL,device=video.device)
                descriptors,after=self.encode_clip(video[group[:,None],frame_index],
                                                    None if state is None else state[group])
                features.append((group[:,None]*WINDOW+frame_index,descriptors))
                new_state.append((group,after))
            state=torch.cat([s for _,s in new_state])[torch.cat([g for g,_ in new_state]).argsort()]
            start=start+lengths
        if not bool((start==CELLS).all()):
            raise ValueError('All four clips must cover the entire window')
        order=torch.cat([i.flatten() for i,_ in features]).argsort()
        sequence=torch.cat([x.flatten(0,1) for _,x in features])[order].reshape(b,WINDOW,-1)
        cache=update_fifo(cache,sequence)
        return self.head(cache),dict(lengths=torch.stack(all_lengths,1)*CELL,stats=stats)


class VariableEFExperiment(CalibratedExperiment):
    def set_checkpointing(self, enabled):
        self.backbone.gradient_checkpointing=bool(enabled)

    def optimizer_metrics(self):
        grad=self.model.observer.actor.weight.grad
        if grad is None:
            return {}
        return {'actor_grad_l2':float(grad.detach().float().norm())}

    def _forward(self,samples,policy=None):
        device=next(self.model.parameters()).device
        if any(int(s['prefix_frames'])!=0 or len(s['video'])!=WINDOW for s in samples):
            raise ValueError('Stage B never adds prefix or padding')
        epoch=self.train.dataset.epoch.value if self.model.training else 0
        draws=[]
        for s in samples:
            # Identical seeded draws across microbatch settings and the two arms.
            rng=random.Random(f"{self.seed}:{epoch}:{s['sample_id']}")
            draws.append([rng.random() for _ in range(CLIPS*MAX_CELLS)])
        video=torch.stack([s['video'] for s in samples]).to(device,non_blocking=True)
        rand=torch.tensor(draws,device=device).reshape(-1,CLIPS,MAX_CELLS)
        forced=None
        if samples[0].get('stress_lengths'):
            forced=torch.tensor([s['stress_lengths'] for s in samples],device=device)
        return self.model(video,rand,policy=policy,forced=forced)

    def calibration_samples(self,batch_size):
        values=super().calibration_samples(batch_size)
        # The long clips align to stress memory; the normal trials still sample actual policies.
        return [dict(s,sample_id=f'train:stress:{i}',stress_lengths=[32,16,8,8]) for i,s in enumerate(values)]

    def loss(self,samples):
        prediction,extra=self._forward(samples)
        ef=self._task_loss(prediction,samples)
        lengths=extra['lengths']
        for length in range(8,33,4):
            self.last_metrics[f'clip_fraction_{length}']=(lengths==length).float().mean()
        self.last_metrics['attention_length_square_sum']=lengths.float().square().sum(1).mean()
        if not extra['stats']:
            return ef
        reward=-(prediction.detach().float()*100-torch.stack([s['target'] for s in samples]).to(prediction.device)).abs()
        logp,entropy,value,free,_,_=(torch.cat(xs,1) for xs in zip(*extra['stats']))
        advantage=(reward[:,None]-value.detach()).clamp(-20,20)
        actor=-(logp*advantage*free).sum(1).mean()/CLIPS
        critic=(F.smooth_l1_loss(value,reward[:,None].expand_as(value),reduction='none')*free).sum(1).mean()/CLIPS
        mean_entropy=(entropy*free).sum()/free.sum().clamp_min(1)
        self.last_metrics.update(policy_entropy=mean_entropy,policy_free_actions=free.sum(1).float().mean(),
                                 actor_loss=actor,critic_loss=critic)
        return ef+.05*actor+.05*critic-.01*mean_entropy

    def evaluate_batch(self,samples,boundaries=False):
        pred,extra=self._forward(samples,getattr(self,'eval_policy',None))
        rows=self._rows(pred,samples,boundaries)
        lengths=extra['lengths'].tolist()
        stats=extra['stats']
        for i,(row,sample) in enumerate(zip(rows,samples)):
            row.update(clip_lengths=lengths[i],recent_start=int(sample['recent_start']),
                       frames=WINDOW,prefix_frames=0)
        if stats:
            _,entropy,_,free,motion,action=(torch.cat(xs,1) for xs in zip(*stats))
            for row,e,f,m,a in zip(rows,entropy.tolist(),free.tolist(),motion.tolist(),action.tolist()):
                row.update(policy_entropy=sum(x for x,y in zip(e,f) if y)/max(1,sum(f)),free_actions=sum(f),
                           free_stop_motion=[x for x,y,z in zip(m,f,a) if y and z],
                           free_continue_motion=[x for x,y,z in zip(m,f,a) if y and not z])
        return self._task_loss(pred,samples),rows

    def preview(self,samples,directory):
        return None

    def model_export(self):
        return dict(stage_b_warmup=self.metadata['stage_b_warmup'],
                    pretraining_data_protocol=self.pretraining_protocol,
                    config=dict(model=self.native_config))


def evaluation_dataset(cfg,manifest,split,task):
    return stage_a.evaluation_dataset(cfg,manifest,split,task)


def build(config,stage,source,device):
    ssl=Path(config['stage_b_ssl']['path']).resolve()
    if file_hash(ssl)!=config['stage_b_ssl']['sha256']:
        raise ValueError('A-R fixed SSL source changed')
    reference=stage_a.build(config,stage,ssl,device)
    backbone=reference.backbone
    native=reference.native_config
    if (stage['kind']!='ef' or stage['mode']!='finetune' or
            (backbone.forward_variant,backbone.read_after_block,backbone.local_frames,
             backbone.embed_dim,native.get('position_embedding'))!=('mid_read',6,16,384,'flat_sinusoid')):
        raise ValueError('Stage B is an A-R full EF adaptation, not an SSL or segmentation job')
    policy=stage['boundary_policy']
    if policy not in ('uniform','observed'):
        raise ValueError('Unknown boundary policy')
    model=VariableEF(backbone,reference.model.head,policy).to(device)
    if not source:
        raise ValueError('Stage B source must be explicit')
    if Path(source).resolve()!=ssl:
        payload=torch.load(source,map_location='cpu',weights_only=True)
        known=payload.get('source')
        if not known or Path(known).resolve()!=ssl or not payload.get('stage_b_warmup') or stage['id']=='B1':
            raise ValueError('B2 requires the VAL-selected B1 full EF checkpoint')
        model.load_state_dict(payload['model'],strict=True)
    elif stage['id']!='B1':
        raise ValueError('B2 must start from completed B1')
    exp=VariableEFExperiment(model,reference.train,reference.val,'ef',reference.metadata,
                             microbatch_safe=True,cacheable=False)
    exp.native_config,exp.pretraining_protocol,exp.seed=native,reference.pretraining_protocol,reference.seed
    exp.metadata.update(boundary_policy=policy,stage_b_warmup=stage['id']=='B1',
                        feature_cache='disabled: segmentation changes the encoder path')
    return exp
