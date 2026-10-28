"""Bounded D1/D2/D4 on immutable checkpoints; D3 uses the public probe engine."""

from contextlib import contextmanager
from collections import deque
import time

import torch

from .artifacts import code_identity, digest, file_hash, layout, mark_done, verify_done, write_json
from .engine import Heartbeat, csv_rows
from utils.seed import get_rng_state, set_rng_state, seed_everything


@contextmanager
def blocked_reads(model, feature=False, decoder=False):
    """Zero only the history-context edge; retain the module/current-token computation."""
    handles = []
    def zero_context(module, args):
        return args[0], torch.zeros_like(args[1])
    try:
        if feature:
            handles.append(model.feature_fusion.register_forward_pre_hook(zero_context))
        if decoder:
            handles.append(model.memory_fusion.register_forward_pre_hook(zero_context))
        yield
    finally:
        for handle in handles:
            handle.remove()


def last_clip_loss(model, video, masks, return_full=False):
    result = model(video, masks=masks)
    count = model.patch_embed.num_patches
    error = (result['pred'][:, -count:].float()-result['target'][:, -count:].detach().float()).square().mean(-1)
    weight = result['mask'][:, -count:]
    loss = (error*weight).sum()/weight.sum()
    return (loss,result['loss']) if return_full else loss


def gradient_credit(model, video, masks, feature=False, decoder=False):
    value = video.detach().clone().requires_grad_(True)
    states, writes = [], []
    def capture(module,args,output):
        writes.append(args[0])
        states.append(output[1])
    handle = model.memory.register_forward_hook(capture)
    try:
        with blocked_reads(model, feature, decoder):
            loss = last_clip_loss(model, value, masks)
        parameters = [(name,p) for name,p in model.named_parameters() if name.startswith(('memory.','content_writer.'))]
        gradients = torch.autograd.grad(loss, [value, *states, *writes, *[p for _,p in parameters]], allow_unused=True)
        def measure(x):
            return dict(connected=x is not None, norm=None if x is None else float(x.norm()),
                        finite=True if x is None else bool(torch.isfinite(x).all()))
        clips = len(states)
        return dict(loss=float(loss.detach()), target='last clip only; target detached',
                    input_clips=[measure(g) for g in gradients[0].split(model.local_frames,1)],
                    states=[measure(g) for g in gradients[1:1+clips]],
                    time_specific_writes=[measure(g) for g in gradients[1+clips:1+2*clips]],
                    shared_writer_parameters={name:measure(g) for (name,_),g in zip(parameters,gradients[1+2*clips:])})
    finally:
        handle.remove()


def stream(model, video, prefix, reset_recent=False, clear_last=False, corrupt=None, capacity=4):
    state, cache, traces = None, deque(maxlen=capacity), []
    clips = list(video.split(model.local_frames,1))
    for index, clip in enumerate(clips):
        if reset_recent and index*model.local_frames == prefix:
            state = None
            cache.clear()
        if clear_last and index == len(clips)-1:
            state = torch.zeros_like(state) if state is not None else None
        if index == corrupt:
            clip = torch.zeros_like(clip)
        out = model.stream_clip(clip,state)
        state = out['final_state']
        cache.append(out['frame_outputs'].mean(2))
        traces.append(dict(state=state, sequence=torch.cat(list(cache),1), frame=out['frame_outputs'][:,-1]))
    return traces


def _head(model, checkpoint, task, recent, source):
    from .echo import TemporalTask
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('source') is None or file_hash(payload['source']) != file_hash(source):
        raise ValueError('Diagnostic frozen head must reference exactly this backbone checkpoint')
    if any(not torch.equal(value.cpu(),payload['model_state_dict'][key].cpu()) for key,value in model.state_dict().items()):
        raise ValueError('Frozen head backbone changed or belongs to a different source')
    head = TemporalTask(model,task,'frozen_linear',recent,'final',False).to(next(model.parameters()).device).eval()
    head.head.load_state_dict({k[5:]:v for k,v in payload['model'].items() if k.startswith('head.')},strict=True)
    return head


def run_diagnostics(cfg, source):
    rng = get_rng_state()
    try:
        return _run_diagnostics(cfg,source)
    finally:
        set_rng_state(rng)


def _run_diagnostics(cfg, source):
    from models.final_temporal_mae import load_final_model
    from utils.final_temporal_data import WindowDataset, _load_manifest
    from utils.final_temporal_training import deterministic_masks
    output = layout(cfg,cfg['stage']['id'])['result']
    output.mkdir(parents=True,exist_ok=True)
    heads = cfg['stage'].get('heads',{})
    identity = dict(config=cfg,source=file_hash(source),manifest=file_hash(cfg['data']['manifest']),
                    heads={k:file_hash(v) for k,v in heads.items()},code=code_identity())
    protocol = output/'protocol.json'
    if protocol.exists():
        import json
        if json.loads(protocol.read_text(encoding='utf-8')) != identity:
            raise ValueError('Diagnostic identity changed')
    write_json(protocol,identity)
    if verify_done(output):
        return
    rng = get_rng_state()
    started = time.perf_counter()
    beat = Heartbeat(output,cfg['runtime']['heartbeat_seconds'])
    model, native, _ = load_final_model(source,seed=cfg['training']['seed'])
    device = torch.device(cfg['runtime']['device'])
    model = model.to(device).eval()
    original = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(True)
    model.gradient_checkpointing = False  # Hooks are time-indexed, not recomputation-indexed.
    recent, local = cfg['data']['recent_frames'], model.local_frames
    from functools import partial
    run_stream = partial(stream,capacity=recent//local)
    manifest = _load_manifest(cfg['data']['manifest'])
    maximum = 2 if cfg.get('smoke') else int(cfg['stage'].get('panel_cases',64))
    gradient_cases = min(maximum,int(cfg['stage'].get('gradient_cases',2)))
    records = []
    for case in manifest['train'][:maximum]:
        h = min(cfg['data']['max_prefix'],(case['frames']-recent)//local*local)
        records.append(dict(patient=case['patient'],recent_start=h,H=h))
    dataset = WindowDataset(manifest,'train',task='mae',recent_frames=recent,local_frames=local,
                            max_prefix=cfg['data']['max_prefix'],records=records)
    readers = {task:_head(model,path,task,recent,source) for task,path in heads.items()}
    # Frozen head construction disables gradients on its shared backbone; restore for D1.
    model.requires_grad_(True)
    routes = {'normal':(False,False),'feature_context_zero':(True,False),
              'decoder_context_zero':(False,True),'both_context_zero':(True,True)}
    rows, gradients, recovery, population = [], [], [], []
    try:
        for index in range(len(dataset)):
            seed_everything(cfg['training']['seed'])
            sample = dataset[index]
            video = sample['video'][None].to(device)
            if model.in_chans == 1:
                video = video[:,:,:1]
            masks = deterministic_masks(model,dict(video=video,record_index=torch.tensor([index])),cfg['training']['seed'])
            population.append(dict(patient=sample['patient'],indices=sample['frame_indices'].tolist(),prefix=sample['prefix_frames'],
                                   video_frames=dataset.cases[sample['patient']]['frames'],fps=sample['fps'],
                                   intensity_std=float(video.std()),mean_absolute_frame_change=float((video[:,1:]-video[:,:-1]).abs().mean()),
                                   proxy_scope='pixel statistics only; not a clinical quality annotation'))
            if index < gradient_cases:
                for route,(f,d) in routes.items():
                    gradients.append(dict(patient=sample['patient'],route=route,
                                           **gradient_credit(model,video,masks,f,d)))
            with torch.no_grad():
                clean = run_stream(model,video,sample['prefix_frames'])
                reference = clean[-1]
                edited = video.clone()
                edited[:,-1] = 0
                identity_out = model.stream_clip(edited[:,-local:],clean[-2]['state'] if len(clean)>1 else None)['frame_outputs']
                pair = model.stream_clip(video[:,-local:],clean[-2]['state'] if len(clean)>1 else None)['frame_outputs']
                recovery.append(dict(patient=sample['patient'],intervention='replace_last_source_frame_in_tubelet',clip=len(clean)-1,
                                     same_pair_feature_l2=float((pair[:,-1]-pair[:,-2]).norm()),
                                     target_response_l2=float((identity_out[:,-1]-pair[:,-1]).norm()),
                                     paired_frame_response_l2=float((identity_out[:,-2]-pair[:,-2]).norm())))
                for route,(f,d) in routes.items():
                    with blocked_reads(model,f,d):
                        loss,full_loss = last_clip_loss(model,video,masks,return_full=True)
                        changed = run_stream(model,video,sample['prefix_frames'])[-1]
                    row = dict(patient=sample['patient'],route=route,reconstruction_last_clip=float(loss),reconstruction_all_clips=float(full_loss),
                               final_sequence_l2=float((changed['sequence']-reference['sequence']).norm()))
                    for task, reader in readers.items():
                        key = 'sequence' if task == 'ef' else 'frame'
                        a = reader([dict(features=reference[key][0])])
                        b = reader([dict(features=changed[key][0])])
                        row[task+'_fixed_head_prediction_l2'] = float((a-b).norm())
                        if task == 'ef':
                            row['ef_fixed_head_delta_pp'] = float((b-a).squeeze()*100)
                        else:
                            row['seg_fixed_head_changed_pixel_fraction'] = float((a.argmax(1)!=b.argmax(1)).float().mean())
                    rows.append(row)
                for intervention, altered in (
                    ('clear_last_state_only_keep_inherited_cache',run_stream(model,video,sample['prefix_frames'],clear_last=True)),
                    ('reset_at_recent_boundary_recompute_cache',run_stream(model,video,sample['prefix_frames'],reset_recent=True)),
                    ('repeat_prefix_clip_keep_recent',run_stream(model,torch.cat((video[:,:local].repeat(1,sample['prefix_frames']//local,1,1,1),video[:,sample['prefix_frames']:] ),1),sample['prefix_frames']))):
                    recovery.append(dict(patient=sample['patient'],intervention=intervention,clip=len(clean)-1,
                                         sequence_l2=float((altered[-1]['sequence']-reference['sequence']).norm()),
                                         state_l2=float((altered[-1]['state']-reference['state']).norm())))
                disturbed = run_stream(model,video,sample['prefix_frames'],corrupt=0)
                for step,(a,b) in enumerate(zip(clean,disturbed)):
                    row = dict(patient=sample['patient'],intervention='one_black_clip_then_clean',clip=step,
                               corrupt_clip_evicted=step>=recent//local,sequence_l2=float((b['sequence']-a['sequence']).norm()),
                               state_l2=float((b['state']-a['state']).norm()))
                    if 'ef' in readers:
                        reader = readers['ef']
                        row['ef_fixed_head_delta_pp'] = float((reader([dict(features=b['sequence'][0])])-reader([dict(features=a['sequence'][0])])).squeeze()*100)
                    recovery.append(row)
            beat(phase='diagnostics',completed=index+1,total=len(dataset))
        if any(not torch.equal(value.cpu(),original[key]) for key,value in model.state_dict().items()):
            raise RuntimeError('Read-only diagnostics modified the source model')
        if file_hash(source) != identity['source']:
            raise ValueError('Source checkpoint changed during diagnostics')
        csv_rows(output/'paths.csv',rows)
        csv_rows(output/'history_recovery.csv',recovery)
        write_json(output/'temporal_credit.json',gradients)
        write_json(output/'population.json',population)
        write_json(output/'summary.json',dict(stage=cfg['stage']['id'],kind='forward_diagnostics',
                    metrics=dict(cases=len(population),gradient_cases=gradient_cases,wall_seconds=time.perf_counter()-started),
                    source_epoch=torch.load(source,map_location='cpu',weights_only=False).get('epoch'),
                    native_model=native['model'],source_unchanged=True,
                    scope='TRAIN diagnostic panel; fixed-head sensitivity is not refitted readability or accuracy',
                    limitations=['Zero-context intervention is off-distribution, not retraining ablation.',
                                 'Read edges are zeroed at every clip during a full re-unroll; changes are not additive path attribution.',
                                 'Single black clip is artificial stress, not a real probe switch.',
                                 'Short videos may not extend beyond FIFO eviction; inspect clip rows.',
                                 'D3 H/F readability is in separate matched probe stages; endpoint state norm is not clinical semantics.']))
        mark_done(output,['protocol.json','summary.json','paths.csv','history_recovery.csv','temporal_credit.json','population.json'])
        beat(force=True,phase='done')
    finally:
        for p, flag in zip(model.parameters(),flags):
            p.requires_grad_(flag)
        set_rng_state(rng)
