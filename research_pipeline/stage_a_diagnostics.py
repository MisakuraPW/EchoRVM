"""Fixed labelled VAL panel; no fitting, no TEST access, no state-only probes."""

import json
from pathlib import Path
import time

import torch

from .artifacts import code_identity, digest, file_hash, layout, mark_done, verify_done, write_json
from .engine import Heartbeat, build_experiment, csv_rows
from .forward_diagnostics import blocked_reads, gradient_credit
from .mae_execution import local_objective
from .metrics import segmentation_row


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_task(request_path, checkpoint, source_hash, device):
    path = Path(request_path)
    if not verify_done(path.parent):
        raise ValueError('A diagnostic head must be a completed VAL-selected FT model')
    summary = read(path.parent/'summary.json')
    request = read(path)
    if (summary['selection'] != 'VAL only' or summary['mode'] != 'finetune'
            or file_hash(request['source']) != source_hash
            or file_hash(checkpoint) != summary['checkpoints'][summary['best_file']]['sha256']):
        raise ValueError('Diagnostic head/SSL binding mismatch')
    exp = build_experiment(request['config'], request['source'], device)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    exp.model.load_state_dict(payload['model'],strict=True)
    return exp.model.requires_grad_(False).eval()


def task_steps(reader, video, task, target, *, disturb=False, prefix=None, zero_feature=False):
    """Use the exact FT head, preserving prior descriptors in the isolated read test."""
    steps = [] if prefix is None else list(prefix)
    state = steps[-1]['state'] if steps else None
    local = reader.backbone.local_frames
    for i in range(len(steps),video.shape[1]//local):
        clip = video[:,i*local:(i+1)*local]
        if disturb and i==0:
            clip = torch.zeros_like(clip)
        with blocked_reads(reader.backbone,feature=zero_feature):
            output = reader.backbone.stream_clip(clip,state)
        state = output['final_state']
        feature = output['frame_outputs']
        step = dict(state=state, descriptors=feature.mean(2), frame=feature[:,-1])
        if task=='ef':
            sequence = torch.cat([s['descriptors'] for s in steps[-3:]]+[step['descriptors']],1)
            estimate = float(reader([dict(features=sequence[0])]).item())*100
            step['metrics'] = dict(prediction_pp=estimate, absolute_error_pp=abs(estimate-target))
        elif i==video.shape[1]//local-1:
            mask = reader([dict(features=step['frame'][0])]).argmax(1)[0].cpu().numpy()
            step['metrics'] = segmentation_row(mask,target,patient='',source_frame=0,boundaries=True)
        else:
            step['metrics'] = {}
        steps.append(step)
    return steps


def masked_routes(model, video, masks):
    state = None
    clips = list(video.split(model.local_frames,1))
    for i,clip in enumerate(clips[:-1]):
        _,_,state = local_objective(model,clip,masks[:,i],None,state)
    rows = {}
    for name,feature,decoder in [('normal',False,False),('feature_S_zero',True,False),('decoder_S_zero',False,True)]:
        with blocked_reads(model,feature,decoder):
            error,count,_ = local_objective(model,clips[-1],masks[:,-1],None,state)
        rows[name] = float((error/count).mean())
    return rows


def verify_visibility(model, video, masks):
    local = model.local_frames
    gt,gh,gw = model.token_grid
    hidden = masks[:,0].reshape(-1,gt,gh,gw).repeat_interleave(model.tubelet_size,1)
    hidden = hidden.repeat_interleave(model.patch_size,2).repeat_interleave(model.patch_size,3)
    with torch.no_grad():
        normal = model(video,masks=masks)
        modified = video.clone()
        modified[:,:local] = torch.where(hidden[:,:,None],1-modified[:,:local],modified[:,:local])
        masked = model(modified,masks=masks)
        torch.testing.assert_close(masked['pred'],normal['pred'])
        torch.testing.assert_close(masked['states'],normal['states'])
        modified = video.clone()
        modified[:,-local:] = 1-modified[:,-local:]
        future = model(modified,masks=masks)
        n = model.patch_embed.num_patches
        torch.testing.assert_close(future['pred'][:,:-n],normal['pred'][:,:-n])
        torch.testing.assert_close(future['states'][:,:-1],normal['states'][:,:-1])
    return dict(hidden_pixels_isolated=True, future_clips_isolated=True,
                scope='Fixed mask, clip-level causality; not within-clip frame causality')


def run_panel(cfg, source):
    from models.final_temporal_mae import load_final_model
    from utils.final_temporal_data import WindowDataset, _complete_trace, _load_manifest
    from utils.final_temporal_training import deterministic_masks

    stage = cfg['stage']
    output = layout(cfg,stage['id'])['result']
    source_hash = file_hash(source)
    identity = dict(config=cfg,source=source_hash,manifest=file_hash(cfg['data']['manifest']),
                    heads={task:file_hash(path) for task,path in stage['heads'].items()},
                    requests={task:file_hash(path) for task,path in stage['head_requests'].items()},code=code_identity())
    protocol = output/'protocol.json'
    if protocol.exists() and read(protocol)!=identity:
        raise ValueError('Stage A diagnostic identity changed')
    if verify_done(output):
        return
    write_json(protocol,identity)
    started = time.perf_counter()
    budget = None if cfg.get('smoke') else .1*sum(read(Path(p)/'summary.json')['wall_seconds'] for p in
                                               [stage['budget_ssl'],*stage['budget_stages'].values()])
    maximum = 2 if cfg.get('smoke') else min(128,stage['panel_cases'])
    data = cfg['data']
    manifest = _load_manifest(data['manifest'])
    records = []
    for case in manifest['val']:
        if case['ef'] is None:
            continue
        targets = [t['frame'] for t in case['traces'] if _complete_trace(case,t['frame'],64,16)]
        if targets:
            frame = min(targets)
            records.append(dict(patient=case['patient'],target_frame=frame,recent_start=frame-63,H=0))
        if len(records)==maximum:
            break
    if not records:
        raise ValueError('No matched labelled VAL panel with a real W64 window')
    dataset = WindowDataset(manifest,'val',task='seg',local_frames=16,recent_frames=64,max_prefix=0,records=records)
    device = torch.device(cfg['runtime']['device'])
    model,_,_ = load_final_model(source,seed=cfg['training']['seed'])
    model = model.to(device).eval()
    model.gradient_checkpointing = False
    readers = {task:load_task(stage['head_requests'][task],path,source_hash,device)
               for task,path in stage['heads'].items()}
    rows, recovery, gradients = [],[],[]
    beat = Heartbeat(output,cfg['runtime']['heartbeat_seconds'])
    for i,sample in enumerate(dataset):
        if budget is not None and time.perf_counter()-started >= budget:
            break
        video = sample['video'][None].to(device)
        masks = deterministic_masks(model,dict(video=video,record_index=torch.tensor([i])),cfg['training']['seed'])
        patient = sample['patient']
        if i<stage['gradient_cases']:
            credit = gradient_credit(model,video,masks)
            early = [*credit['states'][:-1],*credit['time_specific_writes'][:-1]]
            if not all(g['connected'] and g['finite'] for g in early):
                raise RuntimeError('Disconnected/nonfinite past credit in the last-clip-only objective')
            gradients.append(dict(patient=patient,**verify_visibility(model,video,masks),**credit))
        with torch.no_grad():
            routes = masked_routes(model,video,masks)
            row = dict(patient=patient,target_frame=sample['target_frame'],**{'masked_mse_'+k:v for k,v in routes.items()})
            for task,reader in readers.items():
                target = float(dataset.cases[patient]['ef']) if task=='ef' else sample['mask'].cpu().numpy()
                clean = task_steps(reader,video,task,target)
                cut = task_steps(reader,video,task,target,prefix=clean[:-1],zero_feature=True)
                changed = task_steps(reader,video,task,target,disturb=True)
                for name,values in [('normal',clean),('feature_S_zero',cut)]:
                    for key,value in values[-1]['metrics'].items():
                        if key in {'dice','assd','hd95','absolute_error_pp','prediction_pp'}:
                            row[f'{task}_{name}_{key}'] = value
                for j,(a,b) in enumerate(zip(clean,changed)):
                    for name,values in [('clean',a),('first_clip_black',b)]:
                        recovery.append(dict(patient=patient,task=task,route=name,clip=j,
                                             context_frames=16*(j+1),clean_clips_since_disturbance=j,
                                             state_delta_l2=float((values['state']-a['state']).norm()),
                                             descriptor_delta_l2=float((values['descriptors']-a['descriptors']).norm()),
                                             **{k:v for k,v in values['metrics'].items() if k in
                                                {'dice','assd','hd95','absolute_error_pp','prediction_pp'}}))
            rows.append(row)
        beat(phase='diagnostics',completed=i+1,total=len(records))
    write_json(output/'panel.json',records)
    csv_rows(output/'paths.csv',rows)
    csv_rows(output/'history_recovery.csv',recovery)
    write_json(output/'temporal_credit.json',gradients)
    write_json(output/'summary.json',dict(stage=stage['id'],kind=stage['kind'],smoke=bool(cfg.get('smoke')),
                metrics=dict(cases=len(rows),gradient_cases=len(gradients),wall_seconds=time.perf_counter()-started),
                budget_seconds=budget,panel_sha256=digest(records),budget_truncated=len(rows)!=len(records),
                scope='VAL-only fixed case panel; masked SSL reconstruction and independent FT task heads',
                limitations=['Final-clip-only feature intervention retains earlier descriptors and state.',
                             'Decoder intervention is SSL only; task exits never use the reconstruction decoder.',
                             'First clip blackout is a stress test. No FIFO eviction occurs within four clips.',
                             'EF prefix predictions are diagnostic, not the W64 endpoint. Dice uses a real final traced frame.']))
    artifacts = ['protocol.json','panel.json','summary.json','temporal_credit.json']
    mark_done(output,artifacts+(['paths.csv','history_recovery.csv'] if rows else []))
    beat(force=True,phase='done')
