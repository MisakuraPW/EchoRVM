"""Stage-two question-based endpoint audits. No backbone updates or new MAE training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.temporal_mae import TemporalMAE
from models.ef_readout import TemporalEFReadout
from tools.diagnose_stage2_interfaces import (SegViews, data_signature, make_loader,
    save_json, seg_probe, validity_rows, file_hash, write_csv, load_model,
    regression_metrics, ridge_fit, ridge_apply)
from tools.run_temporal_research import archive_analysis
from utils.temporal_data import TemporalEchoDataset
from utils.streaming_features import StreamingFeatureCache, ef_sequences
from utils.stage2_diagnostics import paired_bootstrap, tubelet_weights
from utils.seed import seed_everything

VERSION = 'stage2_questions_v1'


def code_hash():
    paths = sorted({*ROOT.joinpath('models').glob('*.py'), *ROOT.joinpath('utils').glob('*.py'),
        *ROOT.joinpath('optim').glob('*.py'),
        Path(__file__).resolve(), ROOT/'tools/diagnose_stage2_interfaces.py',
        ROOT/'tools/evaluate_temporal_mae.py', ROOT/'tools/evaluate_representation_quality.py',
        ROOT/'trainers/train_rmae.py', ROOT/'trainers/train_finetune.py',
        ROOT/'tools/tune_rmae_runtime.py', ROOT/'tools/run_temporal_research.py',
        ROOT/'echo_aug_validation/io_utils.py'})
    return hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_bytes()
                                  for p in paths)).hexdigest()


@torch.inference_mode()
def ef_batch(model, video, indices, recent_frames):
    """One actual prefix followed by a bounded recent window; no overlap/replay."""
    if recent_frames % model.local_frames or video.shape[1] % model.local_frames:
        raise ValueError('Prefix/recent window must contain complete local clips')
    if model.memory_mode == 'none':
        if model.local_frames != recent_frames:
            raise ValueError('No-memory endpoint is reserved for the whole-window reference')
        clip = video[:, -recent_frames:]
        features = model.frame_features(model.forward_features(clip)).mean(2)
        return dict(joint_recent=features)
    cache = StreamingFeatureCache(model, recent_frames//model.local_frames)
    ids = [str(i) for i in range(len(video))]
    for start in range(0, video.shape[1], model.local_frames):
        cache.update(video[:, start:start+model.local_frames], ids,
                     indices[:, start:start+model.local_frames])
    sequences = ef_sequences(cache.read())
    # Group B: reset at the same recent-window boundary as the joint reference.
    # Group C: keep the prefix-conditioned cache, compare explicit state vs empty.
    cache.reset()
    for start in range(video.shape[1]-recent_frames, video.shape[1], model.local_frames):
        cache.update(video[:, start:start+model.local_frames], ids,
                     indices[:, start:start+model.local_frames])
    recent = cache.read()
    sequences.update(last=recent['last'], cache=recent['fused'], state=recent['final_state'])
    return sequences


@torch.inference_mode()
def seg_batch(model, video, valid, targets, context):
    if context == 'repeat_target':
        video = video[torch.arange(len(video), device=video.device), targets][:, None].expand_as(video)
        # Keep absent source context absent, even for this distribution intervention.
        video = video * valid[:, :, None, None, None]
    result = model.diagnostic_features(video, valid)
    native = result['features']
    good, partial = tubelet_weights(valid, model.tubelet_size)
    expanded = model.frame_features(native, good[:, :, None].expand(-1, -1, native.shape[2]))
    expanded = expanded * good.repeat_interleave(model.tubelet_size, 1)[:, :, None, None]
    chosen = expanded[torch.arange(len(video), device=video.device), targets]
    gh, gw = model.token_grid[1:]
    maps = dict(fused=chosen.transpose(1, 2).reshape(len(video), model.embed_dim, gh, gw))
    if model.frame_readout != 'repeat':
        base = native[torch.arange(len(video), device=video.device), targets//model.tubelet_size]
        maps['native'] = base.transpose(1, 2).reshape_as(maps['fused'])
    return maps, dict(complete=good.sum(1), partial=partial.sum(1))


def tune_extraction(model, ef_sample, seg_sample, args, device):
    if device.type != 'cuda':
        return 1, [dict(batch_size=1, status='cpu')]
    trials = []
    for size in sorted({1, args.max_batch_size, *(s for s in (2,4,8,16,32) if s <= args.max_batch_size)}):
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        allowance = min(total*.85, torch.cuda.memory_allocated(device)+free*.85)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            def trial():
                with torch.autocast('cuda'):
                    video = ef_sample['video'][None].expand(size,-1,-1,-1,-1).to(device)
                    idx = torch.arange(video.shape[1], device=device)[None].expand(size,-1)
                    ef_batch(model, video, idx, args.recent_frames)
                    video = seg_sample['video'][None].expand(size,-1,-1,-1,-1).to(device)
                    valid = seg_sample['frame_valid'][None].expand(size,-1).to(device)
                    target = torch.full((size,), int(seg_sample['target_index']), device=device)
                    seg_batch(model, video, valid, target, 'real')
            trial()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(2):
                trial()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device)
            trials.append(dict(batch_size=size, status='ok' if peak < allowance else 'headroom',
                               samples_per_second=2*size/(time.perf_counter()-start), peak_bytes=peak))
            if peak >= allowance:
                break
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch_size=size, status='oom'))
            break
    torch.cuda.empty_cache()
    good = [r for r in trials if r['status']=='ok']
    if not good:
        raise RuntimeError('No extraction batch fits with memory reserve')
    best = max(r['samples_per_second'] for r in good)
    return min(r['batch_size'] for r in good if r['samples_per_second'] >= .97*best), trials


def extract(model, dataset, args, device, task, context, cache, identity):
    signature = json.dumps(dict(identity=identity, files=data_signature(dataset),task=task,context=context),sort_keys=True)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            if str(data['signature'].item()) != signature:
                raise RuntimeError('Cache protocol changed; use a new run_tag')
            out = {k:data[k].copy() for k in data.files if k not in {'signature','rows','excluded'}}
            out.update(rows=json.loads(str(data['rows'].item())),excluded=json.loads(str(data['excluded'].item())))
            return out
    columns, rows, excluded = {}, [], []
    for batch in tqdm(make_loader(dataset,args), desc=f'{task} {context} extract'):
        if task == 'ef':
            keep = batch['frame_valid'].all(1)
            excluded.extend(dict(id=batch['id'][i],valid_frames=int(batch['frame_valid'][i].sum()),
                                 reason='incomplete real prefix+recent window') for i in range(len(keep)) if not keep[i])
            if not bool(keep.any()):
                continue
            for key,value in list(batch.items()):
                batch[key] = value[keep] if torch.is_tensor(value) else [v for v,k in zip(value,keep) if k]
        video, valid = batch['video'].to(device), batch['frame_valid'].to(device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type=='cuda'):
            if task == 'ef':
                features = ef_batch(model, video, batch['frame_indices'].to(device), args.recent_frames)
                for i,case in enumerate(batch['id']):
                    idx = batch['frame_indices'][i]
                    boundary = args.prefix_frames
                    rows.append(dict(id=case,patient=case,source_start=int(idx[0]),
                        recent_start=int(idx[boundary]),source_end=int(idx[-1]),
                        prefix_frames=boundary,recent_frames=args.recent_frames,
                        length_extrapolation=video.shape[1]>model.frames))
            else:
                features, counts = seg_batch(model,video,valid,batch['target_index'].to(device),context)
                rows.extend(validity_rows(batch,model,'seg',counts))
                images = batch['video'][torch.arange(len(video)),batch['target_index']].mean(1)
                columns.setdefault('images',[]).append(images.numpy().astype(np.float16))
        for key,value in features.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite features')
            columns.setdefault(key,[]).append(value.float().cpu().numpy().astype(np.float16))
        columns.setdefault('y',[]).append(batch['target' if task=='ef' else 'mask'].numpy())
    if not rows:
        raise ValueError('No eligible cases; need real prefix+recent frames, never repeated padding')
    out = {k:np.concatenate(v) for k,v in columns.items()}
    cache.parent.mkdir(parents=True,exist_ok=True)
    temp = cache.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f,signature=np.asarray(signature),rows=np.asarray(json.dumps(rows)),
                 excluded=np.asarray(json.dumps(excluded)),**out)
    temp.replace(cache)
    out.update(rows=rows,excluded=excluded)
    return out


def plot_loss(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(6,3.5))
    ax.plot([r['step'] for r in rows],[r['loss'] for r in rows])
    ax.set(xlabel='Optimizer step',ylabel='Training loss',title='Frozen-backbone probe (not MAE)')
    fig.tight_layout()
    fig.savefig(path,dpi=130)
    plt.close(fig)


def fit_ef(train, val, mode, args, device):
    x,xv = torch.from_numpy(train[mode]).float(),torch.from_numpy(val[mode]).float()
    y,yv = torch.from_numpy(train['y']).float(),torch.from_numpy(val['y']).float()
    if len(y)<2 or len(yv)<2:
        raise ValueError('At least two complete-history cases per split are required')
    if args.ef_head == 'ridge':
        pred = ridge_apply(ridge_fit(x.mean(1),y,args.ridge_alpha),xv.mean(1))
        parameters, losses = x.shape[-1]+1, []
    else:
        slots = args.history_slots if mode.endswith(('_empty','_history')) else 0
        recent = x[:, :-slots] if slots else x
        # Real/empty history controls use the SAME recent-cache normalization.
        mean,std = recent.mean((0,1),keepdim=True),recent.std((0,1),unbiased=False,keepdim=True).clamp_min(1e-5)
        x,xv = (x-mean)/std,(xv-mean)/std
        ym,ys = y.mean(),y.std(unbiased=False).clamp_min(1)
        torch.manual_seed(args.seed+1001)
        head = TemporalEFReadout(x.shape[-1],args.ef_hidden).to(device)
        opt = torch.optim.AdamW(head.parameters(),lr=args.ef_lr,weight_decay=.01)
        rng = torch.Generator().manual_seed(args.seed)
        losses = []
        for step in tqdm(range(args.ef_steps),desc='EF '+mode):
            idx = torch.randint(len(x),(min(args.probe_batch_size,len(x)),),generator=rng)
            prediction = head(x[idx].to(device),slots)
            loss = F.smooth_l1_loss(prediction,((y[idx]-ym)/ys).to(device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite EF loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            opt.step()
            losses.append(dict(step=step+1,loss=float(loss.detach())))
        head.eval()
        with torch.inference_mode():
            pred = torch.cat([head(v.to(device),slots).cpu() for v in xv.split(args.probe_batch_size)])*ys+ym
        parameters = sum(p.numel() for p in head.parameters())
    metrics = regression_metrics(pred,yv)
    metrics = {k:(float(v) if np.isfinite(v) else None) for k,v in metrics.items()}
    metrics.update(parameters=parameters,head=args.ef_head,steps=0 if args.ef_head=='ridge' else args.ef_steps)
    rows = [dict(r,target=float(yv[i]),prediction=float(pred[i]),error=float(abs(pred[i]-yv[i])))
            for i,r in enumerate(val['rows'])]
    return metrics,rows,losses


@torch.inference_mode()
def benchmark_stream(model, sample, device):
    video=sample['video'][None].to(device)
    length=model.local_frames
    sync=lambda: torch.cuda.synchronize(device) if device.type=='cuda' else None
    capacity=model.frames//length
    cache=StreamingFeatureCache(model,capacity)
    warmup=max(2,capacity)
    timings=[]
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup+5):
        clip=video[:,:length]
        indices=torch.arange(i*length,(i+1)*length,device=device)[None]
        sync()
        start=time.perf_counter()
        with torch.autocast(device.type,enabled=device.type=='cuda'):
            cache.update(clip,['benchmark'],indices)
        sync()
        if i>=warmup:
            timings.append((time.perf_counter()-start)*1000)
    stored={id(v):v for e in cache.entries for v in e.values() if torch.is_tensor(v)}
    for v in (cache.state,cache.short_state):
        if v is not None:
            stored[id(v)]=v
    return dict(batch_size=1,local_frames=length,cache_frames=model.frames,measured_updates=5,
        clip_update_ms_p50=float(np.median(timings)),clip_update_ms_p95=float(np.quantile(timings,.95)),
        resident_fifo_bytes=sum(v.numel()*v.element_size() for v in stored.values()),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        max_buffer_intervals=length-1,
        note='GPU-resident repeated workload; excludes decoding, H2D, EF head and acquisition wait. '
             'Milliseconds are compute latency, not clinical end-to-end latency; source FPS not assumed.')


def fit_seg(train, val, mode, args, device, tubelet, out):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    boundaries = []
    def callback(start,pred,target):
        for j,(p,y) in enumerate(zip(pred.numpy().astype(bool),target.numpy().astype(bool))):
            pb,yb = p ^ binary_erosion(p),y ^ binary_erosion(y)
            if pb.any() and yb.any():
                dist = np.concatenate((distance_transform_edt(~yb)[pb],distance_transform_edt(~pb)[yb]))
                assd,hd95 = float(dist.mean()),float(np.quantile(dist,.95))
            else:
                assd,hd95 = None,None
            boundaries.append(dict(boundary_mean_px=assd,hd95_px=hd95,empty_prediction=not bool(p.any())))
            if start+j<8:
                fig,ax = plt.subplots(figsize=(3,3))
                ax.imshow(val['images'][start+j],cmap='gray',vmin=0,vmax=1)
                if yb.any():
                    ax.contour(y,levels=[.5],colors=['lime'],linewidths=.8)
                if pb.any():
                    ax.contour(p,levels=[.5],colors=['magenta'],linewidths=.8)
                ax.set_title('GT green / prediction magenta',fontsize=8)
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(out/f'overlay_{mode}_{start+j:03d}.png',dpi=110)
                plt.close(fig)
    metrics,rows,losses = seg_probe(train,val,mode,args,device,tubelet,callback)
    for row,extra in zip(rows,boundaries):
        row.update(extra)
    for offset in sorted({r['offset'] for r in rows}):
        metrics[f'dice_offset_{offset}'] = float(np.mean([r['dice'] for r in rows if r['offset']==offset]))
    # Two traced phases are ordered by annotated area, never inferred cardiac labels.
    areas = {}
    for row,y in zip(rows,val['y']):
        areas.setdefault(row['patient'],{})[row['source_frame']] = int((y==1).sum())
    for row in rows:
        values = areas[row['patient']]
        row['traced_phase'] = ('larger_area' if values[row['source_frame']]==max(values.values())
                               else 'smaller_area')
    metrics['dice_p10'] = float(np.quantile([r['dice'] for r in rows],.1))
    metrics['empty_predictions'] = sum(r['empty_prediction'] for r in rows)
    for phase in ('larger_area','smaller_area'):
        values = [r['dice'] for r in rows if r['traced_phase']==phase]
        metrics['dice_'+phase] = float(np.mean(values)) if values else None
    values = [r['boundary_mean_px'] for r in rows if r['boundary_mean_px'] is not None]
    metrics['boundary_mean_px_nonempty'] = float(np.mean(values)) if values else None
    return metrics,rows,losses


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    model,cfg,meta = load_model(args.checkpoint,device)
    if not isinstance(model,TemporalMAE) or model.frequency_conditioned or model.frequency_loss_weight:
        raise ValueError('Stage 2 requires a pure TemporalMAE checkpoint')
    if model.img_size!=112 or model.frames != args.recent_frames or args.prefix_frames % model.local_frames:
        raise ValueError('Require 112px, recent_frames=native frames, prefix divisible by local_frames')
    if model.memory_mode=='none' and model.local_frames!=args.recent_frames:
        raise ValueError('No-memory input must be the whole-window reference')
    args.input_protocol = meta['input_protocol']
    args.seg_frames,args.seg_target_index = model.frames,model.frames-2
    args.offset_probe = True
    # Identical two target positions even for tubelet1; only its indexing changes.
    args.history_slots = 1 if model.memory_mode=='global' else model.memory_grid**2
    model.requires_grad_(False).eval()
    model.gradient_checkpointing = False
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    scientific = {k:v for k,v in vars(args).items() if k not in
        {'output_dir','cache_dir','batch_size','max_batch_size','num_workers','prefetch_factor','cpu_threads','keep_cache'}}
    scientific.update(version=VERSION,checkpoint_sha256=file_hash(args.checkpoint),code_sha256=code_hash(),
        filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
        traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'))
    data = {}
    for split in ('train','val'):
        data['ef_'+split] = TemporalEchoDataset(args.data_root,split,args.prefix_frames+args.recent_frames,
            channels=model.in_chans,input_protocol=args.input_protocol,limit=getattr(args,'ef_'+split+'_cases'),
            seed=args.seed,random_start=False)
        data['seg_'+split] = SegViews(args.data_root,split,args,model.in_chans,2)
    for task in ('ef','seg'):
        ids = lambda d: d.ids if task=='ef' else d.patient_ids
        if set(ids(data[task+'_train'])) & set(ids(data[task+'_val'])):
            raise ValueError('Train/val patient overlap')
    scientific['data_manifest'] = {k:data_signature(v) for k,v in data.items()}
    path = out/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=scientific:
        raise RuntimeError('Protocol/checkpoint/data changed; use a new output directory')
    save_json(path,scientific)
    if (out/'DONE').exists():
        if not (out/'metrics.json').exists():
            raise RuntimeError('DONE without metrics')
        return
    trials = []
    if not args.batch_size:
        args.batch_size,trials = tune_extraction(model,data['ef_train'][0],data['seg_train'][0],args,device)
    save_json(out/'runtime.json',dict(batch_size=args.batch_size,num_workers=args.num_workers,trials=trials,
                                    device=str(device),metadata=meta))
    cache = Path(args.cache_dir)
    started = time.perf_counter()
    metrics, predictions = dict(ef={},seg={},paired=[],
        inference=benchmark_stream(model,data['ef_train'][0],device),
        parameters_including_mae_decoder=sum(p.numel() for p in model.parameters()),
        source_checkpoint_bytes=Path(args.checkpoint).stat().st_size),{}
    owned = []
    def get(key,context='real'):
        save_json(out/'status.json',dict(stage='extract_'+key+'_'+context,updated_unix=time.time()))
        path = cache/(key+'_'+context+'.npz')
        owned.append(path)
        result = extract(model,data[key],args,device,key.split('_')[0],context,path,scientific)
        write_csv(out/('validity_'+key+'_'+context+'.csv'),result['rows'])
        write_csv(out/('excluded_'+key+'_'+context+'.csv'),result['excluded'])
        return result
    tr,va = get('ef_train'),get('ef_val')
    modes = (['joint_recent'] if model.memory_mode=='none' else
             ['last','cache','cache_empty','cache_history','local_empty','local_history'])
    if args.state_probe and model.memory_mode!='none':
        modes.append('state')
    def fit(task,name,fn):
        save_json(out/'status.json',dict(stage='fit_'+task+'_'+name,updated_unix=time.time()))
        saved = out/(task+'_'+name+'.json')
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            value,rows = result['metrics'],result['rows']
        else:
            value,rows,losses = fn()
            if losses:
                write_csv(out/(task+'_'+name+'_loss.csv'),losses)
                plot_loss(losses,out/(task+'_'+name+'_loss.png'))
            save_json(saved,dict(metrics=value,rows=rows))
        write_csv(out/(task+'_'+name+'.csv'),rows)
        metrics[task][name],predictions[task+'_'+name] = value,rows
    for mode in modes:
        fit('ef',mode,lambda mode=mode:fit_ef(tr,va,mode,args,device))
    del tr,va
    for context in ('real','repeat_target'):
        tr,va = get('seg_train',context),get('seg_val',context)
        for mode in (['fused','native'] if model.frame_readout=='learned' else ['fused']):
            fit('seg',context+'_'+mode,lambda mode=mode:fit_seg(tr,va,mode,args,device,model.tubelet_size,out/context))
        del tr,va
    contrasts = [('seg','real_fused','repeat_target_fused','dice')]
    if model.frame_readout=='learned':
        contrasts.append(('seg','real_fused','real_native','dice'))
    if model.memory_mode!='none':
        contrasts += [('ef','cache','last','error'),('ef','cache_history','cache_empty','error'),
                      ('ef','local_history','local_empty','error')]
    for task,a,b,field in contrasts:
        metrics['paired'].append(dict(task=task,candidate=a,control=b,
            **paired_bootstrap(predictions[task+'_'+a],predictions[task+'_'+b],field,args.seed)))
    metrics['wall_seconds'] = time.perf_counter()-started
    save_json(out/'metrics.json',metrics)
    write_csv(out/'paired_differences.csv',metrics['paired'])
    lines = ['# Stage 2 endpoint questions','',
        'Frozen validation probes, not full fine-tuning or test performance. No per-window EF ground truth.',
        'EF last/cache/state reset at recent-window start. History/empty pairs instead use prefix-conditioned caches.',
        'All EF modes use identical complete-prefix cases; prefix-conditioned fused caches already contain history.',
        'History vs empty adds explicit access to the state BEFORE the recent window. No claim of universal superiority.',
        'repeat_target is a distribution intervention, not a separately pretrained single-frame baseline.',
        'Two views of each real traced frame have different future context. Larger/smaller traced area is not a new label.',
        'Boundary distances are pixels; empty predictions are counted, not assigned a fabricated distance.',
        'Streaming sees each complete local clip before emitting its frame maps; not zero-latency per-frame causality.',
        'A longer prefix than pretraining exposes inference-length extrapolation. Memory is reset between patients.',
        'Learned EF heads preserve ordered descriptors; ridge averages them and is only a cheap screening option.','',
        '| Task | Readout | Metric |','|---|---|---:|']
    for task in ('ef','seg'):
        field = 'mae' if task=='ef' else 'dice_patient_mean'
        lines += [f'| {task} | {name} | {m[field]:.5f} |' for name,m in metrics[task].items()]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'status.json',dict(stage='completed',updated_unix=time.time()))
    (out/'DONE').write_text(VERSION+'\n')
    archive_analysis(out)
    if not args.keep_cache:
        for path in owned:
            path.unlink(missing_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--cache_dir',required=True)
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--prefix_frames',type=int,default=64)
    p.add_argument('--recent_frames',type=int,default=64)
    p.add_argument('--batch_size',type=int,default=0)
    p.add_argument('--max_batch_size',type=int,default=16)
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--prefetch_factor',type=int,default=2)
    p.add_argument('--cpu_threads',type=int,default=4)
    p.add_argument('--ef_train_cases',type=int,default=512)
    p.add_argument('--ef_val_cases',type=int,default=256)
    p.add_argument('--seg_train_cases',type=int,default=64)
    p.add_argument('--seg_val_cases',type=int,default=64)
    p.add_argument('--ef_steps',type=int,default=400)
    p.add_argument('--seg_steps',type=int,default=200)
    p.add_argument('--ef_lr',type=float,default=.001)
    p.add_argument('--seg_lr',type=float,default=.01)
    p.add_argument('--ef_hidden',type=int,default=64)
    p.add_argument('--ef_head',choices=['attention','ridge'],default='attention')
    p.add_argument('--ridge_alpha',type=float,default=10.)
    p.add_argument('--probe_batch_size',type=int,default=32)
    p.add_argument('--state_probe',action='store_true')
    p.add_argument('--keep_cache',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    args = p.parse_args()
    if args.batch_size<0 or args.num_workers<0:
        p.error('batch_size/workers must be nonnegative')
    for key in ('prefix_frames','recent_frames','max_batch_size','prefetch_factor','cpu_threads',
                'ef_train_cases','ef_val_cases','seg_train_cases','seg_val_cases','ef_steps','seg_steps',
                'ef_lr','seg_lr','ef_hidden','ridge_alpha','probe_batch_size'):
        if getattr(args,key)<=0:
            p.error(key+' must be positive')
    if args.smoke:
        args.ef_train_cases,args.ef_val_cases=16,8
        args.seg_train_cases,args.seg_val_cases=2,2
        args.ef_steps=args.seg_steps=2
        args.batch_size,args.num_workers=1,0
    return args


if __name__=='__main__':
    evaluate(parse_args())
