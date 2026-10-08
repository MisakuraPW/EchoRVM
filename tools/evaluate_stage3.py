"""Frozen, endpoint-only memory diagnostics. Never starts MAE pretraining."""

from __future__ import annotations

import argparse
import copy
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
from tools.diagnose_stage2_interfaces import (load_model, save_json, file_hash,
    make_loader, data_signature, write_csv, SegViews, regression_metrics, ridge_fit, ridge_apply)
from tools.evaluate_stage2 import extract as extract_seg, fit_seg, plot_loss, code_hash, seg_batch
from tools.run_temporal_research import archive_analysis
from utils.temporal_data import TemporalEchoDataset
from utils.stage2_diagnostics import paired_bootstrap
from utils.stage3_diagnostics import conditions, intervention_video, stream_audit, gradient_audit
from utils.seed import seed_everything

VERSION = 'stage3_memory_v2'


def fit_head(train, key, args, device):
    x = torch.from_numpy(train[key]).float()
    y = torch.from_numpy(train['y']).float()
    slots = args.history_slots if key.endswith(('_history', '_empty')) else 0
    recent = x[:, :-slots] if slots else x
    mean = recent.mean((0, 1), keepdim=True)
    std = recent.std((0, 1), unbiased=False, keepdim=True).clamp_min(1e-5)
    x = (x-mean)/std
    ym, ys = y.mean(), y.std(unbiased=False).clamp_min(1)
    torch.manual_seed(args.seed+1001)
    head = TemporalEFReadout(x.shape[-1], args.ef_hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.ef_lr, weight_decay=.01)
    rng = torch.Generator().manual_seed(args.seed)
    losses = []
    progress = tqdm(range(args.ef_steps), desc='EF '+key)
    for step in progress:
        idx = torch.randint(len(x), (min(args.probe_batch_size, len(x)),), generator=rng)
        loss = F.smooth_l1_loss(head(x[idx].to(device), slots), ((y[idx]-ym)/ys).to(device))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite probe loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        losses.append(dict(step=step+1, loss=float(loss.detach())))
        if step % 20 == 0 or step+1 == args.ef_steps:
            progress.set_postfix(loss=f'{losses[-1]["loss"]:.4f}')
    head.eval()
    @torch.no_grad()
    def predict(values):
        values = (torch.from_numpy(values).float()-mean)/std
        return torch.cat([head(v.to(device), slots).cpu() for v in values.split(args.probe_batch_size)])*ys+ym
    return predict, losses, sum(p.numel() for p in head.parameters())


def tune_runtime(model, dataset, args, device, seg_sample=None, benchmark_fn=None):
    """Tune execution knobs only, without looking at any task labels/scores."""
    rows = []
    if args.batch_size == 0:
        args.batch_size = 1
        if device.type == 'cuda':
            sample = dataset[0]['video']
            for size in sorted({1, args.max_batch_size, *(s for s in (2,4,8,16,32) if s <= args.max_batch_size)}):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                free, total = torch.cuda.mem_get_info(device)
                allowance = min(total*.85, torch.cuda.memory_allocated(device)+free*.85)
                try:
                    def trial():
                        video = sample[None].expand(size,-1,-1,-1,-1).to(device)
                        with torch.autocast('cuda'):
                            if benchmark_fn is None:
                                stream_audit(model, video, args.recent_frames)
                            else:
                                benchmark_fn(video)
                            if seg_sample is not None:
                                frames=seg_sample['video'][None].expand(size,-1,-1,-1,-1).to(device)
                                valid=seg_sample['frame_valid'][None].expand(size,-1).to(device)
                                targets=torch.full((size,),int(seg_sample['target_index']),device=device)
                                seg_batch(model,frames,valid,targets,'real')
                    trial()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(2):
                        trial()
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved()
                    rows.append(dict(batch_size=size, status='ok' if peak<allowance else 'headroom',
                                     throughput=2*size/(time.perf_counter()-start), peak_bytes=peak))
                    if peak >= allowance:
                        break
                except torch.cuda.OutOfMemoryError:
                    rows.append(dict(batch_size=size, status='oom'))
                    break
            good = [r for r in rows if r['status']=='ok']
            if not good:
                raise RuntimeError('No extraction batch fits with memory headroom')
            speed = max(r['throughput'] for r in good)
            args.batch_size = min(r['batch_size'] for r in good if r['throughput']>=speed*.97)
            torch.cuda.empty_cache()
    worker_rows = []
    if args.auto_workers:
        for workers in sorted({0, args.num_workers, *(w for w in (2,4,8) if w<=args.num_workers)}):
            candidate = copy.copy(args)
            candidate.num_workers = workers
            loader = make_loader(dataset, candidate)
            iterator = iter(loader)
            next(iterator)
            start, seen = time.perf_counter(), 0
            for _ in range(3):
                batch = next(iterator, None)
                if batch is None:
                    break
                seen += len(batch['video'])
            if seen:
                worker_rows.append(dict(workers=workers, throughput=seen/(time.perf_counter()-start)))
            del iterator, loader
        if worker_rows:
            speed = max(r['throughput'] for r in worker_rows)
            args.num_workers = min(r['workers'] for r in worker_rows if r['throughput']>=speed*.95)
    return dict(batch_size=args.batch_size, num_workers=args.num_workers,
                batch_trials=rows, worker_trials=worker_rows,
                note='Short warm loader-only trial, not a guarantee of GPU saturation; no scientific knobs tuned.')


def extract_ef(model, dataset, args, device, split, cache, identity):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            if str(f['identity'].item()) != identity:
                raise ValueError('Cached features mismatch; choose a new run_tag')
            return {k:f[k].copy() for k in f.files if k not in {'identity','rows','excluded','traces','recovery'}}, {
                k:json.loads(str(f[k].item())) for k in ('rows','excluded','traces','recovery')}
    collected, rows, excluded, traces, recovery = {}, [], [], [], []
    specs = conditions(args.prefixes)
    if split == 'train':
        specs = [v for v in specs if v[2]=='clean']
    maximum = max(args.prefixes)
    budget = getattr(args,'ef_'+split+'_cases')
    progress = tqdm(make_loader(dataset, args), desc='stage3 EF '+split)
    for batch in progress:
        keep = batch['frame_valid'].all(1)
        for i, case in enumerate(batch['id']):
            if not keep[i]:
                excluded.append(dict(id=case, reason='incomplete maximum-history window',
                                     valid_frames=int(batch['frame_valid'][i].sum())))
        if not keep.any():
            continue
        if args.eligible_budget:
            positions = keep.nonzero().flatten()
            keep[positions[max(0,budget-len(rows)):]] = False
        ids = [v for v,k in zip(batch['id'], keep) if k]
        video = batch['video'][keep].to(device, non_blocking=True)
        indices = batch['frame_indices'][keep]
        for i, case in enumerate(ids):
            rows.append(dict(id=case, patient=case, source_start=int(indices[i,0]),
                recent_start=int(indices[i,maximum]), source_end=int(indices[i,-1])))
        collected.setdefault('y', []).append(batch['target'][keep].numpy())
        reference = []
        for name, prefix, kind in specs:
            def trajectory(index, state, short, fused):
                if name == f'history_{maximum}':
                    reference.append((state, short, fused))
                elif name in ('repeat_prefix','zero_prefix_clip'):
                    rs, rh, rf = reference[index]
                    values = dict(feature_distance_rms=(fused.float()-rf.float()).square().mean((1,2)).sqrt())
                    if state is not None:
                        distance=(state.float()-rs.float()).square().mean((1,2)).sqrt()
                        values.update(state_distance_rms=distance,
                            state_relative_distance=distance/rs.float().square().mean((1,2)).sqrt().clamp_min(1e-6))
                    if short is not None:
                        values['short_distance_rms']=(short.float()-rh.float()).square().mean((1,2)).sqrt()
                    values={k:v.cpu().tolist() for k,v in values.items()}
                    recovery.extend(dict(id=case,patient=case,condition=name,clip=index,
                        clips_since_prefix=index-maximum//model.local_frames+1,
                        **{k:v[i] for k,v in values.items()}) for i,case in enumerate(ids))
            with torch.autocast(device.type, enabled=device.type=='cuda'):
                inp = intervention_video(video, prefix, args.recent_frames, kind, model.local_frames)
                features, trace = stream_audit(model, inp, args.recent_frames, observe=split=='val',
                                               trajectory_callback=trajectory if split=='val' else None)
            # These traces are batch averages, not independent patient measurements.
            traces.extend(dict(condition=name, patients=ids, **row) for row in trace)
            names = ['cache']
            if name == f'history_{maximum}':
                names += ['local','local_history','local_empty']
                if model.memory_mode != 'none':
                    names += ['compressed','updated']
            for feature in names:
                value = features[feature]
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite stage3 features')
                collected.setdefault(name+'_'+feature, []).append(value.float().cpu().numpy().astype(np.float16))
        progress.set_postfix(eligible=len(rows),excluded=len(excluded),batch=args.batch_size,
                            memory=f'{torch.cuda.memory_allocated(device)/2**30:.1f}G' if device.type=='cuda' else 'cpu')
        del reference
        if args.eligible_budget and len(rows)>=budget:
            break
    if args.eligible_budget and len(rows)<budget:
        raise ValueError(f'Eligible {split} patients {len(rows)} < requested {budget}; lower explicit budget or inspect data')
    if len(rows)<2:
        raise ValueError('Fewer than two complete-history patients; do not silently use repeated padding')
    arrays = {k:np.concatenate(v) for k,v in collected.items()}
    metadata = dict(rows=rows, excluded=excluded, traces=traces, recovery=recovery)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f, identity=np.asarray(identity), **{k:np.asarray(json.dumps(v)) for k,v in metadata.items()}, **arrays)
    temp.replace(cache)
    return arrays, metadata


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    model, cfg, meta = load_model(args.checkpoint, device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    training_config = payload.get('config', {})
    partial = payload.get('partial_epoch', False)
    del payload
    if args.expected_epoch is not None and (meta['epoch']!=args.expected_epoch or partial):
        raise ValueError('Checkpoint is not the requested complete training epoch')
    if args.expected_memory is not None and getattr(model,'memory_mode',None)!=args.expected_memory:
        raise ValueError('Checkpoint memory type does not match the queue label')
    if not isinstance(model, TemporalMAE) or model.frequency_conditioned or model.frequency_loss_weight:
        raise ValueError('Pure TemporalMAE checkpoints required')
    if model.frames != args.recent_frames or model.img_size != 112 or any(p%model.local_frames for p in args.prefixes):
        raise ValueError('Use native recent window, 112px, and prefix multiples of local_frames')
    model.requires_grad_(False).eval()
    model.gradient_checkpointing = False
    args.history_slots = 1 if model.memory_mode=='global' else model.memory_grid**2
    args.input_protocol = meta['input_protocol']
    args.seg_frames, args.seg_target_index, args.offset_probe = model.frames, model.frames-2, True
    out, cache = Path(args.output_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {split:TemporalEchoDataset(args.data_root, split, max(args.prefixes)+args.recent_frames,
        channels=model.in_chans, input_protocol=args.input_protocol,
        limit=None if args.eligible_budget else getattr(args, 'ef_'+split+'_cases'),
        seed=args.seed, random_start=False) for split in ('train','val')}
    if set(datasets['train'].ids) & set(datasets['val'].ids):
        raise ValueError('Train/val patient overlap')
    segdata = {s:SegViews(args.data_root,s,args,model.in_chans,2) for s in ('train','val')} if args.with_seg else {}
    if segdata and set(segdata['train'].patient_ids) & set(segdata['val'].patient_ids):
        raise ValueError('Segmentation patient overlap')
    scientific = {k:v for k,v in vars(args).items() if k not in {
        'batch_size','max_batch_size','num_workers','auto_workers','cpu_threads','prefetch_factor','keep_cache',
        'output_dir','cache_dir'}}
    scientific.update(version=VERSION, source_config=training_config, checkpoint_metadata=meta, checkpoint_sha256=file_hash(args.checkpoint),
        code_sha256=hashlib.sha256((code_hash()+file_hash(__file__)+file_hash(ROOT/'utils/stage3_diagnostics.py')).encode()).hexdigest(),
        filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
        traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'),
        data={k:data_signature(v) for k,v in datasets.items()},
        segmentation_data={k:data_signature(v) for k,v in segdata.items()})
    protocol = out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != scientific:
        raise ValueError('Protocol/data/code changed; use a new run_tag')
    save_json(protocol, scientific)
    if (out/'DONE').exists():
        if not (out/'metrics.json').is_file():
            raise ValueError('DONE without metrics')
        return
    start = time.perf_counter()
    save_json(out/'status.json', dict(stage='autotune', updated_unix=time.time()))
    save_json(out/'runtime.json', tune_runtime(model, datasets['train'], args, device,
                                             segdata['train'][0] if segdata else None))
    identity = json.dumps(scientific, sort_keys=True)
    extracted, metadata = {}, {}
    for split in ('train','val'):
        save_json(out/'status.json', dict(stage='extract_'+split, updated_unix=time.time()))
        extracted[split], metadata[split] = extract_ef(model,datasets[split],args,device,split,cache/(split+'.npz'),identity)
        write_csv(out/('validity_'+split+'.csv'), metadata[split]['rows'])
        write_csv(out/('excluded_'+split+'.csv'), metadata[split]['excluded'])
    save_json(out/'state_traces.json', metadata['val']['traces'])
    write_csv(out/'recovery_patient.csv',metadata['val']['recovery'])
    train, val = extracted['train'], extracted['val']
    maximum = max(args.prefixes)
    keys = [f'history_{p}_cache' for p in args.prefixes]
    if model.memory_mode != 'none':
        keys += [f'history_{maximum}_local_empty', f'history_{maximum}_local_history']
    metrics, predictions = {}, {}
    for key in keys:
        save_json(out/'status.json', dict(stage='probe_'+key, updated_unix=time.time()))
        # One fixed clean-trained head for all its perturbation evaluations.
        predict, losses, count = fit_head(train, key, args, device)
        write_csv(out/(key+'_loss.csv'), losses)
        plot_loss(losses, out/(key+'_loss.png'))
        eval_keys = [key]
        if key == f'history_{maximum}_cache':
            eval_keys += ['repeat_prefix_cache','zero_prefix_clip_cache','degraded_history_cache']
        if key == 'history_0_cache':
            eval_keys += ['degraded_no_history_cache']
        for name in eval_keys:
            pred = predict(val[name])
            target = torch.from_numpy(val['y'])
            value = regression_metrics(pred, target)
            value = {k:float(v) if np.isfinite(v) else None for k,v in value.items()}
            value.update(head_parameters=count, trained_on=key)
            rows = [dict(r, target=float(target[i]), prediction=float(pred[i]),
                         error=float(abs(pred[i]-target[i]))) for i,r in enumerate(metadata['val']['rows'])]
            metrics[name], predictions[name] = value, rows
            write_csv(out/(name+'.csv'), rows)
        if key == f'history_{maximum}_cache':
            # Same fitted head, different history inputs: separate from the clean refit comparison.
            for prefix in args.prefixes[:-1]:
                name=f'fixed_head_history_{prefix}'
                pred=predict(val[f'history_{prefix}_cache'])
                target=torch.from_numpy(val['y'])
                value=regression_metrics(pred,target)
                value={k:float(v) if np.isfinite(v) else None for k,v in value.items()}
                value.update(head_parameters=count,trained_on=key)
                rows=[dict(r,target=float(target[i]),prediction=float(pred[i]),error=float(abs(pred[i]-target[i])))
                      for i,r in enumerate(metadata['val']['rows'])]
                metrics[name],predictions[name]=value,rows
                write_csv(out/(name+'.csv'),rows)
        del predict
    contrasts = [(f'history_{p}_cache','history_0_cache') for p in args.prefixes if p]
    contrasts += [(k,f'history_{maximum}_cache') for k in ('repeat_prefix_cache','zero_prefix_clip_cache')]
    contrasts += [('degraded_history_cache','degraded_no_history_cache')]
    contrasts += [(f'history_{maximum}_cache',f'fixed_head_history_{p}') for p in args.prefixes[:-1]]
    if model.memory_mode != 'none':
        contrasts += [(f'history_{maximum}_local_history',f'history_{maximum}_local_empty')]
    paired = [dict(candidate=a,control=b,**paired_bootstrap(predictions[a],predictions[b],'error',args.seed)) for a,b in contrasts]
    write_csv(out/'paired.csv', paired)
    # Match every bottleneck probe to D inputs via mean pooling. Auxiliary only:
    # averaging removes ordering and cannot establish sufficiency of a state.
    bottlenecks = {}
    for feature in ('local','cache','compressed','updated'):
        key = f'history_{maximum}_'+feature
        if key not in train:
            continue
        prediction = ridge_apply(ridge_fit(torch.from_numpy(train[key]).float().mean(1),
            torch.from_numpy(train['y']), 10.), torch.from_numpy(val[key]).float().mean(1))
        values = regression_metrics(prediction, torch.from_numpy(val['y']))
        bottlenecks[feature] = {k:float(v) if np.isfinite(v) else None for k,v in values.items()}
    save_json(out/'bottleneck_ridge_auxiliary.json',bottlenecks)
    # A clean native training-window input, independent of the long inference intervention.
    sample = datasets['train'][datasets['train'].ids.index(metadata['train']['rows'][0]['id'])]
    if sample['frame_valid'][-model.frames:].all():
        try:
            audit = gradient_audit(model, sample['video'][None].to(device), args.seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            audit = dict(status='skipped_oom', note='FP32 gradient diagnostic did not fit; other results remain valid')
    else:
        audit = dict(status='skipped_incomplete_sample')
    save_json(out/'gradient_audit.json', audit)
    seg = None
    if args.with_seg:
        save_json(out/'status.json', dict(stage='segmentation', updated_unix=time.time()))
        segargs = copy.copy(args)
        segargs.prefix_frames = 0
        tr, va = [extract_seg(model,segdata[s],segargs,device,'seg','real',cache/('seg_'+s+'.npz'),scientific) for s in ('train','val')]
        seg, rows, losses = fit_seg(tr,va,'fused',args,device,model.tubelet_size,out/'seg')
        write_csv(out/'seg.csv',rows)
        write_csv(out/'seg_loss.csv',losses)
        plot_loss(losses,out/'seg_loss.png')
    result = dict(ef=metrics,seg=seg,paired=paired,wall_seconds=time.perf_counter()-start,
                  memory_mode=model.memory_mode,local_frames=model.local_frames,
                  train_patients=len(train['y']),val_patients=len(val['y']),
                  total_parameters=sum(p.numel() for p in model.parameters()))
    save_json(out/'metrics.json',result)
    lines = ['# Stage 3 frozen memory diagnostics','',
        'No new MAE training. Validation screening, not full fine-tuning or a test result.',
        'All EF conditions use identical complete maximum-prefix cases and recent source frames.',
        'Each clean history length has a separately fitted identical head; interventions reuse its clean head.',
        'local_empty/history share unmodified local features and matching explicit state slots.',
        'State traces are batch averages. Gradient sensitivity is not a representation-quality score.',
        'Artificial repeats/zeros/occlusion are distribution interventions, not physiological labels.',
        'A no-memory checkpoint is a trained control; resetting a memory model does not replace it.',
        'Global/spatial/dual change capacity and training; scores alone do not isolate compression mechanisms.',
        'Prefixes beyond the native training window are length extrapolation. Clip outputs wait for clip end.',
        'Segmentation uses native-window target positions T-2/T-1, not long-prefix EF cases.',
        'Bootstrap covers patients only, not seed variance. No automatic ranking or candidate training.','',
        '| EF condition | MAE |','|---|---:|']
    lines += [f'| {k} | {v["mae"]:.5f} |' for k,v in metrics.items()]
    if seg:
        lines += ['',f'Segmentation patient Dice: {seg["dice_patient_mean"]:.5f}']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'status.json',dict(stage='completed',updated_unix=time.time()))
    (out/'DONE').write_text(VERSION+'\n')
    archive_analysis(out)
    if not args.keep_cache:
        for name in ('train.npz','val.npz','seg_train.npz','seg_val.npz'):
            (cache/name).unlink(missing_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','output_dir','cache_dir'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--expected_epoch',type=int)
    p.add_argument('--expected_memory',choices=['none','global','spatial','dual'])
    p.add_argument('--prefixes',nargs='+',type=int,default=[0,64,128])
    p.add_argument('--recent_frames',type=int,default=64)
    for key,default in dict(batch_size=0,max_batch_size=32,num_workers=8,prefetch_factor=2,cpu_threads=4,
        ef_train_cases=256,ef_val_cases=128,seg_train_cases=32,seg_val_cases=32,ef_steps=200,seg_steps=100,
        ef_hidden=64,probe_batch_size=32,seed=42).items():
        p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--ef_lr',type=float,default=.001)
    p.add_argument('--seg_lr',type=float,default=.01)
    p.add_argument('--auto_workers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--with_seg',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--keep_cache',action='store_true')
    p.add_argument('--eligible_budget',action='store_true',help='Scan the seeded split until the requested count of complete-history patients is met')
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    args.prefixes = sorted(set(args.prefixes))
    if not args.prefixes or args.prefixes[0]!=0 or max(args.prefixes)<=0:
        p.error('prefixes must include zero and a positive prefix')
    for key in ('recent_frames','max_batch_size','prefetch_factor','cpu_threads','ef_train_cases','ef_val_cases',
                'seg_train_cases','seg_val_cases','ef_steps','seg_steps','ef_hidden','probe_batch_size','ef_lr','seg_lr'):
        if getattr(args,key)<=0:
            p.error(key+' must be positive')
    if args.batch_size<0 or args.num_workers<0:
        p.error('batch_size/workers must be nonnegative')
    if args.smoke:
        args.ef_train_cases,args.ef_val_cases = 16,8
        args.seg_train_cases=args.seg_val_cases=2
        args.ef_steps=args.seg_steps=2
        args.batch_size,args.num_workers,args.auto_workers=1,0,False
    return args


if __name__=='__main__':
    evaluate(parse_args())
