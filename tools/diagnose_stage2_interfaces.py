"""Zero-new-pretraining stage-2 diagnostics on existing TemporalMAE checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.temporal_mae import TemporalMAE
from tools.evaluate_temporal_mae import ProbeSegDataset, ridge_fit, ridge_apply, write_csv
from tools.evaluate_representation_quality import load_model, regression_metrics
from tools.evaluate_temporal_screen import file_hash
from utils.temporal_data import TemporalEchoDataset
from utils.stage2_diagnostics import extract_batch, tune_batch, paired_bootstrap, OffsetReadout
from utils.seed import seed_everything
from echo_aug_validation.io_utils import find_echonet_video

VERSION = 'stage2_interface_v1'


def save_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class SegViews(Dataset):
    def __init__(self, root, split, args, channels, tubelet):
        self.views = []
        for offset in range(tubelet if args.offset_probe else 1):
            cfg = copy.copy(args)
            cfg.audit_frames = args.seg_frames
            cfg.seg_target_index = args.seg_target_index+offset
            self.views.append(ProbeSegDataset(root, split, cfg, channels))
        self.samples = self.views[0].samples
        self.patient_ids = sorted({s['stem'] for s in self.samples})

    def __len__(self):
        return len(self.samples)*len(self.views)

    def __getitem__(self, index):
        view, sample = divmod(index, len(self.samples))
        row = self.views[view][sample]
        source = self.samples[sample]['frame']
        row['patient'] = self.samples[sample]['stem']
        row['source_frame'] = source
        row['id'] += f':view{view}'
        return row


def make_loader(dataset, args):
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                   pin_memory=torch.cuda.is_available())
    if args.num_workers:
        options.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(dataset, **options)


def validity_rows(batch, model, task, result):
    rows = []
    for i, case in enumerate(batch['id']):
        valid = batch['frame_valid'][i].bool()
        row = dict(id=case, patient=batch['patient'][i] if task=='seg' else case,
                   valid_frames=int(valid.sum()), padded_frames=int((~valid).sum()),
                   complete_tubelets=int(result['complete'][i]), partial_tubelets=int(result['partial'][i]))
        if task == 'seg':
            index = int(batch['target_index'][i])
            tube_start = index//model.tubelet_size*model.tubelet_size
            clip_start = index//model.local_frames*model.local_frames
            window_start = index//model.frames*model.frames
            row.update(source_frame=int(batch['source_frame'][i]), target_index=index,
                       offset=index%model.tubelet_size, target_valid=bool(valid[index]),
                       target_tubelet_complete=bool(valid[tube_start:tube_start+model.tubelet_size].all()),
                       real_history_frames=int(valid[window_start:clip_start].sum()),
                       effective_history_tubelets=int(valid[window_start:clip_start].reshape(-1,model.tubelet_size).all(-1).sum()),
                       real_future_frames=int(valid[index+1:clip_start+model.local_frames].sum()),
                       full_context=bool(valid.all()))
            if not row['target_valid']:
                raise ValueError(f'Invalid labeled target: {case}')
        rows.append(row)
    return rows


def data_signature(dataset):
    ids = dataset.patient_ids if isinstance(dataset, SegViews) else dataset.ids
    root = dataset.views[0].root if isinstance(dataset, SegViews) else dataset.root
    files = []
    for case in ids:
        path = find_echonet_video(root, case)
        if path is None:
            raise FileNotFoundError(case)
        stat = path.stat()
        files.append([case,str(path.resolve()),stat.st_size,stat.st_mtime_ns])
    return files


def extract(model, dataset, args, device, task, cache, identity):
    signature = json.dumps(dict(identity=identity, files=data_signature(dataset), task=task), sort_keys=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data['signature'].item()) != signature:
                raise RuntimeError(f'Feature cache identity changed: {cache}; use a new run_tag')
            result = {k:data[k].copy() for k in data.files if k not in {'signature','rows','stats'}}
            result['rows'] = json.loads(str(data['rows'].item()))
            result['stats'] = json.loads(str(data['stats'].item()))
            return result
    columns, rows = {}, []
    stats = dict(fusion_square_sum=0., fusion_elements=0, fusion_max_abs=0.)
    start = time.perf_counter()
    progress = tqdm(make_loader(dataset,args), desc=f'{task} extract {len(dataset)}')
    for batch in progress:
        try:
            result = extract_batch(model,batch['video'],batch['frame_valid'],device,
                                   batch.get('target_index'),args.amp)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError('Extraction OOM. Rerun same command with --batch_size 2; completed split caches are reused.') from exc
        for key in stats:
            if key=='fusion_max_abs':
                stats[key] = max(stats[key],result[key])
            else:
                stats[key] += result[key]
        features = result['maps'] if task=='seg' else result['pooled']
        for name,value in features.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite {name} features')
            # Cache only target maps (not full clips), in half precision on CPU.
            columns.setdefault(name,[]).append(value.numpy().astype(np.float16 if task=='seg' else np.float32))
        columns.setdefault('y',[]).append(batch['mask' if task=='seg' else 'target'].numpy())
        rows.extend(validity_rows(batch,model,task,result))
        progress.set_postfix(batch=args.batch_size, memory=f'{torch.cuda.memory_allocated()/2**30:.1f}G' if device.type=='cuda' else 'cpu')
    output = {k:np.concatenate(v) for k,v in columns.items()}
    stats['extraction_seconds'] = time.perf_counter()-start
    output.update(rows=rows,stats=stats)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary = cache.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, signature=np.asarray(signature), rows=np.asarray(json.dumps(rows)),
                 stats=np.asarray(json.dumps(stats)), **{k:v for k,v in output.items() if k not in {'rows','stats'}})
    temporary.replace(cache)
    return output


def ef_probes(train, val, args):
    result, predictions = {}, {}
    for name in ('local','fused','legacy_fused'):
        pred = ridge_apply(ridge_fit(torch.from_numpy(train[name]),torch.from_numpy(train['y']),args.ridge_alpha),
                           torch.from_numpy(val[name]))
        y = torch.from_numpy(val['y'])
        result[name] = regression_metrics(pred,y)
        predictions[name] = [dict(id=r['id'],patient=r['patient'],target=float(y[i]),prediction=float(pred[i]),
                                  error=float((pred[i]-y[i]).abs())) for i,r in enumerate(val['rows'])]
    return result,predictions


def seg_probe(train, val, name, args, device, tubelet):
    feature = 'fused' if name in {'shared_bank','offset_bank'} else name
    bank = name in {'shared_bank','offset_bank'}
    torch.manual_seed(args.seed+2001)
    x, xv = torch.from_numpy(train[feature]).float(), torch.from_numpy(val[feature]).float()
    mean = x.mean((0,2,3),keepdim=True)
    std = x.std((0,2,3),keepdim=True,unbiased=False).clamp_min(1e-5)
    x, xv = (x-mean)/std, (xv-mean)/std
    y = torch.from_numpy(train['y']).long()
    offsets = torch.tensor([r['offset'] for r in train['rows']])
    head = OffsetReadout(x.shape[1],tubelet if bank else 1,name=='offset_bank').to(device)
    opt = torch.optim.AdamW(head.parameters(),lr=args.seg_lr,weight_decay=.001)
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    progress = tqdm(range(args.seg_steps),desc=f'seg head {name}')
    for step in progress:
        idx = torch.randint(len(x),(min(args.probe_batch_size,len(x)),),generator=generator)
        logits = F.interpolate(head(x[idx].to(device),offsets[idx].to(device)), y.shape[-2:],mode='bilinear',align_corners=False)
        loss = F.cross_entropy(logits,y[idx].to(device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite segmentation probe loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(dict(step=step+1,loss=float(loss.detach())))
        if step%20==0 or step+1==args.seg_steps:
            progress.set_postfix(loss=f'{losses[-1]["loss"]:.4f}')
    rows = []
    with torch.inference_mode():
        for start in range(0,len(xv),args.probe_batch_size):
            stop = start+args.probe_batch_size
            offsets = torch.tensor([r['offset'] for r in val['rows'][start:stop]],device=device)
            logits = head(xv[start:stop].to(device),offsets)
            pred = F.interpolate(logits,val['y'].shape[-2:],mode='bilinear',align_corners=False).argmax(1).cpu()
            target = torch.from_numpy(val['y'][start:stop])
            inter = ((pred==1)&(target==1)).sum((1,2))
            denom = (pred==1).sum((1,2))+(target==1).sum((1,2))
            for j in range(len(pred)):
                rows.append(dict(val['rows'][start+j],dice=float((2*inter[j]+1e-6)/(denom[j]+1e-6)),
                                 intersection=int(inter[j]),denominator=int(denom[j])))
    patients = {}
    for r in rows:
        patients.setdefault(r['patient'],[]).append(r['dice'])
    clean = [r['dice'] for r in rows if r['full_context']]
    return dict(dice_mean=float(np.mean([r['dice'] for r in rows])),
                dice_patient_mean=float(np.mean([np.mean(v) for v in patients.values()])),
                dice_global=2*sum(r['intersection'] for r in rows)/max(1,sum(r['denominator'] for r in rows)),
                full_context_dice=float(np.mean(clean)) if clean else None,
                full_context_frames=len(clean),parameters=sum(p.numel() for p in head.parameters()),
                steps=args.seg_steps),rows,losses


def summarize(root, methods, seed):
    comparison, pairs = [], []
    completed = {}
    for name in methods:
        path = root/name
        if not (path/'DONE').exists():
            continue
        m = json.loads((path/'metrics.json').read_text(encoding='utf-8'))
        completed[name] = m
        for task in ('ef','seg'):
            for readout,values in m[task].items():
                comparison.append(dict(method=name,task=task,readout=readout,
                                       value=values['mae' if task=='ef' else 'dice_patient_mean'],
                                       metric='MAE_pp' if task=='ef' else 'patient_Dice_0_1'))
        for row in m['paired']:
            pairs.append(dict(method=name,**row))
    cross_pairs = []
    if len(completed) > 1:
        control = next(iter(completed))
        for candidate in list(completed)[1:]:
            pa = json.loads((root/candidate/'protocol.json').read_text(encoding='utf-8'))
            pb = json.loads((root/control/'protocol.json').read_text(encoding='utf-8'))
            pa.pop('checkpoint_sha256')
            pb.pop('checkpoint_sha256')
            ma = json.loads((root/candidate/'data_manifest.json').read_text(encoding='utf-8'))
            mb = json.loads((root/control/'data_manifest.json').read_text(encoding='utf-8'))
            if pa != pb or ma != mb:
                raise RuntimeError('Cross-checkpoint comparison requires identical data and probe protocols')
            for task,field in [('ef','error'),('seg','dice')]:
                rows = []
                for method in (candidate,control):
                    with (root/method/f'{task}_fused.csv').open(encoding='utf-8') as handle:
                        rows.append(list(csv.DictReader(handle)))
                cross_pairs.append(dict(candidate=candidate,control=control,task=task,
                                        **paired_bootstrap(*rows,field,seed)))
    write_csv(root/'cross_checkpoint_differences.csv',cross_pairs)
    write_csv(root/'comparison.csv',comparison)
    write_csv(root/'paired_differences.csv',pairs)
    lines = ['# Stage 2 frozen interface diagnosis','',
             'No new MAE pretraining. No full fine-tuning. Validation probes, not test or clinical performance.',
             'EF: MAE in percentage points, lower is better. Seg: patient-averaged Dice, higher is better.',
             'All readouts are refit on training cases; fixed endpoint, no validation checkpoint selection.','',
             '| Checkpoint | Task | Readout | Value |','|---|---|---|---:|']
    lines += [f"| {r['method']} | {r['task']} | {r['readout']} | {r['value']:.5f} |" for r in comparison]
    lines += ['','## Paired differences','',
              'Candidate minus control. Negative favors candidate for EF, positive for Dice. Unadjusted patient bootstrap; not training-seed uncertainty.',
              '| Checkpoint | Contrast | Delta | 95% low | 95% high | Patients |','|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['contrast']} | {r['delta']:.5f} | {r['low']:.5f} | {r['high']:.5f} | {r['patients']} |" for r in pairs]
    lines += ['','## Cross-checkpoint (descriptive, not causal)','',
              '| Candidate | Control | Task | Delta | 95% low | 95% high |',
              '|---|---|---|---:|---:|---:|']
    lines += [f"| {r['candidate']} | {r['control']} | {r['task']} | {r['delta']:.5f} | {r['low']:.5f} | {r['high']:.5f} |" for r in cross_pairs]
    lines += ['','## Interpretation boundaries','',
              '- local vs fused changes a whole trained fusion block, including self-attention/MLP. It is not an isolated causal memory ablation.',
              '- Target vs local-clip mean tests temporal-location readability, not whether the encoder must process single frames.',
              '- Offset vs shared bank uses equal parameter counts and identical examples. Extra routing capacity remains an alternative explanation, not proof of recovered motion.',
              '- Offset views shift the sampling window by one frame. They change future context and history alignment. There are no invented adjacent-frame ground truths.',
              '- EF uses complete valid tubelets in the denominator; legacy_fused refits the old denominator as a sensitivity control. Padding still enters encoder attention and memory.',
              '- Partial target tubelets remain zero as in the checkpoint implementation. They are reported and included, never silently filtered. Full-context Dice is secondary.',
              '- Native windows reset state. This audit is not a continuous online inference demonstration.',
              '- Checkpoints may have different training histories. Cross-checkpoint ranking is descriptive, not a matched memory treatment effect.',
              '- No reconstruction, state-only EF, temporal-order classifier or repeated full fine-tuning is run. A decoder bypass mechanism cannot be concluded from this audit.','',
              '## Next decision','',
              'If fused loses spatial readability but helps EF, investigate fusion protection before enlarging state. If target beats mean, preserve location-specific outputs. If offset routing improves both alignment views, consider a frame-aware readout before new pretraining. If differences are uncertain, do not declare equivalence or add an entire ablation matrix.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(root/'analysis.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in {'.csv','.json','.md','.png','.log'}:
                archive.write(path,path.relative_to(root))


def run_one(name, checkpoint, args, root):
    out = root/name
    out.mkdir(parents=True,exist_ok=True)
    cache_root = root.parent.parent/'cache'/args.run_tag/name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, meta = load_model(str(checkpoint),device)
    if not isinstance(model,TemporalMAE) or model.frequency_conditioned or model.frequency_loss_weight:
        raise ValueError('This pure-temporal diagnostic requires an existing TemporalMAE (no frequency branch)')
    if model.img_size != 112 or args.ef_frames%model.frames or args.seg_frames != model.frames:
        raise ValueError('Require native 112px model, EF frames a native-window multiple, seg_frames equal native window')
    if args.seg_target_index%model.tubelet_size or not model.local_frames <= args.seg_target_index < args.seg_frames-model.tubelet_size+1:
        raise ValueError('Target must start a tubelet after the first local clip, with room for offset views')
    args.input_protocol = meta['input_protocol']
    model.requires_grad_(False).eval()
    meta['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.gradient_checkpointing = False
    scientific = {k:v for k,v in vars(args).items() if k not in {
        'batch_size','max_batch_size','num_workers','prefetch_factor','cpu_threads','keep_cache','output_root','source_run','checkpoints','methods'}}
    sources = sorted({*ROOT.joinpath('models').glob('*.py'), *ROOT.joinpath('utils').glob('*.py'),
                      ROOT/'echo_aug_validation/io_utils.py',ROOT/'tools/evaluate_representation_quality.py',
                      ROOT/'tools/evaluate_temporal_mae.py',Path(__file__).resolve()})
    scientific.update(version=VERSION,checkpoint_sha256=file_hash(checkpoint),
                      filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
                      traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'),
                      code_sha256=hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_bytes()
                                                       for p in sources)).hexdigest())
    protocol = out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != scientific:
        raise RuntimeError('Protocol changed; choose a new run_tag, do not mix results')
    save_json(protocol,scientific)
    data = {}
    for split in ('train','val'):
        data[f'ef_{split}'] = TemporalEchoDataset(args.data_root,split,args.ef_frames,channels=model.in_chans,
            limit=getattr(args,f'ef_{split}_cases'),seed=args.seed,random_start=False,input_protocol=args.input_protocol)
        data[f'seg_{split}'] = SegViews(args.data_root,split,args,model.in_chans,model.tubelet_size)
    for task in ('ef','seg'):
        get_ids = lambda d: d.ids if task=='ef' else d.patient_ids
        if set(get_ids(data[task+'_train'])) & set(get_ids(data[task+'_val'])):
            raise ValueError('Patient overlap between train and validation')
    manifest = {k:data_signature(v) for k,v in data.items()}
    manifest_path = out/'data_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Selected data changed; choose a new run_tag')
    save_json(manifest_path,manifest)
    if (out/'DONE').exists():
        if not (out/'metrics.json').exists():
            raise RuntimeError('DONE exists without metrics')
        print(f'Skip completed {name}',flush=True)
        return
    begin = time.perf_counter()
    trials = []
    if args.batch_size == 0:
        args.batch_size,trials = tune_batch(model,data['ef_train'][0],device,args.max_batch_size,amp=args.amp)
    save_json(out/'runtime.json',dict(batch_size=args.batch_size,num_workers=args.num_workers,
                                     device=str(device),trials=trials,amp=args.amp))
    save_json(out/'checkpoint_config.json',dict(model=cfg,metadata=meta))
    print(f'{name}: frozen backbone; extraction batch={args.batch_size}, workers={args.num_workers}',flush=True)
    def stage(label):
        save_json(out/'status.json',dict(stage=label,updated_unix=time.time(),checkpoint=str(checkpoint)))
    metrics = dict(metadata=meta,ef={},seg={},paired=[])
    # Keep the EF and segmentation caches separate so interruption only repeats an unfinished split.
    stage('extract_ef_train')
    eftrain = extract(model,data['ef_train'],args,device,'ef',cache_root/'ef_train.npz',scientific)
    stage('extract_ef_val')
    efval = extract(model,data['ef_val'],args,device,'ef',cache_root/'ef_val.npz',scientific)
    if model.memory_mode == 'none' and any(x['stats']['fusion_max_abs'] != 0 for x in (eftrain,efval)):
        raise RuntimeError('No-memory control local/fused exits must be exactly equal')
    metrics['ef'], predictions = ef_probes(eftrain,efval,args)
    for mode,rows in predictions.items():
        write_csv(out/f'ef_{mode}.csv',rows)
    for a,b in [('fused','local'),('fused','legacy_fused')]:
        metrics['paired'].append(dict(contrast=f'ef:{a}-{b}',**paired_bootstrap(predictions[a],predictions[b],'error',args.seed)))
    metrics['extraction'] = {f'ef_{split}':item['stats'] for split,item in [('train',eftrain),('val',efval)]}
    for split,item in [('train',eftrain),('val',efval)]:
        write_csv(out/f'validity_ef_{split}.csv',item['rows'])
    del eftrain,efval
    stage('extract_seg_train')
    train = extract(model,data['seg_train'],args,device,'seg',cache_root/'seg_train.npz',scientific)
    stage('extract_seg_val')
    val = extract(model,data['seg_val'],args,device,'seg',cache_root/'seg_val.npz',scientific)
    for split,item in [('train',train),('val',val)]:
        write_csv(out/f'validity_seg_{split}.csv',item['rows'])
        metrics['extraction'][f'seg_{split}'] = item['stats']
    # Backbone no longer occupies GPU memory while fitting tiny readouts.
    tubelet, mode = model.tubelet_size,model.memory_mode
    del model
    if device.type=='cuda':
        torch.cuda.empty_cache()
    predictions = {}
    modes = ['local','fused','local_mean','fused_mean']
    if args.offset_probe:
        modes += ['shared_bank','offset_bank']
    for name_readout in modes:
        stage('fit_seg_'+name_readout)
        head_result = out/f'head_{name_readout}.json'
        if head_result.exists():
            saved = json.loads(head_result.read_text(encoding='utf-8'))
            values,rows = saved['metrics'],saved['rows']
        else:
            values,rows,losses = seg_probe(train,val,name_readout,args,device,tubelet)
            write_csv(out/f'loss_{name_readout}.csv',losses)
            save_json(head_result,dict(metrics=values,rows=rows))
        write_csv(out/f'seg_{name_readout}.csv',rows)
        metrics['seg'][name_readout],predictions[name_readout] = values,rows
    contrasts = [('fused','local'),('local','local_mean'),('fused','fused_mean')]
    if args.offset_probe:
        contrasts += [('offset_bank','shared_bank')]
    for a,b in contrasts:
        metrics['paired'].append(dict(contrast=f'seg:{a}-{b}',**paired_bootstrap(predictions[a],predictions[b],'dice',args.seed)))
        if a=='offset_bank':
            for offset in range(tubelet):
                ar = [r for r in predictions[a] if r['offset']==offset]
                br = [r for r in predictions[b] if r['offset']==offset]
                metrics['paired'].append(dict(contrast=f'seg:{a}-{b}:offset{offset}',**paired_bootstrap(ar,br,'dice',args.seed)))
    metrics['boundaries'] = dict(memory_mode=mode,train_seg_frames=len(train['rows']),val_seg_frames=len(val['rows']),
        partial_target_rows=sum(not r['target_tubelet_complete'] for r in val['rows']),
        no_real_history_rows=sum(r['real_history_frames']==0 for r in val['rows']),
        full_context_rows=sum(r['full_context'] for r in val['rows']),
        trainable_backbone_parameters=0,mae_optimizer_steps=0,weight_files_written=0)
    metrics['wall_seconds'] = time.perf_counter()-begin
    save_json(out/'metrics.json',metrics)
    stage('completed')
    (out/'DONE').write_text('Frozen interface diagnosis completed. No new MAE training.\n')
    if not args.keep_cache and cache_root.exists():
        # Delete only the four caches owned by this evaluator after successful completion.
        for key in data:
            (cache_root/f'{key}.npz').unlink(missing_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_run',default='/root/autodl-tmp/outputs_temporal/temporal_gray_20260909')
    p.add_argument('--methods',nargs='+',default=['clip_mae_pool64','hier_spatial'])
    p.add_argument('--checkpoints',nargs='+',help='Explicit NAME=PATH pairs; overrides source_run/methods')
    p.add_argument('--epoch',type=int,default=400)
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--output_root',default='/root/autodl-tmp/outputs_stage2_diagnostics')
    p.add_argument('--run_tag',default='stage2_interface_v1')
    p.add_argument('--batch_size',type=int,default=0,help='0: bounded forward-only GPU throughput search')
    p.add_argument('--max_batch_size',type=int,default=16)
    p.add_argument('--num_workers',type=int,default=min(8,os.cpu_count() or 1))
    p.add_argument('--prefetch_factor',type=int,default=2)
    p.add_argument('--cpu_threads',type=int,default=4)
    p.add_argument('--ef_frames',type=int,default=256)
    p.add_argument('--seg_frames',type=int,default=64)
    p.add_argument('--seg_target_index',type=int,default=48)
    p.add_argument('--ef_train_cases',type=int,default=512)
    p.add_argument('--ef_val_cases',type=int,default=256)
    p.add_argument('--seg_train_cases',type=int,default=64)
    p.add_argument('--seg_val_cases',type=int,default=64)
    p.add_argument('--seg_steps',type=int,default=200)
    p.add_argument('--probe_batch_size',type=int,default=32)
    p.add_argument('--seg_lr',type=float,default=.01)
    p.add_argument('--ridge_alpha',type=float,default=10.)
    p.add_argument('--offset_probe',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--keep_cache',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    args = p.parse_args()
    if args.batch_size < 0 or args.num_workers < 0 or args.seg_target_index < 0:
        p.error('Invalid batch/workers/target index')
    for key in ('max_batch_size','prefetch_factor','cpu_threads','ef_frames','seg_frames','ef_train_cases',
                'ef_val_cases','seg_train_cases','seg_val_cases','seg_steps','probe_batch_size','seg_lr','ridge_alpha'):
        if getattr(args,key) <= 0:
            p.error(f'{key} must be positive')
    if Path(args.run_tag).name != args.run_tag or args.run_tag in {'.','..'} or '\\' in args.run_tag:
        p.error('run_tag must be a directory name')
    if args.smoke:
        args.ef_train_cases,args.ef_val_cases = 8,4
        args.seg_train_cases,args.seg_val_cases,args.seg_steps = 2,2,2
        args.batch_size,args.num_workers = 1,0
        args.run_tag = 'smoke_'+args.run_tag
    return args


def main():
    args = parse_args()
    checkpoints = {name:Path(args.source_run)/'ckpt'/name/f'epoch_{args.epoch:04d}.pt' for name in args.methods}
    if args.checkpoints:
        checkpoints = {}
        for item in args.checkpoints:
            name,path = item.split('=',1)
            if name in checkpoints:
                raise ValueError('Duplicate checkpoint name')
            checkpoints[name] = Path(path)
    for name in checkpoints:
        if not name or Path(name).name!=name or name in {'.','..'} or '\\' in name:
            raise ValueError('Invalid checkpoint label')
    root = Path(args.output_root)/'result'/args.run_tag
    print('No MAE training or weight saving. Existing checkpoints only.',flush=True)
    for name,path in checkpoints.items():
        print(f'{name}: {path}',flush=True)
    print(f'Results: {root}',flush=True)
    if args.dry_run:
        return
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing checkpoints (no fallback to random initialization): '+', '.join(missing))
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    for name,checkpoint in checkpoints.items():
        run_one(name,checkpoint,copy.copy(args),root)
        summarize(root,list(checkpoints),args.seed)
    print(f'Completed. Download {root / "analysis.zip"}',flush=True)


if __name__=='__main__':
    main()
