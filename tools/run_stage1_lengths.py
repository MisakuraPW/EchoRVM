"""Matched local-length screening: fixed spatial memory, 64 frames, endpoint probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from tools.run_temporal_research import make_config, run_command, archive_analysis
from tools.diagnose_stage2_interfaces import (
    SegViews, data_signature, extract, save_json, seg_probe,
    ridge_fit, ridge_apply, regression_metrics, load_model, file_hash, write_csv,
)
from utils.research_storage import check_protocol
from utils.seed import seed_everything
from utils.stage2_diagnostics import tune_batch, paired_bootstrap
from utils.temporal_data import TemporalEchoDataset

VERSION = 'stage1_lengths_v1'


def stage_config(args, length, checkpoint_dir):
    base = SimpleNamespace(seed=args.seed, data_root=args.data_root,
        input_protocol='gray_repeat3', num_workers=args.num_workers, prefetch_factor=4,
        init_checkpoint=args.init_checkpoint, audit_epochs=[0, args.epochs],
        save_last_every=10, min_free_gb=3., epochs=args.epochs,
        batch_size=8, grad_accum_steps=4, baseline_batch_size=32, smoke=args.smoke)
    cfg = make_config('temporal', f'spatial_l{length}',
                      dict(local_frames=length, clip_count=64//length, memory_mode='spatial'), base)
    cfg['experiment']['description'] = VERSION + ': fixed spatial memory; local length only'
    cfg['checkpoint'].update(dir=str(checkpoint_dir), save_initial=True, save_best=False)
    # Validation reconstruction is monitoring only; no best selection or early stop.
    cfg['train']['val_interval'] = 1 if args.smoke else 25
    cfg['train']['plot_interval'] = 1 if args.smoke else 5
    return cfg


def probe_args(args):
    return SimpleNamespace(seed=args.seed, input_protocol='gray_repeat3',
        ef_frames=256, seg_frames=64, seg_target_index=63, offset_probe=False,
        ef_train_cases=8 if args.smoke else 512, ef_val_cases=4 if args.smoke else 256,
        seg_train_cases=2 if args.smoke else 64, seg_val_cases=2 if args.smoke else 64,
        seg_steps=2 if args.smoke else 200, probe_batch_size=32, seg_lr=.01, ridge_alpha=10.,
        batch_size=1 if args.smoke else args.eval_batch_size,
        num_workers=0 if args.smoke else args.num_workers, prefetch_factor=2, amp=True)


def code_hash():
    paths = sorted({*ROOT.joinpath('models').glob('*.py'), *ROOT.joinpath('utils').glob('*.py'),
        ROOT/'echo_aug_validation/io_utils.py', ROOT/'tools/evaluate_temporal_mae.py',
        ROOT/'tools/evaluate_representation_quality.py', ROOT/'tools/diagnose_stage2_interfaces.py',
        ROOT/'tools/run_temporal_research.py', ROOT/'trainers/train_rmae.py', Path(__file__).resolve()})
    return hashlib.sha256(b''.join(p.relative_to(ROOT).as_posix().encode()+b'\0'+p.read_bytes()
                                  for p in paths)).hexdigest()


@torch.inference_mode()
def benchmark(model, sample, device):
    """GPU-resident batch-one native window; excludes disk, H2D, and heads."""
    video = sample['video'][None, :model.frames].to(device)
    valid = sample['frame_valid'][None, :model.frames].to(device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    for _ in range(2):
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            model.forward_features(video, valid)
    sync()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(5):
        begin = time.perf_counter()
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            model.forward_features(video, valid)
        sync()
        timings.append((time.perf_counter()-begin)*1000)
    return dict(native_frames=model.frames, batch_size=1, repetitions=5,
        native_window_ms_median=float(np.median(timings)),
        per_frame_ms_amortized=float(np.median(timings)/model.frames),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        local_buffer_intervals=model.local_frames-1, sampling_stride=1,
        note='Compute only, not end-to-end latency. Buffer seconds=(L-1)/source_fps; FPS not assumed. '
             'State resets each native window; this is not infinite streaming.')


def evaluate(checkpoint, out, args):
    out.mkdir(parents=True, exist_ok=True)
    a = probe_args(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    model, cfg, meta = load_model(str(checkpoint), device)
    if (model.core_type != 'temporal_mae' or model.frames != 64 or model.img_size != 112
            or model.memory_mode != 'spatial' or model.tubelet_size != 2
            or meta['input_protocol'] != 'gray_repeat3' or model.frequency_conditioned
            or model.frequency_loss_weight):
        raise ValueError('Stage 1 requires the declared pure-temporal spatial model contract')
    model.requires_grad_(False).eval()
    model.gradient_checkpointing = False
    scientific = {k:v for k,v in vars(a).items() if k not in {'batch_size','num_workers','prefetch_factor'}}
    scientific.update(version=VERSION, code_sha256=code_hash(), checkpoint_sha256=file_hash(checkpoint),
        filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
        traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'))
    protocol = out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != scientific:
        raise RuntimeError('Evaluation protocol changed; use a new run_tag')
    save_json(protocol, scientific)
    data = {}
    for split in ('train','val'):
        data['ef_'+split] = TemporalEchoDataset(args.data_root, split, a.ef_frames, channels=3,
            limit=getattr(a,'ef_'+split+'_cases'),seed=a.seed,random_start=False,input_protocol=a.input_protocol)
        data['seg_'+split] = SegViews(args.data_root, split, a, 3, 2)
    for task in ('ef','seg'):
        ids = lambda d: d.ids if task=='ef' else d.patient_ids
        if set(ids(data[task+'_train'])) & set(ids(data[task+'_val'])):
            raise ValueError('Patient split overlap')
    manifest = {k:data_signature(v) for k,v in data.items()}
    mp = out/'data_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError('Selected data changed')
    save_json(mp, manifest)
    if (out/'DONE').exists():
        if not (out/'metrics.json').is_file():
            raise RuntimeError('DONE without metrics')
        return
    cache = Path(args.output_root)/args.run_tag/'cache'/out.parent.name
    begin = time.perf_counter()
    trials = []
    if a.batch_size == 0:
        a.batch_size, trials = tune_batch(model, data['ef_train'][0], device, maximum=16)
    save_json(out/'runtime.json', dict(batch_size=a.batch_size, trials=trials, device=str(device)))
    speed = benchmark(model, data['ef_train'][0], device)
    records, metrics = {}, dict(local_frames=model.local_frames, ef={}, seg={}, inference=speed,
        parameters=sum(p.numel() for p in model.parameters()), checkpoint_bytes=checkpoint.stat().st_size)
    def get(key):
        save_json(out/'status.json',dict(stage='extract_'+key,updated_unix=time.time()))
        result = extract(model, data[key], a, device, key.split('_')[0],cache/(key+'.npz'),scientific)
        write_csv(out/('validity_'+key+'.csv'),result['rows'])
        records[key] = result['stats']
        return result
    tr, va = get('ef_train'), get('ef_val')
    pred = ridge_apply(ridge_fit(torch.from_numpy(tr['fused']),torch.from_numpy(tr['y']),a.ridge_alpha),
                       torch.from_numpy(va['fused']))
    target = torch.from_numpy(va['y'])
    metrics['ef'] = regression_metrics(pred,target)
    rows = [dict(id=r['id'],patient=r['patient'],target=float(target[i]),prediction=float(pred[i]),
                 error=float(abs(pred[i]-target[i]))) for i,r in enumerate(va['rows'])]
    write_csv(out/'ef_predictions.csv',rows)
    del tr,va
    tr,va = get('seg_train'), get('seg_val')
    metrics['validity'] = dict(val_frames=len(va['rows']),
        partial_targets=sum(not r['target_tubelet_complete'] for r in va['rows']),
        full_context_frames=sum(r['full_context'] for r in va['rows']),
        future_frames=max(r['real_future_frames'] for r in va['rows']))
    if metrics['validity']['future_frames']:
        raise RuntimeError('End-of-window segmentation must not see future frames')
    del model
    if device.type=='cuda':
        torch.cuda.empty_cache()
    save_json(out/'status.json',dict(stage='fit_seg_fused',updated_unix=time.time()))
    metrics['seg'],rows,losses = seg_probe(tr,va,'fused',a,device,2)
    write_csv(out/'seg_predictions.csv',rows)
    write_csv(out/'seg_loss.csv',losses)
    metrics.update(extraction=records,wall_seconds=time.perf_counter()-begin)
    save_json(out/'metrics.json',metrics)
    save_json(out/'status.json',dict(stage='completed',updated_unix=time.time()))
    (out/'DONE').write_text(VERSION+'\n')
    for key in data:
        (cache/(key+'.npz')).unlink(missing_ok=True)


def summarize(root, lengths, seed):
    rows, pairs, completed = [], [], []
    for length in lengths:
        path = root/f'spatial_l{length}'/'endpoint'
        if not (path/'DONE').exists():
            continue
        m = json.loads((path/'metrics.json').read_text())
        timing = root/'stage_times.csv'
        with timing.open() as handle:
            stages = list(csv.DictReader(handle))
        train_seconds = sum(float(r['seconds']) for r in stages
                            if r['stage']==f'spatial_l{length}/pretrain')
        rows.append(dict(local_frames=length,clip_count=64//length,ef_mae=m['ef']['mae'],
            ef_rmse=m['ef']['rmse'],seg_dice=m['seg']['dice_patient_mean'],
            seg_global=m['seg']['dice_global'],train_process_seconds=train_seconds,
            evaluation_seconds=m['wall_seconds'],parameters=m['parameters'],
            checkpoint_mib=m['checkpoint_bytes']/2**20,**m['inference']))
        completed.append((length,path))
    for i,(control,pc) in enumerate(completed):
        for candidate,pa in completed[i+1:]:
            protocols = [json.loads((p/'protocol.json').read_text()) for p in (pc,pa)]
            for item in protocols:
                item.pop('checkpoint_sha256')
            if protocols[0]!=protocols[1] or (pc/'data_manifest.json').read_bytes()!=(pa/'data_manifest.json').read_bytes():
                raise RuntimeError('Cannot compare mismatched probe/data protocols')
            for task,field in [('ef','error'),('seg','dice')]:
                def read(path):
                    with (path/(task+'_predictions.csv')).open() as handle:
                        return list(csv.DictReader(handle))
                pairs.append(dict(candidate=candidate,control=control,task=task,
                    **paired_bootstrap(read(pa),read(pc),field,seed)))
    write_csv(root/'comparison.csv',rows)
    write_csv(root/'paired_differences.csv',pairs)
    lines = ['# Stage 1 local temporal length screening','',
        'Fixed spatial memory, total 64 frames, tubelet2, stride1. Endpoint frozen probes only.',
        'Validation screening, not held-out test; one training seed. No memory/no-memory comparison.',
        'EF reads fused outputs from all valid tubelets; segmentation reads target index63 with no future frames.',
        'EF extraction uses 256 frames with state reset every64; not unlimited streaming.',
        'Candidate-control: lower EF is better, higher Dice is better. Patient-bootstrap intervals exclude training-seed variation.',
        'Training process seconds include startup/autotune and recorded retries, not idle downtime.',
        'Local buffer wait in seconds=(L-1)/source_fps. FPS is not invented from NPY files.',
        'Different L changes local attention AND state-update frequency. This is a time-partition study, not a universal optimal frame count.','',
        '| L | EF MAE (pp) | Patient Dice | Train process (h) | Native64 batch1 (ms) |',
        '|---|---:|---:|---:|---:|']
    lines += [f"| {r['local_frames']} | {r['ef_mae']:.4f} | {r['seg_dice']:.5f} | {r['train_process_seconds']/3600:.2f} | {r['native_window_ms_median']:.2f} |" for r in rows]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive_analysis(root)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_tag',default='stage1_lengths_20260928')
    p.add_argument('--output_root',default='/root/autodl-tmp/outputs_stage1')
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--init_checkpoint',default='ckpt/mae/videomae_vit_s.pth')
    p.add_argument('--lengths',nargs='+',type=int,default=[8,16,32])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--eval_batch_size',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--autotune',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    p.add_argument('--evaluate_checkpoint',help=argparse.SUPPRESS)
    p.add_argument('--evaluation_output',help=argparse.SUPPRESS)
    args = p.parse_args()
    if (not args.lengths or len(set(args.lengths))!=len(args.lengths)
            or any(n not in (4,8,16,32) for n in args.lengths)):
        p.error('lengths must be unique members of 4,8,16,32')
    if args.epochs<1 or args.num_workers<0 or args.eval_batch_size<0:
        p.error('Invalid budget or runtime settings')
    if Path(args.run_tag).name!=args.run_tag or args.run_tag in ('.','..') or '\\' in args.run_tag:
        p.error('run_tag must be a directory name')
    if args.smoke:
        args.epochs=1
        if not args.run_tag.startswith('smoke_'):
            args.run_tag='smoke_'+args.run_tag
    return args


def main():
    args = parse_args()
    torch.set_num_threads(4)
    if args.evaluate_checkpoint:
        evaluate(Path(args.evaluate_checkpoint),Path(args.evaluation_output),args)
        return
    run = Path(args.output_root)/args.run_tag
    root, ckpts = run/'result',run/'ckpt'
    configs = {n:stage_config(args,n,ckpts/f'spatial_l{n}') for n in args.lengths}
    for n,cfg in configs.items():
        print(f'L={n} clips={64//n} memory=spatial epochs={args.epochs} effective_batch='
              f'{cfg["train"]["batch_size"]*cfg["train"]["grad_accum_steps"]} result={root}',flush=True)
    if args.dry_run:
        return
    for path in (Path(args.init_checkpoint),Path(args.data_root)/'FileList.csv',Path(args.data_root)/'VolumeTracings.csv'):
        if not path.is_file():
            raise FileNotFoundError(path)
    root.mkdir(parents=True,exist_ok=True)
    identity = dict(version=VERSION,code_sha256=code_hash(),initialization_sha256=file_hash(args.init_checkpoint),
        filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
        traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'))
    guard = root/'identity.json'
    if guard.exists() and json.loads(guard.read_text())!=identity:
        raise RuntimeError('Code/initialization/data changed; use a new run_tag')
    save_json(guard,identity)
    save_json(root/'plan.json',dict(arguments=vars(args),configs=configs,identity=identity))
    times = []
    if (root/'stage_times.csv').exists():
        with (root/'stage_times.csv').open() as handle:
            times=list(csv.DictReader(handle))
    for n,cfg in configs.items():
        name=f'spatial_l{n}'
        out=root/name
        out.mkdir(exist_ok=True)
        digest=check_protocol(out,cfg)
        (out/'protocol.sha256').write_text(digest+'\n')
        requested=out/'requested_config.yaml'
        requested.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        final=ckpts/name/f'epoch_{args.epochs:04d}.pt'
        if not (out/'PRETRAIN_DONE').exists():
            command=[sys.executable,'trainers/train_rmae.py','--config',str(requested),'--output_dir',str(out)]
            if args.autotune and not args.smoke:
                command.append('--autotune')
            run_command(command,name+'/pretrain',root,times)
            if not final.exists():
                raise RuntimeError(f'Missing final checkpoint: {final}')
            (out/'PRETRAIN_DONE').write_text(digest+'\n')
        command=[sys.executable,str(Path(__file__).resolve()),'--evaluate_checkpoint',str(final),
            '--evaluation_output',str(out/'endpoint'),'--run_tag',args.run_tag,'--output_root',args.output_root,
            '--data_root',args.data_root,'--num_workers',str(args.num_workers),
            '--eval_batch_size',str(args.eval_batch_size),'--seed',str(args.seed)]
        if args.smoke:
            command.append('--smoke')
        run_command(command,name+'/endpoint',root,times)
        summarize(root,args.lengths,args.seed)
    save_json(root/'current_stage.json',dict(stage='all_completed',status='completed',updated_unix=time.time()))
    (root/'DONE').write_text(VERSION+'\n')
    archive_analysis(root)
    print(f'Completed. Download {root / "analysis.zip"}',flush=True)


if __name__=='__main__':
    main()
