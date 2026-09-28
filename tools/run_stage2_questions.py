"""Stage 2: reuse checkpoints first; opt into matched structural pretraining."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import yaml
import torch

from tools.run_temporal_research import make_config, run_command, archive_analysis
from tools.evaluate_stage2 import code_hash
from tools.diagnose_stage2_interfaces import save_json, file_hash, write_csv
from utils.research_storage import check_protocol
from utils.stage2_diagnostics import paired_bootstrap

VARIANTS = ('repeat','learned','tubelet1','joint')


def reuse_training_contract(config):
    """Compare scientific settings, not output paths or tuned runtime settings."""
    cfg = copy.deepcopy(config)
    cfg['experiment'] = {'seed': cfg['experiment']['seed']}
    for key in ('checkpoint', 'logging', 'runtime', 'autotune'):
        cfg.pop(key, None)
    model = cfg['model']
    model.pop('gradient_checkpointing', None)
    model.setdefault('frame_readout', 'repeat')
    for key in ('num_workers', 'pin_memory', 'persistent_workers', 'prefetch_factor'):
        cfg['data'].pop(key, None)
    train = cfg['train']
    batch = int(train.pop('batch_size'))
    train['effective_batch'] = batch * int(train.pop('grad_accum_steps', 1))
    train.setdefault('epoch_sample_batch', batch)
    for key in ('log_interval', 'val_interval', 'plot_interval'):
        train.pop(key, None)
    return cfg


def validate_reused_control(path, expected):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('partial_epoch') or payload.get('epoch') != expected['train']['epochs']:
        raise ValueError('Reused control must be a complete checkpoint at the requested epoch')
    if not payload.get('model_state_dict') or not isinstance(payload.get('config'), dict):
        raise ValueError('Reused control needs model_state_dict and its saved training config')
    actual = reuse_training_contract(payload['config'])
    target = reuse_training_contract(expected)
    if actual != target:
        changed = sorted(k for k in actual.keys() | target.keys() if actual.get(k) != target.get(k))
        raise ValueError(f'Reused control training protocol mismatch in: {changed}')
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path), epoch=payload['epoch'],
                training_contract=actual, source_config=payload['config'],
                note='Reused without training or copying weights. Saved config checked; '
                     'historical source code/initial-file bytes are not certified by this check.')


def stage_config(args, variant, checkpoint_dir):
    base = SimpleNamespace(seed=args.seed,data_root=args.data_root,input_protocol='gray_repeat3',
        num_workers=args.num_workers,prefetch_factor=4,init_checkpoint=args.init_checkpoint,
        audit_epochs=[0,args.epochs],save_last_every=10,min_free_gb=3.,epochs=args.epochs,
        batch_size=8,grad_accum_steps=4,baseline_batch_size=32,smoke=args.smoke)
    changes = dict(local_frames=args.local_frames,clip_count=64//args.local_frames,memory_mode='spatial',
                   tubelet_size=2,frame_readout='repeat')
    if variant=='learned':
        changes['frame_readout']='learned'
    elif variant=='tubelet1':
        changes.update(tubelet_size=1,patch_init_temporal_sum=True)
    elif variant=='joint':
        changes.update(local_frames=64,clip_count=1,memory_mode='none')
    elif variant!='repeat':
        raise ValueError(variant)
    cfg = make_config('temporal','stage2_'+variant,changes,base)
    cfg['experiment']['description'] = 'Stage 2 matched structure; no automatic stage-1 length selection'
    cfg['checkpoint'].update(dir=str(checkpoint_dir),save_best=False)
    cfg['train'].update(val_interval=1 if args.smoke else 25,plot_interval=1 if args.smoke else 5)
    return cfg


def summarize(root, names, seed=42):
    records,completed = [],[]
    for name in names:
        out=root/name/'endpoint'
        if not (out/'DONE').exists():
            continue
        result=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
        completed.append(name)
        for task in ('ef','seg'):
            field='mae' if task=='ef' else 'dice_patient_mean'
            for readout,m in result[task].items():
                records.append(dict(model=name,task=task,readout=readout,metric=field,value=m[field],
                    readout_parameters=m['parameters'],model_parameters=result['parameters_including_mae_decoder'],
                    source_checkpoint_mib=result['source_checkpoint_bytes']/2**20,
                    stream_clip_ms_p50=result['inference']['clip_update_ms_p50'],
                    stream_clip_ms_p95=result['inference']['clip_update_ms_p95'],
                    cache_bytes=result['inference']['resident_fifo_bytes'],
                    evaluation_seconds=result['wall_seconds']))
    write_csv(root/'comparison.csv',records)
    pairs=[]
    if 'repeat' in completed:
        for name in completed:
            if name=='repeat':
                continue
            # These queues use a common fixed sample protocol and initialization.
            for task,a,b,field in [('seg','real_fused','real_fused','dice'),
                                   ('ef','joint_recent' if name=='joint' else 'cache','cache','error')]:
                def read(label,mode):
                    with (root/label/'endpoint'/f'{task}_{mode}.csv').open(encoding='utf-8') as f:
                        return list(csv.DictReader(f))
                ar,br=read(name,a),read('repeat',b)
                byid={r['id']:r for r in br}
                for row in ar:
                    keys=('source_start','recent_start','source_end') if task=='ef' else ('source_frame','target_index')
                    if row['id'] not in byid or any(row[k]!=byid[row['id']][k] for k in keys):
                        raise RuntimeError('Cannot pair different source frames/windows')
                pairs.append(dict(candidate=name,control='repeat',task=task,
                                  **paired_bootstrap(ar,br,field,seed)))
    write_csv(root/'cross_structure_differences.csv',pairs)
    text=['# Stage 2 comparison','',
          'Frozen validation screening; no held-out test claims. Head-only results are not full fine-tuning.',
          'MAE training uses 64 real-frame slots. History probes extend to 128, reporting complete-history cases only.',
          'repeat/learned/tubelet1 vary representation AND decoder tokenization/cost; report these as structure treatments.',
          'joint is a separately trained 64-frame no-memory encoder, not an extended-position inference trick.',
          'EF joint/cache both reset at the recent-window boundary; C history/empty probes separately use the older prefix.',
          'Bootstrap intervals reflect patient sampling, not training-seed variation.','',
          '| Model | Task | Readout | Value |','|---|---|---|---:|']
    text += [f"| {r['model']} | {r['task']} | {r['readout']} | {r['value']:.5f} |" for r in records]
    (root/'comparison.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    archive_analysis(root)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['diagnose','train'],default='diagnose')
    p.add_argument('--checkpoint',help='Existing TemporalMAE; required for diagnose')
    p.add_argument('--reuse_control',help='Reuse a matched final repeat checkpoint in a training queue')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['repeat','learned'])
    p.add_argument('--local_frames',type=int,choices=[4,8,16,32],default=16)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--run_tag',default='stage2_questions_v1')
    p.add_argument('--output_root',default='/root/autodl-tmp/outputs_stage2')
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--init_checkpoint',default='ckpt/mae/videomae_vit_s.pth')
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--eval_batch_size',type=int,default=0)
    p.add_argument('--ef_head',choices=['attention','ridge'],default='attention')
    p.add_argument('--autotune',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.phase=='diagnose' and not args.checkpoint:
        p.error('--checkpoint is required for diagnose; no fallback to random initialization')
    if args.reuse_control and (args.phase!='train' or 'repeat' not in args.variants or args.smoke):
        p.error('--reuse_control requires train phase with repeat, and cannot be used with --smoke')
    if len(set(args.variants))!=len(args.variants) or args.epochs<1 or args.num_workers<0 or args.eval_batch_size<0:
        p.error('Invalid budget or duplicate variants')
    if not args.run_tag or Path(args.run_tag).name!=args.run_tag or args.run_tag in ('.','..') or '\\' in args.run_tag:
        p.error('run_tag must be a directory name')
    if args.smoke:
        args.epochs=1
        if not args.run_tag.startswith('smoke_'):
            args.run_tag='smoke_'+args.run_tag
    return args


def main():
    args=parse_args()
    run=Path(args.output_root)/args.run_tag
    result,weights=run/'result',run/'ckpt'
    names=args.variants if args.phase=='train' else ['existing']
    configs={name:stage_config(args,name,weights/name) for name in names} if args.phase=='train' else {}
    for name in names:
        print(f'{args.phase}: {name}, L={configs[name]["model"]["local_frames"] if configs else "checkpoint"}, '
              f'epochs={args.epochs if configs else 0}, '
              f'action={"reuse+evaluate" if name=="repeat" and args.reuse_control else args.phase}, '
              f'endpoint only; result={result/name}',flush=True)
    if args.dry_run:
        return
    init=Path(args.init_checkpoint if args.phase=='train' else args.checkpoint)
    for path in (init,Path(args.data_root)/'FileList.csv',Path(args.data_root)/'VolumeTracings.csv'):
        if not path.is_file():
            raise FileNotFoundError(path)
    reused = validate_reused_control(args.reuse_control, configs['repeat']) if args.reuse_control else None
    result.mkdir(parents=True,exist_ok=True)
    identity=dict(code_sha256=code_hash(),runner_sha256=file_hash(__file__),init_sha256=file_hash(init),
        phase=args.phase,local_frames=args.local_frames,epochs=args.epochs,seed=args.seed,ef_head=args.ef_head,
        filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
        traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'))
    if reused:
        identity['reused_control_sha256'] = reused['sha256']
    guard=result/'identity.json'
    if guard.exists() and json.loads(guard.read_text(encoding='utf-8'))!=identity:
        raise RuntimeError('Run identity changed; use a new run_tag')
    save_json(guard,identity)
    (result/'DONE').unlink(missing_ok=True)
    save_json(result/'plan.json',dict(arguments=vars(args),configs=configs))
    times=[]
    if (result/'stage_times.csv').exists():
        with (result/'stage_times.csv').open() as f:
            times=list(csv.DictReader(f))
    for name in names:
        out=result/name
        out.mkdir(exist_ok=True)
        checkpoint=init
        if configs and name=='repeat' and reused:
            checkpoint = Path(reused['path'])
            save_json(out/'reused_control.json', reused)
        elif configs:
            cfg=configs[name]
            digest=check_protocol(out,cfg)
            (out/'protocol.sha256').write_text(digest+'\n')
            requested=out/'requested_config.yaml'
            requested.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            checkpoint=weights/name/f'epoch_{args.epochs:04d}.pt'
            if not (out/'PRETRAIN_DONE').exists():
                cmd=[sys.executable,'trainers/train_rmae.py','--config',str(requested),'--output_dir',str(out)]
                if args.autotune and not args.smoke:
                    cmd.append('--autotune')
                run_command(cmd,name+'/pretrain',result,times)
                if not checkpoint.is_file():
                    raise RuntimeError('Training did not produce the requested final epoch')
                (out/'PRETRAIN_DONE').write_text(digest+'\n')
        cmd=[sys.executable,'tools/evaluate_stage2.py','--checkpoint',str(checkpoint),
            '--output_dir',str(out/'endpoint'),'--cache_dir',str(run/'cache'/name),
            '--data_root',args.data_root,'--batch_size',str(args.eval_batch_size),
            '--num_workers',str(args.num_workers),'--ef_head',args.ef_head,'--seed',str(args.seed)]
        if args.smoke:
            cmd.append('--smoke')
        run_command(cmd,name+'/endpoint',result,times)
        summarize(result,names,args.seed)
    (result/'DONE').write_text('Selected stage-two jobs completed\n')
    save_json(result/'current_stage.json',dict(stage='all_completed',status='completed'))
    archive_analysis(result)
    print(f'Download only {result / "analysis.zip"}',flush=True)


if __name__=='__main__':
    main()
