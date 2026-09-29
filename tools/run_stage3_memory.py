"""Reuse the four archived temporal checkpoints; run bounded endpoint diagnostics."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.run_temporal_research import run_command, archive_analysis
from tools.diagnose_stage2_interfaces import write_csv, save_json, file_hash
from utils.stage2_diagnostics import paired_bootstrap

NAMES = ('clip_mae_pool64','hier_global','hier_spatial','hier_dual')


def summarize(result, names, seed):
    rows, results = [], {}
    for name in names:
        folder = result/name
        if not (folder/'DONE').exists():
            continue
        metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        results[name] = metrics
        for condition, value in metrics['ef'].items():
            rows.append(dict(model=name,task='ef',condition=condition,metric='mae',value=value['mae'],
                             patients=metrics['val_patients'],seconds=metrics['wall_seconds']))
        if metrics['seg']:
            rows.append(dict(model=name,task='seg',condition='native_real',metric='dice_patient_mean',
                             value=metrics['seg']['dice_patient_mean'],seconds=metrics['wall_seconds']))
    write_csv(result/'comparison.csv',rows)
    pairs = []
    control = 'clip_mae_pool64'
    if control in results:
        def read(name,file):
            with (result/name/file).open(encoding='utf-8') as f:
                return list(csv.DictReader(f))
        for name in results:
            if name==control:
                continue
            # Compare actual identical windows. Do not silently intersect patients.
            modes = [(key+'.csv','error',key) for key in results[control]['ef'] if key.startswith('history_') and key.endswith('_cache')]
            if results[name]['seg'] and results[control]['seg']:
                modes.append(('seg.csv','dice','seg_native'))
            for file,field,condition in modes:
                a,b = read(name,file),read(control,file)
                byid={r['id']:r for r in b}
                fields=('source_frame','target_index') if field=='dice' else ('source_start','recent_start','source_end')
                for row in a:
                    if row['id'] not in byid or any(row[k]!=byid[row['id']][k] for k in fields):
                        raise ValueError('Cannot compare mismatched source windows')
                pairs.append(dict(candidate=name,control=control,condition=condition,
                                  **paired_bootstrap(a,b,field,seed)))
    write_csv(result/'cross_model_paired.csv',pairs)
    lines=['# Stage 3 checkpoint screening','',
           'No MAE pretraining or full fine-tuning. See per-model protocol.json before interpreting differences.',
           'Cross-model scores are descriptive: capacity, training epoch and saved configurations may differ.',
           'Patient bootstrap does not measure seed variability. State-only probes are auxiliary, never selection targets.','',
           '| Model | Task | Condition | Value |','|---|---|---|---:|']
    lines += [f"| {r['model']} | {r['task']} | {r['condition']} | {r['value']:.5f} |" for r in rows]
    (result/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_tag',default='stage3_memory_v1')
    p.add_argument('--checkpoint_root',default='/root/autodl-tmp/outputs_temporal/temporal_gray_20260909/ckpt')
    p.add_argument('--checkpoints_json',help='Optional JSON object: model name -> exact checkpoint path')
    p.add_argument('--epoch',type=int,default=400)
    p.add_argument('--only',nargs='+',choices=NAMES,default=list(NAMES))
    p.add_argument('--output_root',default='/root/autodl-tmp/outputs_stage3')
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--batch_size',type=int,default=0)
    p.add_argument('--max_batch_size',type=int,default=32)
    p.add_argument('--auto_workers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--with_seg',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--prefixes',nargs='+',type=int,default=[0,64,128])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args()
    if Path(args.run_tag).name!=args.run_tag or args.run_tag in ('','.','..') or '\\' in args.run_tag:
        p.error('run_tag must be a directory name')
    if len(set(args.only))!=len(args.only):
        p.error('Duplicate model names')
    if args.smoke and not args.run_tag.startswith('smoke_'):
        args.run_tag='smoke_'+args.run_tag
    paths={name:str(Path(args.checkpoint_root)/name/f'epoch_{args.epoch:04d}.pt') for name in args.only}
    if args.checkpoints_json:
        mapping=json.loads(Path(args.checkpoints_json).read_text(encoding='utf-8'))
        paths={name:mapping[name] for name in args.only}
    result=Path(args.output_root)/args.run_tag/'result'
    commands=[]
    for name,path in paths.items():
        cmd=[sys.executable,'tools/evaluate_stage3.py','--checkpoint',path,'--data_root',args.data_root,
             '--expected_epoch',str(args.epoch),'--expected_memory',dict(zip(NAMES,('none','global','spatial','dual')))[name],
             '--output_dir',str(result/name),'--cache_dir',str(result.parent/'cache'/name),
             '--num_workers',str(args.num_workers),'--batch_size',str(args.batch_size),
             '--max_batch_size',str(args.max_batch_size),'--seed',str(args.seed),
             '--prefixes',*[str(v) for v in args.prefixes]]
        if not args.auto_workers:
            cmd.append('--no-auto_workers')
        if not args.with_seg:
            cmd.append('--no-with_seg')
        if args.smoke:
            cmd.append('--smoke')
        print(f'{name}: reuse {path}; MAE epochs=0; endpoint-only',flush=True)
        commands.append((name,cmd))
    if args.dry_run:
        return
    # Preflight ALL requested files before consuming compute on the first model.
    for path in [*paths.values(),str(Path(args.data_root)/'FileList.csv'),str(Path(args.data_root)/'VolumeTracings.csv')]:
        if not Path(path).is_file():
            raise FileNotFoundError(f'{path}; supply --checkpoint_root or --checkpoints_json; never substitute random weights')
    result.mkdir(parents=True,exist_ok=True)
    identity=dict(checkpoints={k:dict(path=v,sha256=file_hash(v)) for k,v in paths.items()},
                  prefixes=args.prefixes,with_seg=args.with_seg,seed=args.seed,smoke=args.smoke)
    guard=result/'queue_identity.json'
    if guard.exists() and json.loads(guard.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Queue changed; choose a new run_tag')
    save_json(guard,identity)
    times=[]
    if (result/'stage_times.csv').exists():
        with (result/'stage_times.csv').open() as f:
            times=list(csv.DictReader(f))
    (result/'DONE').unlink(missing_ok=True)
    for name,cmd in commands:
        run_command(cmd,name+'/diagnose',result,times)
        summarize(result,args.only,args.seed)
    save_json(result/'current_stage.json',dict(stage='all_completed',status='completed'))
    (result/'DONE').write_text('Requested stage3 diagnostics completed\n')
    archive_analysis(result)
    print(f'Download only {result/"analysis.zip"}',flush=True)


if __name__=='__main__':
    main()
