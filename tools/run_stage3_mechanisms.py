"""Minimal matched mechanism screen: one compression, one write-source change."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import yaml

from tools.run_stage1_lengths import stage_config as length_config
from tools.run_stage2_questions import validate_reused_control
from tools.run_temporal_research import run_command, archive_analysis
from tools.diagnose_stage2_interfaces import file_hash, save_json, write_csv
from utils.research_storage import check_protocol
from utils.stage2_diagnostics import paired_bootstrap

VARIANTS=('baseline','none','temporal_pool','local_write')


def configuration(args, name, weights):
    cfg=length_config(args,16,weights)
    cfg['experiment']['name']='stage3_'+name
    cfg['experiment']['description']='Matched 100-epoch mechanism screen; endpoint evaluation only'
    if name=='none':
        cfg['model']['memory_mode']='none'
    elif name=='temporal_pool':
        cfg['model']['memory_compression']='temporal_attention'
    elif name=='local_write':
        cfg['model']['memory_write_source']='local'
    elif name!='baseline':
        raise ValueError(name)
    cfg['checkpoint']['save_initial']=False
    return cfg


def summarize(root,names,seed):
    rows,pairs=[],[]
    for name in names:
        out=root/name/'endpoint'
        if not (out/'DONE').is_file():
            continue
        m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(model=name,ef_no_prefix=m['ef']['history_0_cache']['mae'],
            ef_prefix128=m['ef']['history_128_cache']['mae'],dice=m['seg']['dice_patient_mean'],
            parameters=m['total_parameters'],train_patients=m['train_patients'],val_patients=m['val_patients']))
    for control in ('baseline','none'):
        if control not in [r['model'] for r in rows]:
            continue
        for row in rows:
            name=row['model']
            if name==control:
                continue
            for file,field in [('history_0_cache.csv','error'),('history_128_cache.csv','error'),('seg.csv','dice')]:
                def read(label):
                    with (root/label/'endpoint'/file).open(encoding='utf-8') as f:
                        return list(csv.DictReader(f))
                a,b=read(name),read(control)
                source=('source_frame','target_index') if field=='dice' else ('source_start','recent_start','source_end')
                byid={r['id']:r for r in b}
                if any(r['id'] not in byid or any(r[k]!=byid[r['id']][k] for k in source) for r in a):
                    raise ValueError('Mechanism pairing has different source windows')
                pairs.append(dict(candidate=name,control=control,metric=file,
                    **paired_bootstrap(a,b,field,seed)))
    write_csv(root/'mechanism_metrics.csv',rows)
    write_csv(root/'mechanism_paired.csv',pairs)


def main():
    os.chdir(ROOT)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_tag',default='stage3_mechanisms_20260929')
    p.add_argument('--output_root',default='/root/autodl-tmp/outputs_stage3')
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--init_checkpoint',default='ckpt/mae/videomae_vit_s.pth')
    p.add_argument('--reuse_baseline',required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--only',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args()
    if Path(args.run_tag).name!=args.run_tag or args.run_tag in ('','.','..') or '\\' in args.run_tag:
        p.error('Invalid run_tag')
    if len(set(args.only))!=len(args.only) or args.epochs<1 or args.num_workers<0:
        p.error('Invalid budget/selection')
    if args.smoke:
        args.epochs=1
        args.run_tag='smoke_'+args.run_tag
        args.only=[n for n in args.only if n!='baseline']
    run=Path(args.output_root)/args.run_tag
    result,weights=run/'result',run/'ckpt'
    configs={n:configuration(args,n,weights/n) for n in args.only}
    for name in args.only:
        print(f'{name}: {"reuse" if name=="baseline" else str(args.epochs)+" epochs"}; endpoint-only',flush=True)
    if args.dry_run:
        return
    for path in (Path(args.init_checkpoint),Path(args.data_root)/'FileList.csv',Path(args.data_root)/'VolumeTracings.csv'):
        if not path.is_file():
            raise FileNotFoundError(path)
    reused=validate_reused_control(args.reuse_baseline,configs['baseline']) if 'baseline' in configs else None
    result.mkdir(parents=True,exist_ok=True)
    identity=dict(seed=args.seed,epochs=args.epochs,init_hash=file_hash(args.init_checkpoint),
                  reuse_hash=reused['sha256'] if reused else None,configs=configs)
    guard=result/'queue_identity.json'
    if guard.exists() and json.loads(guard.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Run protocol changed')
    save_json(guard,identity)
    times=[]
    if (result/'stage_times.csv').exists():
        with (result/'stage_times.csv').open() as f:
            times=list(csv.DictReader(f))
    for name,cfg in configs.items():
        out=result/name
        out.mkdir(exist_ok=True)
        checkpoint=Path(reused['path']) if name=='baseline' else weights/name/f'epoch_{args.epochs:04d}.pt'
        if name=='baseline':
            save_json(out/'reused_baseline.json',reused)
        elif not (out/'PRETRAIN_DONE').exists():
            digest=check_protocol(out,cfg)
            (out/'protocol.sha256').write_text(digest+'\n')
            requested=out/'requested_config.yaml'
            requested.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            command=[sys.executable,'trainers/train_rmae.py','--config',str(requested),'--output_dir',str(out)]
            if not args.smoke:
                command.append('--autotune')
            run_command(command,name+'/pretrain',result,times)
            if not checkpoint.is_file():
                raise ValueError('Missing requested final checkpoint')
            (out/'PRETRAIN_DONE').write_text(digest+'\n')
        cmd=[sys.executable,'tools/evaluate_stage3.py','--checkpoint',str(checkpoint),
            '--output_dir',str(out/'endpoint'),'--cache_dir',str(run/'cache'/name),'--data_root',args.data_root,
            '--expected_epoch',str(args.epochs),'--expected_memory','none' if name=='none' else 'spatial',
            '--ef_train_cases','512','--ef_val_cases','256','--seg_train_cases','64','--seg_val_cases','64',
            '--ef_steps','400','--seg_steps','200','--seed',str(args.seed),'--num_workers',str(args.num_workers)]
        if args.smoke:
            cmd.append('--smoke')
        else:
            cmd.append('--eligible_budget')
        run_command(cmd,name+'/endpoint',result,times)
        summarize(result,args.only,args.seed)
    (result/'DONE').write_text('Requested mechanism screening completed; no automatic winning-model claim\n')
    save_json(result/'current_stage.json',dict(stage='all_completed',status='completed'))
    archive_analysis(result)


if __name__=='__main__':
    main()
