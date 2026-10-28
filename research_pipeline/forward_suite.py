"""Bounded B0 + D + A queue; W and extensions require explicit CLI choices."""

import argparse
import copy
import json
from pathlib import Path
import re

from .artifacts import layout, verify_done, write_json, file_hash
from .config import load, stage_config, validate
from .cli import execute, source_path
from .engine import csv_rows
from .metrics import paired_bootstrap


def build_plan(base, endpoint=None, with_w=False, baseline_result=None, confirm_final=False):
    cfg = copy.deepcopy(base)
    baseline = next(s for s in cfg['stages'] if s['id']=='pretrain')
    train = stage_config(cfg,baseline)['training']
    n = train.get('schedule_epochs') or train['epochs']
    k1,k2 = n//4,n//2
    endpoint = endpoint or k1
    if endpoint not in (k1,k2,n):
        raise ValueError('Only the predeclared K1/K2/N endpoints are allowed')
    if confirm_final and endpoint != n:
        raise ValueError('Full confirmation requires an explicitly selected N endpoint')
    if baseline_result:
        root = Path(baseline_result)
        actual = json.loads((root/'plan.json').read_text(encoding='utf-8'))
        saved = next(s for s in actual['stages'] if s['id']=='pretrain')
        actual_training = stage_config(actual,saved)['training']
        actual_n = actual_training.get('schedule_epochs') or actual_training['epochs']
        if actual_n != n:
            raise ValueError('Use the actual baseline plan as --config; do not substitute a new schedule')
        cfg['stages'] = []
        def b0(epoch):
            if epoch <= endpoint:
                return source_path(actual,dict(source=f'@pretrain/epoch_{epoch:04d}'))
            return str(layout(actual,'pretrain')['ckpt']/f'epoch_{epoch:04d}.pt')
    else:
        def b0(epoch):
            return f'@pretrain/epoch_{epoch:04d}'
        pins = set(baseline.get('training',{}).get('save_epochs',[]))
        if not {0,k1,k2,n}.issubset(pins):
            raise ValueError('Baseline recipe must already pin 0/K1/K2/N; do not silently alter a locked trajectory')
    selected = [s['id'] for s in cfg['stages']]
    for epoch in (k1,k2,n):
        for task in ('ef','seg'):
            name = f'{task}_linear_{epoch:04d}'
            if not any(s['id']==name for s in cfg['stages']):
                cfg['stages'].append(dict(id=name,kind=task,mode='frozen_linear',source=b0(epoch)))
            if epoch<=endpoint:
                selected.append(name)
    for exit_name in ('local','final'):
        for task in ('ef','seg'):
            name = f'D3_{exit_name}_{task}'
            cfg['stages'].append(dict(id=name,kind=task,mode='frozen_linear',source=b0(k1),exit_name=exit_name,
                                      data=dict(limit_train=64,limit_val=64)))
            selected.append(name)
    cfg['stages'].append(dict(id='D_paths_history',kind='forward_diagnostics',source=b0(k1),panel_cases=64,
                              gradient_cases=2,heads={task:f'@D3_final_{task}/best' for task in ('ef','seg')}))
    cfg['stages'].append(dict(id='D_visibility_causality',kind='mechanisms',source=b0(k1)))
    selected += ['D_paths_history','D_visibility_causality']
    for variant in ('A','W'):
        scientific = dict(forward_variant='mid_read',read_after_block=6) if variant=='A' else dict(forward_variant='content_write',writer_width=64)
        training = copy.deepcopy(train)
        training.update(schedule_epochs=n,run_until_epoch=endpoint,save_epochs=sorted({0,k1,k2,n}))
        cfg['stages'].append(dict(id=variant,kind='mae',source=b0(0),model=dict(overrides=scientific),training=training))
        enabled = variant=='A' or with_w
        if enabled:
            selected.append(variant)
        for epoch in (k1,k2,n):
            for task in ('ef','seg'):
                name = f'{variant}_{task}_{epoch:04d}'
                cfg['stages'].append(dict(id=name,kind=task,mode='frozen_linear',source=f'@{variant}/epoch_{epoch:04d}'))
                if enabled and epoch<=endpoint:
                    selected.append(name)
        cfg['stages'].append(dict(id=variant+'_diagnostics',kind='forward_diagnostics',source=f'@{variant}/epoch_{k1:04d}',
                             panel_cases=64,gradient_cases=2,heads={task:f'@{variant}_{task}_{k1:04d}/best' for task in ('ef','seg')}))
        if enabled:
            selected.append(variant+'_diagnostics')
        cfg['stages'].append(dict(id=variant+'_streaming',kind='streaming',source=f'@{variant}/epoch_{k1:04d}',
                                  audit=dict(max_frames=512,max_cases=2)))
        if enabled:
            selected.append(variant+'_streaming')
        for task in ('ef','seg'):
            for mode in ('frozen_lite','finetune'):
                name = f'{variant}_confirm_{task}_{mode}'
                cfg['stages'].append(dict(id=name,kind=task,mode=mode,source=f'@{variant}/final',
                     data=dict(augmentation=dict(enabled=mode=='finetune',preset='A4_tgc_zoom_speckle',per_frame_random=False))))
                if enabled and confirm_final:
                    selected.append(name)
    cfg['forward_budget'] = dict(schedule_epochs=n,k1=k1,k2=k2,
                                default_new_mae_runs=1,max_single_change_candidates=2,
                                no_automatic_promotion=True,baseline_external=bool(baseline_result),
                                baseline_result=str(Path(baseline_result).resolve()) if baseline_result else None)
    return validate(cfg),list(dict.fromkeys(selected))


def compare(cfg, selected):
    if cfg.get('smoke'):
        write_json(layout(cfg)['result']/'forward_comparison.json',dict(smoke=True,
                   scope='Checkpoint references remapped to one-epoch software fixture; no scientific ranking'))
        return
    rows, intervals = [], []
    stages = {s['id']:s for s in cfg['stages']}
    for name in selected:
        match = re.fullmatch(r'([AW])_(ef|seg)_([0-9]{4})',name)
        if not match:
            continue
        variant,task,epoch = match.groups()
        baseline = f'{task}_linear_{epoch}'
        a,b = layout(cfg,baseline)['result'],layout(cfg,name)['result']
        if not verify_done(a) or not verify_done(b):
            rows.append(dict(candidate=name,baseline=baseline,status='missing_matched_endpoint'))
            continue
        protocol_a = json.loads((a/'protocol.json').read_text())['protocol']
        protocol_b = json.loads((b/'protocol.json').read_text())['protocol']
        for key in ('training','evaluation','data'):
            if protocol_a[key] != protocol_b[key]:
                raise ValueError(f'Unmatched {key} in {baseline} vs {name}')
        for key in ('train_records_sha256','val_records_sha256','head'):
            if protocol_a['inputs'][key] != protocol_b['inputs'][key]:
                raise ValueError('Probe head or patient/window population mismatch')
        # Epoch comes from the loaded native checkpoint, not the supervised head fit epoch.
        if protocol_a['inputs']['initialization']['source_epoch'] != int(epoch) or protocol_b['inputs']['initialization']['source_epoch'] != int(epoch):
            raise ValueError('Checkpoint epoch mismatch')
        sa,sb = (json.loads((directory/'summary.json').read_text()) for directory in (a,b))
        metric = 'mae_pp' if task=='ef' else 'dice_mean'
        def cost(directory):
            import csv
            with (directory/'metrics.csv').open(encoding='utf-8-sig') as handle:
                history = [row for row in csv.DictReader(handle) if int(row['epoch'])<=int(epoch)]
            return dict(train_wall_hours=sum(float(r['train_seconds']) for r in history)/3600,
                        frames=sum(int(r.get('observed_frames') or 0) for r in history),
                        updates=int(history[-1]['successful_updates']) if history else 0)
        baseline_dir = (Path(cfg['forward_budget']['baseline_result'])/'pretrain' if cfg['forward_budget']['baseline_external']
                        else layout(cfg,'pretrain')['result'])
        costs = {f'{label}_{key}':value for label,directory in (('b0',baseline_dir),(variant,layout(cfg,variant)['result']))
                 for key,value in cost(directory).items()}
        rows.append(dict(candidate=name,baseline=baseline,source_epoch=int(epoch),status='matched',task=task,
                         baseline_protocol=sa['protocol_sha256'],candidate_protocol=sb['protocol_sha256'],
                         baseline_checkpoint_sha256=file_hash(protocol_a['stage']['source']) if not protocol_a['stage']['source'].startswith('@') else file_hash(source_path(cfg,stages[baseline])),
                         candidate_checkpoint_sha256=file_hash(source_path(cfg,stages[name])),**costs,
                         baseline_metric=sa['metrics'][metric],candidate_metric=sb['metrics'][metric],
                         delta_candidate_minus_baseline=sb['metrics'][metric]-sa['metrics'][metric],
                         candidate_head_train_seconds=sb['completed_train_seconds'],baseline_head_train_seconds=sa['completed_train_seconds']))
        patients = [json.loads((directory/'patients.json').read_text()) for directory in (a,b)]
        intervals.append(dict(candidate=name,task=task,**paired_bootstrap(*patients,'absolute_error' if task=='ef' else 'dice')))
    root = layout(cfg)['result']
    csv_rows(root/'forward_comparison.csv',rows)
    write_json(root/'forward_comparison.json',dict(rows=rows,paired_patient_intervals=intervals,
                selection='Human decision required; no CI/loss threshold automatically extends or selects candidates',
                budget=cfg['forward_budget']))
    hf = []
    for task in ('ef','seg'):
        directories = [layout(cfg,f'D3_{exit_name}_{task}')['result'] for exit_name in ('local','final')]
        if not all(verify_done(p) for p in directories):
            continue
        protocols = [json.loads((p/'protocol.json').read_text())['protocol'] for p in directories]
        for key in ('train_records_sha256','val_records_sha256','head'):
            if protocols[0]['inputs'][key] != protocols[1]['inputs'][key]:
                raise ValueError('H/F probe population or capacity mismatch')
        summaries = [json.loads((p/'summary.json').read_text()) for p in directories]
        metric = 'mae_pp' if task=='ef' else 'dice_mean'
        hf.append(dict(task=task,H=summaries[0]['metrics'][metric],F=summaries[1]['metrics'][metric],
                       paired=paired_bootstrap(*[json.loads((p/'patients.json').read_text()) for p in directories],
                                                'absolute_error' if task=='ef' else 'dice'),
                       scope='Same-capacity refitted TRAIN/VAL diagnostic panel heads, not fixed-head sensitivity'))
    write_json(root/'hf_readability.json',hf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pipeline/clip_recurrent_global.yaml')
    parser.add_argument('--tag',default='clip_recurrent_forward_v1')
    parser.add_argument('--root')
    parser.add_argument('--set',action='append',default=[])
    parser.add_argument('--baseline-result')
    parser.add_argument('--endpoint',type=int)
    parser.add_argument('--with-w',action='store_true')
    parser.add_argument('--confirm-final',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--plan-output')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    base = load(args.config,args.set,args.tag,args.root)
    cfg,selected = build_plan(base,args.endpoint,args.with_w,args.baseline_result,args.confirm_final)
    if args.smoke:
        # Apply the same real-model truncation as the public smoke command.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'smoke.json'
            write_json(path,cfg)
            cfg = load(path,smoke=True)
        # Diagnostic head references remain references to the independently fitted smoke heads.
    if args.plan_output:
        write_json(args.plan_output,cfg)
    print(json.dumps(dict(budget=cfg['forward_budget'],selected=selected,
                         note='W is present but disabled unless --with-w; no automatic K2/N extension'),indent=2),flush=True)
    if args.plan_only:
        return
    execute(cfg,selected)
    compare(cfg,selected)


if __name__=='__main__':
    main()
