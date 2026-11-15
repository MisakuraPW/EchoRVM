"""Bounded A-R Stage B: EF-only warmup and matched variable-boundary comparison."""

import argparse
from collections import Counter
import copy
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import ROOT,code_identity,file_hash,layout,mark_done,run_lock,verify_done,write_json
from research_pipeline.cli import configure,execute
from research_pipeline.config import safe_name,stage_config,validate
from tools import run_patch8_stage_a as a
from tools.run_spatial_ef_readout import matched_observations,paired


IDS=('B1','B_uniform','B_observed')


def baseline(args):
    ar=args.ar_root/'result'/'patch8_stage_a_v1_AR'
    stage='ef_l1_ft'
    ft=a.completed(ar/stage)
    request=a.read(ar/stage/'request.json')
    protocol=a.verify_inputs(ar/stage)
    pre=a.completed(ar/'pretrain')
    source=request['source']
    sha=file_hash(source)
    if (sha!=protocol['source'] or sha!=pre['checkpoints']['epoch_0400.pt']['sha256']
            or pre['epoch']!=400 or pre['smoke'] or pre['synthetic']
            or ft['mode']!='finetune' or ft['selection']!='VAL only'):
        raise ValueError('Stage B requires unchanged Stage A A-R400 and its original EF FT recipe')
    return request['config'],protocol['inputs'],dict(path=source,sha256=sha),ft


def plan(args,epochs=40):
    original,inputs,source,ft=baseline(args)
    cfg=copy.deepcopy(original)
    prior=cfg.pop('stage')
    cfg.update(adapter='research_pipeline.variable_ef:build',
               run=dict(root=str(args.root),tag=args.tag),stage_b_ssl=source)
    cfg['evaluation']['cache_frozen']=False
    cfg['runtime'].update(batch_candidates=[2,4,8,16],worker_candidates=[0,4,8],
                          checkpoint_candidates=[False,True],thread_candidates=[2],
                          probe_warmup_updates=1,probe_updates=3,min_free_gb=5,evict_completed_feature_caches=False)
    cfg['training'].update(resume_every=1)
    cfg['stages']=[]
    for i,name in enumerate(IDS):
        stage=dict(copy.deepcopy(prior),id=name,
                   source=source['path'] if i==0 else '@B1/best',
                   boundary_policy='observed' if i==2 else 'uniform',
                   training=dict(epochs=20 if i==0 else epochs,patience=None if i==0 else 10,
                                 min_epochs=0 if i==0 else 10))
        stage['matched_inputs']={key:inputs[key] for key in (
            'train_records_sha256','val_records_sha256','data_files','head','loss')}
        cfg['stages'].append(stage)
    return validate(cfg),ft


def smoke_config(cfg):
    cfg=copy.deepcopy(cfg)
    cfg['run']['tag']+='_smoke'
    cfg['smoke']=True
    cfg['data'].update(limit_train=6,limit_val=3)
    cfg['training'].update(effective_batch=2,tensorboard=False)
    for stage in cfg['stages']:
        stage['training'].update(epochs=1,patience=None,min_epochs=0)
    cfg['runtime'].update(micro_batch=2,worker_candidates=[0],autotune=False)
    return validate(cfg)


def pace(cfg,control):
    from research_pipeline.runtime import run_probe_process
    directory=control/'calibration'
    directory.mkdir(parents=True,exist_ok=True)
    rows={}
    for policy in ('uniform','observed'):
        spec=stage_config(cfg,dict(cfg['stages'][0],boundary_policy=policy))
        spec['runtime'].update(probe_updates=5,probe_warmup_updates=2)
        options=[]
        for micro,checkpoint in ((4,False),(8,False),(16,False),(8,True)):
            name=f'{policy}_{micro}_{int(checkpoint)}'
            path=directory/f'{name}.request.json'
            response=directory/f'{name}.response.json'
            selection=dict(micro_batch=micro,num_workers=4,gradient_checkpointing=checkpoint,threads=2)
            write_json(path,dict(config=spec,source=cfg['stage_b_ssl']['path'],selection=selection,response=str(response)))
            with (directory/f'{name}.log').open('w',encoding='utf-8') as log:
                run_probe_process([sys.executable,'-m','research_pipeline','_probe','--request',str(path)],log,180)
            if response.exists():
                row=a.read(response)
                if row['status']=='error':
                    raise RuntimeError(f'Budget probe failed: {row}')
                if row['status']=='ok':
                    options.append(row)
            print(f'budget_probe {name}: {a.read(response) if response.exists() else "timeout"}',flush=True)
        if not options:
            raise RuntimeError(f'No budget-sized calibration succeeded for {policy}')
        rows[policy]=max(options,key=lambda item:item['samples_per_second'])
    return rows


def fit_budget(rates,train_cases,val_cases,budget_h):
    # Inflate measured steady-state cost; reserve all three TESTs and extra VAL/audit.
    estimates={k:1.20*(train_cases/v['samples_per_second']+val_cases/v['samples_per_second']*.55)+8.
               for k,v in rates.items()}
    reserve=1800
    remaining=budget_h*3600-reserve-20*estimates['uniform']
    epochs=min(40,math.floor(remaining/sum(estimates.values())))
    if epochs<10:
        raise RuntimeError(f'20 warmup + two matched 10-epoch arms do not fit {budget_h}h: {estimates}')
    return dict(b2_epochs=epochs,epoch_seconds=estimates,reserve_seconds=reserve,
                estimated_hours=(reserve+20*estimates['uniform']+epochs*sum(estimates.values()))/3600,
                budget_hours=budget_h, measured=rates)


def report(cfg,control,ar_ft,data_root):
    from research_pipeline.engine import Heartbeat,amp_context,build_experiment,evaluate as eval_dataset
    from statistics import median
    from tools.evaluate_patch8_test import locked_checkpoint
    import torch
    results=layout(cfg)['result']
    endpoints={name:locked_checkpoint(Path(cfg['run']['root']),cfg['run']['tag'],name) for name in IDS}
    rows={name:a.read(results/name/'observations.json') for name in IDS}
    subjects={name:a.read(results/name/'patients.json') for name in IDS}
    for name in IDS[1:]:
        matched_observations(rows['B1'],rows[name])
    val_pair=paired(subjects['B_uniform'],subjects['B_observed'])
    lock={name:file_hash(x[2]) for name,x in endpoints.items()}
    a.lock_selection(control/'VAL_LOCK.json',dict(checkpoints=lock,
                     gain_pp=-val_pair['candidate_minus_reference'],
                     reaches_practical_gain=-val_pair['candidate_minus_reference']>=.10))
    warm=control/'B1_fixed_val'
    if not verify_done(warm):
        directory,summary,path,request=endpoints['B1']
        settings=request['config']
        configure(settings)
        exp=build_experiment(settings,request['source'],torch.device(settings['runtime']['device']))
        exp.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'],strict=True)
        exp.model.eval()
        exp.eval_policy='fixed'
        runtime=a.read(directory/'runtime.json')['selected']
        begin=time.perf_counter()
        _,metrics,pats,observations=eval_dataset(exp,settings,runtime,Heartbeat(warm,15))
        matched_observations(rows['B1'],observations)
        write_json(warm/'summary.json',dict(metrics=metrics,scope='Same B1 weights, fixed L16 inference on VAL only',
                   checkpoint_sha256=lock['B1'],paired_fixed_minus_random=paired(subjects['B1'],pats),
                   wall_seconds=time.perf_counter()-begin))
        mark_done(warm,['summary.json'])
        del exp
        torch.cuda.empty_cache()
    test_root=results.with_name(results.name+'_test')
    # One TEST evaluation per formally trained arm, using the previously selected weights.
    a.evaluate(cfg,IDS,data_root)
    test={name:a.completed(test_root/name) for name in IDS}
    if not all(test[name]['checkpoint_sha256']==lock[name] for name in IDS):
        raise ValueError('TEST does not bind all three VAL weights')
    for name in IDS[1:]:
        matched_observations(a.read(test_root/'B1'/'observations.json'),a.read(test_root/name/'observations.json'))
    test_pair=paired(*(a.read(test_root/name/'patients.json') for name in IDS[1:]))
    timing={}
    for name in IDS:
        _,_,checkpoint,request=endpoints[name]
        settings=request['config']
        configure(settings)
        device=torch.device(settings['runtime']['device'])
        exp=build_experiment(settings,request['source'],device)
        exp.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        exp.model.eval()
        measurements=[]
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync=lambda: torch.cuda.synchronize(device) if device.type=='cuda' else None
        with torch.no_grad(),amp_context(settings,device):
            for i in range(min(35,len(exp.val))):
                sample=exp.val[i]
                sync()
                begin=time.perf_counter()
                exp._forward([sample])
                sync()
                if i>=3:
                    measurements.append(time.perf_counter()-begin)
        timing[name]=dict(p50_ms=1000*median(measurements),p90_ms=1000*sorted(measurements)[int(.9*(len(measurements)-1))],
                          patients=len(measurements),scope='batch1, predecoded video, H2D+observer+encoder+head; no acquisition',
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        del exp
        torch.cuda.empty_cache()
    result=dict(val={name:endpoints[name][1] for name in IDS},val_pair=val_pair,
                ar_fixed_reference=dict(metrics=ar_ft['metrics'],epochs=ar_ft['epoch'],scope='different budget, descriptive only'),
                b1_fixed_val=a.read(warm/'summary.json'),test=test,test_pair=test_pair,inference_timing=timing,
                scope='Supervised EF-only adaptation of A-R SSL400. Transductive SSL. No segmentation or SSL claims.')
    result['val_policy']={name:dict(length_counts=dict(Counter(l for p in rows[name] for l in p['clip_lengths'])),
                                  sum_length_squared=sum(sum(l*l for l in p['clip_lengths']) for p in rows[name])/len(rows[name]),
                                  acquisition_delay_frames='clip representations released only at their boundary (8-32 observed frames)')
                          for name in IDS}
    from statistics import mean
    motion={key:[v for row in rows['B_observed'] for v in row[key]]
            for key in ('free_stop_motion','free_continue_motion')}
    result['observed_motion']={key:dict(count=len(v),mean=mean(v) if v else None) for key,v in motion.items()}
    write_json(control/'comparison.json',result)
    mark_done(control,['budget.json','protocol.json','VAL_LOCK.json','comparison.json','B1_fixed_val/DONE.json'])
    print('VAL observed minus uniform',val_pair,flush=True)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','smoke','run'))
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-fs/outputs_cr_stage_b'))
    parser.add_argument('--ar-root',type=Path,default=Path('/root/autodl-fs/outputs_cr_stage_a'))
    parser.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    parser.add_argument('--tag',default='stage_b_w64_20261116')
    args=parser.parse_args()
    safe_name(args.tag)
    cfg,ft=plan(args)
    if args.action=='plan':
        print(json.dumps(cfg,indent=2))
        return
    if args.action=='smoke':
        execute(smoke_config(cfg))
        return
    control=args.root/'control'/args.tag
    with run_lock(control):
        if verify_done(control):
            print(control/'comparison.json')
            return
        if shutil.disk_usage(args.root).free < 12*1024**3:
            raise OSError('Stage B needs 12 GiB including the 5 GiB write floor; no old weights are removed')
        if (control/'budget.json').is_file():
            budget=a.read(control/'budget.json')
        else:
            _,inputs,_,_=baseline(args)
            rates=pace(cfg,control)
            budget=fit_budget(rates,len(inputs['train_ids']),len(inputs['val_ids']),8)
            write_json(control/'budget.json',budget)
        cfg,ft=plan(args,budget['b2_epochs'])
        a.lock_selection(control/'protocol.json',dict(config=cfg,code=code_identity([__file__])))
        print('Locked budget',budget,flush=True)
        execute(cfg)
        report(cfg,control,ft,args.data_root)


if __name__=='__main__':
    main()
