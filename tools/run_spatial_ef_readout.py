"""Train only two frozen EF query heads and run one paired, VAL-locked TEST pass."""

import argparse
import copy
import json
import shutil
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import code_identity, file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import configure, execute
from research_pipeline.config import safe_name, stage_config, validate
from tools import run_patch8_stage_a as a


ARMS = ('C1','C2')
MIN_EFFECT_PP = .10


def plan(args):
    c0 = args.a2_root/'result'/f'{args.a2_tag}_H'/'ef_pool_frozen'
    summary = a.completed(c0)
    protocol = a.verify_inputs(c0)
    request = a.read(c0/'request.json')
    source = request['source']
    ar = Path(source).parent
    ar_results = ar.parents[2]/'result'/ar.parent.name/'pretrain'
    ar_summary = a.completed(ar_results)
    native = ar_summary['checkpoints']['epoch_0400.pt']
    if (Path(source).name!='epoch_0400.pt' or not ar.parent.name.endswith('_AR') or
            file_hash(source)!=native['sha256'] or native['sha256']!=protocol['source'] or
            ar_summary['epoch']!=400 or ar_summary['selection']!='final fixed epoch' or ar_summary['smoke'] or
            summary['smoke'] or summary['synthetic'] or summary['mode']!='frozen_lite' or summary['selection']!='VAL only' or
            protocol['inputs']['head']!='TemporalEFReadout, zero last weight, bias=.556'):
        raise ValueError('C0 is not a matched original A-R SSL-400 frozen readout')
    cfg = copy.deepcopy(request['config'])
    prior = cfg.pop('stage')
    data,train = cfg['data'],cfg['training']
    if (not prior.get('calibrated_ef') or data['augmentation'].get('enabled') or
            (data['local_frames'],data['recent_frames'],data['prefix'],data['max_prefix'])!=(16,64,0,0) or
            (train['epochs'],train['patience'],train['effective_batch'])!=(100,10,32) or
            any(cfg[k]!=protocol[k] for k in ('data','evaluation')) or
            {k:v for k,v in train.items() if k!='run_until_epoch'}!=protocol['training'] or
            cfg['runtime']['precision']!=protocol['precision']):
        raise ValueError('C0 recipe differs from the locked frozen EF protocol')
    cfg.update(adapter='research_pipeline.spatial_ef:build',
               run=dict(root=str(args.root),tag=args.tag,cache_root=str(args.cache_root)))
    cfg['runtime'].update(cache_gb=41,min_free_gb=3,evict_completed_feature_caches=False,
                          batch_candidates=[4,8,16,32], worker_candidates=[0,2,4],
                          checkpoint_candidates=[False], thread_candidates=[2],
                          prefetch_factor=1, probe_warmup_updates=1, probe_updates=4)
    cfg['stages'] = [dict(copy.deepcopy(prior),id=name,source=source,spatial_ef_view=view,
                          matched_inputs={key:protocol['inputs'][key] for key in
                                          ('train_records_sha256','val_records_sha256','data_files','loss')})
                     for name,view in zip(ARMS,('mean','dense'))]
    return validate(cfg),c0,source


def smoke_plan(cfg):
    cfg=copy.deepcopy(cfg)
    cfg['run']['tag'] += '_smoke'
    cfg['smoke']=True
    cfg['data'].update(limit_train=12,limit_val=6)
    cfg['runtime'].update(autotune=False,micro_batch=4,worker_candidates=[0],min_free_gb=.1)
    cfg['training'].update(epochs=1,effective_batch=4,tensorboard=False)
    return validate(cfg)


def paired(reference,candidate):
    from research_pipeline.metrics import paired_bootstrap
    first={r['patient']:r['target'] for r in reference}
    second={r['patient']:r['target'] for r in candidate}
    if first!=second or len(first)!=len(reference) or len(second)!=len(candidate):
        raise ValueError('EF comparison requires identical patient IDs and labels')
    return paired_bootstrap(reference,candidate,'absolute_error')


def matched_observations(a_rows,b_rows):
    keys = ('sample_id','patient','recent_start','frames','prefix_frames','target')
    if [tuple(r[k] for k in keys) for r in a_rows] != [tuple(r[k] for k in keys) for r in b_rows]:
        raise ValueError('Readout cohorts, frame windows or labels differ')


def endpoint(cfg,arm,source):
    directory = layout(cfg,arm)['result']
    summary = a.completed(directory)
    checkpoint = layout(cfg,arm)['ckpt']/summary['best_file']
    protocol = a.verify_inputs(directory)
    if (summary['selection']!='VAL only' or summary['mode']!='frozen_lite' or summary['smoke'] or
            protocol['source']!=file_hash(source) or file_hash(checkpoint)!=summary['checkpoints'][checkpoint.name]['sha256']):
        raise ValueError('Expected a verified VAL-selected frozen head bound to the same SSL checkpoint')
    return directory,summary,checkpoint


def val_comparison(cfg,c0,source):
    reports={name:endpoint(cfg,name,source) for name in ARMS}
    rows={name:a.read(d/'observations.json') for name,(d,_,_) in reports.items()}
    matched_observations(*rows.values())
    integrity=[a.read(d/'frozen_integrity.json') for d,_,_ in reports.values()]
    protocols=[a.verify_inputs(d) for d,_,_ in reports.values()]
    c0_protocol=a.verify_inputs(c0)
    for p in protocols:
        if any(p[k]!=c0_protocol[k] for k in ('source','data','training','evaluation','precision')):
            raise ValueError('Frozen head protocol differs from C0')
    if (integrity[0]!=integrity[1] or len({v['cache']['directory'] for _,v,_ in reports.values()})!=1
            or len({v['trainable_parameters'] for _,v,_ in reports.values()})!=1
            or protocols[0]['inputs']['head_initial_sha256']!=protocols[1]['inputs']['head_initial_sha256']):
        raise ValueError('Heads do not share cache, frozen backbone or parameter budget')
    old_cache = Path(a.completed(c0)['cache']['directory'])
    new_cache = Path(reports['C1'][1]['cache']['directory'])
    key = ('patient','sample_id','recent_start','prefix_frames')
    compared_splits = []
    for split in ('train','val'):
        previous = old_cache/split/'records.jsonl'
        if previous.is_file():
            def windows(path):
                return [tuple(json.loads(line)[k] for k in key) for line in path.read_text(encoding='utf-8').splitlines()]
            if windows(previous)!=windows(new_cache/split/'records.jsonl'):
                raise ValueError(f'C0 {split} cached windows differ')
            compared_splits.append(split)
    patients={name:a.read(d/'patients.json') for name,(d,_,_) in reports.items()}
    subjects=a.read(c0/'patients.json')
    values={name:s['metrics'] for name,(_,s,_) in reports.items()}
    values['C0']=a.completed(c0)['metrics']
    return dict(metrics=values,paired=dict(C2_minus_C1=paired(patients['C1'],patients['C2']),
                C1_minus_C0=paired(subjects,patients['C1']),C2_minus_C0=paired(subjects,patients['C2'])),
                cost={name:{k:s[k] for k in ('selected_epoch','epoch','wall_seconds','completed_train_seconds',
                                            'completed_validation_seconds','trainable_parameters','cache')}
                      for name,(_,s,_) in reports.items()},
                C0_scope='Frozen VAL only; no historic full fine-tuning or historical TEST result is substituted',
                C0_cached_window_splits_verified=compared_splits)


def test_pair(cfg, source, data_root, control):
    import torch
    from research_pipeline.echo import IdentifiedDataset
    from research_pipeline.engine import Heartbeat, amp_context, build_experiment, csv_rows
    from research_pipeline.metrics import summarize
    from research_pipeline.runtime import loader
    from research_pipeline.spatial_ef import QueryEF, spatial_mean
    from tools.watch_pipeline_progress import StageDisplay
    from utils.final_temporal_data import WindowDataset, build_manifest
    if cfg.get('smoke'):
        raise ValueError('Smoke must never access TEST labels')
    endpoints = {name:endpoint(cfg,name,source) for name in ARMS}
    weights = {name:dict(sha256=file_hash(p),selected_epoch=s['selected_epoch']) for name,(_,s,p) in endpoints.items()}
    identity = dict(weights=weights, source_sha256=file_hash(source), data_root=str(data_root),
                    code=code_identity([__file__]), sampling='same unaugmented W64/H0 frozen EF protocol')
    a.lock_selection(control/'TEST_LOCK.json',identity)
    dest=control/'test'
    if verify_done(dest):
        return a.read(dest/'summary.json')
    settings = stage_config(cfg,cfg['stages'][0])
    configure(settings)
    device=torch.device(settings['runtime']['device'])
    exp=build_experiment(settings,source,device)
    heads={'C1':exp.model,'C2':QueryEF(True).to(device)}
    for name,(directory,_,path) in endpoints.items():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        report=a.read(directory/'frozen_integrity.json')
        if (not payload.get('head_only') or payload['epoch']!=weights[name]['selected_epoch']
                or payload['frozen_backbone_digest']!=exp.initial_digest
                or report['backbone_digest']!=exp.initial_digest):
            raise ValueError('Selected head payload does not bind this backbone')
        heads[name].load_state_dict(payload['model'],strict=True)
        heads[name].eval()
    started=time.perf_counter()
    spec=settings['data']
    root=Path(a.read(spec['manifest'])['root']).resolve()
    if data_root.resolve()!=root:
        raise ValueError('TEST must use the same dataset root')
    manifest=build_manifest(root,seed=spec['seed'],recent_frames=64,local_frames=16,
                            max_prefix=0,splits=('test',))
    data=IdentifiedDataset(WindowDataset(manifest,'test',task='ef',allow_test=True,
                recent_frames=64,local_frames=16,max_prefix=0,prefix=0,seed=spec['seed']), 'test')
    if {r['patient'] for r in data.dataset.records} & (set(exp.metadata['train_ids']) | set(exp.metadata['val_ids'])):
        raise ValueError('TEST patient overlap')
    beat=Heartbeat(dest,settings['runtime']['heartbeat_seconds'])
    batch=dict(micro_batch=8,num_workers=0)
    rows={name:[] for name in ARMS}
    with StageDisplay(dest,'shared EF TEST'),torch.no_grad():
        for samples in loader(data,settings,batch,training=False):
            native=exp.features(samples)
            values={'C1':spatial_mean(native),'C2':native.half()}
            with amp_context(settings,device):
                for name in ARMS:
                    pred=exp.forward_head(heads[name],values[name])
                    rows[name].extend(exp._rows(pred,samples))
            beat(phase='validation',completed=len(rows['C1']),total=len(data))
        matched_observations(*rows.values())
        metrics,patients={},{}
        artifacts=[]
        for name in ARMS:
            metrics[name],patients[name]=summarize(rows[name],'ef')
            for label,value in [('observations',rows[name]),('patients',patients[name])]:
                artifact=f'{name}_{label}'
                write_json(dest/(artifact+'.json'),value)
                artifacts.append(artifact+'.json')
            csv_rows(dest/f'{name}_patients.csv',patients[name])
        result=dict(metrics=metrics,paired=paired(patients['C1'],patients['C2']),weights=weights,
                    split='TEST',manifest_sha256=manifest['manifest_sha256'],shared_cache_bytes=0,
                    wall_seconds=time.perf_counter()-started,selection='VAL-locked; no TEST tuning',
                    ssl_data_exposure=exp.metadata['ssl_data_exposure'])
        write_json(dest/'summary.json',result)
        mark_done(dest,artifacts+['summary.json'])
        beat(force=True,phase='done')
    return result


def conclusion(val,test):
    gain=-val['paired']['C2_minus_C1']['candidate_minus_reference']
    gain_test=-test['paired']['candidate_minus_reference']
    if gain>=MIN_EFFECT_PP:
        note='VAL supports a frozen spatial readout benefit; TEST direction agrees.' if gain_test>0 else (
             'VAL supports a frozen spatial readout benefit, but TEST does not corroborate it.')
    elif abs(gain)<MIN_EFFECT_PP and all(
            -val['paired'][f'{name}_minus_C0']['candidate_minus_reference']>=MIN_EFFECT_PP for name in ARMS):
        note='Both heads improve over C0; the shared readout structure/capacity is likelier than spatial retention to explain the gains.'
    else:
        note='This comparison does not show a material benefit of retaining spatial tokens in the frozen readout. It does not rule out useful spatial information.'
    return dict(val_spatial_gate_pp=MIN_EFFECT_PP,val_gain_c2_pp=gain,test_gain_c2_pp=gain_test,
                val_supports_considering_ft=gain>=MIN_EFFECT_PP,interpretation=note,
                limits='Single trained seed. Frozen readout only; no claim of better fine-tuning or pretraining. TEST is descriptive only.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','smoke','run'))
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-fs/outputs_ef_spatial_readout'))
    parser.add_argument('--cache-root',type=Path,default=Path('/root/autodl-tmp/outputs_ef_spatial_readout_cache'))
    parser.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    parser.add_argument('--tag',default='ar400_spatial_ef_20261116')
    parser.add_argument('--a2-root',type=Path,default=Path('/root/autodl-fs/outputs_cr_stage_a2'))
    parser.add_argument('--a2-tag',default='stage_a2_recurrent_20261114')
    args=parser.parse_args()
    safe_name(args.tag)
    safe_name(args.a2_tag)
    cfg,c0,source=plan(args)
    if args.action=='plan':
        from research_pipeline.spatial_ef import BATCH_BYTES
        inputs=a.verify_inputs(c0)['inputs']
        count=len(inputs['train_ids'])+len(inputs['val_ids'])
        path=args.cache_root if args.cache_root.exists() else args.cache_root.parent
        print(json.dumps(dict(config=cfg,reference=str(c0),cases=count,
              estimated_cache_gib=count*BATCH_BYTES/1024**3,cache_disk_free_gib=shutil.disk_usage(path).free/1024**3),indent=2))
        return
    if args.action=='smoke':
        execute(smoke_plan(cfg))
        return
    control=args.root/'control'/args.tag
    with run_lock(control):
        a.lock_selection(control/'protocol.json',dict(config=cfg,source_sha256=file_hash(source),
                         c0_summary_sha256=file_hash(c0/'summary.json'),code=code_identity([__file__])))
        if verify_done(control):
            print(control/'comparison.json')
            return
        execute(cfg)
        val=val_comparison(cfg,c0,source)
        write_json(control/'VAL.json',val)
        test=test_pair(cfg,source,args.data_root,control)
        report=dict(val=val,test=test,decision=conclusion(val,test))
        write_json(control/'comparison.json',report)
        mark_done(control,['protocol.json','VAL.json','TEST_LOCK.json','test/DONE.json','comparison.json'])
        print(json.dumps(report['decision'],indent=2),flush=True)


if __name__=='__main__':
    main()
