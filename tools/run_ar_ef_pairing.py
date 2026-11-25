"""E0 inference aggregation, then optional matched head-only A-R EF comparisons."""

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import code_identity, file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import configure, execute
from research_pipeline.config import safe_name, stage_config, validate
from tools import run_patch8_stage_a as stage_a
from tools import run_spatial_ef_readout as frozen
from tools.evaluate_patch8_test import locked_checkpoint


def uniform_starts(frames):
    last = frames-64
    if last < 0:
        raise ValueError('E0 does not pad cases shorter than W64')
    return sorted({(i*last+2)//4 for i in range(5)})


def endpoint_group(case, start, rasterize):
    if len(case['traces']) != 2:
        return 'uncovered'
    a,b = sorted(case['traces'], key=lambda t:t['frame'])
    fa,fb = a['frame'],b['frame']
    if not start <= fa < fb < start+64:
        return 'uncovered'
    if rasterize(a).sum() <= rasterize(b).sum():
        return 'uncovered'
    return 'same' if (fa-start)//16==(fb-start)//16 else 'cross'


def plan(args):
    cfg,c0,source = frozen.plan(args)
    request = stage_a.read(c0/'request.json')
    prior = request['config']['stage']
    ft_dir,summary,ckpt,ft_req = locked_checkpoint(args.stage_a_root,args.stage_a_tag,'ef_l1_ft')
    ft_protocol = stage_a.verify_inputs(ft_dir)
    c0_protocol = stage_a.verify_inputs(c0)
    data = cfg['data']
    manifest = stage_a.read(data['manifest'])
    if (summary['selected_epoch'] != 21 or summary.get('smoke') or summary.get('synthetic')
        or file_hash(source) != ft_protocol['source'] or file_hash(ft_req['source']) != ft_protocol['source']
        or Path(manifest['root']).resolve() != args.data_root.resolve() or 'test' in manifest
        or cfg['model'] != ft_req['config']['model']
        or any(ft_protocol['inputs'][k] != c0_protocol['inputs'][k] for k in
               ('train_records_sha256','val_records_sha256','data_files','loss'))
        or any(ft_req['config']['data'].get(k)!=data.get(k) for k in
               ('manifest','seed','local_frames','recent_frames','max_prefix','prefix','limit_train','limit_val'))):
        raise ValueError('Locked A-R epoch21 FT and SSL400 frozen reference must share the exact cohort/source')
    arms = ('I','J','R','P') + (('I_stride4',) if args.with_stride4 else ())
    cfg.update(adapter='research_pipeline.ef_pairing:build', stages=[
        dict(copy.deepcopy(prior),id=arm,source=source,ef_pairing_arm=arm,
             matched_inputs={k:c0_protocol['inputs'][k] for k in
                             ('train_records_sha256','val_records_sha256','data_files','loss')}) for arm in arms])
    cfg['runtime'].update(cache_gb=args.cache_gb,min_free_gb=args.reserve_gb,micro_batch=4,
                          worker_candidates=[0,2], checkpoint_candidates=[False],precision='fp32')
    cfg['evaluation']['cache_frozen'] = True
    if args.action == 'smoke':
        cfg = frozen.smoke_plan(cfg)
        cfg['runtime']['min_free_gb'] = args.reserve_gb
        if args.cpu:
            cfg['runtime'].update(device='cpu',precision='fp32',threads=2)
    return validate(cfg), source, (ft_dir, summary, ckpt, ft_req), manifest


def dataset(manifest, split, seed, limit=None, records=None):
    from utils.final_temporal_data import WindowDataset
    return WindowDataset(manifest,split,task='ef',allow_test=split=='test', seed=seed,
                         recent_frames=64,local_frames=16,max_prefix=0,prefix=0,
                         records=records,limit=limit)


def cohort(data):
    import numpy as np
    return {r['patient']: float(np.float32(data.cases[r['patient']]['ef'])) for r in data.records}


def reduce_results(rows, strata, comparisons):
    from research_pipeline.metrics import summarize
    patients, report = {}, dict(metrics={}, pairs={})
    for arm, arm_rows in rows.items():
        report['metrics'][arm],patients[arm] = summarize(arm_rows,'ef')
        report['metrics'][arm]['strata'] = {}
        for key,ids in strata.items():
            selected = [r for r in patients[arm] if r['patient'] in ids]
            report['metrics'][arm]['strata'][key] = summarize(selected,'ef')[0] if selected else {'patients':0}
    for a,b in comparisons:
        pair = frozen.paired(patients[a],patients[b])
        pair['strata'] = {}
        for key,ids in strata.items():
            first,second = [[r for r in patients[arm] if r['patient'] in ids] for arm in (a,b)]
            pair['strata'][key] = frozen.paired(first,second) if first else {'patients':0}
        report['pairs'][f'{b}_minus_{a}'] = pair
    return report,patients


def population_strata(data):
    from utils.final_temporal_data import _trace_mask
    groups = {key:set() for key in ('EF_lt40','EF_40_lt50','EF_ge50','same','cross','uncovered')}
    for i in range(len(data)):
        record = data.get_record(i)
        y = record['ef']
        key = 'EF_lt40' if y<40 else 'EF_40_lt50' if y<50 else 'EF_ge50'
        groups[key].add(record['patient'])
        groups[endpoint_group(record,record['recent_start'],_trace_mask)].add(record['patient'])
    return groups


def save_predictions(dest, rows, patients):
    from research_pipeline.engine import csv_rows
    files = []
    for arm in rows:
        for kind,value in (('windows',rows[arm]),('patients',patients[arm])):
            name = f'{arm}_{kind}.csv'
            csv_rows(dest/name,value)
            files.append(name)
    return files


def e0(data, ft, cfg, dest):
    import numpy as np
    import torch
    from research_pipeline.echo import IdentifiedDataset
    from research_pipeline.engine import Heartbeat, amp_context, build_experiment
    from research_pipeline.runtime import loader
    from tools.watch_pipeline_progress import StageDisplay
    if verify_done(dest):
        return stage_a.read(dest/'summary.json')
    ft_dir,summary,ckpt,request = ft
    settings = copy.deepcopy(request['config'])
    if cfg.get('smoke'):
        settings['runtime'].update(device=cfg['runtime']['device'],precision=cfg['runtime']['precision'])
    configure(settings)
    device = torch.device(settings['runtime']['device'])
    exp = build_experiment(settings,request['source'],device)
    payload = torch.load(ckpt,map_location='cpu',weights_only=False)
    if payload['epoch']!=21 or file_hash(payload['source'])!=file_hash(request['source']):
        raise ValueError('E0 payload is not the locked A-R EF endpoint')
    exp.model.load_state_dict(payload['model'],strict=True)
    exp.model.requires_grad_(False).eval()
    del payload
    split = 'test' if 'test' in data.manifest else 'val'
    if split=='val':
        previous = {r['patient']:r['target'] for r in stage_a.read(ft_dir/'observations.json')}
        actual = cohort(data)
        if any(previous.get(k)!=v for k,v in actual.items()) or (not cfg.get('smoke') and actual!=previous):
            raise ValueError('E0 VAL is not the original FT evaluation cohort')
    else:
        if set(cohort(data)) & (set(exp.metadata['train_ids'])|set(exp.metadata['val_ids'])):
            raise ValueError('E0 TEST overlaps supervised TRAIN/VAL')
    runtime = stage_a.read(ft_dir/'runtime.json')['selected']
    runtime = dict(runtime,num_workers=0) if cfg.get('smoke') else runtime
    beat = Heartbeat(dest,10)
    rows,cost = {},{}
    strata = population_strata(data)
    for arm in ('single','five'):
        records = []
        for i in range(len(data)):
            record = data.get_record(i)
            starts = [record['recent_start']] if arm=='single' else uniform_starts(record['frames'])
            records.extend(dict(patient=record['patient'],recent_start=start,H=0) for start in starts)
        windows = IdentifiedDataset(dataset(data.manifest,split,data.seed,records=records),split)
        rows[arm] = []
        started = time.perf_counter()
        with StageDisplay(dest,f'E0 {split}/{arm}'),torch.inference_mode():
            for samples in loader(windows,settings,runtime,training=False):
                with amp_context(settings,device):
                    pred = exp.model(samples).float().tolist()
                rows[arm].extend(dict(patient=s['patient'], target=float(s['target']),prediction=p*100,
                                      recent_start=int(s['recent_start']),fps=float(s['fps']),
                                      covered_ms=63000/float(s['fps']))
                                 for s,p in zip(samples,pred))
                beat(phase='validation',arm=arm,completed=len(rows[arm]),total=len(windows))
        elapsed = time.perf_counter()-started
        cost[arm] = dict(wall_seconds=elapsed,windows=len(records),clip_calls=4*len(records),
                         frames_encoded=64*len(records),windows_per_second=len(records)/elapsed,
                         elapsed_ms_per_case=elapsed*1000/len(data),runtime=runtime)
    result,patients = reduce_results(rows,strata,[('single','five')])
    del exp
    by_case = {}
    for row in rows['five']:
        case = by_case.setdefault(row['patient'],[])
        case.append(row['prediction'])
    for row in patients['five']:
        values = by_case[row['patient']]
        row.update(actual_windows=len(values),start_sd_pp=float(np.std(values)))
    result.update(split=split,cost=cost,epoch=21,checkpoint_sha256=file_hash(ckpt),
                  distinct_window_histogram={str(n):sum(len(v)==n for v in by_case.values()) for n in range(1,6)},
                  mean_start_sd_pp=float(np.mean([r['start_sd_pp'] for r in patients['five']])),
                  endpoint_strata_scope='both arms grouped by the original single window',
                  timing_scope='batched end-to-end wall time including loading; not single-exam clinical latency')
    artifacts = save_predictions(dest,rows,patients)
    write_json(dest/'summary.json',result)
    mark_done(dest,['summary.json',*artifacts])
    beat(force=True,phase='done')
    return result


def compare_heads(cfg, source, data, dest, *, test=False):
    import torch
    from research_pipeline.cache import FeatureDataset
    from research_pipeline.ef_pairing import VIEWS, describe
    from research_pipeline.engine import Heartbeat, amp_context, build_experiment
    from research_pipeline.echo import IdentifiedDataset
    from research_pipeline.runtime import loader
    from tools.watch_pipeline_progress import StageDisplay
    if verify_done(dest):
        return stage_a.read(dest/'summary.json')
    arms = tuple(stage['id'] for stage in cfg['stages'])
    requested = [stage_config(cfg,s) for s in cfg['stages']]
    ends = {arm:frozen.endpoint(cfg,arm,source) for arm in arms}
    if len({stage_a.read(path/'frozen_integrity.json')['backbone_digest'] for path,_,_ in ends.values()}) != 1:
        raise ValueError('Frozen arms used different backbone states')
    if (ends['I'][1]['cache']['directory']!=ends['J'][1]['cache']['directory'] or
            ends['R'][1]['cache']['directory']!=ends['P'][1]['cache']['directory'] or
            ends['I'][1]['cache']['directory']==ends['R'][1]['cache']['directory']):
        raise ValueError('Expected shared cache within, and different cache across, encoder views')
    pairs = [('I','J'),('J','R'),('R','P')]
    if 'I_stride4' in arms:
        pairs.append(('I','I_stride4'))
    rows = {arm:[] for arm in arms}
    if not test:
        for arm,(path,_,_) in ends.items():
            rows[arm] = stage_a.read(path/'observations.json')
    settings = requested[0]
    configure(settings)
    device = torch.device(settings['runtime']['device'])
    heads = {}
    backbone = None
    for settings,arm in zip(requested,arms):
        exp = build_experiment(settings,source,torch.device('cpu'))
        payload = torch.load(ends[arm][2],map_location='cpu',weights_only=True)
        if (not payload.get('head_only') or payload.get('arm')!=arm or
            payload['epoch']!=ends[arm][1]['selected_epoch'] or payload['source']!=source):
            raise ValueError('Readout is not the VAL-locked arm or correct SSL source')
        exp.model.load_state_dict(payload['model'],strict=True)
        heads[arm] = exp.model.to(device).requires_grad_(False).eval()
        if arm=='I' and test:
            if set(cohort(data)) & (set(exp.metadata['train_ids'])|set(exp.metadata['val_ids'])):
                raise ValueError('Head TEST overlaps supervised TRAIN/VAL')
            backbone = exp.backbone.to(device)
        del exp,payload
    beat = Heartbeat(dest,10)
    runtime = dict(micro_batch=min(4,cfg['training']['effective_batch']),num_workers=0)
    strata = population_strata(data)
    started = time.perf_counter()
    if test:
        feed = IdentifiedDataset(data,'test')
        groups = {view:[arm for arm in arms if VIEWS[arm]==view] for view in dict.fromkeys(VIEWS[a] for a in arms)}
    else:
        feed = FeatureDataset(Path(ends['P'][1]['cache']['directory'])/'val')
        rows['P_perturbed'] = []
    with StageDisplay(dest,'head TEST' if test else 'VAL pairing stress test'),torch.inference_mode():
        for samples in loader(feed,settings,runtime,training=False):
            fps = torch.tensor([s['fps'] for s in samples],device=device)
            if test:
                video = torch.stack([s['video'] for s in samples]).to(device)
                predictions = {}
                for view,view_arms in groups.items():
                    features = describe(backbone,video,view)
                    with amp_context(settings,device):
                        predictions.update({arm:heads[arm](features,fps) for arm in view_arms})
            else:
                features = torch.stack([s['features'] for s in samples]).to(device)
                with amp_context(settings,device):
                    predictions = dict(P_perturbed=heads['P'](features,fps,perturb=True))
            for arm,pred in predictions.items():
                rows[arm].extend(dict(patient=s['patient'],target=float(s['target']),prediction=p*100,
                                      recent_start=int(s['recent_start']),fps=float(s['fps']))
                                 for s,p in zip(samples,pred.float().tolist()))
            beat(phase='validation',completed=len(rows[arms[0] if test else 'P_perturbed']),total=len(feed))
    observations = {r['patient']:(r['recent_start'],r['ef']) for r in (data.get_record(i) for i in range(len(data)))}
    actual_targets = cohort(data)
    if any(len(items)!=len(data) or {r['patient']:(r['recent_start'],r['target']) for r in items} !=
           {k:(v[0],actual_targets[k]) for k,v in observations.items()} for items in rows.values()):
        raise ValueError('Head arms do not match precisely on patients, labels and windows')
    if not test:
        pairs.append(('P','P_perturbed'))
    result,patients = reduce_results(rows,strata,pairs)
    result.update(split='test' if test else 'val',wall_seconds=time.perf_counter()-started,
                  weights={arm:dict(sha256=file_hash(path),epoch=summary['selected_epoch'],
                                    parameters=summary['trainable_parameters'])
                           for arm,(_,summary,path) in ends.items()},
                  fit_cost={arm:{k:summary[k] for k in ('cache','wall_seconds','completed_train_seconds',
                                                      'completed_validation_seconds')}
                            for arm,(_,summary,_) in ends.items()},
                  endpoint_strata_scope='Two within-window area-ordered annotations; not a proof of same-beat phase',
                  supervision='TRAIN EF only; no annotations used as model input or pair targets')
    files=save_predictions(dest,rows,patients)
    write_json(dest/'summary.json',result)
    mark_done(dest,['summary.json',*files])
    beat(force=True,phase='done')
    return result


def run(args):
    from utils.final_temporal_data import build_manifest
    cfg,source,ft,manifest = plan(args)
    views = 2+int(args.with_stride4)
    num = sum(sum(r['ef'] is not None and r['frames']>=64 for r in manifest[split]) for split in ('train','val'))
    gib = views*num*32*384*4/1024**3
    if args.action=='plan':
        print(json.dumps(dict(config=cfg,head_cache_estimated_gib=gib,ft_checkpoint=str(ft[2]),
                              labels_opened={'TEST':False},auto_ft=False),indent=2))
        return
    control = layout(cfg)['result']/'ef_pairing'
    seed = cfg['data']['seed']
    smoke = args.action=='smoke'
    fit = args.action!='e0'
    with run_lock(control):
        binding = dict(config=cfg,action=args.action,ft_sha256=file_hash(ft[2]),ssl_sha256=file_hash(source),
                       manifest_sha256=file_hash(cfg['data']['manifest']),code=code_identity([__file__]))
        stage_a.lock_selection(control/'protocol.json',binding)
        if verify_done(control):
            print(control/'comparison.json')
            return
        val = dataset(manifest,'val',seed,limit=cfg['data'].get('limit_val') if smoke else None)
        result = dict(val=dict(E0=e0(val,ft,cfg,control/'e0_val')),test=None)
        gc.collect()
        if fit:
            cache = layout(cfg)['cache']
            cache.mkdir(parents=True,exist_ok=True)
            if not smoke and (gib+.05>args.cache_gb or
                    shutil.disk_usage(cache).free+sum(f.stat().st_size for f in cache.rglob('*.npy')) <
                    (gib+.05+args.reserve_gb)*1024**3):
                raise OSError('Insufficient compact-cache budget or disk safety reserve')
            execute(cfg)
            result['val']['heads'] = compare_heads(cfg,source,val,control/'heads_val')
            evidence = result['val']['heads']['metrics']
            gain = evidence['R']['mae_pp']-evidence['P']['mae_pp']
            result['decision_from_val_only'] = dict(P_gain_vs_R_pp=gain,
                eligible_to_consider_matched_ft=gain>=.10,
                skill_limit='Frozen only. No full fine-tuning is started by this entry point.')
        if not smoke:
            lock = dict(ft_epoch=21,ft_sha256=binding['ft_sha256'],ssl_sha256=binding['ssl_sha256'],
                        E0_fixed_policies=['single','five'],heads=result['val'].get('heads',{}).get('weights',{}),
                        val_decision=result.get('decision_from_val_only'))
            stage_a.lock_selection(control/'TEST_LOCK.json',lock)
            test = build_manifest(args.data_root,seed=seed,recent_frames=64,local_frames=16,
                                  max_prefix=0,splits=('test',))
            stage_a.lock_selection(control/'test_manifest_lock.json',dict(sha256=test['manifest_sha256']))
            test_data=dataset(test,'test',seed)
            result['test'] = dict(E0=e0(test_data,ft,cfg,control/'e0_test'))
            if fit:
                result['test']['heads'] = compare_heads(cfg,source,test_data,control/'heads_test',test=True)
        write_json(control/'comparison.json',result)
        artifacts = ['protocol.json','comparison.json','e0_val/DONE.json']
        if fit:
            artifacts += ['heads_val/DONE.json']
        if not smoke:
            artifacts += ['TEST_LOCK.json','test_manifest_lock.json','e0_test/DONE.json']
            if fit:
                artifacts += ['heads_test/DONE.json']
        mark_done(control,artifacts)
        print(control/'comparison.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','smoke','e0','run'])
    p.add_argument('--tag',required=True,type=safe_name,help='Use a separate tag for a standalone E0 run')
    defaults = dict(root='/root/autodl-fs/outputs_ar_ef_pairing',
        cache_root='/root/autodl-tmp/outputs_ar_ef_pairing_cache', data_root='/root/autodl-tmp/datasets/EchoNet-Dynamic',
        stage_a_root='/root/autodl-fs/outputs_cr_stage_a',a2_root='/root/autodl-fs/outputs_cr_stage_a2')
    for key,value in defaults.items():
        p.add_argument('--'+key.replace('_','-'),type=Path,default=Path(value))
    p.add_argument('--stage-a-tag',type=safe_name,default='patch8_stage_a_v1_AR')
    p.add_argument('--a2-tag',type=safe_name,default='stage_a2_recurrent_20261114')
    p.add_argument('--cache-gb',type=float,default=3)
    p.add_argument('--reserve-gb',type=float,default=3)
    p.add_argument('--with-stride4',action='store_true')
    p.add_argument('--cpu',action='store_true',help='CPU smoke only')
    args=p.parse_args()
    if args.cpu and args.action!='smoke':
        p.error('--cpu is smoke-only')
    run(args)


if __name__=='__main__':
    main()
