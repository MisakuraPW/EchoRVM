"""Stage A: two independent fixed-400 schedules, matched FT, explicit TEST selection."""

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import ROOT, file_hash, layout, run_lock, verify_done, write_json
from research_pipeline.cli import execute, source_path
from research_pipeline.config import load, safe_name, stage_config, validate
from tools.evaluate_patch8_test import locked_checkpoint


VARIANTS = {'AR': dict(forward_variant='mid_read', read_after_block=6),
            'AW': dict(forward_variant='content_write', writer_width=64)}
TASK_STAGES = {'seg': 'seg_full_coverage_ft', 'ef': 'ef_l1_ft'}
PROBES = {'seg': 'seg_linear_0400', 'ef': 'ef_linear_0400'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completed(directory):
    if not verify_done(directory):
        raise ValueError(f'Missing verified result: {directory}')
    return read(directory / 'summary.json')


def verify_inputs(directory):
    protocol = read(directory / 'protocol.json')['protocol']
    for path, expected in protocol['inputs']['data_files'].items():
        if file_hash(path) != expected:
            raise ValueError(f'Baseline input changed: {path}')
    return protocol


def baseline(source_root, gap_root, source_tag, coverage_tag, data_root):
    parent = source_root / 'result' / source_tag
    base = load(parent / 'plan.json')
    if Path(base['run']['root']).resolve() != source_root.resolve() or base['run']['tag'] != source_tag:
        raise ValueError('Use the original B8 result layout and absolute manifest paths')
    stages = {s['id']: s for s in base['stages']}
    ssl = stage_config(base, stages['pretrain'])
    recipe = ssl['training']
    expected = dict(epochs=400, schedule_epochs=None, run_until_epoch=None, lr=.0001,
                    weight_decay=.05, effective_batch=64, warmup_epochs=40, patience=None,
                    betas=[.9, .95], lr_schedule='cosine', resume_every=1, save_epochs=[0, 100, 200, 400])
    if any(recipe[k] != value for k, value in expected.items()):
        raise ValueError('B8 must be the independent 400-epoch recipe, not a long-run snapshot')
    if ssl['model']['overrides'] or ssl['model']['config'].get('forward_variant'):
        raise ValueError('B8 must use the unchanged read/write baseline')
    model = ssl['model']['config']
    required = dict(img_size=112, patch_size=8, local_frames=16, clip_count=4, tubelet_size=2,
                    depth=12, memory_mode='global', memory_compression='mean', memory_write_source='fused',
                    memory_read_location='tokens', frame_readout='repeat', reconstruction_recent_frames=64,
                    mask_ratio=.9, research_mask='tube', norm_pix_loss=True, candidate_rank=0,
                    dynamic_orthogonal_weight=0., frequency_conditioned=False, frequency_loss_weight=0.)
    if any(model.get(k) != value for k, value in required.items()):
        raise ValueError('B8 native architecture mismatch')
    if tuple(ssl['data'][k] for k in ('local_frames','recent_frames','max_prefix','prefix')) != (16,64,0,0):
        raise ValueError('B8 data must be L16/W64/H0')
    mae = completed(parent / 'pretrain')
    protocol = verify_inputs(parent / 'pretrain')
    if (mae['epoch'] != 400 or mae['smoke'] or mae['synthetic'] or protocol['training'] !=
            {k:v for k,v in recipe.items() if k != 'run_until_epoch'} or
            mae['ssl_data_exposure'] != 'transductive_all_splits_unlabeled'):
        raise ValueError('B8 completed SSL evidence does not match its plan')
    zero, endpoint = [source_path(base, dict(source=f'@pretrain/epoch_{epoch:04d}')) for epoch in (0,400)]
    endpoint_hash = file_hash(endpoint)
    ft = {'seg': locked_checkpoint(gap_root, coverage_tag, TASK_STAGES['seg']),
          'ef': locked_checkpoint(source_root, source_tag, TASK_STAGES['ef'])}
    ef_probe = parent / PROBES['ef']
    completed(ef_probe)
    inputs = {'pretrain': protocol['inputs']}
    for name,directory in [(PROBES['ef'],ef_probe), *((v[0].name,v[0]) for v in ft.values())]:
        task_protocol = verify_inputs(directory)
        if task_protocol['source'] != endpoint_hash:
            raise ValueError(f'B8 endpoint lineage mismatch: {directory}')
        inputs[name] = task_protocol['inputs']
    for directory,_,_,request in ft.values():
        protocol = read(directory/'protocol.json')['protocol']
        cfg = request['config']
        if (file_hash(request['source']) != endpoint_hash or
                any(cfg[k] != protocol[k] for k in ('data','evaluation')) or
                {k:v for k,v in cfg['training'].items() if k!='run_until_epoch'} != protocol['training'] or
                cfg['runtime']['precision'] != protocol['precision']):
            raise ValueError(f'B8 FT request disagrees with the sealed protocol: {directory}')
    for suffix in ('_val_full','_test'):
        path = gap_root/'result'/(coverage_tag+suffix)/TASK_STAGES['seg']
        if completed(path)['checkpoint_sha256'] != file_hash(ft['seg'][2]):
            raise ValueError(f'B8 full segmentation evaluation has the wrong checkpoint: {path}')
    seg_cfg = ft['seg'][3]['config']
    if seg_cfg['adapter'] != 'research_pipeline.coverage_repair:build':
        raise ValueError('B8 segmentation FT must be the coverage-repaired run')
    counts = verify_inputs(ft['seg'][0])['inputs']['coverage']
    if (counts['unique_source_frames'], counts['draws_per_check']) != (14920,8108):
        raise ValueError('Unexpected B8 coverage/optimizer-update population')
    full = read(seg_cfg['data']['coverage_manifest'])
    if full.get('minimum_frames') != 1 or Path(full['root']).resolve() != data_root.resolve():
        raise ValueError('Data root must match the verified full-supervision manifest')
    return dict(base=base, stages=stages, zero=zero, endpoint=endpoint, ft=ft, ef_probe=ef_probe, inputs=inputs,
                lock=dict(epoch0=file_hash(zero), epoch400=endpoint_hash,
                          source_tag=source_tag, coverage_tag=coverage_tag))


def task_stage(request, source, name=None):
    cfg = request['config']
    stage = copy.deepcopy(cfg['stage'])
    stage.update(source=source, **{k:copy.deepcopy(cfg[k]) for k in
                                  ('data','training','runtime','evaluation')})
    if name:
        stage['id'] = name
    return stage


def build_plan(ref, root, tag, arm):
    cfg = copy.deepcopy(ref['base'])
    cfg.update(run=dict(root=str(root), tag=safe_name(tag+'_'+arm)),
               adapter='research_pipeline.stage_a:build', stage_a_baseline=ref['lock'])
    cfg['stages'] = []
    if arm in VARIANTS:
        cfg['stages'].append(dict(copy.deepcopy(ref['stages']['pretrain']), source=ref['zero'],
                                  model=dict(overrides=VARIANTS[arm])))
        endpoint = '@pretrain/epoch_0400'
        cfg['stages'] += [task_stage(ref['ft'][task][3], endpoint) for task in ('ef','seg')]
        cfg['stages'].append(dict(copy.deepcopy(ref['stages'][PROBES['ef']]), source=endpoint))
    elif arm == 'B8':
        endpoint = ref['endpoint']
    else:
        raise ValueError('Unknown Stage A arm')
    coverage = ref['ft']['seg'][3]['config']['data']
    probe = dict(copy.deepcopy(ref['stages'][PROBES['seg']]), source=endpoint)
    probe['data'] = dict(coverage, augmentation={'enabled':False})
    cfg['stages'].append(probe)
    for stage in cfg['stages']:
        source_inputs = ref.get('inputs',{}).get(stage['id'])
        if source_inputs:
            stage['matched_inputs'] = {key:source_inputs[key] for key in
                                       ('train_records_sha256','val_records_sha256','data_files','head','loss')}
    heads = {task:ref['ft'][task][2] if arm=='B8' else f'@{stage}/best'
             for task,stage in TASK_STAGES.items()}
    requests = {task:str((ref['ft'][task][0] if arm=='B8' else layout(cfg,stage)['result'])/'request.json')
                for task,stage in TASK_STAGES.items()}
    cfg['stages'].append(dict(id='diagnostics', kind='forward_diagnostics', source=endpoint,
                              audit_scope='patch8_stage_a', panel_cases=32, gradient_cases=1,
                              heads=heads, head_requests=requests,
                              budget_stages={task:str(Path(req).parent) for task,req in requests.items()},
                              budget_ssl=str(Path(ref['ef_probe']).parent/'pretrain' if arm=='B8'
                                             else layout(cfg,'pretrain')['result'])))
    return validate(cfg)


def summary_pair(reference, candidate, task):
    from research_pipeline.metrics import paired_bootstrap

    a,b = (completed(path) for path in (reference,candidate))
    pa,pb = (read(path/'protocol.json')['protocol'] for path in (reference,candidate))
    for key in ('training','evaluation','data','precision'):
        if pa[key] != pb[key]:
            raise ValueError(f'Unmatched {key}: {reference} vs {candidate}')
    for key in ('train_records_sha256','val_records_sha256','head','loss','data_files'):
        if pa['inputs'][key] != pb['inputs'][key]:
            raise ValueError(f'Unmatched {key}: {reference} vs {candidate}')
    metric, field = ('mae_pp','absolute_error') if task=='ef' else ('dice_mean','dice')
    return dict(reference=a['metrics'], candidate=b['metrics'], primary=metric,
                paired=paired_bootstrap(read(reference/'patients.json'),read(candidate/'patients.json'),field),
                reference_wall_seconds=a['wall_seconds'], candidate_wall_seconds=b['wall_seconds'])


def phase_metrics(directory, manifest, split):
    from research_pipeline.metrics import stratify
    from utils.final_temporal_data import _trace_mask

    rows = read(directory/'observations.json')
    population = {str(r['patient']) for r in rows}
    phases = {}
    for case in manifest[split]:
        if case['patient'] not in population or len(case['traces']) != 2:
            continue
        events = sorted((int(_trace_mask(t).sum()),t['frame']) for t in case['traces'])
        if events[0][0] != events[1][0]:
            phases.update({(case['patient'],str(frame)):phase for (_,frame),phase in zip(events,('ES','ED'))})
    for r in rows:
        r['phase'] = phases.get((str(r['patient']),str(r['source_frame'])))
    return dict(label='larger/smaller traced area ED/ES proxy; ties and non-pairs excluded',
                unknown_observations=sum(r['phase'] is None for r in rows),
                metrics=stratify(rows,'seg',fields=('phase',)).get('phase',{}))


def match_observations(reference, candidate, task):
    fields = ('patient','sample_id','target') if task=='ef' else (
        'patient','source_frame','target_position','class_id','target_pixels')
    def keys(directory):
        return sorted(tuple(row[k] for k in fields) for row in read(directory/'observations.json'))
    if keys(reference) != keys(candidate):
        raise ValueError(f'Mismatched {task} evaluation population: {reference} vs {candidate}')


def cost(directory):
    summary = completed(directory)
    with (directory/'metrics.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return dict(wall_seconds=summary['wall_seconds'], train_seconds=summary['completed_train_seconds'],
                updates=int(rows[-1]['successful_updates']),
                frames=sum(int(r.get('observed_frames') or 0) for r in rows),
                peak_allocated_bytes=max(int(r.get('train_peak_allocated_bytes') or 0) for r in rows),
                parameters=summary['model_parameters'])


def compare(ref, plans, control, gap_root, coverage_tag):
    full = read(ref['ft']['seg'][3]['config']['data']['coverage_manifest'])
    result = dict(baseline=ref['lock'], decision='VAL only, human selection; no automatic promotion', arms={}, diagnostics={},
                  baseline_cost={**{task:cost(item[0]) for task,item in ref['ft'].items()},
                                 'pretrain':cost(ref['ef_probe'].parent/'pretrain')})
    for arm,cfg in plans.items():
        directory = layout(cfg,'diagnostics')['result']
        if verify_done(directory):
            result['diagnostics'][arm] = completed(directory)
    panel_hashes = {value['panel_sha256'] for value in result['diagnostics'].values()}
    if len(panel_hashes)>1:
        raise ValueError('Diagnostic case panels are not matched')
    for arm in VARIANTS:
        cfg = plans[arm]
        out = layout(cfg)['result']
        if not all(verify_done(out/s) for s in TASK_STAGES.values()):
            continue
        comparisons = {task:summary_pair(ref['ft'][task][0],out/stage,task) for task,stage in TASK_STAGES.items()}
        comparisons['seg']['common_val_phases'] = {name:phase_metrics(path,full,'val') for name,path in
                                                   [('B8',ref['ft']['seg'][0]),(arm,out/TASK_STAGES['seg'])]}
        full_dirs = {'B8':gap_root/'result'/(coverage_tag+'_val_full')/TASK_STAGES['seg'],
                     arm:out.parent/(out.name+'_val_full')/TASK_STAGES['seg']}
        if all(verify_done(path) for path in full_dirs.values()):
            match_observations(full_dirs['B8'],full_dirs[arm],'seg')
            for name,path in full_dirs.items():
                parent = ref['ft']['seg'][0] if name=='B8' else out/TASK_STAGES['seg']
                ft = completed(parent)
                if completed(path)['checkpoint_sha256'] != ft['checkpoints'][ft['best_file']]['sha256']:
                    raise ValueError('Full VAL result does not bind the selected FT checkpoint')
        comparisons['seg']['full_val'] = {name:dict(metrics=completed(path)['metrics'],
                                                   phases=phase_metrics(path,full,'val'))
                                          for name,path in full_dirs.items() if verify_done(path)}
        probes = {}
        for task,stage in PROBES.items():
            base = ref['ef_probe'] if task=='ef' else layout(plans['B8'],stage)['result']
            if verify_done(base) and verify_done(out/stage):
                probes[task] = summary_pair(base,out/stage,task)
        result['arms'][arm] = dict(ft=comparisons, probes=probes,
                                   cost={stage:cost(out/stage) for stage in ('pretrain',*TASK_STAGES.values())})
    write_json(control/'matched_val.json',result)
    return result


def evaluate(cfg, stages, data_root, split='test'):
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/evaluate_patch8_test.py'),
                    '--root',cfg['run']['root'],'--tag',cfg['run']['tag'],'--data-root',str(data_root),
                    '--split',split,'--stages',*stages],cwd=ROOT,check=True)


def lock_selection(path, selection):
    if path.exists() and read(path) != selection:
        raise ValueError(f'Selection already locked: {path}')
    write_json(path,selection)


def smoke_config(config, cpu=False):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'plan.json'
        write_json(path,config)
        cfg = load(path,smoke=True)
    # Pinning real epoch 100/200/400 has no meaning in a one-epoch software check.
    for stage in cfg['stages']:
        if stage['kind']=='mae':
            stage['training']['save_epochs'] = [0,1]
        if stage.get('head_requests'):
            stage['head_requests'] = {task:str(layout(cfg,name)['result']/'request.json')
                                      for task,name in TASK_STAGES.items()}
        stage['runtime'] = dict(stage.get('runtime',{}),autotune=False,micro_batch=1,worker_candidates=[0])
        if cpu:
            stage['runtime'].update(device='cpu',precision='fp32')
    if cpu:
        cfg['runtime'].update(device='cpu',precision='fp32')
    return validate(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','smoke','run','compare','test','confirm'])
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/outputs_cr_stage_a'))
    parser.add_argument('--tag', default='patch8_stage_a_v1')
    parser.add_argument('--source-root', type=Path, default=Path('/root/autodl-fs/outputs_cr_spatial'))
    parser.add_argument('--gap-root', type=Path, default=Path('/root/autodl-fs/outputs_cr_gap'))
    parser.add_argument('--source-tag', default='cr_patch8_w64_400_v1')
    parser.add_argument('--coverage-tag', default='seg_gap_stage1_v1_coverage')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    parser.add_argument('--candidate', choices=['AR','AW','B8'])
    parser.add_argument('--cpu-smoke', action='store_true')
    args = parser.parse_args()
    if args.cpu_smoke and args.action!='smoke':
        parser.error('--cpu-smoke only applies to software smoke')
    if args.action in {'test','confirm'} and not args.candidate:
        parser.error('Explicit VAL-selected --candidate is required')
    if args.candidate=='B8' and args.action!='test':
        parser.error('B8 is reused; --candidate B8 is only for baseline TEST')
    for name in (args.source_tag,args.coverage_tag,args.tag):
        safe_name(name)
    ref = baseline(args.source_root,args.gap_root,args.source_tag,args.coverage_tag,args.data_root)
    plans = {arm:build_plan(ref,args.root,args.tag,arm) for arm in ('B8',*VARIANTS)}
    if args.action=='plan':
        print(json.dumps(plans,indent=2))
        return
    control = args.root/'control'/args.tag
    arms = [args.candidate] if args.candidate else list(VARIANTS)
    with run_lock(control):
        if args.action=='smoke':
            for arm in arms:
                execute(smoke_config(plans[arm],args.cpu_smoke))
            return
        if args.action=='run':
            execute(plans['B8'])
            for arm in arms:
                cfg = plans[arm]
                execute(cfg)
                evaluate(cfg,[TASK_STAGES['seg']],args.data_root,'val_full')
                compare(ref,plans,control,args.gap_root,args.coverage_tag)
            return
        result = compare(ref,plans,control,args.gap_root,args.coverage_tag)
        if args.action=='compare':
            return
        arm = args.candidate
        if arm!='B8' and (arm not in result['arms'] or arm not in result['arms'][arm]['ft']['seg']['full_val']):
            raise ValueError('Both FT tasks and complete VAL must finish before selection')
        if args.action=='confirm':
            lock_selection(control/'confirmation_lock.json',dict(candidate=arm,seed=43))
            for name in ('B8',arm):
                cfg = copy.deepcopy(plans[name])
                source = ref['endpoint'] if name=='B8' else source_path(cfg,dict(source='@pretrain/epoch_0400'))
                cfg['run']['tag'] += '_confirm43'
                cfg['stages'] = [task_stage(ref['ft'][task][3],source) for task in ('ef','seg')]
                for stage in cfg['stages']:
                    stage['training']['seed'] = 43
                execute(validate(cfg))
            a,b = [args.root/'result'/f'{args.tag}_{name}_confirm43' for name in ('B8',arm)]
            write_json(control/'confirmation_val.json',{task:summary_pair(a/name,b/name,task)
                                                         for task,name in TASK_STAGES.items()})
            return
        cfg = plans[arm]
        selected = ref['ft'] if arm=='B8' else {
            task:locked_checkpoint(args.root,cfg['run']['tag'],name) for task,name in TASK_STAGES.items()}
        locked = {name:{task:file_hash(item[2]) for task,item in entries.items()}
                  for name,entries in [('B8',ref['ft']),(arm,selected)]}
        base_seg = ref['ft']['seg'][0]
        base_seg_test = base_seg.parent.with_name(base_seg.parent.name+'_test')/base_seg.name
        if completed(base_seg_test)['checkpoint_sha256'] != locked['B8']['seg']:
            raise ValueError('Existing B8 full segmentation TEST does not bind the selected checkpoint')
        lock_selection(control/('b8_test_lock.json' if arm=='B8' else 'candidate_test_lock.json'),
                       dict(candidate=arm, weights=locked, selection='explicit VAL decision, no TEST tuning'))
        evaluate(ref['base'],[TASK_STAGES['ef']],args.data_root)
        if arm!='B8':
            evaluate(cfg,list(TASK_STAGES.values()),args.data_root)
        from utils.final_temporal_data import build_manifest
        manifest = build_manifest(args.data_root, seed=ref['base']['data']['seed'],
                                  max_prefix=0, splits=('test',), minimum_frames=1)
        dirs = {'B8': {task:item[0].parent.with_name(item[0].parent.name+'_test')/item[0].name
                       for task,item in ref['ft'].items()}}
        if arm!='B8':
            dirs[arm] = {task:layout(cfg)['result'].with_name(cfg['run']['tag']+'_test')/name
                         for task,name in TASK_STAGES.items()}
            for task in TASK_STAGES:
                match_observations(dirs['B8'][task],dirs[arm][task],task)
        report = {}
        for name,paths in dirs.items():
            report[name] = {task:completed(path) for task,path in paths.items()}
            for task in paths:
                if report[name][task]['checkpoint_sha256'] != locked[name][task]:
                    raise ValueError('TEST artifact does not match its VAL lock')
            report[name]['seg_phases'] = phase_metrics(paths['seg'],manifest,'test')
        write_json(control/f'{arm}_test.json',report)


if __name__=='__main__':
    main()
