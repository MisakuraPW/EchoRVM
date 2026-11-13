"""A-R follow-up: matched EF readout screen and two independent 400-epoch runs."""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import execute, source_path
from research_pipeline.config import safe_name, stage_config, validate
from tools import run_patch8_stage_a as a


VARIANTS = {
    'RT': dict(forward_variant='mid_temporal_write', read_after_block=6, temporal_writer_width=96),
    'RM': dict(forward_variant='mid_read', read_after_block=6, asymmetric_mask=True),
}
H_FROZEN = ('ef_pool_frozen', 'ef_relation_frozen')
PROBE200 = 'ef_linear_0200'


def reference(args):
    b8 = a.baseline(args.source_root, args.gap_root, args.source_tag, args.coverage_tag, args.data_root)
    b8_plan = a.build_plan(b8, args.stage_a_root, args.stage_a_tag, 'B8')
    ar_plan = a.read(args.stage_a_root/'result'/(args.stage_a_tag+'_AR')/'plan.json')
    if ar_plan != a.build_plan(b8, args.stage_a_root, args.stage_a_tag, 'AR'):
        raise ValueError('A-R plan is not the expected Stage A comparison')
    out = layout(ar_plan)['result']
    ssl = a.completed(out/'pretrain')
    ssl_protocol = a.verify_inputs(out/'pretrain')
    if (ssl['epoch'] != 400 or ssl['smoke'] or ssl['synthetic']
            or ssl_protocol['source'] != b8['lock']['epoch0']
            or ssl_protocol['model']['overrides'] != a.VARIANTS['AR']):
        raise ValueError('A-R does not match the required fixed-400 shared initialization')
    endpoints = a.task_endpoints(ar_plan)
    for task, entry in endpoints.items():
        test = a.completed(out.with_name(out.name+'_test')/a.TASK_STAGES[task])
        if test['checkpoint_sha256'] != file_hash(entry[2]):
            raise ValueError('A-R TEST does not bind the VAL-selected checkpoint')
    return b8, b8_plan, ar_plan, endpoints


def plans(args, b8, ar):
    def shell(name):
        cfg = copy.deepcopy(ar)
        cfg['run'] = dict(root=str(args.root), tag=safe_name(args.tag+'_'+name), cache_root=str(args.cache_root))
        return cfg

    epoch400 = source_path(ar, dict(source='@pretrain/epoch_0400'))
    epoch200 = source_path(ar, dict(source='@pretrain/epoch_0200'))
    ef_probe = next(stage for stage in ar['stages'] if stage['id'] == a.PROBES['ef'])
    ef_ft = next(stage for stage in ar['stages'] if stage['id'] == a.TASK_STAGES['ef'])
    def probe200(source):
        return dict(copy.deepcopy(ef_probe), id=PROBE200, source=source)

    result = {}
    trajectory = shell('AR200')
    trajectory['stages'] = [probe200(epoch200)]
    result['AR200'] = trajectory
    h = shell('H')
    frozen = copy.deepcopy(ef_ft)
    frozen.update(mode='frozen_lite', calibrated_ef=True, source=epoch400)
    frozen['data']['augmentation'] = {'enabled': False}
    frozen.pop('matched_inputs', None)
    h['stages'] = [
        dict(copy.deepcopy(frozen), id=H_FROZEN[0]),
        dict(copy.deepcopy(frozen), id=H_FROZEN[1], ef_temporal_relation=True),
    ]
    h_ft = dict(copy.deepcopy(ef_ft), source=epoch400, ef_temporal_relation=True)
    # The head is the sole change; keep the other dataset/loss identities locked.
    if 'matched_inputs' in h_ft:
        h_ft['matched_inputs'].pop('head', None)
    h['stages'].append(h_ft)
    result['H'] = h

    for name, change in VARIANTS.items():
        cfg = shell(name)
        cfg['stage_a_baseline'] = b8['lock']
        pretrain = cfg['stages'][0]
        pretrain['model']['overrides'] = change
        cfg['stages'].insert(1, probe200('@pretrain/epoch_0200'))
        panel = cfg['stages'][-1]
        panel['head_requests'] = {task:str(layout(cfg,stage)['result']/'request.json')
                                  for task,stage in a.TASK_STAGES.items()}
        panel['budget_stages'] = {task:str(Path(request).parent) for task,request in panel['head_requests'].items()}
        panel['budget_ssl'] = str(layout(cfg,'pretrain')['result'])
        result[name] = cfg
    return {key:validate(cfg) for key,cfg in result.items()}


def pair(cfg, stages, *, same_head):
    return a.summary_pair(*(layout(cfg,s)['result'] for s in stages),'ef',same_head=same_head)


def gate_h(row):
    # Near parity merits one full FT check, but not a search over alternative heads.
    return row['candidate']['mae_pp'] <= row['reference']['mae_pp']+.05


def gate_r(ef, seg):
    return (ef['reference']['mae_pp']-ef['candidate']['mae_pp'] >= .10
            and seg['reference']['dice_mean']-seg['candidate']['dice_mean'] <= .001)


def compare_r(cfg, ar, ar200, endpoints, name):
    out, ref = layout(cfg)['result'], layout(ar)['result']
    full_ref = ref.with_name(ref.name+'_val_full')/a.TASK_STAGES['seg']
    full_candidate = out.with_name(out.name+'_val_full')/a.TASK_STAGES['seg']
    a.match_observations(full_ref,full_candidate,'seg')
    for directory, endpoint in ((full_ref,endpoints['seg']), (full_candidate,a.task_endpoints(cfg)['seg'])):
        if a.completed(directory)['checkpoint_sha256'] != file_hash(endpoint[2]):
            raise ValueError('Complete VAL is not bound to its selected FT weights')
    ssl = [a.verify_inputs(p/'pretrain') for p in (ref,out)]
    for key in ('source','training','data','evaluation','precision'):
        if ssl[0][key] != ssl[1][key]:
            raise ValueError(f'Mismatched SSL protocol: {key}')
    ft = {task:a.summary_pair(ref/stage,out/stage,task) for task,stage in a.TASK_STAGES.items()}
    probes = {task:a.summary_pair(ref/stage,out/stage,task) for task,stage in a.PROBES.items()}
    probes['ef_0200'] = a.summary_pair(layout(ar200,PROBE200)['result'],out/PROBE200,'ef')
    diag = {key:a.completed(p/'diagnostics') for key,p in (('AR',ref),(name,out))}
    if diag['AR']['panel_sha256'] != diag[name]['panel_sha256']:
        raise ValueError('A-R and candidate diagnostic panels differ')
    return dict(ft=ft, probes=probes, diagnostics=diag,
                cost={stage:{'AR':a.cost(ref/stage),name:a.cost(out/stage)}
                      for stage in ('pretrain',*a.TASK_STAGES.values())},
                full_val_seg={'AR':a.completed(full_ref)['metrics'],name:a.completed(full_candidate)['metrics']},
                crosses_practical_val_gate=gate_r(ft['ef'],ft['seg']))


def run_queue(configs, args, ar, endpoints, control):
    h = configs['H']
    execute(h, only=H_FROZEN)
    if a.completed(layout(h,H_FROZEN[0])['result'])['cache']['directory'] != a.completed(
            layout(h,H_FROZEN[1])['result'])['cache']['directory']:
        raise ValueError('H frozen controls did not use the same feature cache')
    val = dict(H=dict(frozen=pair(h,H_FROZEN,same_head=False)), arms={},
               decision='VAL only; frozen probes do not gate R candidate FT; no automatic R combinations')
    gate = gate_h(val['H']['frozen'])
    a.lock_selection(control/'H_val_gate.json',dict(run_ft=gate, tolerance_pp=.05,
                     reference=val['H']['frozen']['reference']['mae_pp'],
                     candidate=val['H']['frozen']['candidate']['mae_pp']))
    results = ['queue_plan.json','H_val_gate.json','matched_val.json']
    if gate:
        ef = a.TASK_STAGES['ef']
        execute(h, only=[ef])
        endpoint = a.locked_checkpoint(Path(h['run']['root']),h['run']['tag'],ef)
        val['H']['ft'] = a.summary_pair(endpoints['ef'][0],endpoint[0],'ef',same_head=False)
        a.test_endpoints({'AR':{'ef':endpoints['ef']},'H':{'ef':endpoint}},[(h,[ef])],
                         args.data_root,ar['data']['seed'],control,'H',reference='AR')
        results += ['H_test_lock.json','H_test.json']
    write_json(control/'matched_val.json',val)
    execute(configs['AR200'])
    for name in VARIANTS:
        cfg = configs[name]
        execute(cfg)
        a.evaluate(cfg,[a.TASK_STAGES['seg']],args.data_root,'val_full')
        val['arms'][name] = compare_r(cfg,ar,configs['AR200'],endpoints,name)
        write_json(control/'matched_val.json',val)
        a.test_endpoints({'AR':endpoints,name:a.task_endpoints(cfg)},[(cfg,list(a.TASK_STAGES.values()))],
                         args.data_root,ar['data']['seed'],control,name,reference='AR')
        results += [f'{name}_test_lock.json',f'{name}_test.json']
    mark_done(control,results)


def run_smoke(configs):
    execute(a.smoke_config(configs['H']))
    for name in VARIANTS:
        cfg = a.smoke_config(configs[name])
        train = cfg['stages'][0]['training']
        train.update(epochs=2, save_epochs=[0,2])
        paused = copy.deepcopy(cfg)
        paused['stages'][0]['training']['run_until_epoch'] = 1
        status = layout(cfg,'pretrain')['result']
        if not verify_done(status):
            if not (status/'PAUSED.json').exists():
                execute(paused,only=['pretrain'])
            if a.read(status/'PAUSED.json')['epoch'] != 1:
                raise ValueError('Smoke did not create its first-epoch recovery point')
        execute(cfg)
        if a.completed(status)['epoch'] != 2:
            raise ValueError('Smoke did not reach the resumed second epoch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','smoke','run'))
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-fs/outputs_cr_stage_a2'))
    parser.add_argument('--cache-root',type=Path,default=Path('/root/autodl-tmp/outputs_cr_stage_a2_cache'))
    parser.add_argument('--tag',default='stage_a2_recurrent_20261114')
    parser.add_argument('--stage-a-root',type=Path,default=Path('/root/autodl-fs/outputs_cr_stage_a'))
    parser.add_argument('--stage-a-tag',default='patch8_stage_a_v1')
    parser.add_argument('--source-root',type=Path,default=Path('/root/autodl-fs/outputs_cr_spatial'))
    parser.add_argument('--gap-root',type=Path,default=Path('/root/autodl-fs/outputs_cr_gap'))
    parser.add_argument('--source-tag',default='cr_patch8_w64_400_v1')
    parser.add_argument('--coverage-tag',default='seg_gap_stage1_v1_coverage')
    parser.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    args = parser.parse_args()
    for value in (args.tag,args.stage_a_tag,args.source_tag,args.coverage_tag):
        safe_name(value)
    b8, _, ar, endpoints = reference(args)
    configs = plans(args,b8,ar)
    if args.action == 'plan':
        import json
        print(json.dumps(configs,indent=2))
        return
    control = args.root/'control'/(args.tag+('_smoke' if args.action=='smoke' else ''))
    with run_lock(control):
        a.lock_selection(control/'queue_plan.json',dict(configs=configs, runner_sha256=file_hash(__file__)))
        if verify_done(control):
            print(f'Verified completed queue: {control}',flush=True)
            return
        if args.action == 'smoke':
            run_smoke(configs)
            return
        run_queue(configs,args,ar,endpoints,control)


if __name__ == '__main__':
    main()
