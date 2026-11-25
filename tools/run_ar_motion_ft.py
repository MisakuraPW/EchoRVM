"""One matched supervised M1 EF run, then one VAL-gated second arm."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import code_identity, file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import configure, execute
from research_pipeline.config import safe_name, validate
from tools.evaluate_patch8_test import locked_checkpoint
from tools.run_ar_ef_geometry import locked_windows, samples
from tools.run_ar_ef_pairing import cohort, dataset, population_strata, reduce_results, save_predictions
from tools.run_patch8_stage_a import lock_selection, read, verify_inputs


GATE = .10


def second_arm(val_gain):
    return 'M1_aux' if val_gain>=GATE else 'M2'


def plan(args):
    reference = locked_checkpoint(args.stage_a_root,args.stage_a_tag,'ef_l1_ft')
    old,summary,checkpoint,request = reference
    protocol = verify_inputs(old)
    cfg = copy.deepcopy(request['config'])
    stage = cfg.pop('stage')
    native = protocol['inputs']['native_config']
    manifest = read(cfg['data']['manifest'])
    ef0 = args.e0/'e0_val'
    e0_lock = read(args.e0/'protocol.json')
    expected_epoch = 21
    source = request['source']
    if (summary['selected_epoch']!=expected_epoch or summary.get('smoke') or summary.get('synthetic')
        or not verify_done(ef0) or not source.endswith('/epoch_0400.pt') or file_hash(source)!=protocol['source']
        or e0_lock['ssl_sha256']!=protocol['source'] or not isinstance(stage['matched_inputs'],dict)
        or read(ef0/'summary.json')['checkpoint_sha256']!=file_hash(checkpoint)
        or e0_lock['manifest_sha256']!=file_hash(cfg['data']['manifest'])
        or Path(manifest['root']).resolve()!=args.data_root.resolve() or 'test' in manifest
        or (native['patch_size'],native['tubelet_size'],native['forward_variant'],native['read_after_block'])!=(8,2,'mid_read',6)
        or cfg['data']['augmentation']!=protocol['data']['augmentation']
        or {k:v for k,v in cfg['training'].items() if k!='run_until_epoch'}!=protocol['training']
        or cfg['evaluation']!=protocol['evaluation']
        or cfg['runtime']['precision']!=protocol['precision']):
        raise ValueError('Original matched A-R FT and E0 are not reusable; do not launch unmatched training')
    cfg.update(adapter='research_pipeline.ar_motion_ft:build',
               run=dict(root=str(args.root),tag=args.tag),stages=[dict(stage,source=source,id=arm,motion_arm=arm)
                    for arm in ('M1','M1_aux','M2')])
    # Short measured runtime trials only; do not change effective batch or training settings.
    cfg['runtime'].update(batch_candidates=[4,8,16,32],worker_candidates=[0,2,4],
                          evict_completed_feature_caches=False)
    if args.action=='smoke':
        cfg['run']['tag'] += '_smoke'
        cfg['smoke'] = True
        cfg['data'].update(limit_train=24,limit_val=8)
        cfg['training'].update(epochs=1,effective_batch=4,tensorboard=False)
        cfg['runtime'].update(autotune=False,micro_batch=2,worker_candidates=[0])
        if args.cpu:
            cfg['runtime'].update(device='cpu',precision='fp32')
    return validate(cfg),reference,manifest


def probe(cfg, source, dest):
    import torch
    from research_pipeline.engine import amp_context, build_experiment
    from research_pipeline.config import stage_config
    from research_pipeline import stage_a
    settings = stage_config(cfg,cfg['stages'][0])
    configure(settings)
    device = torch.device(settings['runtime']['device'])
    exp = build_experiment(settings,source,device)
    baseline = stage_a.build(settings,settings['stage'],source,device).model.eval()
    exp.model.eval()
    rows = exp.calibration_samples(2)
    if not all(r.get('motion_target') for r in rows):
        raise ValueError('Smoke requires actual in-window TRAIN endpoint pairs')
    seen = []
    handles = [m.head.register_forward_pre_hook(lambda head,args:seen.append(args[0].detach()))
               for m in (baseline,exp.model)]
    with torch.no_grad(), amp_context(settings,device):
        base = baseline(rows)
        zero = exp.model(rows)
        torch.testing.assert_close(base,zero)
        torch.testing.assert_close(seen[0],seen[1],rtol=1e-3,atol=1e-3)
        maximum = float((seen[0]-seen[1]).abs().max())
    for h in handles:
        h.remove()
    del baseline,seen
    exp.model.train()
    opt = torch.optim.AdamW(exp.model.parameters(),lr=settings['training']['lr'])
    write_grads,projection_grads = [],[]
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with amp_context(settings,device):
            loss = exp.loss(rows)
        loss.backward()
        motion = exp.model.motion
        write_grads.append(float(motion.write.weight.grad.abs().sum()))
        projection_grads.append(float(motion.project[1].weight.grad.abs().sum()))
        opt.step()
    with torch.no_grad(),amp_context(settings,device):
        enabled,disabled = exp.model(rows),exp.model(rows,residual=False)
    # BF16 may round the scalar EF after only three updates; check a nonzero EF-only gradient instead.
    opt.zero_grad(set_to_none=True)
    with amp_context(settings,device):
        clean = [{k:v for k,v in r.items() if k!='motion_target'} for r in rows]
        pred = exp.model(clean)
        loss = exp._task_loss(pred,clean)
    loss.backward()
    ps = exp.model.motion
    ef_path = float(ps.write.weight.grad.abs().sum())
    ef_projection = float(ps.project[1].weight.grad.abs().sum())
    if max(write_grads)<=0 or max(projection_grads)<=0 or min(ef_path,ef_projection)<=0:
        raise ValueError('No effective gradient from downstream EF through the previous-clip motion write')
    if any('motion_target' in exp.val[i] for i in range(len(exp.val))):
        raise ValueError('VAL accidentally carries TRAIN auxiliary targets')
    write_json(dest/'gradient_check.json',dict(zero_init_descriptor_max_abs=maximum,
        write_grads=write_grads,aux_projection_grads=projection_grads,ef_only_write_grad=ef_path,
        ef_only_projection_grad=ef_projection,empty_aux_validation=True,
        prediction_delta_after_three_updates_pp=float((enabled-disabled).abs().max())*100))


def arm_endpoint(cfg,name,source):
    info = locked_checkpoint(Path(cfg['run']['root']),cfg['run']['tag'],name)
    d,s,_,req = info
    protocol = verify_inputs(d)
    if req['config']['stage']['motion_arm']!=name or file_hash(req['source'])!=file_hash(source) or protocol['source']!=file_hash(source) or s.get('smoke'):
        raise ValueError('Cannot evaluate a mismatched formal motion checkpoint')
    return info


def evaluate_arm(info, manifest, split, data, baseline, dest):
    import torch
    from research_pipeline.engine import Heartbeat, amp_context, build_experiment
    from research_pipeline.runtime import loader
    from tools.watch_pipeline_progress import StageDisplay
    if verify_done(dest):
        return read(dest/'summary.json')
    d,summary,ckpt,req = info
    cfg = req['config']
    configure(cfg)
    device = torch.device(cfg['runtime']['device'])
    exp = build_experiment(cfg,req['source'],device)
    if split=='test' and set(cohort(data)) & (set(exp.metadata['train_ids'])|set(exp.metadata['val_ids'])):
        raise ValueError('Inherited TEST population overlaps with supervised development cases')
    weights = torch.load(ckpt,map_location='cpu',weights_only=True)
    if weights['variant']!=exp.model.arm or weights['epoch']!=summary['selected_epoch']:
        raise ValueError('Wrong weights for VAL-locked motion arm')
    exp.model.load_state_dict(weights['model'],strict=True)
    exp.model.eval().requires_grad_(False)
    del weights
    timing = {}
    groups = population_strata(data)
    runtime = dict(read(d/'runtime.json')['selected'])
    beat = Heartbeat(dest,5)
    result = {}
    for policy,reference in zip(('single','five'),baseline):
        windows = samples(manifest,split,data.seed,reference)
        rows = []
        started = time.perf_counter()
        beat(force=True,phase='validation',arm=policy,completed=0,total=len(windows))
        with StageDisplay(dest,f'{exp.model.arm} {split}/{policy}'),torch.inference_mode():
            for batch in loader(windows,cfg,runtime,training=False):
                clean = [{k:r[k] for k in ('video','prefix_frames','fps')} for r in batch]
                with amp_context(cfg,device):
                    p = exp.model(clean).float().tolist()
                rows.extend(dict(patient=r['patient'],recent_start=int(r['recent_start']),
                                 target=float(r['target']),prediction=v*100) for v,r in zip(p,batch))
                beat(phase='validation',arm=policy,completed=len(rows),total=len(windows))
        timing[policy] = dict(wall_seconds=time.perf_counter()-started,windows=len(rows),clips=len(rows)*4,
                              frames=len(rows)*64)
        truth = [dict(patient=r['patient'],recent_start=r['start'],target=r['target'],prediction=r['prediction'])
                 for r in reference]
        keys = ('patient','recent_start','target')
        if [tuple(r[k] for k in keys) for r in rows]!=[tuple(r[k] for k in keys) for r in truth]:
            raise ValueError('A-R baseline and candidate source windows differ')
        arm_rows = dict(AR=truth,candidate=rows)
        result[policy],patients = reduce_results(arm_rows,groups,[('AR','candidate')])
        save_predictions(dest/policy,arm_rows,patients)
    result.update(split=split,cost=timing,checkpoint=file_hash(ckpt),epoch=summary['selected_epoch'],
                  training_wall_seconds=summary['wall_seconds'],trainable_parameters=summary['trainable_parameters'])
    write_json(dest/'summary.json',result)
    paths = [str(p.relative_to(dest)) for p in sorted(dest.rglob('*.csv'))]
    mark_done(dest,['summary.json',*paths])
    beat(force=True,phase='done')
    return result


def run(args):
    cfg,ref,manifest = plan(args)
    source = ref[3]['source']
    if args.action=='plan':
        print(json.dumps(dict(config=cfg,plan='M1 -> M1_aux if VAL single gain >=0.10 pp, else M2; then locked TEST',
                              reference=str(ref[2]),test_labels_opened=False),indent=2))
        return
    import torch
    from utils.final_temporal_data import build_manifest
    smoke = args.action=='smoke'
    dest = layout(cfg)['result']/'motion_queue'
    with run_lock(dest):
        lock_selection(dest/'protocol.json',dict(config=cfg,baseline=file_hash(ref[2]),ssl=file_hash(source),
            val_e0=file_hash(args.e0/'e0_val'/'DONE.json'),code=code_identity([__file__]),gate=GATE))
        if verify_done(dest):
            print(dest/'comparison.json')
            return
        disk = layout(cfg)['ckpt']
        disk.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(disk).free < 5*1024**3:
            raise OSError('Need at least 5 GiB free on the checkpoint destination before FT')
        if smoke and not (dest/'gradient_check.json').is_file():
            probe(cfg,source,dest)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        seed = cfg['data']['seed']
        val = dataset(manifest,'val',seed)
        baseline = locked_windows(args.e0/'e0_val',val)
        execute(cfg,only=['M1'])
        if smoke:
            write_json(dest/'comparison.json',dict(scope='synthetic-budget development-only smoke',test=None))
            mark_done(dest,['protocol.json','gradient_check.json','comparison.json'])
            return
        first = arm_endpoint(cfg,'M1',source)
        report = dict(val={},test={})
        report['val']['M1'] = evaluate_arm(first,manifest,'val',val,baseline,dest/'M1_val')
        gain = -report['val']['M1']['single']['pairs']['candidate_minus_AR']['candidate_minus_reference']
        second = second_arm(gain)
        decision = dict(val_single_gain_pp=gain,threshold_pp=GATE,second_arm=second,
                        selected_checkpoint=file_hash(first[2]),TEST_used_for_branch=False)
        lock_selection(dest/'BRANCH.json',decision)
        print(json.dumps(decision),flush=True)
        execute(cfg,only=[second])
        other = arm_endpoint(cfg,second,source)
        report['val'][second] = evaluate_arm(other,manifest,'val',val,baseline,dest/(second+'_val'))
        entries = {'M1':first,second:other}
        lock_selection(dest/'TEST_LOCK.json',dict(branch=decision,checkpoints={a:file_hash(i[2]) for a,i in entries.items()},
             comparison='all fitted arms; VAL-locked; two fixed inference policies; no TEST tuning'))
        source_test = args.e0/'e0_test'
        if not verify_done(source_test) or read(source_test/'summary.json')['checkpoint_sha256']!=file_hash(ref[2]):
            raise ValueError('Original locked E0 TEST is not available')
        tm = build_manifest(args.data_root,seed=seed,recent_frames=64,local_frames=16,max_prefix=0,splits=('test',))
        if read(args.e0/'test_manifest_lock.json')['sha256']!=tm['manifest_sha256']:
            raise ValueError('TEST identity changed since matched A-R baseline')
        test = dataset(tm,'test',seed)
        test_base = locked_windows(source_test,test)
        for arm,info in entries.items():
            report['test'][arm] = evaluate_arm(info,tm,'test',test,test_base,dest/(arm+'_test'))
        report.update(branch=decision,analysis_scope='supervised EF only; do not infer shared SSL or segmentation benefit')
        write_json(dest/'comparison.json',report)
        mark_done(dest,['protocol.json','BRANCH.json','TEST_LOCK.json','comparison.json',
                        *(a+s+'/DONE.json' for a in entries for s in ('_val','_test'))])
        print(dest/'comparison.json',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','smoke','run'])
    p.add_argument('--tag',type=safe_name,required=True)
    p.add_argument('--cpu',action='store_true')
    defaults = dict(root='/root/autodl-fs/outputs_ar_motion_ft',
        stage_a_root='/root/autodl-fs/outputs_cr_stage_a',data_root='/root/autodl-tmp/datasets/EchoNet-Dynamic',
        e0='/root/autodl-fs/outputs_ar_ef_pairing/result/ef_pair_20261125/ef_pairing')
    for key,value in defaults.items():
        p.add_argument('--'+key.replace('_','-'),type=Path,default=Path(value))
    p.add_argument('--stage-a-tag',type=safe_name,default='patch8_stage_a_v1_AR')
    args = p.parse_args()
    if args.cpu and args.action!='smoke':
        p.error('--cpu is smoke only')
    run(args)


if __name__=='__main__':
    main()
