"""Run D and G EF FT serially, then sealed matched VAL/TEST and a result-only archive."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import code_identity, export_results, file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import execute
from research_pipeline.config import safe_name, validate
from tools.evaluate_patch8_test import locked_checkpoint
from tools.run_ar_ef_geometry import locked_windows
from tools.run_ar_ef_pairing import cohort, dataset, population_strata, reduce_results
from tools.run_ar_motion_aux import check_draws, reuse_m1_runtime
from tools.run_ar_motion_ft import evaluate_arm, plan
from tools.run_patch8_stage_a import lock_selection, read, verify_inputs


ARMS = ('D','G')
GATE = .10


def gap_plan(args):
    cfg,ref,manifest = plan(args)
    base = {k:v for k,v in cfg['stages'][0].items() if k!='motion_arm'}
    cfg['adapter'] = 'research_pipeline.ar_gap_ft:build'
    cfg['stages'] = [dict(base,id=arm,gap_arm=arm) for arm in ARMS]
    if not cfg.get('smoke'):
        cfg['runtime'].update(batch_candidates=[4,8,16,32],worker_candidates=[0,2,4],thread_candidates=[2])
    return validate(cfg),ref,manifest


def endpoint(cfg, arm, source):
    value = locked_checkpoint(Path(cfg['run']['root']),cfg['run']['tag'],arm)
    d,s,_,request = value
    protocol = verify_inputs(d)
    if (request['config']['stage']['gap_arm']!=arm or s.get('smoke') or s.get('synthetic')
            or file_hash(request['source'])!=file_hash(source) or protocol['source']!=file_hash(source)):
        raise ValueError('Expected a matching VAL-locked D/G endpoint')
    return value


def paired_split(dest, split, data):
    groups = population_strata(data)
    result = {}
    for policy in ('single','five'):
        rows = {}
        for arm in ARMS:
            p = dest/f'{arm}_{split}'/policy/'candidate_windows.csv'
            with p.open(encoding='utf-8-sig',newline='') as handle:
                rows[arm] = [dict(patient=r['patient'],start=int(r['recent_start']),
                                  target=float(r['target']),prediction=float(r['prediction']))
                             for r in csv.DictReader(handle)]
        keys = ('patient','start','target')
        if [tuple(r[k] for k in keys) for r in rows['D']]!=[tuple(r[k] for k in keys) for r in rows['G']]:
            raise ValueError('D/G post-training inference windows differ')
        result[policy],_ = reduce_results(rows,groups,[('D','G')])
    return result


def smoke_validate(cfg):
    """Validate the gradient path, independent chronology and original head tensor initialization."""
    import torch
    from research_pipeline.ar_gap_ft import source_groups
    from research_pipeline.cli import configure
    from research_pipeline.config import stage_config
    from research_pipeline.engine import amp_context, build_experiment
    head = None
    updates = {}
    for stage in cfg['stages']:
        settings = stage_config(cfg,stage)
        configure(settings)
        device = torch.device(settings['runtime']['device'])
        exp = build_experiment(settings,stage['source'],device)
        b = exp.backbone
        descriptor = source_groups(stage['id']).flatten()
        torch.testing.assert_close(descriptor[descriptor.argsort()],torch.arange(64))
        weights = {n:v.detach().cpu().clone() for n,v in exp.model.head.state_dict().items()}
        if head is not None:
            for n, v in weights.items():
                torch.testing.assert_close(head[n],v,atol=0,rtol=0)
        head = weights
        called = []
        original = b.encode_current
        def observe(x, mask, state):
            called.append((x.shape[1],mask is None,state is None))
            return original(x,mask,state)
        b.encode_current = observe
        opt = torch.optim.AdamW(exp.model.parameters(),lr=settings['training']['lr'])
        exp.model.train()
        batch = [exp.train[0]]
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with amp_context(settings,device):
                value = exp.loss(batch)
            value.backward()
            opt.step()
        grad = b.patch_embed.proj.weight.grad
        if called != [(16,True,True)]*8 or grad is None or not grad.isfinite().all() or not bool(grad.abs().sum()>0):
            raise ValueError('Smoke did not observe effective encoder gradients and four independent L16 calls')
        if any(p.grad is not None for p in b.memory.parameters()):
            raise ValueError('Independent EF group unexpectedly trained recurrent state')
        updates[stage['id']] = dict(real_updates=2,patch_embed_gradient=float(grad.abs().sum()),
                                    encoder_calls=called,head_initialization='identical')
        del exp,opt,b,original,grad
        if device.type=='cuda':
            torch.cuda.empty_cache()
    return updates


def run(args):
    cfg,ref,manifest = gap_plan(args)
    source = ref[3]['source']
    if args.action=='plan':
        print(json.dumps(dict(config=cfg,queue=['D','G','VAL and locked TEST','result-only ZIP'],
                              baseline_checkpoint=str(ref[2]),test_labels_opened=False),indent=2))
        return
    if args.action=='run' and verify_done(args.runtime_reference):
        # The same 4090, input shapes, and encoder are used, without M1's extra recurrence.
        # Both arms receive the same measured execution settings; otherwise tune as usual.
        reuse_m1_runtime(cfg,args.runtime_reference)
    from utils.final_temporal_data import build_manifest
    dest = layout(cfg)['result']/'gap_queue'
    out_zip = Path(cfg['run']['root'])/'exports'/(cfg['run']['tag']+'.zip')
    smoke = args.action=='smoke'
    with run_lock(dest):
        lock_selection(dest/'protocol.json',dict(config=cfg,baseline=file_hash(ref[2]),ssl=file_hash(source),
            e0_val=file_hash(args.e0/'e0_val'/'DONE.json'),code=code_identity([__file__]),
            scientific_scope='same raw 64 frames, four independent L16s; changes both tubelet span and attention input'))
        if verify_done(dest):
            if not smoke and not out_zip.is_file():
                export_results(layout(cfg)['result'],out_zip)
            print(dest/'comparison.json')
            return
        ckpts = layout(cfg)['ckpt']
        ckpts.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(ckpts).free<8*1024**3:
            raise OSError('Need at least 8 GiB free for two independent EF FT endpoints and recovery states')
        if smoke:
            write_json(dest/'gradient_smoke.json',smoke_validate(cfg))
        execute(cfg,only=list(ARMS))
        if smoke:
            write_json(dest/'comparison.json',dict(smoke=True,test=None))
            mark_done(dest,['protocol.json','gradient_smoke.json','comparison.json'])
            return
        end = {arm:endpoint(cfg,arm,source) for arm in ARMS}
        matched = check_draws(end['D'][0],end['G'][0])
        seed = cfg['data']['seed']
        val = dataset(manifest,'val',seed)
        val_base = locked_windows(args.e0/'e0_val',val)
        results = dict(val={},test={},direct={},matched_training_draw_epochs=matched)
        for arm in ARMS:
            results['val'][arm] = evaluate_arm(end[arm],manifest,'val',val,val_base,dest/(arm+'_val'))
        results['direct']['val'] = paired_split(dest,'val',val)
        gain = -results['direct']['val']['single']['pairs']['G_minus_D']['candidate_minus_reference']
        decision = dict(G_gain_vs_D_single_val_pp=gain,practical_threshold_pp=GATE,
                        reached_threshold=gain>=GATE,no_auto_promotion=True)
        lock_selection(dest/'TEST_LOCK.json',dict(arms={a:dict(sha256=file_hash(v[2]),epoch=v[1]['selected_epoch'])
                     for a,v in end.items()},val_decision=decision,
                     test_policy='every fitted arm, identical fixed windows, never used for selection'))
        old_test = args.e0/'e0_test'
        if not verify_done(old_test) or read(old_test/'summary.json')['checkpoint_sha256']!=file_hash(ref[2]):
            raise ValueError('Original sealed A-R TEST unavailable or changed')
        tm = build_manifest(args.data_root,seed=seed,recent_frames=64,local_frames=16,max_prefix=0,splits=('test',))
        if read(args.e0/'test_manifest_lock.json')['sha256']!=tm['manifest_sha256']:
            raise ValueError('Official TEST population differs from original A-R comparison')
        test = dataset(tm,'test',seed)
        if set(cohort(test)) & {r['patient'] for split in ('train','val') for r in manifest[split]}:
            raise ValueError('Supervised development and TEST populations overlap')
        test_base = locked_windows(old_test,test)
        for arm in ARMS:
            results['test'][arm] = evaluate_arm(end[arm],tm,'test',test,test_base,dest/(arm+'_test'))
        results['direct']['test'] = paired_split(dest,'test',test)
        results['decision_from_val_only'] = decision
        results['scope'] = 'Offline W64 grouping, conditional EF FT, no SSL/segmentation/clinical inference claim'
        write_json(dest/'comparison.json',results)
        mark_done(dest,['protocol.json','comparison.json','TEST_LOCK.json',
                        *(a+s+'/DONE.json' for a in ARMS for s in ('_val','_test'))])
        print(json.dumps(export_results(layout(cfg)['result'],out_zip)),flush=True)
        print(dest/'comparison.json',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','smoke','run'])
    p.add_argument('--tag',type=safe_name,required=True)
    p.add_argument('--cpu',action='store_true')
    defaults = dict(root='/root/autodl-fs/outputs_ar_gap_ft',
        runtime_reference='/root/autodl-fs/outputs_ar_motion_ft/result/ar_motion_ft_20261125/M1',
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
