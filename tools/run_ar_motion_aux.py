"""One matched M1 auxiliary-only FT and one VAL-only residual intervention."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import code_identity, file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import execute
from research_pipeline.config import safe_name
from tools.evaluate_patch8_test import locked_checkpoint
from tools.run_ar_ef_geometry import locked_windows
from tools.run_ar_motion_ft import arm_endpoint, evaluate_arm, plan
from tools.run_ar_ef_pairing import cohort, dataset
from tools.run_patch8_stage_a import lock_selection, read, verify_inputs


def check_draws(old, new):
    def draws(path):
        with path.open(encoding='utf-8-sig',newline='') as handle:
            return {int(r['epoch']):r['window_sequence_sha256'] for r in csv.DictReader(handle)}
    first,second = draws(old/'metrics.csv'),draws(new/'metrics.csv')
    matched = sorted(first.keys() & second.keys())
    if not matched or any(first[k]!=second[k] for k in matched):
        raise ValueError('M1 and M1_aux have different original sampled windows')
    return matched


def reference_windows(directory, baseline):
    results = []
    for policy,expected in zip(('single','five'),baseline):
        with (directory/policy/'candidate_windows.csv').open(encoding='utf-8-sig',newline='') as handle:
            rows = [dict(patient=r['patient'],start=int(r['recent_start']),target=float(r['target']),
                         prediction=float(r['prediction'])) for r in csv.DictReader(handle)]
        keys = ('patient','start','target')
        if [tuple(r[k] for k in keys) for r in rows]!=[tuple(r[k] for k in keys) for r in expected]:
            raise ValueError(f'M1 {policy} windows differ from the locked A-R baseline')
        results.append(rows)
    return results


def reuse_m1_runtime(cfg, old):
    import torch
    runtime = read(old/'runtime.json')
    hw, selected = runtime['hardware'],runtime['selected']
    peak = max((row['peak_reserved_bytes'] for row in runtime['trials']
                if row.get('status')=='ok' and row['selection']==selected),default=0)
    device = torch.device(cfg['runtime']['device'])
    if device.type!='cuda' or not torch.cuda.is_available() or selected['gradient_checkpointing'] or not peak:
        return
    free,total = torch.cuda.mem_get_info(device)
    if (hw['name']==torch.cuda.get_device_name(device) and hw['total_bytes']==total
        and hw['torch']==str(torch.__version__) and free>=peak+1024**3):
        cfg['runtime'].update(autotune=False,micro_batch=selected['micro_batch'],
            worker_candidates=[selected['num_workers']],threads=selected.get('threads',hw['threads']))
        print(json.dumps(dict(runtime='reuse measured M1 execution settings',selected=selected)),flush=True)


def run(args):
    cfg,ref,manifest = plan(args)
    source = ref[3]['source']
    m1 = locked_checkpoint(args.previous_root,args.previous_tag,'M1')
    previous = args.previous_root/'result'/args.previous_tag/'motion_queue'
    m1_val = previous/'M1_val'
    if not verify_done(previous) or not verify_done(m1_val):
        raise ValueError('Old M1 formal queue and VAL predictions must already be sealed')
    p = verify_inputs(m1[0])
    if (file_hash(source)!=p['source'] or m1[1].get('smoke')
        or m1[3]['config']['stage']['motion_arm']!='M1'
        or read(m1_val/'summary.json')['checkpoint']!=file_hash(m1[2])
        or not read(m1_val/'summary.json').get('residual',True)
        or (args.action!='smoke' and (
        {k:v for k,v in cfg['training'].items() if k!='run_until_epoch'}!=p['training']
        or cfg['data']!=p['data'] or cfg['model']!=p['model'] or cfg['evaluation']!=p['evaluation']
        or cfg['runtime']['precision']!=p['precision']))):
        raise ValueError('M1_aux must reuse M1 scientific recipe, restored SSL source and evaluation cohort')
    cfg['stages'] = [s for s in cfg['stages'] if s['id']=='M1_aux']
    if args.action=='plan':
        print(json.dumps(dict(stages=['M1_aux'],source=source,previous_m1=str(m1[2]),test_labels_opened=False),indent=2))
        return
    if args.action=='run':
        reuse_m1_runtime(cfg,m1[0])
    dest = layout(cfg)['result']/'aux_queue'
    from utils.final_temporal_data import build_manifest
    with run_lock(dest):
        protocol_path = dest/'protocol.json'
        sources = dict(source=file_hash(source),m1=file_hash(m1[2]),
                       previous_m1_val=file_hash(m1_val/'DONE.json'))
        eval_only = args.action=='evaluate'
        if eval_only:
            prior = read(protocol_path)
            check = lambda c: {k:v for k,v in c.items() if k!='runtime'}
            if check(prior['config'])!=check(cfg) or any(prior.get(k)!=v for k,v in sources.items()):
                raise ValueError('Existing training run does not match the immutable auxiliary evaluation inputs')
            cfg = prior['config']
            info = arm_endpoint(cfg,'M1_aux',source)
            if read(info[0]/'request.json')['config']['training']!=cfg['training']:
                raise ValueError('Existing M1_aux checkpoint does not match the original training budget')
        else:
            lock_selection(protocol_path,dict(config=cfg,**sources,code=code_identity([__file__])))
        if verify_done(dest):
            print(dest/'comparison.json')
            return
        if not eval_only:
            disk = layout(cfg)['ckpt']
            disk.mkdir(parents=True,exist_ok=True)
            if shutil.disk_usage(disk).free<5*1024**3:
                raise OSError('Need at least 5 GiB free for recoverable FT checkpoints')
            execute(cfg,only=['M1_aux'])
        if args.action=='smoke':
            write_json(dest/'comparison.json',dict(smoke=True,test=None))
            mark_done(dest,['protocol.json','comparison.json'])
            return
        info = info if eval_only else arm_endpoint(cfg,'M1_aux',source)
        if eval_only:
            lock_selection(dest/'EVAL_PROTOCOL.json',dict(training_protocol_sha256=file_hash(protocol_path),
                checkpoint_sha256=file_hash(info[2]),code=code_identity([__file__]),training=False))
        epochs = check_draws(m1[0],info[0])
        seed = cfg['data']['seed']
        val = dataset(manifest,'val',seed)
        baseline = locked_windows(args.e0/'e0_val',val)
        off_dest = dest/'M1_residual_off_val'
        aux_dest = dest/'M1_aux_val'
        off_val = evaluate_arm(m1,manifest,'val',val,reference_windows(m1_val,baseline),off_dest,residual=False)
        aux_val = evaluate_arm(info,manifest,'val',val,baseline,aux_dest)
        lock_selection(dest/'TEST_LOCK.json',dict(M1_aux_sha256=file_hash(info[2]),
            m1_intervention_scope='VAL only; one existing checkpoint',selection='M1_aux VAL-selected weights only'))
        previous_test = args.e0/'e0_test'
        if not verify_done(previous_test) or read(previous_test/'summary.json')['checkpoint_sha256']!=file_hash(ref[2]):
            raise ValueError('Original A-R TEST source changed')
        tm = build_manifest(args.data_root,seed=seed,recent_frames=64,local_frames=16,max_prefix=0,splits=('test',))
        if read(args.e0/'test_manifest_lock.json')['sha256']!=tm['manifest_sha256']:
            raise ValueError('TEST manifest differs from locked A-R TEST')
        test = dataset(tm,'test',seed)
        if set(cohort(test)) & {r['patient'] for split in ('train','val') for r in manifest[split]}:
            raise ValueError('Supervised development and TEST overlap')
        test_dest = dest/'M1_aux_test'
        aux_test = evaluate_arm(info,tm,'test',test,locked_windows(previous_test,test),test_dest)
        write_json(dest/'comparison.json',dict(matched_training_draw_epochs=epochs,M1_aux_val=aux_val,
            M1_aux_test=aux_test,M1_residual_off_val=off_val,
            residual_off_reference='original locked M1 predictions (residual on); AR label in child output names reference'))
        mark_done(dest,['comparison.json','protocol.json','TEST_LOCK.json',
            *(['EVAL_PROTOCOL.json'] if eval_only else []),
            *[p.name+'/DONE.json' for p in (off_dest,aux_dest,test_dest)]])
        print(dest/'comparison.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','smoke','run','evaluate'])
    p.add_argument('--tag',type=safe_name,required=True)
    p.add_argument('--cpu',action='store_true')
    defaults = dict(root='/root/autodl-fs/outputs_ar_motion_ft',previous_root='/root/autodl-fs/outputs_ar_motion_ft',
        stage_a_root='/root/autodl-fs/outputs_cr_stage_a',data_root='/root/autodl-tmp/datasets/EchoNet-Dynamic',
        e0='/root/autodl-fs/outputs_ar_ef_pairing/result/ef_pair_20261125/ef_pairing')
    for k,v in defaults.items():
        p.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    p.add_argument('--stage-a-tag',type=safe_name,default='patch8_stage_a_v1_AR')
    p.add_argument('--previous-tag',type=safe_name,default='ar_motion_ft_20261125')
    args=p.parse_args()
    if args.tag in (args.previous_tag,args.stage_a_tag):
        p.error('Use an independent run tag')
    if args.cpu and args.action!='smoke':
        p.error('--cpu is smoke only')
    run(args)


if __name__=='__main__':
    main()
