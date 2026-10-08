"""Bounded final temporal round: reuse diagnostics, two matched new trainings, endpoints."""

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from tools.run_stage1_lengths import stage_config as length_config
from tools.run_stage2_questions import stage_config as frame_config, validate_reused_control
from tools.run_stage3_mechanisms import configuration as mechanism_config
from tools.run_temporal_research import run_command, archive_analysis
from tools.diagnose_stage2_interfaces import save_json, file_hash, write_csv
from utils.dynamic_data import build_dynamic_manifest
from utils.research_storage import check_protocol
from utils.stage2_diagnostics import paired_bootstrap

CONTROLS = ('baseline', 'none', 'temporal_pool', 'learned', 'tubelet1')
VARIANTS = ('combined', 'factorized', 'query')


def configuration(args, name, weights):
    cfg = length_config(args, 16, weights)
    cfg['experiment']['name'] = 'dynamic_' + name
    cfg['experiment']['description'] = 'L16/64-frame matched dynamic refinement; endpoints only'
    cfg['model'].update(memory_mode='spatial', memory_compression='temporal_attention',
                        frame_readout={'combined':'learned', 'factorized':'factorized', 'query':'query'}[name])
    if name == 'factorized':
        cfg['model'].update(dynamic_rank=args.dynamic_rank,
                            dynamic_orthogonal_weight=args.orthogonal_weight)
    cfg['checkpoint'].update(save_initial=False, save_best=False, save_epochs=[args.epochs],
                             save_last_every_n_epochs=args.save_last_every, min_free_gb=args.min_free_gb)
    return cfg


def control_configuration(args, name):
    from types import SimpleNamespace
    control = SimpleNamespace(**vars(args))
    control.epochs = 100
    control.local_frames = 16
    control.smoke = False
    if name in ('learned', 'tubelet1'):
        return frame_config(control, name, Path('unused'))
    return mechanism_config(control, name, Path('unused'))


def checkpoints(args):
    paths = dict(baseline=Path(args.stage1_root)/'ckpt/spatial_l16/epoch_0100.pt',
                 none=Path(args.stage3_root)/'ckpt/none/epoch_0100.pt',
                 temporal_pool=Path(args.stage3_root)/'ckpt/temporal_pool/epoch_0100.pt',
                 learned=Path(args.stage2_root)/'ckpt/learned/epoch_0100.pt',
                 tubelet1=Path(args.stage2_root)/'ckpt/tubelet1/epoch_0100.pt')
    for override in args.checkpoint:
        name, sep, value = override.partition('=')
        if not sep or name not in CONTROLS or not value:
            raise ValueError('--checkpoint must be CONTROL=/absolute/path/epoch_0100.pt')
        paths[name] = Path(value)
    return {n:paths[n] for n in args.controls}


def run_endpoint(command, stage, result, times, output_dir, required_files):
    output_dir = Path(output_dir)
    done = output_dir / 'DONE'
    if done.is_file():
        missing = [name for name in required_files if not (output_dir / name).is_file()]
        if missing:
            raise RuntimeError(f'{stage} has DONE but is missing required outputs: {missing}')
        print(f'\n========== {stage}: reuse completed outputs ==========', flush=True)
        return
    run_command(command, stage, result, times)


@contextmanager
def queue_lock(path):
    """Kernel-owned lock is released on exit/crash; stale text cannot block recovery."""
    with path.open('a+') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not path.stat().st_size:
                handle.write(' ')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('This run_tag already has a running queue') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('This run_tag already has a running queue') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def summarize(result, names, seed):
    overview, completed, comparisons = [], {}, []
    for name in names:
        medical = result/name/'medical'
        if not (medical/'DONE').exists():
            continue
        m = json.loads((medical/'metrics.json').read_text(encoding='utf-8'))
        for row in m['summary']:
            overview.append(dict(model=name, **row))
        with (medical/'patient_metrics.csv').open(encoding='utf-8') as f:
            completed[name] = list(csv.DictReader(f))
    for candidate, control in (('temporal_pool','baseline'), ('learned','baseline'),
                               ('combined','temporal_pool'), ('combined','learned'),
                               ('factorized','combined'), ('query','combined')):
        if candidate not in completed or control not in completed:
            continue
        for field in ('ed_within100ms','es_within100ms','pair_accuracy'):
            a, b = [[r for r in completed[n] if r['branch']=='frame'] for n in (candidate, control)]
            convert = lambda v: float(v == 'True') if v in ('True','False') else float(v)
            if [r['id'] for r in a] != [r['id'] for r in b]:
                raise ValueError('Dynamic comparisons require identical matched cases')
            keep = [i for i in range(len(a)) if a[i][field] and b[i][field]]
            if not keep:
                continue
            pairs = [[dict(r, **{field:convert(r[field])}) for i,r in enumerate(rows) if i in keep] for rows in (a,b)]
            comparisons.append(dict(candidate=candidate,control=control,metric=field,
                                    **paired_bootstrap(*pairs,field,seed)))
    write_csv(result/'dynamic_comparison.csv',overview)
    write_csv(result/'dynamic_paired.csv',comparisons)
    task_rows = []
    for name in names:
        ef, seg = result/name/'ef/metrics.json', result/name/'positions/seg_position_summary.csv'
        if ef.exists():
            m = json.loads(ef.read_text())
            for readout in ('history_0_cache','history_128_cache'):
                task_rows.append(dict(model=name,task='EF',readout=readout,value=m['ef'][readout]['mae'],unit='pp'))
        if seg.exists():
            with seg.open() as f:
                rows = list(csv.DictReader(f))
            if rows and all(r['dice_patient_mean'] for r in rows):
                task_rows.append(dict(model=name,task='seg',readout='all16_positions',
                    value=sum(float(r['dice_patient_mean']) for r in rows)/len(rows),unit='Dice_0_1'))
    write_csv(result/'task_comparison.csv',task_rows)
    lines = ['# Dynamic latent refinement results', '',
             'Matched frozen validation diagnostics. No test-set model selection or automatic success declaration.',
             'Motion-axis sign uses training sparse traced-area labels; filtered event analysis is offline.',
             'Event MAE excludes missed detections; detection rate and all-case within100ms must be read alongside it.',
             'A PCA loop, smoothness or low effective rank alone is not medical correctness.',
             'Two matched new100-epoch treatments by default. Old controls are reused100-epoch models.', '',
             '| Model | Branch | ED within100ms | ES within100ms | Pair identity |',
             '|---|---|---:|---:|---:|']
    for row in overview:
        if row['branch'] == 'frame':
            fmt = lambda v:'NA' if v is None else f'{v:.4f}'
            lines.append(f"| {row['model']} | frame | {fmt(row['ed_within100ms'])} | {fmt(row['es_within100ms'])} | {fmt(row['pair_accuracy'])} |")
    lines += ['', 'See dynamic_comparison.csv for all objects, patient_metrics.csv for misses/collapse/resolution,',
              'within_tubelet_swap.csv for the content-exchange challenge, task_comparison.csv for EF/Dice,',
              'and dynamic_paired.csv for uncertainty. Cross-zero intervals are not a veto or an all-patient requirement.']
    (result/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_tag',default='dynamic_refinement_20261008')
    p.add_argument('--output_root',default='/root/autodl-tmp/outputs_dynamic')
    p.add_argument('--data_root',default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--stage1_root',default='/root/autodl-tmp/outputs_stage1/stage1_lengths_20260928')
    p.add_argument('--stage2_root',default='/root/autodl-tmp/outputs_stage2/stage2_l16_20260929')
    p.add_argument('--stage3_root',default='/root/autodl-tmp/outputs_stage3/stage3_mechanisms_20260929')
    p.add_argument('--init_checkpoint',default='ckpt/mae/videomae_vit_s.pth')
    p.add_argument('--checkpoint',action='append',default=[])
    p.add_argument('--controls',nargs='+',choices=CONTROLS,default=['baseline','none','temporal_pool','learned'])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['combined','factorized'])
    p.add_argument('--phase',choices=['diagnose','full'],default='full')
    for key, value in dict(epochs=100,seed=42,num_workers=8,save_last_every=10,dynamic_rank=16,
                           audit_frames=192,train_cases=128,val_cases=128).items():
        p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--orthogonal_weight',type=float,default=.001)
    p.add_argument('--min_free_gb',type=float,default=3.)
    p.add_argument('--endpoint',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry_run',action='store_true')
    args = p.parse_args()
    if Path(args.run_tag).name != args.run_tag or args.run_tag in ('','.','..') or '\\' in args.run_tag:
        p.error('Invalid run_tag')
    if min(args.epochs,args.save_last_every,args.dynamic_rank,args.train_cases,args.val_cases) < 1 or args.num_workers < 0:
        p.error('Invalid budget')
    if args.audit_frames < 64 or args.audit_frames % 16 or args.orthogonal_weight < 0 or args.min_free_gb < 0:
        p.error('Invalid temporal/rank/storage settings')
    if len(set(args.controls)) != len(args.controls) or len(set(args.variants)) != len(args.variants):
        p.error('Duplicate experiment selection')
    if args.phase == 'full' and args.epochs != 100 and not args.smoke:
        p.error('Matched reused controls are100 epochs; use100 for this bounded round')
    if args.smoke:
        args.epochs,args.train_cases,args.val_cases,args.num_workers = 1,2,2,0
        if args.phase == 'full':
            args.controls = []
        args.run_tag = 'smoke_' + args.run_tag
    sources = checkpoints(args)
    names = list(sources) + (args.variants if args.phase == 'full' else [])
    if not names:
        p.error('No experiments selected')
    run = Path(args.output_root)/args.run_tag
    result, weights = run/'result', run/'ckpt'
    configs = {n:configuration(args,n,weights/n) for n in args.variants} if args.phase == 'full' else {}
    for name in names:
        print(f'{name}: '+('reuse frozen100' if name in sources else f'{args.epochs} epochs + frozen endpoints'),flush=True)
    if args.dry_run:
        return
    for path in [Path(args.data_root)/'FileList.csv',Path(args.data_root)/'VolumeTracings.csv',
                 *sources.values(), *([Path(args.init_checkpoint)] if configs else [])]:
        if not path.is_file():
            raise FileNotFoundError(f'{path}; use --checkpoint NAME=PATH or the root overrides')
    result.mkdir(parents=True,exist_ok=True)
    with queue_lock(run/'queue.lock'):
        reused = {n:validate_reused_control(path,control_configuration(args,n)) for n,path in sources.items()}
        manifest_file = result/'matched_manifest.json'
        manifest = build_dynamic_manifest(args.data_root,args.audit_frames,args.train_cases,args.val_cases,args.seed)
        if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
            raise ValueError('Matched patient/source manifest changed')
        save_json(manifest_file,manifest)
        identity = dict(configs=configs, controls={n:r['sha256'] for n,r in reused.items()},
                        manifest_sha256=file_hash(manifest_file), endpoint=args.endpoint,
                        init_sha256=file_hash(args.init_checkpoint) if configs else None)
        guard = result/'queue_identity.json'
        if guard.exists() and json.loads(guard.read_text()) != identity:
            raise ValueError('Queue scientific protocol changed; use a new run_tag')
        save_json(guard,identity)
        save_json(result/'preserved_components.json',dict(reused=reused,
            recent_cache='Four recent clip descriptors; strongest prior EF evidence',
            learned_expansion='Existing candidate, previously trained separately from temporal_pool',
            temporal_pool='Prior EF-positive compression candidate with segmentation tradeoff',
            combined='New matched training, not a merge of independently trained checkpoints',
            factorized='Per-clip spatial reference + low-rank dynamic residual, not a Functa reproduction'))
        times = []
        if (result/'stage_times.csv').exists():
            with (result/'stage_times.csv').open() as f:
                times = list(csv.DictReader(f))
        for name in names:
            out = result/name
            out.mkdir(exist_ok=True)
            checkpoint = sources.get(name, weights/name/f'epoch_{args.epochs:04d}.pt')
            if name in configs:
                cfg = configs[name]
                if not (out/'PRETRAIN_DONE').exists():
                    digest = check_protocol(out,cfg)
                    (out/'protocol.sha256').write_text(digest+'\n')
                    requested = out/'requested_config.yaml'
                    requested.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
                    cmd = [sys.executable,'trainers/train_rmae.py','--config',str(requested),'--output_dir',str(out)]
                    if not args.smoke:
                        cmd.append('--autotune')
                    run_command(cmd,name+'/pretrain',result,times)
                    if not checkpoint.exists():
                        raise ValueError('Requested final checkpoint was not saved')
                    (out/'PRETRAIN_DONE').write_text(digest+'\n')
            cmd = [sys.executable,'tools/evaluate_dynamic_latent.py','--checkpoint',str(checkpoint),
                   '--output_dir',str(out/'medical'),'--cache_dir',str(run/'cache'/name/'medical'),
                   '--manifest',str(manifest_file),'--seed',str(args.seed),'--num_workers',str(args.num_workers)]
            if args.smoke:
                cmd += ['--batch_size','1','--no-auto_workers','--plot_cases','1','--nuisance_cases','1']
            run_endpoint(cmd,name+'/medical',result,times,out/'medical',
                         ('metrics.json','patient_metrics.csv','summary.csv'))
            if args.endpoint:
                ef = [sys.executable,'tools/evaluate_stage3.py','--checkpoint',str(checkpoint),
                      '--output_dir',str(out/'ef'),'--cache_dir',str(run/'cache'/name/'ef'),
                      '--data_root',args.data_root,'--no-with_seg','--eligible_budget',
                      '--ef_train_cases','512','--ef_val_cases','256','--ef_steps','400',
                      '--seed',str(args.seed),'--num_workers',str(args.num_workers)]
                if args.smoke:
                    ef.append('--smoke')
                run_endpoint(ef,name+'/ef_endpoint',result,times,out/'ef',('metrics.json',))
                seg = [sys.executable,'tools/audit_stage3_streaming.py','--checkpoint',str(checkpoint),
                       '--output_dir',str(out/'positions'),'--cache_dir',str(run/'cache'/name/'positions'),
                       '--data_root',args.data_root,'--train_all_positions','--stream_cases','1' if args.smoke else '2',
                       '--stream_frames','80' if args.smoke else '512','--seed',str(args.seed),
                       '--num_workers',str(args.num_workers)]
                if args.smoke:
                    seg += ['--seg_train_cases','2','--seg_val_cases','2','--seg_steps','2',
                            '--batch_size','1','--no-auto_workers']
                run_endpoint(seg,name+'/position_stream_endpoint',result,times,out/'positions',
                             ('seg_position_summary.csv','stream_checks.json'))
            summarize(result,names,args.seed)
        (result/'DONE').write_text('Bounded dynamic refinement queue completed\n')
        save_json(result/'current_stage.json',dict(stage='all_completed',status='completed'))
        archive_analysis(result)


if __name__ == '__main__':
    main()
