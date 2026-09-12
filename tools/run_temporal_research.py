"""Sequential, resumable server runner for the temporal representation study."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from utils.config import load_config
from tools.evaluate_temporal_mae import write_csv, bootstrap_mean
from utils.research_storage import (check_protocol, prepare_layout, prune_audited_snapshot,
                                    prune_completed_resume, remove_owned_checkpoint)


def experiment_matrix(suite):
    entries = [
        ('echocardmae_video_port', 'official', {}),
        ('videomae_matched', 'matched', {}),
        ('videomae_standard', 'standard', {}),
        ('frame_mae_pool64', 'temporal', dict(local_frames=1, clip_count=64, memory_mode='none', tubelet_size=1)),
        ('frame_rvm64', 'temporal', dict(local_frames=1, clip_count=64, memory_mode='global', tubelet_size=1)),
        ('clip_mae_pool64', 'temporal', dict(local_frames=16, clip_count=4, memory_mode='none')),
    ]
    for mode in ('global', 'spatial', 'dual'):
        entries.append(('hier_' + mode, 'temporal', dict(local_frames=16, clip_count=4, memory_mode=mode)))
    if suite == 'full':
        for k in (4, 8, 16, 32):
            for n in (2, 4, 8):
                if (k, n) == (16, 4):
                    continue
                for mode in ('none', 'global'):
                    entries.append((f'scale_k{k}_n{n}_{mode}', 'temporal',
                                    dict(local_frames=k, clip_count=n, memory_mode=mode)))
        for strategy in ('random_frame', 'temporal_block', 'complementary'):
            for mode in ('none', 'global'):
                entries.append((f'mask_{strategy}_{mode}', 'temporal',
                                dict(local_frames=16, clip_count=4, memory_mode=mode, research_mask=strategy)))
    return entries


def make_config(kind, name, changes, args):
    baseline = 'stage0_echonet_echocardmae_400.yaml' if kind == 'official' else 'stage0_echonet_videomae_matched_400.yaml'
    cfg = load_config(ROOT / 'configs' / 'pretrain' / baseline)
    cfg['experiment'].update(name=name, seed=args.seed, description='Temporal study v1: ' + name)
    cfg['data'].update(data_root=args.data_root, sampling_protocol='temporal_v1',
                       input_protocol=args.input_protocol,
                       num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    cfg['model'].update(init_checkpoint=args.init_checkpoint, gradient_checkpointing=True,
                        require_init_complete_encoder=True)
    cfg['checkpoint'].update(save_initial=True, save_epochs=args.audit_epochs[1:],
                             save_every_n_epochs=0, save_best=False, save_last=True, auto_resume=True,
                             epoch_name_width=4, save_last_every_n_epochs=args.save_last_every,
                             min_free_gb=args.min_free_gb)
    cfg['train'].update(epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
                        plot_interval=5, val_interval=5)
    if kind != 'temporal':
        cfg['train'].update(batch_size=args.baseline_batch_size, grad_accum_steps=1)
    cfg['early_stopping']['enabled'] = False
    if kind == 'standard':
        # Native VideoMAE mask/patch/normalized-pixel target on the same RGB echo input.
        cfg['model'].update(patch_size=16, mask_ratio=.9, norm_pix_loss=True, mask_strategy='tube',
                            target_normalization='videomae')
    if kind == 'temporal':
        cfg['model'].update(name='temporal_mae', baseline_family='temporal_study_v1',
                            sampling_rate=1, two_views=False, local_frames=16, clip_count=4,
                            memory_grid=4, core_depth=1, research_mask='tube', norm_pix_loss=False)
        cfg['model'].update(changes)
        cfg['model']['frames'] = cfg['model']['local_frames'] * cfg['model']['clip_count']
    if args.smoke:
        cfg['data'].update(limit=8, num_workers=0)
        cfg['train'].update(max_steps=2, batch_size=2, grad_accum_steps=1, plot_interval=1, val_interval=1)
    return cfg


def run_command(command, label, root, timings):
    print(f'\n========== {label} ==========\n' + subprocess.list2cmdline(command), flush=True)
    begin = time.perf_counter()
    status = 'failed'
    stage_file = root / 'current_stage.json'
    stage_file.write_text(json.dumps(dict(stage=label, status='running', command=command,
                                         started=datetime.now().isoformat()), indent=2), encoding='utf-8')
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        status = 'completed'
    finally:
        stage_file.write_text(json.dumps(dict(stage=label, status=status,
                                             ended=datetime.now().isoformat()), indent=2), encoding='utf-8')
        timings.append(dict(stage=label, seconds=time.perf_counter()-begin, status=status,
                            ended=datetime.now().isoformat()))
        write_csv(root / 'stage_times.csv', timings)


def summarize(root):
    import csv
    rows = []
    for path in sorted(root.glob('*/audit/epoch_*/metrics.json')):
        m = json.loads(path.read_text(encoding='utf-8'))
        budget = max(map(int, m['ef']))
        probe = m['ef'][str(budget)]
        rows.append(dict(method=path.parents[2].name, epoch=m['metadata']['epoch'],
                         input_protocol=m.get('protocol', {}).get('input_protocol', 'rgb'),
                         ef_train_cases=budget, ef_mae=probe['mae'], ef_rmse=probe['rmse'],
                         ef_mae_ci_low=probe['mae_ci95']['low'], ef_mae_ci_high=probe['mae_ci95']['high'],
                         seg_dice=m['segmentation']['dice_mean'],
                         seg_global=m['segmentation']['dice_global'],
                         state_ef_mae=m.get('state_only_ef',{}).get('mae'),
                         state_seg_dice=m['segmentation'].get('state_only_seg_dice'),
                         state_ed_es=m['segmentation'].get('state_area_derived_ed_es_accuracy'),
                         order_accuracy=m['temporal_order_accuracy'],
                         ed_es_area_proxy=m['segmentation']['area_derived_ed_es_accuracy'],
                         effective_rank=m['effective_rank'], parameters=m['metadata']['parameters'],
                         checkpoint_mb=m['metadata']['checkpoint_mb'],
                         encoder_ms_batch=m['inference']['normal_encoder_ms_per_batch'],
                         encoder_batch_size=m['inference']['batch_size']))
    initial = {r['method']:r for r in rows if r['epoch']==0}
    for row in rows:
        zero = initial.get(row['method'])
        row['ef_mae_improvement_from_epoch0'] = zero['ef_mae']-row['ef_mae'] if zero else None
        row['seg_dice_gain_from_epoch0'] = row['seg_dice']-zero['seg_dice'] if zero else None
    write_csv(root / 'comparison.csv', rows)
    paired_report(root,rows)
    lines = ['# Temporal Representation Study', '',
             'Validation probes, not held-out test performance. Lower EF MAE is better.',
             'Do not rank models by reconstruction loss or feature sensitivity alone.', '',
             '| Method | Epoch | EF MAE | Seg Dice | Order accuracy |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["epoch"]} | {r["ef_mae"]:.4f} | {r["seg_dice"]:.4f} | {r["order_accuracy"]:.4f} |')
    anchors = []
    for path in sorted(root.glob('*/full_finetune/*/logs/metrics.csv')):
        with path.open(encoding='utf-8') as handle:
            records = list(csv.DictReader(handle))
        if not records:
            continue
        task = path.parents[1].name
        best = (min if task == 'echonet_ef' else max)(records,key=lambda row:float(row['monitor_value']))
        anchors.append(dict(method=path.parents[3].name,task=task,best_epoch=best['epoch'],
                            monitor=best['monitor'],best_validation=best['monitor_value'],
                            completed=(path.parents[1]/'DONE').exists()))
    write_csv(root/'full_finetune.csv',anchors)
    if anchors:
        lines += ['', '## Full Fine-tuning Anchors', '',
                  '| Method | Task | Best epoch | Validation metric | Value | Complete |',
                  '|---|---|---:|---|---:|---|']
        for row in anchors:
            lines.append(f'| {row["method"]} | {row["task"]} | {row["best_epoch"]} | {row["monitor"]} | {float(row["best_validation"]):.4f} | {row["completed"]} |')
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for method in sorted({row['method'] for row in rows}):
            values = sorted((r for r in rows if r['method'] == method), key=lambda r: r['epoch'])
            for ax, key in zip(axes, ('ef_mae', 'seg_dice')):
                ax.plot([r['epoch'] for r in values], [r[key] for r in values], marker='.', label=method)
                ax.set_xlabel('Pretraining epoch')
                ax.set_ylabel(key)
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(root / 'representation_trajectory.png', dpi=140)
        plt.close(fig)


def archive_analysis(root):
    allowed = {'.json','.jsonl','.csv','.md','.yaml','.log','.png','.sha256'}
    with zipfile.ZipFile(root/'analysis.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in allowed and 'checkpoints' not in path.parts:
                archive.write(path,path.relative_to(root))


def paired_report(root,rows):
    import csv
    reference = dict(hier_global='clip_mae_pool64',hier_spatial='clip_mae_pool64',
                     hier_dual='clip_mae_pool64',frame_rvm64='frame_mae_pool64',
                     echocardmae_video_port='videomae_matched')
    for row in rows:
        name = row['method']
        if name.startswith(('scale_', 'mask_')) and name.endswith('_global'):
            reference[name] = name[:-len('global')] + 'none'
    comparisons = []
    available = {(r['method'],r['epoch']) for r in rows}
    for method,control in reference.items():
        for epoch in sorted({r['epoch'] for r in rows}):
            if (method,epoch) not in available or (control,epoch) not in available:
                continue
            predictions = []
            for name in (method,control):
                path = root/name/'audit'/f'epoch_{epoch:04d}'/'ef_predictions.csv'
                with path.open(encoding='utf-8') as handle:
                    data = list(csv.DictReader(handle))
                budget = max(int(r['train_cases']) for r in data)
                predictions.append({r['id']:r for r in data if r['mode']=='normal' and int(r['train_cases'])==budget})
            if predictions[0].keys() != predictions[1].keys():
                raise RuntimeError('Paired comparison requires identical validation case IDs.')
            delta = []
            for case,a in predictions[0].items():
                b = predictions[1][case]
                if float(a['target']) != float(b['target']):
                    raise RuntimeError('Paired targets do not match.')
                delta.append(abs(float(a['prediction'])-float(a['target'])) -
                             abs(float(b['prediction'])-float(b['target'])))
            ci = bootstrap_mean(delta)
            comparisons.append(dict(method=method,control=control,epoch=epoch,
                                    mae_difference=ci['mean'],ci95_low=ci['low'],ci95_high=ci['high'],
                                    cases=ci['n'],meaning='negative favors method; unadjusted bootstrap interval'))
    write_csv(root/'paired_comparisons.csv',comparisons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('core', 'full'), default='core')
    parser.add_argument('--run_tag', default=os.environ.get('RUN_TAG', 'temporal_gray_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--output_root', default='/root/autodl-tmp/outputs_temporal')
    parser.add_argument('--data_root', default=None,
                        help='Existing data root. Default: local EchoNet-Dynamic grayscale cache; no files are generated.')
    parser.add_argument('--input_protocol', choices=('gray_repeat3', 'rgb'), default='gray_repeat3',
                        help='gray_repeat3: existing grayscale NPY, expand only sampled clips to three channels.')
    parser.add_argument('--prepare_rgb_cache', action='store_true')
    parser.add_argument('--source_root', default='/root/autodl-fs/datasets/EchoNet-Dynamic')
    parser.add_argument('--init_checkpoint', default='ckpt/mae/videomae_vit_s.pth')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--audit_epochs', type=int, nargs='+', default=[0, 50, 100, 150, 200, 250, 300, 350, 400])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--baseline_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--audit_batch_size', type=int, default=4)
    parser.add_argument('--autotune', action='store_true',
                        help='Calibrate each pretraining model in isolated processes; preserve effective batch.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start_experiment', type=int, default=1)
    parser.add_argument('--only', nargs='+', help='Exact experiment names from --dry_run.')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--no_audit', action='store_true')
    parser.add_argument('--anchors', choices=('none','baselines','core'), default='none',
                        help='Optional full fine-tuning at the final checkpoint; frozen probes remain primary.')
    parser.add_argument('--save_last_every', type=int, default=10,
                        help='Overwrite resumable last.pt every N epochs; also first/final/evaluation epochs.')
    parser.add_argument('--min_free_gb', type=float, default=2,
                        help='Free GiB reserve required in addition to a temporary checkpoint write.')
    parser.add_argument('--keep_stage_checkpoints', action='store_true',
                        help='Keep epoch 0/intermediate snapshots even after successful audits.')
    parser.add_argument('--keep_completed_resume', action='store_true',
                        help='Keep full last.pt/interrupt.pt after an experiment fully completes.')
    parser.add_argument('--migrate_legacy_layout', action='store_true',
                        help='Move this run_tag old mixed layout into result/ and ckpt/. Stop active jobs first.')
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args()
    if args.save_last_every < 1 or args.min_free_gb < 0:
        parser.error('--save_last_every must be positive and --min_free_gb must be nonnegative')
    if args.data_root is None:
        args.data_root = '/root/autodl-tmp/datasets/EchoNet-Dynamic' + ('-rgb' if args.input_protocol == 'rgb' else '')
    if args.prepare_rgb_cache and args.input_protocol != 'rgb':
        parser.error('--prepare_rgb_cache requires --input_protocol rgb. Gray runs reuse existing NPY without caching.')
    if args.prepare_rgb_cache and not args.data_root.endswith('-rgb'):
        args.data_root += '-rgb'
    if args.smoke:
        args.epochs, args.audit_epochs = 1, [0, 1]
        if not args.run_tag.startswith('smoke_'):
            args.run_tag = 'smoke_' + args.run_tag
    args.audit_epochs = sorted({0, args.epochs, *(e for e in args.audit_epochs if 0 <= e <= args.epochs)})
    entries = experiment_matrix(args.suite)
    if args.only:
        unknown = set(args.only) - {name for name, _, _ in entries}
        if unknown:
            raise ValueError(f'Unknown experiments: {unknown}')
    run_root = Path(args.output_root) / args.run_tag
    root, ckpt_root = run_root / 'result', run_root / 'ckpt'
    if args.summarize_only:
        root, ckpt_root = prepare_layout(run_root, [n for n, _, _ in experiment_matrix('full')],
                                         migrate=args.migrate_legacy_layout)
        summarize(root)
        archive_analysis(root)
        return
    configs = [(name, make_config(kind, name, changes, args)) for name, kind, changes in entries]
    for i, (name, cfg) in enumerate(configs, 1):
        if i < args.start_experiment or (args.only and name not in args.only):
            continue
        m = cfg['model']
        print(f'{i:02d} {name:34s} T={m["frames"]:3d} local={m.get("local_frames", m["frames"]):2d} '
              f'memory={m.get("memory_mode", "none"):7s} epochs={args.epochs}')
    selected_count = sum(i >= args.start_experiment and (not args.only or name in args.only)
                         for i, (name, _) in enumerate(configs, 1))
    print(f'Selected experiments={selected_count}/{len(entries)}; result={root}; ckpt={ckpt_root}')
    print(f'Data={args.data_root}; prepare_rgb_cache={args.prepare_rgb_cache}; '
          f'input_protocol={args.input_protocol}; last_every={args.save_last_every}; keep_stages={args.keep_stage_checkpoints}')
    if args.dry_run:
        return
    root, ckpt_root = prepare_layout(run_root, [n for n, _, _ in experiment_matrix('full')],
                                     migrate=args.migrate_legacy_layout)
    # Even --only must not append gray results to a run containing RGB experiments.
    for old_config in root.glob('*/requested_config.yaml'):
        previous = yaml.safe_load(old_config.read_text(encoding='utf-8'))
        if previous.get('data', {}).get('input_protocol', 'rgb') != args.input_protocol:
            raise RuntimeError('Input protocol changed: use a new run_tag instead of mixing RGB and gray experiments.')
    if not Path(args.init_checkpoint).is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if not args.no_audit or args.anchors != 'none':
        try:
            import skimage.draw
        except ImportError as exc:
            raise RuntimeError('Install the official mask dependency before training: python -m pip install scikit-image') from exc
    if args.prepare_rgb_cache:
        subprocess.run([sys.executable,'tools/cache_echonet_npy.py','--input-root',args.source_root,
                        '--output-root',args.data_root,'--rgb','--num-workers',str(max(1,args.num_workers))],
                       cwd=ROOT,check=True)
    for filename in ('FileList.csv', 'VolumeTracings.csv'):
        if not (Path(args.data_root) / filename).is_file():
            raise FileNotFoundError(Path(args.data_root) / filename)
    root.mkdir(parents=True, exist_ok=True)
    import torch
    revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    environment = dict(torch=torch.__version__,python=sys.version,git_commit=revision.stdout.strip(),
                       cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                       argv=sys.argv,timestamp=datetime.now().isoformat())
    with (root/'invocations.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(environment)+'\n')
    from utils.temporal_data import TemporalEchoDataset
    preflight = TemporalEchoDataset(args.data_root,'val',16,channels=3,limit=1,
                                    input_protocol=args.input_protocol)[0]
    (root/'input_contract.json').write_text(json.dumps(dict(
        video_shape=list(preflight['video'].shape),source_path=preflight['source_path'],
        valid_frames=int(preflight['frame_valid'].sum()),data_root=args.data_root,
        input_protocol=args.input_protocol,
        channels_equal=bool(torch.equal(preflight['video'][:,0],preflight['video'][:,1]) and
                            torch.equal(preflight['video'][:,0],preflight['video'][:,2])),
        color_contract=('Grayscale NPY sampled first, then repeated to 3 channels in memory; model normalization unchanged'
                        if args.input_protocol == 'gray_repeat3' else 'RGB NPY or AVI decoded to RGB')),
        indent=2),encoding='utf-8')
    timings = []
    if (root / 'stage_times.csv').exists():
        import csv
        with (root / 'stage_times.csv').open(encoding='utf-8') as handle:
            timings = list(csv.DictReader(handle))
    manifest = dict(arguments=vars(args), experiments=[dict(name=n, config=c) for n, c in configs])
    (root / 'plan.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for i, (name, cfg) in enumerate(configs, 1):
        if i < args.start_experiment or (args.only and name not in args.only):
            continue
        destination = root / name
        destination.mkdir(exist_ok=True)
        checkpoint_dir = ckpt_root / name
        cfg['checkpoint']['dir'] = str(checkpoint_dir)
        digest = check_protocol(destination, cfg)
        identity = destination / 'protocol.sha256'
        identity.write_text(digest + '\n')
        config_path = destination / 'requested_config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        complete = destination / 'PRETRAIN_DONE'
        if not complete.exists():
            cmd = [sys.executable, 'trainers/train_rmae.py', '--config', str(config_path),
                   '--output_dir', str(destination)]
            if args.autotune:
                cmd.append('--autotune')
            run_command(cmd, name + '/pretrain', root, timings)
            final = checkpoint_dir / f'epoch_{args.epochs:04d}.pt'
            if not final.exists():
                raise RuntimeError(f'Incomplete training: {final} missing.')
            complete.write_text(digest + '\n')
        if not args.no_audit:
            for epoch in args.audit_epochs:
                audit = destination / 'audit' / f'epoch_{epoch:04d}'
                if (audit / 'DONE').exists():
                    if not (audit / 'metrics.json').is_file():
                        raise RuntimeError(f'Audit marker without metrics: {audit}')
                    prune_audited_snapshot(checkpoint_dir, destination, epoch, args.epochs,
                                           keep=args.keep_stage_checkpoints)
                    continue
                checkpoint = checkpoint_dir / f'epoch_{epoch:04d}.pt'
                if not checkpoint.is_file():
                    raise FileNotFoundError(f'Missing unaudited snapshot: {checkpoint}. '
                                            'Pruned snapshots cannot be re-evaluated; preserve existing audit results.')
                cmd = [sys.executable, 'tools/evaluate_temporal_mae.py', '--checkpoint', str(checkpoint),
                       '--data_root', args.data_root, '--output_dir', str(audit),
                       '--batch_size', str(args.audit_batch_size), '--num_workers', str(args.num_workers),
                       '--seed', str(args.seed)]
                if args.smoke:
                    cmd.append('--smoke')
                run_command(cmd, name + f'/audit{epoch}', root, timings)
                summarize(root)
                prune_audited_snapshot(checkpoint_dir, destination, epoch, args.epochs,
                                       keep=args.keep_stage_checkpoints)
        anchor_names = {entry[0] for entry in experiment_matrix('core')}
        run_anchor = args.anchors == 'core' and name in anchor_names
        run_anchor = run_anchor or (args.anchors == 'baselines' and name in {'echocardmae_video_port','videomae_matched','videomae_standard'})
        if run_anchor:
            for task in ('echonet_ef','echonet_seg'):
                anchor_dir = destination / 'full_finetune' / task
                anchor_ckpt = checkpoint_dir / 'full_finetune' / task
                if (anchor_dir / 'DONE').exists():
                    if not args.keep_completed_resume and (anchor_ckpt / 'best.pt').is_file():
                        for file in ('last.pt', 'interrupt.pt'):
                            remove_owned_checkpoint(anchor_ckpt / file, checkpoint_dir, anchor_dir,
                                                    'fine-tuning completed; best model retained')
                    continue
                anchor_dir.mkdir(parents=True,exist_ok=True)
                anchor = load_config(ROOT/'configs'/f'finetune_{task}.yaml')
                anchor['model'].update(backbone_checkpoint=str(checkpoint_dir/f'epoch_{args.epochs:04d}.pt'),
                                       frames=cfg['model']['frames'],img_size=112,seg_use_temporal_context=True,
                                       strict_backbone=True)
                anchor['data'].update(data_root=args.data_root,num_workers=args.num_workers,
                                      input_protocol=args.input_protocol)
                anchor['experiment']['seed'] = args.seed
                # Full-token video fine-tuning is much larger than masked MAE.
                anchor['train'].update(batch_size=2,grad_accum_steps=16)
                anchor['checkpoint'].update(auto_resume=True,save_every_n_epochs=0,dir=str(anchor_ckpt),
                                            save_last_every_n_epochs=args.save_last_every,
                                            best_weights_only=True,min_free_gb=args.min_free_gb)
                if args.smoke:
                    anchor['train'].update(epochs=1,max_steps=2,batch_size=2,grad_accum_steps=1)
                anchor_config = anchor_dir/'requested_config.yaml'
                anchor_config.write_text(yaml.safe_dump(anchor,sort_keys=False),encoding='utf-8')
                run_command([sys.executable,'trainers/train_finetune.py','--task',task,
                             '--config',str(anchor_config),'--output_dir',str(anchor_dir)],
                            name+'/'+task,root,timings)
                (anchor_dir/'DONE').write_text('completed\n')
                if not args.keep_completed_resume and (anchor_ckpt / 'best.pt').is_file():
                    for file in ('last.pt', 'interrupt.pt'):
                        remove_owned_checkpoint(anchor_ckpt / file, checkpoint_dir, anchor_dir,
                                                'fine-tuning completed; best model retained')
        prune_completed_resume(checkpoint_dir, destination, args.epochs, args.audit_epochs,
                               keep=args.keep_completed_resume, audited=not args.no_audit)
    summarize(root)
    archive_analysis(root)
    print(f'Completed selected stages. Reports: {root}')


if __name__ == '__main__':
    main()
