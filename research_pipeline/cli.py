"""One CLI for planning, smoke, execution, status, evaluation and export."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .artifacts import ROOT, alive, digest, event, export_results, file_hash, layout, mark_done, run_lock, verify_done, write_json
from .config import load, stage_config


def configure(config):
    # Set the start method before a dataset creates shared epoch counters or CUDA is initialized.
    multiprocessing.set_start_method('spawn', force=True)
    import torch
    torch.set_num_threads(config['runtime']['threads'])
    os.environ.setdefault('OMP_NUM_THREADS', str(config['runtime']['threads']))
    os.environ.setdefault('MKL_NUM_THREADS', str(config['runtime']['threads']))
    device = config['runtime']['device']
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU smoke must explicitly set runtime.device=cpu')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('Pipeline v1 is single-device; do not silently multiply effective batch with torchrun')


def source_path(cfg, stage):
    value = stage.get('source')
    if not value:
        return None
    if value.startswith('@'):
        name, which = value[1:].split('/')
        directory = layout(cfg, name)['result']
        if not verify_done(directory):
            raise ValueError(f'Source stage not completed: {name}')
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        target = 'final.pt' if which == 'final' else summary['best_file']
        info = summary['checkpoints'][target]
        if not Path(info['path']).is_file() or file_hash(info['path']) != info['sha256']:
            raise ValueError('Source checkpoint missing or changed')
        return info['path']
    if not Path(value).is_file():
        raise FileNotFoundError(value)
    return str(Path(value).resolve())


def legacy_audit(cfg, source):
    import torch
    from models.final_temporal_mae import load_final_model
    output = layout(cfg, cfg['stage']['id'])['result']
    output.mkdir(parents=True, exist_ok=True)
    if cfg['adapter'] != 'research_pipeline.echo:build':
        raise ValueError('Existing streaming/mechanism audits are explicitly limited to the reference temporal adapter')
    if not source:
        raise ValueError('Audit requires a trained native checkpoint')
    manifest = cfg['data']['manifest']
    identity = dict(config=cfg, source=file_hash(source), manifest=file_hash(manifest))
    from .artifacts import code_identity
    identity['code'] = code_identity()
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Audit protocol changed')
    write_json(path, identity)
    if verify_done(output):
        return
    device = torch.device(cfg['runtime']['device'])
    write_json(output / 'status.json', dict(pid=os.getpid(), hostname=socket.gethostname(), time=time.time(),
                                            phase=cfg['stage']['kind']+'_audit',
                                            note='Legacy audit progress is in worker.log and audit artifacts; no per-step heartbeat'))
    if cfg['stage']['kind'] == 'streaming':
        from utils.final_temporal_streaming import run_streaming_audit
        report = run_streaming_audit(dict(checkpoint=source, output_dir=str(output / 'audit'),
                                         seed=cfg['training']['seed'], model_overrides=cfg['model'].get('overrides', {}),
                                         smoke=bool(cfg.get('smoke')), **cfg['stage'].get('audit', {})), manifest, device)
    else:
        from utils.final_temporal_mechanisms import run_mechanism_audit
        model, _, _ = load_final_model(source, cfg['model'].get('overrides'), cfg['training']['seed'])
        audit = dict(seed=cfg['training']['seed'], recent_frames=cfg['data'].get('recent_frames', 64),
                     max_prefix=cfg['data'].get('max_prefix', 128))
        audit.update(cfg['stage'].get('audit', {}))
        audit['smoke'] = bool(cfg.get('smoke'))
        report = run_mechanism_audit(model.to(device).eval(), manifest, output / 'audit',
                                     audit, device)
    write_json(output / 'summary.json', dict(stage=cfg['stage']['id'], kind=cfg['stage']['kind'], metrics=report,
                                            scope='Existing bounded temporal audit; not a new full training'))
    mark_done(output, ['protocol.json', 'summary.json'] +
              [str(p.relative_to(output)) for p in sorted((output / 'audit').rglob('*')) if p.is_file()])
    write_json(output / 'status.json', dict(pid=os.getpid(), hostname=socket.gethostname(), time=time.time(), phase='done'))


def work(request):
    from .engine import run_representation, run_training
    cfg = request['config']
    output = layout(cfg, cfg['stage']['id'])['result']
    with run_lock(output):
        try:
            configure(cfg)
            kind = cfg['stage']['kind']
            if kind in {'streaming', 'mechanisms'}:
                return legacy_audit(cfg, request.get('source'))
            if kind == 'representation':
                return run_representation(cfg, request.get('source'))
            return run_training(cfg, request.get('source'))
        except BaseException as exc:
            write_json(output / 'status.json', dict(pid=os.getpid(), hostname=socket.gethostname(),
                                                    time=time.time(), phase='failed', error=repr(exc)))
            raise


def report(config):
    root = layout(config)['result']
    rows = []
    for stage in config['stages']:
        output = layout(config, stage['id'])['result']
        if verify_done(output):
            rows.append(json.loads((output / 'summary.json').read_text(encoding='utf-8')))
    write_json(root / 'comparison.json', rows)
    flat = []
    for row in rows:
        entry = {k:v for k,v in row.items() if isinstance(v, (str, int, float, bool)) or v is None}
        entry.update({k:v for k,v in row['metrics'].items() if isinstance(v, (str, int, float, bool)) or v is None})
        flat.append(entry)
    from .engine import csv_rows
    csv_rows(root / 'comparison.csv', flat)
    lines = ['# Pipeline Run Summary', '',
             'Development evidence only. Different sources, heads, budgets, future access or populations are not a matched ranking.', '',
             '| Stage | Kind | Mode | Epoch | EF MAE (pp) | Patient Dice |', '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        metrics = row['metrics']
        lines.append(f"| {row['stage']} | {row['kind']} | {row.get('mode', '')} | {row.get('epoch', '')} | "
                     f"{metrics.get('mae_pp', '')} | {metrics.get('dice_mean', '')} |")
    if any(row.get('synthetic') for row in rows):
        lines += ['', '**SYNTHETIC FIXTURE: these results are software checks, not research findings.**']
    (root / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def execute(config, only=None):
    root = layout(config)['result']
    root.mkdir(parents=True, exist_ok=True)
    plan_sha = digest(config)
    with run_lock(root):
        path = root / 'plan.json'
        if path.exists() and digest(json.loads(path.read_text(encoding='utf-8'))) != plan_sha:
            raise ValueError('Run plan changed; do not silently resume a different experiment')
        write_json(path, config)
        selected = set(only or [s['id'] for s in config['stages']])
        unknown = selected - {s['id'] for s in config['stages']}
        if unknown:
            raise ValueError(f'Unknown selected stages: {unknown}')
        for stage in config['stages']:
            if stage['id'] not in selected:
                continue
            output = layout(config, stage['id'])['result']
            output.mkdir(parents=True, exist_ok=True)
            source = source_path(config, stage)
            cfg = stage_config(config, stage)
            request = output / 'request.json'
            write_json(request, dict(config=cfg, source=source))
            print(f"========== {stage['id']} ({stage['kind']}/{stage.get('mode', '-')}) ==========", flush=True)
            event(root / 'queue.jsonl', stage=stage['id'], state='starting')
            with (output / 'worker.log').open('a', encoding='utf-8') as log:
                completed = subprocess.run([sys.executable, '-u', '-m', 'research_pipeline', '_worker', '--request', str(request)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            event(root / 'queue.jsonl', stage=stage['id'], state='done' if completed.returncode == 0 else 'failed',
                  returncode=completed.returncode)
            if completed.returncode != 0:
                raise RuntimeError(f"{stage['id']} failed; queue stopped. Inspect {output / 'worker.log'}. "
                                   'No later experiment started; rerun the same command after fixing the cause.')
            if not verify_done(output):
                raise RuntimeError('Worker exited successfully without verified results')
            report(config)
        mark_done(root, ['comparison.json', 'comparison.csv', 'comparison.md']) if len(selected) == len(config['stages']) else None


def status(root, tag):
    directory = Path(root) / 'result' / tag
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    rows = []
    for stage in plan['stages']:
        output = directory / stage['id']
        value = dict(stage=stage['id'], state='pending')
        if (output / 'status.json').exists():
            value.update(json.loads((output / 'status.json').read_text(encoding='utf-8')))
            value['heartbeat_age_seconds'] = round(time.time()-value['time'], 1)
            same_host = value.get('hostname') == socket.gethostname()
            value['process_alive'] = alive(value['pid']) if same_host else None
            value['state'] = 'running' if value['process_alive'] else 'inactive_or_other_host'
            if value['heartbeat_age_seconds'] > 900 and value['state'] == 'running':
                value['state'] = 'needs_inspection_stale_heartbeat_not_proof_of_deadlock'
        if verify_done(output):
            value['state'] = 'done'
        rows.append(value)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'smoke', 'run', 'profile'):
        item = sub.add_parser(command)
        item.add_argument('--config', required=True)
        item.add_argument('--tag')
        item.add_argument('--root')
        item.add_argument('--set', action='append', default=[])
        item.add_argument('--only', nargs='+')
    for command in ('status', 'export', 'report'):
        item = sub.add_parser(command)
        item.add_argument('--root', required=True)
        item.add_argument('--tag', required=True)
        if command == 'export':
            item.add_argument('--destination', required=True)
    item = sub.add_parser('manifest')
    item.add_argument('--data-root', required=True)
    item.add_argument('--output', required=True)
    item.add_argument('--recent-frames', type=int, default=64)
    item.add_argument('--local-frames', type=int, default=16)
    item.add_argument('--max-prefix', type=int, default=128)
    item.add_argument('--smoke', action='store_true')
    item = sub.add_parser('metrics', help='Re-evaluate saved observations with the common definitions')
    item.add_argument('--observations', required=True)
    item.add_argument('--task', choices=['mae', 'ef', 'seg'], required=True)
    item.add_argument('--output', required=True)
    item = sub.add_parser('compare', help='Patient-matched bootstrap; does not select models')
    item.add_argument('--reference', required=True)
    item.add_argument('--candidate', required=True)
    item.add_argument('--field', required=True)
    item.add_argument('--output', required=True)
    item.add_argument('--repetitions', type=int, default=2000)
    item.add_argument('--seed', type=int, default=42)
    for command in ('_worker', '_probe'):
        sub.add_parser(command).add_argument('--request', required=True)
    args = parser.parse_args()
    if args.command in {'plan', 'run', 'smoke', 'profile'}:
        config = load(args.config, args.set, args.tag, args.root, smoke=args.command == 'smoke')
        if args.command == 'plan':
            print(json.dumps(config, ensure_ascii=False, indent=2))
        elif args.command == 'profile':
            if not args.only or len(args.only) != 1:
                raise ValueError('profile requires --only ONE_STAGE_ID')
            stage = next((s for s in config['stages'] if s['id'] == args.only[0]), None)
            if not stage:
                raise ValueError('Unknown profile stage')
            cfg = stage_config(config, stage)
            configure(cfg)
            from .engine import profile_experiment
            with run_lock(layout(config)['result']):
                with run_lock(layout(config, stage['id'])['result']):
                    profile_experiment(cfg, source_path(config, stage), layout(config, stage['id'])['result'] / 'profile')
        else:
            execute(config, args.only)
    elif args.command == 'status':
        print(json.dumps(status(args.root, args.tag), ensure_ascii=False, indent=2))
    elif args.command == 'export':
        print(json.dumps(export_results(Path(args.root) / 'result' / args.tag, args.destination), ensure_ascii=False))
    elif args.command == 'report':
        report(json.loads((Path(args.root) / 'result' / args.tag / 'plan.json').read_text(encoding='utf-8')))
    elif args.command == 'manifest':
        from utils.final_temporal_data import build_manifest
        value = build_manifest(args.data_root, recent_frames=args.recent_frames, local_frames=args.local_frames,
                               max_prefix=args.max_prefix, smoke=args.smoke)
        write_json(args.output, value)
    elif args.command == 'metrics':
        from .metrics import summarize
        summary, patients = summarize(json.loads(Path(args.observations).read_text(encoding='utf-8')), args.task)
        write_json(args.output, dict(metrics=summary, patients=patients))
    elif args.command == 'compare':
        from .metrics import paired_bootstrap
        write_json(args.output, paired_bootstrap(json.loads(Path(args.reference).read_text(encoding='utf-8')),
                                                json.loads(Path(args.candidate).read_text(encoding='utf-8')),
                                                args.field, args.repetitions, args.seed))
    else:
        request = json.loads(Path(args.request).read_text(encoding='utf-8'))
        if args.command == '_worker':
            work(request)
        else:
            from .runtime import probe
            import torch
            try:
                configure(request['config'])
                result = probe(request)
            except torch.cuda.OutOfMemoryError:
                result = dict(status='oom', selection=request['selection'])
            except Exception as exc:
                traceback.print_exc()
                result = dict(status='error', selection=request['selection'], error=repr(exc))
            write_json(request['response'], result)
            if result['status'] == 'error':
                raise SystemExit(1)


if __name__ == '__main__':
    main()
