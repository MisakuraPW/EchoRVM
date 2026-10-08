"""Isolated, bounded real-data throughput probes for MAE pretraining."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from utils.autotune import apply_selection, batch_candidates, choose_trial, fingerprint


def hardware():
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, total_memory=p.total_memory, torch=torch.__version__, cuda=torch.version.cuda)


def calibrate(config, run_dir):
    if not torch.cuda.is_available():
        print('[autotune] CPU run: unchanged settings', flush=True)
        return config
    if config['train'].get('torch_compile', False):
        raise ValueError('Autotune does not support torch_compile; disable it explicitly before calibrating.')
    folder = Path(run_dir) / 'autotune'
    identity = dict(config=fingerprint(config), hardware=hardware(), version=2,
                    warm_updates=3, measured_windows=8)
    report_path = folder / 'runtime.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('identity') == identity:
            print('[autotune] reusing ' + str(report['selected']), flush=True)
            return apply_selection(config, report['selected'])
        if report.get('identity', {}).get('version') == 1:
            # Keep prior measurements intact while allowing the corrected search
            # to run in the same resumed output directory.
            import datetime
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            preserved = folder.with_name(folder.name + '_v1_preserved_' + stamp)
            shutil.move(str(folder), str(preserved))
            print('[autotune] preserved old v1 measurements at ' + str(preserved), flush=True)
        else:
            raise RuntimeError('Autotune cache/config/hardware changed. Preserve this report and use a new output directory.')
    folder.mkdir(parents=True, exist_ok=True)
    trials = []
    base = dict(batch_size=int(config['train']['batch_size']),
                gradient_checkpointing=bool(config['model'].get('gradient_checkpointing', False)),
                num_workers=int(config['data'].get('num_workers', 0)))

    def attempt(selection):
        for previous in trials:
            if all(previous[k] == v for k, v in selection.items()):
                return previous
        index = len(trials)
        request = folder / f'trial_{index:02d}.json'
        result = folder / f'trial_{index:02d}_result.json'
        request.write_text(json.dumps(dict(config=config, selection=selection,
                                           warm_updates=3, measured_windows=8)), encoding='utf-8')
        print('[autotune] probe ' + str(selection), flush=True)
        with (folder / f'trial_{index:02d}.log').open('w', encoding='utf-8') as log:
            try:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                          '--trial', str(request), '--result', str(result)],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=os.name != 'nt',
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise RuntimeError('Autotune trial timed out; inspect trial log and worker processes before retrying.')
        if child.returncode != 0 or not result.exists():
            raise RuntimeError(f'Autotune probe failed (not a handled OOM); inspect {log.name}')
        measured = json.loads(result.read_text())
        trials.append(measured)
        print('[autotune] ' + json.dumps(measured), flush=True)
        return measured

    attempt(base)
    candidates = batch_candidates(config)
    # Try growing first. Only search smaller batches if the original one failed.
    search = [b for b in candidates if b > base['batch_size']]
    if trials[0]['status'] != 'ok':
        search = sorted((b for b in candidates if b < base['batch_size']), reverse=True)
    for b in search:
        row = attempt(dict(base, batch_size=b))
        if row['status'] != 'ok' and b > base['batch_size']:
            break
    # Compare checkpointing jointly with worker counts for the two fastest
    # safe microbatches. Worker count affects input overlap, while checkpointing
    # changes GPU compute and memory; tuning them independently can miss the
    # best combination.
    finalist_batches = []
    for row in sorted((r for r in trials if r.get('status') == 'ok'),
                      key=lambda r: r['samples_per_second'], reverse=True):
        if row['batch_size'] not in finalist_batches:
            finalist_batches.append(row['batch_size'])
        if len(finalist_batches) == 2:
            break
    if not finalist_batches:
        raise RuntimeError('No safe batch/checkpointing candidate; inspect autotune/trials.')
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    worker_candidates = sorted({0, min(cpus, base['num_workers']),
                                min(cpus, max(1, base['num_workers'] // 2))})
    for batch in finalist_batches:
        for checkpointing in (False, True):
            for workers in worker_candidates:
                attempt(dict(base, batch_size=batch,
                             gradient_checkpointing=checkpointing,
                             num_workers=workers))
    best = choose_trial(trials)
    selected = {k: best[k] for k in base}
    report = dict(identity=identity, selected=selected, trials=trials,
                  effective_batch=int(config['train']['batch_size']) * int(config['train'].get('grad_accum_steps', 1)),
                  note='Finite throughput search, not guaranteed global optimum. No probe weights are retained.')
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(report_path)
    print('[autotune] selected ' + str(selected) + '; report=' + str(report_path), flush=True)
    return apply_selection(config, selected)


def probe(request):
    from trainers.train_rmae import build_training_model, build_loader, move_batch, TemporalMAE
    from optim import build_optimizer
    from utils.seed import seed_everything
    from utils.pretrained_init import load_videomae_init

    selection = request['selection']
    cfg = apply_selection(request['config'], selection)
    seed_everything(int(cfg.get('experiment', {}).get('seed', 42)))
    torch.set_float32_matmul_precision(str(cfg['train'].get('matmul_precision', 'high')))
    device = torch.device('cuda')
    model = build_training_model(cfg['model'], device)
    if cfg['model'].get('init_checkpoint'):
        load_videomae_init(model, cfg['model']['init_checkpoint'], map_location='cpu')
    model.train()
    optimizer, _ = build_optimizer(model, cfg.get('optimizer', {}))
    amp = bool(cfg['train'].get('mixed_precision', True))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loader = build_loader(cfg, 'train', None)
    if not len(loader):
        raise ValueError('Empty probe loader')
    iterator = iter(loader)
    accum = int(cfg['train']['grad_accum_steps'])
    samples, elapsed, wait_time, updates = 0, 0., 0., 0
    warm_updates, measured_windows = 0, 0
    torch.cuda.reset_peak_memory_stats()
    # Real forward/backward/optimizer steps allocate lazy optimizer state.
    # Warmup is excluded, and each timed window ends with CUDA synchronization.
    warm_target = int(request.get('warm_updates', 3))
    measured_target = int(request.get('measured_windows', 8))
    for window in range(warm_target + measured_target + 8):
        measuring = warm_updates >= warm_target
        torch.cuda.synchronize()
        begin = time.perf_counter()
        n, data_seconds = 0, 0.
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            data_seconds += time.perf_counter() - start
            batch = move_batch(batch, device)
            n += batch['video'].shape[0]
            with torch.amp.autocast('cuda', enabled=amp):
                if 'video_view2' in batch:
                    out = model(batch['video'], video_view2=batch['video_view2'])
                elif isinstance(model, TemporalMAE):
                    out = model(batch['video'], frame_valid=batch.get('frame_valid'))
                else:
                    out = model(batch['video'])
                loss = out['loss'] / accum
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite calibration loss')
            scaler.scale(loss).backward()
        clip = cfg['train'].get('clip_grad_norm')
        if clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        old = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        successful = scaler.get_scale() >= old
        if measuring:
            samples += n
            elapsed += time.perf_counter() - begin
            wait_time += data_seconds
            updates += int(successful)
            measured_windows += 1
            if measured_windows == measured_target:
                break
        else:
            warm_updates += int(successful)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    # Account for memory held by the parent/other processes, not just our tensors.
    external = max(0, total - free - torch.cuda.memory_reserved())
    limit = min(total * .85, total - 2 * 1024**3)
    status = 'ok' if reserved + external <= limit and updates == measured_target else 'unsafe'
    return dict(selection, status=status, samples_per_second=samples / max(elapsed, 1e-9),
                data_wait_fraction=wait_time / max(elapsed, 1e-9), peak_allocated_bytes=allocated,
                peak_reserved_bytes=reserved, external_bytes=external, successful_updates=updates,
                warm_updates=warm_target, measured_windows=measured_windows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.trial).read_text())
    try:
        result = probe(request)
    except torch.cuda.OutOfMemoryError:
        result = dict(request['selection'], status='oom')
    Path(args.result).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
