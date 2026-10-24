"""Measured execution tuning in disposable processes, never LR/budget search."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Sampler

from .artifacts import ROOT, digest, write_json
from .contracts import collate_samples, worker_init


class UpdateBatchSampler(Sampler):
    def __init__(self, count, effective, micro, seed, epoch=0, shuffle=True):
        if count < 1 or effective < 1 or micro < 1 or effective % micro:
            raise ValueError('Micro batch must divide effective batch; dataset cannot be empty')
        self.count, self.effective, self.micro = count, effective, micro
        self.seed, self.epoch, self.shuffle = seed, epoch, shuffle

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.count, generator=generator).tolist() if self.shuffle else list(range(self.count))
        for start in range(0, self.count, self.effective):
            group = order[start:start+self.effective]
            for offset in range(0, len(group), self.micro):
                yield group[offset:offset+self.micro]

    def __len__(self):
        full, tail = divmod(self.count, self.effective)
        return full * (self.effective // self.micro) + math.ceil(tail / self.micro)


def loader(dataset, cfg, selection, epoch=0, training=True):
    run, train = cfg['runtime'], cfg['training']
    workers = int(selection['num_workers'])
    options = dict(batch_sampler=UpdateBatchSampler(len(dataset), train['effective_batch'], selection['micro_batch'],
                                                   train['seed'], epoch, training),
                   num_workers=workers, pin_memory=str(run['device']).startswith('cuda'),
                   collate_fn=collate_samples, worker_init_fn=worker_init,
                   generator=torch.Generator().manual_seed(train['seed']+epoch))
    if workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True,
                       prefetch_factor=run['prefetch_factor'], timeout=run['loader_timeout_seconds'])
    return DataLoader(dataset, **options)


def hardware(device):
    result = dict(torch=str(torch.__version__), device=str(device), cpus=os.cpu_count(),
                  threads=torch.get_num_threads(), cuda=torch.version.cuda)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        result.update(name=torch.cuda.get_device_name(device), total_bytes=total, free_bytes=free,
                      capability=list(torch.cuda.get_device_capability(device)))
    return result


def choose(trials):
    valid = [row for row in trials if row.get('status') == 'ok']
    if not valid:
        raise RuntimeError('No safe runtime trial succeeded; inspect runtime/trials and worker logs')
    fastest = max(row['samples_per_second'] for row in valid)
    return min((row for row in valid if row['samples_per_second'] >= .97*fastest),
               key=lambda row: (row['peak_reserved_bytes'], row['selection']['num_workers']))


def run_probe_process(command, log, timeout):
    options = dict(cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options['start_new_session'] = True
    process = subprocess.Popen(command, **options)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Only the disposable probe's owned process group, including its readers.
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        return None


def select_runtime(cfg, experiment, directory, source, identity, heartbeat):
    run, train = cfg['runtime'], cfg['training']
    device = torch.device(run['device'])
    effective = train['effective_batch']
    invariant = experiment.microbatch_safe and not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                                      for m in experiment.model.modules())
    identity = digest(dict(protocol=identity, hardware={k:v for k,v in hardware(device).items() if k != 'free_bytes'}))
    path = directory / 'runtime.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Resume hardware/runtime identity changed; explicit new run required')
        return saved['selected']
    requested = dict(micro_batch=min(run['micro_batch'], effective), num_workers=0, gradient_checkpointing=False)
    if effective % requested['micro_batch']:
        raise ValueError('Requested micro_batch must divide effective_batch')
    if not run['autotune'] or device.type != 'cuda':
        requested['num_workers'] = min(run['worker_candidates'])
        write_json(path, dict(identity=identity, hardware=hardware(device), selected=requested, trials=[],
                             note='Explicit runtime or CPU path; no GPU tuning claim'))
        return requested
    # Parent must not retain a CUDA copy while probes measure the same device.
    experiment.model.cpu()
    torch.cuda.empty_cache()
    trial_dir = directory / 'runtime'
    trial_dir.mkdir(parents=True, exist_ok=True)
    trials = []

    def attempt(selection):
        number = len(trials)
        request, response = trial_dir / f'{number:03d}.request.json', trial_dir / f'{number:03d}.response.json'
        write_json(request, dict(config=cfg, source=source, selection=selection, response=str(response)))
        heartbeat(phase='autotune', trial=number, selection=selection)
        with (trial_dir / f'{number:03d}.log').open('w', encoding='utf-8') as log:
            returncode = run_probe_process([sys.executable, '-m', 'research_pipeline', '_probe', '--request', str(request)],
                                           log, run['probe_timeout_seconds'])
            if returncode is None:
                row = dict(status='timeout', selection=selection)
            else:
                if not response.exists():
                    raise RuntimeError(f'Probe exited {returncode} without a result: {trial_dir / f"{number:03d}.log"}')
                row = json.loads(response.read_text(encoding='utf-8'))
                if row['status'] == 'error':
                    raise RuntimeError(f'Non-OOM probe failure: {row}; see trial log')
        trials.append(row)
        write_json(trial_dir / 'trials.json', trials)

    batches = sorted({int(b) for b in run['batch_candidates'] if b <= effective and effective % b == 0}) if invariant else [requested['micro_batch']]
    if not batches:
        raise ValueError('No candidate batch divides the fixed effective batch')
    checkpoints = run['checkpoint_candidates'] if hasattr(experiment.model, 'gradient_checkpointing') or experiment.metadata.get('checkpointing') else [False]
    for checkpointing in checkpoints:
        for batch in batches:
            attempt(dict(micro_batch=batch, num_workers=0, gradient_checkpointing=checkpointing))
            if trials[-1]['status'] == 'oom':
                break
    best = choose(trials)['selection']
    for workers in sorted(set(run['worker_candidates']) - {0}):
        attempt(dict(best, num_workers=workers))
    selected = choose(trials)['selection']
    write_json(path, dict(identity=identity, hardware=hardware(device), selected=selected, trials=trials,
                         microbatch_safe=invariant, objective='end-to-end measured samples/s, 3% tie favors headroom',
                         note='No LR, effective batch, objective, data or epoch search; floating point/dropout need not be bit-identical'))
    return selected


def probe(request):
    from .engine import build_experiment, prepare_features, amp_context, make_scaler, update_step
    from utils.seed import seed_everything
    cfg, selection = request['config'], request['selection']
    device = torch.device(cfg['runtime']['device'])
    seed_everything(cfg['training']['seed'])
    exp = build_experiment(cfg, request.get('source'), device)
    prepare_features(exp, cfg, request.get('source'), lambda **kw: None)
    exp.model.to(device)
    exp.set_checkpointing(selection['gradient_checkpointing'])
    exp.train_mode()
    optimizer = torch.optim.AdamW([p for p in exp.model.parameters() if p.requires_grad], lr=cfg['training']['lr'],
                                 weight_decay=cfg['training']['weight_decay'], betas=tuple(cfg['training']['betas']))
    scaler = make_scaler(cfg, device)
    data = loader(exp.train, cfg, selection)
    iterator = iter(data)
    effective = cfg['training']['effective_batch']
    free, _ = torch.cuda.mem_get_info(device)
    baseline = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    stress = exp.calibration_samples(selection['micro_batch'])
    optimizer.zero_grad(set_to_none=True)
    with amp_context(cfg, device):
        stress_loss = exp.loss(stress)
    scaler.scale(stress_loss).backward()
    if not update_step(exp.model, optimizer, scaler, cfg['training']['clip_norm']):
        raise FloatingPointError('AMP failed the maximum-context stress update')
    optimizer.zero_grad(set_to_none=True)
    del stress_loss
    warmup, measured = cfg['runtime']['probe_warmup_updates'], cfg['runtime']['probe_updates']
    begin, samples, successes = None, 0, 0
    for update in range(warmup + measured):
        if update == warmup:
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        used = 0
        chunks = []
        while used < effective:
            batch = next(iterator, None)
            if batch is None:
                iterator = iter(data)
                batch = next(iterator)
            # Probe uses a full effective update, including tiny smoke populations.
            batch = batch[:effective-used]
            chunks.append(batch)
            used += len(batch)
        for batch in chunks:
            with amp_context(cfg, device):
                loss = exp.loss(batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite probe loss')
            scaler.scale(loss * len(batch) / used).backward()
        success = update_step(exp.model, optimizer, scaler, cfg['training']['clip_norm'])
        if update >= warmup:
            samples += used
            successes += int(success)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter()-begin
    peak = torch.cuda.max_memory_reserved(device)
    reserve = max(cfg['runtime']['memory_headroom_gb']*1024**3, free*cfg['runtime']['memory_headroom_fraction'])
    safe = peak-baseline+reserve <= free and successes == measured
    return dict(status='ok' if safe else 'headroom_or_skipped_update', selection=selection,
                samples_per_second=samples/seconds, seconds=seconds, peak_reserved_bytes=peak,
                successful_updates=successes, samples=samples,
                stress_context_frames=len(stress[0]['video']) if 'video' in stress[0] else None)
