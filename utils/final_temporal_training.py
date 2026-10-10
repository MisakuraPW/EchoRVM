"""Successful-update-budget adaptation with exact deterministic draw replay."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.final_temporal_mae import load_final_model, CandidateLowRankRVM
from models.temporal_mae import temporal_mask
from utils.final_temporal_data import WarmPlanDataset, WarmBatchSampler, WindowDataset
from utils.checkpoint import atomic_torch_save
from utils.seed import get_rng_state, set_rng_state, seed_everything
from utils.logger import setup_logger
from utils.metrics_logger import MetricsLogger
from utils.plotting import plot_loss_curves


def native_video(video, model, smoke=False):
    if video.shape[-2:] == (model.img_size, model.img_size):
        return video
    if not smoke:
        raise ValueError('Formal training cannot silently resize the native data/model contract')
    shape = video.shape
    import torch.nn.functional as F
    return F.interpolate(video.flatten(0, 1), (model.img_size, model.img_size), mode='bilinear',
                         align_corners=False).reshape(*shape[:3], model.img_size, model.img_size)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def job_protocol(job, manifest):
    scientific = {k:v for k,v in job.items() if k not in {
        'output_dir', 'checkpoint_dir', 'cache_dir', 'num_workers', 'micro_batch', 'autotune', 'device'}}
    return dict(version=2, job=scientific, manifest=digest(manifest), source=file_digest(job['checkpoint']),
                model_code=file_digest(Path(__file__).parents[1] / 'models/final_temporal_mae.py'),
                trainer_code=file_digest(__file__), data_code=file_digest(Path(__file__).with_name('final_temporal_data.py')))


def guard_job(out, protocol, done_files=('metrics.json',)):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Job code/data/checkpoint/scientific protocol changed; use a new run_tag')
    write_json(path, protocol)
    if (out / 'DONE').exists():
        done = json.loads((out / 'DONE').read_text(encoding='utf-8'))
        if any(not (out / name).is_file() or file_digest(out / name) != done['artifacts'].get(name)
               for name in done_files):
            raise ValueError('Completed job has missing or altered outputs')
        return True
    return False


def complete_job(out, files=('metrics.json',)):
    write_json(out / 'DONE', dict(artifacts={name:file_digest(out / name) for name in files}))


def scaler_for(device):
    return torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=128.)


def amp(device):
    return torch.autocast(device.type, enabled=device.type == 'cuda')


def initialize_candidate(model, manifest, job, device):
    if not isinstance(model.memory, CandidateLowRankRVM):
        return None
    # Statistics come from the unrestricted candidate with the same registered background.
    base, _, _ = load_final_model(job['checkpoint'], dict(job.get('overrides', {}), candidate_rank=0), job.get('seed', 42))
    base.to(device).eval()
    samples = []
    hook = base.memory.norm.register_forward_hook(lambda _, __, value: samples.append(value.detach().float().cpu().flatten(0, 1)))
    try:
        dataset = WindowDataset(manifest, 'train', task='mae', recent_frames=job.get('recent_frames', 64),
                                local_frames=model.local_frames, max_prefix=job.get('max_prefix', 128),
                                prefix=None, limit=2 if job.get('smoke') else 32, seed=job.get('seed', 42))
        with torch.no_grad():
            for sample in dataset:
                base(native_video(sample['video'][None].to(device), base, bool(job.get('smoke'))))
        values = torch.cat(samples).double()
        _, _, vt = torch.linalg.svd(values, full_matrices=False)
        rank = model.memory.candidate_down.out_features
        if len(vt) < rank:
            covariance = values.T @ values
            _, basis = torch.linalg.eigh(covariance)
            basis = basis[:, -rank:].flip(1)
        else:
            basis = vt[:rank].T
        with torch.no_grad():
            model.memory.candidate_up.weight.copy_(basis.to(model.memory.candidate_up.weight))
            model.memory.candidate_down.weight.copy_(basis.T.to(model.memory.candidate_down.weight))
        return dict(rank=rank, samples=len(values), centered=False, split='train', method='candidate SVD')
    finally:
        hook.remove()
        del base
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def deterministic_masks(model, batch, seed):
    masks = []
    count = batch['video'].shape[1] // model.local_frames
    device = batch['video'].device
    for index in batch['record_index'].tolist():
        generator = torch.Generator(device=device).manual_seed(seed + int(index) * 1009)
        order = torch.rand(1, model.token_grid[1] * model.token_grid[2], generator=generator, device=device).argsort(-1)
        masks.append(torch.stack([temporal_mask(1, model.token_grid, model.mask_ratio, model.research_mask,
                                               device, i, order)[0] for i in range(count)]))
    return torch.stack(masks)


def tune_warm(model, dataset, out, job, device):
    requested = dict(batch_size=int(job.get('micro_batch') or 2), num_workers=int(job.get('num_workers', 8)),
                     gradient_checkpointing=bool(job.get('overrides', {}).get('gradient_checkpointing', True)))
    if device.type != 'cuda' or not job.get('autotune', True) or job.get('smoke'):
        requested['num_workers'] = 0 if device.type == 'cpu' else requested['num_workers']
        write_json(out / 'runtime.json', dict(selected=requested, hardware=str(device), trials=[]))
        return requested
    path = out / 'runtime.json'
    identity = dict(model=digest(job.get('overrides', {})), gpu=torch.cuda.get_device_name(),
                    effective_batch=dataset.effective_batch, checkpointing=requested['gradient_checkpointing'])
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('identity') != identity:
            raise ValueError('Runtime identity changed; use a new run tag')
        return report['selected']
    initial = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    rng = get_rng_state()
    longest = max(dataset.eligible_cases, key=lambda c:c['frames'])
    start = min(job.get('max_prefix', 128), longest['frames'] - dataset.recent_frames)
    start = start // model.local_frames * model.local_frames
    stress = WindowDataset(dataset.manifest, 'train', task='mae', recent_frames=dataset.recent_frames,
                           local_frames=model.local_frames, max_prefix=job.get('max_prefix', 128),
                           records=[dict(patient=longest['patient'], recent_start=start, H=start)])
    video = native_video(stress[0]['video'][None].to(device), model, bool(job.get('smoke')))
    trials = []
    try:
        for size in (1, 2, 4, 8, 16, 32):
            if size > dataset.effective_batch:
                continue
            optimizer = None
            try:
                model.load_state_dict(initial); set_rng_state(rng)
                model.gradient_checkpointing = requested['gradient_checkpointing']
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = scaler_for(device)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                free_before, _ = torch.cuda.mem_get_info(device)
                baseline = torch.cuda.memory_reserved(device)
                torch.cuda.synchronize(); begin = time.perf_counter()
                successful = 0
                for _ in range(8):
                    optimizer.zero_grad(set_to_none=True)
                    with amp(device):
                        loss = model(video.expand(size, -1, -1, -1, -1))['loss']
                    scaler.scale(loss).backward()
                    before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                    successful += int(scaler.get_scale() >= before)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_reserved()
                reserve = max(512 * 1024**2, int(free_before * .15))
                feasible = peak - baseline + reserve <= free_before and successful == 8
                trials.append(dict(batch_size=size, status='ok' if feasible else 'headroom',
                                   samples_per_second=size * successful / (time.perf_counter() - begin),
                                   peak_reserved_bytes=peak, successful_updates=successful))
            except torch.cuda.OutOfMemoryError:
                trials.append(dict(batch_size=size, status='oom'))
                break
            finally:
                model.zero_grad(set_to_none=True); del optimizer
                loss = scaler = None
                torch.cuda.empty_cache()
        valid = [row for row in trials if row['status'] == 'ok']
        if not valid:
            raise RuntimeError('No batch fits with CUDA headroom')
        speed = max(row['samples_per_second'] for row in valid)
        selected = min((row for row in valid if row['samples_per_second'] >= speed * .97),
                       key=lambda row:row['peak_reserved_bytes'])
        requested['batch_size'] = selected['batch_size']
        workers = []
        for count in sorted({0, min(4, requested['num_workers']), requested['num_workers']}):
            options = dict(batch_sampler=WarmBatchSampler(dataset, requested['batch_size']), num_workers=count,
                           pin_memory=True)
            if count:
                options.update(persistent_workers=True, prefetch_factor=4)
            loader = DataLoader(dataset, **options)
            iterator = iter(loader)
            for _ in range(4):
                if next(iterator, None) is None:
                    break
            begin, n = time.perf_counter(), 0
            for _ in range(20):
                batch = next(iterator, None)
                if batch is None:
                    break
                n += len(batch['video'])
            workers.append(dict(num_workers=count, samples_per_second=n / (time.perf_counter() - begin)))
            del iterator, loader
        requested['num_workers'] = max(workers, key=lambda r:r['samples_per_second'])['num_workers']
        write_json(path, dict(identity=identity, selected=requested, trials=trials, worker_trials=workers,
                             note='Real-window backward stress benchmark; no trial weights retained; common checkpointing fixed'))
        return requested
    finally:
        model.load_state_dict(initial); set_rng_state(rng); model.zero_grad(set_to_none=True)
        del initial


def run_warm_job(job, manifest, device):
    device = torch.device(device)
    out, weights = Path(job['output_dir']), Path(job['checkpoint_dir'])
    protocol = job_protocol(job, manifest)
    if guard_job(out, protocol):
        if not (weights / 'final.pt').exists():
            raise FileNotFoundError('Completed adaptation has no final checkpoint')
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    weights.mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    logger = setup_logger(out / 'logs/train.log')
    seed = int(job.get('seed', 42)); seed_everything(seed)
    model, config, loaded = load_final_model(job['checkpoint'], job.get('overrides'), seed)
    model.to(device)
    updates, effective = int(job.get('updates', 1500)), int(job.get('effective_batch', 32))
    dataset = WarmPlanDataset(manifest, updates, effective, job.get('recent_frames', 64), model.local_frames,
                              job.get('max_prefix', 128), seed)
    last = weights / 'last.pt'
    pca = initialize_candidate(model, manifest, job, device) if isinstance(model.memory, CandidateLowRankRVM) and not last.exists() else None
    if not (out / 'initialization.json').exists():
        write_json(out / 'initialization.json', dict(**loaded, candidate_initialization=pca,
                                                    initialized_sha=model_digest(model)))
    runtime = tune_warm(model, dataset, out, job, device)
    model.gradient_checkpointing = runtime['gradient_checkpointing']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05, betas=(.9, .95))
    scaler = scaler_for(device)
    cursor = 0
    history, update_times, prior_wall = {}, [], 0.
    coverage = dict(samples=0, observed_frames=0, prefix_frames=0,
                    recent_reconstruction_frames=0, prefix_seconds=0., observed_seconds=0.)
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['protocol'] != protocol:
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(saved['model_state_dict']); optimizer.load_state_dict(saved['optimizer_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict']); set_rng_state(saved['rng_state'])
        cursor = int(saved['cursor'])
        history = saved.get('prefix_counts', {})
        update_times = saved.get('update_times', [])
        prior_wall = float(saved.get('wall_seconds', 0))
        coverage.update(saved.get('training_coverage', {}))
        logger.info('resumed successful_update=%d', cursor)
    config['model'].update(gradient_checkpointing=model.gradient_checkpointing)
    config['data'] = dict(input_protocol='gray_repeat3', sampling_protocol='final_variable_prefix_v2')
    config['train'] = dict(updates=updates, effective_batch=effective, prefix_max=job.get('max_prefix', 128))
    options = dict(batch_sampler=WarmBatchSampler(dataset, runtime['batch_size'], cursor),
                   num_workers=runtime['num_workers'], pin_memory=device.type == 'cuda',
                   generator=torch.Generator().manual_seed(seed + 801))
    if runtime['num_workers']:
        options.update(persistent_workers=True, prefetch_factor=4)
    loader = iter(DataLoader(dataset, **options))
    metrics = MetricsLogger(out / 'logs')
    log = out / 'logs/train_metrics.jsonl'
    if log.exists():
        retained = []
        for line in log.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row['step']) <= cursor:
                retained.append(row)
        log.write_text(''.join(json.dumps(row) + '\n' for row in retained), encoding='utf-8')
        metrics.csv_path.unlink(missing_ok=True)
        for row in retained:
            metrics.update_csv(row)
    start = time.perf_counter(); progress = tqdm(total=updates, initial=cursor, desc='MAE adaptation')
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / 'tensorboard'), purge_step=cursor + 1)
    except ImportError:
        logger.warning('TensorBoard unavailable; JSONL, CSV and PNG remain enabled')
    def save():
        atomic_torch_save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                              scaler_state_dict=scaler.state_dict(), rng_state=get_rng_state(), config=config,
                              cursor=cursor, global_step=cursor, epoch=0, protocol=protocol,
                              prefix_counts=history, update_times=update_times,
                              training_coverage=coverage,
                              wall_seconds=prior_wall + time.perf_counter() - start), last,
                          min_free_gb=float(job.get('min_free_gb', 5)))
    try:
        model.train()
        if not last.exists():
            save()
        while cursor < updates:
            begin = time.perf_counter(); batches, count = [], 0
            while count < effective:
                batch = next(loader)
                if not bool((batch['update_index'] == cursor).all()):
                    raise ValueError('Warm sample cursor mismatch')
                batches.append(batch); count += len(batch['video'])
            if count != effective:
                raise ValueError('Ragged update did not preserve effective sample count')
            data_time = time.perf_counter() - begin
            retry = 0
            while True:
                optimizer.zero_grad(set_to_none=True); loss_value = recon_value = orth_value = 0.
                ftime = btime = 0.
                for batch in batches:
                    batch = dict(batch, video=native_video(batch['video'].to(device, non_blocking=True), model, bool(job.get('smoke'))))
                    tick = time.perf_counter()
                    with amp(device):
                        result = model(batch['video'], masks=deterministic_masks(model, batch, seed))
                        loss = result['loss'] * (len(batch['video']) / effective)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite MAE loss')
                    ftime += time.perf_counter() - tick; tick = time.perf_counter()
                    scaler.scale(loss).backward(); btime += time.perf_counter() - tick
                    loss_value += float(loss.detach())
                    recon_value += float(result['loss_recon']) * len(batch['video']) / effective
                    orth_value += float(result['loss_dynamic_orthogonal']) * model.dynamic_orthogonal_weight * len(batch['video']) / effective
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not scaler.is_enabled() and not torch.isfinite(grad_norm):
                    raise FloatingPointError('Nonfinite MAE gradient')
                before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= before:
                    break
                retry += 1
                if retry > 8:
                    raise FloatingPointError('Repeated AMP overflows; no successful update')
            cursor += 1
            for batch in batches:
                for h, fps in zip(batch['prefix_frames'].tolist(), batch['fps'].tolist()):
                    history[str(h)] = history.get(str(h), 0) + 1
                    coverage['samples'] += 1
                    coverage['prefix_frames'] += h
                    coverage['observed_frames'] += h + dataset.recent_frames
                    coverage['recent_reconstruction_frames'] += dataset.recent_frames
                    coverage['prefix_seconds'] += h / fps
                    coverage['observed_seconds'] += (h + dataset.recent_frames) / fps
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin; update_times.append(elapsed)
            allocated = torch.cuda.memory_allocated() / 2**30 if device.type == 'cuda' else 0.
            row = dict(epoch=cursor, step=cursor, train_loss=loss_value, val_loss=None, data_time=data_time,
                       reconstruction_loss=recon_value, weighted_orthogonal_loss=orth_value,
                       forward_time=ftime, backward_time=btime, step_time=elapsed, gpu_mem_allocated=allocated,
                       lr=1e-4, successful_update=True, retries=retry)
            metrics.write_jsonl('train_metrics.jsonl', row); metrics.update_csv(row)
            if writer:
                for name, value in (('loss', loss_value), ('reconstruction', recon_value), ('weighted_orthogonal', orth_value)):
                    writer.add_scalar('train/' + name, value, cursor)
            progress.update(1); progress.set_postfix(loss=f'{loss_value:.4f}', data=f'{data_time:.3f}',
                                                   step=f'{elapsed:.3f}', mem=f'{allocated:.1f}G', batch=runtime['batch_size'])
            if cursor % 20 == 0:
                logger.info('update=%d loss=%.6f data=%.3f step=%.3f memory=%.2f', cursor, loss_value, data_time, elapsed, allocated)
                plot_loss_curves(out / 'logs/metrics.csv', out / 'plots/loss_latest.png')
            if cursor % int(job.get('save_every_updates', 300)) == 0 or cursor == updates:
                save(); write_json(out / 'status.json', dict(status='training', successful_updates=cursor, total=updates))
        atomic_torch_save(dict(model_state_dict=model.state_dict(), config=config, epoch=0, global_step=cursor,
                              final_adaptation=True, protocol=protocol), weights / 'final.pt', float(job.get('min_free_gb', 5)))
        final = dict(successful_updates=cursor, effective_batch=effective, prefix_sample_counts=history,
                     training_coverage=dict(coverage,
                         duration_convention='frame count / source FPS, summed over successful draws; not unique video duration'),
                     runtime=runtime, wall_seconds=prior_wall + time.perf_counter() - start,
                     optimizer_update_seconds_median=float(np.median(update_times)) if update_times else None,
                     max_training_prefix=job.get('max_prefix', 128), memory_slots=model.memory_slots,
                     final_checkpoint=str(weights / 'final.pt'), source=job['checkpoint'])
        write_json(out / 'metrics.json', final)
        plot_loss_curves(out / 'logs/metrics.csv', out / 'plots/loss_latest.png')
        complete_job(out, files=('metrics.json', 'logs/metrics.csv', 'plots/loss_latest.png', 'initialization.json', 'runtime.json'))
        done = json.loads((out / 'DONE').read_text(encoding='utf-8'))
        done['checkpoints'] = {name:file_digest(weights / name) for name in ('last.pt', 'final.pt')}
        write_json(out / 'DONE', done)
        return final
    except BaseException:
        # Partial gradients/updates are discarded; only the durable success boundary resumes.
        write_json(out / 'interrupt.json', dict(last_checkpoint=str(last), next_safe_update='from checkpoint cursor',
                                               observed_update=cursor, note='partial update not serialized'))
        raise
    finally:
        progress.close()
        if writer:
            writer.close()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
