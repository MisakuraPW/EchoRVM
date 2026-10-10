"""Matched variable-history capability fits and fixed-head dependence curves."""

from __future__ import annotations

from collections import defaultdict
import copy
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.final_temporal_mae import load_final_model
from models.ef_readout import TemporalEFReadout
from utils.final_temporal_data import WindowDataset
from utils.final_temporal_execution import encode_window, task_features
from utils.final_temporal_training import write_json, guard_job, complete_job, job_protocol, native_video, file_digest
from utils.final_temporal_tasks import load_frozen_task_head
from utils.checkpoint import atomic_torch_save


def rows_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)


def paired_delta(a, b, field, seed=42, repetitions=2000):
    if repetitions < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    if a and b and 'recent_start' in a[0] and 'recent_start' in b[0]:
        def keys(rows):
            values = [(r['patient'], r['recent_start'], r.get('source_frame', -1),
                       r.get('position', -1), r.get('source_path')) for r in rows]
            if len(set(values)) != len(values):
                raise ValueError('Duplicate matched window/source/position')
            return set(values)
        if keys(a) != keys(b):
            raise ValueError('Window/source/position pairing differs; never silently intersect')
    def aggregate(rows):
        sources = defaultdict(list)
        for row in rows:
            value = float(row[field])
            if not np.isfinite(value):
                raise ValueError('Nonfinite paired observation')
            sources[(row['patient'], row.get('source_frame', -1))].append(value)
        grouped = defaultdict(list)
        for (patient, _), values in sources.items():
            grouped[patient].append(float(np.mean(values)))
        return {key:float(np.mean(value)) for key,value in grouped.items()}
    x, y = aggregate(a), aggregate(b)
    if set(x) != set(y) or not x:
        raise ValueError('Patient pairing differs; never silently intersect a different population')
    delta = np.array([x[k] - y[k] for k in sorted(x)])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repetitions)])
    return dict(delta=float(delta.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), patients=len(delta))


def history_records(manifest, split, prefix, recent=64, limit=None):
    if prefix < 0 or recent < 1 or (limit is not None and limit < 0):
        raise ValueError('Invalid real history/window/case limit')
    cases = [case for case in manifest[split] if case['ef'] is not None and case['frames'] >= prefix + recent]
    if limit is not None:
        cases = cases[:limit]
    # The chosen recent window is independent of the requested prefix length.
    return [dict(patient=case['patient'], recent_start=case['frames'] - recent, H=prefix,
                 prefix_kind='clean') for case in cases]


def seg_history_records(manifest, split, prefix, recent=64, local=16, limit=None):
    """Original labels complete at ALL last-clip positions, including real H.

    Limit patients before expanding labels/positions. The source label and
    recent_start=source-position remain identical in real/H0/repeat views.
    """
    if (local < 1 or recent < local or recent % local or prefix < 0 or prefix % local
            or (limit is not None and limit < 0)):
        raise ValueError('Require valid complete clips and a nonnegative patient limit')
    eligible = []
    for case in manifest[split]:
        labels = [trace['frame'] for trace in case['traces']
                  if trace['frame'] - (recent - 1) >= prefix and trace['frame'] + local <= case['frames']]
        if labels:
            eligible.append((case, labels))
    if limit is not None:
        eligible = eligible[:limit]
    return [dict(patient=case['patient'], recent_start=frame-position, H=prefix,
                 prefix_kind='clean', target_frame=frame, target_position=position)
            for case, frames in eligible for frame in frames
            for position in range(recent-local, recent)]


def _options(job, device):
    options = dict(batch_size=int(job.get('micro_batch') or 2), num_workers=int(job.get('num_workers', 0)),
                   pin_memory=device.type == 'cuda')
    if options['num_workers']:
        options.update(persistent_workers=True, prefetch_factor=4)
    return options


def select_anchor(manifest, recent=64, local=16, smoke=False, max_prefix=128):
    available = {}
    for h in (local, local * 2, local * 4, local * 8):
        if h > max_prefix:
            continue
        available[h] = {split:len(history_records(manifest, split, h, recent)) for split in ('train', 'val')}
    minimum = dict(train=2 if smoke else 256, val=1 if smoke else 128)
    eligible = [h for h,count in available.items() if all(count[k] >= minimum[k] for k in minimum)]
    preferred = local * 4
    anchor = preferred if preferred in eligible else max(eligible, default=None)
    return anchor, available


@torch.no_grad()
def extract_history(model, manifest, split, records, job, device):
    if not records:
        return None
    dataset = WindowDataset(manifest, split, task='ef', recent_frames=job.get('recent_frames', 64),
                            local_frames=model.local_frames, max_prefix=job.get('max_prefix', 128), records=records)
    cache, local, boundary, targets, metadata = [], [], [], [], []
    model.eval()
    for batch in tqdm(DataLoader(dataset, **_options(job, device)), desc='Matched history ' + split,
                      disable=job.get('quiet', False)):
        h = int(batch['prefix_frames'][0])
        if not bool((batch['prefix_frames'] == h).all()):
            raise ValueError('History extraction requires one real-length bucket')
        video = native_video(batch['video'].to(device, non_blocking=True), model, bool(job.get('smoke')))
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            values = encode_window(model, video, h, job.get('recent_frames', 64))
        cache.append(values['sequences']['final'].float().cpu())
        local.append(values['sequences']['local'].float().cpu())
        old = values['boundary']
        if old is None:
            old = video.new_zeros(len(video), model.memory_slots, model.embed_dim)
        boundary.append(old.float().cpu()); targets.append(batch['target'])
        for i, patient in enumerate(batch['patient']):
            metadata.append(dict(patient=patient, recent_start=int(batch['recent_start'][i]), prefix=h,
                                 source_path=batch['source_path'][i],
                                 fps=float(batch['fps'][i]), prefix_seconds=h / float(batch['fps'][i])))
    return dict(cache=torch.cat(cache), local=torch.cat(local), boundary=torch.cat(boundary),
                targets=torch.cat(targets), rows=metadata)


@torch.no_grad()
def fixed_head_predictions(model, head, manifest, split, records, task, job, device):
    """Reuse a frozen task module, its exit and train-fitted normalization."""
    if not records:
        return []
    if head.task != task or head.task_protocol['recent_frames'] != job.get('recent_frames', 64):
        raise ValueError('Frozen task/recent-window protocol mismatch')
    dataset = WindowDataset(manifest, split, task=task, recent_frames=job.get('recent_frames', 64),
                            local_frames=model.local_frames, max_prefix=job.get('max_prefix', 128),
                            records=records)
    head.eval()
    rows = []
    for batch in DataLoader(dataset, **_options(job, device)):
        h = int(batch['prefix_frames'][0])
        if not bool((batch['prefix_frames'] == h).all()):
            raise ValueError('Fixed-head history requires one real H bucket')
        video = native_video(batch['video'].to(device), model, bool(job.get('smoke')))
        target = batch['target_index'].to(device) if task == 'seg' else None
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            prediction = head(video, prefix_frames=h, recent_frames=job.get('recent_frames', 64),
                              target_index=target)
        if task == 'ef':
            physical = prediction.float().cpu() * head.normalization['target_std'] + head.normalization['target_mean']
        else:
            masks = batch['mask'].to(device).bool()
            hard = F.interpolate(prediction.float(), masks.shape[-2:], mode='bilinear',
                                 align_corners=False).argmax(1).bool()
            physical = ((2 * (hard & masks).sum((1, 2)).float() + 1e-6) /
                        (hard.sum((1, 2)) + masks.sum((1, 2)) + 1e-6)).cpu()
        for i, patient in enumerate(batch['patient']):
            row = dict(patient=patient, recent_start=int(batch['recent_start'][i]),
                       source_frame=int(batch['target_frame'][i]) if task == 'seg' else -1,
                       position=int(batch['target_recent_index'][i]), target_index=int(batch['target_index'][i]),
                       prefix=h, fps=float(batch['fps'][i]), prefix_seconds=h / float(batch['fps'][i]),
                       source_path=batch['source_path'][i], full_context=bool(batch['full_context'][i]))
            if task == 'ef':
                row.update(target=float(batch['target'][i]), prediction=float(physical[i]),
                           error=float(abs(physical[i] - batch['target'][i])))
            else:
                row['dice'] = float(physical[i])
            rows.append(row)
    return rows


def _population(records, manifest, split, h):
    patients = sorted({r['patient'] for r in records})
    cases = {case['patient']: case for case in manifest[split]}
    seconds = [h / cases[patient]['fps'] for patient in patients]
    return dict(patients=len(patients), source_frames=len({(r['patient'], r.get('target_frame', -1)) for r in records}),
                windows=len(records), patient_keys=patients,
                physical_prefix_seconds_mean=float(np.mean(seconds)) if seconds else None,
                physical_prefix_seconds_min=min(seconds) if seconds else None,
                physical_prefix_seconds_max=max(seconds) if seconds else None)


def _patient_mean(rows, field):
    sources, patients = defaultdict(list), defaultdict(list)
    for row in rows:
        sources[(row['patient'], row.get('source_frame', -1))].append(row[field])
    for (patient, _), values in sources.items():
        patients[patient].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in patients.values()])) if patients else None


def _seg_primary(patient_count, smoke=False):
    return patient_count >= (1 if smoke else 64)


def _plot_history(out, pairs, losses):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    plots = out / 'plots'; plots.mkdir(exist_ok=True)
    figure = Figure(figsize=(10, 4), layout='constrained'); FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, task, label in zip(axes, ('ef', 'seg'), ('Real minus control EF MAE (pp)', 'Real minus control patient Dice')):
        axis.axhline(0, color='black', linewidth=.8)
        for control, color, marker in (('h0', '#0072b2', 'o'), ('repeat', '#d55e00', 's')):
            observed = [p for p in pairs if p.get('task', 'ef') == task and p.get('scope', '').startswith('fixed')
                        and p['contrast'].endswith('_' + control)]
            if observed:
                axis.errorbar([p['prefix'] for p in observed], [p['delta'] for p in observed],
                              yerr=[[max(0., p['delta']-p['low']) for p in observed],
                                    [max(0., p['high']-p['delta']) for p in observed]],
                              color=color, marker=marker, linestyle='none', capsize=3, label=control)
                for p in observed:
                    axis.annotate(f"n={p['patients']}", (p['prefix'], p['delta']), xytext=(3, 5),
                                  textcoords='offset points', fontsize=8)
        axis.set(xlabel='Available history frames (separate matched C_h cohorts)', ylabel=label)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    figure.savefig(plots / 'paired_history.png', dpi=140)
    if losses:
        figure = Figure(figsize=(10, 4), layout='constrained'); FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        for name, curve in losses.items():
            axes[0].plot([r['epoch'] for r in curve], [r['train_loss'] for r in curve], label=name)
            axes[1].plot([r['epoch'] for r in curve], [r['val_mae'] for r in curve], label=name)
        axes[0].set(xlabel='Epoch', ylabel='Train normalized SmoothL1')
        axes[1].set(xlabel='Epoch', ylabel='Validation EF MAE (pp)')
        for axis in axes:
            axis.legend(fontsize=8)
        figure.savefig(plots / 'capability_loss.png', dpi=140)


def fit_head(x, y, xv, yv, history_slots, job, device):
    seed = int(job.get('seed', 42))
    recent = x[:, :-history_slots] if history_slots else x
    mean = recent.mean((0, 1), keepdim=True); std = recent.std((0, 1), unbiased=False, keepdim=True).clamp_min(1e-5)
    ym, ys = y.mean(), y.std(unbiased=False).clamp_min(1)
    torch.manual_seed(seed + 1001)
    head = TemporalEFReadout(x.shape[-1], int(job.get('hidden', 64))).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
    best, bad, best_state, curve = math.inf, 0, None, []
    epochs = int(job.get('head_epochs', 2 if job.get('smoke') else 60))
    batch = min(32, len(x))
    @torch.no_grad()
    def predict(values):
        values = (values - mean) / std
        return torch.cat([head(part.to(device), history_slots).cpu() for part in values.split(64)]) * ys + ym
    for epoch in range(epochs):
        head.train(); order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))
        losses = []
        lr = 1e-3 * min(1., (epoch + 1) / 3) * (.01 + .99 * .5 * (1 + math.cos(math.pi * epoch / max(1, epochs))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        for indices in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            prediction = head(((x[indices] - mean) / std).to(device), history_slots)
            loss = F.smooth_l1_loss(prediction, ((y[indices] - ym) / ys).to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
        head.eval(); error = float((predict(xv) - yv).abs().mean())
        curve.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), val_mae=error, lr=lr))
        if error < best:
            best = error; bad = 0; best_state = copy.deepcopy(head.state_dict())
        else:
            bad += 1
        if bad >= int(job.get('head_patience', 12)):
            break
    head.load_state_dict(best_state); head.eval()
    return predict, dict(head=head.state_dict(), mean=mean, std=std, target_mean=ym, target_std=ys,
                         history_slots=history_slots, hidden=int(job.get('hidden', 64))), curve


def predictions(features, predict, tensor_key='cache'):
    values = predict(features[tensor_key])
    return [dict(row, target=float(features['targets'][i]), prediction=float(values[i]),
                 error=float(abs(values[i] - features['targets'][i]))) for i,row in enumerate(features['rows'])]


def run_history_job(job, manifest, device):
    """Fit only a count-qualified capability anchor; always audit fixed heads.

    job accepts seg_head_checkpoint/seg_head_metrics and optional
    ef_head_checkpoint/ef_head_metrics (frozen task best checkpoints). The EF
    head is the descriptive fallback when no count-qualified anchor fit exists.
    seg_fixed_head_curve and fixed_head_coverage separate primary/auxiliary
    populations, missing heads, and zero available real-history cohorts.
    """
    device = torch.device(device); out = Path(job['output_dir']); weights = Path(job['checkpoint_dir'])
    protocol = job_protocol(job, manifest)
    protocol['history_code'] = file_digest(Path(__file__))
    protocol['fixed_head_sources'] = {task:file_digest(Path(job[task + '_head_checkpoint']))
                                      for task in ('ef', 'seg') if job.get(task + '_head_checkpoint')}
    delivered = ('metrics.json', 'cohort_counts.json', 'length_curve.csv', 'paired.csv', 'plots/paired_history.png')
    if guard_job(out, protocol, delivered):
        import json
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    weights.mkdir(parents=True, exist_ok=True)
    model = load_final_model(job['checkpoint'])[0].to(device).eval()
    recent = int(job.get('recent_frames', 64)); local = model.local_frames
    anchor, counts = select_anchor(manifest, recent, local, bool(job.get('smoke')), int(job.get('max_prefix', 128)))
    smoke = bool(job.get('smoke'))
    minimum = dict(train=2 if smoke else 256, val=1 if smoke else 128)
    limits = {split:int(job.get(split + '_cases', 4 if smoke else 1024 if split == 'train' else 512))
              for split in ('train', 'val')}
    records = ({split:history_records(manifest, split, anchor, recent, limits[split]) for split in ('train', 'val')}
               if anchor is not None else dict(train=[], val=[]))
    write_json(out / 'anchor_records.json', records)
    fitted_counts = {split:len(rows) for split,rows in records.items()}
    fit_allowed = anchor is not None and all(fitted_counts[split] >= minimum[split] for split in minimum)
    capability_status = ('complete' if fit_allowed else 'insufficient_anchor_population' if anchor is None
                         else 'insufficient_selected_anchor_population')
    curves, seg_curves, pairs, capabilities, losses = [], [], [], {}, {}
    anchor_predict = None
    seed, repetitions = int(job.get('seed', 42)), int(job.get('bootstrap_repetitions', 2000))
    if fit_allowed:
        actual = {split:extract_history(model, manifest, split, rows, job, device) for split,rows in records.items()}
        empty_records = {split:[dict(row, H=0) for row in rows] for split,rows in records.items()}
        empty = {split:extract_history(model, manifest, split, rows, job, device) for split,rows in empty_records.items()}
        rows_by_name = {}
        for name, dataset, key, slots in (
                ('anchor_h0', empty, 'cache', 0), ('anchor_real', actual, 'cache', 0),
                ('local_empty', actual, 'local_empty', model.memory_slots),
                ('local_history', actual, 'local_history', model.memory_slots)):
            if key.startswith('local_'):
                for values in dataset.values():
                    state = values['boundary'] if key == 'local_history' else torch.zeros_like(values['boundary'])
                    values[key] = torch.cat((values['local'], state), 1)
            predict, saved, curve = fit_head(dataset['train'][key], dataset['train']['targets'],
                                            dataset['val'][key], dataset['val']['targets'], slots, job, device)
            atomic_torch_save(saved, weights / (name + '.pt'))
            losses[name] = curve
            rows_csv(out / (name + '_loss.csv'), curve)
            rows = predictions(dataset['val'], predict, key); rows_by_name[name] = rows
            rows_csv(out / (name + '_predictions.csv'), rows)
            capabilities[name] = _patient_mean(rows, 'error')
            if name == 'anchor_real':
                anchor_predict = predict
        for contrast, left, right in (('capability_real_minus_h0', 'anchor_real', 'anchor_h0'),
                                       ('capability_local_history_minus_zero', 'local_history', 'local_empty')):
            pairs.append(dict(contrast=contrast, prefix=anchor, task='ef', scope='capability',
                              **paired_delta(rows_by_name[left], rows_by_name[right], 'error', seed, repetitions)))
    heads = {task:(load_frozen_task_head(job[task + '_head_checkpoint'], model, job.get(task + '_head_metrics'))
                   if job.get(task + '_head_checkpoint') else None) for task in ('ef', 'seg')}
    coverage = dict(ef={}, seg={})
    lengths = [0] + [h for h in (local, local*2, local*4, local*8) if h <= int(job.get('max_prefix', 128))]
    for task in ('ef', 'seg'):
        for h in lengths:
            all_records = (history_records(manifest, 'val', h, recent) if task == 'ef' else
                           seg_history_records(manifest, 'val', h, recent, local))
            available = _population(all_records, manifest, 'val', h)
            same = (history_records(manifest, 'val', h, recent, limits['val']) if task == 'ef' else
                    seg_history_records(manifest, 'val', h, recent, local, limits['val']))
            selected = _population(same, manifest, 'val', h)
            primary = task == 'seg' and _seg_primary(selected['patients'], smoke)
            scope = ('fixed_anchor_head' if task == 'ef' and anchor_predict is not None
                     else 'fixed_frozen_' + task + '_head')
            feasible = (anchor_predict is not None if task == 'ef' else False) or heads[task] is not None
            status = 'evaluated' if same and feasible else 'no_real_cohort' if not same else 'missing_fixed_head'
            coverage[task][str(h)] = dict(available=available, selected=selected, evaluation_status=status,
                                         population_scope='primary' if primary else 'auxiliary',
                                         primary_patient_minimum=(1 if smoke else 64) if task == 'seg' else None)
            dependencies = {}
            conditions = [('h0', same)] if h == 0 else [('real', same), ('h0', [dict(r, H=0) for r in same]),
                          ('repeat', [dict(r, prefix_kind='repeat_prefix') for r in same])]
            for condition, altered in conditions:
                rows = []
                if same and feasible:
                    if task == 'ef' and anchor_predict is not None:
                        rows = predictions(extract_history(model, manifest, 'val', altered, job, device), anchor_predict)
                    else:
                        rows = fixed_head_predictions(model, heads[task], manifest, 'val', altered, task, job, device)
                dependencies[condition] = rows
                name = 'fixed_head' if task == 'ef' else 'seg_fixed_head'
                rows_csv(out / f'{name}_h{h}_{condition}.csv', rows)
                entry = dict(prefix=h, condition=condition, scope=scope, task=task,
                             population_scope='primary' if primary else 'auxiliary', evaluation_status=status,
                             available_patients=available['patients'], **selected,
                             processed_prefix_frames=0 if condition == 'h0' else h,
                             processed_prefix_seconds_mean=float(np.mean([r['prefix_seconds'] for r in rows])) if rows else None)
                entry['mae' if task == 'ef' else 'patient_dice'] = _patient_mean(rows, 'error' if task == 'ef' else 'dice')
                (curves if task == 'ef' else seg_curves).append(entry)
            if h and status == 'evaluated':
                for control in ('h0', 'repeat'):
                    contrast = ('fixed' if task == 'ef' else 'seg_fixed') + '_real_minus_' + control
                    pairs.append(dict(contrast=contrast, prefix=h, task=task, scope=scope,
                                      population_scope='primary' if primary else 'auxiliary',
                                      physical_prefix_seconds_mean=selected['physical_prefix_seconds_mean'],
                                      source_frames=selected['source_frames'], windows=selected['windows'],
                                      delta_units='percentage_points' if task == 'ef' else 'dice_fraction',
                                      better_direction='negative' if task == 'ef' else 'positive',
                                      **paired_delta(dependencies['real'], dependencies[control],
                                                     'error' if task == 'ef' else 'dice', seed, repetitions)))
    write_json(out / 'cohort_counts.json', dict(anchor=anchor, counts=counts, capability_minimum=minimum,
                                              selected_anchor_counts=fitted_counts, capability_status=capability_status,
                                              fixed_head_coverage=coverage,
                                              note='H128 is diagnostic only; count thresholds apply to actual selected patients'))
    rows_csv(out / 'length_curve.csv', curves + seg_curves); rows_csv(out / 'paired.csv', pairs)
    _plot_history(out, pairs, losses)
    supported = False
    for task in ('ef', 'seg'):
        for h in lengths[1:]:
            matched = [p for p in pairs if p['task'] == task and p['prefix'] == h and p['scope'].startswith('fixed')]
            if (len(matched) == 2 and (task == 'ef' or all(p['population_scope'] == 'primary' for p in matched))
                    and all(p['delta'] < 0 if task == 'ef' else p['delta'] > 0 for p in matched)):
                supported = True
    metrics = dict(anchor=anchor, status='complete' if any(c['evaluation_status'] == 'evaluated'
                   for values in coverage.values() for c in values.values()) else 'insufficient_history_evidence',
                   capability_status=capability_status, cohorts=counts, capabilities=capabilities,
                   capability_minimum=minimum, selected_anchor_counts=fitted_counts,
                   fixed_head_coverage=coverage, fixed_head_curve=curves, seg_fixed_head_curve=seg_curves,
                   paired=pairs, history_content_supported=supported,
                   note='Within-C_h real-minus-controls effects only; no cross-population absolute trend or clinical claim. '
                        'Fixed-head curves are descriptive dependence tests; capability fits require registered TRAIN/VAL counts. '
                        'H128 is diagnostic, not an application precondition. Auxiliary segmentation does not support the main conclusion. '
                        'Curve physical_prefix_seconds describe the available true cohort H; processed_prefix_seconds describe the actual view. '
                        'H0 retains recent-window RVM updates; it is not a memory-disabled model.')
    # Match the JSON round-trip, including integer keys in the older cohort map.
    import json
    metrics = json.loads(json.dumps(metrics, allow_nan=False))
    write_json(out / 'metrics.json', metrics); complete_job(out, delivered)
    return metrics
