"""Explicit patient-level metrics; empty masks and units are never implicit."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def segmentation_row(prediction, target, *, patient, source_frame, class_id=1,
                     spacing=None, boundaries=False, **metadata):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError('Boundary/segmentation metrics require matching 2D native masks')
    if spacing is not None and (len(spacing) != 2 or not all(np.isfinite(x) and x > 0 for x in spacing)):
        raise ValueError('Physical spacing requires two positive finite axis-ordered values')
    tp, p, t = int((pred & truth).sum()), int(pred.sum()), int(truth.sum())
    row = dict(patient=str(patient), source_frame=str(source_frame), class_id=int(class_id),
               intersection=tp, predicted_pixels=p, target_pixels=t,
               dice=2 * tp / (p + t) if p + t else 1.,
               empty_prediction=not p, empty_target=not t,
               boundary_failure=bool(p != 0) != bool(t != 0),
               assd=None, hd95=None, boundary_unit='mm' if spacing is not None else 'pixel', **metadata)
    if boundaries and p and t:
        a = pred ^ binary_erosion(pred, border_value=0)
        b = truth ^ binary_erosion(truth, border_value=0)
        ab = distance_transform_edt(~b, sampling=spacing)[a]
        ba = distance_transform_edt(~a, sampling=spacing)[b]
        distances = np.concatenate((ab, ba))
        row.update(assd=float(distances.mean()), hd95=float(np.percentile(distances, 95)))
    return row


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def summarize(rows, task):
    if not rows:
        raise ValueError('Cannot report an empty evaluation population')
    if task == 'mae':
        return dict(reconstruction_loss=_mean([r['loss'] for r in rows]), samples=len(rows)), rows
    if task == 'ef':
        groups = defaultdict(list)
        for row in rows:
            groups[str(row['patient'])].append(row)
        patients = []
        for patient, values in sorted(groups.items()):
            labels = {float(v['target']) for v in values}
            if len(labels) != 1:
                raise ValueError('Patient EF targets differ; aggregate by exam first or fix patient/exam IDs')
            y, p = labels.pop(), _mean([float(v['prediction']) for v in values])
            patients.append(dict(patient=patient, target=y, prediction=p, error=p-y, absolute_error=abs(p-y)))
        y, p = (np.array([r[key] for r in patients]) for key in ('target', 'prediction'))
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Nonfinite EF predictions or labels')
        error = p-y
        var = float(((y-y.mean())**2).sum())
        corr = float(np.corrcoef(p, y)[0, 1]) if len(y) > 1 and p.std() > 0 and y.std() > 0 else None
        return dict(mae_pp=float(np.abs(error).mean()), rmse_pp=float(np.sqrt((error**2).mean())),
                    bias_pp=float(error.mean()), r2=1-float((error**2).sum())/var if var else None,
                    corr=corr, p90_absolute_error_pp=float(np.percentile(np.abs(error), 90)),
                    within_5=float((np.abs(error) <= 5).mean()), within_10=float((np.abs(error) <= 10).mean()),
                    patients=len(patients), windows=len(rows), aggregation='mean prediction per patient'), patients
    if task != 'seg':
        raise ValueError(f'Unknown metric task: {task}')
    frames = defaultdict(list)
    for row in rows:
        if not np.isfinite(row['dice']) or not 0 <= row['dice'] <= 1:
            raise ValueError('Dice observations must be finite fractions in [0,1]')
        if row['intersection'] < 0 or row['intersection'] > min(row['predicted_pixels'], row['target_pixels']):
            raise ValueError('Invalid segmentation pixel counts')
        frames[(str(row['patient']), str(row['source_frame']), int(row['class_id']))].append(row)
    patients, by_class = defaultdict(list), defaultdict(list)
    pixel_counts = defaultdict(lambda: np.zeros(3, dtype=float))
    boundary_values, boundary_failures = [], 0
    for (patient, frame, class_id), values in sorted(frames.items()):
        if len({v['target_pixels'] for v in values}) != 1:
            raise ValueError('Repeated positions of a source frame must use identical GT coordinates')
        # Repeated target positions/windows count once per source frame.
        dice = _mean([v['dice'] for v in values])
        patients[patient].append(dice)
        by_class[class_id].append((patient, dice))
        pixel_counts[class_id] += np.array([[v['intersection'], v['predicted_pixels'], v['target_pixels']]
                                          for v in values]).mean(0)
        boundary_failures += int(any(v['boundary_failure'] for v in values))
        finite = [v for v in values if v['assd'] is not None]
        if finite:
            units = {v['boundary_unit'] for v in finite}
            if len(units) != 1:
                raise ValueError('Cannot aggregate physical and pixel distances')
            boundary_values.append((patient, _mean([v['assd'] for v in finite]),
                                    _mean([v['hd95'] for v in finite]), units.pop()))
    patient_rows = [dict(patient=p, dice=_mean(values)) for p, values in sorted(patients.items())]
    result = dict(dice_mean=_mean([r['dice'] for r in patient_rows]), patients=len(patient_rows),
                  source_frames=len({(patient, frame) for patient,frame,_ in frames}), observations=len(rows), boundary_failure_frames=boundary_failures,
                  empty_prediction_observations=sum(bool(r['empty_prediction']) for r in rows),
                  aggregation='position/window mean -> source frame mean -> patient mean')
    ordered = sorted(r['dice'] for r in patient_rows)
    result['worst_patient_decile_dice'] = _mean(ordered[:max(1, int(np.ceil(len(ordered)*.1)))])
    global_dice = []
    for class_id, values in by_class.items():
        grouped = defaultdict(list)
        for patient, value in values:
            grouped[patient].append(value)
        result[f'dice_class_{class_id}'] = _mean([_mean(v) for v in grouped.values()])
        tp, p, t = pixel_counts[class_id]
        result[f'dice_global_class_{class_id}'] = float(2*tp/(p+t)) if p+t else 1.
        global_dice.append(result[f'dice_global_class_{class_id}'])
    result['dice_global_mean'] = _mean(global_dice)
    if boundary_values:
        units = {r[3] for r in boundary_values}
        if len(units) != 1:
            raise ValueError('Do not pool mm and pixel metrics')
        result['boundary_unit'] = units.pop()
        for index, key in ((1, 'assd'), (2, 'hd95')):
            grouped = defaultdict(list)
            for row in boundary_values:
                grouped[row[0]].append(row[index])
            result[key + '_valid_only'] = _mean([_mean(v) for v in grouped.values()])
        result['boundary_valid_frames'] = len(boundary_values)
        result['boundary_warning'] = 'Empty-mask failures counted separately; valid-only distances are not an all-case score'
    return result, patient_rows


def stratify(rows, task, fields=('phase', 'view', 'target_position')):
    result = {}
    for field in fields:
        groups = defaultdict(list)
        for row in rows:
            if row.get(field) is not None:
                groups[str(row[field])].append(row)
        if groups:
            result[field] = {key: summarize(values, task)[0] for key, values in sorted(groups.items())}
    return result


def paired_bootstrap(reference, candidate, field, repetitions=2000, seed=42):
    a, b = ({str(r['patient']): r for r in values} for values in (reference, candidate))
    if len(a) != len(reference) or len(b) != len(candidate) or set(a) != set(b):
        raise ValueError('Paired comparisons require unique, exactly matched patient IDs')
    delta = np.array([float(b[p][field])-float(a[p][field]) for p in sorted(a)])
    if not len(delta) or not np.isfinite(delta).all() or repetitions < 1:
        raise ValueError('Invalid paired observations or repetition budget')
    rng = np.random.default_rng(seed)
    draws = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repetitions)])
    return dict(candidate_minus_reference=float(delta.mean()), ci95=list(map(float, np.quantile(draws, [.025, .975]))),
                patients=len(delta), repetitions=repetitions, seed=seed,
                note='Patient sampling uncertainty, not independent pretraining-seed stability; not a selection veto')


def representation_summary(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('Need finite [samples, features] with at least two samples/features')
    singular = np.linalg.svd(x-x.mean(0), compute_uv=False)
    eigenvalues = singular**2
    total = eigenvalues.sum()
    probabilities = eigenvalues / total if total else np.zeros_like(eigenvalues)
    nonzero = probabilities[probabilities > 0]
    return dict(samples=len(x), dimensions=x.shape[1],
                effective_rank=float(np.exp(-(nonzero*np.log(nonzero)).sum())) if total else 0.,
                variance_mean=float(x.var(0).mean()), top1_variance_fraction=float(probabilities[0]),
                interpretation='Geometry/collapse diagnostics only; rank does not establish anatomy or motion semantics')


def linear_cka(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('CKA requires paired sample order')
    x, y = x-x.mean(0), y-y.mean(0)
    denominator = np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2 / denominator) if denominator else None
