"""Physical-time trajectory diagnostics; labels never enter event detection."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def validate_trajectory(features, seconds):
    x, t = np.asarray(features, dtype=np.float64), np.asarray(seconds, dtype=np.float64)
    if x.ndim != 2 or len(x) < 3 or t.shape != (len(x),):
        raise ValueError('Need [time,channels] features with matching timestamps')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Nonfinite trajectory or non-increasing timestamps')
    return x, t


def trajectory_statistics(features, seconds):
    x, t = validate_trajectory(features, seconds)
    centered = x - x.mean(0)
    variance = np.linalg.svd(centered, compute_uv=False) ** 2
    total = variance.sum()
    probabilities = variance / max(total, 1e-12)
    velocity = np.diff(x, axis=0) / np.diff(t)[:, None]
    acceleration = np.diff(velocity, axis=0) / ((np.diff(t)[1:] + np.diff(t)[:-1]) / 2)[:, None]
    eligible = [(i, np.mean(np.sum(centered[:-i] * centered[i:], axis=1)))
                for i in range(1, len(x)) if .4 <= np.median(t[i:] - t[:-i]) <= 1.5]
    recurrence = None
    if total > 1e-12 and eligible:
        energy = np.mean(np.sum(centered ** 2, axis=1))
        lag, score = max(eligible, key=lambda item: item[1] / energy)
        recurrence = dict(lag_seconds=float(np.median(t[lag:] - t[:-lag])), score=float(score / energy))
    return dict(effective_rank=float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12)))) if total > 1e-12 else 0.,
                rank_ceiling=min(len(x) - 1, x.shape[1]),
                explained_variance_2=float(probabilities[:2].sum()),
                temporal_variance=float(total / len(x)), collapsed=bool(total <= 1e-12),
                velocity_rms=float(np.sqrt(np.mean(velocity ** 2))),
                acceleration_rms=float(np.sqrt(np.mean(acceleration ** 2))),
                median_interval_ms=float(1000 * np.median(np.diff(t))),
                recurrence=recurrence)


def fit_motion_axis(trajectories):
    """One shared axis fitted only on training motion; one training-label sign bit."""
    directions, differences = [], []
    for item in trajectories:
        x, t = validate_trajectory(item['features'], item['seconds'])
        d = np.diff(x, axis=0)
        norm = np.linalg.norm(d, axis=1)
        directions.append(d[norm > 1e-8] / norm[norm > 1e-8, None])
        ed, es = item['ed_seconds'], item['es_seconds']
        differences.append(x[np.argmin(abs(t - ed))] - x[np.argmin(abs(t - es))])
    d = np.concatenate(directions)
    if not len(d):
        return np.zeros(trajectories[0]['features'].shape[1], dtype=np.float64)
    _, _, vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
    axis = vt[0]
    if np.mean(np.asarray(differences) @ axis) < 0:
        axis = -axis
    return axis


def motion_signal(features, seconds, axis, smoothing_seconds=.15):
    x, t = validate_trajectory(features, seconds)
    signal = (x - x.mean(0)) @ axis
    raw = signal.copy()
    # Offline diagnostics intentionally see the whole trajectory. This is not a causal detector.
    signal -= np.polyval(np.polyfit(t - t[0], signal, 1), t - t[0])
    window = max(3, int(round(smoothing_seconds / np.median(np.diff(t)))) | 1)
    window = min(window, len(signal) if len(signal) % 2 else len(signal) - 1)
    if window >= 3:
        signal = savgol_filter(signal, window, min(2, window - 1))
    return raw, signal


def detect_events(signal, seconds, prominence=.25, separation_seconds=.2):
    s, t = np.asarray(signal), np.asarray(seconds)
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite event signal')
    if np.ptp(s) <= 1e-10:
        return dict(ed=[], es=[])
    distance = max(1, int(np.ceil(separation_seconds / np.median(np.diff(t)))))
    options = dict(prominence=prominence * np.ptp(s), distance=distance)
    return dict(ed=find_peaks(s, **options)[0].tolist(), es=find_peaks(-s, **options)[0].tolist())


def event_scores(events, seconds, ed_seconds, es_seconds, fps):
    t = np.asarray(seconds)
    scores = {}
    for phase, target in (('ed', ed_seconds), ('es', es_seconds)):
        candidates = t[events[phase]]
        error = float(abs(candidates - target).min()) if len(candidates) else None
        scores.update({phase + '_error_ms': None if error is None else error * 1000,
                       phase + '_error_frames': None if error is None else error * fps,
                       phase + '_detected': bool(len(candidates)),
                       phase + '_within100ms': error is not None and error <= .100,
                       phase + '_candidates': len(candidates)})
    scores['candidates_per_second'] = (len(events['ed']) + len(events['es'])) / (t[-1] - t[0])
    return scores


def pair_identity(prediction, target, min_energy=1e-6):
    """Adjacent-pair assignment; identical predictions receive chance credit, never a win."""
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2 or len(p) % 2:
        raise ValueError('Pair identity requires matching even [frames,descriptor] arrays')
    p, y = p.reshape(-1, 2, p.shape[-1]), y.reshape(-1, 2, y.shape[-1])
    y = y - y.mean(-1, keepdims=True)
    p = p - p.mean(-1, keepdims=True)
    energy = np.mean((y[:, 0] - y[:, 1]) ** 2, axis=1)
    eligible = energy > min_energy
    correct = np.mean((p - y) ** 2, axis=(1, 2))
    swapped = np.mean((p - y[:, ::-1]) ** 2, axis=(1, 2))
    margin = swapped - correct
    tied = np.isclose(margin, 0., atol=1e-10, rtol=1e-6)
    credit = np.where(tied, .5, margin > 0).astype(float)
    return dict(pair_accuracy=float(credit[eligible].mean()) if eligible.any() else None,
                eligible_pairs=int(eligible.sum()), excluded_near_identical_pairs=int((~eligible).sum()),
                tied_fraction=float(tied[eligible].mean()) if eligible.any() else None,
                mean_assignment_margin=float(margin[eligible].mean()) if eligible.any() else None)


def safe_correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if min(np.std(a), np.std(b)) <= 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def motion_agreement(a, b):
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    denominator = np.linalg.norm(da) * np.linalg.norm(db)
    return float(np.sum(da * db) / denominator) if denominator > 1e-10 else None


def clip_boundary_ratio(features, local_frames=16, tubelet=2):
    """Compare like-for-like tubelet transitions, not repeat-induced zero steps."""
    speed = np.linalg.norm(np.diff(features, axis=0), axis=1)
    indices = np.arange(len(speed))
    boundary = indices % local_frames == local_frames - 1
    internal = (indices % tubelet == tubelet - 1) & ~boundary
    if not boundary.any() or not internal.any() or speed[internal].mean() <= 1e-10:
        return None
    return float(speed[boundary].mean() / speed[internal].mean())
