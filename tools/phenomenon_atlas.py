"""Model-free EchoNet/TED development measurements for the phenomenon atlas."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAGS = (1, 2, 4, 8, 16)
DEV = {"TRAIN", "VAL"}


def split_rows(path, split):
    with path.open(newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["Split"].strip().upper() in split:
                yield row


def load_reference(root):
    rows = {}
    for row in split_rows(root / "FileList.csv", DEV):
        key = Path(row["FileName"]).stem
        if key in rows:
            raise ValueError(f"Duplicate development ID: {key}")
        rows[key] = row
    coords = {}
    with (root / "VolumeTracings.csv").open(newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            key = Path(row["FileName"]).stem
            if key in rows:
                points = [float(row[k]) for k in ("X1", "Y1", "X2", "Y2")]
                if np.isfinite(points).all():
                    coords.setdefault(key, {}).setdefault(int(row["Frame"]), []).append(points)
    return rows, coords


def rasterize(points):
    from skimage.draw import polygon

    p = np.asarray(points)
    if len(p) < 3:
        return np.zeros((112, 112), dtype=bool)
    rr, cc = polygon(np.r_[p[1:, 1], p[1:, 3][::-1]],
                     np.r_[p[1:, 0], p[1:, 2][::-1]], shape=(112, 112))
    mask = np.zeros((112, 112), dtype=bool)
    mask[rr, cc] = True
    return mask


def gray(raw):
    x = np.asarray(raw)
    if x.ndim == 4 and x.shape[-1] in (1, 3):
        x = x.mean(axis=-1)
    if x.ndim != 3 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        raise ValueError("Expected finite grayscale frames within [0,255]")
    x = x.astype(np.float32)
    return x / 255 if raw.dtype == np.uint8 or x.max() > 1 else x


def region_mask(x):
    # Static nonblack support across time approximates the scan sector.
    support = ndi.binary_fill_holes(np.max(x, axis=0) > 5 / 255)
    if not support.any():
        raise ValueError("Empty scan region")
    return support


def geom(m, n):
    return float(np.count_nonzero(m ^ n) / max(1, np.count_nonzero(m | n)))


def edge(m):
    return m & ~ndi.binary_erosion(m)


def boundary_distance(m, n):
    a, b = edge(m), edge(n)
    return float((ndi.distance_transform_edt(~a)[b].mean()
                  + ndi.distance_transform_edt(~b)[a].mean()) / 2)


def common_output_floor(m, n):
    return float(np.mean(m != n) / 2)


def block_mask(m, patch):
    if any(d % patch for d in m.shape):
        raise ValueError("Mask shape must divide into patches")
    h, w = m.shape
    coarse = m.reshape(h // patch, patch, w // patch, patch).mean((1, 3)) >= .5
    return coarse.repeat(patch, 0).repeat(patch, 1)


def contrast(x, m):
    outer = ndi.binary_dilation(m, iterations=2) & ~m
    inner = m & ~ndi.binary_erosion(m, iterations=2)
    mu = x[outer].mean() - x[inner].mean()
    return float(abs(mu) / max(.01, np.sqrt((x[outer].var() + x[inner].var()) / 2)))


def rmse(diff, mask):
    return float(np.sqrt(np.mean(np.square(diff[mask])))) if mask.any() else None


def anon(key):
    return sha256(key.encode()).hexdigest()[:16]


def trace_data(root, limit, seed, metadata_only=False):
    rows, coords = load_reference(root)
    metadata, spatial, history, scale, perturb = [], [], [], [], []
    counts = {"development_rows": len(rows), "with_endpoint_pair": 0, "image_cases": 0}
    candidates = []
    for key, row in sorted(rows.items()):
        p = coords.get(key, {})
        if len(p) != 2:
            continue
        frames = sorted(p)
        masks = [rasterize(p[f]) for f in frames]
        if not all(m.any() for m in masks):
            continue
        fps = float(row["FPS"] or "nan")
        if not np.isfinite(fps) or fps <= 0:
            continue
        if masks[0].sum() == masks[1].sum():
            continue
        if masks[0].sum() < masks[1].sum():
            frames.reverse()
            masks.reverse()
        ed, es = frames
        length = int(row["NumberOfFrames"] or 0)
        if not 0 <= min(frames) <= max(frames) < length:
            continue
        key_out = anon(key)
        # Uniform valid W64 starts, integrated within each patient.
        starts = np.arange(max(0, length - 63))
        covered = (starts <= min(ed, es)) & (starts + 63 >= max(ed, es))
        same = covered & (((ed - starts) // 16) == ((es - starts) // 16))
        ef = float(row["EF"] or "nan")
        metadata.append(dict(case=key_out, split=row["Split"].upper(), fps=fps, ed=ed, es=es,
                             frames=length, seconds=length / fps, gap_frames=es-ed,
                             gap_ms=(es-ed)*1000/fps, ed_area=int(masks[0].sum()),
                             es_area=int(masks[1].sum()), ef=ef if np.isfinite(ef) else None,
                             both=float(covered.mean()) if len(starts) else None,
                             same=float(same.mean()) if len(starts) else None))
        if row["Split"].strip().upper() == "VAL":
            candidates.append((key, frames, masks))
    counts["with_endpoint_pair"] = len(metadata)
    if metadata_only:
        return metadata, spatial, history, scale, perturb, counts
    rng = np.random.default_rng(seed)
    if limit:
        candidates = [candidates[i] for i in rng.permutation(len(candidates))[:limit]]
    counts["requested_val_image_cases"] = len(candidates)
    for idx, (key, frames, masks) in enumerate(candidates, 1):
        path = root / "npy" / f"{key}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete NPY cache: {path}")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim not in (3, 4) or raw.shape[1:3] != (112, 112) or max(frames) >= len(raw):
            raise ValueError(f"Incompatible cached video and trace coordinates: {path}")
        # Read only the span used by the lag analysis, plus the annotated frames.
        offsets = {0, 1, *LAGS, *(2*lag for lag in LAGS)}
        offsets |= {lag+1 for lag in offsets}
        indices = sorted({t - lag for t in frames for lag in offsets if t >= lag})
        x = dict(zip(indices, gray(raw[indices])))
        support = region_mask(np.stack(list(x.values())))
        case = anon(key)
        for phase, (t, m) in zip(("ED", "ES"), zip(frames, masks)):
            image = x[t]
            band = ndi.binary_dilation(m, iterations=2) & ~ndi.binary_erosion(m, iterations=2)
            region = dict(edge=band & support, cavity=(m & ~band) & support,
                          remainder=support & ~m & ~band)
            for patch in (1, 4, 8, 16):
                coarse = block_mask(m, patch)
                spatial.append(dict(case=case, phase=phase, patch=patch,
                                    geometry=geom(m, coarse), cnr=contrast(image, m)))
            yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
            shade = np.exp(-((xx-60)**2 + (yy-70)**2) / (2*16**2))
            for strength in (0, .25, .5):
                degraded = np.clip(image*(1-strength*shade) + rng.normal(0, strength*.07, image.shape), 0, 1)
                for sigma in (0, .7, 1.4, 2.1):
                    smooth = ndi.gaussian_filter(degraded, sigma)
                    perturb.append(dict(case=case, phase=phase, strength=strength, sigma=sigma,
                                        cnr=contrast(smooth, m),
                                        interior_error=rmse(smooth-image, region["cavity"]),
                                        edge_error=rmse(smooth-image, region["edge"])))
            visible = np.repeat(np.repeat(rng.random((14, 14)) < .1, 8, 0), 8, 1) & support
            if not visible.any():
                continue
            loc = ndi.distance_transform_edt(~visible, return_distances=False, return_indices=True)
            spatial_pred = image[tuple(loc)]
            for lag in LAGS:
                if t < 2*lag:
                    continue
                old, stale = x[t-lag], x[t-2*lag]
                # The historical frame is privileged (fully observed). Use only current visible pixels
                # to fit its brightness offset. No current masked pixels enter the predictor.
                def adapt(past, current=image):
                    return np.clip(past + np.median((current-past)[visible]), 0, 1)
                past_pred, stale_pred = adapt(old), adapt(stale)
                current_delta = previous_delta = None
                if t > 2*lag:
                    one_before = x[t-1]
                    gt_delta = image-one_before
                    current_delta = spatial_pred - one_before[tuple(loc)] - gt_delta
                    previous_delta = past_pred - adapt(x[t-1-lag], one_before) - gt_delta
                for name, r in region.items():
                    keep = r & ~visible
                    history.append(dict(case=case, phase=phase, lag=lag, region=name,
                                        now=rmse(spatial_pred-image, keep),
                                        past=rmse(past_pred-image, keep),
                                        stale=rmse(stale_pred-image, keep),
                                        now_delta=rmse(current_delta, keep) if current_delta is not None else None,
                                        past_delta=rmse(previous_delta, keep) if previous_delta is not None else None))
                    for sigma in (1, 2, 4):
                        diff = (image-ndi.gaussian_filter(image, sigma)
                                - old+ndi.gaussian_filter(old, sigma))
                        scale.append(dict(case=case, phase=phase, lag=lag, region=name,
                                          sigma=sigma, error=rmse(diff, r)))
        counts["image_cases"] += 1
        if idx % 25 == 0 or idx == len(candidates):
            print(f"EchoNet image panel: {idx}/{len(candidates)}", flush=True)
    return metadata, spatial, history, scale, perturb, counts


def read_ted_case(path):
    with np.load(path, allow_pickle=False) as p:
        images = gray(p["video"])
        labels = p["mask"]
        fps = float(p["fps"])
        ef = float(p["ef"]) if "ef" in p else float("nan")
    if (images.shape != labels.shape or images.ndim != 3 or len(images) < 8
            or not np.isfinite(fps) or fps <= 0 or not np.isin(labels, (0, 1)).all()
            or not np.all(labels.any(axis=(1, 2)))):
        raise ValueError(f"Bad dense-reference archive: {path}")
    return images, labels.astype(bool), fps, ef if np.isfinite(ef) else None


def dense_measurements(root, allowed):
    pairs, curves, budget, references = [], [], [], []
    if root is None:
        return pairs, curves, budget, references
    if allowed is None:
        raise ValueError("--dense-root also requires a development ID allowlist")
    ids = set(allowed.read_text(encoding="utf-8-sig").split())
    for p in sorted(root.glob("*.npz")):
        if p.stem not in ids:
            continue
        print(f"Dense reference {len(curves)+1}", flush=True)
        case = anon(p.stem)
        video, labels, fps, ef = read_ted_case(p)
        support = region_mask(video)
        area = labels.mean((1, 2))
        amin, amax = int(area.argmin()), int(area.argmax())
        curves.append(dict(case=case, fps=fps, area=area.tolist(), es=amin, ed=amax, ef=ef))
        pair_rows = []
        for lag in LAGS:
            if lag >= len(video):
                continue
            for i in range(len(video)-lag):
                j = i+lag
                m, n = labels[i], labels[j]
                row = dict(case=case, lag=lag, lag_ms=1000*lag/fps,
                                  phase=(i+j)/(2*(len(video)-1)), es_phase=amin/(len(video)-1),
                                  image=rmse(video[i]-video[j], support),
                                  geometry=geom(m, n), boundary=boundary_distance(m, n),
                                  floor=common_output_floor(m, n),
                                  contrast=contrast(video[j], n))
                pairs.append(row)
                if lag == 1:
                    pair_rows.append((i, j, row))
                if lag in (1, 4, 16):
                    references.append(dict(case=case, lag=lag, phase_distance=lag/(len(video)-1),
                                           contrast=contrast(video[j], n), geometry=geom(m, n)))
        # Deterministic example: highest disagreement among the less different images.
        cap = np.quantile([r["image"] for _, _, r in pair_rows], .25)
        i, j, example = max((v for v in pair_rows if v[2]["image"] <= cap),
                            key=lambda v: v[2]["geometry"])
        curves[-1]["example"] = dict(image=example["image"], geometry=example["geometry"],
                                     frames=[(video[k]*255).round().astype("uint8").tolist() for k in (i, j)],
                                     outlines=[edge(labels[k]).astype("uint8").tolist() for k in (i, j)])
        # Fixed descriptor budget: four scalar means, decoded piecewise-constant.
        # Test the same length multiset in two orders; do not pick the winner per patient.
        if len(area) >= 16:
            cuts = np.rint(np.asarray([0, .25, .5, .75, 1])*len(area)).astype(int)
            shift = max(1, (cuts[2]-cuts[1])//3)
            for name, delta in (("equal", 0), ("earlier", -shift), ("later", shift)):
                b = cuts.copy()
                b[2] += delta
                yh = np.concatenate([np.repeat(area[a:z].mean(), z-a) for a,z in zip(b[:-1],b[1:])])
                budget.append(dict(case=case, strategy=name, slots=4, task="partition",
                                   mae=float(np.mean(np.abs(yh-area))),
                                   amplitude_error=float(abs(np.ptp(yh)-np.ptp(area))),
                                   max_delay_ms=float(1000*np.diff(b).max()/fps)))
        for keep in (4, 8):
            if len(area) <= keep:
                continue
            grid = np.rint(np.linspace(0, len(area)-1, keep)).astype(int)
            activity = np.r_[0., np.mean(np.abs(np.diff(video[:, ::4, ::4], axis=0)), (1, 2))]
            motion = np.argsort(activity[1:-1])[-(keep-2):] + 1
            points = [("uniform", grid), ("offline_image_motion", sorted([0, len(area)-1, *motion]))]
            for name, indices in points:
                y = np.interp(np.arange(len(area)), indices, area[indices])
                budget.append(dict(case=case, strategy=name, slots=keep, task="sampling",
                                   mae=float(np.mean(np.abs(y-area))),
                                   amplitude_error=float(abs(np.ptp(y)-np.ptp(area))),
                                   max_delay_ms=None))
    if not curves:
        raise ValueError("No dense-reference archives matched the development allowlist")
    typical = np.median([c["example"]["geometry"] for c in curves])
    example_case = min(curves, key=lambda c: abs(c["example"]["geometry"]-typical))
    for c in curves:
        if c is not example_case:
            del c["example"]
    return pairs, curves, budget, references


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--echo-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dense-root", type=Path)
    p.add_argument("--dense-dev-ids", type=Path)
    p.add_argument("--reuse-echo", type=Path, help="Sealed EchoNet measurements.json from an earlier pass")
    p.add_argument("--image-cases", type=int, default=0, help="0 = all eligible VAL image cases")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metadata-only", action="store_true")
    a = p.parse_args()
    if a.image_cases < 0:
        p.error("--image-cases must be zero or positive")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "measurements.json").exists():
        raise FileExistsError("Use a fresh output directory to avoid mixing protocols")
    sources = {name: sha256((a.echo_root / name).read_bytes()).hexdigest()
               for name in ("FileList.csv", "VolumeTracings.csv")}
    if a.reuse_echo:
        echo = json.loads(a.reuse_echo.read_text(encoding="utf-8"))
        if echo["schema"] != 1 or echo["seed"] != a.seed or echo["sources"] != sources:
            raise ValueError("Cached EchoNet pass does not match current metadata/seed")
        meta, spatial, hist, scale, pert, counts = (echo[k] for k in (
            "metadata", "spatial", "history", "scale", "perturb", "counts"))
    else:
        meta, spatial, hist, scale, pert, counts = trace_data(
            a.echo_root, a.image_cases, a.seed, a.metadata_only)
    pairs, curves, budget, refs = dense_measurements(a.dense_root, a.dense_dev_ids)
    result = dict(schema=1, seed=a.seed, metadata=meta, spatial=spatial, history=hist,
                  scale=scale, perturb=pert, dense_pairs=pairs, curves=curves, budget=budget,
                  references=refs, counts=counts, dense_cases=len(curves),
                  sources=sources)
    dest = a.output / "measurements.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    tmp.replace(dest)
    print(f"Saved {dest}: {counts}, dense={len(curves)}", flush=True)


if __name__ == "__main__":
    main()
