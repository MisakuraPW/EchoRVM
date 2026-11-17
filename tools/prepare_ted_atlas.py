"""Fetch a bounded TED panel; match both image endpoints to CAMUS dev before use."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

import numpy as np
from scipy.ndimage import zoom


API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/"
TED_ID = "6284ce6c73e9f00479084fbe"


def get(route, **params):
    with urllib.request.urlopen(API + route + "?" + urllib.parse.urlencode(params), timeout=30) as r:
        return json.load(r)


def download(item, path):
    destination = path / item["name"]
    if destination.is_file() and destination.stat().st_size == item["size"]:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    route = "item/" + item["_id"] + "/download"
    with urllib.request.urlopen(API + route, timeout=45) as reader, tmp.open("wb") as writer:
        remaining = item["size"]
        while remaining:
            block = reader.read(min(remaining, 1024 * 1024))
            if not block:
                raise IOError("Incomplete TED item")
            writer.write(block)
            remaining -= len(block)
        if reader.read(1):
            raise IOError("TED item exceeded declared size")
    tmp.replace(destination)
    return destination


def read_cfg(path):
    return dict(s.split(":", 1) for s in path.read_text().splitlines() if ":" in s)


def resize(x, hw=48, order=1):
    return zoom(x.astype(np.float32), (hw/x.shape[0], hw/x.shape[1]), order=order)


def normalize(x):
    x = x.flatten().astype(np.float32)
    x -= x.mean()
    return x / max(np.linalg.norm(x), 1e-8)


def orientation_pairs(x):
    for transpose in (False, True):
        x0 = x.T if transpose else x
        for y, h in ((False, False), (True, False), (False, True), (True, True)):
            yield x0[::(-1 if y else 1), ::(-1 if h else 1)]


def load_camus(root):
    import SimpleITK as sitk

    splits = root / "database_split"
    dev = set()
    for split in ("training", "validation"):
        dev.update((splits / f"subgroup_{split}.txt").read_text().split())
    # Verify split identities without reading any held-out annotations.
    test = set((splits / "subgroup_testing.txt").read_text().split())
    if dev & test or not dev:
        raise ValueError("CAMUS development split intersects TEST or is empty")
    names, imgs, fps, efs = [], [], [], []
    for name in sorted(dev):
        if not re.fullmatch(r"patient\d+", name):
            raise ValueError("Invalid CAMUS development ID")
        folder = root / "database_nifti" / name
        cfg = read_cfg(folder / "Info_4CH.cfg")
        if "FrameRate" not in cfg:
            continue
        frames = [sitk.GetArrayFromImage(sitk.ReadImage(str(folder / f"{name}_4CH_{s}.nii.gz"))).squeeze()
                  for s in ("ED", "ES")]
        imgs.append(np.stack([normalize(resize(x)) for x in frames]))
        names.append(name)
        fps.append(float(cfg["FrameRate"]))
        efs.append(float(cfg["EF"]) if "EF" in cfg else None)
    if not names:
        raise ValueError("No CAMUS development endpoints with FPS")
    return names, np.stack(imgs), fps, efs


def match_endpoints(ed, es, references):
    scores = []
    for a, b in zip(orientation_pairs(ed), orientation_pairs(es)):
        d = (references * np.stack([normalize(resize(a)), normalize(resize(b))])).sum(axis=-1)
        scores.append(d.min(axis=1))
    values = np.amax(scores, axis=0)
    idx = np.argsort(values)[::-1]
    return int(idx[0]), float(values[idx[0]]), float(values[idx[0]]-values[idx[1]]) if len(idx)>1 else 1.


def main():
    import SimpleITK as sitk

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camus-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", type=int, default=16, help="Number of candidate TED items (before validation)")
    p.add_argument("--max-mib", type=int, default=768)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    prepared = a.output / "dense"
    prepared.mkdir(exist_ok=True)
    if any(prepared.iterdir()):
        raise ValueError("Use a fresh output directory")
    folders = get("folder", parentType="folder", parentId=TED_ID, limit=200)
    names, references, fps, efs = load_camus(a.camus_root)
    accepted, records = [], []
    budget = a.max_mib * 2**20
    # Predefined order, limited bytes, and no performance-based selection.
    folds = [z for z in sorted(folders, key=lambda z: z["name"]) if re.fullmatch(r"patient\d+", z["name"])]
    chosen = np.random.default_rng(42).permutation(len(folds))[:a.cases]
    for ix in chosen:
        f = folds[ix]
        patient = f["name"]
        files = {i["name"]: i for i in get("item", folderId=f["_id"], limit=100)}
        expected = [f"{patient}_4CH_{name}" for name in
                    ("info.cfg", "sequence.mhd", "sequence.raw", "sequence_gt.mhd", "sequence_gt.raw")]
        items = [files[n] for n in expected]
        size = sum(i["size"] for i in items)
        if size > budget:
            continue
        budget -= size
        temp = a.output / patient
        temp.mkdir(exist_ok=True)
        print(f"TED candidate {len(records)+1}; {size/2**20:.0f} MiB", flush=True)
        try:
            with ThreadPoolExecutor(4) as pool:
                cfg, mhd, _, gt, _ = list(pool.map(lambda i: download(i, temp), items))
            info = read_cfg(cfg)
            video = sitk.GetArrayFromImage(sitk.ReadImage(str(mhd)))
            ed, es = int(info["ED"])-1, int(info["ES"])-1
            if not 0 <= ed < es < len(video):
                raise ValueError("Bad TED phase indices")
            idx, similarity, margin = match_endpoints(video[ed], video[es], references)
            if similarity < .995 or margin < .005 or names[idx] in accepted:
                records.append(dict(folder=patient, match=False, score=similarity, margin=margin))
                continue
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(gt))) == 1
            if seg.shape != video.shape or not seg.any(axis=(1, 2)).all():
                raise ValueError("Dense mask/video mismatch")
            # Full native frames remain server-side; the inexpensive common grid
            # for measurement is explicitly recorded and not a clinical spacing.
            x = np.stack([resize(im, 112) for im in video]).round().clip(0, 255).astype("uint8")
            y = np.stack([resize(im, 112, 0) for im in seg]).astype("uint8")
            np.savez_compressed(prepared / f"{names[idx]}.npz", video=x, mask=y, fps=fps[idx],
                                ef=efs[idx] if efs[idx] is not None else np.nan)
            accepted.append(names[idx])
            records.append(dict(folder=patient, match=True, score=similarity, margin=margin,
                                metadata="matched two images against CAMUS TRAIN/VAL only"))
        except (OSError, ValueError) as exc:
            records.append(dict(folder=patient, match=False, error=str(exc)))
        finally:
            shutil.rmtree(temp)
    (a.output / "dev_ids.txt").write_text("\n".join(accepted), encoding="ascii")
    (a.output / "matches.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"TED verified development panel: {len(accepted)} of {len(records)} attempted", flush=True)
    if not accepted:
        raise SystemExit("No TED endpoints could be independently matched to the development set")


if __name__ == "__main__":
    main()
