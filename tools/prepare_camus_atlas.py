"""Convert official CAMUS development half-sequences to an explicitly proxy panel."""

import argparse
from pathlib import Path
import re

import numpy as np

from prepare_ted_atlas import read_cfg, resize


def main():
    import SimpleITK as sitk

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camus-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    splits = a.camus_root / "database_split"
    names = (splits / "subgroup_validation.txt").read_text().split()
    test = set((splits / "subgroup_testing.txt").read_text().split())
    if set(names) & test or len(names) != len(set(names)):
        raise ValueError("Invalid CAMUS validation split")
    dense = a.output / "dense"
    dense.mkdir(parents=True, exist_ok=False)
    accepted = []
    for name in sorted(names):
        if not re.fullmatch(r"patient\d+", name):
            raise ValueError("Invalid CAMUS identifier")
        folder = a.camus_root / "database_nifti" / name
        cfg = read_cfg(folder / "Info_4CH.cfg")
        x, y = [sitk.GetArrayFromImage(sitk.ReadImage(str(folder / f"{name}_4CH_half_sequence{s}.nii.gz")))
                for s in ("", "_gt")]
        fps = float(cfg.get("FrameRate", "nan"))
        if x.ndim != 3 or x.shape != y.shape or len(x) < 8 or not np.isfinite(fps) or fps <= 0:
            continue
        if (not np.isfinite(x).all() or x.min() < 0 or x.max() > 255 or not (y == 1).any(axis=(1, 2)).all()):
            continue
        image = np.stack([resize(frame, 112) for frame in x])
        masks = np.stack([resize(frame == 1, 112, 0) for frame in y]).astype("uint8")
        np.savez_compressed(dense / f"{name}.npz", video=image if image.max() <= 1 else image/255,
                            mask=masks, fps=fps)
        accepted.append(name)
    if not accepted:
        raise ValueError("No valid CAMUS validation half-sequences")
    (a.output / "dev_ids.txt").write_text("\n".join(accepted), encoding="ascii")
    print(f"CAMUS half-cycle proxy cases: {len(accepted)} / {len(names)}", flush=True)


if __name__ == "__main__":
    main()
