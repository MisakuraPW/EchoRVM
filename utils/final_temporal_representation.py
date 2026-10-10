"""Frozen, train/validation-only R1--R5 audit for the final temporal protocol.

No optimizer, backbone update, test-set access, or persistent dense feature cache.
Config may reduce (never enlarge) patient/attention/2 GiB budgets. ``recent_frames``
defaults to four local clips; ``layers`` defaults to (6, 9, 12). R5 requires a
canonical C100 reference and an explicit positive ``memory_prefix``. Tiny local
models may use smaller image/clip sizes and layer indices for contract tests.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.video_mae import tubelet_patchify
from tools.evaluate_temporal_mae import ridge_apply, ridge_fit
from utils.datasets import _as_video_tensor
from utils.downstream_datasets import _rasterize_echonet_trace


_EXITS = {"local": "local_base_outputs", "H": "frame_base_outputs",
          "F": "frame_outputs", "local_F": "local_frame_outputs"}
_LIMITS = {"max_train": 512, "max_val": 256, "max_seg_train": 128,
           "max_seg_val": 128, "attention_patients": 32}
_REQUIRED = ("geometry.csv", "spectra.csv", "cka.csv", "norms.csv", "probes.csv",
             "identity.csv", "anatomy.csv", "tokens.csv", "attention.csv",
             "memory_prediction.csv", "patient_observations.json", "metrics.json",
             "protocol.json", "spectra.png", "evidence.png")
REQUIRED_ARTIFACTS = (*_REQUIRED, "DONE")
REQUIRED = REQUIRED_ARTIFACTS


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _hash_json(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_contract(model):
    fields = ("memory_mode", "memory_compression", "memory_write_source", "memory_read_location",
              "frame_readout", "soft_beta", "memory_slots", "memory_grid", "local_frames",
              "img_size", "patch_size", "tubelet_size", "embed_dim", "in_chans", "token_grid")
    return dict(type=f"{type(model).__module__}.{type(model).__qualname__}",
                attributes={name: getattr(model, name, None) for name in fields},
                blocks=len(model.blocks))


def _write_json(path, value):
    Path(path).write_text(_json(value) + "\n", encoding="utf-8")


def _write_table(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def _config(config, model, detailed, memory):
    cfg = dict(config)
    for name, maximum in _LIMITS.items():
        cfg[name] = int(cfg.get(name, maximum))
        if not 0 < cfg[name] <= maximum:
            raise ValueError(f"{name} must be in [1,{maximum}]")
    defaults = dict(seed=42, token_reservoir=4096, anatomy_train_tokens=8192,
                    anatomy_val_tokens=8192, token_budget_bytes=2 * 1024**3,
                    tokens_per_window=64, anatomy_patches_per_view=32,
                    bootstrap_repetitions=1000, identity_tie_epsilon=1e-6,
                    low_change_quantile=.05, memory_prefix=0, memory_pca_dim=16)
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    for name in ("token_reservoir", "anatomy_train_tokens", "anatomy_val_tokens",
                 "tokens_per_window", "anatomy_patches_per_view", "bootstrap_repetitions"):
        cfg[name] = int(cfg[name])
        if cfg[name] < 1:
            raise ValueError(f"{name} must be positive")
    cfg["token_budget_bytes"] = int(cfg["token_budget_bytes"])
    if not 0 < cfg["token_budget_bytes"] <= 2 * 1024**3:
        raise ValueError("Token cache budget must be positive and at most 2 GiB")
    cfg["recent_frames"] = int(cfg.get("recent_frames", 4 * model.local_frames))
    cfg["prefix"] = int(cfg.get("prefix", 0))
    for name in ("recent_frames", "prefix", "memory_prefix"):
        value = int(cfg[name])
        if value < 0 or value % model.local_frames or (name == "recent_frames" and value == 0):
            raise ValueError(f"{name} must contain real complete local clips")
        cfg[name] = value
    cfg["layers"] = list(map(int, cfg.get("layers", (6, 9, 12)))) if detailed else []
    if len(set(cfg["layers"])) != len(cfg["layers"]) or any(
            x < 1 or x > len(model.blocks) for x in cfg["layers"]):
        raise ValueError("Requested detailed layers are absent from the backbone")
    cfg["attention_layers"] = list(map(int, cfg.get("attention_layers", (9, 12)))) if detailed else []
    tiny = model.img_size == 16 and model.local_frames == 4
    if not tiny and (cfg["layers"] != ([6, 9, 12] if detailed else []) or
                     cfg["attention_layers"] != ([9, 12] if detailed else [])):
        raise ValueError("Production detailed R uses only layers6/9/12 and attention9/12")
    if len(cfg.get("detailed_model_ids", [])) > 5:
        raise ValueError("Detailed R permits at most five registered checkpoints")
    if any(x not in cfg["layers"] for x in cfg["attention_layers"]):
        raise ValueError("Attention layers must be registered detailed layers")
    cfg["model_id"] = str(cfg.get("model_id", "unspecified"))
    allowed = {"C", "C100", "F", "F100", "B0", "B0-spatial", "B1", "B1-global"}
    if cfg.get("mixed_selected"):
        allowed.update(("B8", "mixed", "B8-mixed"))
    if detailed and cfg["model_id"] not in allowed:
        raise ValueError("Detailed R is restricted to C/F/B0/B1 and selected mixed")
    cfg["detailed"] = bool(detailed)
    cfg["with_memory_prediction"] = bool(memory or cfg.get("with_memory_prediction", False))
    if cfg["with_memory_prediction"]:
        if cfg["model_id"] not in {"B0", "B0-spatial", "B1", "B1-global", "B8", "mixed", "B8-mixed"}:
            raise ValueError("R5 is only registered for B0/B1/selected mixed")
        if cfg["model_id"] in {"B8", "mixed", "B8-mixed"} and not cfg.get("mixed_selected"):
            raise ValueError("R5 mixed requires the registered selected mixed")
        if cfg["memory_prefix"] <= 0 or cfg["memory_pca_dim"] != 16:
            raise ValueError("R5 needs a positive real prefix and fixed train PCA16")
        if cfg.get("reference_id") != "C100":
            raise ValueError("R5 requires explicitly registered canonical C100")
    if not 0 <= float(cfg["low_change_quantile"]) < 1 or float(cfg["identity_tie_epsilon"]) < 0:
        raise ValueError("Invalid fixed identity rules")
    # Optional branches must not silently turn into unregistered interventions.
    if cfg.get("perturbation") or cfg.get("probe_angles"):
        raise ValueError("Optional perturbation/angles are not enabled in this audit")
    return cfg


def _records(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256"):
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        if digest != manifest["manifest_sha256"]:
            raise ValueError("Manifest hash mismatch")
    splits = manifest.get("splits", manifest)
    if "test" in splits or "TEST" in splits:
        raise ValueError("Test-set records are forbidden in representation audit")
    rows = {split: [dict(r) for r in splits.get(split, [])] for split in ("train", "val")}
    for split, records in rows.items():
        if not records:
            raise ValueError(f"Missing {split} records")
        for r in records:
            if str(r.get("split", split)).lower() != split:
                raise ValueError("Record/split mismatch")
            r["patient"] = str(r["patient"])
            r["path"] = str(Path(r.get("path", r.get("source_path", ""))).resolve())
            if not Path(r["path"]).is_file():
                raise FileNotFoundError(r["path"])
            stat = Path(r["path"]).stat()
            if ("source_bytes" in r and stat.st_size != r["source_bytes"]) or (
                    "source_mtime_ns" in r and stat.st_mtime_ns != r["source_mtime_ns"]):
                raise ValueError("Source changed after manifest creation")
            for name in ("frames", "start"):
                if name in r and (int(r[name]) != r[name] or r[name] < 0):
                    raise ValueError(f"Invalid integer source {name}")
            if not math.isfinite(float(r["fps"])) or float(r["fps"]) <= 0:
                raise ValueError("Positive source FPS required")
        if len({r["patient"] for r in records}) != len(records):
            raise ValueError("Manifest must have one case per patient per split")
    if {r["patient"] for r in rows["train"]} & {r["patient"] for r in rows["val"]}:
        raise ValueError("Training/validation patient overlap")
    return rows


def _source_length(r):
    if Path(r["path"]).suffix.lower() == ".npy":
        raw = np.load(r["path"], mmap_mode="r", allow_pickle=False)
        try:
            return len(raw)
        finally:
            if isinstance(raw, np.memmap):
                raw._mmap.close()
    import cv2
    capture = cv2.VideoCapture(r["path"])
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open source video")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _source_provenance(r):
    """Cheap header/stat provenance, intentionally not an entire video content hash."""
    path = Path(r["path"])
    before = path.stat()
    if path.suffix.lower() == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            shape, dtype = list(raw.shape), str(raw.dtype)
        finally:
            if isinstance(raw, np.memmap):
                raw._mmap.close()
    else:
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Unable to open source video")
            shape = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))]
            dtype = "uint8"
        finally:
            capture.release()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while checking provenance")
    result = dict(path=str(path.resolve()), source_bytes=before.st_size,
                  source_mtime_ns=before.st_mtime_ns, shape=shape, dtype=dtype)
    for name in ("source_bytes", "source_mtime_ns", "dtype"):
        if name in r and r[name] != result[name]:
            raise ValueError("Source changed after manifest creation")
    if "shape" in r and list(r["shape"]) != shape:
        # AVI manifests record decoded grayscale as THW or THW1/3.
        if path.suffix.lower() == ".npy" or list(r["shape"])[:3] != shape:
            raise ValueError("Source shape changed after manifest creation")
    return result


def _read_slice(r, start, count, model):
    """Copy only the required real interval; never pad/resample past boundaries."""
    if start < 0 or count <= 0:
        raise ValueError("Invalid real source interval")
    path = Path(r["path"])
    if path.suffix.lower() == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            clip = np.array(raw[start:start + count], copy=True)
        finally:
            if isinstance(raw, np.memmap):
                raw._mmap.close()
    else:
        import cv2
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            # Decode from the beginning: codec frame seeking is not exact enough
            # for the strict source/target identity contract.
            for index in range(start + count):
                ok, frame = capture.read()
                if not ok:
                    break
                if index >= start:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        finally:
            capture.release()
        clip = np.asarray(frames)
    if len(clip) != count:
        raise ValueError("Incomplete real interval; padding is forbidden")
    was_uint8 = clip.dtype == np.uint8
    if not np.issubdtype(clip.dtype, np.number) or not np.isfinite(clip).all() or clip.min() < 0 or clip.max() > 255:
        raise ValueError("Pixels must be finite gray_repeat3 [0,1]/[0,255]")
    clip = clip.astype(np.float32)
    if was_uint8 or clip.max() > 1:
        clip /= 255.
    if clip.ndim == 4:
        if clip.shape[-1] not in (1, 3, 4):
            raise ValueError("Source must be THW or THWC")
        clip = clip[..., :3].mean(-1)
    if clip.ndim != 3:
        raise ValueError("Source must be THW or THWC")
    return _as_video_tensor(clip, count, model.img_size, channels=model.in_chans).unsqueeze(0)


def _pixels(video, patch):
    raw = tubelet_patchify(video.float().mean(2, keepdim=True), 1, patch)
    b, t, _, h, w = video.shape
    raw = raw.reshape(-1, 1, patch, patch)
    # Exactly the train-fitted 8x8 local pixel descriptor, not dense reconstruction.
    return F.adaptive_avg_pool2d(raw, (8, 8)).reshape(b, t, (h // patch) * (w // patch), 64)


def _positions(frames, patches, grid_width):
    index = torch.arange(patches)
    spatial = torch.stack((index // grid_width, index % grid_width), -1).float()
    spatial /= max(1, grid_width - 1)
    temporal = torch.arange(frames).float() / max(1, frames - 1)
    return torch.cat((temporal[:, None, None].expand(-1, patches, 1),
                      spatial[None].expand(frames, -1, -1)), -1)


def centered_cka(x, y):
    """Centered linear CKA on strictly paired observations, not unpaired Gram sets."""
    x, y = x.double(), y.double()
    if len(x) != len(y):
        raise ValueError("CKA observations must be paired")
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = torch.linalg.vector_norm(x.T @ x) * torch.linalg.vector_norm(y.T @ y)
    return None if denominator <= 1e-20 else float((x.T @ y).square().sum() / denominator)


def geometry(x):
    x = x.double()
    eigenvalues = torch.linalg.eigvalsh((x - x.mean(0)).T @ (x - x.mean(0))).clamp_min(0).flip(0)
    total = eigenvalues.sum()
    spectrum = eigenvalues / total if total > 1e-20 else eigenvalues * 0
    positive = spectrum[spectrum > 0]
    rank = float(torch.exp(-(positive * positive.log()).sum())) if len(positive) else 0.
    return dict(effective_rank=rank, n=len(x), norm_mean=float(x.norm(dim=-1).mean()),
                norm_p99=float(torch.quantile(x.norm(dim=-1), .99))), spectrum.tolist()


def gini(values):
    x = torch.sort(torch.as_tensor(values).double().flatten().clamp_min(0)).values
    if not len(x) or x.sum() <= 0:
        return 0.
    return float(((2 * torch.arange(1, len(x) + 1) - len(x) - 1) * x).sum() / (len(x) * x.sum()))


class _Budget:
    def __init__(self, maximum):
        self.maximum, self.used, self.peak = maximum, 0, 0

    def change(self, delta):
        if self.used + delta > self.maximum:
            raise MemoryError("Representation token/descriptor cache exceeds registered budget")
        self.used += delta
        self.peak = max(self.peak, self.used)


class _Reservoir:
    """Bounded random-priority sampling; all exits/labels share identical indices."""
    def __init__(self, capacity, budget, seed):
        self.capacity, self.budget = capacity, budget
        self.rng = np.random.default_rng(seed)
        self.data, self.keys, self.priority = {}, [], np.empty(0)
        self.bytes, self.seen = 0, 0

    def add(self, data, keys):
        n = len(keys)
        if not n:
            return
        data = {name: value.detach().cpu().float() for name, value in data.items()}
        if any(len(v) != n for v in data.values()) or (self.data and self.data.keys() != data.keys()):
            raise ValueError("Paired reservoir schema/index mismatch")
        priority = np.concatenate((self.priority, self.rng.random(n)))
        keep = np.argsort(priority, kind="stable")[:self.capacity]
        previous = len(self.keys)
        row_bytes = sum(v[0].numel() * v.element_size() for v in data.values())
        new_bytes = len(keep) * row_bytes
        self.budget.change(new_bytes - self.bytes)
        joined_keys = self.keys + keys
        self.keys = [joined_keys[i] for i in keep]
        self.data = {name: torch.cat((self.data[name], value), 0)[keep] if previous else value[keep]
                     for name, value in data.items()}
        self.priority, self.bytes = priority[keep], new_bytes
        self.seen += n


@torch.no_grad()
def _attention_received(module, x):
    """Independent FP32 head-mean diagnostic; never replaces training SDPA."""
    x = x.float()
    b, n, c = x.shape
    if b != 1:
        raise ValueError("Attention diagnostic requires B=1")
    bias = module.qkv.bias
    if getattr(module, "q_bias", None) is not None:
        bias = torch.cat((module.q_bias, torch.zeros_like(module.q_bias), module.v_bias))
    qkv = F.linear(x, module.qkv.weight.float(), None if bias is None else bias.float())
    q, k, _ = qkv.reshape(b, n, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    received = torch.zeros(n, device=x.device)
    entropy = x.new_zeros(())
    for query in q.split(128, dim=2):
        probability = ((query @ k.transpose(-1, -2)) * module.scale).softmax(-1)
        received += probability.sum((0, 1, 2)) / (module.num_heads * n)
        entropy += -(probability * probability.clamp_min(1e-20).log()).sum() / (module.num_heads * n)
    return received.cpu(), float(entropy)


class _Capture:
    def __init__(self, model, layers, attention_layers):
        self.model, self.layers, self.attention_layers = model, layers, attention_layers
        self.handles, self.outputs, self.norms, self.attention = [], {}, {}, {}
        self.enabled, self.with_attention = True, False

    def __enter__(self):
        for layer in self.layers:
            block = self.model.blocks[layer - 1]
            def output(module, inputs, value, layer=layer):
                if self.enabled:
                    self.outputs[layer] = value.detach().float().cpu()
            self.handles.append(block.register_forward_hook(output))
            for name in ("norm1", "norm2"):
                self._norm(getattr(block, name), f"layer{layer}.{name}")
            if layer in self.attention_layers:
                def attention(module, inputs, layer=layer):
                    if self.enabled and self.with_attention:
                        self.attention[layer] = _attention_received(module, inputs[0])
                self.handles.append(block.attn.register_forward_pre_hook(attention))
        self._norm(self.model.norm, "encoder.final_norm")
        return self

    def _norm(self, module, name):
        def capture(module, inputs, value):
            if self.enabled:
                self.norms[name] = (float(inputs[0].float().norm(dim=-1).mean()),
                                    float(value.float().norm(dim=-1).mean()))
        self.handles.append(module.register_forward_hook(capture))

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def _stream(model, video, recent, capture=None):
    if video.shape[1] % model.local_frames or video.shape[1] < recent:
        raise ValueError("Invalid complete stream/window")
    state = short = boundary = None
    fifo = deque(maxlen=recent // model.local_frames)
    attention_requested = capture.with_attention if capture else False
    for start in range(0, video.shape[1], model.local_frames):
        if start == video.shape[1] - recent:
            boundary = None if state is None else state.detach().cpu().float().clone()
        if capture:
            capture.outputs.clear()
            capture.norms.clear()
            capture.attention.clear()
            capture.with_attention = attention_requested and start == video.shape[1] - model.local_frames
        out = model.stream_clip(video[:, start:start + model.local_frames], state, short)
        state, short = out["final_state"], out["final_short"]
        exits = {}
        for name, key in _EXITS.items():
            if key not in out:
                raise ValueError(f"Missing FinalTemporalMAE frame exit {key}")
            value = out[key].detach().cpu().float()
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != model.local_frames:
                raise ValueError("Frame exits must be [1,L,P,D], not native tubelets")
            exits[name] = value
        for name, key in (("native_local", "local_features"), ("native_fused", "features")):
            value = out[key].detach().cpu().float()
            gt, gh, gw = model.token_grid
            if value.shape != (1, gt, gh * gw, model.embed_dim):
                raise ValueError("Native token exit has unexpected tubelet grid")
            exits[name] = value.repeat_interleave(model.tubelet_size, 1)
        if capture:
            gt, gh, gw = model.token_grid
            for layer, value in capture.outputs.items():
                exits[f"layer{layer}"] = value.reshape(1, gt, gh * gw, -1).repeat_interleave(model.tubelet_size, 1)
        fifo.append({name: value.mean(2).squeeze(0) for name, value in exits.items()})
    descriptors = {name: torch.cat([row[name] for row in fifo]) for name in fifo[0]}
    if boundary is None:
        slots = int(model.memory_slots)
        boundary = torch.zeros(1, slots, model.embed_dim)
    if capture:
        capture.with_attention = attention_requested
        capture.state = None if state is None else state.detach().cpu().float()
    return exits, descriptors, boundary


def _trace_labels(r, trace, model):
    import pandas as pd
    from echo_aug_validation.augment_recipes import resize_mask_with_pad
    points = trace["points"]
    if isinstance(points, dict):
        frame = pd.DataFrame(points)
    else:
        frame = pd.DataFrame(points, columns=["X1", "Y1", "X2", "Y2"])
    shape = tuple(r.get("trace_shape", (112, 112)))
    mask = _rasterize_echonet_trace(frame, shape)
    mask = torch.from_numpy(resize_mask_with_pad(mask, model.img_size)).float()[None, None]
    # Boundary is the one-pixel 3x3 morphological band, including inner/outer edge.
    dilated = F.max_pool2d(mask, 3, 1, 1)
    eroded = -F.max_pool2d(-F.pad(mask, (1, 1, 1, 1), value=0), 3, 1)
    boundary = (dilated - eroded).clamp_min(0)
    lv = F.avg_pool2d(mask, model.patch_size, model.patch_size).flatten()
    edge = F.avg_pool2d(boundary, model.patch_size, model.patch_size).flatten()
    labels = (lv >= .5).long()
    labels[edge > 0] = 2
    return labels, lv


def _bootstrap(rows, field, cfg):
    patients = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            patients[row["patient"]].append(float(row[field]))
    values = np.asarray([np.mean(v) for _, v in sorted(patients.items())])
    if not len(values):
        return dict(mean=None, low=None, high=None, n=0)
    rng = np.random.default_rng(cfg["seed"])
    samples = [float(values[rng.integers(len(values), size=len(values))].mean())
               for _ in range(cfg["bootstrap_repetitions"])]
    return dict(mean=float(values.mean()), low=float(np.quantile(samples, .025)),
                high=float(np.quantile(samples, .975)), n=len(values))


def _balanced_fit(x, labels, seed):
    rng = np.random.default_rng(seed)
    groups = [torch.where(labels == c)[0].numpy() for c in range(3)]
    if any(not len(group) for group in groups):
        return None
    count = min(map(len, groups))
    indices = np.concatenate([rng.choice(group, count, replace=False) for group in groups])
    return ridge_fit(x[indices], F.one_hot(labels[indices].long(), 3).float(), alpha=10.)


def _anatomy_scores(prediction, labels):
    scores = []
    for category in range(3):
        p, y = prediction == category, labels == category
        denominator = int(p.sum() + y.sum())
        scores.append(None if denominator == 0 else float(2 * (p & y).sum() / denominator))
    valid = [x for x in scores if x is not None]
    return dict(lv_dice=scores[1], background_dice=scores[0], boundary_dice=scores[2],
                macro_dice=float(np.mean(valid)), accuracy=float((prediction == labels).float().mean()))


def _anatomy_distances(train, val, labels):
    """Descriptive natural separation in the same train-fitted feature scale."""
    mean, scale = train.mean(0), train.std(0, unbiased=False).clamp_min(1e-5)
    x = (val - mean) / scale
    groups = [x[labels == c] for c in range(3)]
    present = [g for g in groups if len(g)]
    if len(present) < 2:
        return dict(within_class_mse=None, between_class_mse=None)
    centers = [g.mean(0) for g in present]
    within = torch.stack([(g - center).square().mean() for g, center in zip(present, centers)]).mean()
    between = torch.stack([(a - b).square().mean() for a, b in itertools.combinations(centers, 2)]).mean()
    return dict(within_class_mse=float(within), between_class_mse=float(between))


def _identity_margin(predicted, target, epsilon):
    predicted, target = predicted.float(), target.float()
    if len(predicted) != 2 or len(target) != 2:
        raise ValueError("Identity is an ordered two-frame content assignment")
    correct = (predicted - target).square().mean()
    exchanged = (predicted - target.flip(0)).square().mean()
    change = (target[0] - target[1]).square().mean()
    scale = change.clamp_min(1e-12)
    margin = float((exchanged - correct) / scale)
    return dict(margin=margin, tie=abs(margin) <= epsilon,
                correct=margin > epsilon, correct_mse=float(correct),
                swapped_mse=float(exchanged), change=float(change))


def _memory_target(reference, video, layer, device):
    values = []
    handle = reference.blocks[layer - 1].register_forward_hook(
        lambda module, inputs, output: values.append(output.detach().float().mean(1).cpu()))
    try:
        reference.encode_video(video.to(device), mask=None)
    finally:
        handle.remove()
    if len(values) != 1:
        raise ValueError("Canonical layer target must be one local encode")
    return values[0].squeeze(0)


def _memory_prediction(model, reference, records, cfg, device, budget):
    if reference is None:
        raise ValueError("R5 requires canonical C100 reference_model")
    if int(model.memory_slots) < 1:
        raise ValueError("R5 true state requires a memory-enabled model")
    layer = int(cfg.get("memory_target_layer", 9))
    if layer != 9 or len(reference.blocks) < 9:
        raise ValueError("R5 target is fixed canonical C100 layer9")
    if (reference.local_frames, reference.img_size) != (model.local_frames, model.img_size):
        raise ValueError("Canonical target/input local grid mismatch")
    if reference.embed_dim < 16:
        raise ValueError("Canonical descriptor width must support fixed PCA16")
    arrays, exclusions = {}, []
    prefix, recent, length = cfg["memory_prefix"], cfg["recent_frames"], model.local_frames
    for split, cases in records.items():
        arrays[split] = []
        for r in cases:
            start = int(r.get("start", prefix))
            if start < prefix or start + recent + length > _source_length(r):
                exclusions.append(dict(patient=r["patient"], split=split, reason="no_real_prefix_and_next_clip"))
                continue
            # Future is read separately only after past/current state construction.
            current = _read_slice(r, start - prefix, prefix + recent, model).to(device)
            _, cache, boundary = _stream(model, current, recent)
            future = _read_slice(r, start + recent, length, reference)
            before = _memory_target(reference, current[:, -length:].cpu(), layer, device)
            after = _memory_target(reference, future, layer, device)
            local = cache["local"].flatten()
            state = boundary.flatten()
            target = after - before
            budget.change((local.numel() + state.numel() + target.numel()) * 4)
            arrays[split].append((r["patient"], local, state, target, start))
    if len(arrays["train"]) < 17 or not arrays["val"]:
        return [], dict(status="insufficient_real_next_clip_patients", delta=None,
                        exclusions=exclusions, train=len(arrays["train"]), val=len(arrays["val"]))
    train_target = torch.stack([x[3] for x in arrays["train"]]).double()
    mean = train_target.mean(0)
    _, _, basis = torch.linalg.svd(train_target - mean, full_matrices=False)
    basis = basis[:16].T
    targets = {split: torch.stack([x[3] for x in rows]).double() @ basis for split, rows in arrays.items()}
    predictions = {}
    for kind in ("true_state", "zero_slots"):
        features = {split: torch.stack([torch.cat((row[1], row[2] if kind == "true_state"
                                                    else torch.zeros_like(row[2]))) for row in rows])
                    for split, rows in arrays.items()}
        predictions[kind] = ridge_apply(ridge_fit(features["train"], targets["train"], 10.), features["val"])
    rows = []
    center = targets["train"].mean(0)
    for i, record in enumerate(arrays["val"]):
        target = targets["val"][i].float()
        true = float((predictions["true_state"][i] - target).square().mean())
        zero = float((predictions["zero_slots"][i] - target).square().mean())
        rows.append(dict(patient=record[0], true_state_mse=true, zero_slots_mse=zero,
                         zero_change_mse=float(target.square().mean()), delta=zero - true,
                         target_variance=float((target - center.float()).square().mean()),
                         cache_start=record[4], state_last_source=record[4] - 1,
                         target_start=record[4] + recent, target_end=record[4] + recent + length - 1))
    denominator = sum(row["target_variance"] for row in rows)
    report = dict(status="complete", delta=_bootstrap(rows, "delta", cfg), exclusions=exclusions,
                  true_state_error=_bootstrap(rows, "true_state_mse", cfg),
                  zero_slots_error=_bootstrap(rows, "zero_slots_mse", cfg),
                  zero_change_error=_bootstrap(rows, "zero_change_mse", cfg),
                  train=len(arrays["train"]), val=len(rows), head="same alpha10 linear ridge, equal input width",
                  target="C100 layer9 next-minus-current descriptor, train PCA16",
                  history_anchor=prefix, cache_frames=recent,
                  cohort_hash=_hash_json({split: [(row[0], row[4], prefix, recent) for row in values]
                                         for split, values in arrays.items()}),
                  paired_patients=[r["patient"] for r in rows],
                  target_projection_hash=_hash_json(dict(train_patients=[r[0] for r in arrays["train"]],
                                                         mean=mean.tolist(), basis=basis.tolist())),
                  true_state_r2=None if denominator <= 1e-20 else 1 - sum(r["true_state_mse"] for r in rows) / denominator,
                  zero_slots_r2=None if denominator <= 1e-20 else 1 - sum(r["zero_slots_mse"] for r in rows) / denominator,
                  interpretation="Unlabelled cache-external information, not medical or task quality")
    return rows, report


def run_memory_prediction_audit(model, reference_model, manifest, output_dir, config, device):
    """Run only conditional R5, after the coordinator's normal task endpoints.

    Use a separate output directory from the full R audit. No geometry, EF,
    anatomy, identity, attention, or plots are computed. DONE also covers a
    completed evaluation with insufficient real next-clip patients; its metrics
    explicitly retain that status rather than claiming supporting evidence.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cfg = _config(config, model, detailed=False, memory=True)
    if reference_model is None:
        raise ValueError("R5 requires canonical C100 reference_model")
    models = [model] + ([] if reference_model is model else [reference_model])
    for item in models:
        if any(p.dtype != torch.float32 for p in item.parameters() if p.is_floating_point()):
            raise ValueError("R5 models must be FP32")
        if next(item.parameters()).device != device:
            raise ValueError("Move R5 models to requested device before calling")
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    records = _records(manifest)
    records = {split: cases[:cfg[f"max_{split}"]] for split, cases in records.items()}
    source = [{"split": split, "record": r, "provenance": _source_provenance(r)}
              for split, cases in records.items() for r in cases]
    if any(item["provenance"]["shape"][0] != int(item["record"].get(
            "raw_frames", item["record"]["frames"])) for item in source):
        raise ValueError("Manifest/source frame count mismatch")
    protocol = dict(version=2, scope="R5_only", config=cfg,
                    data_hash=_hash_json(source), manifest_metadata_hash=_hash_json(manifest),
                    model_hash=_model_hash(model), reference_hash=_model_hash(reference_model),
                    model_contract=_model_contract(model), reference_contract=_model_contract(reference_model),
                    code_hash=_file_hash(__file__),
                    ridge_code_hash=_file_hash(Path(__file__).parents[1] / "tools/evaluate_temporal_mae.py"),
                    helper_code_hashes={name: _file_hash(Path(__file__).parent / name) for name in
                                        ("datasets.py", "downstream_datasets.py")},
                    model_code_hashes={name: _file_hash(Path(__file__).parents[1] / "models" / name) for name in
                                       ("temporal_mae.py", "frame_readout.py", "vit_blocks.py", "video_mae.py", "rvm_core.py")},
                    source_hash_policy="Manifest metadata plus stat/shape/dtype; no full video/NPY content scan",
                    input="Same ordered LOCAL recent cache, plus true pre-cache state or equal-shaped zero slots",
                    history_anchor=cfg["memory_prefix"], target="Canonical C100 layer9 next-real-clip change, train PCA16",
                    label_use="No EF/seg/test labels; future clip used only for frozen diagnostic target",
                    head="Same alpha10 linear ridge, train-fitted scaling, equal input width",
                    inference="Paired patient errors/bootstrap; zero-change baseline; no clinical or task-quality claim")
    final_code = Path(__file__).parents[1] / "models/final_temporal_mae.py"
    if final_code.is_file():
        protocol["model_code_hashes"]["final_temporal_mae.py"] = _file_hash(final_code)
    protocol["hash"] = _hash_json(protocol)
    required = ("memory_prediction.csv", "metrics.json", "protocol.json")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        old = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if old.get("scope") != "R5_only" or old.get("hash") != protocol["hash"]:
            raise ValueError("R5 restart protocol/data/model hash mismatch; use a separate output directory")
        if (output / "DONE").exists():
            done = json.loads((output / "DONE").read_text(encoding="utf-8"))
            if done.get("scope") != "R5_only" or done.get("protocol_hash") != protocol["hash"] or any(
                    not (output / name).is_file() or done.get("artifacts", {}).get(name) != _file_hash(output / name)
                    for name in required):
                raise ValueError("R5 DONE cache has missing/corrupt artifacts")
            return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    elif (output / "DONE").exists():
        raise ValueError("R5 DONE without protocol is not a valid restart cache")
    _write_json(output / "protocol.json", protocol)
    modes = [{module: module.training for module in item.modules()} for item in models]
    budget = _Budget(cfg["token_budget_bytes"])
    try:
        for item in models:
            item.eval()
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            rows, report = _memory_prediction(model, reference_model, records, cfg, device, budget)
        interval = report["delta"]
        metrics = dict(status=report["status"], audit_completed=True, scope="R5_only",
                       protocol_hash=protocol["hash"], model_id=cfg["model_id"], reference_id="C100",
                       memory_prediction_delta=None if interval is None else interval["mean"],
                       memory_prediction=report, patient_observations=rows,
                       paired_consistency=dict(memory_prediction_delta=interval),
                       role_evidence=dict(role="Cache-external state information in unlabelled change prediction",
                           comparison="True pre-cache state versus equal zero slots, matched LOCAL cache and future target",
                           available=bool(rows), patient_matched=bool(rows), paired_patients=len(rows),
                           history_anchor=cfg["memory_prefix"], cohort_hash=report.get("cohort_hash"),
                           target_projection_hash=report.get("target_projection_hash"),
                           clinical_evidence=False, task_selection_authority=False),
                       coverage=dict(train=report["train"], val=report["val"]),
                       exclusions=report["exclusions"], token_cache_peak_bytes=budget.peak,
                       geometry_only_success=False, artifacts=[*required, "DONE"])
        _write_table(output / "memory_prediction.csv", rows)
        _write_json(output / "metrics.json", metrics)
        if any(_source_provenance(item["record"]) != item["provenance"] for item in source):
            raise ValueError("Source data changed during R5 audit")
        _write_json(output / "DONE", dict(scope="R5_only", protocol_hash=protocol["hash"],
                                         artifacts={name: _file_hash(output / name) for name in required}))
        return metrics
    finally:
        for mode in modes:
            for module, training in mode.items():
                module.training = training


def _plots(output, spectra, probes, identity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in sorted({r["exit"] for r in spectra}):
        selected = [r for r in spectra if r["exit"] == name]
        ax.plot([r["component"] for r in selected], np.cumsum([r["energy"] for r in selected]), label=name)
    ax.set(xlabel="Centered covariance component", ylabel="Cumulative spectral energy", ylim=(0, 1.02))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "spectra.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ef = [r for r in probes if r["task"] == "ef"]
    axes[0].bar([r["exit"] for r in ef], [r["mae"] for r in ef], color="#247c86")
    axes[0].set(ylabel="Validation patient EF MAE (pp)")
    axes[0].tick_params(axis="x", rotation=45)
    for condition, color in (("real", "#247c86"), ("repeat", "#aa545b"), ("swap", "#6f893a")):
        values = [r["margin"] for r in identity if r["exit"] == "F" and r["condition"] == condition and r["retained"]]
        if values:
            axes[1].hist(values, bins=20, histtype="step", label=condition, color=color)
    axes[1].set(xlabel="FP32 content assignment margin", ylabel="Frame-pair count (descriptive)")
    if identity:
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "evidence.png", dpi=150)
    plt.close(fig)


def run_representation_audit(model, manifest, output_dir, config, device,
                             reference_model=None, detailed=False, with_memory_prediction=False):
    """Return JSON-serializable complementary evidence and verified report artifacts.

    Restarts accept DONE only when protocol, data provenance, model, code AND artifact hashes
    match. Incomplete matching runs are recomputed; mismatched runs are rejected.
    Source manifests accept ``splits.train/val`` or top-level ``train/val`` records
    with patient/path/frames/fps/ef/traces[{frame,points}]. Geometry uses a bounded
    paired token reservoir; anatomical Dice is explicitly sampled patch Dice,
    not the coordinator's full task segmentation endpoint.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cfg = _config(config, model, detailed, with_memory_prediction)
    if cfg["with_memory_prediction"] and reference_model is None:
        raise ValueError("R5 requires canonical C100 reference_model")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Identity audit requires FP32 model parameters")
    if reference_model is not None and any(p.dtype != torch.float32 for p in reference_model.parameters() if p.is_floating_point()):
        raise ValueError("Canonical reference must be FP32")
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    records = _records(manifest)
    # Validate disjointness before truncation; otherwise an overlap can hide outside budgets.
    records = {split: rows[:cfg[f"max_{split}"]] for split, rows in records.items()}
    source = [{"split": split, "record": r, "provenance": _source_provenance(r)}
              for split, rows in records.items() for r in rows]
    protocol = dict(version=2, config=cfg, data_hash=_hash_json(source),
                    manifest_metadata_hash=_hash_json(manifest),
                    model_hash=_model_hash(model), code_hash=_file_hash(__file__),
                    model_contract=_model_contract(model),
                    ridge_code_hash=_file_hash(Path(__file__).parents[1] / "tools/evaluate_temporal_mae.py"),
                    helper_code_hashes={name: _file_hash(Path(__file__).parent / name) for name in
                                        ("datasets.py", "downstream_datasets.py")},
                    model_code_hashes={name: _file_hash(Path(__file__).parents[1] / "models" / name) for name in
                                       ("temporal_mae.py", "frame_readout.py", "vit_blocks.py", "video_mae.py", "rvm_core.py")},
                    reference_hash=None if reference_model is None else _model_hash(reference_model),
                    reference_contract=None if reference_model is None else _model_contract(reference_model),
                    input="gray_repeat3, consecutive real frames, no test data",
                    source_hash_policy="Manifest metadata plus file stat/shape/dtype, no full video/NPY scan; stat-preserving tampering is not detected",
                    labels="Official LV trace; bg/LV/3x3 one-pixel derived boundary; no myocardium",
                    sampling="Paired source-frame/patch indices; train balanced anatomy; patient bootstrap",
                    geometry="Centered covariance spectra/CKA; detailed layer native tokens repeated to source frames",
                    normalization="Ridge feature scaling fitted on train only; norm hooks labelled pre/post LN",
                    identity="Train-only image-change exclusion; FP32 real/repeat/tubelet-internal swap; fixed margins/ties",
                    evidence_boundary="Geometry alone is never quality, mechanism success, or clinical evidence",
                    optional="No probe angles or perturbation; R5 only if explicitly requested")
    final_code = Path(__file__).parents[1] / "models/final_temporal_mae.py"
    if final_code.is_file():
        protocol["model_code_hashes"]["final_temporal_mae.py"] = _file_hash(final_code)
    protocol["hash"] = _hash_json(protocol)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        old = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if old.get("hash") != protocol["hash"]:
            raise ValueError("Restart cache protocol/data/model hash mismatch; use a new output directory")
        if (output / "DONE").exists():
            done = json.loads((output / "DONE").read_text(encoding="utf-8"))
            if done.get("protocol_hash") != protocol["hash"] or any(
                    not (output / name).is_file() or done.get("artifacts", {}).get(name) != _file_hash(output / name)
                    for name in _REQUIRED):
                raise ValueError("DONE cache has missing/corrupt artifacts")
            return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    elif (output / "DONE").exists():
        raise ValueError("DONE without protocol is not a valid restart cache")
    _write_json(output / "protocol.json", protocol)
    budget = _Budget(cfg["token_budget_bytes"])
    pools = {split: _Reservoir(cfg["token_reservoir"], budget, cfg["seed"]) for split in records}
    anatomy_pools = {split: _Reservoir(cfg[f"anatomy_{split}_tokens"], budget, cfg["seed"] + 1) for split in records}
    ef, windows, norm_rows, token_rows, attention_rows, invariant_rows, exclusions = [], [], [], [], [], [], []
    state_rows, state_means = [], []
    train_changes = []
    models = [model] + ([] if reference_model is None or reference_model is model else [reference_model])
    modes = [{module: module.training for module in item.modules()} for item in models]
    for item in models:
        if next(item.parameters()).device != device:
            raise ValueError("Move audit models to requested device before calling")
    for item in models:
        item.eval()
    try:
        with torch.no_grad(), torch.autocast(device.type, enabled=False), _Capture(
                model, cfg["layers"], cfg["attention_layers"]) as capture:
            for split, cases in records.items():
                for case_index, r in enumerate(cases):
                    recent, prefix = cfg["recent_frames"], cfg["prefix"]
                    start = int(r.get("start", prefix))
                    count = _source_length(r)
                    if count != int(r.get("raw_frames", r["frames"])):
                        raise ValueError("Manifest/source frame count mismatch")
                    if start < prefix or start + recent > count:
                        exclusions.append(dict(split=split, patient=r["patient"], reason="no_complete_real_window"))
                        continue
                    video = _read_slice(r, start - prefix, prefix + recent, model).to(device)
                    capture.with_attention = split == "val" and case_index < cfg["attention_patients"]
                    exits, descriptors, _ = _stream(model, video, recent, capture)
                    if capture.state is not None:
                        state = capture.state.squeeze(0)
                        state_rows.append(dict(patient=r["patient"], split=split, slots=len(state),
                            state_slot_norm_mean=float(state.norm(dim=-1).mean()),
                            state_slot_norm_gini=gini(state.norm(dim=-1)),
                            state_slot_variance=float(state.var(0, unbiased=False).mean())))
                        if split == "val":
                            budget.change(state.shape[-1] * 4)
                            state_means.append(state.mean(0))
                    ef_value = r.get("ef")
                    if ef_value is not None:
                        if not math.isfinite(float(ef_value)):
                            raise ValueError("Nonfinite EF label")
                        features = {k: v.mean(0) for k, v in descriptors.items()}
                        budget.change(sum(v.numel() * 4 for v in features.values()))
                        ef.append(dict(split=split, patient=r["patient"], ef=float(ef_value), features=features))
                    length, patches = model.local_frames, exits["F"].shape[2]
                    pixels = _pixels(video[:, -length:].cpu(), model.patch_size).squeeze(0)
                    position = _positions(length, patches, model.img_size // model.patch_size)
                    indices = np.random.default_rng(cfg["seed"] + case_index).choice(
                        length * patches, min(cfg["tokens_per_window"], length * patches), replace=False)
                    data = {name: value.squeeze(0).flatten(0, 1)[indices] for name, value in exits.items()}
                    for name in ("H", "F"):
                        value = exits[name]
                        data[f"variation_{name}"] = (value - value.mean(1, keepdim=True)).squeeze(0).flatten(0, 1)[indices]
                    data.update(pixel=pixels.flatten(0, 1)[indices], position=position.flatten(0, 1)[indices])
                    keys = [(r["patient"], start + recent - length + int(i) // patches, int(i) % patches) for i in indices]
                    pools[split].add(data, keys)
                    windows.append(dict(split=split, patient=r["patient"], start=start, end=start + recent - 1,
                                        prefix=prefix, fps=float(r["fps"])))
                    if split == "train":
                        train_changes.extend(float((pixels[i] - pixels[i + 1]).square().mean()) for i in range(0, length - 1, 2))
                    h, f = exits["H"], exits["F"]
                    h_energy = float((h - h.mean(1, keepdim=True)).square().mean())
                    f_energy = float((f - f.mean(1, keepdim=True)).square().mean())
                    invariant_rows.append(dict(patient=r["patient"], split=split,
                        mean_max_error=float((h.mean(1) - f.mean(1)).abs().max()), H_variation_energy=h_energy,
                        F_variation_energy=f_energy, variation_ratio=None if h_energy <= 1e-20 else f_energy / h_energy))
                    for name, (pre, post) in capture.norms.items():
                        norm_rows.append(dict(patient=r["patient"], split=split, module=name, pre_ln_norm=pre,
                                              post_ln_norm=post, source_start=start + recent - length))
                    for layer, (received, entropy) in capture.attention.items():
                        top_index = int(received.argmax())
                        attention_rows.append(dict(patient=r["patient"], layer=layer, entropy=entropy,
                            entropy_over_uniform=entropy / max(1e-12, math.log(len(received))),
                            received_gini=gini(received), received_max_over_uniform=float(received.max() * len(received)),
                            source_start=start + recent - length, head_rule="mean", token_count=len(received),
                            top_patch=top_index % patches, top_tubelet=top_index // patches,
                            top_source_start=start + recent - length + model.tubelet_size * (top_index // patches),
                            attention_scope="local ViT patch sequence, independent FP32 diagnostic"))
                    for name, value in exits.items():
                        norms = value.squeeze(0).norm(dim=-1)
                        top = torch.topk(norms.flatten(), max(1, math.ceil(.01 * norms.numel()))).indices
                        for index in top.tolist():
                            token_rows.append(dict(patient=r["patient"], split=split, exit=name,
                                source_frame=start + recent - length + index // patches, patch=index % patches,
                                norm=float(norms.flatten()[index]), norm_gini=gini(norms),
                                norm_p50=float(norms.median()), norm_p99=float(torch.quantile(norms.flatten(), .99)),
                                anatomy="unlabelled", ln="post_encoder_final_LN" if name == "native_local" else
                                "native_post_fusion" if name == "native_fused" else
                                "block_output" if name.startswith("layer") else "expanded_frame_exit"))
                    if case_index >= cfg[f"max_seg_{split}"]:
                        continue
                    for trace_index, trace in enumerate(r.get("traces", [])):
                        source_frame = int(trace["frame"])
                        if source_frame != trace["frame"]:
                            raise ValueError("Trace frame must be an integer source index")
                        if not 0 <= source_frame < count:
                            raise ValueError("Official trace frame outside source")
                        labels, lv = _trace_labels(r, trace, model)
                        # Complete-context common all-offset intersection, before sampling.
                        offsets = list(range(length))
                        if any(source_frame - (recent - length + pos) < prefix or
                               source_frame - (recent - length + pos) + recent > count for pos in offsets):
                            exclusions.append(dict(patient=r["patient"], split=split, source_frame=source_frame,
                                                   reason="no_common_complete_all_position_anatomy"))
                            continue
                        if split == "train":
                            offsets = [(case_index + trace_index) % length]
                        for pos in offsets:
                            seg_start = source_frame - (recent - length + pos)
                            target_index = prefix + recent - length + pos
                            if seg_start - prefix + target_index != source_frame:
                                raise ValueError("Target/source frame index mismatch")
                            capture.with_attention = False
                            seg_video = _read_slice(r, seg_start - prefix, prefix + recent, model).to(device)
                            # Detailed layer capture remains necessary for same-layer anatomy probes.
                            capture.enabled = True
                            seg_exits, _, _ = _stream(model, seg_video, recent, capture)
                            chosen = np.random.default_rng(cfg["seed"] + case_index * 997 + trace_index).choice(
                                patches, min(patches, cfg["anatomy_patches_per_view"]), replace=False)
                            features = {name: value[0, pos, chosen] for name, value in seg_exits.items()}
                            features.update(label=labels[chosen, None].float())
                            anatomy_pools[split].add(features, [(r["patient"], source_frame, pos, int(i)) for i in chosen])
                            for name, value in seg_exits.items():
                                norms = value[0, pos].norm(dim=-1)
                                top = torch.topk(norms, max(1, math.ceil(.01 * patches))).indices
                                for i in top.tolist():
                                    token_rows.append(dict(patient=r["patient"], split=split, exit=name,
                                        source_frame=source_frame, target_offset=pos, patch=i, norm=float(norms[i]),
                                        norm_gini=gini(norms), anatomy=("bg", "LV", "boundary")[int(labels[i])],
                                        lv_fraction=float(lv[i]), ln="exit_or_block_output"))
            if not pools["train"].keys or not pools["val"].keys:
                raise ValueError("No eligible train/val representation observations")
            geometry_rows, spectra_rows, cka_rows, probes, anatomy_rows, identity_rows = [], [], [], [], [], []
            geometry_names = [name for name in pools["train"].data if name not in ("pixel", "position")]
            names = [name for name in geometry_names if not name.startswith("variation_")]
            for name in geometry_names:
                stats, spectrum = geometry(pools["val"].data[name])
                geometry_rows.append(dict(exit=name, **stats))
                spectra_rows.extend(dict(exit=name, component=i + 1, energy=value) for i, value in enumerate(spectrum))
            if state_means:
                stats, spectrum = geometry(torch.stack(state_means))
                geometry_rows.append(dict(exit="state_slot_mean", **stats))
                spectra_rows.extend(dict(exit="state_slot_mean", component=i + 1, energy=value) for i, value in enumerate(spectrum))
            for left, right in itertools.combinations(names, 2):
                cka_rows.append(dict(left=left, right=right, cka=centered_cka(
                    pools["val"].data[left], pools["val"].data[right]), paired_tokens=len(pools["val"].keys)))
            pixel_fits = {}
            patient_evidence = defaultdict(dict)
            for name in names:
                train, val = [r for r in ef if r["split"] == "train"], [r for r in ef if r["split"] == "val"]
                if train and val:
                    prediction = ridge_apply(ridge_fit(torch.stack([r["features"][name] for r in train]),
                        torch.tensor([r["ef"] for r in train]), 10.), torch.stack([r["features"][name] for r in val]))
                    errors = []
                    for row, pred in zip(val, prediction):
                        error = abs(float(pred) - row["ef"])
                        errors.append(dict(patient=row["patient"], error=error))
                        patient_evidence[row["patient"]][f"ef_{name}_error"] = error
                    score = _bootstrap(errors, "error", cfg)
                    probes.append(dict(task="ef", exit=name, alpha=10., train_patients=len(train),
                                       val_patients=len(val), mae=score["mean"], low=score["low"], high=score["high"]))
                for target in ("pixel", "position"):
                    fit = ridge_fit(pools["train"].data[name], pools["train"].data[target], 10.)
                    if target == "pixel":
                        pixel_fits[name] = fit
                    prediction = ridge_apply(fit, pools["val"].data[name])
                    error_rows = [dict(patient=key[0], error=float(err)) for key, err in zip(
                        pools["val"].keys, (prediction - pools["val"].data[target]).square().mean(-1))]
                    score = _bootstrap(error_rows, "error", cfg)
                    probes.append(dict(task=target, exit=name, alpha=10., patient_mse=score["mean"],
                                       low=score["low"], high=score["high"], val_patients=score["n"]))
                    norm_threshold = float(torch.quantile(pools["train"].data[name].norm(dim=-1), .99))
                    high = pools["val"].data[name].norm(dim=-1) >= norm_threshold
                    for subset, mask in (("high_norm", high), ("ordinary_norm", ~high)):
                        selected = [row for row, keep in zip(error_rows, mask) if keep]
                        score = _bootstrap(selected, "error", cfg)
                        probes.append(dict(task=f"{target}_{subset}", exit=name, alpha=10.,
                                           patient_mse=score["mean"], train_norm_p99=norm_threshold,
                                           low=score["low"], high=score["high"], val_patients=score["n"]))
                if anatomy_pools["train"].keys and anatomy_pools["val"].keys:
                    train_pool, val_pool = anatomy_pools["train"], anatomy_pools["val"]
                    labels = train_pool.data["label"].flatten().long()
                    fit = _balanced_fit(train_pool.data[name], labels, cfg["seed"])
                    if fit is None:
                        probes.append(dict(task="anatomy", exit=name, status="missing_train_class", alpha=10.))
                        continue
                    prediction = ridge_apply(fit, val_pool.data[name]).argmax(-1)
                    distances = _anatomy_distances(train_pool.data[name], val_pool.data[name],
                                                   val_pool.data["label"].flatten().long())
                    groups = defaultdict(list)
                    for i, key in enumerate(val_pool.keys):
                        groups[key[:3]].append(i)
                    for (patient, source_frame, offset), indices in groups.items():
                        scores = _anatomy_scores(prediction[indices], val_pool.data["label"][indices].flatten().long())
                        anatomy_rows.append(dict(patient=patient, source_frame=source_frame, offset=offset,
                                                  exit=name, sampled_patches=len(indices), **scores))
                    selected = [r for r in anatomy_rows if r["exit"] == name]
                    score = _bootstrap(selected, "macro_dice", cfg)
                    probes.append(dict(task="anatomy", exit=name, alpha=10., patient_macro_patch_dice=score["mean"],
                                       low=score["low"], high=score["high"], val_patients=score["n"], class_balance="train equal count",
                                       **distances))
            cutoff = float(np.quantile(train_changes, cfg["low_change_quantile"]))
            capture.enabled = False
            for window in [w for w in windows if w["split"] == "val"]:
                r = next(r for r in records["val"] if r["patient"] == window["patient"])
                video = _read_slice(r, window["start"] - cfg["prefix"], cfg["prefix"] + cfg["recent_frames"], model).to(device)
                length = model.local_frames
                targets = _pixels(video[:, -length:].cpu(), model.patch_size).squeeze(0)
                variants = {"real": video}
                repeat, swap = video.clone(), video.clone()
                for i in range(0, length - 1, 2):
                    j = video.shape[1] - length + i
                    repeat[:, j + 1] = video[:, j]
                    swap[:, j:j + 2] = video[:, j:j + 2].flip(1)
                variants.update(repeat=repeat, swap=swap)
                for condition, value in variants.items():
                    exits, _, _ = _stream(model, value, cfg["recent_frames"])
                    for name in _EXITS:
                        predictions = ridge_apply(pixel_fits[name], exits[name].squeeze(0))
                        for i in range(0, length - 1, 2):
                            target = targets[i:i + 2]
                            change = float((target[0] - target[1]).square().mean())
                            if condition == "swap":
                                target = target.flip(0)
                            scores = _identity_margin(predictions[i:i + 2], target, cfg["identity_tie_epsilon"])
                            identity_rows.append(dict(patient=r["patient"], exit=name, condition=condition,
                                source_a=window["end"] - length + 1 + i, source_b=window["end"] - length + 2 + i,
                                input_source_a=window["end"] - length + (2 if condition == "swap" else 1) + i,
                                input_source_b=window["end"] - length + (2 if condition == "real" else 1) + i,
                                target_source_a=window["end"] - length + (2 if condition == "swap" else 1) + i,
                                target_source_b=window["end"] - length + (1 if condition == "swap" else 2) + i,
                                offset_a=i, offset_b=i + 1, retained=change > cutoff, train_change_cutoff=cutoff, **scores))
            memory_rows, memory_report = [], dict(status="not_requested", delta=None)
            if cfg["with_memory_prediction"]:
                memory_rows, memory_report = _memory_prediction(model, reference_model, records, cfg, device, budget)
            identity_report = {}
            for name in _EXITS:
                identity_report[name] = {}
                for condition in ("real", "repeat", "swap"):
                    rows = [r for r in identity_rows if r["exit"] == name and r["condition"] == condition]
                    retained = [r for r in rows if r["retained"]]
                    identity_report[name][condition] = dict(accuracy=_bootstrap(retained, "correct", cfg),
                        tie_rate=_bootstrap(retained, "tie", cfg), margin=_bootstrap(retained, "margin", cfg),
                        retained_pairs=len(retained), excluded_pairs=len(rows) - len(retained))
            consistency = []
            for patient in sorted({r["patient"] for r in records["val"]}):
                item = dict(patient=patient, **patient_evidence[patient])
                for name in _EXITS:
                    for condition in ("real", "repeat", "swap"):
                        rows = [r for r in identity_rows if r["patient"] == patient and r["exit"] == name
                                and r["condition"] == condition and r["retained"]]
                        item[f"{name}_{condition}_margin"] = None if not rows else float(np.mean([r["margin"] for r in rows]))
                    rows = [r for r in anatomy_rows if r["patient"] == patient and r["exit"] == name]
                    item[f"anatomy_{name}"] = None if not rows else float(np.mean([r["macro_dice"] for r in rows]))
                memory_row = next((r for r in memory_rows if r["patient"] == patient), None)
                item["memory_prediction_delta"] = None if memory_row is None else memory_row["delta"]
                a, b = item["anatomy_F"], item["anatomy_H"]
                item["anatomy_F_minus_H"] = None if a is None or b is None else a - b
                a, b = item["F_real_margin"], item["F_repeat_margin"]
                item["real_minus_repeat_margin"] = None if a is None or b is None else a - b
                a, b = item["F_swap_margin"], item["F_real_margin"]
                item["swap_minus_real_margin"] = None if a is None or b is None else a - b
                for right, left in (("F", "local"), ("H", "local")):
                    a, b = item[f"anatomy_{right}"], item[f"anatomy_{left}"]
                    item[f"anatomy_{right}_minus_{left}"] = None if a is None or b is None else a - b
                a, b = item.get("ef_H_error"), item.get("ef_F_error")
                item["ef_H_minus_F_error"] = None if a is None or b is None else a - b
                consistency.append(item)
            anatomy_score = _bootstrap([r for r in anatomy_rows if r["exit"] == "F"], "macro_dice", cfg)
            metrics = dict(status="complete", protocol_hash=protocol["hash"], model_id=cfg["model_id"],
                anatomy_linear_score=anatomy_score["mean"], anatomy_linear_interval=anatomy_score,
                content_identity_accuracy=identity_report["F"]["real"]["accuracy"]["mean"], identity=identity_report,
                memory_prediction_delta=None if memory_report["delta"] is None else memory_report["delta"]["mean"],
                memory_prediction=memory_report, geometry=geometry_rows, probes=probes,
                state_statistics=state_rows,
                paired_consistency=dict(anatomy_F_minus_H=_bootstrap(consistency, "anatomy_F_minus_H", cfg),
                    anatomy_F_minus_local=_bootstrap(consistency, "anatomy_F_minus_local", cfg),
                    anatomy_H_minus_local=_bootstrap(consistency, "anatomy_H_minus_local", cfg),
                    ef_H_minus_F_error=_bootstrap(consistency, "ef_H_minus_F_error", cfg),
                    real_minus_repeat_margin=_bootstrap(consistency, "real_minus_repeat_margin", cfg),
                    swap_minus_real_margin=_bootstrap(consistency, "swap_minus_real_margin", cfg)),
                mean_invariance=invariant_rows, patient_observations=consistency,
                coverage={split: len([w for w in windows if w["split"] == split]) for split in records},
                token_cache_peak_bytes=budget.peak, reservoir={split: dict(retained=len(p.keys), seen=p.seen,
                    index_hash=_hash_json(p.keys)) for split, p in pools.items()}, exclusions=exclusions,
                anatomy_reservoir={split: dict(retained=len(p.keys), seen=p.seen,
                    index_hash=_hash_json(p.keys)) for split, p in anatomy_pools.items()},
                geometry_only_success=False, interpretation="Complementary evidence only; pair with coordinator EF/seg endpoints",
                anatomy_unit="Patient mean of source-frame/offset sampled patch macro Dice, not full pixel task Dice",
                artifacts=list(REQUIRED_ARTIFACTS))
            tables = {"geometry": geometry_rows, "spectra": spectra_rows, "cka": cka_rows, "norms": norm_rows,
                      "probes": probes, "identity": identity_rows, "anatomy": anatomy_rows, "tokens": token_rows,
                      "attention": attention_rows, "memory_prediction": memory_rows}
            for name, rows in tables.items():
                _write_table(output / f"{name}.csv", rows)
            _write_json(output / "patient_observations.json", dict(patients=consistency, windows=windows,
                        exclusions=exclusions, mean_invariance=invariant_rows))
            _write_json(output / "metrics.json", metrics)
            _plots(output, spectra_rows, probes, identity_rows)
            # A source modified during extraction cannot produce a trusted DONE.
            if any(_source_provenance(item["record"]) != item["provenance"] for item in source):
                raise ValueError("Source data changed during representation audit")
            _write_json(output / "DONE", dict(protocol_hash=protocol["hash"],
                        artifacts={name: _file_hash(output / name) for name in _REQUIRED}))
            return metrics
    finally:
        for mode in modes:
            for module, training in mode.items():
                module.training = training
