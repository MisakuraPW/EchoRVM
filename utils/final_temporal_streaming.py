"""Read-only, real-video streaming contract checks for the final temporal study."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import torch

from models.final_temporal_mae import load_final_model
from utils.final_temporal_representation import (
    _file_hash, _hash_json, _model_contract, _model_hash, _read_slice,
    _source_provenance, _write_json, _write_table,
)


REQUIRED_ARTIFACTS = ("clips.csv", "patients.csv", "prefix_checks.csv", "checks.csv",
                      "metrics.json", "protocol.json", "DONE")


def _tensor_digest(value):
    if value is None:
        return None
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _storage_bytes(values):
    storages = {}
    for value in values:
        if value is not None:
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class _ClipStream:
    """One patient, one state, four ordered pooled clips; no dense token retention."""
    capacity = 4

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self, patient=None):
        self.patient, self.state, self.next_index = patient, None, None
        self.entries = deque(maxlen=self.capacity)
        self.observed_clips = 0

    def update(self, clip, patient, indices):
        if self.model.training:
            raise RuntimeError("Streaming audit requires model.eval()")
        length = self.model.local_frames
        if clip.shape != (1, length, self.model.in_chans, self.model.img_size, self.model.img_size):
            raise ValueError("Streaming requires one complete real clip; no padding")
        indices = torch.as_tensor(indices, device=clip.device)
        if indices.dtype not in (torch.int32, torch.int64) or indices.shape != (1, length):
            raise ValueError("Source indices must be [1,L] integers")
        if bool((indices < 0).any()) or bool((indices[:, 1:] - indices[:, :-1] != 1).any()):
            raise ValueError("Source frames must be nonnegative and consecutive")
        if self.patient is not None and patient != self.patient:
            raise ValueError("Patient changed: explicit reset required")
        if self.next_index is not None and int(indices[0, 0]) != self.next_index:
            raise ValueError("Duplicated/skipped/reordered source frames require reset")
        _sync(clip.device)
        begin = time.perf_counter()
        out = self.model.stream_clip(clip, self.state)
        _sync(clip.device)
        self.last_model_seconds = time.perf_counter() - begin
        final = out.get("frame_outputs")
        patches = self.model.token_grid[1] * self.model.token_grid[2]
        if final is None or final.shape != (1, length, patches, self.model.embed_dim):
            raise ValueError("Explicit frame_outputs must be [1,L,P,D]; never expand frames twice")
        native = out.get("features")
        if native is None or native.shape != (1, self.model.token_grid[0], patches, self.model.embed_dim):
            raise ValueError("Native features must remain tubelets")
        if not bool(torch.isfinite(final).all()):
            raise ValueError("Nonfinite frame output")
        state = out.get("final_state")
        slots = self.model.memory_slots
        if (slots == 0 and state is not None) or (slots and (state is None or
                state.shape != (1, slots, self.model.embed_dim))):
            raise ValueError("State does not match actual memory slots")
        if out.get("final_short") is not None:
            raise ValueError("Final study does not use dual short state")
        self.patient, self.state = patient, None if state is None else state.detach()
        self.entries.append(dict(descriptors=final.mean(2).detach(), indices=indices.long().clone()))
        self.next_index = int(indices[0, -1]) + 1
        self.observed_clips += 1
        return out

    def read(self):
        if not self.entries:
            raise RuntimeError("Empty stream")
        return dict(descriptors=torch.cat([r["descriptors"] for r in self.entries], 1),
                    source_indices=torch.cat([r["indices"] for r in self.entries], 1),
                    final_state=self.state, clips=len(self.entries),
                    warmup=len(self.entries) < self.capacity)

    def tensor_bytes(self):
        return _storage_bytes([self.state, *[v for r in self.entries for v in r.values()]])


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _close(actual, expected, atol, rtol, label):
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)
    return float((actual - expected).abs().max()) if actual.numel() else 0.


def _cases(manifest):
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    splits = manifest.get("splits", manifest)
    if "test" in splits or "TEST" in splits:
        raise ValueError("Test records are forbidden in streaming audit")
    if manifest.get("manifest_sha256"):
        payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        if digest != manifest["manifest_sha256"]:
            raise ValueError("Manifest hash mismatch")
    train = {str(r["patient"]) for r in splits.get("train", [])}
    val = [dict(r) for r in splits.get("val", [])]
    ids = [str(r["patient"]) for r in val]
    if len(set(ids)) != len(ids) or train.intersection(ids):
        raise ValueError("Duplicate validation patients or train/val overlap")
    for r in val:
        r["patient"] = str(r["patient"])
        r["path"] = str(Path(r.get("path", r.get("source_path", ""))).resolve())
        if not math.isfinite(float(r["fps"])) or float(r["fps"]) <= 0:
            raise ValueError("Positive source FPS is required")
    return manifest, val


def run_streaming_audit(job, manifest, device):
    """Validate actual consecutive VAL clips, at most two patients x 512 frames.

    Job requires checkpoint/output_dir. Optional smoke, max_frames (<=512),
    max_cases (<=2), seed, model_overrides, atol/rtol are protocol-hashed. Formal
    coverage means two actual 512-frame videos; smaller observations are valid
    contract evidence but explicitly limited. Smoke uses up to eight local clips
    by default and still requires more than one real complete clip. Tails are
    left unprocessed, never padded. Offline verification tensors are temporary,
    separately reported and excluded from the live state's fixed storage bound.
    """
    job = dict(job)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    smoke = bool(job.get("smoke", False))
    max_cases = int(job.get("max_cases", 2))
    if not 1 <= max_cases <= 2:
        raise ValueError("Streaming audit permits at most two validation cases")
    model, model_config, load_report = load_final_model(
        job["checkpoint"], overrides=job.get("model_overrides"), seed=int(job.get("seed", 42)))
    model = model.to(device).eval()
    length = model.local_frames
    maximum = int(job.get("max_frames", min(512, 8 * length) if smoke else 512))
    if not length * 2 <= maximum <= 512:
        raise ValueError("max_frames must allow >one real clip and be at most512")
    atol, rtol = float(job.get("atol", 1e-5)), float(job.get("rtol", 1e-4))
    if not all(math.isfinite(v) and 0 <= v <= 1e-3 for v in (atol, rtol)):
        raise ValueError("Streaming comparison tolerances must be finite in [0,1e-3]")
    manifest, cases = _cases(manifest)
    sources, excluded, eligible = [], [], []
    for r in cases:
        provenance = _source_provenance(r)
        frames = provenance["shape"][0]
        if frames != int(r.get("raw_frames", r["frames"])):
            raise ValueError("Manifest/source frame count mismatch")
        start = r.get("stream_start", 0)
        if int(start) != start or not 0 <= start <= frames:
            raise ValueError("Invalid continuous source start")
        source = dict(record=r, provenance=provenance)
        sources.append(source)
        available = frames - int(start)
        count = min(maximum, available) // length * length
        if count < 2 * length:
            excluded.append(dict(patient=r["patient"], reason="fewer_than_two_real_complete_clips",
                                 available_frames=available, complete_frames=count))
            continue
        eligible.append(dict(record=r, start=int(start), available=available, frames=count))
    # Selection is blind to task scores: maximize real length, manifest order breaks ties.
    selected = sorted(eligible, key=lambda r: -r["frames"])[:max_cases]
    root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__), root / "utils/final_temporal_representation.py",
                  root / "utils/datasets.py", root / "models/final_temporal_mae.py",
                  root / "models/temporal_mae.py", root / "models/frame_readout.py",
                  root / "models/video_mae.py", root / "models/vit_blocks.py", root / "models/rvm_core.py"]
    protocol = dict(version=2, scope="streaming", job={k: v for k, v in job.items() if k != "output_dir"},
                    manifest_hash=_hash_json(manifest), source_hash=_hash_json(sources),
                    checkpoint_hash=_file_hash(job["checkpoint"]), model_hash=_model_hash(model),
                    model_contract=_model_contract(model), model_config=model_config,
                    code_hashes={str(p.relative_to(root)): _file_hash(p) for p in code_paths},
                    device=str(device), precision="FP32", fifo_capacity=4, max_frames=maximum,
                    selected=[dict(patient=r["record"]["patient"], start=r["start"], frames=r["frames"]) for r in selected],
                    sampling="VAL only; largest actual real length first, manifest-order ties; no padding/repeats",
                    source_hash_policy="Metadata/stat/shape/dtype only; no full video/NPY scan",
                    comparison="Offline diagnostic explicit frame_outputs, not a second frame expansion",
                    causality="Clip-internal bidirectional; past-prefix invariance across complete clip arrivals",
                    storage="Live recurrent state plus four pooled frame-descriptor clips and source indices only",
                    limitations="Finite-length engineering audit, not clinical readiness or arbitrarily long accuracy")
    protocol["hash"] = _hash_json(protocol)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    required = REQUIRED_ARTIFACTS[:-1]
    if (output / "protocol.json").exists():
        old = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        if old.get("hash") != protocol["hash"]:
            raise ValueError("Streaming restart protocol/data/model hash mismatch")
        if (output / "DONE").exists():
            done = json.loads((output / "DONE").read_text(encoding="utf-8"))
            if done.get("protocol_hash") != protocol["hash"] or any(
                    not (output / name).is_file() or done.get("artifacts", {}).get(name) != _file_hash(output / name)
                    for name in required):
                raise ValueError("Streaming DONE has missing/corrupt artifacts")
            return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    elif (output / "DONE").exists():
        raise ValueError("Streaming DONE without protocol")
    _write_json(output / "protocol.json", protocol)
    clips, patients, prefix_rows, checks = [], [], [], []
    stream = _ClipStream(model)

    def check(patient, name, actual, expected=None):
        if expected is None:
            if not actual:
                raise AssertionError(f"{patient}: {name}")
            error = None
        else:
            error = _close(actual, expected, atol, rtol, f"{patient}: {name}")
        checks.append(dict(patient=patient, check=name, passed=True, max_abs_error=error))
        return error

    try:
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for case in selected:
                r, start, count = case["record"], case["start"], case["frames"]
                patient = r["patient"]
                begin = time.perf_counter()
                video = _read_slice(r, start, count, model)
                read_seconds = time.perf_counter() - begin
                # Independent legal offline unroll, bounded by the same real <=512 interval.
                offline = model.diagnostic_features(video.to(device))
                if "frame_outputs" not in offline or offline["frame_outputs"].shape != (
                        1, count, model.token_grid[1] * model.token_grid[2], model.embed_dim):
                    raise AssertionError("Offline diagnostic must expose final frame_outputs exactly once")
                reference = offline["frame_outputs"].cpu()
                state_means = offline["states"].cpu()
                offline_bytes = _storage_bytes(offline.values())
                del offline
                stream.reset(patient)
                check(patient, "patient_reset_empty_state_fifo", stream.state is None and not stream.entries)
                last_state_digest, sizes, errors = None, [], []
                h2d_total = compute_total = readout_total = 0.
                state_bytes = model.memory_slots * model.embed_dim * next(model.parameters()).element_size()
                descriptor_bytes = length * model.embed_dim * next(model.parameters()).element_size()
                bound = state_bytes + 4 * (descriptor_bytes + length * 8)
                for offset in range(0, count, length):
                    _sync(device)
                    begin = time.perf_counter()
                    clip = video[:, offset:offset + length].to(device)
                    _sync(device)
                    transfer_seconds = time.perf_counter() - begin
                    indices = torch.arange(start + offset, start + offset + length, device=device)[None]
                    before_digest = _tensor_digest(stream.state)
                    if offset:
                        check(patient, "state_continuity", before_digest == last_state_digest)
                    begin = time.perf_counter()
                    out = stream.update(clip, patient, indices)
                    _sync(device)
                    update_seconds = time.perf_counter() - begin
                    compute_seconds = stream.last_model_seconds
                    final = out["frame_outputs"]
                    error = check(patient, "offline_stream_final_frames", final, reference[:, offset:offset + length])
                    errors.append(error)
                    if stream.state is not None:
                        check(patient, "offline_stream_state_mean", stream.state.mean(1), state_means[:, offset // length])
                    last_state_digest = _tensor_digest(stream.state)
                    begin = time.perf_counter()
                    cached = stream.read()
                    _sync(device)
                    readout_seconds = time.perf_counter() - begin
                    check(patient, "read_does_not_update_state", _tensor_digest(stream.state) == last_state_digest)
                    expected_start = max(0, offset + length - 4 * length)
                    expected_indices = torch.arange(start + expected_start, start + offset + length)[None]
                    check(patient, "fifo_source_order_identity", torch.equal(cached["source_indices"].cpu(), expected_indices))
                    check(patient, "fifo_descriptor_identity", cached["descriptors"],
                          reference[:, expected_start:offset + length].mean(2))
                    live_bytes = stream.tensor_bytes()
                    sizes.append(live_bytes)
                    check(patient, "bounded_state_fifo_storage", live_bytes <= bound)
                    check(patient, "one_update_per_primary_clip", stream.observed_clips == offset // length + 1)
                    if offset >= 4 * length and model.memory_slots:
                        check(patient, "state_persists_after_fifo_eviction", before_digest is not None and
                              before_digest == clips[-1]["output_state_digest"])
                    clips.append(dict(patient=patient, clip=offset // length + 1,
                        source_start=start + offset, source_end=start + offset + length - 1,
                        source_clip_digest=_tensor_digest(clip), frame_exit_length=final.shape[1],
                        native_tubelets=out["features"].shape[1], fifo_clips=cached["clips"],
                        fifo_frames=cached["descriptors"].shape[1], fifo_first_source=int(cached["source_indices"][0, 0]),
                        warmup=cached["warmup"], state_slots=model.memory_slots,
                        input_state_digest=before_digest, output_state_digest=last_state_digest,
                        persistent_tensor_bytes=live_bytes, persistent_bound_bytes=bound,
                        offline_max_abs_error=error, model_seconds=compute_seconds, h2d_seconds=transfer_seconds,
                        stream_update_seconds=update_seconds,
                        validation_fifo_update_seconds=max(0., update_seconds - compute_seconds),
                        readout_seconds=readout_seconds, fps=float(r["fps"]),
                        clip_end_max_wait_seconds=(length - 1) / float(r["fps"])))
                    for trace in r.get("traces", []):
                        source_frame = trace["frame"]
                        if int(source_frame) != source_frame:
                            raise ValueError("Target source index must be an integer")
                        if start + offset <= source_frame < start + offset + length:
                            target_offset = int(source_frame) - start - offset
                            check(patient, "target_source_frame_identity", final[:, target_offset],
                                  reference[:, int(source_frame) - start])
                            checks[-1].update(source_frame=int(source_frame), target_offset=target_offset,
                                               observed_at_source=start + offset + length - 1)
                    h2d_total += transfer_seconds
                    compute_total += compute_seconds
                    readout_total += readout_seconds
                    del out, final, cached, clip
                if len(sizes) >= 4:
                    check(patient, "storage_plateau_after_four_clips", len(set(sizes[3:])) == 1)
                # Short prefixes use only arrived real frames. The longer reference
                # includes future clips, which must not alter any earlier exit.
                requested = sorted(set([length * n for n in (1, 2, 3, 4, 5, 8, 12, 32)] + [count]))
                for prefix in requested:
                    if prefix > count:
                        prefix_rows.append(dict(patient=patient, frames=prefix, status="not_available", available_frames=count))
                        continue
                    result = model.diagnostic_features(video[:, :prefix].to(device))
                    error = check(patient, "causal_prefix_invariance", result["frame_outputs"], reference[:, :prefix])
                    prefix_rows.append(dict(patient=patient, frames=prefix, status="passed", available_frames=count,
                                            max_abs_error=error, physical_duration_seconds=prefix / float(r["fps"])))
                    del result
                first = video[:, :length].to(device)
                stream.reset(patient)
                reset_out = stream.update(first, patient, torch.arange(start, start + length, device=device)[None])
                check(patient, "explicit_reset_matches_fresh_first_clip", reset_out["frame_outputs"], reference[:, :length])
                # Invalid patient/index updates must be rejected before model/state mutation.
                stable_digest, stable_calls = _tensor_digest(stream.state), stream.observed_clips
                invalid = [("patient_change_rejected", patient + "__different", torch.arange(start + length, start + 2 * length)[None]),
                           ("duplicate_frames_rejected", patient, torch.arange(start, start + length)[None]),
                           ("skipped_frames_rejected", patient, torch.arange(start + length + 1, start + 2 * length + 1)[None]),
                           ("reordered_frames_rejected", patient, torch.arange(start + length, start + 2 * length).flip(0)[None]),
                           ("padded_indices_rejected", patient, torch.full((1, length), -1, dtype=torch.long))]
                for name, bad_patient, indices in invalid:
                    rejected = False
                    try:
                        stream.update(first, bad_patient, indices.to(device))
                    except ValueError:
                        rejected = True
                    check(patient, name, rejected and _tensor_digest(stream.state) == stable_digest and
                          stream.observed_clips == stable_calls)
                check(patient, "incomplete_tail_rejected", _reject_tail(stream, first[:, :-1], patient, start + length, device))
                patients.append(dict(patient=patient, source_start=start, source_end=start + count - 1,
                    source_frames=case["available"], observed_frames=count, observed_clips=count // length,
                    excluded_tail_frames=min(maximum, case["available"]) - count,
                    unobserved_source_frames=case["available"] - count, fps=float(r["fps"]),
                    physical_duration_seconds=count / float(r["fps"]),
                    first_to_last_frame_seconds=(count - 1) / float(r["fps"]),
                    clip_end_max_wait_seconds=(length - 1) / float(r["fps"]),
                    source_read_seconds=read_seconds, model_seconds=compute_total, h2d_seconds=h2d_total,
                    fifo_readout_seconds=readout_total, model_frames_per_second=count / max(compute_total, 1e-12),
                    persistent_peak_bytes=max(sizes), persistent_bound_bytes=bound,
                    storage_plateau_checked=len(sizes) >= 4, fifo_eviction_checked=count > 4 * length,
                    state_persistence="checked" if model.memory_slots and count > 4 * length else
                        "not_applicable_memory_none" if not model.memory_slots else "limited_no_fifo_eviction",
                    offline_max_abs_error=max(errors), offline_temporary_tensor_bytes=offline_bytes,
                    offline_reference_tensor_bytes=_storage_bytes((reference, state_means)),
                    formal_512_covered=count == 512))
                stream.reset()
                del reset_out, first, reference, state_means, video
        for source in sources:
            if _source_provenance(source["record"]) != source["provenance"]:
                raise ValueError("Source changed during streaming audit")
        full = len(patients) == 2 and all(r["formal_512_covered"] for r in patients)
        smoke_complete = smoke and len(patients) == max_cases and all(r["observed_clips"] >= 2 for r in patients)
        metrics = dict(status="complete" if full or smoke_complete else "limited", protocol_hash=protocol["hash"],
            audit_completed=True, checks_passed=True if patients else None, formal_512_coverage=full, smoke=smoke,
            patients=patients, exclusions=excluded, observed_cases=len(patients), maximum_total_frames_per_case=512,
            fifo_capacity=4, local_frames=length, persistent_peak_bytes=max([r["persistent_peak_bytes"] for r in patients], default=0),
            patient_reset_checked=bool(patients), cross_patient_reset_checked=len(patients) > 1,
            first_three_clip_coverage={str(k): sum(r["observed_clips"] >= k for r in patients) for k in (1, 2, 3)},
            load_report=load_report, artifacts=list(REQUIRED_ARTIFACTS),
            limitation=None if full else "Only the listed real frames/patients were tested; no fabricated 512-frame sequence",
            latency_note="Clip-internal bidirectional; clip-end collection wait excludes compute/H2D; not zero-latency per-frame",
            storage_note="Measured live state/FIFO tensor storages; temporary offline reference and weights are separate",
            clinical_deployment_claim=False, arbitrary_length_accuracy_claim=False)
        _write_json(output / "metrics.json", metrics)
        for name, rows in (("clips", clips), ("patients", patients), ("prefix_checks", prefix_rows), ("checks", checks)):
            _write_table(output / f"{name}.csv", rows)
        _write_json(output / "DONE", dict(protocol_hash=protocol["hash"],
                                         artifacts={name: _file_hash(output / name) for name in required}))
        return metrics
    except Exception as error:
        _write_json(output / "metrics.json", dict(status="failed", audit_completed=False,
                    protocol_hash=protocol["hash"], error=f"{type(error).__name__}: {error}",
                    patients=patients, exclusions=excluded, checks_passed=False))
        for name, rows in (("clips", clips), ("patients", patients), ("prefix_checks", prefix_rows), ("checks", checks)):
            _write_table(output / f"{name}.csv", rows)
        raise


def _reject_tail(stream, clip, patient, start, device):
    previous, calls = _tensor_digest(stream.state), stream.observed_clips
    try:
        stream.update(clip, patient, torch.arange(start, start + clip.shape[1], device=device)[None])
    except ValueError:
        return _tensor_digest(stream.state) == previous and stream.observed_clips == calls
    return False
