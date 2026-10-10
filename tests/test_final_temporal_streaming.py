"""CPU/file-backed checks of real bounded streaming, no external dataset."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.final_temporal_mae import FinalTemporalMAE
from utils.final_temporal_streaming import _ClipStream, run_streaming_audit


def tiny_config(mode="spatial", readout="factorized", late=False):
    return dict(img_size=16, patch_size=4, local_frames=4, clip_count=4,
                tubelet_size=2, in_chans=3, embed_dim=24, depth=1, num_heads=3,
                decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=3,
                mask_ratio=.5, norm_pix_loss=False, memory_mode=mode, memory_grid=2,
                core_depth=1, frame_readout=readout, dynamic_rank=6,
                memory_write_source="local" if late else "fused",
                memory_read_location="frames" if late else "tokens")


def fixture(root, lengths=(37, 29), mode="spatial", readout="factorized", late=False):
    torch.manual_seed(42)
    cfg = tiny_config(mode, readout, late)
    model = FinalTemporalMAE(**cfg)
    checkpoint = root / "tiny.pt"
    torch.save(dict(config=dict(model=cfg), model_state_dict=model.state_dict(), epoch=100), checkpoint)
    rng = np.random.default_rng(7)
    records = []
    for i, length in enumerate(lengths):
        path = root / f"val_{i}.npy"
        np.save(path, rng.integers(0, 256, (length, 16, 16), dtype=np.uint8))
        stat = path.stat()
        records.append(dict(patient=f"val_{i}", path=str(path), frames=length, fps=50.,
                            shape=[length, 16, 16], dtype="uint8", source_bytes=stat.st_size,
                            source_mtime_ns=stat.st_mtime_ns, traces=[dict(frame=min(6, length - 1))]))
    # The training record is never opened, including its intentionally missing file.
    manifest = dict(train=[dict(patient="unread_train", path=str(root / "DO_NOT_OPEN.npy"))], val=records)
    job = dict(checkpoint=str(checkpoint), output_dir=str(root / "audit"), smoke=True,
               max_frames=36, max_cases=2)
    return model, manifest, job


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class StreamingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_file_backed_smoke_contracts_target_indices_warmup_storage_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root)
            report = run_streaming_audit(job, manifest, "cpu")
            json.dumps(report, allow_nan=False)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["smoke"])
            self.assertFalse(report["formal_512_coverage"])
            self.assertTrue(report["cross_patient_reset_checked"])
            self.assertEqual([r["observed_frames"] for r in report["patients"]], [36, 28])
            self.assertEqual(report["first_three_clip_coverage"], {"1": 2, "2": 2, "3": 2})
            for patient in report["patients"]:
                self.assertLessEqual(patient["persistent_peak_bytes"], patient["persistent_bound_bytes"])
                self.assertTrue(patient["fifo_eviction_checked"])
                self.assertEqual(patient["state_persistence"], "checked")
                self.assertEqual(patient["physical_duration_seconds"], patient["observed_frames"] / 50.)
                self.assertEqual(patient["clip_end_max_wait_seconds"], .06)
                self.assertGreater(patient["model_frames_per_second"], 0)
            output = Path(job["output_dir"])
            self.assertEqual({p.name for p in output.iterdir()}, set(report["artifacts"]))
            clips = rows(output / "clips.csv")
            for patient in ("val_0", "val_1"):
                selected = [r for r in clips if r["patient"] == patient]
                self.assertEqual([int(r["fifo_frames"]) for r in selected[:5]], [4, 8, 12, 16, 16])
                self.assertEqual([r["warmup"] for r in selected[:5]], ["True", "True", "True", "False", "False"])
                self.assertTrue(all(int(r["frame_exit_length"]) == 4 and int(r["native_tubelets"]) == 2 for r in selected))
                self.assertEqual(len({r["persistent_tensor_bytes"] for r in selected[3:]}), 1)
                self.assertTrue(all(r["input_state_digest"] == selected[i - 1]["output_state_digest"]
                                    for i, r in enumerate(selected) if i))
            targets = [r for r in rows(output / "checks.csv") if r["check"] == "target_source_frame_identity"]
            self.assertEqual(len(targets), 2)
            self.assertTrue(all(int(r["source_frame"]) == 6 and int(r["target_offset"]) == 2 for r in targets))
            self.assertTrue(any(r["status"] == "not_available" for r in rows(output / "prefix_checks.csv")))
            with patch.object(FinalTemporalMAE, "stream_clip", side_effect=AssertionError("Restart must skip forward")):
                self.assertEqual(run_streaming_audit(job, manifest, "cpu"), report)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_streaming_audit(dict(job, max_frames=32), manifest, "cpu")
            (output / "clips.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                run_streaming_audit(job, manifest, "cpu")

    def test_formal_real512_cap_and_no_fabricated_short_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(517, 516), mode="none", readout="repeat")
            job.update(smoke=False, max_frames=512)
            report = run_streaming_audit(job, manifest, "cpu")
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["formal_512_coverage"])
            self.assertEqual([r["observed_frames"] for r in report["patients"]], [512, 512])
            self.assertTrue(all(r["state_persistence"] == "not_applicable_memory_none" for r in report["patients"]))
            self.assertEqual(len(rows(root / "audit/clips.csv")), 256)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(17, 7))
            job.update(smoke=False, max_frames=512)
            report = run_streaming_audit(job, manifest, "cpu")
            self.assertEqual(report["status"], "limited")
            self.assertFalse(report["formal_512_coverage"])
            self.assertEqual(report["observed_cases"], 1)
            self.assertEqual(report["patients"][0]["observed_frames"], 16)
            self.assertEqual(report["patients"][0]["excluded_tail_frames"], 1)
            self.assertEqual(report["exclusions"][0]["reason"], "fewer_than_two_real_complete_clips")

    def test_late_read_mixed_slots_and_no_extra_frame_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(24,), mode="spatial_global", late=True)
            job.update(max_cases=1, max_frames=24)
            original = FinalTemporalMAE.frame_features
            def local_native_only(model, value, valid=None):
                # Core's local exit may expand native tubelets, audit may not expand
                # any explicit final [B,L,P,D] exit a second time.
                self.assertEqual(value.shape[1], model.token_grid[0])
                return original(model, value, valid)
            with patch.object(FinalTemporalMAE, "frame_features", local_native_only):
                result = run_streaming_audit(job, manifest, "cpu")
            self.assertEqual(result["patients"][0]["state_persistence"], "checked")
            self.assertTrue(all(int(r["state_slots"]) == 5 for r in rows(root / "audit/clips.csv")))

    def test_stream_rejects_discontinuity_and_reset_does_not_touch_weights(self):
        model = FinalTemporalMAE(**tiny_config()).eval()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        stream = _ClipStream(model)
        clip = torch.rand(1, 4, 3, 16, 16)
        with torch.inference_mode():
            stream.update(clip, "a", torch.arange(4)[None])
            before = stream.state.clone()
            for patient, indices in (("b", torch.arange(4, 8)[None]), ("a", torch.arange(4)[None]),
                                     ("a", torch.arange(5, 9)[None]), ("a", torch.arange(4, 8).flip(0)[None])):
                with self.assertRaises(ValueError):
                    stream.update(clip, patient, indices)
                torch.testing.assert_close(stream.state, before)
                self.assertEqual(stream.observed_clips, 1)
            stream.reset("b")
            self.assertIsNone(stream.state)
            self.assertFalse(stream.entries)
            self.assertEqual(stream.tensor_bytes(), 0)
            stream.update(clip, "b", torch.arange(10, 14)[None])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_hash_stat_overlap_and_budget_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(12, 12))
            for options in (dict(max_frames=513), dict(max_frames=4), dict(max_cases=3), dict(atol=.01)):
                with self.assertRaises(ValueError):
                    run_streaming_audit(dict(job, **options), manifest, "cpu")
            with self.assertRaisesRegex(ValueError, "Test records"):
                run_streaming_audit(job, dict(manifest, test=[]), "cpu")
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_streaming_audit(job, dict(manifest, train=[dict(patient="val_0")]), "cpu")
            bad = json.loads(json.dumps(manifest))
            bad["val"][0]["source_mtime_ns"] -= 1
            with self.assertRaisesRegex(ValueError, "changed after manifest"):
                run_streaming_audit(job, bad, "cpu")

    def test_failure_no_done_and_empty_real_cohort_explicitly_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(12,))
            with patch.object(FinalTemporalMAE, "diagnostic_features", side_effect=RuntimeError("diagnostic failure")):
                with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
                    run_streaming_audit(job, manifest, "cpu")
            self.assertFalse((root / "audit/DONE").exists())
            self.assertEqual(json.loads((root / "audit/metrics.json").read_text())["status"], "failed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(7, 3))
            result = run_streaming_audit(job, manifest, "cpu")
            self.assertEqual(result["status"], "limited")
            self.assertEqual(result["observed_cases"], 0)
            self.assertIsNone(result["checks_passed"])
            self.assertEqual(len(result["exclusions"]), 2)
            self.assertFalse(result["patient_reset_checked"])
            self.assertTrue((root / "audit/DONE").is_file())

    def test_reversed_or_doubled_frame_exits_fail_without_done(self):
        original = FinalTemporalMAE.stream_clip
        for bad_kind in ("reversed", "doubled"):
            with self.subTest(bad_kind=bad_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest, job = fixture(root, lengths=(12,))
                def bad_stream(model, *args, **kwargs):
                    result = original(model, *args, **kwargs)
                    value = result["frame_outputs"]
                    result["frame_outputs"] = value.flip(1) if bad_kind == "reversed" else value.repeat_interleave(2, 1)
                    return result
                expected = AssertionError if bad_kind == "reversed" else ValueError
                with patch.object(FinalTemporalMAE, "stream_clip", bad_stream), self.assertRaises(expected):
                    run_streaming_audit(job, manifest, "cpu")
                self.assertFalse((root / "audit/DONE").exists())
                self.assertEqual(json.loads((root / "audit/metrics.json").read_text())["status"], "failed")

    def test_source_provenance_and_checkpoint_changes_no_full_npy_hash(self):
        from utils import final_temporal_streaming as audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, job = fixture(root, lengths=(12,))
            original = audit._file_hash
            def no_npy_hash(path):
                self.assertNotEqual(Path(path).suffix, ".npy")
                return original(path)
            with patch.object(audit, "_file_hash", side_effect=no_npy_hash):
                run_streaming_audit(job, manifest, "cpu")
            checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
            checkpoint["epoch"] += 1
            torch.save(checkpoint, job["checkpoint"])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_streaming_audit(job, manifest, "cpu")
            np.save(manifest["val"][0]["path"], np.zeros((12, 16, 16), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "changed after manifest"):
                run_streaming_audit(job, manifest, "cpu")


if __name__ == "__main__":
    unittest.main()
