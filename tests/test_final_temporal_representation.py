"""Small real-slice tests of the frozen final R audit, without external DATA."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.temporal_mae import TemporalMAE
from utils.final_temporal_representation import (
    _Budget, _Capture, _Reservoir, _anatomy_scores, _attention_received,
    _bootstrap, _config, _identity_margin, _memory_prediction, _read_slice,
    _records, _stream, centered_cka, geometry, gini, run_representation_audit,
    run_memory_prediction_audit,
)


def tiny_config(depth=1, readout="factorized"):
    return dict(img_size=16, patch_size=4, local_frames=4, clip_count=2,
                tubelet_size=2, in_chans=3, embed_dim=24, depth=depth,
                num_heads=3, decoder_embed_dim=24, decoder_depth=1,
                decoder_num_heads=3, mask_ratio=.5, memory_mode="spatial",
                memory_grid=2, core_depth=1, norm_pix_loss=False,
                frame_readout=readout, dynamic_rank=6, separate_qv_bias=True)


class ContractModel(TemporalMAE):
    """Local stand-in for the parent-owned FinalTemporalMAE frame exit contract."""
    memory_slots = 4

    def stream_clip(self, video, state=None, short_state=None, frame_valid=None):
        result = super().stream_clip(video, state, short_state, frame_valid)
        expansion = self.frame_expansion
        base = expansion.base if hasattr(expansion, "base") else expansion
        result["frame_base_outputs"] = base(result["features"])
        result["local_base_outputs"] = base(result["local_features"])
        result["frame_outputs"] = self.frame_features(result["features"])
        result["local_frame_outputs"] = self.frame_features(result["local_features"])
        return result


def audit_config(**overrides):
    result = dict(seed=42, model_id="C100", max_train=4, max_val=3, max_seg_train=4,
                  max_seg_val=3, attention_patients=2, recent_frames=8,
                  layers=[1], attention_layers=[1], token_reservoir=256,
                  anatomy_train_tokens=512, anatomy_val_tokens=1024,
                  anatomy_patches_per_view=16, tokens_per_window=32,
                  bootstrap_repetitions=40, token_budget_bytes=8 * 1024**2)
    result.update(overrides)
    return result


def make_manifest(root, train=4, val=3, frames=24):
    rng = np.random.default_rng(17)
    manifest = {"train": [], "val": []}
    # Official paired-border points, deliberately yielding all three patch classes.
    points = [[0, 0, 0, 0], [0, 0, 56, 0], [0, 28, 56, 28], [0, 56, 56, 56]]
    for split, count in (("train", train), ("val", val)):
        for i in range(count):
            patient = f"{split}_{i}"
            path = root / f"{patient}.npy"
            video = rng.integers(0, 256, (frames, 16, 16), dtype=np.uint8)
            np.save(path, video)
            manifest[split].append(dict(patient=patient, path=str(path), frames=frames,
                fps=50., ef=40. + i * 4., start=4,
                traces=[dict(frame=11, points=points)]))
    return manifest


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_centered_cka_rank_and_degenerate_inputs(self):
        x = torch.tensor([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
        self.assertAlmostEqual(centered_cka(x, x * 3 + 19), 1., places=10)
        self.assertIsNone(centered_cka(torch.ones_like(x), x))
        with self.assertRaisesRegex(ValueError, "paired"):
            centered_cka(x, x[:2])
        stats, spectrum = geometry(x)
        self.assertAlmostEqual(stats["effective_rank"], 1., places=6)
        self.assertAlmostEqual(sum(spectrum), 1.)
        self.assertEqual(geometry(torch.ones_like(x))[0]["effective_rank"], 0.)
        self.assertEqual(gini([1, 1, 1]), 0.)
        self.assertGreater(gini([0, 0, 3]), .6)

    def test_fp32_identity_margins_and_ties(self):
        target = torch.stack((torch.zeros(3, 64), torch.ones(3, 64)))
        correct = _identity_margin(target, target, 1e-6)
        reverse = _identity_margin(target.flip(0), target, 1e-6)
        tie = _identity_margin(torch.ones_like(target) * .5, target, 1e-6)
        self.assertTrue(correct["correct"])
        self.assertFalse(reverse["correct"])
        self.assertLess(reverse["margin"], 0)
        self.assertTrue(tie["tie"])
        self.assertFalse(tie["correct"])

    def test_patient_bootstrap_not_token_weighted(self):
        rows = [dict(patient="a", value=0.)] * 100 + [dict(patient="b", value=1.)]
        result = _bootstrap(rows, "value", audit_config())
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean"], .5)

    def test_reservoir_alignment_and_hard_budget(self):
        budget = _Budget(1000)
        reservoir = _Reservoir(5, budget, 42)
        for start in (0, 10, 20):
            x = torch.arange(start, start + 10).float()[:, None]
            reservoir.add(dict(x=x, y=x + 100), [(str(int(i)),) for i in x.flatten()])
        self.assertEqual(len(reservoir.keys), 5)
        self.assertEqual(reservoir.seen, 30)
        torch.testing.assert_close(reservoir.data["y"], reservoir.data["x"] + 100)
        self.assertEqual(budget.peak, 40)
        for key, value in zip(reservoir.keys, reservoir.data["x"]):
            self.assertEqual(int(key[0]), int(value))
        with self.assertRaises(MemoryError):
            _Reservoir(100, _Budget(1), 42).add(dict(x=torch.ones(2, 10)), [("a",), ("b",)])

    def test_attention_independent_matches_explicit_softmax_and_keeps_sdpa(self):
        model = ContractModel(**tiny_config())
        attention = model.blocks[0].attn
        x = torch.randn(1, 32, 24)
        bias = torch.cat((attention.q_bias, torch.zeros_like(attention.q_bias), attention.v_bias))
        qkv = torch.nn.functional.linear(x, attention.qkv.weight, bias)
        q, k, _ = qkv.reshape(1, 32, 3, 3, 8).permute(2, 0, 3, 1, 4).unbind(0)
        probabilities = ((q @ k.transpose(-1, -2)) * attention.scale).softmax(-1)
        received, entropy = _attention_received(attention, x)
        torch.testing.assert_close(received, probabilities.mean((0, 1, 2)))
        self.assertGreater(entropy, 0.)
        with patch("models.vit_blocks.F.scaled_dot_product_attention", wraps=torch.nn.functional.scaled_dot_product_attention) as sdpa:
            attention(x)
        self.assertEqual(sdpa.call_count, 1)

    def test_stream_exits_and_true_pre_cache_boundary(self):
        model = ContractModel(**tiny_config()).eval()
        video = torch.rand(1, 12, 3, 16, 16)
        with torch.no_grad():
            expected = model.stream_clip(video[:, :4])["final_state"].cpu()
            with _Capture(model, [1], [1]) as capture:
                capture.with_attention = True
                exits, descriptors, boundary = _stream(model, video, 8, capture)
                self.assertIn(1, capture.attention)
                self.assertEqual(exits["F"].shape, (1, 4, 16, 24))
                self.assertEqual(descriptors["local"].shape, (8, 24))
                torch.testing.assert_close(boundary, expected)
                torch.testing.assert_close(exits["H"].mean(1), exits["F"].mean(1), atol=1e-6, rtol=1e-5)
        self.assertEqual(len(model.norm._forward_hooks), 0)

    def test_input_indices_disjointness_and_budget_validation(self):
        model = ContractModel(**tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_manifest(Path(directory))
            r = manifest["train"][0]
            video = _read_slice(r, 2, 4, model)
            self.assertEqual(video.shape, (1, 4, 3, 16, 16))
            torch.testing.assert_close(video[:, :, 0], video[:, :, 2])
            with self.assertRaisesRegex(ValueError, "padding"):
                _read_slice(r, 23, 4, model)
            with self.assertRaisesRegex(ValueError, "overlap"):
                bad = copy.deepcopy(manifest)
                bad["val"][0]["patient"] = bad["train"][0]["patient"]
                _records(bad)
            with self.assertRaisesRegex(ValueError, "Test-set"):
                _records(dict(manifest, test=[]))
            for overrides in (dict(max_train=513), dict(attention_patients=33),
                              dict(token_budget_bytes=2 * 1024**3 + 1), dict(model_id="B7")):
                with self.assertRaises(ValueError):
                    _config(audit_config(**overrides), model, True, False)

    def test_end_to_end_outputs_restart_and_mode_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root)
            model = ContractModel(**tiny_config()).train()
            model.blocks[0].attn.eval()
            flags = [module.training for module in model.modules()]
            weights = {name: value.clone() for name, value in model.state_dict().items()}
            output = root / "report"
            report = run_representation_audit(model, manifest, output, audit_config(), "cpu", detailed=True)
            json.dumps(report, allow_nan=False)
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["geometry_only_success"])
            self.assertIsNotNone(report["anatomy_linear_score"])
            self.assertIsNotNone(report["content_identity_accuracy"])
            self.assertEqual(report["coverage"], dict(train=4, val=3))
            self.assertLess(report["token_cache_peak_bytes"], 8 * 1024**2)
            self.assertEqual(flags, [module.training for module in model.modules()])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, weights[name])
            for name in report["artifacts"]:
                self.assertTrue((output / name).is_file(), name)
            from PIL import Image
            for name in ("spectra.png", "evidence.png"):
                with Image.open(output / name) as image:
                    self.assertGreater(float(np.asarray(image.convert("RGB")).std()), 10.)
            observations = json.loads((output / "patient_observations.json").read_text())
            self.assertEqual(len(observations["patients"]), 3)
            import csv
            with (output / "anatomy.csv").open() as handle:
                anatomy = list(csv.DictReader(handle))
            self.assertEqual({int(row["offset"]) for row in anatomy}, {0, 1, 2, 3})
            self.assertTrue(all(int(row["source_frame"]) == 11 for row in anatomy))
            with (output / "attention.csv").open() as handle:
                attention = list(csv.DictReader(handle))
            self.assertEqual(len(attention), 2)
            with patch.object(model, "stream_clip", side_effect=AssertionError("cache must be reused")):
                self.assertEqual(run_representation_audit(model, manifest, output, audit_config(), "cpu", detailed=True), report)
            model.soft_beta = .5
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_representation_audit(model, manifest, output, audit_config(), "cpu", detailed=True)
            del model.soft_beta
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_representation_audit(model, manifest, output, audit_config(seed=7), "cpu", detailed=True)
            raw = np.load(manifest["train"][0]["path"])
            raw[0, 0, 0] ^= 1
            np.save(manifest["train"][0]["path"], raw)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_representation_audit(model, manifest, output, audit_config(), "cpu", detailed=True)

    def test_corrupt_done_and_failure_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root, train=2, val=1)
            model = ContractModel(**tiny_config()).train()
            report = root / "report"
            cfg = audit_config(max_train=2, max_val=1)
            run_representation_audit(model, manifest, report, cfg, "cpu")
            (report / "geometry.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                run_representation_audit(model, manifest, report, cfg, "cpu")
            with patch.object(model, "stream_clip", side_effect=RuntimeError("failed extraction")):
                with self.assertRaisesRegex(RuntimeError, "failed extraction"):
                    run_representation_audit(model, manifest, root / "failed", cfg, "cpu")
            self.assertTrue(model.training)
            self.assertFalse((root / "failed" / "DONE").exists())
            self.assertFalse(model.norm._forward_hooks)

    def test_real_parent_model_and_stat_only_hash_policy(self):
        from models.final_temporal_mae import FinalTemporalMAE
        from utils import final_temporal_representation as audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = make_manifest(root, train=2, val=1)
            for split in records.values():
                for record in split:
                    stat = Path(record["path"]).stat()
                    record.update(source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                  shape=[24, 16, 16], dtype="uint8")
            model = FinalTemporalMAE(**dict(tiny_config(), memory_read_location="frames", memory_write_source="local"))
            hash_file = audit._file_hash
            def forbid_source_content_hash(path):
                self.assertNotEqual(Path(path).suffix, ".npy")
                return hash_file(path)
            with patch.object(audit, "_file_hash", side_effect=forbid_source_content_hash):
                report = run_representation_audit(model, records, root / "real", audit_config(max_train=2, max_val=1), "cpu")
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["state_statistics"])
            self.assertTrue(any(row["exit"] == "native_fused" for row in report["geometry"]))
            provenance = copy.deepcopy(records)
            provenance["train"][0]["source_mtime_ns"] -= 1
            with self.assertRaisesRegex(ValueError, "changed after manifest"):
                run_representation_audit(model, provenance, root / "bad", audit_config(), "cpu")

    def test_nonfinite_pixels_and_source_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = make_manifest(root, train=2, val=1)
            model = ContractModel(**tiny_config())
            path = Path(records["train"][0]["path"])
            raw = np.load(path).astype(np.float32)
            raw[0, 0, 0] = float("nan")
            np.save(path, raw)
            with self.assertRaisesRegex(ValueError, "finite"):
                _read_slice(records["train"][0], 0, 4, model)
            np.save(path, np.zeros((24, 16, 16), dtype=np.uint8))
            records["train"][0]["frames"] = 23
            with self.assertRaisesRegex(ValueError, "frame count mismatch"):
                run_representation_audit(model, records, root / "bad", audit_config(), "cpu")

    def test_r5_fixed_target_no_future_input_equal_heads_and_no_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = make_manifest(root, train=17, val=2, frames=20)
            # Labels cannot be read by this branch: EF and anatomy need not exist.
            for split in records.values():
                for record in split:
                    record.pop("ef")
                    record.pop("traces")
            model = ContractModel(**tiny_config()).eval()
            reference = ContractModel(**tiny_config(depth=9, readout="learned")).eval()
            cfg = _config(audit_config(model_id="B0", max_train=17, max_val=2,
                reference_id="C100", memory_prefix=4), model, False, True)
            intervals, target_lengths, widths = [], [], []
            original_read = _read_slice
            original_fit = __import__("utils.final_temporal_representation", fromlist=["ridge_fit"]).ridge_fit
            def read(record, start, count, backbone):
                intervals.append((backbone is model, start, count))
                return original_read(record, start, count, backbone)
            def fit(x, y, alpha):
                widths.append((tuple(x.shape), tuple(y.shape), alpha))
                return original_fit(x, y, alpha)
            handle = reference.blocks[8].register_forward_hook(lambda m, args, out: target_lengths.append(out.shape[1]))
            try:
                with torch.no_grad(), patch("utils.final_temporal_representation._read_slice", side_effect=read), patch(
                        "utils.final_temporal_representation.ridge_fit", side_effect=fit):
                    rows, report = _memory_prediction(model, reference, records, cfg, torch.device("cpu"), _Budget(8 * 1024**2))
            finally:
                handle.remove()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["delta"]["n"], 2)
            self.assertEqual(widths[0], widths[1])
            self.assertEqual(widths[0][1][-1], 16)
            self.assertTrue(all(start == 0 and count == 12 for is_model, start, count in intervals if is_model))
            self.assertTrue(all(start == 12 and count == 4 for is_model, start, count in intervals if not is_model))
            self.assertTrue(all(n == 32 for n in target_lengths))
            self.assertTrue(all(row["state_last_source"] == 3 and row["target_start"] == 12 for row in rows))
            self.assertTrue(all(row["target_end"] == 15 for row in rows))
            self.assertIsInstance(report["true_state_r2"], float)

    def test_r5_insufficient_real_future_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            records = make_manifest(Path(directory), train=2, val=1, frames=12)
            model = ContractModel(**tiny_config()).eval()
            reference = ContractModel(**tiny_config(depth=9)).eval()
            cfg = _config(audit_config(model_id="B1", reference_id="C100", memory_prefix=4), model, False, True)
            with torch.no_grad():
                rows, report = _memory_prediction(model, reference, records, cfg, torch.device("cpu"), _Budget(1024**2))
            self.assertFalse(rows)
            self.assertEqual(report["status"], "insufficient_real_next_clip_patients")
            self.assertEqual(len(report["exclusions"]), 3)

    def test_r5_only_public_helper_outputs_restart_and_no_other_audits(self):
        from utils import final_temporal_representation as audit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root, train=17, val=2, frames=20)
            for cases in manifest.values():
                for record in cases:
                    record.pop("ef")
                    record.pop("traces")
            model = ContractModel(**tiny_config()).train()
            reference = ContractModel(**tiny_config(depth=9, readout="learned")).train()
            reference.blocks[0].attn.eval()
            flags = [[module.training for module in item.modules()] for item in (model, reference)]
            weights = [{k: v.clone() for k, v in item.state_dict().items()} for item in (model, reference)]
            cfg = audit_config(model_id="B0", max_train=17, max_val=2, reference_id="C100", memory_prefix=4)
            output = root / "r5"
            with patch.object(audit, "run_representation_audit", side_effect=AssertionError("No full R")), patch.object(
                    audit, "geometry", side_effect=AssertionError("No geometry")), patch.object(
                    audit, "_plots", side_effect=AssertionError("No plots")), patch.object(
                    audit, "_trace_labels", side_effect=AssertionError("No anatomy")):
                result = run_memory_prediction_audit(model, reference, manifest, output, cfg, "cpu")
            json.dumps(result, allow_nan=False)
            self.assertEqual(result["scope"], "R5_only")
            self.assertEqual(result["status"], "complete")
            self.assertIsInstance(result["memory_prediction_delta"], float)
            self.assertEqual(len(result["patient_observations"]), 2)
            self.assertTrue(result["role_evidence"]["patient_matched"])
            self.assertFalse(result["role_evidence"]["clinical_evidence"])
            self.assertEqual(result["memory_prediction"]["delta"]["n"], 2)
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"memory_prediction.csv", "metrics.json", "protocol.json", "DONE"})
            for item, expected_flags, expected_weights in zip((model, reference), flags, weights):
                self.assertEqual([m.training for m in item.modules()], expected_flags)
                self.assertTrue(all(p.grad is None for p in item.parameters()))
                for key, value in item.state_dict().items():
                    torch.testing.assert_close(value, expected_weights[key])
            self.assertFalse(reference.blocks[8]._forward_hooks)
            with patch.object(audit, "_memory_prediction", side_effect=AssertionError("Cache must skip extraction")):
                self.assertEqual(run_memory_prediction_audit(model, reference, manifest, output, cfg, "cpu"), result)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_memory_prediction_audit(model, reference, manifest, output, dict(cfg, memory_prefix=8), "cpu")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_representation_audit(model, manifest, output, cfg, "cpu", reference_model=reference)
            (output / "memory_prediction.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                run_memory_prediction_audit(model, reference, manifest, output, cfg, "cpu")

    def test_r5_only_insufficient_status_failures_and_source_stat_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root, train=2, val=1, frames=12)
            model = ContractModel(**tiny_config()).train()
            reference = ContractModel(**tiny_config(depth=9)).train()
            cfg = audit_config(model_id="B1", reference_id="C100", memory_prefix=4)
            output = root / "insufficient"
            result = run_memory_prediction_audit(model, reference, manifest, output, cfg, "cpu")
            self.assertEqual(result["status"], "insufficient_real_next_clip_patients")
            self.assertIsNone(result["memory_prediction_delta"])
            self.assertFalse(result["patient_observations"])
            self.assertFalse(result["role_evidence"]["available"])
            self.assertEqual(len(result["exclusions"]), 3)
            self.assertTrue((output / "DONE").is_file())
            with patch("utils.final_temporal_representation._memory_prediction", side_effect=RuntimeError("R5 failure")):
                with self.assertRaisesRegex(RuntimeError, "R5 failure"):
                    run_memory_prediction_audit(model, reference, manifest, root / "failed", cfg, "cpu")
            self.assertTrue(model.training)
            self.assertTrue(reference.training)
            self.assertFalse((root / "failed" / "DONE").exists())
            np.save(manifest["train"][0]["path"], np.ones((12, 16, 16), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_memory_prediction_audit(model, reference, manifest, output, cfg, "cpu")
            for override in (dict(memory_prefix=0), dict(reference_id="F100"), dict(memory_pca_dim=8), dict(model_id="C100")):
                with self.assertRaises(ValueError):
                    run_memory_prediction_audit(model, reference, manifest, root / "invalid", dict(cfg, **override), "cpu")


if __name__ == "__main__":
    unittest.main()
