"""File-backed read-only Q1.4 diagnostics, including honest gradient coverage."""

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from models.final_temporal_mae import FinalTemporalMAE
from utils.final_temporal_data import build_manifest
import utils.final_temporal_mechanisms as audit
from utils.seed import get_rng_state, seed_everything


def _model_config(**changes):
    cfg = dict(img_size=16, patch_size=4, local_frames=4, clip_count=2, in_chans=3,
               tubelet_size=2, embed_dim=24, depth=1, num_heads=3, decoder_embed_dim=24,
               decoder_depth=1, decoder_num_heads=3, mask_ratio=.5, memory_mode='spatial',
               memory_grid=2, core_depth=1, memory_write_source='local', memory_compression='mean',
               frame_readout='factorized', dynamic_rank=4, dynamic_orthogonal_weight=.01,
               memory_read_location='frames', norm_pix_loss=False, reconstruction_recent_frames=8,
               gradient_checkpointing=False, position_embedding='flat_sinusoid')
    cfg.update(changes)
    if cfg['frame_readout'] not in ('factorized', 'soft_factorized'):
        cfg['dynamic_orthogonal_weight'] = 0
    return cfg


def _fixture(root):
    (root / 'npy').mkdir()
    rows = []
    for index, (name, split, frames) in enumerate((('short', 'TRAIN', 8), ('middle', 'TRAIN', 16),
                                                  ('long', 'TRAIN', 20), ('validation', 'VAL', 12))):
        rng = np.random.default_rng(index + 30)
        np.save(root / 'npy' / (name + '.npy'), rng.integers(0, 256, (frames, 112, 112), dtype=np.uint8))
        rows.append(dict(FileName=name, Split=split, EF=35 + index * 10, FPS=50, NumberOfFrames=frames))
    pd.DataFrame(rows).to_csv(root / 'FileList.csv', index=False)
    pd.DataFrame(columns=['FileName', 'Frame', 'X1', 'Y1', 'X2', 'Y2']).to_csv(root / 'VolumeTracings.csv', index=False)
    manifest = build_manifest(root, recent_frames=8, local_frames=4, max_prefix=12)
    # A TRAIN-only audit must not touch even this now-unavailable VAL source.
    (root / 'npy' / 'validation.npy').unlink()
    return manifest


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class FinalTemporalMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = _fixture(self.root)
        self.config = dict(smoke=True, seed=42, recent_frames=8, max_prefix=12, quiet=True,
                           num_windows=2, saturation_epsilon=.01, prediction_atol=1e-6, prediction_rtol=1e-5)

    def model(self, **changes):
        torch.manual_seed(5)
        model = FinalTemporalMAE(**_model_config(**changes))
        if model.frame_readout in ('factorized', 'soft_factorized'):
            # A tiny, realistic deviation gives a small loss but measurable regularizer gradient.
            with torch.no_grad():
                model.frame_expansion.basis.mul_(1.0001)
        return model

    def run_audit(self, model, name='audit', config=None, manifest=None):
        output = self.root / name
        metrics = audit.run_mechanism_audit(model, manifest or self.manifest, output,
                                           config or self.config, 'cpu')
        return metrics, output

    def assert_artifacts(self, metrics, output):
        self.assertEqual(_read(output / 'metrics.json'), metrics)
        for name in (*audit._ARTIFACTS, 'DONE'):
            self.assertTrue((output / name).is_file(), name)
        done = _read(output / 'DONE')
        for name in audit._ARTIFACTS:
            self.assertEqual(done['artifacts'][name], audit.file_digest(output / name))
        with (output / 'state_traces.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), metrics['state_trace_rows'])
        self.assertEqual({row['mode'] for row in rows}, {'masked', 'unmasked'})
        self.assertTrue(all(row['patient'] != 'validation' for row in rows))
        return rows

    def test_content_gradients_mask_isolation_and_actual_weighted_orth(self):
        model = self.model(memory_mode='spatial_global', candidate_rank=4)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with mock.patch('torch.optim.AdamW', side_effect=AssertionError('No optimizer allowed')), \
                mock.patch('torch.optim.SGD', side_effect=AssertionError('No optimizer allowed')):
            metrics, output = self.run_audit(model)
        rows = self.assert_artifacts(metrics, output)
        self.assertEqual(metrics['windows'], 2)
        self.assertEqual(metrics['memory_slots'], 5)
        self.assertTrue(metrics['source_weights_unchanged'])
        self.assertTrue(metrics['state_traces_finite'])
        self.assertTrue(metrics['content_gradients_finite'])
        self.assertEqual(metrics['technical_findings'], [])
        self.assertEqual(metrics['requested_gradient_prefix_frames'], 12)
        self.assertEqual(metrics['actual_gradient_prefix_frames'], 12)
        for name, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        gradients = _read(output / 'gradients.json')
        self.assertEqual(len(gradients['prefix_clips']), 3)
        self.assertEqual([row['source_start'] for row in gradients['prefix_clips']], [0, 4, 8])
        self.assertTrue(all(row['finite'] and row['l2'] > 0 for row in gradients['prefix_clips']))
        for group in ('memory', 'memory_update_gates', 'memory_candidate', 'frame', 'feature_fusion'):
            self.assertGreater(gradients['groups'][group]['l2'], 0, group)
        for name in ('memory.candidate_down.weight', 'memory.candidate_up.weight', 'frame_expansion.basis'):
            self.assertGreater(gradients['parameters'][name]['l2'], 0)
        orth = gradients['orthogonal']
        self.assertLess(orth['loss'], 1e-6)
        self.assertGreater(orth['raw_gradient']['l2'], 0)
        self.assertGreater(orth['weighted_gradient']['l2'], 0)
        self.assertAlmostEqual(orth['weighted_gradient']['l2'] / orth['raw_gradient']['l2'], .01, places=6)
        self.assertGreater(orth['weighted_to_reconstruction_ratio'], 0)
        self.assertIsNotNone(orth['weighted_vs_reconstruction_cosine'])
        isolation = metrics['hidden_pixel_isolation']
        self.assertTrue(isolation['only_fully_hidden_tubelet_patches_changed'])
        self.assertTrue(isolation['prediction_invariant'])
        self.assertGreater(isolation['changed_pixel_elements'], 0)
        self.assertGreater(isolation['target_max_abs_difference'], 0)
        self.assertEqual(isolation['visible_pixel_max_difference'], 0)
        self.assertTrue(metrics['preceding_clip_future_causality']['first_clip_invariant'])
        self.assertEqual({int(row['state_slots']) for row in rows}, {5})
        self.assertTrue(all(0 <= float(row['update_gate_low_saturation']) <= 1 for row in rows))

    def test_restore_mixed_modes_requires_grad_existing_gradients_and_rng(self):
        model = self.model()
        model.eval()
        model.blocks[0].train()
        model.reconstruction_recent_frames = 4
        for index, parameter in enumerate(model.parameters()):
            parameter.requires_grad_(index % 3 != 0)
            if index % 5 == 0:
                parameter.grad = torch.full_like(parameter, .123)
        modes = [module.training for module in model.modules()]
        flags = [parameter.requires_grad for parameter in model.parameters()]
        grads = [parameter.grad for parameter in model.parameters()]
        values = [None if value is None else value.clone() for value in grads]
        seed_everything(717)
        rng = get_rng_state()
        with torch.no_grad():
            metrics, _ = self.run_audit(model)
        self.assertTrue(metrics['content_gradient_verified'])
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertEqual([parameter.requires_grad for parameter in model.parameters()], flags)
        self.assertFalse(model.gradient_checkpointing)
        self.assertEqual(model.reconstruction_recent_frames, 4)
        for parameter, original, expected in zip(model.parameters(), grads, values):
            self.assertIs(parameter.grad, original)
            if expected is not None:
                self.assertTrue(torch.equal(parameter.grad, expected))
        after = get_rng_state()
        self.assertEqual(after['python'], rng['python'])
        np.testing.assert_array_equal(after['numpy'][1], rng['numpy'][1])
        self.assertEqual(after['numpy'][2:], rng['numpy'][2:])
        self.assertTrue(torch.equal(after['torch'], rng['torch']))
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_global_spatial_mixed_and_no_memory_shapes(self):
        for mode, slots in (('global', 1), ('spatial', 4), ('spatial_global', 5), ('none', 0)):
            with self.subTest(mode=mode):
                model = self.model(memory_mode=mode, frame_readout='learned', memory_read_location='tokens')
                metrics, output = self.run_audit(model, mode)
                rows = self.assert_artifacts(metrics, output)
                self.assertEqual(metrics['memory_slots'], slots)
                self.assertEqual({int(row['state_slots']) for row in rows}, {slots})
                self.assertTrue(metrics['hidden_pixel_isolation']['prediction_invariant'])
                gradients = _read(output / 'gradients.json')
                self.assertFalse(gradients['orthogonal']['applicable'])
                if mode == 'global':
                    self.assertTrue(all(float(row['state_slot_variance']) == 0 for row in rows))
                if mode == 'none':
                    self.assertEqual(gradients['groups']['memory']['parameters'], 0)
                    self.assertTrue(all(row['l2'] == 0 for row in gradients['prefix_clips']))
                    self.assertNotIn('no_measured_prefix_content_gradient', metrics['technical_findings'])

    def test_h0_short_source_is_not_a_failure(self):
        manifest = copy.deepcopy(self.manifest)
        manifest.pop('manifest_sha256')
        manifest['train'] = [case for case in manifest['train'] if case['patient'] == 'short']
        metrics, output = self.run_audit(self.model(memory_mode='global'), 'h0', manifest=manifest)
        gradients = _read(output / 'gradients.json')
        self.assertEqual(metrics['actual_gradient_prefix_frames'], 0)
        self.assertEqual(gradients['prefix_clips'], [])
        self.assertFalse(gradients['prefix_applicable'])
        self.assertIn('not a failure', gradients['no_prefix_reason'])
        self.assertEqual(metrics['technical_findings'], [])

    def test_mask_expansion_changes_only_entire_hidden_tubelet_patches(self):
        model = self.model()
        masks = audit._fixed_masks(model, 3, 42, 'tube')
        hidden = audit._hidden_pixels(masks, model)
        self.assertEqual(tuple(hidden.shape), (1, 12, 1, 16, 16))
        expected = masks.reshape(1, 6, 4, 4)
        for temporal in range(6):
            for y in range(4):
                for x in range(4):
                    patch = hidden[:, temporal * 2:(temporal + 1) * 2, :, y * 4:(y + 1) * 4, x * 4:(x + 1) * 4]
                    self.assertTrue(torch.all(patch == expected[:, temporal, y, x]))
        self.assertTrue(torch.equal(masks, audit._fixed_masks(model, 3, 42, 'tube')))

    def test_stream_causality_calls_never_receive_future_frames(self):
        model = self.model()
        records, representative = audit._plan(self.manifest, model, self.config)
        sample = audit._dataset(self.manifest, [representative], model, self.config)[0]
        with mock.patch.object(model, 'stream_clip', wraps=model.stream_clip) as calls:
            result = audit._future_causality(model, sample, self.config, torch.device('cpu'))
        self.assertTrue(result['first_clip_invariant'])
        self.assertGreater(result['future_changed_elements'], 0)
        self.assertEqual(len(calls.call_args_list), 10)
        for call in calls.call_args_list:
            self.assertEqual(call.args[0].shape[1], 4)
        self.assertIn('not frame causality', result['boundary'])

    def test_actual_gate_hooks_match_recurrence_and_bottleneck_candidate(self):
        model = self.model(memory_mode='global', candidate_rank=4)
        records, representative = audit._plan(self.manifest, model, self.config)
        sample = audit._dataset(self.manifest, [representative], model, self.config)[0]
        captured = []

        def memory_hook(module, args, result):
            tokens, old = args
            old = torch.zeros_like(tokens) if old is None else old
            update = (module.update_x(tokens) + module.update_s(old)).sigmoid()
            reset = (module.reset_x(tokens) + module.reset_s(old)).sigmoid()
            candidate = tokens
            for block in module.integration:
                candidate = block(candidate, reset * old)
            candidate = module.candidate_up(module.candidate_down(module.norm(candidate)))
            captured.append(dict(update_median=float(update.median()),
                                 candidate_l2=float(candidate.double().norm()),
                                 delta_l2=float((result[1] - old).double().norm())))

        model.eval()
        handle = model.memory.register_forward_hook(memory_hook)
        try:
            with torch.no_grad(), audit._StateRecorder(model, sample, 0, 'unmasked', self.config) as recorder:
                model.diagnostic_features(audit._video(sample, model, self.config, torch.device('cpu')))
        finally:
            handle.remove()
        self.assertEqual(len(recorder.rows), 5)
        for row, expected in zip(recorder.rows, captured):
            self.assertAlmostEqual(row['candidate_l2'], expected['candidate_l2'], places=6)
            self.assertAlmostEqual(row['update_l2'], expected['delta_l2'], places=6)
            # torch.median uses a lower order statistic; the audit deliberately uses interpolated quantiles.
            self.assertLess(abs(row['update_gate_median'] - expected['update_median']), .1)

    def test_gradient_oom_reduces_only_real_prefix_and_records_unverified_length(self):
        model = self.model()
        original = audit._gradient_once
        attempted = []

        def bounded(*args, **kwargs):
            h = args[4]
            attempted.append(h)
            if h > 4:
                raise torch.cuda.OutOfMemoryError('simulated gradient-only capacity')
            return original(*args, **kwargs)

        with mock.patch.object(audit, '_gradient_once', side_effect=bounded):
            metrics, output = self.run_audit(model, 'reduced')
        gradients = _read(output / 'gradients.json')
        self.assertEqual(attempted, [12, 4])
        self.assertEqual(gradients['status'], 'verified_reduced_prefix')
        self.assertEqual(gradients['actual_prefix_frames'], 4)
        self.assertEqual(gradients['unverified_prefix_frames'], 8)
        self.assertFalse(gradients['prefix_coverage_complete'])
        self.assertEqual(gradients['prefix_clips'][0]['source_start'], 8)
        self.assertGreater(gradients['prefix_clips'][0]['l2'], 0)
        # Non-gradient diagnostics still use the original maximal real window.
        self.assertEqual(metrics['hidden_pixel_isolation']['prefix_frames'], 12)
        self.assertEqual(metrics['hidden_pixel_isolation']['total_frames'], 20)

    def test_total_gradient_oom_does_not_fabricate_zero_or_verified_gradients(self):
        with mock.patch.object(audit, '_gradient_once', side_effect=torch.cuda.OutOfMemoryError('no fit')):
            metrics, output = self.run_audit(self.model(), 'oom')
        gradients = _read(output / 'gradients.json')
        self.assertFalse(metrics['content_gradient_verified'])
        self.assertIsNone(metrics['content_gradients_finite'])
        self.assertIsNone(gradients['actual_prefix_frames'])
        self.assertNotIn('parameters', gradients)
        self.assertEqual([attempt['prefix_frames'] for attempt in gradients['attempts']], [12, 4, 0])

    def test_reuse_hash_guard_and_corrupted_done_failure(self):
        model = self.model()
        metrics, output = self.run_audit(model)
        with mock.patch.object(audit, '_gradient_probe', side_effect=AssertionError('Should reuse')):
            self.assertEqual(self.run_audit(model)[0], metrics)
        done = _read(output / 'DONE')
        done['artifacts']['state_traces.csv'] = 'wrong'
        (output / 'DONE').write_text(json.dumps(done), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'missing or altered outputs'):
            self.run_audit(model)
        self.assertTrue((output / 'DONE').exists())

    def test_model_data_and_protocol_changes_are_rejected(self):
        for mutation in ('model', 'pixels', 'config'):
            with self.subTest(mutation=mutation):
                model = self.model()
                metrics, output = self.run_audit(model, mutation)
                config = dict(self.config)
                if mutation == 'model':
                    with torch.no_grad():
                        model.norm.bias.add_(.01)
                elif mutation == 'pixels':
                    path = self.root / 'npy' / 'long.npy'
                    stat = path.stat()
                    array = np.load(path, mmap_mode='r+')
                    array[0, 0, 0] ^= 1
                    array.flush()
                    array._mmap.close()
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                else:
                    config['seed'] = 43
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    self.run_audit(model, mutation, config=config)

    def test_exception_restores_flags_rng_and_hooks_without_done(self):
        model = self.model()
        model.eval()
        model.requires_grad_(False)
        rng = get_rng_state()
        modes = [module.training for module in model.modules()]
        with mock.patch.object(audit, '_hidden_isolation', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                self.run_audit(model, 'failure')
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.gradient_checkpointing)
        self.assertTrue(torch.equal(get_rng_state()['torch'], rng['torch']))
        self.assertFalse((self.root / 'failure' / 'DONE').exists())
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_zero_orth_weight_reports_zero_actual_weighted_gradient_not_missing_graph(self):
        metrics, output = self.run_audit(self.model(dynamic_orthogonal_weight=0), 'zero_orth')
        orth = _read(output / 'gradients.json')['orthogonal']
        self.assertGreater(orth['raw_gradient']['l2'], 0)
        self.assertEqual(orth['weighted_gradient']['l2'], 0)
        self.assertGreater(orth['weighted_gradient']['connected_parameters'], 0)
        self.assertIsNone(orth['weighted_vs_reconstruction_cosine'])

    def test_config_fixed_budget_native_formal_and_train_only_plan(self):
        formal = SimpleNamespace(img_size=112, local_frames=16)
        cfg = audit._config({}, formal)
        cases = [dict(patient=str(index), frames=64 + index * 16) for index in range(20)]
        records, representative = audit._plan(dict(train=cases, val=[]), formal, cfg)
        self.assertEqual(len(records), 32)
        self.assertEqual(min(record['H'] for record in records), 0)
        self.assertEqual(max(record['H'] for record in records), 128)
        self.assertEqual(representative['H'], 128)
        self.assertEqual(records, audit._plan(dict(train=cases, val=[]), formal, cfg)[0])
        with self.assertRaisesRegex(ValueError, 'Formal Q1.4'):
            audit._config({}, self.model())
        for config in (dict(self.config, num_windows=3), dict(self.config, max_prefix=132),
                       dict(self.config, recent_frames=7)):
            with self.assertRaises(ValueError):
                audit._config(config, self.model())
        with self.assertRaisesRegex(ValueError, 'cannot silently resize'):
            audit.native_video(torch.zeros(1, 8, 3, 112, 112), self.model(), False)

    def test_non_fp32_caller_uses_clone_without_changing_source_dtype(self):
        model = self.model().double()
        before = audit._model_hash(model)
        metrics, output = self.run_audit(model, 'double')
        self.assertEqual(audit._model_hash(model), before)
        self.assertTrue(all(parameter.dtype == torch.float64 for parameter in model.parameters()))
        self.assertTrue(metrics['source_weights_unchanged'])
        self.assertEqual(metrics['precision'], 'fp32')

    def test_nonfinite_audit_is_serializable_and_does_not_claim_mask_leakage(self):
        model = self.model()
        with torch.no_grad():
            model.norm.weight.fill_(float('nan'))
        metrics, output = self.run_audit(model, 'nonfinite')
        self.assert_artifacts(metrics, output)
        self.assertFalse(metrics['state_traces_finite'])
        self.assertFalse(metrics['content_gradients_finite'])
        self.assertIsNone(_read(output / 'gradients.json')['content_loss'])
        self.assertIn('hidden_pixel_invariance_unverified_nonfinite', metrics['technical_findings'])
        self.assertNotIn('hidden_pixel_prediction_dependency', metrics['technical_findings'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_checkpointed_real_gradient_probe(self):
        model = self.model().cuda().eval()
        output = self.root / 'cuda'
        metrics = audit.run_mechanism_audit(model, self.manifest, output, self.config, 'cuda')
        self.assertTrue(metrics['content_gradient_verified'])
        self.assertTrue(metrics['content_gradients_finite'])
        self.assertFalse(model.gradient_checkpointing)


if __name__ == '__main__':
    unittest.main()
