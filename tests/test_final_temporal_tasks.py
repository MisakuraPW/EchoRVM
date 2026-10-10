"""File-backed CPU task fits and exact resume; optional CUDA precision checks."""

import csv
import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from models.final_temporal_mae import FinalTemporalMAE
from utils.final_temporal_data import build_manifest, WindowDataset
import utils.final_temporal_tasks as tasks
from utils.seed import get_rng_state


def _fixture(root):
    (root / 'npy').mkdir()
    rows, traces = [], []
    for i, (patient, split, ef) in enumerate((('train_a', 'TRAIN', 35.0), ('train_b', 'TRAIN', 65.0),
                                            ('train_c', 'TRAIN', 50.0), ('val_a', 'VAL', 90.0),
                                            ('val_b', 'VAL', 20.0))):
        rng = np.random.default_rng(i)
        video = rng.integers(0, 256, (18, 112, 112), dtype=np.uint8)
        np.save(root / 'npy' / (patient + '.npy'), video)
        rows.append(dict(FileName=patient, Split=split, EF=ef, FPS=50, NumberOfFrames=18))
        for frame in (7, 12):
            for x1, y1, x2, y2 in ((0, 0, 112, 112), (35, 30, 75, 30), (25, 55, 85, 55), (42, 85, 68, 85)):
                traces.append(dict(FileName=patient + '.avi', Frame=frame, X1=x1, Y1=y1, X2=x2, Y2=y2))
    pd.DataFrame(rows).to_csv(root / 'FileList.csv', index=False)
    pd.DataFrame(traces).to_csv(root / 'VolumeTracings.csv', index=False)
    return build_manifest(root, recent_frames=8, local_frames=4, max_prefix=8)


def _config():
    return dict(name='temporal_final', img_size=16, patch_size=8, local_frames=4, clip_count=2,
                tubelet_size=2, in_chans=3, embed_dim=24, depth=1, num_heads=3,
                decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=3,
                memory_mode='spatial', memory_grid=2, memory_write_source='local', core_depth=1,
                frame_readout='learned', norm_pix_loss=False, mask_ratio=.5,
                position_embedding='flat_sinusoid', gradient_checkpointing=True, drop_path_rate=.15)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


class FinalTemporalTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = _fixture(self.root)
        torch.manual_seed(19)
        model = FinalTemporalMAE(**_config())
        self.source = self.root / 'source.pt'
        torch.save(dict(model_state_dict=model.state_dict(), config=dict(model=_config()), epoch=7), self.source)

    def job(self, name='run', task='ef', freeze=True, **changes):
        result = dict(checkpoint=str(self.source), overrides=dict(gradient_checkpointing=True),
                      output_dir=str(self.root / name / 'result'), checkpoint_dir=str(self.root / name / 'weights'),
                      cache_dir=str(self.root / 'cache'), task=task, freeze=freeze, epochs=2, patience=4,
                      seed=42, num_workers=0, micro_batch=2, effective_batch=3,
                      autotune=True, smoke=True, head_dim=12, head_depth=1, head_heads=3,
                      recent_frames=8, max_prefix=8, precision='fp32', quiet=True)
        result.update(changes)
        return result

    def assert_delivered(self, job, metrics):
        output, weights = Path(job['output_dir']), Path(job['checkpoint_dir'])
        for name in ('metrics.json', 'patient_predictions.csv', 'normalization.json', 'protocol.json',
                     'logs/metrics.csv', 'logs/metrics.jsonl', 'logs/train.log', 'plots/loss_latest.png', 'DONE'):
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual({p.name for p in weights.iterdir()}, {'last.pt', 'best.pt'})
        self.assertEqual(json.loads((output / 'metrics.json').read_text()), metrics)
        self.assertEqual(metrics['best_checkpoint'], str((weights / 'best.pt').resolve()))
        self.assertEqual(metrics['normalization_path'], str((output / 'normalization.json').resolve()))
        best, last = _load(weights / 'best.pt'), _load(weights / 'last.pt')
        self.assertNotIn('optimizer_state_dict', best)
        self.assertTrue(last['optimizer_state_dict']['state'])
        self.assertEqual(best['epoch'], metrics['best_epoch'])
        self.assertEqual(last['progress']['best_predictions'], list(self._predictions(output, typed=True)))
        done = json.loads((output / 'DONE').read_text())
        for name, digest in done['artifacts'].items():
            self.assertEqual(tasks._file_digest(output / name), digest)
        from matplotlib.image import imread
        image = imread(output / 'plots/loss_latest.png')
        self.assertGreater(float(image[..., :3].std()), .05)

    @staticmethod
    def _predictions(output, typed=False):
        with (output / 'patient_predictions.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if typed:
            for row in rows:
                for key in ('source_frames', 'positions', 'windows', 'full_context_windows', 'full_context_sources'):
                    row[key] = int(row[key])
                for key in ('prediction', 'target', 'absolute_error_pp', 'dice'):
                    if key in row:
                        row[key] = float(row[key])
        return rows

    def test_frozen_ef_all_cases_normalization_cache_and_outputs(self):
        job = self.job()
        metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        self.assert_delivered(job, metrics)
        self.assertEqual(metrics['mae'], metrics['mae_pp'])
        self.assertEqual(metrics['patients'], 2)
        output = Path(job['output_dir'])
        stats = json.loads((output / 'normalization.json').read_text())
        self.assertEqual(stats['target_mean'], 50)
        self.assertAlmostEqual(stats['target_std'], float(np.std([35, 65, 50])))
        self.assertEqual(set(stats['fit_patients']), {'train_a', 'train_b', 'train_c'})
        self.assertEqual(stats['feature_count'], 3 * 8)
        protocol = json.loads((output / 'protocol.json').read_text())
        self.assertEqual(protocol['train_samples'], 3)
        self.assertEqual(protocol['optimizer']['lr'], 1e-3)
        self.assertTrue(protocol['model']['gradient_checkpointing'])
        self.assertTrue(all(name.startswith('head.') for name in protocol['active_parameters']))
        self.assertFalse(any('backbone.' in name for name in _load(Path(job['checkpoint_dir']) / 'last.pt')['model_state_dict']))
        shards = list((Path(job['cache_dir']) / protocol['cache_identity']).glob('*.pt'))
        self.assertEqual(len(shards), 5)
        for shard in shards:
            sample = _load(shard)['sample']
            self.assertNotIn('video', sample)
            self.assertNotIn('frame_indices', sample)
            self.assertEqual(tuple(sample['features'].shape), (8, 24))
        with mock.patch.object(tasks, 'encode_window', side_effect=AssertionError('Cache not reused')):
            second = tasks.run_task_job(self.job('second'), self.manifest, 'cpu')
        self.assertEqual(second['mae'], metrics['mae'])

    def test_frozen_and_ft_ef_share_head_normalization_and_scale(self):
        frozen = self.job('frozen', epochs=1)
        ft = self.job('ft', freeze=False, epochs=1)
        frozen_metrics = tasks.run_task_job(frozen, self.manifest, 'cpu')
        ft_metrics = tasks.run_task_job(ft, self.manifest, 'cpu')
        frozen_stats = json.loads((Path(frozen['output_dir']) / 'normalization.json').read_text())
        ft_stats = json.loads((Path(ft['output_dir']) / 'normalization.json').read_text())
        self.assertEqual(frozen_stats, ft_stats)
        saved = _load(Path(ft['checkpoint_dir']) / 'last.pt')
        names = saved['active_parameters']
        for module in ('patch_embed.', 'blocks.', 'norm.', 'frame_expansion.', 'memory.', 'feature_fusion.'):
            self.assertTrue(any(name.startswith('backbone.' + module) for name in names), module)
        self.assertFalse(any(name.startswith('backbone.' + prefix) for name in names for prefix in tasks._MAE_ONLY))
        initial = _load(self.source)['model_state_dict']
        for module in ('patch_embed.', 'blocks.', 'frame_expansion.', 'memory.', 'feature_fusion.'):
            self.assertTrue(any(not torch.equal(tensor, initial[name.removeprefix('backbone.')])
                                for name, tensor in saved['model_state_dict'].items()
                                if name.startswith('backbone.' + module)), module)
        self.assert_delivered(ft, ft_metrics)
        self.assertEqual(frozen_metrics['mae'], frozen_metrics['mae_pp'])
        self.assertLess(ft_metrics['mae'], 100)
        protocol = json.loads((Path(ft['output_dir']) / 'protocol.json').read_text())
        self.assertEqual(protocol['optimizer']['lr'], 5e-5)
        self.assertEqual(protocol['augmentation']['preset'], 'A4_tgc_zoom_speckle')

    def test_seg_frozen_and_ft_repeated_ids_original_sources_all_positions(self):
        for freeze in (True, False):
            with self.subTest(freeze=freeze):
                job = self.job('seg_' + str(freeze), task='seg', freeze=freeze, epochs=1)
                metrics = tasks.run_task_job(job, self.manifest, 'cpu')
                self.assert_delivered(job, metrics)
                self.assertEqual(metrics['dice_patient_mean'], metrics['patient_dice'])
                self.assertEqual(metrics['patients'], 2)
                self.assertEqual(metrics['source_frames'], 4)
                self.assertEqual(metrics['windows'], 16)
                self.assertEqual(metrics['full_context_windows'], 16)
                self.assertGreaterEqual(metrics['patient_dice'], 0)
                self.assertLessEqual(metrics['patient_dice'], 1)
                for row in self._predictions(Path(job['output_dir']), typed=True):
                    self.assertEqual(row['source_frames'], 2)
                    self.assertEqual(row['positions'], 8)
                    self.assertEqual(row['full_context_sources'], 2)
                protocol = json.loads((Path(job['output_dir']) / 'protocol.json').read_text())
                self.assertEqual(protocol['train_samples'], 6)
                self.assertEqual(protocol['head']['name'], 'ViTPatchSegDecoder')
                self.assertEqual(protocol['validation_cache'], 'none_streaming')
                if freeze:
                    shards = list((Path(job['cache_dir']) / protocol['cache_identity']).glob('*.pt'))
                    self.assertEqual(len(shards), 6)
                    self.assertTrue(all(_load(path)['sample']['patient'].startswith('train_') for path in shards))

    def test_approved_defaults_are_by_adaptation_mode_not_task(self):
        formal = mock.Mock(img_size=112, local_frames=16)
        for task in ('ef', 'seg'):
            for freeze in (True, False):
                job = tasks._resolve_job(dict(task=task, freeze=freeze), formal)
                self.assertEqual(job['epochs'], 60 if freeze else 80)
                self.assertEqual(job['patience'], 12 if freeze else 20)
                self.assertEqual(job['warmup_epochs'], 3 if freeze else 5)
                self.assertEqual(job['effective_batch'], 12 if task == 'ef' else 64)
                self.assertEqual(job['head_dim'], 64 if task == 'ef' else 192)
                self.assertEqual(job['head_depth'], 4)
                self.assertEqual(job['head_heads'], 3)
                self.assertEqual(job['min_free_gb'], 5)
                self.assertEqual(job['dataset_seed'], 42)

    def test_head_seed_43_retains_exact_common_validation_records_and_provenance(self):
        for task in ('ef', 'seg'):
            results = []
            for seed in (42, 43):
                job = self.job(f'common_population_{task}_{seed}', task=task, seed=seed, epochs=1)
                metrics = tasks.run_task_job(job, self.manifest, 'cpu')
                population = json.loads((Path(job['output_dir']) / 'evaluation_population.json').read_text())
                protocol = json.loads((Path(job['output_dir']) / 'protocol.json').read_text())
                self.assertEqual(protocol['dataset_seed'], 42)
                self.assertEqual(population['dataset_seed'], 42)
                self.assertEqual(metrics['evaluation_population_sha256'], tasks._digest(population))
                self.assertEqual(metrics['evaluation_population_provenance']['source_provenance'], population['source_provenance'])
                self.assertTrue(all(record['id'] and record['full_context'] for record in population['records']))
                results.append((metrics, population))
            self.assertEqual(results[0][1], results[1][1])
            self.assertEqual(results[0][0]['evaluation_population_sha256'], results[1][0]['evaluation_population_sha256'])
        source = WindowDataset(self.manifest, 'val', task='ef', recent_frames=8, local_frames=4, max_prefix=8, seed=42)
        a = tasks._evaluation_population(source)
        source.seed = 43
        b = tasks._evaluation_population(source)
        self.assertEqual(len(a['records']), len(b['records']))
        self.assertNotEqual(tasks._digest(a), tasks._digest(b))

    def test_patient_aggregation_weights_source_before_patient(self):
        rows = [dict(patient='a', source_frame=1, position=i, full_context=True, dice=0.0) for i in range(3)]
        rows += [dict(patient='a', source_frame=2, position=0, full_context=True, dice=1.0),
                 dict(patient='b', source_frame=1, position=0, full_context=True, dice=1.0)]
        metrics, patients = tasks._aggregate(rows, 'seg')
        self.assertEqual(patients[0]['dice'], .5)
        self.assertEqual(metrics['patient_dice'], .75)
        with self.assertRaisesRegex(ValueError, 'positions'):
            tasks._aggregate(rows, 'seg', range(4))
        with self.assertRaisesRegex(ValueError, 'complete-context'):
            tasks._aggregate([dict(rows[0], full_context=False)], 'seg')
        ef = [dict(patient='a', source_frame=-1, position=-1, full_context=True, prediction=10., target=40.),
              dict(patient='a', source_frame=-1, position=-1, full_context=True, prediction=50., target=40.),
              dict(patient='b', source_frame=-1, position=-1, full_context=True, prediction=50., target=50.)]
        self.assertEqual(tasks._aggregate(ef, 'ef')[0]['mae_pp'], 5.)

    def test_microbatch_effective_update_ragged_and_resume_permutation(self):
        ds = WindowDataset(self.manifest, 'train', task='seg', recent_frames=8, local_frames=4,
                           max_prefix=8, prefix=None, training=True)
        ds.set_epoch(2)
        reference = None
        for micro in (1, 2, 3):
            batches = list(tasks._UpdateBatches(ds, micro, 5, 42, 2, training=True))
            draws = {}
            for batch in batches:
                self.assertLessEqual(len(batch), micro)
                self.assertEqual(len({ds.get_record(index)['prefix_frames'] for index, _, _, _ in batch}), 1)
                for index, cursor, size, last in batch:
                    draws.setdefault(cursor, []).append(index)
                    self.assertEqual(size, 5 if cursor == 0 else 1)
            draws = {key: sorted(value) for key, value in draws.items()}
            if reference is None:
                reference = draws
            self.assertEqual(draws, reference)
            resumed = list(tasks._UpdateBatches(ds, micro, 5, 42, 2, start=1, training=True))
            self.assertEqual(resumed, [b for b in batches if b[0][1] >= 1])

    def test_exact_epoch_resume_no_optimizer_reset_and_best_prediction_recovery(self):
        uninterrupted = self.job('reference', freeze=False, effective_batch=2, micro_batch=1)
        tasks.run_task_job(uninterrupted, self.manifest, 'cpu')
        interrupted = self.job('interrupted', freeze=False, effective_batch=2, micro_batch=1)
        real_publish = tasks._publish_history

        def stop_after_first_epoch(output, history):
            real_publish(output, history)
            if len(history) == 1:
                raise KeyboardInterrupt('test exact epoch stop')

        with mock.patch.object(tasks, '_publish_history', side_effect=stop_after_first_epoch):
            with self.assertRaises(KeyboardInterrupt):
                tasks.run_task_job(interrupted, self.manifest, 'cpu')
        before = _load(Path(interrupted['checkpoint_dir']) / 'last.pt')
        self.assertEqual(before['progress']['epoch'], 1)
        self.assertEqual(before['progress']['global_step'], 2)
        (Path(interrupted['output_dir']) / 'patient_predictions.csv').unlink()
        actual = tasks.run_task_job(interrupted, self.manifest, 'cpu')
        self.assert_delivered(interrupted, actual)
        a = _load(Path(uninterrupted['checkpoint_dir']) / 'last.pt')
        b = _load(Path(interrupted['checkpoint_dir']) / 'last.pt')
        self.assertEqual(a['progress']['best_predictions'], b['progress']['best_predictions'])
        self.assertEqual(a['progress']['best_metric'], b['progress']['best_metric'])
        self.assertEqual(a['scheduler_state_dict'], b['scheduler_state_dict'])
        self.assertEqual(a['scaler_state_dict'], b['scaler_state_dict'])
        for key in a['model_state_dict']:
            self.assertTrue(torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key)
        for index, state in a['optimizer_state_dict']['state'].items():
            for key in state:
                self.assertTrue(torch.equal(state[key], b['optimizer_state_dict']['state'][index][key]), key)
        self.assertTrue(torch.equal(a['rng_state']['torch'], b['rng_state']['torch']))

    def test_partial_gradient_interrupt_replays_only_unsafe_update(self):
        reference = self.job('partial_reference', task='seg', freeze=False, effective_batch=2, micro_batch=1, epochs=1)
        tasks.run_task_job(reference, self.manifest, 'cpu')
        interrupted = self.job('partial_interrupted', task='seg', freeze=False, effective_batch=2, micro_batch=1, epochs=1)
        original = tasks._loss
        calls = 0

        def stop_on_second_accumulation(prediction, samples, task, normalization):
            nonlocal calls
            if torch.is_grad_enabled():
                calls += 1
                if calls == 4:
                    raise KeyboardInterrupt('discard partial effective batch')
            return original(prediction, samples, task, normalization)

        with mock.patch.object(tasks, '_loss', side_effect=stop_on_second_accumulation):
            with self.assertRaises(KeyboardInterrupt):
                tasks.run_task_job(interrupted, self.manifest, 'cpu')
        checkpoint = _load(Path(interrupted['checkpoint_dir']) / 'last.pt')
        self.assertEqual(checkpoint['progress']['epoch'], 0)
        self.assertEqual(checkpoint['progress']['next_update'], 1)
        self.assertEqual(checkpoint['progress']['global_step'], 1)
        tasks.run_task_job(interrupted, self.manifest, 'cpu')
        a = _load(Path(reference['checkpoint_dir']) / 'last.pt')
        b = _load(Path(interrupted['checkpoint_dir']) / 'last.pt')
        for key in a['model_state_dict']:
            self.assertTrue(torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key)
        self.assertEqual(a['progress']['best_predictions'], b['progress']['best_predictions'])
        self.assertTrue(torch.equal(a['rng_state']['torch'], b['rng_state']['torch']))

    def test_interruption_inside_optimizer_keeps_previous_durable_checkpoint(self):
        reference = self.job('step_reference', freeze=False, epochs=1)
        tasks.run_task_job(reference, self.manifest, 'cpu')
        interrupted = self.job('step_interrupted', freeze=False, epochs=1)
        original = torch.optim.AdamW.step

        def broken_step(optimizer, *args, **kwargs):
            original(optimizer, *args, **kwargs)
            raise KeyboardInterrupt('optimizer may be partially mutated')

        with mock.patch.object(torch.optim.AdamW, 'step', broken_step):
            with self.assertRaises(KeyboardInterrupt):
                tasks.run_task_job(interrupted, self.manifest, 'cpu')
        saved = _load(Path(interrupted['checkpoint_dir']) / 'last.pt')
        self.assertEqual(saved['progress']['global_step'], 0)
        self.assertEqual(saved['optimizer_state_dict']['state'], {})
        tasks.run_task_job(interrupted, self.manifest, 'cpu')
        a = _load(Path(reference['checkpoint_dir']) / 'last.pt')
        b = _load(Path(interrupted['checkpoint_dir']) / 'last.pt')
        for key in a['model_state_dict']:
            self.assertTrue(torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key)

    def test_completed_run_is_idempotent_and_artifact_tampering_rejected(self):
        job = self.job('complete', epochs=1)
        metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        with mock.patch.object(tasks, 'encode_window', side_effect=AssertionError('Completed run recomputed')):
            self.assertEqual(tasks.run_task_job(job, self.manifest, 'cpu'), metrics)
        (Path(job['output_dir']) / 'patient_predictions.csv').write_text('altered', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'altered outputs'):
            tasks.run_task_job(job, self.manifest, 'cpu')

    def test_cached_and_completed_runs_cannot_hide_source_changes(self):
        job = self.job('source_guard', epochs=1)
        tasks.run_task_job(job, self.manifest, 'cpu')
        case = self.manifest['train'][0]
        path = Path(case['path'])
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10000000))
        with self.assertRaisesRegex(ValueError, 'cached features cannot bypass provenance'):
            tasks.run_task_job(job, self.manifest, 'cpu')
        with self.assertRaisesRegex(ValueError, 'cached features cannot bypass provenance'):
            tasks.run_task_job(self.job('source_guard_new', epochs=1), self.manifest, 'cpu')

    def test_early_stopping_keeps_observed_best_and_bad_epochs(self):
        job = self.job('early', epochs=4, patience=1)
        original = tasks._evaluate
        calls = 0

        def worsening_validation(*args, **kwargs):
            nonlocal calls
            metrics, patients = original(*args, **kwargs)
            calls += 1
            for patient in patients:
                patient.update(prediction=patient['target'] + calls, absolute_error_pp=float(calls))
            metrics.update(mae_pp=float(calls), rmse_pp=float(calls))
            return metrics, patients

        with mock.patch.object(tasks, '_evaluate', side_effect=worsening_validation):
            metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        self.assertTrue(metrics['stopped_early'])
        self.assertEqual(metrics['best_epoch'], 1)
        self.assertEqual(metrics['epochs_completed'], 2)
        self.assertEqual(metrics['mae'], 1.)
        last = _load(metrics['last_checkpoint'])
        self.assertEqual(last['progress']['bad_epochs'], 1)
        self.assertEqual(_load(metrics['best_checkpoint'])['epoch'], 1)
        self.assertEqual(tasks.run_task_job(job, self.manifest, 'cpu'), metrics)

    def test_prefix_boundary_gradients_and_same_complete_seg_cohort(self):
        job = self.job('prefix', freeze=False, prefix=None, append_boundary=True, epochs=1, micro_batch=None)
        metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        self.assert_delivered(job, metrics)
        protocol = json.loads((Path(job['output_dir']) / 'protocol.json').read_text())
        self.assertIn('head.history_type', protocol['active_parameters'])
        seg = self.job('prefix_seg', task='seg', freeze=True, prefix=4, epochs=1)
        result = tasks.run_task_job(seg, self.manifest, 'cpu')
        self.assertEqual(result['source_frames'], 2)
        self.assertEqual(result['windows'], 8)

    def test_resume_protocol_cache_identity_and_checkpointing_are_guarded(self):
        job = self.job('guard', epochs=1)
        tasks.run_task_job(job, self.manifest, 'cpu')
        for change in (dict(exit_name='base'), dict(overrides=dict(gradient_checkpointing=False)),
                       dict(effective_batch=2), dict(epochs=2), dict(subset=dict(train=1))):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                tasks.run_task_job(dict(job, **change), self.manifest, 'cpu')
        changed = self.job('different', exit_name='base', epochs=1)
        tasks.run_task_job(changed, self.manifest, 'cpu')
        a = json.loads((Path(job['output_dir']) / 'protocol.json').read_text())
        b = json.loads((Path(changed['output_dir']) / 'protocol.json').read_text())
        self.assertNotEqual(a['cache_identity'], b['cache_identity'])
        payload = _load(self.source)
        payload['model_state_dict']['norm.bias'] += .01
        torch.save(payload, self.source)
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            tasks.run_task_job(job, self.manifest, 'cpu')

    def test_cache_ram_and_global_disk_bounded_no_video_storage(self):
        root = self.root / 'bounded'
        cache = tasks._FeatureCache(root, 'a', disk_bytes=6000, ram_bytes=500)
        for i in range(8):
            cache.put(str(i), dict(features=torch.full((50,), float(i))))
            self.assertLessEqual(cache.ram_used, 500)
            self.assertLessEqual(sum(p.stat().st_size for p in root.glob('*/*.pt')), 6000)
        second = tasks._FeatureCache(root, 'b', disk_bytes=6000, ram_bytes=0)
        second.put('new', dict(features=torch.zeros(50)))
        self.assertLessEqual(sum(p.stat().st_size for p in root.glob('*/*.pt')), 6000)
        self.assertIsNone(second.get('0'))
        with self.assertRaisesRegex(ValueError, 'Videos'):
            cache.put('bad', dict(video=torch.zeros(2, 3, 16, 16)))
        second.root.joinpath('poison.pt').parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(identity='other', key='poison', sample={}), second.root / 'poison.pt')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            second.get('poison')

    def test_tune_clone_does_not_change_model_rng_or_scientific_settings(self):
        job = self.job('tune', freeze=False, micro_batch=None, epochs=1)
        backbone, _, _ = tasks.load_final_model(self.source)
        job = tasks._resolve_job(job, backbone)
        dataset = WindowDataset(self.manifest, 'train', task='ef', recent_frames=8, local_frames=4, max_prefix=8)
        stats = tasks._fit_normalization(backbone, tasks._Inputs(dataset), job, torch.device('cpu'), None)
        model = tasks._TaskModel(backbone, job, stats)
        tasks._active_parameters(model, [dataset[0]], job, torch.device('cpu'))
        state = {k: v.clone() for k, v in model.state_dict().items()}
        rng = get_rng_state()
        micro, report = tasks._autotune(model, [dataset[0]], job, stats, torch.device('cpu'))
        self.assertEqual(micro, 1)
        self.assertTrue(report['synthetic_labels'])
        self.assertTrue(report['gradient_checkpointing'])
        for key in state:
            self.assertTrue(torch.equal(model.state_dict()[key], state[key]), key)
        self.assertTrue(torch.equal(get_rng_state()['torch'], rng['torch']))
        metrics = tasks.run_task_job(self.job('end_to_end_tune', micro_batch=None, epochs=1), self.manifest, 'cpu')
        self.assertEqual(metrics['micro_batch'], 1)

    def test_cached_head_autotune_does_not_call_encoder(self):
        job = self.job('head_tune', micro_batch=None, epochs=1)
        backbone, _, _ = tasks.load_final_model(self.source, job['overrides'])
        job = tasks._resolve_job(job, backbone)
        source = WindowDataset(self.manifest, 'train', task='ef', recent_frames=8, local_frames=4, max_prefix=8)
        inputs = tasks._Inputs(source)
        stats = tasks._fit_normalization(backbone, inputs, job, torch.device('cpu'), None)
        model = tasks._TaskModel(backbone, job, stats)
        tasks._active_parameters(model, [source[0]], job, torch.device('cpu'))
        samples = [inputs[0]]
        tasks._features(backbone, samples, job, torch.device('cpu'))
        samples = [{key: value for key, value in sample.items() if key not in ('video', 'frame_indices')}
                   for sample in samples]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with mock.patch.object(tasks, 'encode_window', side_effect=AssertionError('Cached head must not encode')):
            micro, report = tasks._autotune(model, samples, job, stats, torch.device('cpu'))
        self.assertEqual(micro, 1)
        self.assertEqual(report['workload'], 'cached_head')
        self.assertEqual(report['trials'][0]['successful_updates'], 1)
        for key, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_worker_tune_is_train_io_only_bounded_and_restores_rng(self):
        job = self.job('workers', num_workers=8)
        backbone, _, _ = tasks.load_final_model(self.source)
        job = tasks._resolve_job(job, backbone)
        job['ram_cache_bytes'] = 9000
        source = WindowDataset(self.manifest, 'train', task='ef', recent_frames=8, local_frames=4, max_prefix=8,
                               training=True)
        source.set_epoch(5)
        cache = tasks._FeatureCache(self.root / 'workers_cache', 'test', ram_bytes=9000)
        for index in range(8):
            cache.put(str(index), dict(features=torch.zeros(50)))
        inputs = tasks._Inputs(source, cache)
        visited = []
        rng = get_rng_state()
        before = {key: value.clone() for key, value in backbone.state_dict().items()}

        def fake_loader(inputs, micro, effective, seed, **kwargs):
            workers = kwargs['workers']
            self.assertTrue(kwargs['training'])
            self.assertEqual(kwargs['benchmark_batches'], 24)
            self.assertEqual(cache.ram_bytes, 9000 // (workers + 1))
            self.assertLessEqual(cache.ram_used, cache.ram_bytes)
            self.assertEqual(effective, job['effective_batch'])
            self.assertEqual(micro, 2)

            def batches():
                for _ in range(24):
                    # Deliberately consumes global RNG: tuning must restore it.
                    torch.rand(1)
                    visited.append(workers)
                    yield [object(), object()]
            return batches()

        with mock.patch.object(tasks, '_loader', side_effect=fake_loader), \
                mock.patch.object(tasks.time, 'perf_counter', side_effect=[0., 1., 2., 2.5, 3., 3.25]):
            selected, report = tasks._tune_workers(inputs, job, 2)
        self.assertEqual(selected, 8)
        self.assertEqual(visited, [0] * 24 + [4] * 24 + [8] * 24)
        self.assertEqual(source.epoch.value, 5)
        self.assertFalse(report['accuracy_consulted'])
        self.assertTrue(torch.equal(get_rng_state()['torch'], rng['torch']))
        self.assertEqual(cache.ram_bytes, 1000)
        for row in report['trials']:
            self.assertEqual(row['measured_batches'], 20)
            self.assertLessEqual(row['ram_cache_bytes_per_process'] * row['ram_cache_processes'], 9000)
        for key, value in before.items():
            self.assertTrue(torch.equal(backbone.state_dict()[key], value), key)
        spawned = pickle.loads(pickle.dumps(cache))
        self.assertEqual(spawned.ram_bytes, 1000)
        self.assertEqual(spawned.ram_used, 0)
        self.assertFalse(spawned.ram)

    def test_worker_real_loader_repetition_and_bounded_spawn(self):
        job = self.job('real_workers', num_workers=1)
        backbone, _, _ = tasks.load_final_model(self.source)
        job = tasks._resolve_job(job, backbone)
        source = WindowDataset(self.manifest, 'train', task='ef', recent_frames=8, local_frames=4, max_prefix=8)
        inputs = tasks._Inputs(source)
        loader = tasks._loader(inputs, 2, 3, 42, benchmark_batches=24)
        batches = list(loader)
        self.assertEqual(len(batches), 24)
        self.assertTrue(all(0 < len(batch) <= 2 for batch in batches))
        selected, report = tasks._tune_workers(inputs, job, 2)
        self.assertIn(selected, (0, 1))
        self.assertEqual([row['num_workers'] for row in report['trials']], [0, 1])
        self.assertTrue(all(row['measured_batches'] == 20 for row in report['trials'] if row['status'] == 'ok'))

    def test_worker_options_threads_and_train_only_guard(self):
        source = WindowDataset(self.manifest, 'train', task='ef', recent_frames=8, local_frames=4, max_prefix=8)
        inputs = tasks._Inputs(source)
        parallel = tasks._loader(inputs, 1, 3, 42, workers=1)
        self.assertTrue(parallel.persistent_workers)
        self.assertEqual(parallel.prefetch_factor, 4)
        self.assertIs(parallel.worker_init_fn, tasks.task_worker_init)
        single = tasks._loader(inputs, 1, 3, 42, workers=0)
        self.assertFalse(single.persistent_workers)
        with mock.patch.object(torch, 'set_num_threads') as threads, mock.patch('cv2.setNumThreads') as cv_threads:
            tasks.task_worker_init(0)
        threads.assert_called_once_with(1)
        cv_threads.assert_called_once_with(1)
        val = WindowDataset(self.manifest, 'val', task='ef', recent_frames=8, local_frames=4, max_prefix=8)
        job = tasks._resolve_job(self.job(), tasks.load_final_model(self.source)[0])
        with self.assertRaisesRegex(ValueError, 'only use TRAIN'):
            tasks._tune_workers(tasks._Inputs(val, split='val'), job, 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_cached_head_batch_tuning_uses_real_head_backward_only(self):
        job = self.job('cuda_head', micro_batch=None, precision='fp16')
        backbone, _, _ = tasks.load_final_model(self.source)
        backbone.cuda()
        job = tasks._resolve_job(job, backbone)
        source = WindowDataset(self.manifest, 'train', task='ef', recent_frames=8, local_frames=4, max_prefix=8)
        stats = tasks._fit_normalization(backbone, tasks._Inputs(source), job, torch.device('cuda'), None)
        model = tasks._TaskModel(backbone, job, stats).cuda()
        tasks._active_parameters(model, [source[0]], job, torch.device('cuda'))
        samples = [source[0]]
        tasks._features(backbone, samples, job, torch.device('cuda'))
        samples = [{key: value for key, value in sample.items() if key not in ('video', 'frame_indices')}
                   for sample in samples]
        with mock.patch.object(tasks, 'encode_window', side_effect=AssertionError('Encoder called during cached-head tuning')):
            micro, report = tasks._autotune(model, samples, job, stats, torch.device('cuda'))
        self.assertEqual(report['workload'], 'cached_head')
        self.assertGreaterEqual(micro, 1)
        self.assertTrue(any(row['successful_updates'] == 2 for row in report['trials']))

    def test_local_exit_excludes_disconnected_memory_and_explicit_limits_only(self):
        job = self.job('local', freeze=False, exit_name='local', epochs=1, subset=dict(train=2, val=1))
        metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        protocol = json.loads((Path(job['output_dir']) / 'protocol.json').read_text())
        self.assertEqual(protocol['train_samples'], 2)
        self.assertEqual(metrics['patients'], 1)
        self.assertFalse(any(name.startswith(('backbone.memory.', 'backbone.feature_fusion.'))
                             for name in protocol['active_parameters']))

    def test_best_publication_interruption_recovers_pending_checkpoint(self):
        job = self.job('publication', epochs=1)
        original = tasks._reconcile_best
        calls = 0

        def stop_publication(path, pending, progress, identity):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt('after last before best publish')
            return original(path, pending, progress, identity)

        with mock.patch.object(tasks, '_reconcile_best', side_effect=stop_publication):
            with self.assertRaises(KeyboardInterrupt):
                tasks.run_task_job(job, self.manifest, 'cpu')
        self.assertTrue((Path(job['checkpoint_dir']) / 'best.pt.pending').exists())
        metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        self.assert_delivered(job, metrics)

    def test_frozen_head_loader_reuses_exact_head_normalization_for_history_seg(self):
        job = self.job('load_head', task='seg', epochs=1)
        metrics = tasks.run_task_job(job, self.manifest, 'cpu')
        backbone, _, _ = tasks.load_final_model(self.source, job['overrides'])
        rng = get_rng_state()
        loaded = tasks.load_frozen_task_head(metrics['best_checkpoint'], backbone, metrics)
        self.assertTrue(torch.equal(get_rng_state()['torch'], rng['torch']))
        self.assertFalse(loaded.training)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        saved = _load(metrics['best_checkpoint'])
        for key, value in saved['model_state_dict'].items():
            self.assertTrue(torch.equal(loaded.state_dict()[key], value), key)
        dataset = WindowDataset(self.manifest, 'val', task='seg', recent_frames=8, local_frames=4,
                                max_prefix=8, prefix=4, positions='all')
        sample = dataset[0]
        native = tasks._video(sample, backbone, torch.device('cpu'))[None]
        with torch.no_grad():
            actual = loaded(native, prefix_frames=4, target_index=[sample['target_index']])
            resolved = tasks._resolve_job(job, backbone)
            features, slots = tasks._extract(backbone, [sample], resolved, torch.device('cpu'))
            expected = loaded.read(features, slots)
        torch.testing.assert_close(actual, expected)
        protocol = dict(loaded.task_protocol, exit_name='base')
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            tasks.load_frozen_task_head(metrics['best_checkpoint'], backbone, protocol)

    def test_progress_postfix_tensorboard_warning_and_checkpoint_reserve(self):
        job = self.job('progress', epochs=1, tensorboard=True, min_free_gb=5)
        original = tasks.atomic_torch_save
        saves = []

        def save_without_disk_requirement(value, path, min_free_gb=0):
            saves.append((Path(path).name, min_free_gb))
            original(value, path)

        with mock.patch.object(tasks, 'atomic_torch_save', side_effect=save_without_disk_requirement), \
                mock.patch.object(tasks, 'tqdm') as progress, \
                mock.patch.dict('sys.modules', {'torch.utils.tensorboard': None}):
            with self.assertWarnsRegex(RuntimeWarning, 'TensorBoard requested but unavailable'):
                tasks.run_task_job(job, self.manifest, 'cpu')
        details = progress.return_value.set_postfix.call_args.kwargs
        for key in ('loss', 'MAE_train_pp', 'data', 'step', 'lr', 'GPU', 'stepped'):
            self.assertIn(key, details)
        self.assertIn(('last.pt', 5), saves)
        self.assertIn(('best.pt.pending', 5), saves)
        self.assertIn('TensorBoard requested but unavailable',
                      (Path(job['output_dir']) / 'logs/train.log').read_text())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable locally')
    def test_cuda_amp_real_tune_and_train(self):
        job = self.job('cuda', freeze=False, precision='fp16', micro_batch=None, epochs=1,
                       tune_reserve_bytes=16 * 1024 ** 2)
        metrics = tasks.run_task_job(job, self.manifest, 'cuda')
        self.assertGreater(metrics['successful_updates'], 0)
        protocol = json.loads((Path(job['output_dir']) / 'protocol.json').read_text())
        self.assertTrue(protocol['autotune']['trials'])
        self.assertTrue(protocol['model']['gradient_checkpointing'])


if __name__ == '__main__':
    unittest.main()
