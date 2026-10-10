"""Measured B=1 workloads, conservative budgets and discarded model lifetimes."""

import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
import pandas as pd
import torch

from models.final_temporal_mae import FinalTemporalMAE
from utils.final_temporal_data import build_manifest
from utils.seed import get_rng_state
import utils.final_temporal_preflight as preflight


class FinalTemporalPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'npy').mkdir()
        rows, traces = [], []
        for index, (name, split) in enumerate((('a', 'TRAIN'), ('b', 'TRAIN'), ('v', 'VAL'))):
            np.save(self.root / 'npy' / (name + '.npy'),
                    np.random.default_rng(index).integers(0, 256, (20, 112, 112), dtype=np.uint8))
            rows.append(dict(FileName=name, Split=split, EF=40 + index * 15, FPS=50, NumberOfFrames=20))
            for x1, y1, x2, y2 in ((0, 0, 112, 112), (30, 30, 80, 30), (25, 60, 85, 60), (40, 80, 70, 80)):
                traces.append(dict(FileName=name + '.avi', Frame=7, X1=x1, Y1=y1, X2=x2, Y2=y2))
        pd.DataFrame(rows).to_csv(self.root / 'FileList.csv', index=False)
        pd.DataFrame(traces).to_csv(self.root / 'VolumeTracings.csv', index=False)
        self.manifest = build_manifest(self.root, recent_frames=8, local_frames=4, max_prefix=8)
        (self.root / 'npy' / 'v.npy').unlink()
        cfg = dict(img_size=16, patch_size=4, local_frames=4, clip_count=2, in_chans=3,
                   tubelet_size=2, embed_dim=24, depth=1, num_heads=3, decoder_embed_dim=24,
                   decoder_depth=1, decoder_num_heads=3, mask_ratio=.5, memory_mode='spatial',
                   memory_grid=2, core_depth=1, memory_write_source='local', frame_readout='learned',
                   norm_pix_loss=False, reconstruction_recent_frames=8, gradient_checkpointing=True,
                   position_embedding='flat_sinusoid')
        self.source = self.root / 'source.pt'
        torch.manual_seed(22)
        model = FinalTemporalMAE(**cfg)
        torch.save(dict(model_state_dict=model.state_dict(), config=dict(model=cfg)), self.source)
        self.job = dict(checkpoint=str(self.source), output_dir=str(self.root / 'output'),
                        checkpoint_dir=str(self.root / 'weights'), cache_dir=str(self.root / 'cache'),
                        smoke=True, seed=42, recent_frames=8, max_prefix=8, overrides=dict(gradient_checkpointing=True),
                        head_dim=12, head_depth=1, head_heads=3, micro_batch=7, num_workers=0,
                        precision='fp32', min_free_gb=0, autotune=True, save_every_updates=1)

    def test_actual_b1_all_task_modes_no_source_update_rng_or_live_models(self):
        original = preflight.load_final_model
        references, batches = [], []

        def load(*args, **kwargs):
            result = original(*args, **kwargs)
            references.append(weakref.ref(result[0]))
            return result

        features = preflight._features

        def observe(backbone, samples, *args, **kwargs):
            batches.append((len(samples), 'video' in samples[0]))
            return features(backbone, samples, *args, **kwargs)

        digest = preflight.file_digest(self.source)
        rng = get_rng_state()
        with mock.patch.object(preflight, 'load_final_model', side_effect=load), \
                mock.patch.object(preflight, '_features', side_effect=observe):
            for frozen in (True, False):
                for task in ('ef', 'seg'):
                    row = preflight._task_timing(self.job, self.manifest, 'cpu', task, frozen)
                    self.assertEqual(row['measurement_batch_size'], 1)
                    self.assertEqual(row['measurement_effective_batch'], 1)
                    self.assertEqual(row['measured_successful_updates'], 2)
                    self.assertEqual(row['warmup_successful_updates'], 2)
                    self.assertFalse(row['validation_scores_consulted'])
                    self.assertTrue(row['gradient_checkpointing'])
                    for field in ('single_sample_update_seconds', 'encoder_extraction_seconds_per_sample',
                                  'evaluation_seconds_per_sample', 'data_wait_seconds_per_sample'):
                        self.assertGreater(row[field], 0)
                    if frozen:
                        self.assertTrue(all(name.startswith('head.') for name in row['active_parameters']))
                    else:
                        self.assertFalse(any(name.startswith('backbone.decoder_') for name in row['active_parameters']))
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertTrue(all(size == 1 for size, _ in batches))
        self.assertTrue(any(not pixels for _, pixels in batches))
        self.assertEqual(preflight.file_digest(self.source), digest)
        self.assertTrue(torch.equal(get_rng_state()['torch'], rng['torch']))

    def test_checkpointing_choice_is_fixed_not_retuned(self):
        row = preflight._task_timing(dict(self.job, overrides=dict(gradient_checkpointing=False)),
                                    self.manifest, 'cpu', 'ef', False)
        self.assertFalse(row['gradient_checkpointing'])

    def test_exception_releases_trial_models_and_restores_rng(self):
        original = preflight.load_final_model
        references = []

        def load(*args, **kwargs):
            result = original(*args, **kwargs)
            references.append(weakref.ref(result[0]))
            return result

        rng = get_rng_state()
        with mock.patch.object(preflight, 'load_final_model', side_effect=load), \
                mock.patch.object(preflight, '_loss', side_effect=RuntimeError('timing failure')):
            with self.assertRaisesRegex(RuntimeError, 'timing failure'):
                preflight._task_timing(self.job, self.manifest, 'cpu', 'seg', False)
        gc.collect()
        self.assertIsNone(references[0]())
        self.assertTrue(torch.equal(get_rng_state()['torch'], rng['torch']))

    def test_preflight_smoke_forces_b1_disposes_weights_and_reuses_guarded_result(self):
        original = preflight.run_warm_job
        jobs = []

        def warm(job, *args):
            jobs.append(dict(job))
            return original(job, *args)

        digest = preflight.file_digest(self.source)
        with mock.patch.object(preflight, 'run_warm_job', side_effect=warm):
            metrics = preflight.run_preflight_job(self.job, self.manifest, 'cpu')
        self.assertEqual(jobs[0]['micro_batch'], 1)
        self.assertFalse(jobs[0]['autotune'])
        self.assertTrue(jobs[0]['overrides']['gradient_checkpointing'])
        self.assertEqual(len(metrics['task_timings']), 4)
        self.assertFalse(metrics['validation_scores_consulted'])
        self.assertEqual(metrics['task_records']['seg']['val'], 4)
        for name in ('last.pt', 'final.pt'):
            self.assertFalse((self.root / 'weights/warm' / name).exists())
        self.assertFalse((self.root / 'output/warm/DONE').exists())
        self.assertEqual(preflight.file_digest(self.source), digest)
        with mock.patch.object(preflight, '_task_timing', side_effect=AssertionError('should reuse')):
            self.assertEqual(preflight.run_preflight_job(self.job, self.manifest, 'cpu'), metrics)
        budget = preflight.estimate_budget(metrics, updates=2, frozen_epochs=1, ft_epochs=1)
        self.assertGreater(budget['mandatory_hours'], 0)
        self.assertGreaterEqual(budget['worst_case_hours'], budget['mandatory_hours'])
        self.assertFalse(budget['validation_scores_consulted'])

    def test_failed_preflight_disposes_measurement_checkpoint_owners(self):
        def warm(job, *args):
            weights, output = Path(job['checkpoint_dir']), Path(job['output_dir'])
            weights.mkdir(parents=True)
            output.mkdir(parents=True)
            (weights / 'last.pt').write_bytes(b'discarded')
            (weights / 'final.pt').write_bytes(b'discarded')
            (output / 'DONE').write_text('{}', encoding='utf-8')
            return dict(optimizer_update_seconds_median=.5)

        with mock.patch.object(preflight, 'run_warm_job', side_effect=warm), \
                mock.patch.object(preflight, '_task_timing', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                preflight.run_preflight_job(self.job, self.manifest, 'cpu')
        self.assertFalse((self.root / 'weights/warm/last.pt').exists())
        self.assertFalse((self.root / 'weights/warm/final.pt').exists())
        self.assertFalse((self.root / 'output/DONE').exists())

    @staticmethod
    def budget_fixture():
        return dict(adaptation=dict(optimizer_update_seconds_median=.5),
                    task_records={task: dict(train=2, val=4) for task in ('ef', 'seg')},
                    task_timings=[dict(task=task, freeze=freeze, single_sample_update_seconds=2.,
                                       encoder_extraction_seconds_per_sample=3., evaluation_seconds_per_sample=5.,
                                       data_wait_seconds_per_sample=1.) for freeze in (True, False) for task in ('ef', 'seg')])

    def test_budget_full_evaluation_no_cache_discount_no_score_consultation(self):
        measured = self.budget_fixture()
        measured['clinical_accuracy_should_be_ignored'] = float('nan')
        result = preflight.estimate_budget(measured, updates=10, frozen_epochs=1, ft_epochs=2,
                                           optional_ft=False, second_seed=False)
        self.assertEqual(result['components_seconds']['frozen_tasks'], 1080.)
        self.assertEqual(result['components_seconds']['calibration_finetuning'], 504.)
        self.assertEqual(result['components_seconds']['mandatory_adaptation'], 35.)
        self.assertAlmostEqual(result['mandatory_hours'], 1619 * 1.15 / 3600)
        self.assertFalse(result['assumes_cache_hits'])
        self.assertEqual(result['updates'], 10)
        larger = preflight.estimate_budget(measured, updates=10, frozen_epochs=1, ft_epochs=2)
        self.assertEqual(result['mandatory_hours'], larger['mandatory_hours'])
        self.assertGreater(larger['worst_case_hours'], result['worst_case_hours'])
        older = self.budget_fixture()
        for record in older['task_timings']:
            record.pop('evaluation_seconds_per_sample')
        self.assertEqual(preflight.estimate_budget(older, 10, 1, 2)['components_seconds']['frozen_tasks'], 1080.)

    def test_budget_rejects_missing_zero_nonfinite_measurements(self):
        for value in (None, 0, float('nan'), float('inf'), -1):
            fixture = self.budget_fixture()
            fixture['adaptation']['optimizer_update_seconds_median'] = value
            with self.assertRaises(ValueError):
                preflight.estimate_budget(fixture)
        fixture = self.budget_fixture()
        fixture['task_timings'].pop()
        with self.assertRaises(ValueError):
            preflight.estimate_budget(fixture)


if __name__ == '__main__':
    unittest.main()
