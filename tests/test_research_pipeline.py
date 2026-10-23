"""CPU regression tests for execution contracts; no private dataset needed."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import torch

from research_pipeline.artifacts import digest, export_results, file_hash, layout, run_lock, verify_done, write_json
from research_pipeline.cache import prepare_cache
from research_pipeline.cli import configure, source_path
from research_pipeline.config import DEFAULTS, load, merge, stage_config, validate
from research_pipeline.engine import build_experiment, run_training
from research_pipeline.metrics import linear_cka, paired_bootstrap, representation_summary, segmentation_row, summarize
from research_pipeline.runtime import UpdateBatchSampler, choose, loader


def config(root, kind='mae', mode='finetune', epochs=2):
    base = load('configs/pipeline/synthetic.yaml', tag='test', root=str(root))
    stage = dict(id='unit', kind=kind, mode=mode,
                 training=dict(epochs=epochs, patience=None if kind == 'mae' else 2))
    base['stages'] = [stage]
    validate(base)
    return stage_config(base, stage)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mae_cannot_accidentally_early_stop(self):
        value = merge(DEFAULTS, dict(run=dict(tag='x'), stages=[dict(id='m', kind='mae')]))
        with self.assertRaisesRegex(ValueError, 'patience'):
            validate(value)
        value['stages'][0]['training'] = dict(patience=None)
        validate(value)

    def test_dag_names_and_source_order(self):
        for bad in ('../outside', 'a/b', '.', 'a b'):
            with self.assertRaises(ValueError):
                load('configs/pipeline/synthetic.yaml', tag=bad)
        value = load('configs/pipeline/synthetic.yaml')
        value['stages'][0]['source'] = '@future/final'
        with self.assertRaisesRegex(ValueError, 'earlier'):
            validate(value)

    def test_smoke_has_separate_identity_and_keeps_mae_no_stop(self):
        value = load('configs/pipeline/synthetic.yaml', smoke=True)
        self.assertTrue(value['run']['tag'].endswith('_smoke'))
        self.assertIsNone(stage_config(value, value['stages'][0])['training']['patience'])
        self.assertEqual(value['training']['epochs'], 1)

    def test_cli_overrides_inherited_stage_values_without_overriding_other_stages(self):
        value = load('configs/pipeline/echo_reference.yaml', overrides=[
            'stages.pretrain.source=ckpt/mae/native.pt', 'stages.ef_ft.training.patience=15'])
        stages = {s['id']: stage_config(value, s) for s in value['stages']}
        self.assertEqual(stages['ef_ft']['training']['patience'], 15)
        self.assertEqual(stages['seg_ft']['training']['patience'], 10)
        with self.assertRaises(ValueError):
            load('configs/pipeline/echo_reference.yaml', overrides=['stages.ef_ft.training.paitence=15'])

    def test_sampler_preserves_order_and_final_effective_window(self):
        for count in (1, 7, 9, 13):
            a = list(UpdateBatchSampler(count, 4, 1, 42))
            b = list(UpdateBatchSampler(count, 4, 4, 42))
            self.assertEqual([i for row in a for i in row], [i for row in b for i in row])
            self.assertEqual(len({i for row in a for i in row}), count)
            self.assertEqual(len(a), len(UpdateBatchSampler(count, 4, 1, 42)))
        with self.assertRaises(ValueError):
            UpdateBatchSampler(7, 4, 3, 42)

    def test_runtime_chooses_throughput_not_fullest_vram(self):
        rows = [dict(status='ok', samples_per_second=speed, peak_reserved_bytes=memory,
                     selection=dict(num_workers=workers)) for speed, memory, workers in ((100, 20, 8), (98, 10, 2), (80, 8, 0))]
        self.assertEqual(choose(rows)['peak_reserved_bytes'], 10)
        with self.assertRaises(RuntimeError):
            choose([dict(status='oom')])

    def test_cache_is_bounded_and_matches_uncached_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, 'ef', 'frozen_linear')
            exp = build_experiment(cfg, None, torch.device('cpu'))
            exp.model.eval()
            original = [exp.val[0]]
            with torch.no_grad():
                expected = exp.model(original).clone()
            prepare_cache(exp, Path(tmp)/'cache', dict(version=1), 16*1024**2, 0)
            with torch.no_grad():
                actual = exp.model([exp.val[0]])
            torch.testing.assert_close(expected, actual)
            with self.assertRaisesRegex(ValueError, 'identity'):
                other = build_experiment(cfg, None, torch.device('cpu'))
                prepare_cache(other, Path(tmp)/'cache', dict(version=2), 16*1024**2, 0)
            with self.assertRaises(OSError):
                other = build_experiment(cfg, None, torch.device('cpu'))
                prepare_cache(other, Path(tmp)/'tiny', dict(version=1), 1, 0)
            exp.train.close()
            exp.val.close()

    def test_microbatch_changes_preserve_mean_loss_tail_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            states = []
            for micro in (1, 2, 4):
                cfg = config(Path(tmp)/str(micro))
                cfg['runtime']['micro_batch'] = micro
                run_training(cfg, None)
                saved = torch.load(layout(cfg, 'unit')['ckpt']/'final.pt', weights_only=False)
                states.append(saved['model'])
            for name in states[0]:
                for state in states[1:]:
                    torch.testing.assert_close(states[0][name], state[name], atol=1e-6, rtol=1e-5)

    def test_resume_replays_only_incomplete_epoch_and_restores_optimizer_rng(self):
        from research_pipeline import engine
        with tempfile.TemporaryDirectory() as tmp:
            reference = config(Path(tmp)/'reference', epochs=3)
            interrupted = config(Path(tmp)/'interrupted', epochs=3)
            run_training(reference, None)
            evaluate = engine.evaluate
            calls = [0]
            def failing(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('simulated validation interruption')
                return evaluate(*args, **kwargs)
            with mock.patch.object(engine, 'evaluate', failing):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    run_training(interrupted, None)
            saved = torch.load(layout(interrupted, 'unit')['ckpt']/'last.pt', weights_only=False)
            self.assertEqual(saved['progress']['epoch'], 1)
            run_training(interrupted, None)
            a = torch.load(layout(reference, 'unit')['ckpt']/'final.pt', weights_only=False)
            b = torch.load(layout(interrupted, 'unit')['ckpt']/'final.pt', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            history = (layout(interrupted, 'unit')['result']/'metrics.csv').read_text(encoding='utf-8-sig')
            self.assertEqual(len(history.strip().splitlines()), 4)

    def test_frozen_training_does_not_change_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, 'seg', 'frozen_linear')
            before = build_experiment(cfg, None, torch.device('cpu')).model.backbone.state_dict()
            run_training(cfg, None)
            final = torch.load(layout(cfg, 'unit')['ckpt']/'final.pt', weights_only=False)['model']
            for name, value in before.items():
                torch.testing.assert_close(value, final['backbone.'+name], rtol=0, atol=0)

    def test_completed_artifact_and_protocol_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp)
            summary = run_training(cfg, None)
            self.assertEqual(run_training(cfg, None)['epoch'], summary['epoch'])
            cfg['training']['lr'] *= 2
            with self.assertRaisesRegex(ValueError, 'protocol changed'):
                run_training(cfg, None)
            (layout(cfg, 'unit')['result']/'patients.json').write_text('[]')
            with self.assertRaisesRegex(RuntimeError, 'modified'):
                verify_done(layout(cfg, 'unit')['result'])

    def test_early_stop_is_not_applied_to_mae(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, 'ef', epochs=6)
            cfg['training'].update(patience=1, min_delta=1000.)
            summary = run_training(cfg, None)
            self.assertEqual(summary['epoch'], 2)
            self.assertTrue(summary['stopped_early'])
            self.assertEqual(summary['selected_epoch'], 1)

    def test_export_excludes_weights_cache_and_checks_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'result'; root.mkdir()
            (root/'summary.json').write_text('{}')
            for name in ('model.pt', 'features.npy', 'data.npz', 'other.zip'):
                (root/name).write_bytes(b'not for export')
            target = Path(tmp)/'results.zip'
            info = export_results(root, target)
            self.assertEqual(info['files'], 1)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(set(archive.namelist()), {'summary.json', 'EXPORT_MANIFEST.json'})

    def test_duplicate_run_is_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, 'Another owner'):
                    with run_lock(Path(tmp)):
                        pass

    def test_spawn_loader_reads_cached_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, 'ef', 'frozen_linear')
            exp = build_experiment(cfg, None, torch.device('cpu'))
            prepare_cache(exp, Path(tmp)/'cache', dict(version=1), 16*1024**2, 0)
            selection = dict(micro_batch=2, num_workers=1, gradient_checkpointing=False)
            batches = list(loader(exp.train, cfg, selection))
            self.assertEqual(sum(map(len, batches)), len(exp.train))
            self.assertIn('features', batches[0][0])


class MetricTests(unittest.TestCase):
    def test_ef_aggregates_window_predictions_before_patient_error(self):
        rows = [dict(patient='a', target=50, prediction=40), dict(patient='a', target=50, prediction=60),
                dict(patient='b', target=60, prediction=62)]
        result, patients = summarize(rows, 'ef')
        self.assertEqual(result['mae_pp'], 1.)
        self.assertEqual(patients[0]['absolute_error'], 0.)

    def test_seg_counts_source_frame_once_and_retains_empty_failures(self):
        truth = np.ones((4, 4), dtype=bool)
        perfect = segmentation_row(truth, truth, patient='a', source_frame=1)
        bad = segmentation_row(~truth, truth, patient='b', source_frame=2, boundaries=True)
        result, patients = summarize([perfect]*8+[bad], 'seg')
        self.assertEqual(result['dice_mean'], .5)
        self.assertEqual(result['source_frames'], 2)
        self.assertEqual(result['boundary_failure_frames'], 1)

    def test_boundary_spacing_and_units(self):
        a = np.zeros((10, 10), bool); a[3:6, 3:6] = True
        b = np.roll(a, 1, axis=0)
        row = segmentation_row(a, b, patient='p', source_frame=0, boundaries=True, spacing=[2, 2])
        pix = segmentation_row(a, b, patient='p', source_frame=0, boundaries=True)
        self.assertAlmostEqual(row['assd'], pix['assd']*2)
        self.assertEqual(row['boundary_unit'], 'mm')

    def test_matched_bootstrap_and_representation_controls(self):
        a = [dict(patient='a', dice=.8), dict(patient='b', dice=.9)]
        b = [dict(patient='a', dice=.81), dict(patient='b', dice=.92)]
        self.assertAlmostEqual(paired_bootstrap(a, b, 'dice', 100)['candidate_minus_reference'], .015)
        with self.assertRaisesRegex(ValueError, 'matched'):
            paired_bootstrap(a, b[:1], 'dice', 10)
        x = np.random.default_rng(1).normal(size=(10, 5))
        self.assertAlmostEqual(linear_cka(x, x), 1.)
        self.assertEqual(representation_summary(np.ones((4, 3)))['effective_rank'], 0.)


if __name__ == '__main__':
    unittest.main()
