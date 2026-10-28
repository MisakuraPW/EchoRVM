import unittest

from tools.tune_mae_batch import select_batch
from research_pipeline.config import load, stage_config
from research_pipeline.forward_suite import build_plan


class BatchSearchTests(unittest.TestCase):
    def test_oom_and_unsafe_candidates_are_not_selected(self):
        rows = [dict(status='ok', samples_per_second=40, selection=dict(micro_batch=32)),
                dict(status='ok', samples_per_second=65, selection=dict(micro_batch=64)),
                dict(status='oom', selection=dict(micro_batch=128)),
                dict(status='headroom_or_skipped_update', samples_per_second=90, selection=dict(micro_batch=256))]
        self.assertEqual(select_batch(rows)['selection']['micro_batch'], 64)

    def test_smaller_batch_wins_three_percent_tie(self):
        rows = [dict(status='ok', samples_per_second=49, selection=dict(micro_batch=32)),
                dict(status='ok', samples_per_second=50, selection=dict(micro_batch=64))]
        self.assertEqual(select_batch(rows)['selection']['micro_batch'], 32)

    def test_no_success_fails_closed(self):
        with self.assertRaises(RuntimeError):
            select_batch([dict(status='oom')])

    def test_ssl_execution_overrides_match_a_without_capping_task_search(self):
        cfg = load('configs/pipeline/clip_recurrent_global.yaml', overrides=[
            'stages.pretrain.training.effective_batch=64',
            'stages.pretrain.runtime.batch_candidates=[32,64]',
            'stages.pretrain.runtime.checkpoint_candidates=[false]'])
        plan, _ = build_plan(cfg)
        a = next(s for s in plan['stages'] if s['id'] == 'A')
        self.assertEqual(a['runtime'], plan['stages'][0]['runtime'])
        self.assertEqual(stage_config(plan, a)['training']['effective_batch'], 64)
        ft = next(s for s in plan['stages'] if s['id'] == 'ef_ft')
        task = stage_config(plan, ft)
        self.assertEqual(task['training']['effective_batch'], 32)
        self.assertEqual(task['runtime']['batch_candidates'], [1, 2, 4, 8, 16, 32])
