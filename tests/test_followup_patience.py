import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools.run_temporal_final import Queue, parse_args
from tools.handoff_temporal_followup_audit import verify_patience_upgrade
from utils.final_temporal_selection import write_final_report


class FollowupPatienceTests(unittest.TestCase):
    def test_p_stays20_followup_ft10_frozen12_and_epoch_caps_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / 'patients.csv'
            predictions.write_text('patient\na\nb\n', encoding='utf-8')
            args = SimpleNamespace(output_root=str(root), run_tag='test', smoke=False, max_prefix=128,
                updates=1500, optional_ft=True, second_head_seed=True, num_workers=8, min_free_gb=5,
                frozen_epochs=60, ft_epochs=80, followup_ft_patience=10, cache_disk_gb=16,
                cache_ram_gb=4, device='cpu')
            queue = Queue(args, {'C': 'source.pt'}, {'C': {'model': {'local_frames': 16}}}, {})
            jobs = []
            def run(label, job):
                jobs.append(job)
                return dict(patient_predictions_path=str(predictions), evaluation_population_sha256='same')
            with mock.patch.object(queue, 'run', side_effect=run):
                for model in ('P', 'C', 'F', 'B0'):
                    for task in ('ef', 'seg'):
                        for freeze in (False, True):
                            result = queue.task(model, 'source.pt', freeze, task)
                            self.assertEqual(result['patience'], 12 if freeze else 20 if model == 'P' else 10)
                            self.assertEqual(jobs[-1]['epochs'], 60 if freeze else 80)
                            self.assertEqual(jobs[-1]['effective_batch'], 12 if task == 'ef' else 64)

    def test_cli_default_preserves_legacy_and_explicit10_is_supported(self):
        with mock.patch.object(sys, 'argv', ['queue']):
            self.assertEqual(parse_args().followup_ft_patience, 20)
        with mock.patch.object(sys, 'argv', ['queue', '--followup_ft_patience', '10']):
            self.assertEqual(parse_args().followup_ft_patience, 10)

    def test_protocol_upgrade_changes_only_future_patience_and_verified_code(self):
        old = dict(code={'tools/run_temporal_final.py': 'old_queue',
                         'utils/final_temporal_selection.py': 'old_report', 'model.py': 'fixed'},
                   config={'ft_epochs': 80, 'frozen_epochs': 60, 'updates': 1500}, sources={'C': 'fixed'})
        new = dict(old['code'], **{'tools/run_temporal_final.py': 'new_queue',
                                  'utils/final_temporal_selection.py': 'new_report'})
        expected = {key: new[key] for key in new if key != 'model.py'}
        result = verify_patience_upgrade(old, new, expected)
        self.assertEqual(result['config'], dict(old['config'], followup_ft_patience=10))
        self.assertEqual(result['sources'], old['sources'])
        self.assertNotIn('followup_ft_patience', old['config'])
        for kind in ('model', 'uncertified', 'already10'):
            before, altered = copy.deepcopy(old), copy.deepcopy(new)
            if kind == 'model':
                altered['model.py'] = 'changed'
            elif kind == 'uncertified':
                altered['tools/run_temporal_final.py'] = 'other'
            else:
                before['config']['followup_ft_patience'] = 10
            with self.assertRaises(ValueError):
                verify_patience_upgrade(before, altered, expected)

    def test_report_discloses_mixed_ft_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            scores = {'P_FT': dict(ef=dict(mae=4, patience=20, epochs_completed=80),
                                   seg=dict(patient_dice=.92, patience=20, epochs_completed=70)),
                      'C_FT': dict(ef=dict(mae=4.1, patience=10, epochs_completed=43),
                                   seg=dict(patient_dice=.91, patience=10, epochs_completed=54))}
            result = write_final_report(folder, {}, scores)
            report = Path(result['report']).read_text(encoding='utf-8')
            self.assertIn('not an equal-stop-policy FT ranking', report)
            with Path(result['comparison']).open(newline='', encoding='utf-8') as handle:
                rows = {row['model_id']: row for row in csv.DictReader(handle)}
            self.assertEqual(rows['P_FT']['seg_patience'], '20')
            self.assertEqual(rows['C_FT']['seg_patience'], '10')
            self.assertEqual(rows['P_FT']['ef_epochs_completed'], '80')


if __name__ == '__main__':
    unittest.main()
