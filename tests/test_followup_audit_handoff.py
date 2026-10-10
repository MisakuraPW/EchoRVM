import copy
import unittest

from tools.handoff_temporal_followup_audit import verify_upgrade


class FollowupUpgradeTests(unittest.TestCase):
    def test_only_two_registered_files_can_change_and_settings_are_preserved(self):
        old = dict(code={'utils/final_temporal_tasks.py': 'old_task',
                         'utils/final_temporal_training.py': 'old_train', 'model.py': 'same'},
                   config={'updates': 1500, 'ft_epochs': 80}, sources={'C': 'fixed'}, manifest='fixed')
        new = dict(old['code'], **{'utils/final_temporal_tasks.py': 'new_task',
                                  'utils/final_temporal_training.py': 'new_train'})
        expected = {key: new[key] for key in new if key.startswith('utils/')}
        result = verify_upgrade(old, new, expected)
        self.assertEqual(result['config'], old['config'])
        self.assertEqual(result['sources'], old['sources'])
        self.assertEqual(result['manifest'], old['manifest'])
        self.assertEqual(old['code']['utils/final_temporal_tasks.py'], 'old_task')
        for change in ('model', 'uncertified', 'missing'):
            altered = copy.deepcopy(new)
            if change == 'model':
                altered['model.py'] = 'changed'
            elif change == 'uncertified':
                altered['utils/final_temporal_tasks.py'] = 'other'
            else:
                altered.pop('model.py')
            with self.assertRaises(ValueError):
                verify_upgrade(old, altered, expected)


if __name__ == '__main__':
    unittest.main()
