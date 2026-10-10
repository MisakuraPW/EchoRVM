import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tools.run_temporal_final import Queue


class QueueContractTests(unittest.TestCase):
    def queue(self, root, smoke=False):
        args = SimpleNamespace(output_root=str(root),run_tag='test',smoke=smoke,max_prefix=128,
            updates=1500,optional_ft=False,second_head_seed=False,num_workers=0,min_free_gb=0,
            frozen_epochs=60,ft_epochs=80,cache_disk_gb=1,cache_ram_gb=1,device='cpu')
        cfg = dict(local_frames=16,frames=64,depth=12,img_size=112,embed_dim=384)
        return Queue(args,{'C':'canonical.pt'}, {'C':{'model':cfg}}, {'train':[],'val':[]})

    def test_r5_merge_preserves_sign_patients_and_rejects_changed_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(Path(folder))
            queue.scores = {name:dict(history=dict(anchor=64), representation=dict(patient_observations=[dict(patient='p')]))
                            for name in ('B0','B1')}
            report = dict(status='complete',cohort_hash='common',target_projection_hash='projection',
                          history_anchor=64,cache_frames=64,true_state_error=dict(mean=.1),zero_slots_error=dict(mean=.3))
            value = dict(memory_prediction_delta=.2,memory_prediction=report,
                         patient_observations=[dict(patient='p',delta=.2)])
            with mock.patch.object(queue,'run',return_value=value):
                queue.representation('B0','B0.pt',memory=True)
            r = queue.scores['B0']['representation']
            self.assertEqual(r['patient_observations'][0]['memory_prediction_delta'],.2)
            self.assertEqual(r['memory_prediction_delta_convention'],'zero_minus_true')
            changed = copy.deepcopy(value)
            changed['memory_prediction']['target_projection_hash'] = 'different'
            with mock.patch.object(queue,'run',return_value=changed), self.assertRaises(ValueError):
                queue.representation('B1','B1.pt',memory=True)

    def test_exit_decision_is_per_task_no_second_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(Path(folder))
            exits = dict(local_base=dict(ef=dict(mae=5.1),seg=dict(dice_patient_mean=.89)),
                         fused_base=dict(ef=dict(mae=4.8),seg=dict(dice_patient_mean=.9)),
                         fused_final=dict(ef=dict(mae=5.4),seg=dict(dice_patient_mean=.92)))
            result = queue.exit_decision('F',exits)
            self.assertEqual(result['task_exits'],dict(ef='fused_base',seg='fused_final'))

    def test_cache_cleanup_stays_in_owned_run(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = self.queue(Path(folder))
            child = queue.cache/'job'; child.mkdir(); (child/'fixture.txt').write_text('test')
            queue.cleanup_cache(dict(cache_dir=str(child)))
            self.assertFalse(child.exists())
            with self.assertRaises(ValueError):
                queue.cleanup_cache(dict(cache_dir=str(queue.result)))


if __name__ == '__main__':
    unittest.main()
