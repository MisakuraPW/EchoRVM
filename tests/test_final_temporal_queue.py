import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import torch

from tools.run_temporal_final import Queue, validate_sources, LEGACY_CACHE_EXECUTION, VALIDATION_CACHE_TASK_SHA256
from utils.final_temporal_training import write_json, file_digest
from models.temporal_mae import TemporalMAE
from test_final_temporal_model import tiny_config


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

    def test_source_defaults_are_resolved_and_bad_tensors_fail_before_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = {}
            for name, frame in (('P','repeat'),('C','learned'),('F','factorized')):
                cfg = tiny_config(frame_readout=frame,dynamic_rank=4)
                if name == 'P':
                    cfg.pop('frame_readout')
                path = Path(folder)/(name+'.pt')
                torch.save(dict(model_state_dict=TemporalMAE(**cfg).state_dict(),config=dict(model=cfg),epoch=100),path)
                sources[name] = path
            contracts = validate_sources(sources,smoke=True)
            self.assertEqual(contracts['P']['model']['frame_readout'],'repeat')
            self.assertTrue(all(value['source_tensor_contract_verified'] for value in contracts.values()))
            value = torch.load(sources['P'],weights_only=False)
            value['model_state_dict']['decoder_pred.weight'] = value['model_state_dict']['decoder_pred.weight'][:1]
            torch.save(value,sources['P'])
            with self.assertRaisesRegex(ValueError,'mismatch'):
                validate_sources(sources,smoke=True)

    def upgrade_fixture(self, root):
        queue = self.queue(root)
        queue.args.adopt_validation_cache = True
        out = queue.result/'P/frozen/ef'; out.mkdir(parents=True)
        write_json(out/'metrics.json',dict(mae=4.5))
        write_json(out/'DONE',dict(artifacts={'metrics.json':file_digest(out/'metrics.json')}))
        write_json(queue.result/'jobs/P__frozen__ef.json',dict(output_dir=str(out),checkpoint_dir=str(queue.ckpt/'P/frozen/ef')))
        previous = dict(version=2, sources={'C':'fixed'}, manifest='fixed',
                        config=dict(cache_disk_gb=12,updates=1500), code=dict(LEGACY_CACHE_EXECUTION, untouched='same'))
        current = copy.deepcopy(previous)
        current['config']['cache_disk_gb']=16
        current['code'].update({'utils/final_temporal_tasks.py':VALIDATION_CACHE_TASK_SHA256,
                               'tools/run_temporal_final.py':'new_controller'})
        return queue,previous,current,out

    def test_execution_upgrade_is_once_only_and_preserves_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            queue,old,new,out = self.upgrade_fixture(Path(directory))
            before=file_digest(out/'metrics.json')
            queue.adopt_validation_cache(old,new)
            self.assertEqual(before,file_digest(out/'metrics.json'))
            self.assertTrue((queue.result/'operations/validation_cache_upgrade.json').exists())
            with self.assertRaises(ValueError):
                queue.adopt_validation_cache(old,new)

    def test_execution_upgrade_rejects_scientific_drift_partial_jobs_and_uncertified_code(self):
        for alteration in ('source','data','updates','model_code','task_code','partial','corrupt'):
            with self.subTest(alteration=alteration),tempfile.TemporaryDirectory() as directory:
                queue,old,new,out=self.upgrade_fixture(Path(directory))
                if alteration=='source': new['sources']['C']='different'
                elif alteration=='data': new['manifest']='different'
                elif alteration=='updates': new['config']['updates']=750
                elif alteration=='model_code': new['code']['untouched']='different'
                elif alteration=='task_code': new['code']['utils/final_temporal_tasks.py']='not_certified'
                elif alteration=='partial': (out/'DONE').unlink()
                else: write_json(out/'metrics.json',dict(mae=0.))
                with self.assertRaises(ValueError):
                    queue.adopt_validation_cache(old,new)
                self.assertFalse((queue.result/'operations/validation_cache_upgrade.json').exists())


if __name__ == '__main__':
    unittest.main()
