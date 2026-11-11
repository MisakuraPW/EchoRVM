import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from models.final_temporal_mae import FinalTemporalMAE, load_final_model
from research_pipeline.artifacts import file_hash, layout, verify_done, write_json
from research_pipeline.cli import execute, work
from research_pipeline.config import load, stage_config
from research_pipeline.ssl_data import build_ssl_manifest
from research_pipeline.stage_a import build
from research_pipeline.stage_a_diagnostics import masked_routes, verify_visibility
from research_pipeline.forward_diagnostics import gradient_credit
from test_final_temporal_data import fixture
from test_research_pipeline_forward import native
from tools.run_patch8_stage_a import build_plan, lock_selection, match_observations, read, smoke_config
from utils.final_temporal_data import build_manifest


def reference():
    cfg = load('configs/pipeline/clip_recurrent_patch8.yaml')
    stages = {s['id']:s for s in cfg['stages']}
    seg = stage_config(cfg,stages['seg_decoder_transfer_ft'])
    seg['stage']['id'] = 'seg_full_coverage_ft'
    seg['data'].update(coverage_manifest='/fixtures/full.json',evaluation_minimum_frames=1)
    ef = stage_config(cfg,stages['ef_l1_ft'])
    return dict(base=cfg, stages=stages, zero='/fixtures/epoch_0000.pt', endpoint='/fixtures/epoch_0400.pt',
                lock={'epoch0':'zero','epoch400':'endpoint'}, ef_probe=Path('/fixtures/ef_linear'),
                ft={task:(Path('/fixtures')/task,{},f'/fixtures/{task}.pt',dict(config=stage))
                    for task,stage in [('ef',ef),('seg',seg)]})


class StageATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_plan_has_two_independent_runs_and_ft_is_not_probe_gated(self):
        ref = reference()
        b8 = build_plan(ref,'/out','stage_a','B8')
        self.assertEqual([s['id'] for s in b8['stages']],['seg_linear_0400','diagnostics'])
        for arm in ('AR','AW'):
            cfg = build_plan(ref,'/out','stage_a',arm)
            self.assertEqual([s['id'] for s in cfg['stages']],['pretrain','ef_l1_ft','seg_full_coverage_ft',
                                                            'ef_linear_0400','seg_linear_0400','diagnostics'])
            ssl = stage_config(cfg,cfg['stages'][0])
            self.assertEqual(ssl['stage']['source'],ref['zero'])
            self.assertEqual(ssl['training']['epochs'],400)
            self.assertIsNone(ssl['training']['patience'])
            self.assertIsNone(ssl['training']['schedule_epochs'])
            self.assertEqual(ssl['training']['save_epochs'],[0,100,200,400])
            self.assertEqual(ssl['model']['overrides']['forward_variant'],'mid_read' if arm=='AR' else 'content_write')
            for stage in cfg['stages'][1:5]:
                self.assertEqual(stage['source'],'@pretrain/epoch_0400')
            smoke = smoke_config(cfg,cpu=True)
            self.assertEqual(smoke['stages'][1]['source'],'@pretrain/final')
            self.assertEqual(smoke['stages'][0]['training']['epochs'],1)
            self.assertEqual(smoke['stages'][0]['training']['save_epochs'],[0,1])
            self.assertEqual(stage_config(smoke,smoke['stages'][1])['training']['lr'],1e-5)
            self.assertTrue(all(not stage_config(smoke,s)['runtime']['autotune'] for s in smoke['stages']))

    def test_frozen_segmentation_uses_all_labels_without_altering_selection_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            common = build_manifest(root,max_prefix=0)
            full = build_manifest(root,max_prefix=0,minimum_frames=1)
            write_json(root/'common.json',common)
            write_json(root/'full.json',full)
            cfg = load('configs/pipeline/clip_recurrent_patch8.yaml')
            cfg['model'] = dict(config=dict(native(),img_size=112,local_frames=16),overrides={})
            cfg['data'].update(manifest=str(root/'common.json'),coverage_manifest=str(root/'full.json'))
            stage = dict(id='seg_probe',kind='seg',mode='frozen_linear')
            exp = build(stage_config(cfg,stage),stage,None,torch.device('cpu'))
            expected = {(case['patient'],trace['frame']) for case in full['train'] for trace in case['traces']}
            records = exp.train.dataset.records
            self.assertEqual({(r['patient'],r['target_frame']) for r in records},expected)
            self.assertEqual(len(exp.train),len(expected))
            self.assertEqual(len(exp.val),32)
            self.assertTrue(exp.cacheable)
            self.assertFalse(any(p.requires_grad for p in exp.backbone.parameters()))
            self.assertFalse(exp.train.dataset.training)

    def test_masked_route_and_credit_keep_the_past_connected(self):
        cfg = native()
        b8 = FinalTemporalMAE(**cfg)
        video = torch.rand(1,12,3,16,16)
        masks = torch.tensor([[[False,False,True,True]*2]*3])
        with tempfile.TemporaryDirectory() as tmp:
            zero = Path(tmp)/'zero.pt'
            torch.save(dict(model_state_dict=b8.state_dict(),config=dict(model=cfg),epoch=0),zero)
            for variant in ('mid_read','content_write'):
                model,_,_ = load_final_model(zero,dict(forward_variant=variant,read_after_block=2))
                model.eval()
                checks = verify_visibility(model,video,masks)
                self.assertTrue(checks['future_clips_isolated'])
                self.assertTrue(checks['hidden_pixels_isolated'])
                credit = gradient_credit(model,video,masks)
                self.assertGreater(credit['time_specific_writes'][0]['norm'],0)
                self.assertFalse(credit['states'][-1]['connected'])
                with torch.no_grad():
                    routes = masked_routes(model,video,masks)
                self.assertAlmostEqual(routes['normal'],credit['loss'],places=5)
                self.assertNotEqual(routes['normal'],routes['decoder_S_zero'])

    def test_selection_and_population_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_selection(root/'lock.json',dict(candidate='AR',sha='a'))
            lock_selection(root/'lock.json',dict(candidate='AR',sha='a'))
            with self.assertRaisesRegex(ValueError,'already locked'):
                lock_selection(root/'lock.json',dict(candidate='AW',sha='b'))
            row = dict(patient='a',source_frame='3',target_position=63,class_id=1,target_pixels=2)
            write_json(root/'a'/'observations.json',[row])
            write_json(root/'b'/'observations.json',[copy.deepcopy(row)])
            match_observations(root/'a',root/'b','seg')
            write_json(root/'b'/'observations.json',[dict(row,target_position=62)])
            with self.assertRaisesRegex(ValueError,'population'):
                match_observations(root/'a',root/'b','seg')

    def test_file_backed_candidate_smoke_ft_probes_and_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            np.save(root/'npy'/'sealed.npy',np.zeros((200,112,112),np.uint8))
            common = build_manifest(root,max_prefix=0)
            full = build_manifest(root,max_prefix=0,minimum_frames=1)
            write_json(root/'common.json',common)
            write_json(root/'full.json',full)
            write_json(root/'ssl.json',build_ssl_manifest(root,local_frames=16,recent_frames=64,max_prefix=0))
            ref = reference()
            model_cfg = dict(native(),img_size=112,local_frames=16,depth=12,clip_count=4,
                             decoder_embed_dim=192,decoder_depth=4,decoder_num_heads=3,
                             decoder_embed_bias=False,separate_qv_bias=True,reconstruction_recent_frames=64)
            model = FinalTemporalMAE(**model_cfg)
            zero = root/'epoch_0000.pt'
            torch.save(dict(model_state_dict=model.state_dict(),config=dict(model=model_cfg),epoch=0),zero)
            ref.update(zero=str(zero),lock={'epoch0':file_hash(zero),'epoch400':'not_used_in_smoke'})
            ref['base']['model'] = dict(config=model_cfg,overrides={})
            data = dict(manifest=str(root/'common.json'),ssl_manifest=str(root/'ssl.json'))
            ref['base']['data'].update(data)
            for task in ('ef','seg'):
                ref['ft'][task][3]['config']['data'].update(data)
            ref['ft']['seg'][3]['config']['data']['coverage_manifest'] = str(root/'full.json')
            cfg = smoke_config(build_plan(ref,root/'out','fixture','AW'),cpu=True)
            cfg['runtime'].update(min_free_gb=0,threads=2)
            cfg['data'].update(limit_train=1,limit_val=1)
            for stage in cfg['stages']:
                stage.setdefault('runtime',{}).update(min_free_gb=0,threads=2)
                stage['data'].update(limit_train=1,limit_val=1)
            # Small real-file CPU run through the same request/worker boundary.
            def worker(command,**kwargs):
                work(read(command[-1]))
                return SimpleNamespace(returncode=0)
            with mock.patch('research_pipeline.cli.subprocess.run',side_effect=worker):
                execute(cfg)
            out = layout(cfg)['result']
            self.assertTrue(verify_done(out))
            self.assertTrue(verify_done(out/'diagnostics'))
            credit = read(out/'diagnostics'/'temporal_credit.json')
            self.assertTrue(credit[0]['hidden_pixels_isolated'])
            self.assertTrue(credit[0]['future_clips_isolated'])
            self.assertFalse(out.with_name(out.name+'_test').exists())


if __name__=='__main__':
    unittest.main()
