import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from test_final_temporal_data import fixture
from research_pipeline.artifacts import digest, layout, verify_done, write_json
from research_pipeline.config import load, stage_config
from research_pipeline.coverage_repair import CoverageDataset, DrawBudgetDataset, build
from research_pipeline.engine import evaluate, run_training
from research_pipeline.official_anchor import FrameDataset
from tools.run_segmentation_gap_stage1 import build_plans
from utils.final_temporal_data import build_manifest, WindowDataset, _trace_mask


class CoverageRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_labels_and_unchanged_common_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            manifest = build_manifest(root, max_prefix=0, minimum_frames=1)
            common = WindowDataset(manifest, 'train', task='seg', max_prefix=0, positions='balanced')
            full = CoverageDataset(manifest, 'train', positions='balanced')
            expected = {(c['patient'], t['frame']) for c in manifest['train'] for t in c['traces']}
            self.assertEqual({(r['patient'], r['target_frame']) for r in full.records}, expected)
            self.assertGreater(len(full), len(common))
            for i, record in enumerate(full.records):
                sample = full[i]
                self.assertEqual(int(sample['frame_indices'][sample['target_index']]), record['target_frame'])
                self.assertTrue(bool((sample['frame_indices'] >= 0).all()))
                self.assertLess(int(sample['frame_indices'].max()), full.cases[record['patient']]['frames'])
                trace = next(t for t in full.cases[record['patient']]['traces'] if t['frame'] == record['target_frame'])
                np.testing.assert_array_equal(sample['mask'].numpy(), _trace_mask(trace))
                if 'legacy_index' in record:
                    original = common[record['legacy_index']]
                    torch.testing.assert_close(sample['video'], original['video'])
                    torch.testing.assert_close(sample['frame_indices'], original['frame_indices'])
                    self.assertEqual(sample['target_index'], original['target_index'])
                else:
                    self.assertEqual(len(sample['video']) % 16, 0)
                    self.assertLessEqual(int(sample['frame_indices'].max()), record['target_frame'])

    def test_fixed_draws_visit_pool_without_replacement_before_next_cycle(self):
        class Pool:
            seed = 42
            epoch = types.SimpleNamespace(value=0)
            records = list(range(11))
            cases, excluded = {}, []
            def __len__(self): return 11
            def __getitem__(self, i): return i
            def set_epoch(self, e): self.epoch.value = e
        ds = DrawBudgetDataset(Pool(), 4)
        visited = []
        for epoch in range(3):
            ds.set_epoch(epoch)
            visited.extend(ds[i] for i in range(len(ds)))
        self.assertEqual(set(visited[:11]), set(range(11)))
        self.assertEqual(len(visited), 12)
        ds.set_epoch(0)
        self.assertEqual([ds[i] for i in range(4)], visited[:4])
        from research_pipeline.runtime import loader
        from research_pipeline.echo import IdentifiedDataset
        config = load('configs/pipeline/clip_recurrent_patch8.yaml')
        config['runtime']['device'] = 'cpu'
        config['training']['effective_batch'] = 4
        self.assertFalse(loader(IdentifiedDataset(ds, 'train'), config,
                        dict(micro_batch=2, num_workers=0)).batch_sampler.shuffle)

    def test_stage_plan_never_adds_pretraining(self):
        cfg = load('configs/pipeline/clip_recurrent_patch8.yaml')
        plans = build_plans(cfg, '/ssl.pt', '/full.json', Path('/out'), 'gap', '/official.pt')
        self.assertEqual(plans['coverage']['adapter'], 'research_pipeline.coverage_repair:build')
        for value in plans.values():
            self.assertEqual(len(value['stages']), 1)
            self.assertEqual(value['stages'][0]['kind'], 'seg')
        self.assertEqual(plans['anchor']['training']['monitor'], 'val_loss')
        self.assertEqual(plans['anchor']['training']['min_epochs'], 25)
        self.assertEqual(plans['coverage']['data']['evaluation_minimum_frames'], 1)

    def test_queue_requires_all_test_calls_and_stops_on_failure(self):
        import subprocess
        from tools.run_segmentation_gap_stage1 import main as queue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            outputs = root/'new'
            write_json(outputs/'manifests/full_supervision.json', build_manifest(root,max_prefix=0,minimum_frames=1))
            source_root = root/'old'
            parent = load('configs/pipeline/clip_recurrent_patch8.yaml')
            parent_dir=source_root/'result/source'
            write_json(parent_dir/'plan.json',parent)
            metric=dict(metrics=dict(dice_mean=.9,dice_global_mean=.91))
            write_json(parent_dir/'seg_decoder_transfer_ft/summary.json',metric)
            source=source_root/'ckpt/source/pretrain/epoch_0400.pt'
            source.parent.mkdir(parents=True)
            source.touch()
            official=root/'official.pt'; official.touch()
            calls=[]
            def run(command, **kwargs):
                calls.append(command)
                if '-m' in command:
                    plan=json.loads(Path(command[-1]).read_text())
                    directory=Path(plan['run']['root'])/'result'/plan['run']['tag']/plan['stages'][0]['id']
                    write_json(directory/'summary.json',metric)
            args=['queue','--action','run','--source-root',str(source_root),'--source-tag','source',
                  '--root',str(outputs),'--data-root',str(root),'--official-checkpoint',str(official)]
            with mock.patch.object(sys,'argv',args), mock.patch('tools.run_segmentation_gap_stage1.verify_done',return_value=True), mock.patch('tools.run_segmentation_gap_stage1.subprocess.run',side_effect=run):
                queue()
            tests=[c for c in calls if any('evaluate_patch8_test.py' in str(v) for v in c) and '--split' not in c]
            self.assertEqual(len(tests),3)
            self.assertFalse(any('pretrain' in c for c in calls))
            status=json.loads((outputs/'control/seg_gap_stage1_v1/status.json').read_text())
            self.assertTrue(status['new_test_completed'])
            with mock.patch.object(sys,'argv',args), mock.patch('tools.run_segmentation_gap_stage1.verify_done',return_value=True), mock.patch('tools.run_segmentation_gap_stage1.subprocess.run',side_effect=subprocess.CalledProcessError(1,'fixture')) as fail:
                with self.assertRaises(subprocess.CalledProcessError): queue()
                self.assertEqual(fail.call_count,1)
            self.assertEqual(json.loads((outputs/'control/seg_gap_stage1_v1/status.json').read_text())['phase'],'failed')

    def test_cpu_full_coverage_forward_gradients_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            common = build_manifest(root, max_prefix=0)
            full = build_manifest(root, max_prefix=0, minimum_frames=1)
            write_json(root/'common.json', common)
            write_json(root/'full.json', full)
            cfg = load('configs/pipeline/clip_recurrent_patch8.yaml', root=str(root/'out'), tag='cpu')
            cfg['model']['init_checkpoint'] = None
            cfg['model']['config'].update(embed_dim=24, depth=1, num_heads=4,
                                         decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3)
            cfg['data'].update(manifest=str(root/'common.json'), coverage_manifest=str(root/'full.json'),
                               limit_train=2, limit_val=2, augmentation={'enabled': False},
                               evaluation_minimum_frames=1)
            cfg['runtime'].update(device='cpu', precision='fp32', autotune=False, micro_batch=1,
                                  worker_candidates=[0], checkpoint_candidates=[False])
            stage = dict(id='seg_full_coverage_ft', kind='seg', mode='finetune',
                         training=dict(epochs=1, effective_batch=2, warmup_epochs=0, tensorboard=False))
            cfg['adapter'] = 'research_pipeline.coverage_repair:build'
            cfg['stages'] = [stage]
            current = stage_config(cfg, stage)
            from models.final_temporal_mae import FinalTemporalMAE
            native = cfg['model']['config']
            native_model = FinalTemporalMAE(**native)
            source = root/'native.pt'
            torch.save(dict(model_state_dict=native_model.state_dict(), config={'model': native}), source)
            exp = build(current, stage, source, torch.device('cpu'))
            dataset = exp.train.dataset.dataset
            recovered = next(i for i, r in enumerate(dataset.records) if 'legacy_index' not in r)
            exp.train_mode()
            loss = exp.loss([dataset[recovered]])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(exp.backbone.patch_embed.proj.weight.grad.abs().sum()), 0)
            result = run_training(current, source)
            self.assertTrue(verify_done(layout(current, stage['id'])['result']))
            self.assertEqual(result['epoch'], 1)
            protocol = layout(current, stage['id'])['result']/'protocol.json'
            before = protocol.read_bytes()
            with mock.patch('research_pipeline.engine.provenance', return_value={'changed_environment': True}):
                run_training(current, source)
            self.assertEqual(protocol.read_bytes(), before)
            directory = layout(current, stage['id'])['result']
            write_json(directory/'request.json', dict(config=current, source=str(source)))
            from tools.evaluate_patch8_test import main as evaluate_endpoint
            arguments = ['evaluate', '--root', str(root/'out'), '--tag', 'cpu',
                         '--stages', stage['id'], '--split', 'val_full', '--data-root', str(root)]
            with mock.patch.object(sys, 'argv', arguments):
                evaluate_endpoint()
            full_result = root/'out/result/cpu_val_full'
            self.assertTrue(verify_done(full_result))
            score = json.loads((full_result/stage['id']/'summary.json').read_text())
            self.assertEqual(score['split'], 'VAL_FULL_NO_SELECTION')
            self.assertEqual(score['metrics']['source_frames'], 3)


class OfficialPortParityTests(unittest.TestCase):
    def test_anchor_cpu_training_and_locked_test_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            np.save(root/'npy/sealed.npy', np.zeros((200,112,112), dtype=np.uint8))
            import pandas as pd
            from test_final_temporal_data import POINTS
            traces = pd.read_csv(root/'VolumeTracings.csv')
            extra = [dict(FileName='sealed.avi', Frame=100, X1=a, Y1=b, X2=c, Y2=d) for a,b,c,d in POINTS]
            pd.concat([traces,pd.DataFrame(extra)]).to_csv(root/'VolumeTracings.csv', index=False)
            write_json(root/'full.json', build_manifest(root,max_prefix=0,minimum_frames=1))
            from models.echocardmae_official import build_echocardmae_official_seg
            model_cfg = dict(img_size=112,patch_size=8,embed_dim=24,depth=1,num_heads=4,
                             decoder_embed_dim=24,decoder_depth=1,decoder_num_heads=4)
            source = root/'official.pt'
            torch.save({'model':build_echocardmae_official_seg(model_cfg).state_dict()},source)
            cfg=load('configs/pipeline/clip_recurrent_patch8.yaml',root=str(root/'out'),tag='anchor')
            cfg['adapter']='research_pipeline.official_anchor:build'
            cfg['model']['config']=model_cfg
            cfg['data'].update(coverage_manifest=str(root/'full.json'),evaluation_minimum_frames=1,
                               limit_train=2,limit_val=2)
            cfg['runtime'].update(device='cpu',precision='fp32',autotune=False,micro_batch=1,worker_candidates=[0])
            stage=dict(id='official_seg_ft',kind='seg',mode='finetune',
                       training=dict(epochs=1,effective_batch=2,warmup_epochs=0,tensorboard=False,
                                     monitor='val_loss',min_epochs=1))
            cfg['stages']=[stage]
            current=stage_config(cfg,stage)
            result=run_training(current,source)
            directory=layout(current,stage['id'])['result']
            write_json(directory/'request.json',dict(config=current,source=str(source)))
            from tools.evaluate_patch8_test import main as endpoint
            with mock.patch.object(sys,'argv',['evaluate','--root',str(root/'out'),'--tag','anchor',
                                               '--stages',stage['id'],'--data-root',str(root)]):
                endpoint()
            scored=json.loads((root/'out/result/anchor_test/official_seg_ft/summary.json').read_text())
            self.assertEqual(scored['metrics']['patients'],1)
            self.assertEqual(scored['metrics']['source_frames'],1)
            self.assertEqual(scored['split'],'TEST')

    def test_forward_loss_gradient_against_downloaded_author_source(self):
        base = Path('资料/EchoCardMAE-main/EchoCardMAE-main/model')
        if not (base/'vit_seg.py').is_file():
            self.skipTest('Local author reference not present; cloud requires recorded parity evidence')
        if importlib.util.find_spec('einops') is None:
            self.skipTest('Author source requires einops; run parity with the isolated test dependency path')
        package = types.ModuleType('_gap_author')
        package.__path__ = [str(base.resolve())]
        sys.modules['_gap_author'] = package
        spec = importlib.util.spec_from_file_location('_gap_author.vit_seg', base/'vit_seg.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from models.echocardmae_official import EchoCardMAEOfficialSeg
        args = dict(img_size=112, patch_size=8, encoder_embed_dim=24, encoder_depth=1,
                    encoder_num_heads=4, decoder_embed_dim=24, decoder_depth=1,
                    decoder_num_heads=4, qkv_bias=True)
        torch.set_num_threads(1)
        reference = module.ViTSeg2D(**args)
        port = EchoCardMAEOfficialSeg(**args, normalize_input=False)
        port.load_state_dict(reference.state_dict(), strict=True)
        image = torch.rand(2, 3, 112, 112)
        target = F.one_hot(torch.randint(0, 2, (2, 112, 112)), 2).permute(0, 3, 1, 2).float()
        a, b = reference(image), port(image)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        def dice(x):
            p = x.softmax(1)
            return (1-(2*(p*target).sum((2,3))+1e-5)/(p.sum((2,3))+target.sum((2,3))+1e-5)).mean()
        la, lb = dice(a), dice(b)
        loss_spec = importlib.util.spec_from_file_location('_gap_author_loss', base.parent/'loss/dice_loss.py')
        loss_module = importlib.util.module_from_spec(loss_spec)
        loss_spec.loader.exec_module(loss_module)
        actual_author_loss = loss_module.DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        la = actual_author_loss(a, target.argmax(1)[:, None])
        torch.testing.assert_close(la, lb)
        la.backward(); lb.backward()
        for name, param in reference.named_parameters():
            other = dict(port.named_parameters())[name]
            if param.grad is None:
                self.assertIsNone(other.grad)
            else:
                torch.testing.assert_close(param.grad, other.grad, atol=3e-6, rtol=3e-4)
        torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=1e-4).step()
        torch.optim.AdamW(port.parameters(), lr=1e-4, weight_decay=1e-4).step()
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, port.state_dict()[name], atol=3e-6, rtol=3e-4)


if __name__ == '__main__':
    unittest.main()
