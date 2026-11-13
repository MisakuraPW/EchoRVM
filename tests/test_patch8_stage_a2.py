import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.ef_readout import TemporalEFReadout
from models.forward_temporal_mae import ForwardTemporalMAE, TemporalWriter
from research_pipeline.config import stage_config
from research_pipeline.mae_execution import active_clip_loss
from research_pipeline.stage_a_diagnostics import verify_visibility
from test_patch8_stage_a import reference
from test_research_pipeline_forward import native
from tools import run_patch8_stage_a2 as runner
from tools.run_patch8_stage_a import build_plan, smoke_config
from utils.final_temporal_training import deterministic_masks


class StageA2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_writer_exact_mean_at_init_ignores_invalid_and_can_learn_order(self):
        torch.manual_seed(21)
        writer = TemporalWriter(24, 4, 16)
        encoded = torch.randn(3, 12, 24)
        visible = torch.tensor([[True]*3+[False]*9, [True, False]*6, [False]*12])
        expected = encoded.masked_fill(~visible[...,None],0).sum(1,keepdim=True)
        expected = expected/visible.sum(1).clamp_min(1)[:,None,None]
        order = torch.arange(3,-1,-1)
        torch.testing.assert_close(writer(encoded,None,visible),expected)
        torch.testing.assert_close(writer(encoded,None,visible,order),expected)
        writer(encoded,None,visible).sum().backward()
        self.assertGreater(writer.out.weight.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(writer.out.weight.grad).all())
        with torch.no_grad():
            writer.out.weight.normal_(std=.1)
            self.assertFalse(torch.allclose(writer(encoded,None,visible),writer(encoded,None,visible,order)))

    def test_rt_rm_match_active_clip_reference_and_preserve_visibility(self):
        for name, change in runner.VARIANTS.items():
            with self.subTest(arm=name):
                cfg = dict(native(), img_size=32, local_frames=4, reconstruction_recent_frames=16,
                           mask_ratio=.65, research_mask='tube', **change)
                cfg['read_after_block'] = 1
                model = ForwardTemporalMAE(**cfg)
                fast = copy.deepcopy(model)
                samples = [dict(video=torch.rand(t,3,32,32),record_index=i) for i,t in enumerate((16,12))]
                expected = 0
                for sample in samples:
                    video = sample['video'][None]
                    masks = deterministic_masks(model,dict(video=video,record_index=torch.tensor([sample['record_index']])),42)
                    keep = (~masks).sum(2)[0]//model.token_grid[0]
                    base = model.visible_spatial_tokens(0,asymmetric=False)
                    self.assertEqual(int(keep.sum()),len(keep)*base)
                    if name == 'RM' and len(keep) == 4:
                        self.assertEqual(keep.tolist(),[base+base//2,base-base//2]*2)
                        common = deterministic_masks(model,dict(video=video,record_index=torch.tensor([sample['record_index']])),42,
                                                     asymmetric=False)
                        self.assertTrue(((~common).sum(2) == base*model.token_grid[0]).all())
                        verify_visibility(model,video,masks)
                    expected = expected+model(video,masks=masks)['loss']/len(samples)
                actual = active_clip_loss(fast,samples,42)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
                expected.backward()
                actual.backward()
                for (key,p), q in zip(model.named_parameters(),fast.parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,key)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=1e-4,msg=key)
                if name == 'RT':
                    self.assertGreater(fast.temporal_writer.out.weight.grad.norm().item(),0)

    def test_h_retains_common_parameters_and_initial_mapping(self):
        torch.manual_seed(72)
        a = TemporalEFReadout(24)
        rng = torch.get_rng_state()
        torch.manual_seed(72)
        b = TemporalEFReadout(24,temporal_relation=True)
        torch.testing.assert_close(torch.get_rng_state(),rng)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(b.state_dict()[name],value,rtol=0,atol=0)
        x = torch.randn(2,64,24)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

    def test_plans_reuse_stage_a_and_keep_ssl_and_downstream_budgets(self):
        ref = reference()
        ar = build_plan(ref,'/out','a','AR')
        args = SimpleNamespace(root=Path('/a2'),cache_root=Path('/cache'),tag='a2')
        with patch.object(runner,'source_path',side_effect=['/ar/epoch_0400.pt','/ar/epoch_0200.pt']):
            plans = runner.plans(args,ref,ar)
        for name in runner.VARIANTS:
            cfg = plans[name]
            ssl = stage_config(cfg,cfg['stages'][0])
            self.assertEqual(ssl['training'],stage_config(ar,ar['stages'][0])['training'])
            self.assertEqual(ssl['stage']['source'],ref['zero'])
            self.assertEqual(cfg['stages'][1]['source'],'@pretrain/epoch_0200')
            self.assertEqual([s['id'] for s in cfg['stages'][2:]], [s['id'] for s in ar['stages'][1:]])
            self.assertEqual(smoke_config(cfg)['stages'][1]['source'],'@pretrain/final')
        h = plans['H']
        old, new = [stage_config(h,s) for s in h['stages'][:2]]
        self.assertEqual(old['data'],new['data'])
        self.assertEqual(old['training'],new['training'])
        self.assertFalse(old['data']['augmentation']['enabled'])
        self.assertEqual(old['stage']['source'],new['stage']['source'])
