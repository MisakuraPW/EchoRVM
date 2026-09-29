import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from models.temporal_mae import TemporalMAE, temporal_mask
from tools.run_stage3_mechanisms import configuration
from tools.run_stage2_questions import reuse_training_contract
from tools.run_stage1_lengths import stage_config


def config():
    return dict(name='temporal_mae',img_size=16,patch_size=4,local_frames=4,clip_count=2,
        tubelet_size=2,in_chans=3,embed_dim=24,depth=1,num_heads=3,decoder_embed_dim=24,
        decoder_depth=1,decoder_num_heads=3,mask_ratio=.5,memory_mode='spatial',memory_grid=2,
        core_depth=1,norm_pix_loss=False,position_embedding='flat_sinusoid')


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_legacy_checkpoint_and_uniform_compression_equivalence(self):
        base=TemporalMAE(**config()).eval()
        candidate=TemporalMAE(**dict(config(),memory_compression='temporal_attention')).eval()
        missing,extra=candidate.load_state_dict(base.state_dict(),strict=False)
        self.assertTrue(all(k.startswith('compression_score.') for k in missing))
        self.assertFalse(extra)
        video=torch.rand(2,8,3,16,16)
        with torch.no_grad():
            torch.testing.assert_close(base.forward_features(video),candidate.forward_features(video),rtol=1e-5,atol=1e-6)
        for masked in (False,True):
            mask=temporal_mask(2,base.token_grid,.5,'tube','cpu') if masked else None
            valid=torch.ones(2,base.patch_embed.num_patches,dtype=torch.bool)
            valid[1,:16]=False
            dense=torch.randn(2,base.patch_embed.num_patches,24)
            tokens=dense[~mask].reshape(2,-1,24) if masked else dense
            torch.testing.assert_close(base._pool(tokens,mask,valid),candidate._pool(tokens,mask,valid),rtol=1e-5,atol=1e-6)

    def test_both_candidates_backpropagate_and_roundtrip(self):
        for changes in ({'memory_compression':'temporal_attention'},{'memory_write_source':'local'}):
            model=TemporalMAE(**dict(config(),**changes)).train()
            out=model(torch.rand(2,8,3,16,16))
            self.assertTrue(torch.isfinite(out['loss']))
            out['loss'].backward()
            if 'memory_compression' in changes:
                gradient=model.compression_score[-1].weight.grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()),0.)
            else:
                self.assertGreater(float(model.memory.update_x.weight.grad.abs().sum()),0.)
            restored=TemporalMAE(**dict(config(),**changes))
            restored.load_state_dict(model.state_dict(),strict=True)

    def test_local_write_pool_is_not_fused_feedback(self):
        model=TemporalMAE(**dict(config(),memory_write_source='local')).eval()
        video=torch.rand(1,8,3,16,16)
        tokens=[]
        h=model.memory.register_forward_pre_hook(lambda m,args:tokens.append(args[0]))
        with torch.no_grad():
            result=model.diagnostic_features(video)
            for i,raw in enumerate(result['local_features'].split(2,1)):
                valid=torch.ones(1,model.patch_embed.num_patches,dtype=torch.bool)
                expected=model._pool(raw.flatten(1,2),None,valid)
                torch.testing.assert_close(tokens[i],expected)
        h.remove()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_candidate_amp_backward(self):
        for changes in ({'memory_compression':'temporal_attention'},{'memory_write_source':'local'}):
            model=TemporalMAE(**dict(config(),**changes)).cuda().train()
            with torch.autocast('cuda'):
                loss=model(torch.rand(2,8,3,16,16,device='cuda'))['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_stage1_reuse_contract_unchanged_and_candidates_one_factor(self):
        args=SimpleNamespace(seed=42,data_root='/data',num_workers=8,init_checkpoint='init.pt',epochs=100,smoke=False)
        old=stage_config(args,16,Path('/old'))
        base=configuration(args,'baseline',Path('/new'))
        self.assertEqual(reuse_training_contract(old),reuse_training_contract(base))
        for name,key,value in [('temporal_pool','memory_compression','temporal_attention'),
                               ('local_write','memory_write_source','local'),('none','memory_mode','none')]:
            candidate=configuration(args,name,Path('/new'))
            expected=copy.deepcopy(base['model'])
            expected[key]=value
            self.assertEqual(candidate['model'],expected)
            self.assertEqual(candidate['optimizer'],base['optimizer'])
            self.assertEqual(candidate['train'],base['train'])


if __name__=='__main__':
    unittest.main()
