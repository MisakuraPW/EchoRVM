import copy
import unittest

import torch

from models.ef_readout import TemporalEFReadout
from models.forward_temporal_mae import ForwardTemporalMAE
from research_pipeline.variable_ef import VariableEF, stop_table, update_fifo
from tools.run_patch8_stage_b import fit_budget
from utils.final_temporal_execution import encode_window


def tiny():
    return ForwardTemporalMAE(img_size=16,patch_size=8,local_frames=16,clip_count=4,
        in_chans=3,embed_dim=24,depth=2,num_heads=4,tubelet_size=2,decoder_embed_dim=24,
        decoder_depth=1,decoder_num_heads=4,memory_mode='global',frame_readout='repeat',
        position_embedding='flat_sinusoid',memory_read_location='tokens',
        forward_variant='mid_read',read_after_block=1)


class VariableEFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_l16_reproduces_old_descriptors_and_state(self):
        backbone=tiny().eval()
        head=TemporalEFReadout(24)
        model=VariableEF(backbone,head,'uniform').eval()
        video=torch.rand(2,64,3,16,16)
        draw=torch.rand(2,4,8)
        ref=encode_window(backbone,video)
        seen=[]
        handle=backbone.memory.register_forward_hook(lambda m,a,out:seen.append(out[1]))
        pred,_=model(video,draw,policy='fixed')
        torch.testing.assert_close(pred,head(ref['sequences']['final']))
        self.assertEqual(len(seen),4)
        torch.testing.assert_close(seen[-1],ref['final_state'])
        handle.remove()

    def test_variable_partitions_only_encode_original_frame_indices(self):
        model=VariableEF(tiny(),TemporalEFReadout(24),'uniform')
        video=torch.arange(2*64).float().reshape(2,64,1,1,1).expand(-1,-1,3,16,16)
        recorded=[]
        def fake(clip,previous):
            start=clip[:,:,0,0,0]
            recorded.extend(start.tolist())
            state=torch.zeros(len(clip),1,24) if previous is None else previous
            return start[...,None].expand(-1,-1,24),state+1
        model.encode_clip=fake
        rows=[]
        hook=model.head.register_forward_pre_hook(lambda m,args:rows.append(args[0].detach()))
        draw=torch.rand(2,4,8)
        pred,extra=model(video,draw,forced=torch.tensor([[8,20,12,24],[32,8,12,12]]))
        self.assertEqual(extra['lengths'].tolist(),[[8,20,12,24],[32,8,12,12]])
        self.assertEqual(sorted(x for row in recorded for x in row),list(range(128)))
        torch.testing.assert_close(rows[0][...,0],video[:,:,0,0,0])
        torch.testing.assert_close(update_fifo(rows[0],rows[0]),rows[0])
        hook.remove()

    def test_feasible_policy_and_valid_gradient_with_causal_arrivals(self):
        model=VariableEF(tiny(),TemporalEFReadout(24),'observed')
        # Initial zero residual at the policy logit gives the matched uniform prior.
        video=torch.rand(3,64,3,16,16)
        draw=torch.rand(3,4,8)
        y,info=model(video,draw)
        _,prior=model(video,draw,policy='uniform')
        torch.testing.assert_close(info['lengths'],prior['lengths'])
        self.assertTrue(bool((info['lengths'].sum(1)==64).all()))
        self.assertTrue(bool(((info['lengths']>=8)&(info['lengths']<=32)).all()))
        self.assertTrue(bool((stop_table()[0,8,:8]==0).all()))
        logp,_,_,free,_,_=info['stats'][0]
        vision,motion=model.observer.cell_features(video)
        last,stats=model.choose(torch.full((3,),12),0,draw[:,0],None,vision,motion,'observed')
        self.assertEqual(last.tolist(),[4,4,4])
        self.assertFalse(bool(stats[3].any()))
        grad=torch.autograd.grad((stats[0]*stats[3]).sum(),model.observer.actor.weight)[0]
        self.assertFalse(bool(grad.any()))
        obj=-(logp*free).sum()
        g=torch.autograd.grad(obj,model.observer.actor.weight)[0]
        self.assertTrue(bool(torch.isfinite(g).all()))
        self.assertGreater(float(g.norm()),0.)
        with torch.no_grad():
            model.observer.actor.weight.normal_()
            vision,motion=model.observer.cell_features(video)
            length,_=model.choose(torch.zeros(3,dtype=torch.long),3,draw[:,0],None,vision,motion,'observed')
            revised=video.clone()
            for i,n in enumerate(length.tolist()):
                revised[i,n*4:]=1-revised[i,n*4:]
            v,m=model.observer.cell_features(revised)
            after,_=model.choose(torch.zeros(3,dtype=torch.long),3,draw[:,0],None,v,m,'observed')
            torch.testing.assert_close(length,after)

    def test_guard_balances_both_comparison_budgets(self):
        self.assertEqual(fit_budget({'uniform':{'samples_per_second':40},'observed':{'samples_per_second':40}},
                                     7427,1279,8)['b2_epochs'],40)
        short=fit_budget({'uniform':{'samples_per_second':28},'observed':{'samples_per_second':27}},7427,1279,8)
        self.assertGreaterEqual(short['b2_epochs'],10)
        self.assertLess(short['b2_epochs'],40)
        with self.assertRaises(RuntimeError):
            fit_budget({'uniform':{'samples_per_second':5},'observed':{'samples_per_second':5}},7427,1279,8)

    def test_model_and_optimizer_state_resume_the_same_next_update(self):
        torch.manual_seed(18)
        model=VariableEF(tiny(),TemporalEFReadout(24),'observed')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        video,draw=torch.rand(2,64,3,16,16),torch.rand(2,4,8)
        def step(net,optimizer):
            optimizer.zero_grad()
            value,other=net(video,draw)
            logp,_,_,free,_,_=other['stats'][0]
            (value.square().mean()-(logp*free).mean()).backward()
            optimizer.step()
        step(model,opt)
        payload=copy.deepcopy((model.state_dict(),opt.state_dict()))
        clone=VariableEF(tiny(),TemporalEFReadout(24),'observed')
        new_opt=torch.optim.AdamW(clone.parameters(),lr=1e-4)
        clone.load_state_dict(payload[0],strict=True)
        new_opt.load_state_dict(payload[1])
        step(model,opt)
        step(clone,new_opt)
        for p,q in zip(model.parameters(),clone.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
