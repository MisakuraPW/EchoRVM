from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from research_pipeline.cache import prepare_cache
from research_pipeline.spatial_ef import QueryEF, spatial_mean
from tools.run_spatial_ef_readout import conclusion, matched_observations, paired, test_pair
from utils.final_temporal_execution import encode_window


class SpatialReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_parameters_four_queries_not_dense_self_attention(self):
        torch.manual_seed(25)
        c1=QueryEF(False)
        c2=QueryEF(True)
        c2.load_state_dict(c1.state_dict())
        self.assertEqual(c1.queries.shape,(1,4,192))
        self.assertEqual(len(c2.interaction),2)
        keys=[]
        handle=c2.interaction.register_forward_pre_hook(lambda m,args:keys.append(args[0].shape))
        dense=torch.randn(1,32,196,384)
        torch.testing.assert_close(c1(spatial_mean(dense)),c2(dense))
        c2(dense).sum().backward()
        self.assertGreater(c2.regress[-1].weight.grad.abs().sum().item(),0)
        self.assertEqual(keys,[(1,4,192)]*2)
        handle.remove()

    def test_old_exit_consistent_and_state_is_independent_between_windows(self):
        def stream_clip(x,state):
            x=x[:,::2].repeat_interleave(2,dim=1)
            y=x if state is None else x+state
            return dict(final_state=y[:,-1:],frame_outputs=y,frame_base_outputs=y,
                        local_base_outputs=y,local_frame_outputs=y)
        backbone=SimpleNamespace(in_chans=1,local_frames=16,stream_clip=stream_clip)
        x=torch.randn(1,64,1,2,3)
        # Use [B,T,S,D] outputs with distinguishable spatial tokens.
        def maps(clip,state):
            return stream_clip(clip.squeeze(2),state)
        backbone.stream_clip=maps
        old=encode_window(backbone,x)['sequences']['final']
        keep=encode_window(backbone,x,keep_final_grid=True)
        full=keep['final_grid']
        torch.testing.assert_close(full[:,::2],full[:,1::2])
        torch.testing.assert_close(spatial_mean(full[:,::2]),old)
        self.assertFalse(torch.equal(full[:,:,0],full[:,:,1]))
        torch.testing.assert_close(encode_window(backbone,x)['sequences']['final'],old)

    def test_dense_and_unquantized_mean_views_use_one_cache(self):
        rows=[dict(target=torch.tensor(60.+i),patient=str(i)) for i in range(3)]
        def fixture():
            return SimpleNamespace(cacheable=True,train=rows,val=rows,model=torch.nn.Identity(),
                                   features=lambda s:torch.stack([r['target'].expand(2,2)+.01 for r in s]))
        with tempfile.TemporaryDirectory() as d:
            first=fixture()
            destination=Path(d)/'cache'
            layouts=dict(feature_dtype='float16',derived_views={'mean':lambda x:x.mean(1)})
            prepare_cache(first,destination,{},10**8,0,view='mean',**layouts)
            second=fixture()
            def fail(_):
                self.fail('Reusing dense view may not extract again')
            second.features=fail
            prepare_cache(second,destination,{},10**8,0,**layouts)
            torch.testing.assert_close(first.train[0]['features'],torch.full((2,),60.01))
            self.assertEqual(second.train[0]['features'].dtype,torch.float16)
            self.assertEqual(first.train[2]['target'],second.train[2]['target'])
            first.train.close(); first.val.close(); second.train.close(); second.val.close()

    def test_pair_requires_window_label_match_and_refuses_smoke_test(self):
        row=dict(patient='p',sample_id='val:0',target=55.,recent_start=1,frames=64,prefix_frames=0,absolute_error=2.)
        matched_observations([row],[row])
        with self.assertRaises(ValueError):
            matched_observations([row],[dict(row,recent_start=2)])
        with self.assertRaises(ValueError):
            paired([row],[dict(row,target=54)])
        with self.assertRaisesRegex(ValueError,'Smoke'):
            test_pair({'smoke':True},None,None,None)
        delta=lambda value:dict(candidate_minus_reference=value)
        val=dict(paired=dict(C2_minus_C1=delta(-.2),C1_minus_C0=delta(-.1),C2_minus_C0=delta(-.3)))
        self.assertTrue(conclusion(val,dict(paired=delta(-.1)))['val_supports_considering_ft'])
