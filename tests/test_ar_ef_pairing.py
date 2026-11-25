import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from models.ef_readout import TemporalEFReadout
from models.forward_temporal_mae import ForwardTemporalMAE
from research_pipeline.artifacts import layout, mark_done, write_json
from research_pipeline.cli import configure
from research_pipeline.config import load, stage_config
from research_pipeline.ef_pairing import FrozenReadout, clip_views, describe
from research_pipeline.engine import build_experiment, run_training
from tools.run_ar_ef_pairing import compare_heads, dataset, e0, endpoint, endpoint_group, reduce_results, uniform_starts
from utils.final_temporal_data import build_manifest


def test_starts_population_and_paired_case_weight():
    assert uniform_starts(64) == [0]
    assert uniform_starts(65) == [0, 1]
    assert uniform_starts(128) == [0, 16, 32, 48, 64]
    rasterize = lambda row: np.ones(row['area'])
    case = {'traces':[dict(frame=9,area=100),dict(frame=19,area=60)]}
    assert [endpoint_group(case,i,rasterize) for i in (0,5,10)] == ['cross','same','uncovered']
    assert endpoint_group({'traces':list(reversed([dict(frame=9,area=60),dict(frame=19,area=100)]))},
                          0,rasterize)=='uncovered'
    a = [dict(patient='a',target=40.,prediction=46.),dict(patient='b',target=50.,prediction=50.)]
    b = a[:1]*5+[dict(a[1],prediction=52.)]
    r,_ = reduce_results(dict(a=a,b=b),dict(b={'b'}),[('a','b')])
    assert r['metrics']['b']['mae_pp']==4.
    assert r['pairs']['b_minus_a']['candidate_minus_reference']==1.


def test_independent_tracks_do_not_carry_state_and_have_identical_frames():
    state_inputs = []
    def stream_clip(clip,state):
        state_inputs.append(state)
        y = clip[:,::2,0].repeat_interleave(2,dim=1)
        if state is not None:
            y = y + state
        return dict(final_state=y[:,-1:],frame_outputs=y,frame_base_outputs=y,
                    local_base_outputs=y,local_frame_outputs=y)
    b = SimpleNamespace(local_frames=16,in_chans=1,frame_readout='repeat',tubelet_size=2,stream_clip=stream_clip)
    x = torch.arange(64.).reshape(1,64,1,1,1)
    for view in ('independent','stride4'):
        assert torch.cat(clip_views(x,view),dim=1).flatten().sort().values.tolist()==list(range(64))
        state_inputs.clear()
        describe(b,x,view)
        assert len(state_inputs)==4 and all(v is None for v in state_inputs)
    state_inputs.clear()
    describe(b,x,'recurrent')
    assert state_inputs[0] is None and all(v is not None for v in state_inputs[1:])


def test_readout_l1_target_and_pair_fallback():
    torch.set_num_threads(1)
    c0 = TemporalEFReadout(8)
    seq = torch.randn(3,32,8)
    fps = torch.tensor([50.,60.,10000.])
    independent = FrozenReadout(c0,'I',8)
    expected = torch.stack([c0(v.repeat_interleave(2,dim=1)) for v in seq.split(8,dim=1)],dim=1).mean(1)
    torch.testing.assert_close(independent(seq,fps),expected)
    p = FrozenReadout(c0,'P',8)
    for mode in (True,False):
        torch.testing.assert_close(p(seq,fps,perturb=mode),c0(seq.repeat_interleave(2,dim=1)))
    with torch.no_grad():
        p.pair.correction[-1].weight.normal_()
    v = p(seq,fps)
    assert torch.isfinite(v).all()
    torch.testing.assert_close(v[2],c0(seq.repeat_interleave(2,dim=1))[2])
    assert not torch.allclose(v[:2],p(seq,fps,perturb=True)[:2])
    v.abs().mean().backward()
    assert p.pair.phase.weight.grad.abs().sum()>0
    with torch.no_grad():
        p.pair.null_score.fill_(30)
        p.eval()
        torch.testing.assert_close(p(seq,fps),c0(seq.repeat_interleave(2,dim=1)))


def test_real_adapter_fit_cache_val_lock_and_heldout_eval():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/'npy').mkdir()
        for name in ('a','b','c'):
            np.save(root/'npy'/f'{name}.npy',np.random.default_rng(ord(name)).integers(
                0,256,(66,112,112),dtype=np.uint8))
        pd.DataFrame([dict(FileName=p,Split=s,FPS=50,EF=ef,FrameHeight=112,FrameWidth=112)
                      for p,s,ef in [('a','TRAIN',45.),('b','VAL',50.),('c','TEST',55.)]]).to_csv(
                          root/'FileList.csv',index=False)
        pd.DataFrame(columns=['FileName','Frame','X1','Y1','X2','Y2']).to_csv(root/'VolumeTracings.csv',index=False)
        manifest=build_manifest(root,max_prefix=0)
        write_json(root/'manifest.json',manifest)
        native=dict(img_size=112,patch_size=8,local_frames=16,clip_count=4,embed_dim=8,depth=7,
            num_heads=2,decoder_embed_dim=8,decoder_depth=1,decoder_num_heads=2,tubelet_size=2,
            in_chans=3,core_depth=1,memory_grid=1,memory_mode='global',memory_read_location='tokens',
            memory_write_source='fused',memory_compression='mean',frame_readout='repeat',
            reconstruction_recent_frames=64,mask_ratio=.9,forward_variant='mid_read',read_after_block=6)
        net=ForwardTemporalMAE(**native)
        source=str(root/'epoch_0400.pt')
        torch.save(dict(epoch=400,model_state_dict=net.state_dict(),config=dict(model=native)),source)
        cfg=load('configs/pipeline/synthetic.yaml',tag='ef_pair',root=str(root/'out'))
        cfg.update(adapter='research_pipeline.ef_pairing:build',model={'overrides':{}},
                   data=dict(manifest=str(root/'manifest.json'),recent_frames=64,local_frames=16,max_prefix=0,prefix=0,seed=42))
        cfg['training'].update(epochs=1,effective_batch=1,warmup_epochs=0)
        cfg['runtime'].update(micro_batch=1,device='cpu',precision='fp32',min_free_gb=0)
        cfg['stages']=[dict(id=arm,kind='ef',mode='frozen_lite',source=source,calibrated_ef=True,
                            ef_pairing_arm=arm) for arm in ('I','J','R','P')]
        configure(cfg)
        settings=stage_config(cfg,dict(id='ef_l1_ft',kind='ef',mode='finetune'))
        settings['adapter']='research_pipeline.stage_a:build'
        native_ft=build_experiment(settings,source,torch.device('cpu'))
        mock_ft=root/'best_epoch_0021.pt'
        torch.save(dict(epoch=21,source=source,model=native_ft.model.state_dict()),mock_ft)
        origin=root/'ft'
        write_json(origin/'runtime.json',dict(selected=dict(micro_batch=1,num_workers=0)))
        write_json(origin/'observations.json',[dict(patient='b',target=50.)])
        ft=(origin,dict(selected_epoch=21),mock_ft,dict(config=settings,source=source))
        e0_val=e0(dataset(manifest,'val',42),ft,cfg,root/'e0_val')
        assert e0_val['cost']['single']['windows']==1 and e0_val['cost']['five']['windows']==3
        for stage in cfg['stages']:
            run_training(stage_config(cfg,stage),source)
        cache=list((layout(cfg)['cache']/'frozen').iterdir())
        assert len(cache)==2
        val=compare_heads(cfg,source,dataset(manifest,'val',42),root/'val')
        assert val['metrics']['P_perturbed']['patients']==1
        write_json(root/'TEST_LOCK.json',val['weights'])
        test=build_manifest(root,max_prefix=0,splits=('test',))
        held=compare_heads(cfg,source,dataset(test,'test',42),root/'test',test=True)
        assert set(held['metrics'])=={'I','J','R','P'}
        assert all(row['patients']==1 for row in held['metrics'].values())
        directory,summary,path=endpoint(cfg,'I',source)
        write_json(directory/'summary.json',dict(summary,smoke=True))
        mark_done(directory,['summary.json','protocol.json'])
        assert endpoint(dict(cfg,smoke=True),'I',source)[2]==path
        with unittest.TestCase().assertRaisesRegex(ValueError,'Smoke cannot'):
            compare_heads(dict(cfg,smoke=True),source,None,root/'invalid_test',test=True)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name,test in globals().items()
                              if name.startswith('test_'))
