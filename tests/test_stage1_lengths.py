import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.temporal_mae import TemporalMAE, temporal_mask
from tools.run_stage1_lengths import stage_config, probe_args, evaluate, summarize
from tools.diagnose_stage2_interfaces import SegViews


def arguments(root, smoke=False):
    return SimpleNamespace(seed=42,data_root=str(root),num_workers=0,init_checkpoint='unused.pt',
        epochs=1 if smoke else 100,smoke=smoke,eval_batch_size=1,
        output_root=str(root/'outputs'),run_tag='test')


class Stage1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_length_and_partition_change(self):
        args=arguments(Path('data'))
        normalized=[]
        for length in (8,16,32):
            cfg=stage_config(args,length,Path('ckpt')/str(length))
            self.assertEqual(cfg['model']['frames'],64)
            self.assertEqual(cfg['model']['memory_mode'],'spatial')
            self.assertEqual(cfg['model']['local_frames']*cfg['model']['clip_count'],64)
            self.assertEqual(cfg['checkpoint']['save_epochs'],[100])
            self.assertFalse(cfg['early_stopping']['enabled'])
            self.assertEqual(cfg['train']['batch_size']*cfg['train']['grad_accum_steps'],32)
            self.assertFalse(cfg['augment']['enabled'])
            cfg=copy.deepcopy(cfg)
            cfg['experiment'].pop('name')
            cfg['model'].pop('local_frames')
            cfg['model'].pop('clip_count')
            cfg['checkpoint'].pop('dir')
            normalized.append(cfg)
        self.assertEqual(normalized[0],normalized[1])
        self.assertEqual(normalized[1],normalized[2])
        self.assertEqual(probe_args(args).seg_target_index,63)
        self.assertFalse(probe_args(args).offset_probe)

    def test_mask_budget_and_backward_all_lengths(self):
        order=torch.rand(2,16).argsort(-1)
        masks=[]
        for length in (8,16,32):
            cfg=dict(img_size=16,patch_size=4,local_frames=length,clip_count=64//length,
                frames=64,in_chans=3,tubelet_size=2,embed_dim=24,depth=1,num_heads=3,
                decoder_embed_dim=24,decoder_depth=1,decoder_num_heads=3,memory_mode='spatial',
                memory_grid=2,mask_ratio=.75,separate_qv_bias=True,position_embedding='flat_sinusoid')
            mask=torch.stack([temporal_mask(2,(length//2,4,4),.75,'tube','cpu',i,order)
                              for i in range(64//length)],1)
            masks.append(mask.reshape(2,32,16))
            model=TemporalMAE(**cfg)
            out=model(torch.rand(1,64,3,16,16),torch.ones(1,64,dtype=torch.bool))
            self.assertTrue(torch.isfinite(out['loss']))
            out['loss'].backward()
            self.assertTrue(any(p.grad is not None for p in model.memory.parameters()))
        torch.testing.assert_close(masks[0],masks[1])
        torch.testing.assert_close(masks[1],masks[2])

    def test_endpoint_real_file_contract_resume_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'npy').mkdir()
            files=['FileName,EF,Split']
            traces=['FileName,Frame,X1,Y1,X2,Y2']
            for i in range(12):
                name=f'p{i:02d}'
                files.append(f'{name},{40+2*i},{"TRAIN" if i<8 else "VAL"}')
                np.save(root/'npy'/f'{name}.npy',np.broadcast_to((50+np.arange(80,dtype=np.uint8))[:,None,None],(80,112,112)).copy())
                for frame in (2,70):
                    for y in (20,30,50,70,90):
                        traces.append(f'{name}.avi,{frame},30,{y},80,{y}')
            (root/'FileList.csv').write_text('\n'.join(files))
            (root/'VolumeTracings.csv').write_text('\n'.join(traces))
            args=arguments(root,True)
            ds=SegViews(root,'val',probe_args(args),3,2)
            for i in range(len(ds)):
                item=ds[i]
                self.assertEqual(item['target_index'],63)
                self.assertAlmostEqual(float(item['video'][63].mean()),(50+item['source_frame'])/255,places=6)
            results=Path(args.output_root)/args.run_tag/'result'
            for length in (8,16):
                cfg=dict(name='temporal_mae',img_size=112,patch_size=8,local_frames=length,
                    clip_count=64//length,frames=64,in_chans=3,tubelet_size=2,embed_dim=12,
                    depth=1,num_heads=3,decoder_embed_dim=12,decoder_depth=1,decoder_num_heads=3,
                    memory_mode='spatial',memory_grid=2,separate_qv_bias=True,position_embedding='flat_sinusoid')
                checkpoint=root/f'{length}.pt'
                torch.save(dict(model_state_dict=TemporalMAE(**cfg).state_dict(),epoch=1,
                    config=dict(model=cfg,data=dict(input_protocol='gray_repeat3'))),checkpoint)
                out=results/f'spatial_l{length}'/'endpoint'
                evaluate(checkpoint,out,args)
                values=json.loads((out/'metrics.json').read_text())
                self.assertEqual(values['validity']['future_frames'],0)
                self.assertEqual(values['seg']['steps'],2)
                before=(out/'metrics.json').read_bytes()
                evaluate(checkpoint,out,args)
                self.assertEqual(before,(out/'metrics.json').read_bytes())
            (results/'stage_times.csv').write_text('stage,seconds\nspatial_l8/pretrain,10\nspatial_l16/pretrain,11\n')
            summarize(results,[8,16],42)
            with (results/'paired_differences.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),2)
            self.assertTrue((results/'analysis.zip').exists())
            self.assertFalse(list((Path(args.output_root)/args.run_tag/'cache').rglob('*.npz')))


if __name__=='__main__':
    unittest.main()
