"""Native temporal model + file-backed official trace path, on CPU."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from research_pipeline.artifacts import layout, write_json
from research_pipeline.config import load, stage_config
from research_pipeline.engine import build_experiment, run_training
from research_pipeline.cli import configure, legacy_audit
from utils.final_temporal_data import build_manifest


class ReferenceAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_mae_to_frozen_seg_and_ef_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            rows, traces = [], []
            for patient, split in (('a', 'TRAIN'), ('b', 'VAL')):
                video = np.random.default_rng(1).integers(0, 256, size=(24, 112, 112), dtype=np.uint8)
                np.save(root/'npy'/f'{patient}.npy', video)
                rows.append(dict(FileName=patient, Split=split, FPS=50, EF=60, FrameHeight=112, FrameWidth=112))
                for x1, y1, x2, y2 in ((0, 0, 112, 112), (30, 25, 80, 25), (25, 55, 85, 55), (40, 85, 70, 85)):
                    traces.append(dict(FileName=patient, Frame=12, X1=x1, Y1=y1, X2=x2, Y2=y2))
            pd.DataFrame(rows).to_csv(root/'FileList.csv', index=False)
            pd.DataFrame(traces).to_csv(root/'VolumeTracings.csv', index=False)
            manifest = build_manifest(root, recent_frames=8, local_frames=4, max_prefix=8)
            write_json(root/'manifest.json', manifest)
            cfg = load('configs/pipeline/synthetic.yaml', tag='native', root=str(root/'out'))
            cfg['adapter'] = 'research_pipeline.echo:build'
            cfg['data'] = dict(manifest=str(root/'manifest.json'), recent_frames=8, local_frames=4, max_prefix=8,
                               prefix=0, augmentation=None)
            cfg['model'] = dict(config=dict(img_size=112, patch_size=16, local_frames=4, clip_count=2,
                              embed_dim=24, depth=1, num_heads=4, decoder_embed_dim=24, decoder_depth=1,
                              decoder_num_heads=4, tubelet_size=2, in_chans=3, memory_grid=2, core_depth=1,
                              memory_mode='spatial', frame_readout='learned', norm_pix_loss=False,
                              reconstruction_recent_frames=8, mask_ratio=.5))
            cfg['training'].update(epochs=1, effective_batch=1, warmup_epochs=0, patience=None)
            cfg['runtime']['micro_batch'] = 1
            configure(cfg)
            pretrain = stage_config(cfg, dict(id='pretrain', kind='mae'))
            result = run_training(pretrain, None)
            self.assertEqual(result['epoch'], 1)
            source = str(layout(cfg, 'pretrain')['ckpt']/'final.pt')
            self.assertTrue((layout(cfg, 'pretrain')['result']/'previews'/'fixed_case_00.png').exists())
            for task, mode in (('ef', 'frozen_linear'), ('seg', 'frozen_linear'),
                               ('ef', 'frozen_lite'), ('seg', 'frozen_lite'),
                               ('ef', 'finetune'), ('seg', 'finetune')):
                name = task+'_'+mode
                stage = dict(id=name, kind=task, mode=mode, head_dim=24, head_heads=4, head_depth=1)
                current = stage_config(cfg, stage)
                before = build_experiment(current, source, torch.device('cpu'))
                batch = [before.val[0]]
                before.model.eval()
                with torch.no_grad():
                    output = before.model(batch)
                    self.assertEqual(len(output), 1)
                summary = run_training(current, source)
                self.assertIn('mae_pp' if task == 'ef' else 'dice_mean', summary['metrics'])
                saved = torch.load(layout(cfg, name)['ckpt']/'final.pt', weights_only=False)
                initial = torch.load(source, weights_only=False)['model_state_dict']
                if mode.startswith('frozen'):
                    for key in initial:
                        torch.testing.assert_close(initial[key], saved['model']['backbone.'+key], rtol=0, atol=0)
                else:
                    self.assertTrue(any(not torch.equal(initial[key], saved['model']['backbone.'+key]) for key in initial))
            directories = list((layout(cfg)['cache']/'frozen').iterdir())
            self.assertEqual(len(directories), 2, 'linear and lite heads must share one feature cache per task')
            cfg['smoke'] = True
            for kind in ('streaming', 'mechanisms'):
                current = stage_config(cfg, dict(id=kind, kind=kind,
                                                 **({'audit': dict(max_frames=24, max_cases=1)} if kind == 'streaming' else {})))
                legacy_audit(current, source)
                self.assertTrue((layout(cfg, kind)['result']/'DONE.json').exists())


if __name__ == '__main__':
    unittest.main()
