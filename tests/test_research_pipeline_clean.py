"""Clean trajectory: labels/exposure, real windows, snapshots and native readouts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from research_pipeline.artifacts import digest, layout, write_json
from research_pipeline.cache import _make_space
from research_pipeline.cli import configure, source_path
from research_pipeline.config import load, stage_config
from research_pipeline.engine import build_experiment, run_training
from research_pipeline.runtime import loader
from research_pipeline.ssl_data import build_ssl_manifest, load_ssl_manifest, SSLWindowDataset


def fixture(root):
    (root/'npy').mkdir()
    rows = []
    for patient, split, frames in [('train', 'TRAIN', 48), ('val', 'VAL', 32), ('test', 'TEST', 7), ('tiny', 'TEST', 3)]:
        video = np.random.default_rng(frames).integers(0, 256, (frames,112,112), dtype=np.uint8)
        np.save(root/'npy'/f'{patient}.npy', video)
        rows.append(dict(FileName=patient, Split=split, FPS=50, EF='DO_NOT_LOAD_THIS_LABEL'))
    pd.DataFrame(rows).to_csv(root/'FileList.csv', index=False)
    value = build_ssl_manifest(root, local_frames=4, recent_frames=8, max_prefix=8)
    write_json(root/'ssl.json', value)
    return value


class CleanBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_manifest_all_splits_no_labels_no_tracing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = fixture(root)
            self.assertFalse((root/'VolumeTracings.csv').exists())
            self.assertEqual(value['counts']['eligible_by_official_split'], dict(train=1,val=1,test=1))
            self.assertEqual(value['label_columns_loaded'], [])
            self.assertEqual(len(value['excluded']), 1)
            self.assertTrue(all('ef' not in case and 'traces' not in case for case in value['videos']))
            bad = copy.deepcopy(value)
            bad['videos'][0]['ef'] = 60
            bad['manifest_sha256'] = digest({k:v for k,v in bad.items() if k != 'manifest_sha256'})
            with self.assertRaisesRegex(ValueError, 'label'):
                load_ssl_manifest(bad)

    def test_missing_cache_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root/'npy'/'test.npy').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'cache'):
                build_ssl_manifest(root, local_frames=4, recent_frames=8, max_prefix=8)

    def test_windows_are_real_deterministic_variable_and_label_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = fixture(root)
            dataset = SSLWindowDataset(value, True)
            signatures = set()
            for epoch in range(12):
                dataset.set_epoch(epoch)
                for index in range(len(dataset)):
                    a,b = dataset[index],dataset[index]
                    torch.testing.assert_close(a['video'],b['video'],rtol=0,atol=0)
                    self.assertTrue(torch.equal(a['frame_indices'],b['frame_indices']))
                    self.assertTrue(bool((a['frame_indices'][1:]-a['frame_indices'][:-1] == 1).all()))
                    self.assertEqual(len(a['video']) % 4, 0)
                    self.assertNotIn('target',a)
                    self.assertNotIn('mask',a)
                    signatures.add((a['patient'],tuple(a['frame_indices'].tolist())))
                    raw = np.load(a['source_path'],allow_pickle=False)
                    expected = torch.from_numpy(raw[a['frame_indices'].numpy()].copy()).float()/255
                    torch.testing.assert_close(a['video'][:,0], expected, rtol=0,atol=0)
            self.assertGreater(len(signatures), 12)
            test_index = next(i for i,r in enumerate(dataset.records) if r['patient'] == 'test')
            self.assertEqual(len(dataset[test_index]['video']),4)
            diagnostic = SSLWindowDataset(value, False, diagnostic_cases=2)
            a = diagnostic[0]
            diagnostic.set_epoch(100)
            self.assertTrue(torch.equal(a['video'],diagnostic[0]['video']))
            stress = dataset.calibration_samples(2)
            self.assertEqual(len(stress[0]['video']),16)
            self.assertEqual(stress[0]['prefix_frames'],8)

    def test_clean_config_and_smoke_keep_architecture_and_initial_anchor(self):
        config = load('configs/pipeline/clip_recurrent_global.yaml')
        ssl = stage_config(config, config['stages'][0])
        self.assertEqual(ssl['training']['epochs'],1600)
        self.assertIsNone(ssl['training']['patience'])
        self.assertEqual(ssl['model']['config']['memory_mode'],'global')
        self.assertEqual(ssl['model']['config']['frame_readout'],'repeat')
        self.assertEqual(ssl['model']['config']['patch_size'],16)
        smoke = load('configs/pipeline/clip_recurrent_global.yaml',smoke=True)
        self.assertEqual(smoke['stages'][1]['source'],'@pretrain/epoch_0000')
        self.assertEqual(next(s for s in smoke['stages'] if s['id']=='ef_linear_0400')['source'],'@pretrain/final')
        self.assertEqual(smoke['model'],config['model'])

    def test_spawn_loader_matches_serial_and_epoch_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = fixture(root)
            cfg = load('configs/pipeline/synthetic.yaml')
            configure(cfg)
            from research_pipeline.echo import IdentifiedDataset
            dataset = IdentifiedDataset(SSLWindowDataset(manifest,True),'ssl')
            selection = dict(micro_batch=1,num_workers=1,gradient_checkpointing=False)
            batches = loader(dataset,cfg,selection)
            rows = list(batches)
            for group in rows:
                for sample in group:
                    torch.testing.assert_close(sample['video'],dataset[sample['record_index']]['video'],rtol=0,atol=0)
            dataset.set_epoch(1)
            batches.batch_sampler.epoch = 1
            for group in batches:
                for sample in group:
                    torch.testing.assert_close(sample['video'],dataset[sample['record_index']]['video'],rtol=0,atol=0)
            del batches

    def test_native_normpix_ssl_initial_and_final_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            cfg = load('configs/pipeline/synthetic.yaml',root=str(root/'output'))
            cfg['adapter'] = 'research_pipeline.echo:build'
            cfg['data'] = dict(ssl_manifest=str(root/'ssl.json'),manifest='DO_NOT_OPEN_LABELS.json',
                               local_frames=4,recent_frames=8,max_prefix=8,prefix=None,diagnostic_cases=2)
            cfg['model'] = dict(config=dict(img_size=112,patch_size=16,local_frames=4,clip_count=2,
                               embed_dim=24,depth=1,num_heads=4,decoder_embed_dim=24,decoder_depth=1,
                               decoder_num_heads=4,in_chans=3,tubelet_size=2,core_depth=1,
                               memory_mode='global',memory_compression='mean',memory_write_source='fused',
                               frame_readout='repeat',norm_pix_loss=True,reconstruction_recent_frames=16,
                               mask_ratio=.5))
            cfg['training'].update(epochs=1,effective_batch=2,warmup_epochs=0,patience=None,save_epochs=[0,1])
            cfg['runtime']['micro_batch'] = 1
            configure(cfg)
            current = stage_config(cfg,dict(id='pretrain',kind='mae'))
            exp = build_experiment(current,None,torch.device('cpu'))
            exp.model.train()
            sample = exp.calibration_samples(1)
            out = exp._mae(sample)
            self.assertEqual(out['pred'].shape[1],len(sample[0]['video'])//2*49)
            out['loss'].backward()
            self.assertIsNotNone(exp.model.memory.update_x.weight.grad)
            exp.model.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                mixed = exp._mae(sample)['loss']
            mixed.backward()
            self.assertTrue(torch.isfinite(mixed))
            initial = {key:value.clone() for key,value in exp.model.state_dict().items()}
            summary = run_training(current,None)
            self.assertTrue(summary['diagnostic_overlap'])
            self.assertIn('Seen-population',summary['scope'])
            self.assertEqual(summary['epoch'],1)
            for key in ('epoch_0000.pt','epoch_0001.pt','final.pt'):
                self.assertIn(key,summary['checkpoints'])
            zero = torch.load(layout(cfg,'pretrain')['ckpt']/'epoch_0000.pt',weights_only=False)
            for key in initial:
                torch.testing.assert_close(zero['model_state_dict'][key],initial[key],rtol=0,atol=0)
            source = source_path(cfg,dict(source='@pretrain/epoch_0000'))
            self.assertTrue(source.endswith('epoch_0000.pt'))
            final = source_path(cfg,dict(source='@pretrain/final'))
            from models.final_temporal_mae import load_final_model
            model,native,_ = load_final_model(final)
            self.assertEqual(model.memory_slots,1)
            self.assertEqual(native['model']['frame_readout'],'repeat')
            self.assertEqual(native['pretraining_data_protocol']['exposure'],'transductive_all_splits_unlabeled')
            preview = layout(cfg,'pretrain')['result']/'previews'/'epoch_0001'
            self.assertTrue((preview/'fixed_case_00.png').is_file())
            policy = json.loads((preview/'preview_policy.json').read_text())
            self.assertIn('visualization only',policy['policy'])
            import csv
            with (layout(cfg,'pretrain')['result']/'metrics.csv').open(encoding='utf-8-sig') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(int(row['visited_windows']),3)
            self.assertGreater(int(row['observed_clips']),0)
            self.assertTrue((layout(cfg,'pretrain')['result']/'initial_diagnostic.json').is_file())
            # SSL has completed: supervised readers must open only TRAIN/VAL, never the all-SSL file.
            from utils.final_temporal_data import build_manifest
            files = pd.read_csv(root/'FileList.csv')
            files['EF'] = 60
            files.to_csv(root/'FileList.csv',index=False)
            traces = []
            for patient in ('train','val'):
                for x1,y1,x2,y2 in ((0,0,112,112),(30,25,80,25),(25,55,85,55),(40,85,70,85)):
                    traces.append(dict(FileName=patient,Frame=12,X1=x1,Y1=y1,X2=x2,Y2=y2))
            pd.DataFrame(traces).to_csv(root/'VolumeTracings.csv',index=False)
            task_manifest = build_manifest(root,recent_frames=8,local_frames=4,max_prefix=8)
            write_json(root/'task.json',task_manifest)
            cfg['data'].update(manifest=str(root/'task.json'),ssl_manifest='NEVER_OPEN_THIS_FOR_SUPERVISION.json')
            (root/'npy'/'test.npy').unlink()
            for task in ('ef','seg'):
                for mode in ('frozen_linear','finetune'):
                    current = stage_config(cfg,dict(id=task+'_'+mode,kind=task,mode=mode,
                                                    head_dim=24,head_depth=1,head_heads=4))
                    task_exp = build_experiment(current,final,torch.device('cpu'))
                    self.assertEqual(task_exp.metadata['train_ids'],['train'])
                    self.assertEqual(task_exp.metadata['val_ids'],['val'])
                    self.assertFalse(task_exp.metadata['test_access'])
                    self.assertEqual(task_exp.metadata['ssl_data_exposure'],'transductive_all_splits_unlabeled')
                    run_training(current,final)
            cfg['smoke'] = True
            from research_pipeline.cli import legacy_audit
            for kind in ('streaming','mechanisms'):
                current = stage_config(cfg,dict(id=kind,kind=kind,
                                     **({'audit':dict(max_frames=24,max_cases=1)} if kind=='streaming' else {})))
                legacy_audit(current,final)

    def test_eviction_removes_only_finished_feature_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            finished,partial,current = (root/'cache'/'frozen'/name for name in ('finished','partial','current'))
            for directory in (finished,partial,current):
                directory.mkdir(parents=True)
                (directory/'features.bin').write_bytes(b'0'*100)
            for split in ('train','val'):
                (finished/split).mkdir()
                write_json(finished/split/'DONE.json',dict(ok=True))
            (root/'ckpt').mkdir()
            checkpoint = root/'ckpt'/'last.pt'
            checkpoint.write_bytes(b'not disposable')
            destination = current/'train'
            destination.mkdir()
            _make_space(current,destination,root/'cache',50,250,0,None)
            self.assertFalse(finished.exists())
            self.assertTrue(partial.is_dir())
            self.assertTrue(current.is_dir())
            self.assertTrue(checkpoint.is_file())


if __name__ == '__main__':
    unittest.main()
