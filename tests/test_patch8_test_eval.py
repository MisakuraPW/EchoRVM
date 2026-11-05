from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_final_temporal_data import fixture, POINTS
from tools.evaluate_patch8_test import locked_checkpoint
from utils.final_temporal_data import build_manifest, WindowDataset
from research_pipeline.artifacts import file_hash, mark_done, write_json


class TestEndpointTests(unittest.TestCase):
    def test_explicit_test_only_manifest_and_training_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            np.save(root / 'npy/sealed.npy', np.zeros((200, 112, 112), dtype=np.uint8))
            traces = pd.read_csv(root / 'VolumeTracings.csv')
            rows = [dict(FileName='sealed.avi', Frame=100, X1=x1, Y1=y1, X2=x2, Y2=y2)
                    for x1, y1, x2, y2 in POINTS]
            pd.concat([traces, pd.DataFrame(rows)]).to_csv(root / 'VolumeTracings.csv', index=False)
            manifest = build_manifest(root, max_prefix=0, splits=('test',))
            self.assertFalse(manifest['train'] or manifest['val'])
            self.assertEqual([r['patient'] for r in manifest['test']], ['sealed'])
            for options in ({}, {'allow_test': True, 'training': True},
                            {'allow_test': True, 'aug_cfg': {'enabled': False}}):
                with self.assertRaises(ValueError):
                    WindowDataset(manifest, 'test', task='seg', max_prefix=0, **options)
            ds = WindowDataset(manifest, 'test', task='seg', allow_test=True,
                               max_prefix=0, positions='all')
            self.assertEqual(len(ds), 16)
            self.assertEqual(ds[0]['patient'], 'sealed')
            development = build_manifest(root, max_prefix=0)
            self.assertNotIn('test', development)
            self.assertNotIn('sealed', [r['patient'] for r in development['train'] + development['val']])

    def test_val_selected_checkpoint_hash_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'result/tag/seg_decoder_transfer_ft'
            checkpoint = root / 'ckpt/tag/seg_decoder_transfer_ft/best_epoch_0003.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fixture')
            summary = dict(selection='VAL only', mode='finetune', selected_epoch=3,
                           best_file=checkpoint.name,
                           checkpoints={checkpoint.name: dict(sha256=file_hash(checkpoint))})
            write_json(directory / 'summary.json', summary)
            write_json(directory / 'request.json', dict(config={}, source='/unused'))
            mark_done(directory, ['summary.json', 'request.json'])
            self.assertEqual(locked_checkpoint(root, 'tag', 'seg_decoder_transfer_ft')[2], checkpoint)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                locked_checkpoint(root, 'tag', 'seg_decoder_transfer_ft')


if __name__ == '__main__':
    unittest.main()
