import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from utils.checkpoint import atomic_torch_save, checkpoint_directory, should_save_last, load_checkpoint
from utils.research_storage import (check_protocol, prepare_layout, prune_audited_snapshot,
                                    prune_completed_resume, remove_owned_checkpoint)


class StorageTests(unittest.TestCase):
    def test_save_schedule_and_legacy_paths(self):
        cfg = {'checkpoint': {'save_last_every_n_epochs': 10, 'save_epochs': [25,50]}}
        saved = [e for e in range(1,54) if should_save_last(cfg,e,53)]
        self.assertEqual(saved,[1,10,20,25,30,40,50,53])
        self.assertTrue(should_save_last(cfg,17,53,stopping=True))
        self.assertEqual(checkpoint_directory('r'),Path('r/checkpoints'))
        self.assertEqual(checkpoint_directory('r',{'checkpoint':{'dir':'separate'}}),Path('separate'))

    def test_disk_guard_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'last.pt'
            atomic_torch_save({'v':torch.ones(2)},path)
            previous = path.read_bytes()
            with patch('utils.checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(OSError,'Previous checkpoint kept'):
                    atomic_torch_save({'v':torch.zeros(2)},path,min_free_gb=2)
            self.assertEqual(path.read_bytes(),previous)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_partial_interrupt_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'interrupt.pt'
            model = torch.nn.Linear(2,1)
            opt = torch.optim.AdamW(model.parameters())
            atomic_torch_save({'partial_epoch': True, 'model_state_dict': model.state_dict()},path)
            with self.assertRaisesRegex(ValueError,'partial epoch'):
                load_checkpoint(path,model,opt)

    def test_retention_requires_success_and_preserves_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt, result = root/'ckpt/m', root/'result/m'
            ckpt.mkdir(parents=True)
            result.mkdir(parents=True)
            for name in ('epoch_0000.pt','epoch_0050.pt','epoch_0400.pt','last.pt','interrupt.pt'):
                (ckpt/name).write_bytes(b'weight')
            prune_audited_snapshot(ckpt,result,50,400)
            self.assertTrue((ckpt/'epoch_0050.pt').exists())
            for e in (0,50,400):
                audit = result/'audit'/f'epoch_{e:04d}'
                audit.mkdir(parents=True)
                (audit/'DONE').write_text('ok')
                (audit/'metrics.json').write_text('{}')
            prune_audited_snapshot(ckpt,result,50,400,keep=True)
            self.assertTrue((ckpt/'epoch_0050.pt').exists())
            prune_audited_snapshot(ckpt,result,50,400)
            prune_audited_snapshot(ckpt,result,400,400)
            self.assertFalse((ckpt/'epoch_0050.pt').exists())
            self.assertTrue((ckpt/'epoch_0400.pt').exists())
            prune_completed_resume(ckpt,result,400,[0,50,400])
            self.assertTrue((ckpt/'last.pt').exists())
            (result/'PRETRAIN_DONE').write_text('ok')
            prune_completed_resume(ckpt,result,400,[0,50,400],audited=False)
            self.assertTrue((ckpt/'last.pt').exists())
            prune_completed_resume(ckpt,result,400,[0,50,400])
            self.assertFalse((ckpt/'last.pt').exists())
            self.assertTrue((ckpt/'epoch_0400.pt').exists())
            self.assertEqual(len((result/'checkpoint_cleanup.jsonl').read_text().splitlines()),3)
            outside = root/'unrelated.pt'
            outside.write_bytes(b'do not touch')
            with self.assertRaises(RuntimeError):
                remove_owned_checkpoint(outside,ckpt,result,'test')
            self.assertTrue(outside.exists())

    def test_migration_and_storage_only_protocol_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root/'method'
            (legacy/'checkpoints').mkdir(parents=True)
            (legacy/'checkpoints/last.pt').write_bytes(b'full checkpoint')
            anchor = legacy/'full_finetune/echonet_ef'
            (anchor/'checkpoints').mkdir(parents=True)
            (anchor/'checkpoints/best.pt').write_bytes(b'best checkpoint')
            (anchor/'metrics.json').write_text('{}')
            cfg = dict(model={'frames':16},checkpoint={'save_epochs':[50,400]})
            (legacy/'requested_config.yaml').write_text(yaml.safe_dump(cfg))
            (legacy/'protocol.sha256').write_text('old version digest')
            (root/'stage_times.csv').write_text('stage,seconds\nx,1\n')
            unrelated = root/'user_notes.txt'
            unrelated.write_text('keep')
            with self.assertRaisesRegex(RuntimeError,'Legacy output'):
                prepare_layout(root,['method'])
            result,ckpt = prepare_layout(root,['method'],migrate=True)
            self.assertEqual((ckpt/'method/last.pt').read_bytes(),b'full checkpoint')
            self.assertTrue((ckpt/'method/full_finetune/echonet_ef/best.pt').is_file())
            self.assertFalse(list(result.rglob('*.pt')))
            self.assertTrue(unrelated.exists())
            new = copy.deepcopy(cfg)
            new['checkpoint'].update(dir=str(ckpt/'method'),save_last_every_n_epochs=10,min_free_gb=2)
            self.assertTrue(check_protocol(result/'method',new))
            new['model']['frames'] = 64
            with self.assertRaisesRegex(RuntimeError,'changed training protocol'):
                check_protocol(result/'method',new)
            prepare_layout(root,['method'],migrate=True)

    def test_migration_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (root/'m/checkpoints/last.pt',root/'ckpt/m/last.pt'):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'keep both')
            with self.assertRaises(FileExistsError):
                prepare_layout(root,['m'],migrate=True)
            self.assertEqual((root/'m/checkpoints/last.pt').read_bytes(),b'keep both')


if __name__ == '__main__':
    unittest.main()
