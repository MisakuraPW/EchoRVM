from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from research_pipeline.ar_gap_ft import GapTask, source_groups
from tools.run_ar_gap_ft import paired_split


class FixtureBackbone(nn.Module):
    in_chans = 1
    embed_dim = 1
    token_grid = (8,1,1)

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def encode_current(self, clip, mask, state):
        if mask is not None or state is not None:
            raise AssertionError('Groups must start without history')
        value = clip.flatten(2).reshape(len(clip),8,2,1).mean(2)*self.weight
        self.calls.append(clip.detach().flatten().tolist())
        return value,value

    def frame_features(self, native):
        return native.repeat_interleave(2,dim=1)


class FixtureTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FixtureBackbone()
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(64,1,bias=False),nn.Flatten())


class GapTests(unittest.TestCase):
    def test_groups_and_tubelets_restore_original_time(self):
        torch.set_num_threads(1)
        video = torch.arange(64.).reshape(1,64,1,1,1)
        for arm in ('D','G'):
            idx = source_groups(arm)
            self.assertEqual(idx.flatten().sort().values.tolist(),list(range(64)))
            self.assertEqual(idx[0].tolist(),list(range(0,16)) if arm=='D' else list(range(0,64,4)))
            model = GapTask(FixtureTask(),arm)
            out = model.descriptors(video)[0,:,0]
            expected = torch.empty(64)
            for x,y in idx.reshape(-1,2).tolist():
                expected[x]=expected[y]=(x+y)/2
            torch.testing.assert_close(out,expected,rtol=0,atol=0)
            self.assertEqual(model.backbone.calls,idx.tolist())
            out.sum().backward()
            self.assertGreater(abs(float(model.backbone.weight.grad)),0)

    def test_direct_comparison_reads_bom_and_requires_matched_windows(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for arm,v in [('D',55),('G',51)]:
                for p in ('single','five'):
                    dest=directory/f'{arm}_val'/p
                    dest.mkdir(parents=True)
                    (dest/'candidate_windows.csv').write_text(f'patient,recent_start,target,prediction\np,8,50,{v}\n',
                                                              encoding='utf-8-sig')
            with patch('tools.run_ar_gap_ft.population_strata',return_value={'example':{'p'}}):
                r=paired_split(directory,'val',None)
                self.assertEqual(r['five']['pairs']['G_minus_D']['candidate_minus_reference'],-4.)
                p=directory/'G_val'/'five'/'candidate_windows.csv'
                p.write_text('patient,recent_start,target,prediction\np,9,50,51\n',encoding='utf-8-sig')
                with self.assertRaisesRegex(ValueError,'windows differ'):
                    paired_split(directory,'val',None)
