import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.downstream import ViTPatchSegDecoder
from research_pipeline.echo import TemporalTask
from research_pipeline.ef_geometry import case_geometry, coefficients, contractions, cross_fitted, deduplicate, predict
from tools.run_ar_ef_geometry import decode_window, equivalence, visual_inputs


def record(start=0, confidence=.99):
    return dict(start=start,fps=50.,area=[1000+int(450*math.cos((i-5)*2*math.pi/20)) for i in range(32)],
                confidence=[confidence]*32)


class GeometryTest(unittest.TestCase):
    def test_pairing_and_fallback_ignore_labels(self):
        row = record()
        a,_ = contractions(row)
        self.assertTrue(a)
        self.assertTrue(all(.16 <= (p['es']-p['ed'])/50 <= .65 for p in a))
        self.assertEqual(contractions(dict(row,target=-1,ed=63,es=0,traces='forbidden'))[0],a)
        self.assertIsNone(case_geometry([record(confidence=.51)])['g'])
        self.assertIsNone(case_geometry([dict(row,area=[1000]*32)])['g'])
        result = predict(dict(G0=37.,g=None),dict(Gauto=[1,1,1],Gcal=[1,1]))
        self.assertTrue(result['fallback'])
        self.assertEqual(result['Gauto'],37.)

    def test_boundary_jump_and_overlap(self):
        row = record()
        row['area'][8:16]=[650]*8
        pairs,broken = contractions(row)
        self.assertGreaterEqual(broken,1)
        self.assertFalse(any(p['ed']<16<p['es'] for p in pairs))
        p = dict(ed=9.5,es=24.5,g=40.,quality=.9,start=0)
        q = dict(p,ed=10.5,es=25.5,g=60.,start=1)
        r = dict(p,ed=55.5,es=73.5,start=9)
        reps = deduplicate([p,q,r],50)
        self.assertEqual(len(reps),2)
        self.assertEqual(reps[0]['g_median'],50.)
        self.assertIn((reps[0]['ed'],reps[0]['es']),[(p['ed'],p['es']),(q['ed'],q['es'])])

    def test_cross_fit_and_exact_l1_affine(self):
        rng = np.random.default_rng(44)
        rows=[]
        for i in range(50):
            ef,g=rng.uniform(30,70,2)
            rows.append(dict(patient=str(i),G0=ef,g=g if i else None,target=ef+.12*(g-50)-.06*(ef-50)+1))
        coeff=coefficients(rows)
        self.assertEqual(coeff['fitted_n'],49)
        self.assertTrue(np.allclose(coeff['Gauto'],[1,-.6,1.2]))
        oof,_=cross_fitted(rows)
        self.assertEqual(len(oof),50)
        self.assertTrue(all(np.isclose(x['Gauto'],x['target']) for x in oof if x['g'] is not None))
        held=oof[1]
        altered=[dict(x,target=99.) if x['patient']==held['patient'] else x for x in rows]
        self.assertAlmostEqual(cross_fitted(altered)[0][1]['Gauto'],held['Gauto'])

    def test_dense_head_matches_original_batch_route(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        # Real TemporalTask and segmentation decoder; deterministic small stream emitting repeated tubelets.
        calls=[]
        def stream_clip(x,state):
            calls.append(1)
            tokens=x[:,::2].flatten(2).reshape(len(x),8,4,8).repeat_interleave(2,dim=1)
            tokens=tokens if state is None else tokens+state
            return dict(final_state=tokens[:,-1:],frame_outputs=tokens,frame_base_outputs=tokens,
                        local_base_outputs=tokens,local_frame_outputs=tokens)
        b=SimpleNamespace(embed_dim=8,img_size=4,patch_size=2,token_grid=(8,2,2),local_frames=16,
                          tubelet_size=2,frame_readout='repeat',in_chans=2,memory_slots=1,stream_clip=stream_clip)
        model=TemporalTask(b,'seg','finetune',64,'final',False,hidden=8,depth=1,heads=2)
        model.head=ViTPatchSegDecoder(8,2,2,2,8,1,2)
        model.eval()
        v=torch.randn(2,64,2,4,4)
        batch=[dict(patient=str(i),video=v[i],frame_indices=torch.arange(64),fps=50.,
                    prefix_frames=0,recent_start=0,mask='forbidden',target='forbidden') for i in range(2)]
        clean=visual_inputs(batch,0,2)
        self.assertEqual(set(clean[0]),{'patient','video','frame_indices','fps','prefix_frames','recent_start'})
        self.assertEqual(equivalence(model,clean,'cpu')['mask_disagreements'],0)
        calls.clear()
        with torch.inference_mode():
            areas = decode_window(model,v,16)
        self.assertEqual(len(calls),4)
        self.assertEqual(len(areas),2)
        self.assertEqual(len(areas[0][0]),32)
