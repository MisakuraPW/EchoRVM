import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.phenomenon_atlas import common_output_floor, dense_measurements, split_rows
from tools.render_phenomenon_atlas import render


class AtlasTest(unittest.TestCase):
    def test_shared_mask_floor(self):
        a = np.array([True, False, True, False])
        b = np.array([False, False, True, True])
        self.assertEqual(common_output_floor(a, b), .25)
        for bits in np.ndindex(2, 2, 2, 2):
            pred = np.array(bits)
            self.assertGreaterEqual((np.mean(pred != a) + np.mean(pred != b))/2, .25)

    def test_development_only(self):
        with tempfile.TemporaryDirectory() as base:
            p = Path(base) / "split.csv"
            p.write_text("FileName,Split\na,TRAIN\nb,TEST\nc,VAL\n", encoding="utf-8")
            self.assertEqual([r["FileName"] for r in split_rows(p, {"TRAIN", "VAL"})], ["a", "c"])

    def test_dense_report_and_missing_path(self):
        with tempfile.TemporaryDirectory() as base:
            p = Path(base)
            yy, xx = np.mgrid[:24, :24]
            mask = np.array([(xx-12)**2+(yy-12)**2 <= r*r for r in range(4, 14)]*2 + [xx**2+yy**2 < 24])
            np.savez(p / "allowed.npz", video=mask.astype(np.float32)*.5+.2,
                     mask=mask.astype(np.uint8), fps=50.)
            ids = p / "ids.txt"
            ids.write_text("allowed\n", encoding="utf-8")
            (p / "not_allowed.npz").write_bytes(b"should not be opened")
            pairs, curves, budget, refs = dense_measurements(p, ids)
            self.assertEqual(len(curves), 1)
            self.assertTrue(all(r["floor"] >= 0 for r in pairs))
            self.assertEqual({r["slots"] for r in budget if r["task"] == "partition"}, {4})
            variable = [sorted(r["lengths"]) for r in budget if r["task"] == "partition" and r["strategy"] != "equal"]
            self.assertEqual(variable[0], variable[1])
            data = dict(metadata=[], spatial=[], history=[], scale=[], perturb=[],
                        dense_pairs=pairs, curves=curves, budget=budget, references=refs,
                        counts=dict(development_rows=0, with_endpoint_pair=0, image_cases=0), dense_cases=1)
            html = render(data, [])
            self.assertIn("data:image/svg+xml", html)
            self.assertIn("未运行", html)


if __name__ == "__main__":
    unittest.main()
