"""Check real data readers after CPU/CUDA thread initialization, without training."""

import argparse
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.final_temporal_worker import configure_worker_runtime
from utils.final_temporal_data import WindowDataset
from utils.final_temporal_tasks import _Inputs, _loader
from utils.final_temporal_training import write_json


def main():
    configure_worker_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    # Reproduce the old failure boundary: initialize thread pools and CUDA first.
    torch.randn(256, 256).square().mean().item()
    torch.randn(256, 256, device=args.device).square().mean().item()
    dataset = WindowDataset(manifest, 'train', task='ef', recent_frames=64,
                            local_frames=16, max_prefix=128, prefix=0,
                            training=False, positions='balanced', limit=4, seed=42)
    inputs = _Inputs(dataset)
    expected = [inputs[i] for i in range(len(inputs))]
    loader = _loader(inputs, 2, 12, 42, workers=4)
    loader.timeout = 45
    started = time.perf_counter()
    actual = []
    for batch in loader:
        actual.extend(batch)
    assert len(actual) == len(expected) and len(actual) > 0
    for before, after in zip(expected, actual):
        assert before['patient'] == after['patient']
        assert before['target_position'] == after['target_position']
        assert torch.equal(before['video'], after['video'])
        assert torch.equal(before['target'], after['target'])
    result = dict(passed=True, start_method=multiprocessing.get_start_method(),
                  workers=4, samples=len(actual), seconds=time.perf_counter() - started,
                  cpu_threads_initialized=True, device=args.device,
                  pixels_labels_and_order_identical=True, no_training=True)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
