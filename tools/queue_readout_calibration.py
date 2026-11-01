"""Wait for the existing queue; smoke and run in an isolated code snapshot."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_identity(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if value[0] == 'Z' else value[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--previous-result', type=Path, required=True)
    parser.add_argument('--tag', default='cr_readout_calibration_20261101')
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    identity = process_identity(args.wait_pid)
    print(f'Waiting for queue pid={args.wait_pid}; no GPU allocation while waiting', flush=True)
    while identity is not None and process_identity(args.wait_pid) == identity:
        time.sleep(30)
    from research_pipeline.artifacts import run_lock, verify_done
    root = args.previous_result
    selected = json.loads((root/'execution_selection.json').read_text())['stages']
    for name in selected:
        directory = root/name
        if not verify_done(directory):
            # A is intentionally paused at the declared 400/1600 budget.
            if name != 'A' or not (directory/'PAUSED.json').is_file():
                raise RuntimeError(f'Previous queue incomplete: {name}; do not bypass failed work')
    if not (root/'forward_comparison.json').is_file():
        raise RuntimeError('Previous comparison missing; inspect original queue before proceeding')
    from research_pipeline.config import load
    cfg = load('configs/pipeline/readout_calibration.yaml', tag=args.tag)
    result = Path(cfg['run']['root'])/'result'/args.tag
    with run_lock(result/'launcher'):
        # Do not compete with another project's GPU workload.
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                      text=True).strip():
            print('GPU has another compute process; waiting without terminating it', flush=True)
            time.sleep(60)
        base = [sys.executable, '-u', '-m', 'research_pipeline']
        common = ['--config', 'configs/pipeline/readout_calibration.yaml', '--tag', args.tag]
        subprocess.run(base+['smoke']+common, check=True)
        subprocess.run(base+['run']+common, check=True)


if __name__ == '__main__':
    main()
