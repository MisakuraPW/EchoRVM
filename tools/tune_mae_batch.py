"""Explicitly authorized SSL effective-batch throughput/safety search."""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_pipeline.artifacts import ROOT, write_json
from research_pipeline.config import load, stage_config
from research_pipeline.runtime import run_probe_process


def select_batch(trials):
    safe = [v for v in trials if v['status'] == 'ok']
    if not safe:
        raise RuntimeError('No safe batch; no launch configuration written')
    fastest = max(v['samples_per_second'] for v in safe)
    return min((v for v in safe if v['samples_per_second'] >= .97*fastest),
               key=lambda v: v['selection']['micro_batch'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pipeline/clip_recurrent_global.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batches) < 1:
        parser.error('Batch candidates must be positive')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = load(args.config)
    stage = next(s for s in base['stages'] if s['id'] == 'pretrain')
    trials = []
    for batch in sorted(set(args.batches)):
        cfg = stage_config(copy.deepcopy(base), stage)
        cfg['training']['effective_batch'] = batch
        cfg['runtime'].update(threads=args.threads, probe_warmup_updates=2, probe_updates=6)
        selection = dict(micro_batch=batch, num_workers=args.workers,
                         gradient_checkpointing=False, threads=args.threads)
        response = root/f'batch_{batch}.response.json'
        request = root/f'batch_{batch}.request.json'
        write_json(request, dict(config=cfg, source=None, selection=selection, response=str(response)))
        print(f'[batch search] testing effective=micro={batch}, workers={args.workers}, threads={args.threads}', flush=True)
        with (root/f'batch_{batch}.log').open('w') as log:
            returncode = run_probe_process([sys.executable, '-u', '-m', 'research_pipeline', '_probe', '--request', str(request)],
                                           log, 240)
        if returncode is None:
            raise RuntimeError(f'Probe timed out; owned probe/reader processes terminated: {root}')
        if not response.exists():
            raise RuntimeError(f'Probe exited {returncode} without response; inspect {root}')
        value = json.loads(response.read_text())
        if value['status'] == 'error':
            raise RuntimeError(f'Non-OOM failure: {value}')
        trials.append(value)
        write_json(root/'trials.json', trials)
        print(json.dumps(value), flush=True)
        if value['status'] in {'oom', 'headroom_or_skipped_update'}:
            # Larger physical batches cannot repair a maximum-context capacity failure.
            break
    selected = select_batch(trials)
    report = dict(selected=selected['selection'], trials=trials,
                  scientific_change='Effective batch changes; same LR, epoch budget, sample policy and objective. Fresh matched B0/A trajectory required.',
                  baseline_only=True, note='Each candidate includes longest real context forward/backward and actual data/AdamW timing; no global-optimum claim')
    write_json(root/'selection.json', report)
    print('[batch search] selected '+json.dumps(report['selected']), flush=True)


if __name__ == '__main__':
    main()
