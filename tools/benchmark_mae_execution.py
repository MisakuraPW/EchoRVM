"""Bounded, real-window reference/active execution check; no training artifacts."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_pipeline.artifacts import write_json
from research_pipeline.cli import configure
from research_pipeline.engine import amp_context, build_experiment
from research_pipeline.mae_execution import active_clip_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    cfg = request.get('config', request)
    cfg['runtime']['threads'] = args.threads
    configure(cfg)
    device = torch.device(cfg['runtime']['device'])
    exp = build_experiment(cfg, request.get('source'), device)
    exp.model.train()
    exp.set_checkpointing(False)
    ids = torch.randperm(len(exp.train), generator=torch.Generator().manual_seed(cfg['training']['seed'])).tolist()
    samples = [exp.train[i] for i in ids[:args.batch]]
    seed = exp.seed

    def reference():
        result = None
        for length in sorted({len(s['video']) for s in samples}):
            group = [s for s in samples if len(s['video']) == length]
            term = exp._mae(group)['loss']*len(group)/len(samples)
            result = term if result is None else result+term
        return result

    report = dict(batch=len(samples), lengths=[len(s['video']) for s in samples], threads=args.threads,
                  scope='Forward/backward on identical real windows, excludes disk reads and Adam step')
    gradients = None
    for name, function in [('reference', reference), ('active', lambda: active_clip_loss(exp.model, samples, seed))]:
        durations = []
        for repeat in range(5):
            exp.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with amp_context(cfg, device):
                loss = function()
            loss.backward()
            torch.cuda.synchronize()
            if repeat:
                durations.append(time.perf_counter()-begin)
        current = {k: p.grad.detach().float().cpu() for k, p in exp.model.named_parameters() if p.grad is not None}
        report[name] = dict(loss=float(loss), seconds=sum(durations)/len(durations),
                            windows_per_second=len(samples)*len(durations)/sum(durations))
        print(json.dumps({name: report[name]}), flush=True)
        if gradients is not None:
            if gradients.keys() != current.keys():
                raise AssertionError('Changed gradient paths')
            difference = sum(float((current[k]-g).double().square().sum()) for k, g in gradients.items())
            magnitude = sum(float(g.double().square().sum()) for g in gradients.values())
            report['relative_gradient_l2'] = (difference/max(magnitude, 1e-30))**.5
        gradients = current
    report['speedup'] = report['reference']['seconds']/report['active']['seconds']
    report['relative_loss_error'] = abs(report['active']['loss']-report['reference']['loss'])/max(abs(report['reference']['loss']), 1e-12)
    # BF16 regrouping is not bitwise identical, but gross semantic mismatches must fail closed.
    report['passed'] = report['relative_loss_error'] < .005 and report['relative_gradient_l2'] < .05
    write_json(Path(args.output), report)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit('Execution comparison failed; do not launch training')


if __name__ == '__main__':
    main()
