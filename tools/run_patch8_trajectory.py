"""Queue frozen probes of existing snapshots after the parent run completes."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def build_config(plan, root, parent_tag):
    cfg = copy.deepcopy(plan)
    cfg['run'] = dict(root=str(root), tag=parent_tag + '_trajectory')
    stages = {stage['id']: stage for stage in plan['stages']}
    cfg['stages'] = []
    for epoch in (400, 800):
        source = str(Path(root) / 'ckpt' / parent_tag / 'pretrain' / f'epoch_{epoch:04d}.pt')
        for task in ('ef', 'seg'):
            stage = copy.deepcopy(stages[f'{task}_linear_1600'])
            stage.update(id=f'{task}_linear_{epoch:04d}', source=source)
            cfg['stages'].append(stage)
    return cfg


def parent_failed(parent):
    queue = parent / 'queue.jsonl'
    if queue.exists():
        events = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
        if events and events[-1]['state'] == 'failed':
            return True
    for name in ('seg_decoder_transfer_ft', 'ef_l1_ft'):
        path = parent / name / 'status.json'
        if path.exists() and json.loads(path.read_text()).get('phase') == 'failed':
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/outputs_cr_spatial'))
    parser.add_argument('--parent-tag', default='cr_patch8_w64_1600_v1')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from research_pipeline.artifacts import run_lock, verify_done, write_json
    from research_pipeline.config import validate

    parent = args.root / 'result' / args.parent_tag
    plan = json.loads((parent / 'plan.json').read_text())
    cfg = validate(build_config(plan, args.root, args.parent_tag))
    if args.plan:
        print(json.dumps(cfg, indent=2))
        return
    control = args.root / 'followup' / cfg['run']['tag']
    with run_lock(control):
        def status(phase, **extra):
            write_json(control / 'status.json', dict(phase=phase, pid=os.getpid(),
                       time=time.time(), parent=str(parent), **extra))
        try:
            deadline = time.monotonic() + 86400
            while not (parent / 'DONE.json').exists():
                if parent_failed(parent):
                    raise RuntimeError('Parent failed; no follow-up probes started')
                if time.monotonic() > deadline:
                    raise TimeoutError('Parent not completed after 24 hours')
                status('waiting_for_parent')
                time.sleep(30)
            if not verify_done(parent):
                raise RuntimeError('Parent completion verification failed')
            for stage in cfg['stages']:
                if not Path(stage['source']).is_file():
                    raise FileNotFoundError(stage['source'])
            config_path = control / 'config.json'
            write_json(config_path, cfg)
            status('running')
            subprocess.run([sys.executable, '-u', '-m', 'research_pipeline', 'run',
                            '--config', str(config_path)], cwd=args.repo, check=True)
            if not verify_done(args.root / 'result' / cfg['run']['tag']):
                raise RuntimeError('Follow-up exited without verified completion')
            status('done')
        except BaseException as exc:
            status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
