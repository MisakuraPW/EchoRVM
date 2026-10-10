"""Apply the reviewed follow-up patch only after the current task is sealed."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# A copied guardian lives in this run's operations folder; its cwd is the repo.
if not (ROOT / 'tools/run_temporal_final.py').exists():
    ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

from tools.handoff_temporal_validation_cache import process, verified_controller, verify_boundary, finish_stage_time
from tools.run_temporal_final import Queue, scientific_code
from utils.final_temporal_training import file_digest, write_json, digest


def verify_upgrade(previous, current, expected):
    changed = {key for key in previous['code'] if previous['code'][key] != current.get(key)}
    allowed = {'utils/final_temporal_tasks.py', 'utils/final_temporal_training.py'}
    if changed != allowed or set(previous['code']) != set(current):
        raise ValueError('Follow-up patch must change exactly tasks and adaptation telemetry, not models/data/selection')
    for key in changed:
        if current[key] != expected[key]:
            raise ValueError('Deployed implementation differs from the verified commit')
    updated = copy.deepcopy(previous)
    updated['code'] = current
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--expected_commit', required=True)
    parser.add_argument('--poll_seconds', type=float, default=10)
    args = parser.parse_args()
    if os.name != 'posix' or args.poll_seconds < 1:
        raise ValueError('Linux process identity and a bounded polling interval are required')
    root = Path(args.result_dir).resolve()
    ops = root / 'operations'
    record = json.loads((ops / 'followup_audit_handoff.json').read_text())
    status = ops / 'followup_audit_status.json'
    verified_controller(record)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != record['old_commit']:
        raise ValueError('Old checkout changed before the task boundary')
    if file_digest(root / 'protocol.json') != record['old_protocol_file_sha256']:
        raise ValueError('Registered root protocol changed')
    write_json(status, dict(state='waiting_for_current_task', guardian_pid=os.getpid(),
                            stage=record['boundary_stage'], worker=record['current_worker_pid']))
    terminated = False
    try:
        while True:
            verified_controller(record)
            worker = process(record['current_worker_pid'])
            if worker is None or worker['state'] == 'Z':
                break
            if worker['start'] != record['current_worker_start_ticks']:
                raise ValueError('Worker PID was reused')
            time.sleep(args.poll_seconds)
        verify_boundary(root, record)
        completed = []
        for request in sorted((root / 'jobs').glob('*.json')):
            job = json.loads(request.read_text())
            if root not in Path(job['output_dir']).resolve().parents:
                raise ValueError('Registered job escapes this run')
            Queue.verify_done(None, job)
            completed.append(dict(job=request.name, done_sha256=file_digest(Path(job['output_dir']) / 'DONE')))
        previous = json.loads((root / 'protocol.json').read_text())
        # The paused parent cannot advance; only its completed subprocess is retired.
        verified_controller(record)
        os.kill(record['controller_pid'], signal.SIGTERM)
        os.kill(record['controller_pid'], signal.SIGCONT)
        terminated = True
        for _ in range(100):
            old = process(record['controller_pid'])
            if old is None or old['state'] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Old controller did not exit; refuse duplicate queue')
        finish_stage_time(root, record)
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise ValueError('Server checkout is dirty; no overwrite allowed')
        subprocess.run(['git', 'fetch', args.bundle, 'HEAD'], cwd=ROOT, check=True)
        fetched = subprocess.check_output(['git', 'rev-parse', 'FETCH_HEAD'], cwd=ROOT, text=True).strip()
        if fetched != args.expected_commit:
            raise ValueError('Bundle does not contain the reviewed commit')
        subprocess.run(['git', 'merge', '--ff-only', fetched], cwd=ROOT, check=True)
        expected = {key: __import__('hashlib').sha256(subprocess.check_output(
            ['git', 'show', fetched + ':' + key], cwd=ROOT)).hexdigest()
            for key in ('utils/final_temporal_tasks.py', 'utils/final_temporal_training.py')}
        updated = verify_upgrade(previous, scientific_code(), expected)
        write_json(status, dict(state='verifying_followup', guardian_pid=os.getpid(), commit=fetched))
        # This covers real spawn lifetime/epoch augmentation as well as CUDA head tuning.
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                 '-p', 'test_final_temporal_tasks.py', '-q'], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
        (ops / 'followup_server_tests.log').write_text(result.stdout, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Server task regression failed; inspect followup_server_tests.log')
        write_json(ops / 'protocol_before_followup_audit.json', previous)
        write_json(ops / 'followup_audit_upgrade.json', dict(commit=fetched,
            previous_protocol_sha256=digest(previous), upgraded_protocol_sha256=digest(updated),
            changed_files=sorted(expected), preserved_completed_jobs=completed,
            changes='persistent loaders, bounded cold-feature retry, additive position/P10 and training coverage telemetry',
            scientific_settings_unchanged=True, old_tasks_not_retrained=True,
            server_regression_passed=True, server_test_log=str(ops / 'followup_server_tests.log')))
        write_json(root / 'protocol.json', updated)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        log_path = ops / 'continued_queue_spawn.log'
        with log_path.open('ab') as log:
            resumed = subprocess.Popen(record['resume_command'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(status, dict(state='continued_queue_started', pid=resumed.pid,
                                preserved_completed_jobs=len(completed), log=str(log_path), commit=fetched))
        for _ in range(120):
            if resumed.poll() is not None:
                raise RuntimeError('Continuation exited; inspect the queue log, do not start a duplicate')
            current = json.loads((root / 'current_stage.json').read_text())
            if current['stage'] != record['boundary_stage'] and current['status'] == 'running':
                write_json(status, dict(state='handoff_complete', pid=resumed.pid,
                                        next_stage=current['stage'], commit=fetched))
                return
            time.sleep(1)
        raise RuntimeError('Continuation has not advanced within120 seconds; inspect before intervening')
    except Exception as error:
        write_json(status, dict(state='failed', error=str(error), old_controller_terminated=terminated))
        if not terminated:
            verified_controller(record)
            os.kill(record['controller_pid'], signal.SIGCONT)
        raise


if __name__ == '__main__':
    main()
