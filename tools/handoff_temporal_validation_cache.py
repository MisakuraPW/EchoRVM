"""Wait for the authorized current task boundary, then continue its queue once."""

import argparse
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.final_temporal_training import file_digest, write_json


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path/'stat').read_text().rsplit(')',1)[1].split()
        return dict(state=stat[0], start=stat[19],
                    command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                    cwd=os.readlink(path/'cwd') if stat[0] != 'Z' else None)
    except (FileNotFoundError, ProcessLookupError):
        return None


def verified_controller(record):
    current = process(record['controller_pid'])
    if (current is None or current['start'] != record['controller_start_ticks']
            or current['command'] != record['controller_command'] or current['cwd'] != str(ROOT)):
        raise ValueError('Controller identity changed; no process may be signalled')
    if current['state'] not in ('T','t'):
        raise ValueError('Expected the explicitly suspended controller; refusing a live scheduling race')
    return current


def verify_boundary(root, record):
    request = root/'jobs'/(record['boundary_stage'].replace('/','__')+'.json')
    job = json.loads(request.read_text())
    output = Path(job['output_dir'])
    if root.resolve() not in output.resolve().parents:
        raise ValueError('Boundary output escapes the registered run')
    done = json.loads((output/'DONE').read_text())
    for name, expected in done['artifacts'].items():
        if file_digest(output/name) != expected:
            raise ValueError('Boundary artifact changed: '+name)
    for name, expected in done.get('checkpoints',{}).items():
        if file_digest(Path(job['checkpoint_dir'])/name) != expected:
            raise ValueError('Boundary checkpoint changed: '+name)
    return job


def finish_stage_time(root, record):
    path = root/'stage_times.csv'
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not any(row['stage'] == record['boundary_stage'] and row['status'] == 'completed' for row in rows):
        current = json.loads((root/'current_stage.json').read_text())
        ended = (root/record['boundary_stage']/'DONE').stat().st_mtime
        began = datetime.fromisoformat(current['started']).timestamp()
        rows.append(dict(stage=record['boundary_stage'], seconds=max(0,ended-began), status='completed',
                         ended=datetime.fromtimestamp(ended).isoformat()))
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text,fieldnames=['stage','seconds','status','ended'])
        writer.writeheader(); writer.writerows(rows)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text.getvalue()); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--expected_commit', required=True)
    parser.add_argument('--poll_seconds', type=float, default=10)
    args = parser.parse_args()
    if os.name != 'posix' or args.poll_seconds < 1:
        raise ValueError('This guarded handoff requires Linux /proc and a bounded polling interval')
    root = Path(args.result_dir).resolve()
    ops = root/'operations'
    record = json.loads((ops/'validation_cache_handoff.json').read_text())
    status_path = ops/'handoff_status.json'
    verified_controller(record)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() != args.expected_commit:
        raise ValueError('Server checkout is not the expected verified cache-fix commit')
    write_json(status_path,dict(state='waiting_for_current_task',controller=record['controller_pid'],
                                worker=record['current_worker_pid'],stage=record['boundary_stage'],guardian_pid=os.getpid()))
    killed = False
    try:
        while True:
            verified_controller(record)
            worker = process(record['current_worker_pid'])
            if worker is None or worker['state'] == 'Z':
                break
            if worker['start'] != record['current_worker_start_ticks']:
                raise ValueError('Worker PID was reused; abort handoff')
            time.sleep(args.poll_seconds)
        job = verify_boundary(root,record)
        certificate = ops/'validation_cache_certificate.json'
        subprocess.run([sys.executable,str(ROOT/'tools/check_frozen_validation_cache.py'),
                        '--job',str(root/'jobs'/(record['boundary_stage'].replace('/','__')+'.json')),
                        '--manifest',str(root/'manifest.json'), '--checkpoint',str(Path(job['checkpoint_dir'])/'best.pt'),
                        '--output',str(certificate)],cwd=ROOT,check=True)
        if not json.loads(certificate.read_text())['passed']:
            raise ValueError('GPU execution-equivalence certificate failed')
        identity = json.loads((root/'protocol.json').read_text())
        config = identity['config']
        command = [sys.executable,str(ROOT/'tools/run_temporal_final.py'), '--run_tag',root.name,
                   '--data_root',config['data_root'], '--output_root',config['output_root'],
                   '--num_workers',str(config.get('num_workers',8)), '--updates',str(config['updates']),
                   '--frozen_epochs',str(config['frozen_epochs']), '--ft_epochs',str(config['ft_epochs']),
                   '--max_prefix',str(config['max_prefix']), '--min_free_gb',str(config['min_free_gb']),
                   '--cache_disk_gb','16','--cache_ram_gb',str(config['cache_ram_gb']), '--adopt_validation_cache']
        for value in config['checkpoint']:
            command.extend(['--checkpoint',value])
        if not config['optional_ft']: command.append('--no-optional_ft')
        if not config['second_head_seed']: command.append('--no-second_head_seed')
        if config.get('max_gpu_hours') is not None: command.extend(['--max_gpu_hours',str(config['max_gpu_hours'])])
        verified_controller(record)
        os.kill(record['controller_pid'],signal.SIGTERM)
        os.kill(record['controller_pid'],signal.SIGCONT)
        killed = True
        for _ in range(100):
            controller = process(record['controller_pid'])
            if controller is None or controller['state'] == 'Z': break
            time.sleep(.1)
        else:
            raise RuntimeError('Old controller did not exit; no duplicate queue launched')
        finish_stage_time(root,record)
        env = dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (ops/'continued_queue.log').open('ab') as log:
            resumed = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(status_path,dict(state='continued_queue_started',pid=resumed.pid,command=command,
                                    old_task_preserved=True,certificate=str(certificate),guardian_pid=os.getpid()))
        for _ in range(60):
            if resumed.poll() is not None:
                raise RuntimeError('Continuation exited immediately; inspect continued_queue.log')
            upgraded = ops/'validation_cache_upgrade.json'
            current = json.loads((root/'current_stage.json').read_text())
            if upgraded.exists() and current['stage'] != record['boundary_stage']:
                write_json(status_path,dict(state='handoff_complete',pid=resumed.pid,next_stage=current['stage'],
                                            old_task_preserved=True,certificate=str(certificate),guardian_pid=os.getpid()))
                return
            time.sleep(1)
        raise RuntimeError('Continuation did not reach the next stage within 60 seconds; inspect logs, do not launch a duplicate')
    except Exception as error:
        write_json(status_path,dict(state='failed',error=str(error),old_controller_terminated=killed,guardian_pid=os.getpid()))
        if not killed:
            # Restore only the already-owned controller, never a reused PID.
            verified_controller(record)
            os.kill(record['controller_pid'],signal.SIGCONT)
        raise


if __name__ == '__main__':
    main()
