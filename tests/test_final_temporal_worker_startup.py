import argparse
import json
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import final_temporal_worker as worker


class WorkerStartupTests(unittest.TestCase):
    def test_file_limit_is_raised_within_hard_limit_and_never_lowered(self):
        for soft, hard, expected in ((1024, 1048576, 65536), (1024, 4096, 4096),
                                     (131072, 1048576, 131072)):
            calls = []
            resource = SimpleNamespace(RLIMIT_NOFILE=7, getrlimit=lambda _: (soft, hard),
                                       setrlimit=lambda limit, values: calls.append((limit, values)))
            with mock.patch.object(worker.os, 'name', 'posix'), \
                 mock.patch.dict(sys.modules, resource=resource):
                worker.raise_worker_file_limit()
            self.assertEqual(calls, [(7, (expected, hard))])

    def test_spawn_is_configured_before_torch_threads(self):
        calls = []
        with mock.patch.object(worker.multiprocessing, 'set_start_method',
                               side_effect=lambda *a, **k: calls.append(('spawn', a, k))), \
             mock.patch.object(worker.torch, 'set_num_threads',
                               side_effect=lambda n: calls.append(('threads', n))):
            worker.configure_worker_runtime()
        self.assertEqual(calls, [('spawn', ('spawn',), {'force': True}), ('threads', 4)])

    def test_main_configures_runtime_before_reading_job_or_dispatch(self):
        calls = []
        args = argparse.Namespace(job='job.json', manifest='manifest.json', device='cpu')
        def read(path, *args, **kwargs):
            calls.append('read')
            return json.dumps({'kind': 'task'} if str(path) == 'job.json' else {})
        with mock.patch.object(worker, 'configure_worker_runtime', side_effect=lambda: calls.append('configure')), \
             mock.patch.object(worker.argparse.ArgumentParser, 'parse_args', return_value=args), \
             mock.patch.object(worker.Path, 'read_text', read), \
             mock.patch('utils.final_temporal_tasks.run_task_job', side_effect=lambda *a: calls.append('dispatch')):
            worker.main()
        self.assertEqual(calls, ['configure', 'read', 'read', 'dispatch'])


if __name__ == '__main__':
    unittest.main()
