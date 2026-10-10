"""One isolated job of the final temporal study; orchestration owns run locks."""

import argparse
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def configure_worker_runtime():
    # Forking after Torch/CUDA work can inherit locked thread pools. Configure
    # spawn before datasets create shared epoch counters or loaders start.
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_num_threads(4)


def main():
    configure_worker_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if job['kind'] == 'preflight':
        from utils.final_temporal_preflight import run_preflight_job
        run_preflight_job(job, manifest, args.device)
    elif job['kind'] == 'adapt':
        from utils.final_temporal_training import run_warm_job
        run_warm_job(job, manifest, args.device)
    elif job['kind'] == 'task':
        from utils.final_temporal_tasks import run_task_job
        run_task_job(job, manifest, args.device)
    elif job['kind'] == 'representation':
        from models.final_temporal_mae import load_final_model
        from utils.final_temporal_representation import run_representation_audit
        model = load_final_model(job['checkpoint'])[0].to(args.device).eval()
        reference = load_final_model(job['reference_checkpoint'])[0].to(args.device).eval() if job.get('reference_checkpoint') else None
        run_representation_audit(model, manifest, job['output_dir'], job.get('representation', {}), args.device,
                                 reference_model=reference, detailed=job.get('detailed', False),
                                 with_memory_prediction=job.get('memory_prediction', False))
    elif job['kind'] == 'history':
        from utils.final_temporal_history import run_history_job
        run_history_job(job, manifest, args.device)
    elif job['kind'] == 'representation_memory':
        from models.final_temporal_mae import load_final_model
        from utils.final_temporal_representation import run_memory_prediction_audit
        model = load_final_model(job['checkpoint'])[0].to(args.device).eval()
        reference = load_final_model(job['reference_checkpoint'])[0].to(args.device).eval()
        run_memory_prediction_audit(model, reference, manifest, job['output_dir'], job.get('representation', {}), args.device)
    elif job['kind'] == 'mechanisms':
        from models.final_temporal_mae import load_final_model
        from utils.final_temporal_mechanisms import run_mechanism_audit
        model = load_final_model(job['checkpoint'])[0].to(args.device).eval()
        run_mechanism_audit(model, manifest, job['output_dir'], job.get('audit', {}), args.device)
    elif job['kind'] == 'streaming':
        from utils.final_temporal_streaming import run_streaming_audit
        run_streaming_audit(job, manifest, args.device)
    else:
        raise ValueError('Unknown final-study job kind')


if __name__ == '__main__':
    main()
