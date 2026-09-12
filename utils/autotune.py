"""Conservative runtime tuning policies; no learning hyperparameters are searched."""

import copy
import hashlib
import json
import math

from torch.utils.data import Sampler


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def batch_candidates(config):
    train, model = config['train'], config['model']
    base = int(train['batch_size'])
    effective = base * int(train.get('grad_accum_steps', 1))
    # Batch contrastive negatives and arbitrary external dataset randomness are
    # not invariant to microbatch changes.
    safe = (config.get('data', {}).get('sampling_protocol') == 'temporal_v1'
            and model.get('name') in {'temporal_mae', 'echo_videomae'}
            and not model.get('two_views', False)
            and not model.get('align_loss_weight', 0))
    if not safe:
        return [base]
    return sorted({base, *(b for b in (1, 2, 4, 8, 16, 32, 64, 128)
                           if b <= effective and effective % b == 0)})


def apply_selection(config, selection):
    result = copy.deepcopy(config)
    effective = int(config['train']['batch_size']) * int(config['train'].get('grad_accum_steps', 1))
    batch = int(selection['batch_size'])
    if batch not in batch_candidates(config) or effective % batch:
        raise ValueError('Unsafe autotune microbatch selection')
    result['train'].update(batch_size=batch, grad_accum_steps=effective // batch)
    result['model']['gradient_checkpointing'] = bool(selection['gradient_checkpointing'])
    result['data']['num_workers'] = int(selection['num_workers'])
    # Retain exactly the original epoch sample budget, including its final
    # partial accumulation window, when the microbatch grouping changes.
    if batch != int(config['train']['batch_size']):
        result['train']['epoch_sample_batch'] = int(config['train']['batch_size'])
    return result


def choose_trial(trials):
    valid = [t for t in trials if t.get('status') == 'ok']
    if not valid:
        raise RuntimeError('No safe autotune trial; inspect autotune/trials before training.')
    fastest = max(t['samples_per_second'] for t in valid)
    # Within 3% measurement noise, prefer more memory headroom.
    return min((t for t in valid if t['samples_per_second'] >= fastest * .97),
               key=lambda t: (t['peak_reserved_bytes'], t['num_workers']))


class SampleBudgetBatchSampler(Sampler):
    def __init__(self, sampler, sample_count, batch_size):
        self.sampler, self.sample_count, self.batch_size = sampler, sample_count, batch_size

    def __len__(self):
        return math.ceil(self.sample_count / self.batch_size)

    def __iter__(self):
        batch = []
        for i, index in enumerate(self.sampler):
            if i >= self.sample_count:
                break
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
