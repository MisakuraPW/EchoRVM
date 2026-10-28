# Clip-Recurrent Execution Repair, 2026-10-29

## Failure and Evidence

- Old run: `cr_forward_v1`, original cloud code `fd75069`.
- Epochs 1/2 took 921.06/948.57 seconds for 10,030 real windows, not one minute.
- Epoch 1 processed 879,776 frames / 54,986 local clips. Prefix lengths remained
  variable, 0 through 128; there was no fixed 192-frame requirement.
- Tuning selected micro-batch 8, workers 0, checkpointing off. Each trial measured
  only two updates. It could not repair the adapter splitting each mixed-length
  micro-batch into many small length groups, then unrolling each independently.
- Observed GPU utilization was 15%, with one CPU core approximately saturated.
  Low utilization alone is not proof of data starvation: measured data wait was
  about 9-10% of the epoch, not the majority.

## Changes

`research_pipeline/mae_execution.py` batches the real active clips at each
recurrent step. Completed windows leave the batch and their states are removed
by index selection. It preserves state gradients, exact source windows, native
resolution, clip length, mask seeds, per-window target normalization and loss
weighting, effective batch, optimizer and LR schedule. There is no fake padding
or truncated prefix. The reference path remains for stochastic mask/layer cases.

The generic VideoMAE encoder no longer reads a visible-token count back to the
CPU on every clip. Runtime tuning now warms up twice, measures six whole updates,
and also tests CPU thread counts. `last.pt` is overwritten each epoch for durable
recovery; milestone snapshots and their retention are unchanged.

Changing batching changes floating-point reduction order. This is mathematical
objective equivalence, not bitwise-identical training. Six CPU comparisons cover
B0/A/W, tube/complementary masks, unequal windows, shortened reconstruction suffix,
losses, all parameter gradients and an AdamW step. Total local suite: 41 tests.

## Native RTX 4090 Checks

`tools/benchmark_mae_execution.py` used identical real windows and BF16. It ran
one warmup and four measured forward/backward passes per implementation, without
changing weights. Timings exclude disk reads and AdamW and are NOT epoch timings.

| Windows | Reference windows/s | Active windows/s | Speedup | Relative loss error | Gradient L2 error |
| --- | --- | --- | --- | --- | --- |
| 8 | 16.01 | 41.28 | 2.58x | 0.0102% | 1.057% |
| 32 | 17.09 | 49.33 | 2.89x | 0.00105% | 0.798% |

These checks justify deployment, not a claim that 1,600 epochs now take one day.
Use the new full-epoch measurements and observed-frame counts for estimates.

## Deployment and Recovery

User explicitly authorized stopping the old queue after verification and
repeating the early epochs. Its only durable recovery checkpoint was epoch 0;
in-memory epoch 3 was not preserved. Old result/weight directories remain intact.

New code commit: `45bc533`. Published to upstream. Cloud GitHub fetching stalled,
so a Git bundle containing exactly the published commits was transferred and
merged fast-forward. No force reset or overwrite was used.

New run: `cr_forward_fast_v2`, same full B0 1,600-epoch / A 400-epoch queue.
Results: `/root/autodl-tmp/outputs_cr_clean/result/cr_forward_fast_v2/`.
Weights: `/root/autodl-tmp/outputs_cr_clean/ckpt/cr_forward_fast_v2/`.

```bash
tail -f /root/autodl-tmp/cr_forward_fast_v2.log
# Read-only structured viewer:
python tools/watch_pipeline_progress.py --tag cr_forward_fast_v2
```

The parent now prints a stage counter and progress lines while the worker runs.
Per-stage full output remains in `worker.log`. Do not pull changing model/engine
code into this active run; its protocol records source hashes.

## Monitoring Correction

The old ten-minute cron is paused; its two new chats were explicitly archived.
It had not actually performed successful unattended monitoring or archiving.
Do not equate a successful manual SSH read with validated scheduled execution.
