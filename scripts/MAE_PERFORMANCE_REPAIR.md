# Clip-Recurrent Execution Repair, 2026-10-29

## Second Pass: Profiling and Larger Batch

`cr_forward_fast_v2` completed epoch 3 and saved full recovery state (942 updates)
before the authorized stop. Its first two epochs took 176.36/183.96 seconds.

The user explicitly authorized effective batches above 32. Before the indexed
repair, batch 32/64 achieved 62.53/64.01 windows/s; batch 128 OOMed on the longest
192-frame real-window stress pass. Merely doubling batch was not enough then.

Profiling identified 76 `nonzero` calls, 105 stream synchronizations and pageable
per-clip CPU concatenations in one 32-window forward/backward. Fixed-size,
ascending visible-token indices now use gather/scatter; complete real windows
are transferred from loader-pinned storage once, with clip stacking on-device.
For global mean memory, the visible-token mean replaces dense zero scattering.
The reference evaluation path and all parameter names stay available/unchanged.
Losses, gradients and AdamW steps match within declared floating-point tolerance.

The 32-window indexed check achieved 88.22 windows/s versus 67.61 in the prior
active path; `nonzero` count became zero and synchronizations fell to 29.
Final actual-data/AdamW probes using loader-pinned inputs measured:

| Physical = Effective Batch | Windows/s | Maximum Reserved Memory | Longest Stress Context |
| --- | --- | --- | --- |
| 32 | 104.10 | 10.42 GiB | 192 real frames |
| 64 | 140.16 | 20.64 GiB | 192 real frames |

These are short probes, not measured full epochs. Different probe batch sizes
cover different numbers of windows, so compare full-epoch observations as well.
The first three FULL epochs of the deployed trajectory subsequently measured
105.76 / 98.58 / 102.16 seconds, with 10,030 windows per epoch and unchanged
window-sequence hashes for matching epochs. Effective batch is 64. In the actual
parent/worker CUDA environment, physical batch 64 failed the conservative 10%
free-memory headroom criterion despite completing its stress update. The live
selector therefore chose micro-batch 32, two accumulation chunks per update,
two readers and four CPU threads. This is intentionally distinguished from the
standalone 64-window probe; production is NOT claimed to reach 140 windows/s.
A bounded torch.compile experiment exceeded its 240-second limit; it is NOT
enabled in production. Its benchmark-only entry point remains explicitly opt-in.

New trajectory: `cr_forward_fast_v3`. SSL effective batch is 64 for BOTH B0 and A,
with LR unchanged. This changes optimizer update counts; it is not claimed to
be bitwise or optimization-trajectory equivalent to batch 32. Both start freshly
from the same declared initialization rather than mixing recipes mid-run. The
old three epochs and weights remain untouched. Supervised heads retain their
original effective batch. Per-model micro-batch safety tuning remains enabled,
restricted to half/full batch, no recomputation, and 0/2 readers. It can reduce
micro-batch without changing the now-fixed SSL effective batch of 64.

`scripts/run_clip_recurrent_fast.sh all` calibrates once and launches the queue;
`run` reuses a saved calibration, and repeated launches do not silently retune
the scientific effective batch under an existing tag. Resume files are still
overwritten once per epoch; pinned milestone snapshots remain unchanged.

```bash
tail -f /root/autodl-tmp/cr_forward_fast_v3.log
python tools/watch_pipeline_progress.py --tag cr_forward_fast_v3
```

The remaining sections document the earlier repair, not the current run tag.

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
