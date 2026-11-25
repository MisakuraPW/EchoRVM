# A-R EF readout screening

Run from the existing EchoRVM checkout on the MAE instance. This is an EF-only experiment. It does not alter the segmentation path or checkpoints, train SSL, or automatically fine-tune a backbone.

```bash
conda activate echocardmae
git pull --ff-only
python tools/run_ar_ef_pairing.py plan --tag ef_pair_20261125
python tools/run_ar_ef_pairing.py smoke --tag ef_pair_20261125
python tools/run_ar_ef_pairing.py run --tag ef_pair_20261125
```

- `run` does E0 VAL first, then fits I/J/R/P and reports their VAL comparison; subsequently it runs locked E0 and all trained arms on TEST. `--with-stride4` adds one equal-frame probe to the same matched cohort. Use the same option for smoke and full run when checking that path.
- `e0 --tag ef_e0_20261125` runs only E0 on VAL and TEST. Use this when deferring all head fitting. The combined `run` includes its own E0; do not run both unnecessarily.
- `plan` and `smoke` do not build a TEST manifest. Smoke uses a separate suffixed tag, tiny development subsets and a one-epoch head schedule. `--cpu` is permitted only for smoke. Formal resume uses the same command/config/tag and the existing engine's epoch-boundary recovery; scientific or code changes require a new tag.
- The original A-R artifact tree, original frozen C0 protocol and manifest are required. The runner verifies the sealed epoch-21 EF FT endpoint and the SSL epoch-400 source hashes, the original population/loss policy and the dataset root. Override `--stage-a-root`, `--stage-a-tag`, `--a2-root`, `--a2-tag` and `--data-root` only to point to those existing artifacts, not another training recipe.

## Interpretation and cost

| Stage | Controlled change | Readout |
| --- | --- | --- |
| E0 | Original deterministic W64 versus up to five distinct uniformly placed W64 starts | Patient mean of window predictions; fixed FT weights and original inference precision. Repeated rounded starts are deduplicated. |
| I/J | Same four independently encoded L16 clips, state cleared each time | I uses one shared C0 head per clip and averages before patient L1. J uses the same descriptors with one ordered C0 pooling operation. |
| R/P | Same recurrently encoded W64 and C0 path | R is a newly fitted C0 head. P adds a zero-initialized residual using soft ordered latent pairs, including cross-clip pairs, with a learned null option. |
| I_stride4 | Four offset/stride4 tracks partition the same 64 frames | Same I head and loss. No historical state is passed between overlapping tracks. |

The head arms use the inherited frozen recipe (up to 100 head epochs, patience 10, effective batch 32), with FP32 for all arms to keep small residuals from being quantized differently. No ED/ES labels, EDV/ESV, or training-side phase supervision enter the model. P tests latent candidate pairs separated by 0.12–0.65 s using FPS. At inference an unavailable pair or a null probability of at least 0.75 gives exactly the C0 prediction. This is a fixed, uncalibrated gate, not a physiological phase estimator. The VAL-only stress test keeps pair scores and elapsed times fixed and perturbs the end descriptors within each start's admissible time band. It cannot establish phase identity.

Outputs include paired bootstrap deltas and MAE/bias by true EF bins and endpoint coverage. Endpoint grouping uses annotations only after the window and predictions are determined: two ordered in-window traces with descending area, categorized as same or cross L16, otherwise uncovered. E0 uses the single-window groups for both arms. No TEST result gates advancement. A VAL MAE gain of 0.10 pp for P over R only makes it eligible for a later matched FT decision; this entry point never starts FT.

## Files

- `outputs_ar_ef_pairing/result/<tag>/ef_pairing/`: `comparison.json`, child summaries, per-window/per-patient CSVs, and formal selection locks.
- `outputs_ar_ef_pairing/result/<tag>/<arm>/`: runtime trials, training metrics and status. `outputs_ar_ef_pairing/ckpt/<tag>/<arm>/`: head-only checkpoints and recovery state.
- `outputs_ar_ef_pairing_cache/<tag>/`: disposable memmapped frozen features, separate from results and weights. About 0.80 GiB for two views at the known development size, about 1.20 GiB with stride4, plus metadata. `--cache-gb` and `--reserve-gb` guard allocation; this code never auto-deletes old artifacts. TRAIN/VAL extraction uses the reference FP32 backbone. Head training uses the existing measured micro-batch/worker autotuner; E0 reuses the original FT runtime selection.

The terminal shows active evaluation and training progress; inspect each stage's `status.json` and `metrics.csv` after disconnecting. Use process-local logs for queue startup or failures before a stage heartbeat.
