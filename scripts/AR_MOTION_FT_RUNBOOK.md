# A-R supervised EF motion FT

```bash
git pull --ff-only
conda activate echocardmae
python -u tools/run_ar_motion_ft.py plan --tag ar_motion_ft_20261125
python -u tools/run_ar_motion_ft.py smoke --tag ar_motion_ft_20261125
python -u tools/run_ar_motion_ft.py run --tag ar_motion_ft_20261125
```

M1 preserves the A-R EF recipe, source SSL400, dataset, W64/L16, patch8/tubelet2, seed and task head. A small shared projection derives per-tubelet motion features; the signed last-minus-first change adds a zero-initialized residual to the original mean write before the existing recurrent update. Only real in-window TRAIN area pairs supervise the change in a scalar projection, using Huber on log(ES area/ED area), weighted 0.75 against EF L1 in percentage points. The sampled window, EF input, and inference do not depend on trace annotations. Validation logs the usual EF loss, not the auxiliary objective. Coverage rates in metrics are fractions of all sampled windows; multiply by visited_windows for approximate counts, or divide low_ef_pair_rate by low_ef_rate for low-EF coverage.

The queue reuses the sealed epoch21 A-R baseline and original single/five-window predictions. VAL single-window paired gain of at least 0.10 pp queues M1_aux, which receives the same area supervision with motion writing disabled. Otherwise it queues M2, which adds a zero-initialized sparse mid-encoder read of at most twelve already-completed stride4 history observations with physical time offsets, along with the original recurrent state. M2 does not use the auxiliary loss. Each branch starts from SSL400, not M1 weights. Only these two fitted arms are evaluated on TEST, after their VAL weights and the branch are locked. Both single and matched five-window protocols are reported separately.

Artifacts: `outputs_ar_motion_ft/result/<tag>/motion_queue/` has the comparison and locks. Single-arm metrics, runtime trials and live status are under `result/<tag>/<arm>/`; recoverable state and weights are under `ckpt/<tag>/<arm>/`. The same formal command resumes the same epoch-boundary state. Smoke uses a separate tag with a small real TRAIN/VAL subset, zero-init agreement and active-gradient checks, and never accesses TEST. No SSL, segmentation fitting, or third FT is launched.
