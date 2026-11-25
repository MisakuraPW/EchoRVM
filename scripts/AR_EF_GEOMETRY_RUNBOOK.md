# Frozen A-R EF with automatic geometry

This reuses the A-R EF FT epoch 21, independent segmentation FT epoch 48, and sealed E0 single/five-window results. It does not update either task model. The two checkpoints do not share fine-tuned encoder features.

```bash
git pull --ff-only
conda activate echocardmae
python -u tools/run_ar_ef_geometry.py plan --tag ef_geom_20261125
python -u tools/run_ar_ef_geometry.py smoke --tag ef_geom_20261125
python -u tools/run_ar_ef_geometry.py run --tag ef_geom_20261125
```

Plan and smoke never open TEST labels. Smoke has its own suffixed tag. Before formal TEST, the final development coefficients, rules, checkpoint hashes and VAL decision are sealed to `TEST_LOCK.json`. An unchanged formal command can reuse atomic area shards when resumed. Use a new tag after scientific or code changes. Outputs live under `/root/autodl-fs/outputs_ar_ef_geometry/result/<tag>/`; there are no new weight files.

E0 scores and every source window are verified against the original manifests and predictions. Two batches per policy are re-forwarded with the original EF precision to audit the archived scores. Segmentation extracts the same target-frame exit in FP32, but decodes all distinct tubelets in batches after a single window pass. A preflight checks logits and hard masks against the original target-frame path, including both halves of a tubelet and clip boundaries. Saved area shards contain only source start, FPS, 32 areas and 32 mean foreground probabilities per window.

The fixed rule is evaluated first on a small TRAIN panel, with no threshold search. Contractions pair a local predicted ED peak with a following ES valley in the same window. Timing is in seconds, cold start and abrupt clip-boundary jumps are screened, and overlapping windows vote once per apparent beat. This is a candidate geometry proxy, not a clinical phase detector or volumetric EF. Artificial endpoint labels and masks are not fed to inference; the separate VAL diagnostic uses two available annotated endpoints only after prediction and does not treat other beats as false positives.

- `G0`: archived five-window EF.
- `Gcal`: EF-only L1 calibration. Identical fitting cohort and fallback policy as Gauto.
- `Gauto`: EF plus automatic area-shortening proxy; both calibrated arms return exactly G0 when no pair is accepted.

The VAL numbers are **patient-level five-fold cross-fitted development diagnostics**, not independently validated scores. Coefficients are then fit once on development data, frozen, and used on the original matching TEST cases. In the diagnostic a gain of at least 0.10 pp against G0 and a gain over Gcal indicate a signal; no additional experiments are started automatically. See `comparison.json`, `val/paired_cases.csv`, `test/paired_cases.csv` and `val/annotation_diagnostic.csv` for the complete evidence.
