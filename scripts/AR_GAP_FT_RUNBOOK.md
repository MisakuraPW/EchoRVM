# Matched D/G EF fine-tuning

```bash
python -u tools/run_ar_gap_ft.py plan --tag ar_gap_ft_dg_20261126
python -u tools/run_ar_gap_ft.py smoke --tag ar_gap_ft_dg_20261126
python -u tools/run_ar_gap_ft.py run --tag ar_gap_ft_dg_20261126
```

The exact A-R SSL400 checkpoint and EF FT recipe are inherited. Both arms independently encode four groups of 16 drawn from the same augmented W64. D splits the contiguous window into four groups; G uses four stride4 tracks. The recurrent state is empty for every group in both arms. Both expand native tubelets with the original repeat rule and restore chronological source indices before the identical C0 head. G changes both the tubelet span and encoder attention input; the comparison does not isolate those effects or validate online inference. No SSL, segmentation, area target, calibration or other branch is fitted.

The serial queue fits only D and G and verifies the original training window order hash on the shared epochs. Each best checkpoint is selected on VAL; the predeclared practical signal is a 0.10 pp G-vs-D gain on VAL single W64. Both models always run locked TEST (original single and five starts), reporting patient-paired D/G and per-arm A-R comparisons, EF strata and timing. Smoke uses a separate result tag, no TEST, and verifies indexing, independent encoding, equal initial head tensors, and encoder gradients.

Default root: `/root/autodl-fs/outputs_ar_gap_ft`. Metrics/locks are under `result/<tag>/gap_queue`, weights under `ckpt/<tag>/<arm>`, and the automatically checked result-only archive under `exports/<tag>.zip`. The same formal command can resume an unchanged protocol.
