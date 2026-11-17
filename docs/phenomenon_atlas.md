# Echocardiography phenomenon atlas

The atlas uses original B-mode images, endpoint contours and a bounded panel
of reviewed full-cycle TED contours. It does not train models. Measurements
are separate from HTML rendering; a report can incorporate historical metrics
without reopening labels. See the dated research plan for the hypotheses and
distinction between data, controlled proxies, and model evidence.

## AutoDL

Activate the existing `echocardmae` environment. Use a fresh `OUT` directory
for each run. The EchoNet root must have `FileList.csv`, `VolumeTracings.csv`
and the project's native-112 grayscale `npy/` cache.

```bash
OUT=/root/autodl-tmp/mae_phenomenon_atlas
ECHO=/root/autodl-tmp/datasets/EchoNet-Dynamic
CAMUS=/root/autodl-fs/datasets/CAMUS
python -u tools/phenomenon_atlas.py --echo-root "$ECHO" --output "$OUT/echo"
python -u tools/prepare_ted_atlas.py --camus-root "$CAMUS" --output "$OUT/ted" \
  --cases 18 --max-mib 768
python -u tools/phenomenon_atlas.py --echo-root "$ECHO" --output "$OUT/combined" \
  --reuse-echo "$OUT/echo/measurements.json" --dense-root "$OUT/ted/dense" \
  --dense-dev-ids "$OUT/ted/dev_ids.txt"
python tools/render_phenomenon_atlas.py "$OUT/combined"
```

Omit the two dense options when using EchoNet only; sections requiring dense
contours show their missing inputs instead of inferred labels. Use
`--image-cases 8` and a separate output folder for an image smoke test.
Dependencies: NumPy, SciPy, scikit-image, Matplotlib; SimpleITK for TED/CAMUS.

For an explicitly labeled fallback, run `tools/prepare_camus_atlas.py`
with `--camus-root` and a fresh `--output`. Pass its dense directory and
ID list to the measurement tool with `--dense-source camus_half_proxy`.
These are official CAMUS validation half-sequences and their released masks;
intermediate masks are proxies, not independent full-cycle contours.

TED items are selected by seeded candidate order and content-verified against
**both** CAMUS development endpoint images. TED serial IDs are not CAMUS IDs.
No holdout images, contours or config labels are used for matching. TED's
published 4CH images and labels are resampled to a common 112-pixel grid;
physical time uses the matched CAMUS metadata. Review `matches.json` and
report exclusions; the resulting panel is an exploratory subset, not the TED
benchmark. TED: N. Painchaud et al., IEEE TMI 2022,
[10.1109/TMI.2022.3173669](https://doi.org/10.1109/TMI.2022.3173669),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Read the report

`report.html` embeds SVG plots and exports them to `figures/`. `--prior-table AGGREGATE.csv` adds previously
reported metrics; `--prior "Title" CURVE.png` is repeatable. Literature
figures are linked to the primary source, not counted as own results.
Keep the aggregate measurement file with the report for provenance.

The initial D1 atlas deliberately uses simple references: pixel-nearest
completion, fully observed historical copy, shared binary masks and
piecewise-constant area summaries. These are not model ablations. It does
not establish causal effects of history, true per-beat EF, clinical early
stopping, benefits of online variable boundaries, or a winning SSL recipe.
