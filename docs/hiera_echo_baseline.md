# Hiera-T Echo MAE Baseline

This baseline is intentionally separate from EchoRVM. It vendors the official
Meta Hiera package under `third_party/hiera` without editing upstream code.

## What It Runs

- Official `mae_hiera_tiny_224` architecture.
- Official ImageNet MAE checkpoint: `ckpt/mae/mae_hiera_tiny_224.pth`.
- Echo input as single grayscale frame normalized to `[0,1]`, repeated to RGB.
- Formal pretraining resolution: `192x192`.
- Default MAE `mask_ratio: 0.6`.
- Default loss mode: `valid_weighted`; use `official` to reproduce upstream loss.

## Smoke Test

```bash
bash scripts/run_hiera_smoke.sh
```

## Pretrain On AutoDL

Put the Hiera checkpoint in the project root:

```text
ckpt/mae/mae_hiera_tiny_224.pth
```

Then run:

```bash
bash scripts/run_hiera_pretrain_echonet.sh
bash scripts/run_hiera_pretrain_camus.sh
```

Or run both:

```bash
bash scripts/run_hiera_stage1.sh
```

For the newer CAMUS NIfTI layout, keep the dataset at:

```text
/root/autodl-fs/datasets/CAMUS/database_nifti
/root/autodl-fs/datasets/CAMUS/database_split
/root/autodl-fs/datasets/CAMUS/jupyter
```

Check discovery before training:

```bash
python tools/check_camus_dataset.py --root /root/autodl-fs/datasets/CAMUS
```

CLI overrides are supported:

```bash
bash scripts/run_hiera_pretrain_echonet.sh --batch_size 96 --num_workers 8 --lr 2e-4
```

## Checkpoint Tools

Inspect:

```bash
python tools/inspect_hiera_checkpoint.py ckpt/mae/mae_hiera_tiny_224.pth
```

Convert 224 weights to a clean 192 checkpoint:

```bash
python tools/convert_hiera_checkpoint.py \
  --source ckpt/mae/mae_hiera_tiny_224.pth \
  --output ckpt/mae/hiera_tiny_mae_in1k_192_init.pth \
  --target_size 192
```

The trainer can load either the original 224 checkpoint directly or the
converted 192 checkpoint.

## Recurrent Extension Hook

`hiera_echo.models.EchoHieraMAE.encode_frame()` returns rerolled stage features
as `[B,C,H,W]`. That is the intended entry point for a later Recurrent-Hiera
module; do not consume raw unrolled Hiera tokens as spatial states.

## Baseline Comparison

VideoMAE single-frame baseline:

```bash
bash scripts/run_videomae_single_frame_stage1.sh
```

Full Hiera-T vs VideoMAE single-frame comparison:

```bash
bash scripts/run_mae_baseline_compare_full.sh
```

Smoke version:

```bash
bash scripts/run_mae_baseline_compare_smoke.sh
```

The full script runs:

1. EchoNet Hiera-T MAE pretrain.
2. EchoNet segmentation fine-tune from Hiera-T.
3. EchoNet VideoMAE single-frame pretrain.
4. EchoNet segmentation fine-tune from VideoMAE single-frame.
5. CAMUS Hiera-T MAE pretrain.
6. CAMUS segmentation fine-tune from Hiera-T.
7. CAMUS VideoMAE single-frame pretrain.
8. CAMUS segmentation fine-tune from VideoMAE single-frame.

Reports are written to:

```text
/root/autodl-tmp/outputs_baseline_compare/<RUN_TAG>/comparison.csv
/root/autodl-tmp/outputs_baseline_compare/<RUN_TAG>/comparison.md
```

The report includes loss/monitor metrics, downstream Dice when present,
stage wall time, dummy inference latency, parameter count, and best checkpoint
size.
