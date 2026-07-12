# Hiera-T Echo MAE Baseline

This baseline is intentionally separate from EchoRVM. It uses the official
Meta Hiera repository under `资料/hiera` without editing upstream code.

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

Put these in the project root:

```text
资料/hiera/
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
