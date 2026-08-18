# MAE Spatiotemporal Experiment Protocol

This document records the implementation of the new 0-400 epoch research
workflow.  The attached planning document is treated as an experiment
requirement, not as executable code.

## Stage Layout

Stage 0 builds controlled baseline trajectories:

- `echonet_echocardmae_repro`
- `echonet_videomae_clean`
- `camus_echocardmae_repro`
- `camus_videomae_clean`

Stage 2 runs the current recurrent candidates:

- `echonet_rvm_mae`
- `echonet_ttt_mae`
- `camus_rvm_mae`
- `camus_ttt_mae`

All default pretraining runs stop at 400 epochs.  Longer training is a manual
promotion decision.

## Checkpoint Rule

`last.pt` is always the recovery checkpoint and keeps full training state.

Fixed scientific-evaluation checkpoints are configured by:

```yaml
checkpoint:
  save_every_n_epochs: 0
  save_epochs: [50, 100, 150, 200, 250, 300, 350, 400, 600, 800, 1200, 1600]
  epoch_name_width: 4
  save_eval_optimizer_state: false
```

This writes names such as:

```text
epoch_0050.pt
epoch_0100.pt
epoch_0400.pt
```

The fixed epoch checkpoints are model-only by default to reduce disk usage.
`last.pt` and `best.pt` still follow the normal full checkpoint config.

## Run Baseline Pretraining

Smoke test without real pretraining data:

```bash
bash scripts/run_stage0_mae_baselines_smoke.sh
```

Full 0-400 baseline trajectory:

```bash
RUN_TAG=stage0_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_stage0_mae_baselines_400.sh
```

Useful overrides:

```bash
BATCH_SIZE=20 GRAD_ACCUM_STEPS=4 NUM_WORKERS=4 PREFETCH_FACTOR=4 \
bash scripts/run_stage0_mae_baselines_400.sh
```

Resume one run directly:

```bash
python trainers/train_rmae.py \
  --config configs/pretrain/stage0_echonet_echocardmae_400.yaml \
  --resume /root/autodl-tmp/outputs/echonet_echocardmae_repro/<RUN_TAG>/checkpoints/last.pt
```

## Run Recurrent Candidates

```bash
RUN_TAG=<same_or_new_tag> bash scripts/run_stage2_recurrent_mae_400.sh
```

For a quick synthetic smoke:

```bash
bash scripts/run_stage2_recurrent_mae_smoke.sh
```

## One-Shot Research Run

This runs Stage 0 baselines, Stage 2 recurrent candidates, then checkpoint-wise
segmentation evaluation:

```bash
RUN_TAG=research_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_mae_research_400_pipeline.sh
```

Disable expensive checkpoint evaluation during pretraining:

```bash
RUN_CHECKPOINT_EVAL=0 bash scripts/run_mae_research_400_pipeline.sh
```

Skip already completed blocks:

```bash
RUN_STAGE0_BASELINES=0 RUN_STAGE2_RECURRENT=1 RUN_CHECKPOINT_EVAL=0 \
RUN_TAG=<existing_tag> bash scripts/run_mae_research_400_pipeline.sh
```

## Checkpoint-Wise Evaluation

Frozen segmentation probe, default 20 epochs:

```bash
RUN_TAG=<pretrain_tag> EVAL_MODE=frozen \
bash scripts/eval_stage0_checkpoints_seg.sh
```

Full fine-tuning, default 80 epochs:

```bash
RUN_TAG=<pretrain_tag> EVAL_MODE=full \
bash scripts/eval_stage0_checkpoints_seg.sh
```

Evaluate recurrent candidates too:

```bash
RUN_TAG=<pretrain_tag> \
METHODS="echonet:echonet_rvm_mae echonet:echonet_ttt_mae camus:camus_rvm_mae camus:camus_ttt_mae" \
bash scripts/eval_stage0_checkpoints_seg.sh
```

Evaluate only selected checkpoints:

```bash
RUN_TAG=<pretrain_tag> CHECKPOINT_EPOCHS="100 200 400" \
bash scripts/eval_stage0_checkpoints_seg.sh
```

The summary is written to:

```text
/root/autodl-tmp/outputs_stage0_eval_reports/<RUN_TAG>/<EVAL_MODE>/
  stage0_checkpoint_eval.csv
  stage0_checkpoint_eval.md
```

## Promotion Rule

After epoch 400:

- clearly worse than baseline: stop;
- close to baseline but mechanistically meaningful: continue to 600 or 800;
- clearly better: continue to 600, 800, then consider 1200 and 1600;
- final paper candidates: train to 1600 with full downstream evaluation.

To continue a selected run from 400 to 800, keep the same config but override
epochs and resume from `last.pt`:

```bash
python trainers/train_rmae.py \
  --config configs/pretrain/echonet_rvm_mae.yaml \
  --epochs 800 \
  --resume /root/autodl-tmp/outputs/echonet_rvm_mae/<RUN_TAG>/checkpoints/last.pt
```
