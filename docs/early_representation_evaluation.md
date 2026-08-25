# Early MAE Representation Evaluation

## Purpose

This protocol decides whether an echocardiography MAE should be rejected or
promoted before expensive 1600-epoch pretraining and downstream experiments.

Three 0-400 epoch video trajectories are the reference:

- echonet_echocardmae_official_video: official-style 16-frame EchoCardMAE,
  stride 4, patch 8, tubelet 2, 75% ROI masking, median-denoised targets and
  two-clip bidirectional InfoNCE.
- echonet_videomae_matched: the attribution control with the same frames,
  stride, tokenization, initialization, optimizer, batch and input
  normalization, but without ROI masking, denoising or clip alignment.
- echonet_videomae_clean: the complete standard VideoMAE system baseline with
  16 frames, patch 16, tubelet 2, 90% tube masking and normalized-pixel loss.

The previously generated echonet_echocardmae_repro run used T=1 and is not an
EchoCardMAE reproduction. It may only be retained as a named single-frame
Echo-style ablation.

## Evaluation Layers

Every fixed checkpoint is evaluated without updating the backbone.

| Group | Metric | Interpretation |
| --- | --- | --- |
| reconstruction | reconstruction_loss | Same-method convergence check; targets differ across methods, so do not directly rank methods with it. |
| geometry | feature_std_mean | Average feature variation; near zero indicates collapse. |
| geometry | collapsed_dim_fraction | Fraction of nearly constant dimensions; lower is better. |
| geometry | effective_rank / participation_ratio | Number of dimensions carrying variance. |
| geometry | mean_pairwise_cosine | Very high values plus low rank indicate collapse. |
| invariance | augmentation_cosine | Stability to a fixed clip-consistent ultrasound perturbation. |
| temporal | adjacent_frame_cosine | Local temporal smoothness; an extreme value can also mean temporal insensitivity. |
| temporal | temporal_reverse_delta | Change after reversing time; near zero means order is ignored. |
| temporal | pixel_feature_dynamics_corr | Whether feature changes follow observed cardiac motion. |
| EF frozen probe | ef_ridge_001/010/100pct | Closed-form regression with 1%, 10% and all audit labels. |
| EF non-parametric | ef_knn5 | kNN regression without a learned head. |
| spatial frozen probe | seg_linear_patch_dice | Patch-level LV Dice from one linear layer. |

The report emits probe_score, a within-report z-score average of 10%-label EF
ridge, EF kNN and patch-linear segmentation. It is a screening summary only.
A candidate should improve several components at multiple checkpoints.

At epoch 400, full fine-tuning remains an anchor for EchoNet segmentation,
EchoNet EF and CAMUS segmentation when CAMUS is available. Full fine-tuning is
not the early discriminator because it can overwrite pretraining differences.

## Checkpoint Schedule

Default: 50, 100, 150, 200, 250, 300, 350, 400.

Use the same schedule, seed, sample limits and probe optimization for every
candidate. Do not tune probes separately for one method.

## Smoke Test

~~~bash
cd /root/autodl-tmp/MAE/EchoRVM
git pull
bash scripts/run_mae_baseline_representation_smoke.sh
~~~

Successful logs must show official EchoCardMAE at frames=16/patch=8/tubelet=2 and both VideoMAE controls at frames=16.

~~~text
name=echo_videomae family=videomae_clean_video frames=16 patch=16 tubelet=2
~~~

The official EchoCardMAE first batch must be [B,16,3,112,112] and include video_view2. Any EchoCardMAE or VideoMAE baseline with T=1 is invalid.

## Full 400-Epoch Baselines

~~~bash
RUN_TAG=baseline400_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_mae_baseline_representation_400.sh
~~~

Order:

~~~text
EchoCardMAE 0-400
-> VideoMAE 0-400
-> frozen audit at every checkpoint
-> epoch-400 full downstream anchors
-> CSV and Markdown summaries
~~~

Run pretraining and low-cost audit without full fine-tuning:

~~~bash
RUN_FULL_FINETUNE_400=0 \
RUN_TAG=baseline400_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_mae_baseline_representation_400.sh
~~~

Continue from stage 2 after a compatible EchoCardMAE stage:

~~~bash
RUN_TAG=<existing_tag> START_STAGE=2 \
bash scripts/run_mae_baseline_representation_400.sh
~~~

Do not reuse the old `echonet_echocardmae_repro` output. A valid run is named `echonet_echocardmae_official_video` and logs frames=16, patch=8, tubelet=2, stride=4 and two_views=true.

## Re-run Only the Audit

~~~bash
RUN_TAG=<pretrain_tag> \
bash scripts/eval_representation_checkpoints.sh
~~~

Bound the audit cost:

~~~bash
RUN_TAG=<pretrain_tag> \
CHECKPOINT_EPOCHS="50 100 200 400" \
EF_TRAIN_SAMPLES=1000 EF_VAL_SAMPLES=256 \
SEG_TRAIN_SAMPLES=256 SEG_VAL_SAMPLES=128 SEG_STEPS=100 \
bash scripts/eval_representation_checkpoints.sh
~~~

Resume an interrupted audit:

~~~bash
RUN_TAG=<pretrain_tag> START_INDEX=7 \
bash scripts/eval_representation_checkpoints.sh
~~~

## Outputs

~~~text
/root/autodl-tmp/outputs/<METHOD>/<RUN_TAG>/checkpoints/
/root/autodl-tmp/outputs_representation_audit/<RUN_TAG>/<METHOD>/epoch_XXXX/
/root/autodl-tmp/outputs_representation_reports/<RUN_TAG>/
  representation_quality.csv
  representation_quality.md
  epoch0400_full_downstream.csv
  epoch0400_full_downstream.md
~~~

## Promotion Rule

- Stop when frozen EF and spatial probes remain below both baselines, collapse
  indicators worsen and temporal metrics do not support the mechanism.
- Continue to 400 when at least two probe families improve at two or more
  checkpoints.
- Promote beyond 400 only when the gain is stable and matches the claimed
  mechanism, then run the full anchors.
- Reserve 800/1600 epochs for promoted methods, not every small attempt.
