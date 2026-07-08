# Downstream Fine-Tuning

本阶段把四个 MAE 预训练 `best.pt` 分别导入三类下游任务，做全量微调：

- EchoNet-Dynamic LV segmentation
- EchoNet-Dynamic EF regression
- CAMUS segmentation

总计 12 组：`echonet_rvm_mae`、`echonet_ttt_mae`、`camus_rvm_mae`、`camus_ttt_mae` 各跑三项任务。

## 一键运行

默认先跑 Echo 分割四组，再跑 Echo EF 四组，最后跑 CAMUS 分割四组：

```bash
bash scripts/run_all_downstream.sh
```

先只跑 Echo 分割四组：

```bash
bash scripts/run_downstream_echonet_seg.sh
```

默认权重路径会自动从这里找最新 run：

```text
/root/autodl-tmp/outputs/<pretrain_name>/<latest_run>/checkpoints/best.pt
```

如果要固定某个预训练 tag：

```bash
PRETRAIN_TAG=20260707_093341_bs16 bash scripts/run_downstream_echonet_seg.sh
```

默认下游参数：

- EchoNet segmentation: `batch_size=64`, `lr=1e-4`
- CAMUS segmentation: `batch_size=64`, `lr=1e-4`
- EchoNet EF: `batch_size=4`, `grad_accum_steps=3`, `frames=32`, `lr=5e-5`

常用覆盖参数：

```bash
BATCH_SIZE=24 NUM_WORKERS=6 LR=3e-5 EPOCHS=80 bash scripts/run_downstream_echonet_seg.sh
```

EF 默认 `frames=32`，显存不够时：

```bash
FRAMES=16 BATCH_SIZE=8 bash scripts/run_downstream_echonet_ef.sh
```

输出目录：

```text
/root/autodl-tmp/outputs_downstream/<task>/<DOWNSTREAM_TAG>/<init_name>/
```

每个 run 内含：

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `logs/metrics.csv`
- `logs/summary.json`
- `plots/loss_latest.png`
- `tensorboard/`

## 模型头

分割头：`EchoSegFineTuner`

- 输入 `[B,1,112,112]`
- 把单帧扩成 `T=1`
- 使用 MAE frame encoder 和 RVM/TTT temporal core 输出 14x14 tokens
- ConvTranspose decoder 上采样到 `[B,C,112,112]`
- Loss 为 CE + Dice
- 监控 `dice_mean`

EF 头：`EchoEFFineTuner`

- 输入 `[B,T,1,112,112]`
- 使用 MAE frame encoder + RVM/TTT core 得到 recurrent states
- temporal query head 学习 ED/ES-like pooling
- SmoothL1 训练
- 监控 `mae`

## 数据

EchoNet EF：

- `FileList.csv` 读取 `EF`
- 视频优先读 `.npy`，找不到再读 `.avi`

EchoNet 分割：

- `VolumeTracings.csv` 读取 `FileName, Frame, X1, Y1, X2, Y2`
- 对每个视频的 ED/ES tracing 栅格化 LV mask

CAMUS 分割：

- 读取 ED/ES image 和 `_gt` mask
- train/val 按 patient key 确定性切分

## 增强

三项下游默认继续使用 A4：

- TGC
- gamma/contrast
- brightness
- small zoom
- light blur
- speckle noise
- no shadow
- no flip/rotation/CutMix/MixUp/elastic

分割任务会对 image/mask 使用同一组几何参数。

## 建议顺序

先跑：

```bash
bash scripts/run_downstream_echonet_seg.sh
```

看到四组 dice 正常上涨后，再跑：

```bash
bash scripts/run_downstream_echonet_ef.sh
bash scripts/run_downstream_camus_seg.sh
```
