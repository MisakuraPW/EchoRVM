# 心超 RVM-MAE / TTT-MAE 模型与训练说明

## 1. 当前实现范围

本阶段实现的是单数据集心超 Recurrent MAE 预训练骨架，先不接入复杂数据层。

训练代码要求 dataloader 输出：

```python
{"video": Tensor[B, T, 1, 112, 112]}
```

`trainers/train_rmae.py` 的正式配置会读取真实 EchoNet/CAMUS 数据。synthetic video batch 只用于 debug smoke test，用来验证模型、优化器、日志、checkpoint、loss 曲线和 resume 流程。

## 2. 模型结构

### EchoRVM-MAE

EchoRVM-MAE = EchoCardMAE-style 单帧 MAE + RVM recurrent core。

每帧流程：

```text
[B,1,112,112]
  -> patchify / patch embed, patch=8
  -> 196 tokens, dim=384
  -> ViT frame encoder
  -> RVMCore 更新 state
  -> MAE decoder 重建 masked patches
```

保留的 EchoCardMAE trick：

```text
112x112 / patch8
ROI/key-area mask 接口
mask token
background token
median-blur target 接口预留
temporal InfoNCE alignment loss 接口预留
```

RVMCore 使用 GRU-style gates：

```text
update gate: 控制保留旧 state 还是写入新信息
reset gate: 控制 cross-attention 读取旧 state 的强度
transformer integration: 当前帧 tokens query，旧 state 作为 context
```

### EchoTTT-MAE

EchoTTT-MAE 与 EchoRVM-MAE 共享 frame encoder、decoder、masking 和 loss，只把 temporal core 换成 `TTTCore`。

第一版 `TTTCore` 是轻量 fast-state 实现，不直接依赖 Spatial-TTT/TTT-LM 的大模型代码。它用当前帧 tokens 生成 key/value/query，通过局部 self-supervised prediction error 更新 fast state，再输出 refined temporal state。

## 3. 优化器

默认优化器是 `muon_adamw_hybrid`，不是纯 AdamW。

参数分组规则：

```text
Muon:
  encoder / decoder / rvm / ttt 中 ndim >= 2 的 hidden matrix weights

AdamW:
  bias
  LayerNorm
  pos_embed
  mask_token
  background_token
  head
  小标量参数
```

训练开始时日志会打印：

```text
muon_tensors / muon_params
adam_tensors / adam_params
```

如果训练不稳定，可临时把配置里的：

```yaml
optimizer:
  name: AdamW
```

用于排查，但正式主实验默认用 Muon+AdamW hybrid。

## 4. 运行命令

### Debug smoke test

```bash
python trainers/train_rmae.py \
  --config configs/mae_debug.yaml \
  --debug \
  --max_steps 2
```

或：

```bash
bash scripts/run_debug.sh
```

### EchoNet RVM-MAE

```bash
bash scripts/run_echonet_rvm_mae.sh
```

### CAMUS RVM-MAE

```bash
bash scripts/run_camus_rvm_mae.sh
```

### EchoNet TTT-MAE

```bash
bash scripts/run_echonet_ttt_mae.sh
```

### CAMUS TTT-MAE

```bash
bash scripts/run_camus_ttt_mae.sh
```

### 指定输出目录

```bash
python trainers/train_rmae.py \
  --config configs/pretrain/echonet_rvm_mae.yaml \
  --output_dir /root/autodl-tmp/outputs/echonet_rvm_test
```

### 断点续训

```bash
python trainers/train_rmae.py \
  --config configs/pretrain/echonet_rvm_mae.yaml \
  --resume /root/autodl-tmp/outputs/xxx/checkpoints/last.pt
```

checkpoint 会恢复：

```text
model
optimizer
scheduler
AMP GradScaler
epoch
global_step
best_metric
random states
```

## 5. 输出目录

每次训练会生成：

```text
run_dir/
  config.yaml
  config_source.yaml
  logs/
    train.log
    train_metrics.jsonl
    val_metrics.jsonl
    metrics.csv
    summary.json
  checkpoints/
    last.pt
    best.pt
    epoch_XXX.pt
    interrupt.pt
  plots/
    loss_latest.png
    loss_epoch_XXX.png
  tensorboard/
```

查看最新 loss 曲线：

```bash
ls run_dir/plots/loss_latest.png
```

查看日志：

```bash
tail -f run_dir/logs/train.log
```

启动 TensorBoard：

```bash
tensorboard --logdir run_dir/tensorboard --port 6006
```

## 6. 防过拟合与稳定训练

当前已实现：

```text
weight decay
DropPath 配置
gradient clipping
mixed precision GradScaler
best.pt
early stopping
last.pt / interrupt.pt resume
```

早停配置：

```yaml
early_stopping:
  enabled: true
  patience: 20
  min_delta: 1.0e-4
```

如果没有验证集，建议关闭 early stopping：

```yaml
early_stopping:
  enabled: false
```

## 7. CPU/GPU 利用率调参

训练效率主要由两类瓶颈决定：

```text
1. GPU 算得慢或显存不够
2. CPU/DataLoader 喂数据慢，GPU 在等数据
```

### 需要优先调的参数

GPU 相关：

```yaml
train:
  batch_size: 16
  grad_accum_steps: 1
  mixed_precision: true
  torch_compile: false
  gradient_checkpointing: false
  clip_grad_norm: 1.0

model:
  frames: 16
  embed_dim: 384
  depth: 12
  decoder_depth: 4
  core_depth: 2
  mask_ratio: 0.75
  ttt_inner_steps: 1
```

DataLoader 相关：

```yaml
data:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  drop_last: true
```

### 如何观察

看 GPU：

```bash
watch -n 1 nvidia-smi
```

或更细：

```bash
nvidia-smi dmon
```

重点看：

```text
GPU-Util: 是否长期接近 80%-100%
Memory-Usage: 显存是否还有大量空余
Power: 功耗是否接近该卡正常训练功耗
```

看 CPU：

```bash
top
```

或：

```bash
htop
```

重点看：

```text
CPU worker 是否吃满
python DataLoader worker 是否持续工作
是否有大量 IO wait
```

训练进度条也会显示：

```text
data_time
forward_time
backward_time
step_time
gpu memory allocated/reserved
```

### 常见情况与处理

GPU 利用率低，`data_time` 高：

```text
优先增加 num_workers，例如 4 -> 8
打开 persistent_workers
增加 prefetch_factor，例如 2 -> 4
确认 pin_memory=true
如果数据在慢盘，考虑预处理/缓存到 /root/autodl-tmp
```

GPU 利用率低，但 `data_time` 不高：

```text
batch_size 可能太小
frames 可能太少
模型太小或 CPU/GPU 同步太频繁
尝试增大 batch_size 或 frames
```

显存 OOM：

```text
先降低 batch_size
再提高 grad_accum_steps 保持等效 batch
再考虑降低 frames
最后才降低 depth/embed_dim/core_depth
```

显存空余很多：

```text
优先增大 batch_size
如果 batch_size 增不上去，再增大 frames
不要一开始就盲目增大模型深度
```

TTT 比 RVM 慢：

```text
先保持 ttt_inner_steps=1
如果需要更强 TTT，再试 2
不要一开始开很大的 inner steps
```

想让吞吐更稳定：

```text
drop_last=true
固定 img_size=112
固定 frames
避免每个 batch 内变长视频直接拼接
真实数据层最好提前缓存成统一 clip
```

## 8. 当前限制

```text
真实 EchoRisk loader 尚未接入
EchoRisk、多数据集联合预训练未接入
下游 EF/分割/风险 head 未接入
future reconstruction/future latent prediction 有接口但默认关闭
median-blur target 接口预留，当前 synthetic smoke test 不启用真实图像滤波
```

下一步接数据层时，只要 dataloader 返回 `{"video": [B,T,1,112,112]}`，当前训练代码即可复用。

## 9. 在线数据增强 A4

正式 MAE 预训练默认使用在线增强，不预先生成固定增强数据集。当前实现复用：

```python
from augment.ultrasound import EchoAugmentConfig, EchoClipAugmentor
from echo_aug_validation.augment_recipes import augment_video, augment_image_mask
```

训练入口中新增了 `utils/augmentation.py`，会把 dataset 返回的单个 clip 从 `[T,1,H,W]` tensor 转成增强代码需要的 `[T,H,W,C]`，增强后再转回 `[T,1,112,112]`。增强只在 `train` split 启用，`val` split 不增强。

注意：`trainers/train_rmae.py` 里的 synthetic dataset 只用于 smoke test。正式 EchoNet/CAMUS 预训练配置会读取真实数据，不会静默使用随机 tensor。只有 `--debug` 或 `data.loader: synthetic` 才允许跑 synthetic 数据。

默认 preset 是 `A4_tgc_zoom_speckle`：

```yaml
augment:
  enabled: true
  preset: A4_tgc_zoom_speckle
  per_frame_random: false
  img_size: 112
  tgc_prob: 0.4
  gamma_contrast_prob: 0.5
  brightness_prob: 0.3
  zoom_prob: 0.3
  blur_prob: 0.2
  speckle_prob: 0.4
  shadow_prob: 0.0
  tgc_min: 0.75
  tgc_max: 1.35
  gamma_min: 0.75
  gamma_max: 1.35
  contrast_min: 0.85
  contrast_max: 1.20
  brightness_delta: 0.08
  zoom_min: 0.92
  zoom_max: 1.12
  speckle_sigma_min: 0.03
  speckle_sigma_max: 0.10
```

`per_frame_random: false` 很重要：同一个 clip 内 TGC、gamma、zoom、blur 等参数保持一致，避免破坏心动周期时序。

EchoNet clip 推荐流程：

```text
[T,H,W] 或 [T,H,W,C]
  -> dataset 采样 clip
  -> 转成 [T,1,H,W] float32
  -> AugmentedVideoDataset 在线 A4 增强
  -> batch [B,T,1,112,112]
```

CAMUS 用于 MAE 预训练时不需要 mask。可以把单帧扩展成短 clip，或重复成 `T` 帧后再走同一套 A4 image-only 增强。如果以后做分割下游，image/mask 要用 `augment_image_mask` 保证 zoom 对齐。

关闭增强只需要：

```yaml
augment:
  enabled: false
```

训练日志会打印当前增强设置，例如：

```text
augment enabled preset=A4_tgc_zoom_speckle per_frame_random=False ...
```

效率调节时按这个规则判断：

```text
GPU 利用率低、CPU 也低：
  增加 num_workers / prefetch_factor

GPU 利用率低、CPU 高：
  不要继续加 worker
  优先把数据从 autodl-fs 复制到 autodl-tmp
  或减少在线增强复杂度

显存低、GPU 有空：
  增大 batch_size

OOM：
  降低 batch_size
  或增加 grad_accum_steps
```
