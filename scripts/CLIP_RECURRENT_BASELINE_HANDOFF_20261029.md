# Clip-Recurrent B0 基线与本轮队列交接说明

整理日期：2026-10-29。面向使用同一代码库的协作者。

> 本文依据仓库源码和已下载的运行记录整理。本次侧会话的 SSH 查询因域名解析失败，未刷新实时进度。下文版本、配置及耗时为最近已核实记录，不代表此刻已经完成的训练轮数。本文不修改或接管主会话中的服务器任务。

## 1. 对应哪一版

| 项目 | 内容 |
| --- | --- |
| 仓库 | `MisakuraPW/EchoRVM` |
| 训练代码提交 | `d7a6b2b7985bbbafa42006e45c8ee41f0abe30c6` |
| 后续说明提交 | `2aedff9`，仅补充实测说明，未改变训练代码 |
| 正式运行标签 | `cr_forward_fast_v3` |
| 主启动入口 | `scripts/run_clip_recurrent_fast.sh` |
| 队列入口 | `research_pipeline/forward_suite.py` |
| 基础配置 | `configs/pipeline/clip_recurrent_global.yaml` |
| 模型实现 | `models/final_temporal_mae.py`、`models/temporal_mae.py`、`models/rvm_core.py` |
| 训练循环 / 数据适配 | `research_pipeline/engine.py`、`research_pipeline/echo.py`、`research_pipeline/ssl_data.py` |

**实际设置是“基础 YAML + 启动覆盖 + 自动校准结果”，不能只看 YAML。** YAML 默认有效 batch 为32，但本轮 SSL 已覆盖为64。服务器运行目录内的 `plan.json`、各阶段 `protocol.json`、`runtime.json` 是实际配置和来源记录的主要依据。

旧标签 `cr_forward_v1`、`cr_forward_fast_v2` 是优化过程中保留的历史运行，不要将其权重或指标混入本轮正式基线。`cr_forward_fast_v3` 从声明的初始化重新开始，而非续接旧标签的若干轮。

## 2. B0 模型定义

这是本项目的 **Clip-Recurrent MAE**：局部 VideoMAE 编码短片段，跨片段以单个全局向量递归传递历史，下游保留有界近期特征缓存。它不是 EchoCardMAE 官方复现，也不是直接使用官方 Recurrent Video Matting 网络。

| 模块 | 设置 |
| --- | --- |
| 初始化文件 | `ckpt/mae/videomae_vit_s.pth`；具体载入张量及文件哈希看运行来源记录 |
| 局部编码器 | ViT-S，384维，12层，6头 |
| 输入 | 原生112×112；灰度缓存读取后在内存中复制为3通道 |
| 空间 patch / 时间 tubelet | 16×16 / 2帧 |
| 局部 clip | 连续16帧，clip之间按时间先后递归处理 |
| 历史状态 | 一个384维全局向量，形状 `[B,1,384]`，无空间对应槽位 |
| 递归核心 | 本项目 GRU式更新/重置门 + Transformer integration，core depth=1 |
| B0 历史读取 | 当前 clip 经过完整编码器后，再用旧状态进行特征融合 |
| 状态写入 | 融合后的可见 token 取均值，再经递归核心更新状态 |
| 近期缓存 | 下游保留近期最多64帧对应的特征，不只依赖最终状态预测 |
| 帧展开 | tubelet特征重复到对应两帧，`frame_readout=repeat` |
| MAE 解码器 | 192维，4层，3头，可读取旧状态 |
| 位置编码 | flat sinusoid；separate Q/V bias；LayerNorm eps=1e-6 |

不加入：spatial/dual memory、频域分支、低秩状态约束、可学习帧展开、可学习时间压缩、内容自适应写入等候选创新。

### 表征与任务接口

- EF：保留近期片段输出，将每帧空间 token 平均成时间描述符，再进行时间聚合。默认不额外拼接独立边界状态 token；历史已通过递归读取影响近期特征。
- 分割：取目标帧对应的空间 token 图，不使用整个视频的单一全局向量代替空间特征。
- 冻结线性 EF 头：时间描述符均值后接线性层。冻结线性分割头：逐空间 token 线性分类后插值。
- 轻量/全量任务头：EF 使用 `TemporalEFReadout` 学习式注意力聚合；分割使用 `ViTPatchSegDecoder`。
- 当前 tubelet=2 且 repeat 展开，相邻两帧共享展开前的 tubelet 表征；不能声称具有独立的可学习逐帧展开。

## 3. 数据与样本组织

数据根目录：`/root/autodl-tmp/datasets/EchoNet-Dynamic`。

清单：

```text
/root/autodl-tmp/outputs_cr_clean/manifests/dynamic_all_ssl.json
/root/autodl-tmp/outputs_cr_clean/manifests/dynamic_train_val.json
```

- SSL 使用官方 TRAIN、VAL、TEST 的无标签视频，清单中有效视频为10030个；不使用 EF 或分割标签来训练 MAE。
- 复用灰度 NPY 缓存，读取选中窗口，不另外生成 RGB 磁盘缓存。
- 每个 epoch 每个视频在线随机取一个窗口，再随机打乱。不是提前穷举所有滑窗，也不是每轮固定读取同一个片段。
- 每个窗口由近期最多64帧及其之前的0–128帧前缀组成；前缀以16帧为单位，且不能超出真实可用历史。
- 视频不足64帧时，可以取更短的完整 clip 窗口。禁止补造历史帧。正常完整窗口最长192帧，并非每条样本都要求192帧。
- 采样由病例、seed、epoch确定；掩码也按记录索引和epoch确定。窗口内连续采样，stride=1。
- 每个独立窗口从空状态开始。窗口内的状态递归保留梯度，不跨打乱后的独立样本延续患者状态。

**数据暴露口径：`transductive_all_splits_unlabeled`。** 下游使用 TRAIN 标签拟合、VAL 标签选择模型，TEST 标签未用于当前适配器拟合或选择；但 SSL 已见过这些 split 的无标签图像。因此不能将本轮描述为“预训练完全没见过测试图像”的 inductive 设定。

## 4. SSL 训练设置

| 参数 | B0 实际设置 |
| --- | --- |
| 预算 | 1600 epoch，固定预算，不早停 |
| 优化器 | AdamW，betas=(0.9,0.95)，weight decay=0.05 |
| 学习率 | 1e-4；40轮 warmup，随后 cosine decay |
| 有效 batch | 64 |
| 本机实际 micro-batch | 32，梯度累积2次；尾批按实际样本数处理 |
| 精度 | BF16 |
| 梯度裁剪 | norm=1.0 |
| 随机 seed | 42 |
| 掩码 | tube masking，mask ratio=0.9 |
| 损失 | 归一化输入对应的 tubelet patch，经逐patch像素标准化后计算 masked MSE |
| 重建范围 | 全部真实观察窗口，包含前缀，不只是近期64帧 |
| 数据增强 | SSL关闭 |
| 输入归一化 | ImageNet mean=(0.485,0.456,0.406)，std=(0.229,0.224,0.225) |
| 重建诊断 | 每25轮；固定64例已暴露人群诊断，不用于选择SSL权重或早停 |

有效 batch 从旧版本32改为64，未自动放大学习率。这会改变每轮优化器更新次数，**不能宣称与旧batch32轨迹等价**。本轮 B0 与 A 使用相同的新有效 batch 和学习率日程。

## 5. 本次队列不只有 B0

默认顺序：**B0预训练 → B0评估 → D诊断 → A候选训练与评估**，共33个已选择阶段。以下为排队内容，不代表均已完成。

### B0 训练与评估

1. B0完成1600轮SSL。
2. 在0、100、200、400、800、1200、1600轮做所配置的表征分析。
3. 在0、400、800、1600轮检查点训练EF与分割冻结线性头。
4. 最终权重做EF/分割轻量冻结头、全量微调、流式检查和机制检查。

时间线评估在B0预训练完成后，读取保留的检查点执行，不是每隔若干轮暂停SSL去训练下游头。

### D 诊断与 A 候选

- D包含400轮检查点的 local/final 出口小规模线性读出比较、历史路径及梯度诊断、可见性/因果性检查；不额外训练新的MAE。
- A的单一改动：将历史读取移到编码器第6层之后，即 `forward_variant=mid_read`、`read_after_block=6`。
- A从B0的epoch-0共同初始化复制共享张量，**不是从B0训练400轮或1600轮后的权重起跑**。
- A沿同一个1600轮学习率日程先训练到400轮，再做配置中的线性评估、诊断及流式检查。
- 默认不运行W内容自适应写入候选，不自动把A扩展至800或1600轮；A最终全量任务确认也不在默认自动扩展范围内。

## 6. 下游默认设置

- 任务为 EchoNet EF 和 EchoNet 左室分割；这份队列没有 CAMUS 任务。
- 监督头默认最多100轮、有效batch32、LR=1e-4、AdamW betas=(0.9,0.999)、weight decay=0.05、5轮warmup。
- 每轮验证；早停 patience=10次验证，min_delta=1e-4。具体阶段合并配置仍以其 `protocol.json` 为准。
- EF拟合目标为EF/100，使用smooth-L1；MAE等评价值转换为EF百分点。
- 分割损失为逐样本CE + 前景soft Dice；输出病例/源帧聚合及global Dice等指标。
- 全量微调启用 `A4_tgc_zoom_speckle`，同一窗口内增强一致；冻结读出不使用这套随机增强，特征可缓存复用。
- SSL专用的batch/运行参数覆盖不会把下游头的有效batch一起改成64，也不会把任务端安全搜索限制成只测大batch。

## 7. 加速实现与实测

已采用：不同长度窗口按真实活跃clip合批；固定可见token索引的gather/scatter；窗口从DataLoader固定内存搬到GPU后在设备端拼片段；全局mean memory避免无必要的稠密零散射。

这些执行改动通过损失、参数梯度和优化器更新对照，但浮点归约顺序不同，不保证逐位一致。未启用 `torch.compile`；一次限时240秒的探索未在预算内完成，不能将它算作已验证的生产优化。

- 硬件记录：单卡RTX4090；PyTorch 2.5.1+cu124；CPU线程4，DataLoader workers=2。
- 独立探测曾测得物理batch64约140窗口/秒，最长192帧窗口压力测试峰值约20.64GiB。
- 在实际父进程/worker环境中，物理batch64未满足10%显存余量标准，因此运行时自动选微批32×累积2次。不要把短测140窗口/秒写成正式训练速度。
- 本轮前三个完整epoch为105.76、98.58、102.16秒，每轮10030个窗口。最初慢版本约921–949秒/轮。
- GPU利用率和显存瞬时值不是吞吐指标；应同时看整轮耗时、窗口/帧数及采样校验值。

## 8. 保存与续训

```text
/root/autodl-tmp/outputs_cr_clean/
├── result/cr_forward_fast_v3/
│   ├── plan.json
│   ├── execution_selection.json
│   ├── batch_search/
│   └── pretrain/
│       ├── protocol.json
│       ├── runtime.json
│       ├── status.json
│       ├── metrics.csv
│       ├── worker.log
│       └── ...
└── ckpt/cr_forward_fast_v3/
    └── pretrain/
        ├── last.pt
        ├── epoch_0000.pt
        └── ...
```

- `last.pt`每轮原子覆盖，包含模型、优化器、scheduler、AMP、RNG及进度，用于epoch边界恢复。
- B0固定保留0、100、200、400、800、1200、1600轮权重。周期快照间隔25轮；未被固定保留的周期快照按清理规则删除，不会积累1600份。
- A的400轮预算边界会保留恢复状态；其日程仍是1600轮，不是重新压缩成400轮cosine。
- 运行与恢复会检查代码、数据和协议身份，不能随意替换参数后继续写入同一个标签。

## 9. 协作注意

1. 核查当前运行时，以服务器 `plan.json`、`protocol.json`、`runtime.json` 和 `execution_selection.json` 为准，不只依赖本文或基础YAML。
2. `run_clip_recurrent_fast.sh all`会校准有效batch并启动队列；换硬件重新校准可能选出不同有效batch，因此不自动构成与本轮完全相同的科学配置。
3. 不要为查看进度重复启动已有队列，也不要直接把新模型/训练代码pull进运行中的实例；代码身份改变会影响后续阶段或恢复。
4. 结果和权重分开保存；下载分析时优先取 `result/cr_forward_fast_v3`，无需默认下载 `ckpt`。
5. 当前应用接口是clip级流式递归：clip之间不重算全部历史；clip内部仍双向注意力。不能宣称每帧零等待或严格逐帧因果。
6. 有界推理状态/近期缓存不等于训练显存与窗口长度无关；本轮有限窗口内仍做递归梯度传播。

只读查看进度：

```bash
tail -f /root/autodl-tmp/cr_forward_fast_v3.log
# 或在仓库目录内：
python tools/watch_pipeline_progress.py --tag cr_forward_fast_v3
```

补充执行优化说明：`scripts/MAE_PERFORMANCE_REPAIR.md`。本文不替代实验结果报告，也不宣称已证明该结构优于其他MAE。
