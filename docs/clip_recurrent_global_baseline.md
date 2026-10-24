# 干净 Clip-Recurrent-Global 长程基线

## 1. 本轮固定什么

这是新的长程对照轨迹，不是把此前探索模型继续训练。代码复用已验证的时序编码与公共 pipeline，模型选项锁定在独立配方 `configs/pipeline/clip_recurrent_global.yaml`。

| 部分 | 本轮设置 |
| --- | --- |
| 数据 | EchoNet-Dynamic 现有原生 112×112 灰度 uint8 NPY；内存展开三通道，不新建RGB磁盘缓存 |
| 初始编码器 | `ckpt/mae/videomae_vit_s.pth`，VideoMAE-S，384维、12层、6头；原样匹配tubelet2、patch16，不缩放卷积核 |
| 局部上下文 | 连续16帧，stride=1；不做gap采样；片段内非因果，片段间按顺序递归 |
| 长程状态 | 单个全局向量 `[B,1,384]`；可见token均值压缩；既有简单门控Transformer递归核 `RVMCore`，不是空间memory，也不宣称官方RVM逐行复现或标准GRUCell |
| 状态读写 | token侧融合上一个状态；融合后可见特征做均值写入；每个独立窗口从空状态开始 |
| 近期缓存 | 最多4个clip的已融合输出；EF用最近64帧的空间池化序列；更早信息经状态间接影响这些输出，不另加state预测分支 |
| 分割 | 读取指定帧的patch特征；tubelet内两帧直接repeat，明确保留其身份分辨能力受限这个基线缺点；不启用可学习展开 |
| SSL目标 | 每个观察到的clip均重建，包含prefix；tube mask ratio=0.9；按patch标准化的masked MSE；无额外辅助loss |
| 优化 | AdamW，SSL betas=(0.9,0.95)，lr=1e-4，wd=0.05，有效batch=32窗口，40epoch warmup后cosine；1600固定epoch，不早停 |
| 输入标准化 | 固定ImageNet mean/std；所有阶段相同。灰度复制不等于原生彩色；这不是EchoCardMAE完全复现协议 |
| 不带入 | spatial/dual记忆、temporal_attention压缩、低秩限制、频域、frame factorization/query、正交损失、学习式tubelet展开 |

近期FIFO不是第二套编码器。SSL按顺序计算所有clip的重建loss；下游只保留近期输出序列，前缀只经状态传递。窗口训练会保留必要的反向计算图，不能把它与 `no_grad` 在线推理的有限驻留内存混为一谈。

EF线性头是时间均值加Linear；lite/FT头沿用公共 `TemporalEFReadout` 的位置编码、轻量attention pooling与MLP。这是监督读取头，不是把前期 `temporal_pool` 记忆压缩创新带回SSL。分割线性头为逐patch Linear+双线性上采样；lite/FT沿用公共 `ViTPatchSegDecoder`，所有后续创新必须保持相同头来对照。

mask ratio受7×7网格的整数可见patch数约束，配置0.9实际保留4/49个空间位置。patch标准化MSE也不能与此前raw-pixel loss的数值大小直接比较。

## 2. 全数据自监督与标签边界

按本次要求，SSL用官方 **TRAIN+VAL+TEST全部无标签影像**。单独的 `dynamic_all_ssl.json` 只读取 `FileName/Split/FPS`，不加载EF字段，不打开 `VolumeTracings.csv`。清单只含路径、shape、dtype、长度、FPS、原split及文件大小/mtime等；标签字段混入会报错。

这属于 **transductive all-split unlabeled SSL**：VAL/TEST影像在预训练时已经见过。之后只在TRAIN标签上训练任务头或全量微调，在VAL选权重。这些结果可以作为同一数据暴露协议下的内部基线，但不能写成“测试病例影像完全未见”的inductive评估，也不能与只用TRAIN影像预训练的论文数值直接公平排名。

本入口**不做最终TEST标签评估**。如要投稿中的正式TEST结果，应先冻结方案，再独立开启TEST评估；其影像仍属于SSL已见，必须披露。未来创新模型须使用相同SSL人口与监督划分。

`dynamic_train_val.json` 是另一份监督清单，仅TRAIN/VAL，包含EF与官方轮廓；它不参与SSL取样。保留既有GT坐标、病例划分与有效上下文筛查。并非声称每个原始标注都能满足W64及所有目标位置；排除原因会保留。

## 3. 在线取窗，不预构建重复视频样本

1. 一次性扫描CSV及NPY头，建立轻量清单。缺失缓存、错误shape/dtype直接报错，不把缺失数据静默排除后仍称“全数据”。不足16帧者记录排除。
2. 每个epoch，每个可用视频访问一次，按 `seed/epoch/patient` 确定一次新窗口；shuffle的是视频索引。
3. 近期窗口为64帧。16–63帧的短视频使用不超过真实长度的完整clip倍数，不填帧。
4. 随机近期起点；历史前缀从当前可用的 `0,16,...,min(128,起点)` 均匀抽取。因此没有192帧长度门槛，也不保证每次都用最长历史。
5. mmap只读取选中灰度帧，然后在内存归一化、复制三通道；不每次解码AVI、不存重叠窗口。不把视频末尾不足一个clip的部分填充进短视频；长视频起点可任意帧，跨epoch覆盖不同位置。

1600epoch意味着1600次每病例随机访问，不等于1600次把每个视频的所有帧完整看一遍。CSV额外记录每轮真实窗口、帧、clip、prefix分布及窗口序列哈希；结合清单与种子可重建采样。变长上下文下也应比较总帧数与真实运行时间，不只比较epoch。

## 4. 检查点、曲线与后续评测

| 顺序 | 操作 | 是否训练 |
| --- | --- | --- |
| 1 | 全数据SSL1600epoch；每25epoch固定64病例、固定窗口/遮罩重建诊断 | 是，不早停 |
| 2 | 0/100/200/400/800/1200/1600快照的表征几何、有限扰动敏感性与VAL端到端特征延时 | 否 |
| 3 | 0/400/800/1600快照的EF线性读取与分割线性读取 | 仅头；各最多100epoch，VAL每轮检查，patience10 |
| 4 | 最终快照EF/分割轻量非线性读取 | 仅头；相同上限与早停 |
| 5 | 最终快照EF/分割全量FT | 全量；相同上限与早停，训练侧窗口一致增强 |
| 6 | 最终编码器有限流式/机制审计：边界、状态、目标帧、历史干预等 | 否，不新增MAE训练 |

实现中的队列按各快照“表征→对应线性头”穿插执行，最终再lite/FT/审计。**这些下游读取都在1600epoch SSL结束后执行**，不会在SSL中间插入数小时微调。期间仍实时更新CSV/JSONL/状态与TensorBoard，每25epoch更新曲线及固定病例重建图。

重建诊断来自SSL已见人口，字段为兼容公共引擎仍叫 `val_loss`，图例及protocol明确标注 seen-data diagnostic；不作为独立验证或最佳SSL权重依据。patch标准化重建图用GT patch均值/方差反标准化，图旁policy明确说明：只用于可视化，不能拿它证明绝对灰度重建能力。

EF指标含MAE/RMSE/相关性/阈值比例；分割统计病例聚合Dice/全局Dice、IoU及边界（原生像素，没有毫米标定不能称mm）。VAL分割遍历最后clip的16种目标位置，先按源标注帧聚合再按病例聚合，不把重复目标当独立病例。冻结训练禁用增强，特征缓存FP32；FT训练增强整窗一致。此处是统一读取协议，不保证与旧实验不同分割头的绝对数值可直接比较。

表征分析是辅助，不等同于临床功能解释性。流式审计只覆盖有限病例/帧数，不证明所有长视频和临床场景。片段内需要收齐16帧，不能声称零延迟逐帧因果；tubelet repeat的两帧不能宣称分别建模。

### 存储与恢复

```text
/root/autodl-tmp/outputs_cr_clean/
  manifests/                         # 两份清单，含绝对源路径
  result/<RUN_TAG>/<stage>/           # CSV、JSON、日志、曲线、重建、统计、执行参数
  ckpt/<RUN_TAG>/<stage>/             # last、最终/最佳、保留的快照
  cache/<RUN_TAG>/frozen/<identity>/   # 可重建的冻结特征，非近期FIFO
```

- SSL每25epoch覆盖一次完整恢复用 `last.pt`；最多回放未持久化的24个完整epoch以及当前未完成epoch。0/100/200/400/800/1200/1600为固定保留的模型快照；其他周期快照不保留。不承诺batch级原地恢复。
- head/FT每10epoch或VAL改善/早停时保存可恢复last；保留最佳和最终，不永久堆积每一轮。
- 固定快照无优化器状态；完整恢复状态只在last。snapshot与final可能同epoch但各有明确用途。
- 冻结特征总预算24GiB，剩余磁盘保留2GiB；超预算时只在阶段边界清理本RUN_TAG下**已经完整提取完成**的其他冻结缓存，按需可再提取；不删ckpt、结果或未完成缓存。单个任务自身超过预算会报错而非偷偷减病例。
- 如果连单任务缓存也放不下，可显式提高 `runtime.cache_gb` 或设 `evaluation.cache_frozen=false`（更慢）；不是显存。云端存储需同时容纳原数据、权重、缓存与原子写临时空间。
- 同tag重复同命令会复用已校验完成结果并从last恢复；代码/科研参数/数据指纹变化会拒绝原tag，避免混合实验。不要在队列运行中pull。

## 5. 云端指令

只需本仓库、既有灰度NPY及原始VideoMAE权重。先确认环境的PyTorch支持CUDA/bf16，不要在已可用环境盲目重装PyTorch。

```bash
cd /root/autodl-tmp/EchoRVM
conda activate echocardmae
git pull --ff-only

# 一键：两份清单 -> 完整真实模型小样本冒烟 -> 正式SSL -> 全部评测
RUN_TAG=crg1600_allssl_v1 bash scripts/run_clip_recurrent_global.sh all
```

建议用已有tmux/screen环境运行，避免SSH断连。脚本不接管定时关机、不启动巡检自动任务。

```bash
# 可先只冒烟；后续run不会再跑冒烟
RUN_TAG=crg1600_allssl_v1 bash scripts/run_clip_recurrent_global.sh smoke
RUN_TAG=crg1600_allssl_v1 bash scripts/run_clip_recurrent_global.sh run

# 如只想先做1600epochSSL，其他保留日后：
RUN_TAG=crg1600_allssl_v1 bash scripts/run_clip_recurrent_global.sh pretrain
# 之后同tag run会跳过完成的SSL，执行后续读取

# 监控
RUN_TAG=crg1600_allssl_v1 bash scripts/run_clip_recurrent_global.sh status
tail -f /root/autodl-tmp/outputs_cr_clean/result/crg1600_allssl_v1/pretrain/worker.log
tensorboard --logdir /root/autodl-tmp/outputs_cr_clean/result/crg1600_allssl_v1

# 结果包不包含ckpt和特征缓存
RUN_TAG=crg1600_allssl_v1 bash scripts/run_clip_recurrent_global.sh export
```

路径改变时用 `DATA_ROOT=... OUTPUT_ROOT=...` 环境变量；权重路径可用 `--set model.init_checkpoint=...`。复跑保持相同RUN_TAG、路径与参数。需要真正更改科研参数时新建tag。`plan`可在不启动GPU任务时打印完整配置。

每个正式训练阶段会先自动实测micro-batch、workers、gradient-checkpointing，以吞吐为目标，保持有效batch、学习率、病例、采样、梯度目标不变；保留显存安全余量。它不是科研超参数搜索，也不保证GPU百分比始终100%，更不能保证某台机器的理论全局最优。冻结特征提取用独立有界批次与OOM减半，之后调优的是实际头训练，不把特征提取当作FT重复跑。

## 6. 验证范围

本地CPU测试覆盖label-free全split清单、短视频/变长取窗、确定性与spawn、原生模型前后向、快照引用/恢复、norm-pix可视化和监督读取边界。真实初始化文件另做编码器完整性核查。CPU测试不等于4090吞吐或完整1600epoch已验证；云端真实模型冒烟仍是正式启动的必要检查。
