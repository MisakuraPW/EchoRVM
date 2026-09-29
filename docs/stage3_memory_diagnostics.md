# 第三阶段：先诊断状态，再选择新训练

本轮实现的是已经确认的“诊断优先”部分：复用四组旧权重，不新增 MAE 预训练，不自动全量微调，不自动展开压缩/更新结构组合。下一轮候选取决于这轮诊断，不预先把假设写成既定结论。

## 1. 这次回答什么

- 固定近期窗口后，更早历史是否带来额外收益？
- 无记忆分段缓存、global、spatial、dual，在统一的新读出下分别如何？
- 信息是否在压缩、状态更新或融合过程中变得不容易被读取？
- 重复片段或一次坏片段是否改变状态？后续正常片段能否恢复？
- 最后一个 clip 的掩码重建损失，是否存在通向更早输入的梯度路径？

**不把状态单独预测 EF 的能力当作选型目标。**最终输出/近期缓存的 EF 和目标帧分割才是任务指标；状态统计和瓶颈 ridge 只解释机制。

## 2. 默认流程和预算

依次处理 `clip_mae_pool64`、`hier_global`、`hier_spatial`、`hier_dual`，使用原始 `epoch_0400.pt`。不会把一阶段的100轮权重混进400轮对比。

每组只做一次终点评估：

| 项目 | 默认设置 |
|---|---|
| 近期窗口 | 64帧，局部长度从权重读取，旧四组为16帧 |
| 历史长度 | 0 / 64 / 128帧 |
| EF候选患者 | 训练256、验证128；再统一排除不足192个真实连续帧者 |
| EF头 | 固定结构的小注意力头，200步，训练病例标准化 |
| 分割 | 训练/验证各32患者，原生窗口末尾两个目标位置，线性头100步 |
| 压缩观测 | 当前局部表示、融合表示、压缩表示、更新状态，均汇聚成D维后用相同ridge诊断 |
| 输入梯度 | 单个原生训练窗口；最后clip的masked loss对各clip输入求导，目标detach |
| 新MAE训练 | 0轮 |

不同历史长度从**同一次抽取的最大窗口截取后缀**，近期64帧及病例完全一致。默认同时排除短视频，不能用重复帧假装真实历史。排除情况写入 `excluded_train/val.csv`；这种长视频筛选偏差必须保留在结果解释中。

每个干净历史长度独立拟合相同结构/相同初始化的EF头。这与干扰测试不同：重复整个历史、将最后一个历史clip置零、近期中央区域遮挡，都使用对应干净条件训练好的头，**不重新适配干扰**。干扰只是人工分布变化，不代表真实生理变化，也不生成新的EF标签。

`local_history` 与 `local_empty` 使用相同、未经记忆融合的近期局部表示，附加真历史状态/同形状零槽位；避免所谓无历史对照已经含有历史融合信息。无记忆模型本身正常处理各局部clip，不伪造递归状态。

分割仍使用原生64帧窗口、目标位置62/63，保留真实padding元信息，避免旧评估目标落在状态重置处。它不是192帧历史长度实验，不能据此声称长历史改善分割。

## 3. 开机运行

```bash
cd /root/autodl-tmp/EchoRVM
git pull
conda activate echocardmae

# 只列计划，不读取大权重，不启动计算
bash scripts/run_stage3_memory.sh --dry_run

# 先用真实数据和旧权重验证完整接口，结果自动加 smoke_ 前缀
bash scripts/run_stage3_memory.sh --smoke

# 正式诊断，无新增MAE训练
RUN_TAG=stage3_memory_20260929 bash scripts/run_stage3_memory.sh
```

默认权重目录：

```text
/root/autodl-tmp/outputs_temporal/temporal_gray_20260909/ckpt/
  clip_mae_pool64/epoch_0400.pt
  hier_global/epoch_0400.pt
  hier_spatial/epoch_0400.pt
  hier_dual/epoch_0400.pt
```

正式运行前会检查**所有**要求的权重和数据文件；不存在就报具体路径，不跳过、不随机初始化、不自动训练。若归档位置改了，用 `--checkpoint_root /实际/ckpt`；不规则布局可用 `--checkpoints_json mapping.json`，内容为四个实验名到完整权重路径的JSON对象。队列会核实实际epoch和memory类型，不只相信文件名。

只跑一组或暂不测分割：

```bash
RUN_TAG=stage3_spatial_only bash scripts/run_stage3_memory.sh --only hier_spatial
RUN_TAG=stage3_ef_only bash scripts/run_stage3_memory.sh --no-with_seg
```

改历史长度、种子或任务开关请使用新run_tag。不能用同一run_tag追加不同科研协议；同一命令重跑则复用已完成的结果。

## 4. 自动调参、监控和恢复

- 默认每个模型分别搜索安全推理batch，以吞吐量选择，保留约15%显存余量；同时测EF流式路径和分割路径，避免只测流式路径导致分割OOM。
- 默认在0/2/4/8（不超过`--num_workers`）中做短暂的预热后数据加载吞吐测试。它是有限采样的建议，不保证100%占满机器，可能受缓存预热影响。
- 只调运行参数，不改帧数、有效病例、训练步数或学习率。这里训练的只是小探针，不能为了占满GPU任意扩大科学预算。
- `--batch_size 4 --no-auto_workers --num_workers 4` 可关闭搜索并显式指定。机器同时运行别的项目时，显存空闲量会影响结果；正式计算途中如果其他项目新占用显存，仍可能OOM。
- 默认采用原灰度缓存和权重记录的输入协议，不生成RGB磁盘缓存。
- EF特征提取按split缓存，完成split后原子写入；中断重跑复用完整split。探针没有巨大checkpoint，中断时最多重算该模型的小头，完成模型直接跳过。代码/权重/数据/科学配置改变时拒绝复用，要求新run_tag。

```bash
# 当前大阶段
cat /root/autodl-tmp/outputs_stage3/stage3_memory_20260929/result/current_stage.json

# 每个模型细分进度
cat /root/autodl-tmp/outputs_stage3/stage3_memory_20260929/result/hier_spatial/status.json

# 机器利用率
nvidia-smi dmon
```

## 5. 存储和需要下载的内容

```text
outputs_stage3/<run_tag>/
  result/                    # 只下载这里的analysis.zip
    comparison.csv/md
    cross_model_paired.csv
    <model>/
      protocol.json          # 原权重完整训练配置、实际epoch、sha256与数据清单
      runtime.json           # batch和worker实测记录
      metrics.json
      report.md
      paired.csv             # 患者级配对bootstrap，非种子区间
      history_*.csv          # EF逐病例预测/标签/误差/源帧位置
      seg.csv                # 分割逐样本指标
      *_loss.csv/png
      seg/                   # 预测叠加图
      state_traces.json      # 每batch、每clip的门值、状态变化、融合幅度
      bottleneck_ridge_auxiliary.json
      gradient_audit.json
  cache/                     # 临时小特征；各模型成功后删除自身缓存
```

不新建或复制MAE权重，不删除旧实验的权重。没有定期权重保存，因为此次无MAE训练。原权重保持在原来的ckpt根目录，与本次结果严格分开。`analysis.zip`不包含特征缓存、权重，也不包含其他项目的文件。

## 6. 如何解读，以及这版尚未自动做什么

- 不同历史长度的结论只在共同长视频子集上成立；帧数尚未等同秒数或心动周期数。
- 旧global/spatial/dual存在状态容量差异；保存完整训练配置供核对，表格不是“仅改变压缩方法”的因果结论。
- 64帧训练、192帧读取属于长度外推。正常状态恢复曲线来自人工干扰，不能当临床可靠性证明。
- `state_traces`是batch平均，不是逐患者独立样本。门值趋于0/1也不能单独判断状态好坏。
- 输入梯度是最后clip重建对更早输入的敏感度，**不是**训练参数梯度、历史信息量或下游收益。FP32诊断OOM会明确标记跳过，不伪造零梯度。
- 瓶颈ridge统一为D维只是降低读出容量混淆，仍然会抹掉时间顺序；状态单独读得差不能证明状态无用。
- 不做全量微调、外部测试集调参、每50轮探针；不自动启动两种新结构或组合模型。若诊断指出训练目标缺乏历史需求，先比较目标；若指出压缩/更新瓶颈，再分别增加一个候选。
- 未来结构候选必须匹配训练预算/样本/目标，不能把100轮新模型直接与400轮旧模型作公平优劣结论。最后仅对入选方案补独立种子及关键下游确认。

本地测试覆盖真实文件接口的合成小数据、四种状态兼容、窗口一致性、梯度路径、随机状态/参数不变、缓存与断点重跑。真正4090吞吐与真实数据筛选数量需开机后确认，本地测试不代替云端冒烟。

## 7. 2026-09-29确认协议：先保证有效样本数

首轮快速筛选的256/128是筛选前预算，实际仅77/53例。确认轮改为**筛选后**512/256例，分割恢复64/64患者与200步，EF恢复400步。仍不新增MAE训练。

```bash
RUN_TAG=stage3_confirm_20260929 bash scripts/run_stage3_memory.sh \
  --eligible_budget --ef_train_cases 512 --ef_val_cases 256 \
  --seg_train_cases 64 --seg_val_cases 64 --ef_steps 400 --seg_steps 200
```

`--eligible_budget`按固定种子的病例顺序扫描，直到收集够真实最大历史窗口病例；不足时明确失败，不补重复帧。四种模型使用同一病例与近期源帧。此轮协议与旧run_tag不同，不能覆盖首轮。

新增 `recovery_patient.csv`：对相同患者、相同clip，直接计算干扰状态与正常状态的向量距离、相对距离和输出特征距离；不再仅凭两个状态范数相近推断恢复。新增 `fixed_head_history_*` 复用最长历史条件训练的同一个EF头测不同历史输入，区别于分别拟合头的主比较。两种比较都不能单独替代训练无记忆对照。
