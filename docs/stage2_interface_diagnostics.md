# 二阶段前置诊断：不新增 MAE 预训练

## 目的与范围

复用旧 `clip_mae_pool64` 与 `hier_spatial` 的 epoch400 权重，回答两类问题：

1. 同一权重的历史融合前后，EF 和分割所需信息的可读性是否变化？
2. 分割是否受限于时间位置的读出，而不是必须另建单帧编码器？

不更新 MAE 参数、不保存新权重、不跑全量微调。会在训练集拟合 EF 岭回归和固定 200 step 的小型分割头，因此“零新增预训练”不等于“完全没有监督拟合”。只做一次端点评估，不按每 50 epoch 重复运行。

## 服务器运行

在服务器原项目目录和已有训练环境运行。默认使用已有灰度 NPY，不生成 RGB 缓存。

先检查权重路径与输出位置，不运行评估：

```bash
bash scripts/run_stage2_diagnostics.sh --dry_run
```

默认权重来自：

```text
/root/autodl-tmp/outputs_temporal/temporal_gray_20260909/ckpt/clip_mae_pool64/epoch_0400.pt
/root/autodl-tmp/outputs_temporal/temporal_gray_20260909/ckpt/hier_spatial/epoch_0400.pt
```

先冒烟，再正式跑：

```bash
bash scripts/run_stage2_diagnostics.sh --run_tag stage2_diag_20260928 --smoke
bash scripts/run_stage2_diagnostics.sh --run_tag stage2_diag_20260928
```

冒烟使用真实数据和真实旧权重，但只有 8/4 个 EF 训练/验证病例、2/2 个分割病例和 2 step 小头。输出自动加 `smoke_` 前缀，与正式结果隔离。正式默认 EF 512/256 病例，分割 64/64 病例并保留各病例的标注帧。

原实验 run_tag 或目录不同时：

```bash
bash scripts/run_stage2_diagnostics.sh \
  --source_run /root/autodl-tmp/outputs_temporal/你的旧run_tag \
  --run_tag stage2_diag_20260928
```

也可直接指定权重，不受旧目录布局限制：

```bash
bash scripts/run_stage2_diagnostics.sh \
  --checkpoints pool=/完整路径/pool_epoch_0400.pt spatial=/完整路径/spatial_epoch_0400.pt \
  --run_tag stage2_diag_custom
```

只诊断一份时：`--methods hier_spatial`。权重不存在会在计算前报错，不自动换权重、不随机初始化。第一版仅支持现有纯时域 TemporalMAE，包括 none/global/spatial/dual，不混入频域模型或不同主干。分割要求原生 112 分辨率。

## 最小诊断矩阵

| 任务 | 比较 | 控制方式 |
|---|---|---|
| EF | local 对 fused | 同一遍 encoder、相同病例；分别拟合相同 alpha 的岭回归 |
| EF 分母敏感性 | fused 对 legacy_fused | 完整有效 tubelet 分母 vs 旧原始有效帧分母；各自重新拟合 |
| 分割融合 | local 对 fused | 目标 tubelet 的空间图；相同单层 1×1 头、初始化、抽样和步数 |
| 分割定位 | target 对 local-clip mean | 平均范围仅为目标所在局部片段，不混成全视频平均 |
| 帧内位置读出 | offset_bank 对 shared_bank | 同参数量的两分支头；按帧内偏移路由 vs 分支均值 |

分割前四个读出名为 `local / fused / local_mean / fused_mean`，另外两个为 `shared_bank / offset_bank`。

默认将同一真实标注帧分别对齐到输入位置 48、49，每个视图都以该帧的真实掩膜监督。偏移条件来自帧在 tubelet 内的位置，不是 ED/ES 标签，不伪造相邻帧掩膜。两种 bank 头看到完全相同的视图、标签和抽样序列，拥有相同参数量。路由仍会改变模型容量的使用方式，所以收益不能直接解释为恢复了被丢失的运动信息。

48/49 对齐会让采样窗口移动一帧，未来上下文也不同。因此同时输出两种 offset 的成对结果，不能把它们当作相同视频输入只改变一个标签。若暂不需要这一项，用 `--no-offset_probe`，只运行前四个分割读出。

每个头按固定预算训练，只报告最后一步，不按验证 Dice 选择头或步数。不同读出分别使用训练特征计算标准化，不使用验证集拟合统计量。当前是单个 probe seed 的诊断，不是正式 test 指标。

## 有效帧与边界

- 不改变旧权重的状态更新、注意力或 padding 行为，避免把接口修复和科学比较混在一起。
- 诊断出口 local/fused 都沿用现有完整 tubelet 有效性。半有效 tubelet 的特征仍为零，分割目标落入其中时记录并纳入主指标，不静默过滤。
- EF 主指标按完整有效 tubelet 归一化，额外拟合旧分母读出以量化稀释影响。此版本不能直接与旧日志分数混用。
- 保存每例真实帧数、补零数、半有效 tubelet 数；分割再记录目标完整性、真实历史、未来帧数和完整上下文标志。
- `full_context_dice` 是次要分层结果，样本为零时写 null，不用于替代全部样本指标。
- 默认 EF 看 256 帧，由原生 64 帧窗口独立展开后累计汇聚。每个窗口重置状态；不是完整视频无限流式推理。
- 无记忆模型的 local/fused 必须完全一致，程序会检查。目标映射、两帧共用接口、病例独立性、前向一致性和增量均值等价有单元测试。

## 速度与资源

默认 `--batch_size 0` 会对每份权重做短暂的无梯度推理测速，在 1/2/4/8/16 内选吞吐较高且保留显存余量的 batch。它只调提取批量，不改变病例数、帧数、学习率或 probe 预算，不保证 GPU 始终 100%。自动测速只测编码前向，CPU/I/O 仍可能限速。

局部/融合出口共享编码，每个原生窗口立即汇聚，只缓存小型 EF 向量或分割目标图，不把全视频稠密特征堆在 GPU。没有把 CPU 密集的岭回归硬搬上 GPU 来追求使用率。

可覆盖运行参数：

```bash
bash scripts/run_stage2_diagnostics.sh --run_tag stage2_diag_20260928 \
  --max_batch_size 32 --num_workers 8 --prefetch_factor 2
```

若 OOM，同一命令追加 `--batch_size 2`，已完成缓存可复用。`--batch_size 4` 等正数会跳过测速。CPU 默认 4 线程，避免与 DataLoader 过度抢占；可设 `--cpu_threads`。只调这些运行参数可沿用 run_tag。改病例预算、种子、帧数、头步数或诊断代码后必须用新 run_tag。

## 结果、缓存与恢复

```text
/root/autodl-tmp/outputs_stage2_diagnostics/
  result/<run_tag>/
    report.md
    comparison.csv
    paired_differences.csv
    cross_checkpoint_differences.csv
    analysis.zip
    <方法名>/
      protocol.json / checkpoint_config.json / runtime.json
      status.json
      data_manifest.json / metrics.json / DONE
      ef_*.csv / seg_*.csv / validity_*.csv / loss_*.csv
  cache/<run_tag>/<方法名>/
    ef_train.npz / ef_val.npz / seg_train.npz / seg_val.npz
```

只下载 `result/<run_tag>/analysis.zip` 即可分析，不带缓存或模型权重。缓存完成一整个 split 后原子保存，中断最多重提未完成的 split。每个分割头完成后保存结果，重启会跳过已完成头；不保存小头权重，也不支持恢复到头的中间 step。全部完成后默认清理本工具的四个缓存，`--keep_cache` 可保留。同一命令再次运行会检查协议与病例清单并跳过 DONE。

`status.json` 记录当前提取/拟合阶段与更新时间，终端也有对应 tqdm 进度。每个小头的逐 step loss 保存在 `loss_*.csv`。

缓存绑定权重 SHA256、数据清单哈希、目标数据文件路径/大小/修改时间、病例选择和诊断源码。没有对数十 GB 视频逐文件内容哈希；若人工改数据又保留大小和修改时间，应主动换 run_tag。原权重只读，不删除。

## 如何回答核心问题

- `fused-local`：EF 的 MAE 差负值更好，分割 Dice 差正值更好。若 EF 改善而分割损失，支持优先检查融合保护和任务出口，而不是立即增大状态。
- `target-mean`：若目标图稳定更好，说明应保留定位读出，不代表必须用单帧 encoder。
- `offset_bank-shared_bank`：若两种对齐都出现收益，支持继续研究帧内读出；不等于已证明 tubelet2 必然丢信息或应改预训练。
- 区间按病例 bootstrap，同病例 ED/ES 与两个对齐视图先平均，再重采样。不能把多个视图当独立患者扩大显著性；区间不包含训练 seed 不确定性，也未作多重检验校正。
- 跨权重对比是描述性参照。旧 pool/spatial 训练运行条件存在差别，不能称严格的记忆因果实验。local/fused 也改变了整个融合块的计算，不仅改变历史内容。
- 本轮不能回答 decoder 历史旁路是否有害，也不能证明冻结读出排名等于全量微调排名。只有定位到明确瓶颈后再决定一项干预或成对微调。

运行前可在研究记录中自行填写判断和预期。本实现不代写 Prediction Lock，不把这些候选比较自动标为已完成实验。
