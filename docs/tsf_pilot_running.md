# 时空频最小试验：两组、100轮、自动调速

## 本轮只跑什么

| 顺序 | 名称 | 内容 |
|---|---|---|
| 1 | tsf_spatial_control | 现有64帧、4×16局部VideoMAE、4×4空间记忆，原像素重建 |
| 2 | tsf_frequency_memory | 相同骨干、decoder与记忆容量，加可见patch小波条件化读写和多尺度L1目标 |

不是9组，也不重跑全量微调。每组100轮，从 ckpt/mae/videomae_vit_s.pth 初始化；seed=42，112×112、patch8、tubelet2、mask0.75、灰度NPY按需复制为三通道。两组均沿用当前时域协议的随机连续片段采样；不额外开启A4增强，避免同时改变输入分布。

学习配置沿用当前时域对照：AdamW，lr=1e-4，weight_decay=0.05，betas=(0.9,0.95)，scheduler=none，有效batch32，AMP。没有MAE早停，固定100轮。本轮不是重现另一套官方训练，也不改变先前已完成实验。

## 一次性运行

在云端项目根目录，更新项目后：

```bash
bash scripts/run_tsf_pilot_smoke.sh
```

冒烟跑两组各1轮、每轮最多2个batch、小样本EF、2步分割探针。**用真实数据和完整模型**，验证新路径能运行；其结果不可用于性能结论。冒烟目录自动带 smoke_ 前缀，不会作为正式模型续训。

确认通过后只需：

```bash
RUN_TAG=tsf_pilot_100_v1 bash scripts/run_tsf_pilot.sh
```

默认输入：/root/autodl-tmp/datasets/EchoNet-Dynamic，包含FileList.csv、VolumeTracings.csv以及npy目录或Videos。不需再缓存频率数据，不需RGB数据目录。

显式改路径或先查看将运行什么：

```bash
bash scripts/run_tsf_pilot.sh --dry_run
RUN_TAG=tsf_pilot_100_v1 bash scripts/run_tsf_pilot.sh --data_root /实际数据路径
```

不要把新run_tag指向旧实验目录；不要把smoke目录作为正式运行目录。

## 自动调速做什么

默认开启 --autotune，无需另外手调YAML：

- 每个模型在独立子进程中，用真实数据测前向、反向和优化器更新吞吐。
- 从micro-batch8、梯度累积4出发，搜索能整除有效batch32的候选；同步调整累积步数，不用增大有效batch来伪装加速。
- 对比梯度检查点重算开关，以及worker数量。prefetch默认4、pin_memory和persistent_workers沿用数据配置。
- 选择吞吐更高且有显存余量的组合。测量中OOM仅淘汰该候选，不污染正式模型；其他异常会明确报错。
- 探测权重不用于正式训练；正式模型重新初始化。结果写入各模型的autotune/runtime.json，断点继续时复用。

搜索是有限候选、短时间测量，不保证全局最优，更不保证GPU利用率常年100%。目标是实际samples/sec。现有安全策略对本进程与其他进程总占用保留约15%显存余量，并至少预留2GiB；不要为了显示占满把这部分去掉。

若服务器同时运行其他项目，调优结果可能受影响。换GPU/配置不能静默复用旧调优结果，应保留原报告并使用新run_tag。CPU会跳过CUDA校准；本地CPU验证不代表已经测过4090速度。

冻结评估默认batch8、workers8，只有少数几次。需调整可在脚本后加 --audit_batch_size 4，不改训练batch与学习参数。

## 评估流程

```text
对照预训练100轮
  -> 冻结EF：epoch0、epoch100
  -> 冻结Dice：仅epoch100
候选预训练100轮
  -> 冻结EF：epoch0、epoch100
  -> 冻结Dice：仅epoch100
  -> 汇总并打包结果
```

EF使用固定512训练病例、256验证病例、256帧上下文与Ridge闭式回归。状态EF从同一次编码中读取，作为辅助诊断，不能与正常EF取最小值排名。特征缓存供以后补不同标注预算，不运行反序、shuffle、reset、相位分类、全量微调或重复推理测速。

Dice使用64/64患者的标注帧、冻结骨干和200步轻量1×1卷积读出；64帧上下文、目标位置48，避免目标恰在原生窗口记忆重置点。不是分割全量微调。比起只看EF，这一次终点Dice能检查空间细节是否有收益，但每50轮都跑没有必要。

0/100 EF正常读出做患者配对区间；最终Dice先按患者平均ED/ES，再做配对bootstrap。EF差值为候选减对照，负数更好；Dice差值正数更好。小样本、单seed仅用于决定是否继续投入。

## 结果、权重与断点

```text
/root/autodl-tmp/outputs_temporal/tsf_pilot_100_v1/
  result/
    tsf_spatial_control/
    tsf_frequency_memory/
    quick_comparison.csv
    quick_comparison.md
    quick_paired_comparisons.csv
    quick_paired_seg.csv
    stage_times.csv
    current_stage.json
    analysis.zip
  ckpt/
    tsf_spatial_control/
    tsf_frequency_memory/
```

每组保留epoch_0000.pt、epoch_0100.pt和last.pt；last第1轮、每10轮和末轮覆盖保存，包含优化器等续训状态。正常键盘中断保存interrupt.pt；意外关机只能恢复最近完整快照。模型评估完也不自动删除这些快照。磁盘默认另保留2GiB写入安全余量，并检查临时文件空间。

整个队列继续：

```bash
RUN_TAG=tsf_pilot_100_v1 bash scripts/run_tsf_pilot.sh
```

相同run_tag恢复尚未完成的训练，已完成的训练和评估阶段不重跑。若只需从第二组开始：

```bash
RUN_TAG=tsf_pilot_100_v1 bash scripts/run_tsf_pilot.sh --start_experiment 2
```

下载 result/analysis.zip 即可交给分析端；不会包含权重或特征NPZ。各模型logs/metrics.csv包含像素loss、原始频带loss、加权频带loss、读写门控均值及运行时间。plots保留训练曲线，TensorBoard保留分项。不同模型的总预训练loss不用于排名，因为候选多了一个目标。

## 实现边界

- models/frequency.py：固定两级正交Haar分析、7个子带描述、固定尺度归一化的子带平均L1、轻量记忆门控。
- models/temporal_mae.py：配置开关frequency_conditioned与frequency_loss_weight；默认关闭，旧权重不需要新增参数。仅在可见patch上提取条件，完整目标只用于loss。
- 两组同一个decoder；候选额外301824个参数（embed_dim384），没有第二个大编码器，也没有扩大16个空间状态的容量。
- 读门控初始为1，范围0至2；写门控初始sigmoid(2)，范围0至1。频率权重默认0.1是固定工程起点，不是医学常数。日志记录其实际贡献，本轮不自动搜索该权重。
- 这是“局部空间频率条件化记忆”，不包含时间FFT、组织跟踪、RF分析或高频去噪宣称。跨片段因果，片段内仍是双向视频注意力。
- 两组模型采用相同可见性规则和受控数据采样。自动micro-batch选择可能带来不同浮点路径与掩码随机分组，不宣称逐bit等价；它用于低成本筛选，显著收益后再补严格多seed确认。

先看候选在EF或Dice上是否有可信收益、另一项是否明显恶化，同时查看训练时长。没有收益不追加整套消融；有收益再补“只改多尺度目标”的第三组，分清目标重加权与条件化记忆的贡献。
