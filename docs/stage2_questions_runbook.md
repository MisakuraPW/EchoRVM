# 二阶段代码与服务器运行说明

本轮实现对应 `stage2_research_plan.md` 的三个问题组，不改变一阶段正在运行的配置，不自动开始任何云端训练。以下是低成本诊断和可选新结构训练的入口，不是已经取得实验结论。

## 1. 实现了什么

| 问题 | 实现 | 输出/边界 |
|---|---|---|
| A 指定帧空间表征 | 原tubelet2复制、共享预训练帧级展开、tubelet1；真实邻域/重复目标邻域；两个目标偏移 | Dice、患者Dice、global Dice、像素边界距离、低分位、标注面积大/小两期、8例叠图；不是未标注帧准确率 |
| B EF如何读出 | 末clip、近期完整缓存、可选末状态；独立训练的joint64参照 | 固定预算的同参数量注意力汇聚头；MAE/RMSE/相关性与逐病例误差；状态不是主指标 |
| C 缓存与更早状态 | 真实前64帧形成状态，再处理最近64帧；最近窗口之前的状态与近期缓存联合读取 | 真历史/空历史配对；local/fused两种缓存；不足128帧的病例单列排除，不重复补造历史 |

`models/temporal_mae.py` 默认仍为旧结构。`frame_readout: learned` 才启用共享帧级展开。展开发生在可见编码token上，进入MAE decoder；隐藏位置只使用mask token，预测按帧重排回原tubelet像素顺序，继续以同一masked raw-pixel MSE计分。下游通过同一个 `frame_features()` 读取，不另外训练完整单帧encoder。

学习式展开会增加decoder token数和参数；tubelet1也改变encoder token数。它们是结构对照，不是只有名字不同的严格单因素实验。tubelet1初始化对原patch投影的时间核求和，空间缩放沿用现有适配器；初始化加载/跳过情况记录在 `initialization.json`。新帧级投影不是冒充从旧权重加载。

流式接口 `stream_clip(video, state, short_state)` 一次消费一个完整local clip。`StreamingFeatureCache` 显式reset患者，维护FIFO、原视频帧索引和窗口开始前状态；重复帧、跳帧、患者顺序变更会报错。查询不更新状态。缓存只保存空间池化后的逐帧描述与有限状态，完整帧图只在本次update返回，调用方可即时分割。

本轮没有实现“稠密轻量空间支路+稀疏重型支路”，也没有加入LV-MAE embedding重建目标。它们仍是后续可选研究，不会被悄悄加入对照。现在借鉴的是紧凑特征序列上的轻量读取，属于监督读出，不宣称新增自监督创新。

## 2. 更新时机与最便宜的第一步

正在跑一阶段时不要中途pull并重启队列。旧进程不会自动更新已加载模型，但后续子进程可能读到新版代码；一阶段还有代码指纹保护。等它结束后再更新，不要覆盖旧指纹。

```bash
cd /root/autodl-tmp/EchoRVM
git pull
conda activate echocardmae
```

先选一份已存在的**纯时域TemporalMAE**权重。下面L16只是显式示例，不代表代码已选定一阶段胜者；使用你实际保留的文件。

```bash
CKPT=/root/autodl-tmp/outputs_stage1/stage1_lengths_20260928/ckpt/spatial_l16/epoch_0100.pt
bash scripts/run_stage2_questions.sh --phase diagnose --checkpoint "$CKPT" --run_tag diag_l16 --smoke
bash scripts/run_stage2_questions.sh --phase diagnose --checkpoint "$CKPT" --run_tag diag_l16
```

此处**没有新增MAE预训练、没有全量微调**。默认一次提取后训练6个小EF读出和2个分割读出；学习式展开权重额外比较展开前/后的分割读取。数据预算EF训练512/验证256个候选视频，分割训练64/验证64个病人、每个真实标注帧两个位置；EF400步、分割200步。EF会在候选中排除不足真实128帧者，具体数量见excluded/validity表。因此不要把该子集分数直接和以前全体256例数字比较。

默认EF使用保留时间位置的轻量attention pooling，空间池化后的每帧向量保持序列。参数量不随输入序列长短变化，空历史/真历史使用同槽位数、同初始化种子、同近期缓存归一化、同训练步数。tubelet2复制出的相邻描述仍然可能相同，不能据此声称已分开帧身份。

更便宜的 `--ef_head ridge` 会先均值池化，只用于粗筛，不能回答完整时序聚合能力。`--state_probe` 是独立 `tools/evaluate_stage2.py` 的可选辅助项，默认不运行。冻结探针阴性不等于架构不适合任务。

## 3. 新结构匹配预训练

运行前用 `--dry_run` 查看选中项。默认只列原结构repeat与帧级展开learned，不自动展开多长度/多状态矩阵。tubelet1和joint参照已经可运行，显式加入即可。

```bash
bash scripts/run_stage2_questions.sh --phase train --local_frames 16 --variants repeat learned tubelet1 joint --run_tag structures --smoke
bash scripts/run_stage2_questions.sh --phase train --local_frames 16 --variants repeat learned --epochs 100 --run_tag structures
```

确认需要时，补上其余两项：

```bash
bash scripts/run_stage2_questions.sh --phase train --local_frames 16 --variants repeat learned tubelet1 joint --epochs 100 --run_tag structures
```

已完成的同协议预训练/读出跳过；追加时保留repeat/learned名字才能一起生成比较表。模型、种子、epoch、初始化或代码指纹变更必须用新tag，不能拼成同一实验。`--smoke` 自动使用独立 `smoke_` tag，2训练步及极小读取预算，绝不当正式成绩。

四项共同设置：总64帧、stride1、112px、patch8、同一个VideoMAE初始化来源、同一训练划分、tube mask .75、原始像素目标、AdamW lr1e-4/wd.05、有效batch32。沿用当前时域基线的无增强协议，不擅自替换为A4。repeat/learned/tubelet1使用同一固定spatial状态；joint是L64、1clip、无状态的匹配整窗模型。

只在终点做上述小头评估，MAE重建验证每25轮，仅监控，不早停、不据它挑最佳表示。不会每50轮训练分割/EF，也不自动全量微调。

## 4. 效率、保存与恢复

- 预训练默认启用原有实测autotune：microbatch/梯度累积、gradient checkpointing、workers；保持有效batch32，不修改学习率、帧数、mask或模型宽度。`--no-autotune` 才关闭。
- 冻结提取默认 `--eval_batch_size 0`，同时测真实EF流式路径与分割路径的前向吞吐，在候选中选较快且留显存余量的batch。workers显式 `--num_workers 8`；不是“保证100%占用”，更不是按利用率盲目放大参数。
- 相同命令可续跑：训练恢复last、优化器、调度器、scaler、RNG；评估复用完整split缓存与完成的小头结果。中断的极小读取头从头重拟合，不保存几百个无用头权重。
- 默认初始/最终各一个轻量MAE快照；完整last每10轮覆盖一次，首轮/结束及中断按原trainer保护逻辑保存。无每轮best或大量历史快照；不自动删除用户旧权重。完整last不能用轻量终点文件替代断点恢复。
- 缓存成功评估后只删除本次拥有的特征NPZ，结果保留。`tools/evaluate_stage2.py --keep_cache` 可保留；不删除原数据。
- 自动汇报模型含MAE decoder参数数、源checkpoint大小、batch1流式clip更新p50/p95、缓存张量容量、峰值显存。计时不含采集等待、磁盘解码和EF头；源checkpoint可能含优化器，不能冒充纯部署模型大小。

布局：

```text
/root/autodl-tmp/outputs_stage2/<run_tag>/
  result/<variant>/endpoint/       # 指标、协议、每例预测、曲线、叠图
  result/<variant>/logs/           # 新MAE训练时才有
  result/comparison.csv, comparison.md, analysis.zip
  ckpt/<variant>/                  # 不在result内
  cache/<variant>/                # 中间特征，成功后清理
```

只下载 `result/analysis.zip`。父表是不同结构终点对比，endpoint内的paired表是同模型不同读取方式的患者配对bootstrap；区间不包含训练种子不确定性。需要完整排错时再下载result目录。

```bash
tail -f /root/autodl-tmp/outputs_stage2/structures/result/learned/logs/train.log
cat /root/autodl-tmp/outputs_stage2/structures/result/current_stage.json
nvidia-smi dmon -s pucm -d 1
```

## 5. 可选全量微调，不自动连跑

有初步信号再用新的两个配置；它们复用原trainer进度条、曲线、early stopping、断点逻辑。学习式帧展开会进入分割的共享骨干梯度，不再复制两帧。EF可选择与诊断同名的读取方式，完整反传prefix，不悄悄detach冒充全量微调。

```bash
python trainers/train_finetune.py --task echonet_seg --config configs/finetune/stage2_seg.yaml --pretrained "$CKPT" --output_dir /root/autodl-tmp/outputs_stage2/ft/result/seg --checkpoint_dir /root/autodl-tmp/outputs_stage2/ft/ckpt/seg
python trainers/train_finetune.py --task echonet_ef --config configs/finetune/stage2_ef.yaml --pretrained "$CKPT" --ef_readout cache_history --output_dir /root/autodl-tmp/outputs_stage2/ft/result/ef_cache_history --checkpoint_dir /root/autodl-tmp/outputs_stage2/ft/ckpt/ef_cache_history
```

EF支持 `last/cache/cache_empty/cache_history/local_empty/local_history/state/joint_recent`。最后一种只能接独立joint模型。不同读取方式务必独立output/checkpoint目录。最近64帧、前缀64帧均为真实连续帧；不够长的数据在建loader时过滤。默认A4在线clip一致增强，80轮上限、15轮patience。全量微调更昂贵，默认保守microbatch1/累积32，不使用仅适用于MAE的autotune冒充已校准微调；先加 `--max_steps 2` 验证显存，另用输出目录，正式运行不带它。

全量分割沿用现有中心窗采样及边界clamp策略，允许目标帧之后的上下文；它不是严格在线因果成绩，也不同于冻结诊断的窗口末端目标索引。各结构必须使用同一配置成对比较，不能直接拿两种协议的Dice差值当结构增益。EF完整历史筛选清单保存在 `history_dataset_manifest.json`。

冻结阶段B-last/cache/state在最近窗口开始处reset，与joint看到同一最近64帧。C-history/empty则使用真实前缀；二者的fused近期缓存都已吸收历史，因此比较的是**额外显式读取边界状态**。local-history/empty用于补充区分未融合特征。禁止混写成“有记忆vs无记忆”的因果结论。

## 6. 仍然要遵守的解释边界

- 64帧训练权重处理128帧，是长度外推诊断，不是已经充分训练长记忆。
- 所有流式输出等待当前clip结束，最早帧可能等待L-1个采集间隔。tubelet1本身不让双向attention变成逐帧因果。
- 两种目标偏移改变邻域/未来可见帧数量，配对差异不能冒充纯offset效应。
- 真实邻域与重复目标邻域分别拟合同规则头，是输入干预，不是公平的单帧MAE预训练对照。
- EF只拥有病例/视频标签，没有每个滚动窗口的真实EF，不报告虚构的在线逐窗准确率。
- 验证集用于研究筛选；不把指标叫最终test成绩，不声称EF/分割证明所有疾病任务通用。
