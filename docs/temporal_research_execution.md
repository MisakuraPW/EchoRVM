# 心超时域表征研究：实现与服务器执行

## 1. 本轮研究边界

本轮实现的是可验证的假设：短片段内的运动由 VideoMAE 编码，跨片段的信息由因果记忆保存。不能预先认定记忆一定提升性能。

- 主研究数据为 EchoNet-Dynamic 的真实视频。仅 TRAIN 用于自监督更新与探针训练，VAL 用于研究阶段比较；TEST 不用于选模型。
- 本文“病例”指 FileList 的视频病例ID，遵循其官方Split；没有额外患者身份映射时，不宣称独立完成了跨视频的患者身份核验。
- CAMUS 的 ED/ES 静态图不伪装成连续心动周期。本轮不把重复静态帧用于长时序实验。
- 新目录、新实验名重新运行，不接续旧的“单帧 EchoCardMAE 对 16 帧 VideoMAE”权重。那组实验不能作为同条件算法优劣的证据。
- 本地只运行针对性的结构与工程测试，正式数据、CUDA 显存、吞吐和效果由服务器冒烟及正式实验确认。

## 2. 先修复的基线问题

涉及 utils/datasets.py、models/video_mae.py、models/echocardmae_video.py、utils/pretrained_init.py：

1. 修复真实加载路径中未定义的 channels、sampling_rate、two_views 参数与成员。
2. EchoCardMAE 真正使用两个独立起点的 16 帧视频，采样 stride=4，两视图共享同一 ROI mask；训练缺少第二视图直接报错。
3. 使用官方展平序列的交错 sin/cos 位置编码，不再替换成因子化三维位置编码。
4. 支持官方的独立 q_bias / v_bias，K bias 固定为零；编码器到解码器映射没有 bias，LayerNorm eps=1e-6。
5. ROI 可见数量采用 int(ROI数 * (1-mask_ratio))，避免遮挡数量差一的错误；背景 token 从零初始化。
6. median target 的边界采用 replicate，与 OpenCV medianBlur 对齐。
7. 记录初始化映射清单；新入口要求编码器所有参数成功初始化，不只是达到某个张量数量。
8. matched VideoMAE 不再残留双视图配置，不使用 ROI / median / InfoNCE。
9. 标准 VideoMAE 分支的像素标准化按原始 [0,1] 像素、逐通道、tubelet 内样本方差实现。
10. 修复两个训练器最后不足累积步数的梯度缩放，以及 AMP 跳过参数更新时错误推进 scheduler/global_step 的问题。
11. 分割 dice_mean 改为逐图 Dice 的平均；dice_global 仍由整个 epoch 的交并计数计算。不能将旧的 batch-global 平均当作逐图平均。
12. EchoNet掩膜强制使用scikit-image的官方polygon栅格化规则，不再在缺少依赖时静默改用OpenCV。连跑前检查依赖；缺少时执行 python -m pip install scikit-image。

本地数值测试直接调用资料内官方 EchoCardMAE 模型：同权重、同 16 帧输入、同 mask 的小宽度模型，比较编码特征与重建输出。该测试验证模型前向语义，不意味着已经复现了论文的完整训练结果。

官方依据：资料/EchoCardMAE-main/EchoCardMAE-main/pretrain.py、model/echocardmae_vit.py、model/utils.py。
VideoMAE target 依据：[官方 engine_for_pretraining.py](https://github.com/MCG-NJU/VideoMAE/blob/main/engine_for_pretraining.py)。

## 3. 统一灰度输入

2026-09-09 最新协议：gray_repeat3。复用旧 tools/cache_echonet_npy.py 生成的灰度 [T,H,W] uint8 NPY，不重新缓存，不依赖已删除的RGB目录。

默认数据根目录 /root/autodl-tmp/datasets/EchoNet-Dynamic。以mmap读取，先采样真实帧，再在内存中把灰度扩展三通道。模型保持in_chans=3及现有三通道初始化，不执行以前将patch权重求平均的单通道结构转换。

MAE预训练、初始化阶段评估、所有冻结探针及可选全量微调均使用同一灰度内容协议。模型内部原有mean/std标准化不变；复制前后三份原始灰度相同，不代表经过不同通道mean/std之后数值仍完全相同。

这是统一灰度输入的算法适配对照，不再宣称原始RGB输入的逐像素复现；与旧RGB实验必须分开run_tag。input_protocol保存在配置、权重、input_contract.json及评估报告中，评估从权重自动读取，防止训练灰度、评价RGB。

数据根目录需要FileList.csv、VolumeTracings.csv及npy/。若只存在AVI，仍可解码后用与旧缓存一致的OpenCV灰度转换；新默认路径优先读取已有NPY，不反复解码。RGB缓存的可选灰度转换约定输入缓存是RGB，而AVI解码为BGR，两条路径使用相应转换，避免通道次序错误。

在线解码节省磁盘，但增加 CPU 解码和可能的网络 I/O，实际吞吐需查看 data_time、step_time。想减少 I/O 可仅复制压缩的原始 AVI 和 CSV 到本地，不必解压为 NPY；先检查空间，不自动额外复制。

保留--input_protocol rgb作为显式兼容选项；只有该选项才能同时使用--prepare_rgb_cache。本轮灰度训练不要添加缓存生成参数。完整命令见 [灰度训练说明](temporal_gray_run.md)。

## 4. 默认 9 组主实验

| 名称 | 输入/主体 | 用途 |
|---|---|---|
| echocardmae_video_port | 16帧、stride4、patch8、双视图 ROI+median+InfoNCE | 官方算法迁移实现，400轮研究预算 |
| videomae_matched | 16帧、stride4、patch8、tube mask 0.75、单视图 | 输入尺寸、编码器规模、优化器匹配的干净对照 |
| videomae_standard | 16帧、stride4、patch16、tube mask 0.90、逐通道标准化 target | 标准 VideoMAE 核心目标的心超适配分支 |
| frame_mae_pool64 | 64张独立单帧，tubelet1，无记忆 | 单帧空间编码对照 |
| frame_rvm64 | 同样64帧单帧编码，逐帧记忆 | 纯逐帧 RVM-style 路线 |
| clip_mae_pool64 | 16帧×4片段，无跨片段记忆 | 与分层模型同长输入的关键对照 |
| hier_global | 16×4，全局 clip 记忆 | A：低成本全局状态 |
| hier_spatial | 16×4，4×4 空间状态 | B：空间 token 状态 |
| hier_dual | 16×4，上一片段短状态＋长期空间状态 | C：短期/长期双状态 |

这里的 RVM 使用项目内已有的 GRU 门控 transformer recurrent core，是明确实现的 RVM-style 研究模块，不声称逐行复现某一外部 RVM 仓库。

单帧组的 ViT 初始化仍来自同一个 VideoMAE 权重，tubelet kernel 转成单帧 kernel。它是控制时域作用的单帧对照，不是额外导入另一份 ImageMAE 权重。

### 哪些比较可以归因

- hier_global / spatial / dual 对 clip_mae_pool64：相同原始64帧、相同局部编码器/目标/遮挡/有效 batch，研究跨片段记忆。
- frame_rvm64 对 frame_mae_pool64：研究逐帧记忆。
- EchoCardMAE 对 matched VideoMAE：完整算法组合比较，不是“只开关一个 trick”的消融。双视图带来的帧暴露量和负样本构成都不同，报告时必须说明。
- patch16 标准分支对 patch8 matched 分支：多个设计差别，不能单独归因于时序。
- 相同 epoch 不代表相同 FLOPs、token 数或帧暴露量。stage_times.csv、模型参数、实际输入长度必须一起报告。
- 移除记忆也减少了参数量，因此单靠该对照不能排除容量收益；记忆重置诊断也不是参数匹配的重新训练实验。论文阶段仍需针对入选方案补充容量匹配等控制。

为保留官方预训练配方并控制变量，本轮MAE阶段统一关闭A4增强、使用AdamW。已有Muon+AdamW实现仍保留，但不把“换时序结构”和“换优化器”同时放进一次对照；需要时另立实验。

## 5. 完整矩阵：37组

--suite full 包含主实验9组，另加：

- k∈{4,8,16,32}、N∈{2,4,8}；每个组合同时运行无记忆与 global 记忆。16×4已经在主实验内，避免重复。
- 在固定16×4上，对 random_frame、temporal_block、complementary 三种遮挡，各跑无记忆与 global。tube 已在主实验内。

所有遮挡保留相同数量的可见 token。random_frame 在每个时间 tubelet 独立采样；tube 跨时间共享空间遮挡；temporal_block 隐藏连续时间范围，边界允许部分 tubelet；complementary 循环移动可见空间集合，减少邻近片段总看同一区域的问题。

这不是带有真实心动相位标签的周期遮挡。没有完整相位标注时不会自动声称实现了 phase-aware masking。频域模块本轮不实现；未来可在局部 token 与记忆接口之间增加模块，但必须另立对照。

时长轴改变 k*N 时，也改变输入信息量和计算量，必须在相同 k,N 的记忆/无记忆配对中看收益。固定64帧时可比较8×8、16×4、32×2。

## 6. 记忆实现中的约束

- 局部编码器只见未遮挡 token。
- 处理第 i 个 clip 时，只能使用此前 clip 的状态；局部 VideoMAE 在当前 clip 内仍是双向注意力，这不等于逐帧因果模型。
- 历史状态参与当前可见特征融合和 MAE 解码。状态不是仅在推理阶段额外拼接的未训练特征。
- 更新状态只使用当前可见编码特征；不读取重建 target，不读取未来片段，不使用 EF/分割标注。
- 默认完整跨片段反向传播，不进行静默 detach / 截断 BPTT。
- 每次视频 forward 从空状态开始，跨病人不共享状态。
- 长度不足采用零填充，提供 frame_valid 和原始帧索引。分层模型的 loss 不计无效 tubelet，全空片段不更新状态；评估池化也排除无效帧。
- 官方基线仍保留其原有零填充训练语义，不将分层模型的 padding-loss 改动偷偷带入官方算法。

## 7. 评价口径

默认保存并审计 epoch 0/50/100/150/200/250/300/350/400。0指加载自然视频预训练权重后、尚未进行心超迁移更新的状态。MAE不早停，不用最低 val reconstruction loss 选取科学比较的 checkpoint。

### 主要指标：冻结 EF

- 默认固定512个 TRAIN病例，256个 VAL病例；病例选择和 clip 起点由种子确定，同一指标在不同 checkpoint / 模型上使用相同样本。
- 使用嵌套的64/256/512病例标签预算拟合 ridge，alpha固定10，不更新主干。
- 模型均读取同一段最多256个原始相邻帧，stride1。按各自的 native window 处理，不会把图像模型伪装成一整个视频 Transformer。
- VideoMAE 无跨 native window 的记忆；分层模型保留设计范围内的跨 clip 记忆。最终均做有效帧特征的平均池化，再接同款线性探针。
- 报告 MAE、RMSE、相关性、R²与 MAE 的病例 bootstrap 区间。
- 额外报告“有效视频长度→EF”的单变量控制，帮助排查 padding/视频长度相关的捷径。

这里的64/256/512是病例数，不冒充全数据集的1%/10%标签预算。需要扩大时可以单独调用 evaluate_temporal_mae.py 的 --ef_train_cases / --ef_val_cases，并对所有模型统一修改。

### 冻结分割与 ED/ES 诊断

- 每个 split 固定64个病例，保留其人工描记帧，依据官方 VolumeTracings polygon 规则生成掩膜。
- 使用真实相邻帧，标注帧位于统一上下文中心；标注不进入主干。
- 特征统一到14×14，训练1×1线性头，插值至112×112后计算CE；固定200步，不选择验证最优步数。
- 同时报告逐图 Dice 平均和全局 Dice。该指标是低成本线性探针，不是全量微调，也不是论文分割结果。
- ED/ES 二分类采用两个描记帧中较大面积作为ED的代理标签。名称明确为 area_derived_ed_es_accuracy；不是官方提供的连续相位真值，也不能证明恢复了完整周期。

### 时域诊断

- 冻结特征对“正序/反序”进行固定 ridge 分类，训练/验证病例不交叉。反序仅翻转真实帧，padding位置不反转。
- 使用正常训练得到的同一个 EF 探针，评估正常记忆、每clip重置、每2/4clip重置、跨病例打乱状态。
- 另外直接冻结并读取记忆状态，训练 state-only EF ridge、ED/ES代理分类和分割线性头，而不只观察“加入记忆后的特征”。EF取有效片段状态的时间加权平均；状态的空间token先平均为一个向量。
- state-only分割采用状态向量线性映射到14×14×2 logits，再插值到112。它的头参数量比主分割探针大，会单独报告；不能拿两种头的Dice差距解释为状态优劣，也不能声称测试了空间状态的每个位置。
- 输出干预前后每病例误差变化及其配对 bootstrap 区间。打乱采用batch循环错配；最后孤立的单病例不伪装成跨病例错配。
- 保存相邻特征距离、状态向量和局部PCA图。曲线只作诊断，不将“变化大”或“轨迹成环”自动解释为良好的心动周期表征。
- 保存 effective rank，帮助检查塌缩；高rank本身不是成功指标。

主冻结探针统一读取不施加重建mask的dense encoder特征。EchoCardMAE这一读出与其训练时仅对可见前景token做InfoNCE的读出并不相同，所以单项探针失败不能直接等同于官方算法失败；它衡量的是本项目规定的下游读出协议。

### 判断是否值得继续

优先看相对 epoch0 的 EF 改善，以及相同长度无记忆对照上的配对差异；再检查分割是否明显退化、顺序判别与记忆干预是否支持时序假设、耗时/参数是否可接受。不把这些指标强行加权成一个未经验证的总分。

paired_comparisons.csv 中 MAE difference < 0 表示方法优于对照。其区间没有多重比较校正；单种子筛选结果不能直接当论文结论。入选方案需至少3个预先确定种子，并通过全量微调锚点验证。早期探针能否预测最终任务性能本身也是待验证假设。

## 8. 一键执行

在项目根目录，激活原训练环境后运行。不需要云端下载官方项目，也不依赖本地“资料”目录。

先检查实验清单，不读取数据、不训练：

```bash
bash scripts/run_temporal_research.sh --dry_run
```

首次真实数据冒烟，独立目录，9组均跑2个训练batch，并检查冻结评价：

```bash
bash scripts/run_temporal_research.sh \
  --run_tag temporal_gray_smoke --smoke
```

灰度缓存不在默认位置时，用 --data_root 指定它。想在线读AVI，也用该参数指向含Videos和CSV的原始根目录，灰度内容协议保持不变。

正式主实验：修正后的三条视频基线＋六条时域研究模型，共9组，每组400轮及全阶段冻结评估：

```bash
bash scripts/run_temporal_research.sh \
  --run_tag temporal_gray_s42
```

也可以显式指定已有灰度缓存：

```bash
bash scripts/run_temporal_research.sh \
  --run_tag temporal_gray_s42 \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic
```

同样的原始命令也是续跑命令，必须保留首次运行的 data_root 和训练参数。已完成阶段跳过，未完成训练从最近的 last.pt 恢复；默认每10轮保存，因此意外关机可能需要重算最多约10轮，而不是逐batch精确恢复。不要把 partial-epoch 的 interrupt.pt 当成完整epoch续训点。

完整37组可直接加 --suite full；也可以主实验结束后用相同 run_tag 加 --suite full，已完成的9组不重跑。这个矩阵预算明显更大，不建议未经冒烟就启动全部37组。

显式选择个别实验：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic \
  --only hier_global hier_spatial hier_dual
```

只重新汇总、重新打包：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 --summarize_only
```

同一run_tag下改变已有实验的训练超参数会报错，防止把两套协议接在一起。改batch/epochs/seed/data_root请用新run_tag；不能把旧baseline400标签填进来接续错误实验。单纯改变权重目录、last保存频率和磁盘保留策略不改变训练协议。

## 9. 可选全量微调锚点

--anchors baselines 在三条视频基线最后一个checkpoint上追加 EchoNet EF 和分割全量微调；--anchors core 对主实验9组追加。默认none，避免把37组消融自动扩成大量昂贵下游实验。

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic \
  --anchors baselines
```

可在冻结审计结束后追加，已完成训练与审计会跳过。复用现有EF头、ViT分割头、监督loss、验证选best和早停配置；设为全量微调、真实native长度视频，microbatch2、累积16，以减少full-token训练OOM风险。各模型native长度不同的锚点不是等计算量比较，必须看保存的配置。

这些锚点采用项目的统一下游流程，不称为官方EchoCardMAE下游逐行复现。CAMUS不在这份时域runner内自动启动。

## 10. 默认资源和磁盘策略

- 原始图像通道数为3，局部编码器384维、12层，MAE解码器192维、4层；本轮记忆core_depth=1，实际参数完整保存于配置。
- 本轮复用已有灰度缓存，不生成三倍大小的RGB缓存。仍须用 df -h /root/autodl-tmp 检查阶段权重和last.pt原子保存临时副本所需空间。

- 官方/视频基线 microbatch32，累积1。EchoCardMAE的InfoNCE负样本数取决于microbatch；不能拿梯度累积假装扩大负样本集合。
- 时域研究 microbatch8，累积4，有效batch32；可用 --batch_size / --grad_accum_steps 修改。所有时域对照同时修改。
- --baseline_batch_size 单独控制视频基线；改小后要记录它对InfoNCE的影响。
- AMP、SDPA注意力、编码器/MAE解码器activation checkpointing开启。CPU数据worker默认8、prefetch4。
- 默认复用autodl-tmp上的灰度缓存；在线AVI可选但不默认。workers优先按data_time调，不以瞬时GPU利用率为唯一目标。
- frame_rvm64每视频有64次串行状态更新，不一定比16×4结构快。增加batch也不保证消除kernel调度开销。
- OOM先将时域batch8改4、累积4改8，在新run_tag重跑同组对照；不自动缩短frames或改变mask比例。
- 本地没有CUDA训练验证，不能保证上述batch在每个GPU/软件环境都适合。服务器冒烟就是正式前的显存与依赖检查。
- last.pt在第1轮、每10轮、评价快照轮及训练结束时覆盖保存完整续训状态。--save_last_every 可修改频率；只有一份last，不累计last历史。
- epoch_0000/0050/.../0400为不带优化器的评价快照；仍先训练完400轮，再逐阶段冻结评估，不改变评价协议。每个阶段评估成功后删除对应中间快照，最终保留epoch_0400.pt。
- 该模型全部请求的评估及可选全量微调完成后，删除其last.pt/interrupt.pt。尚未完成、评估失败或--no_audit时不会删除续训文件。可选全量微调完成后保留轻量best.pt。
- --keep_stage_checkpoints 保留所有阶段快照；--keep_completed_resume 保留已完成实验的完整续训文件。默认省空间策略下，中间权重删除后不能再对它们新增评价；已有指标/图/日志不删。需要未来重评时从一开始使用保留开关。
- 保存前估算临时checkpoint所需空间，并额外保留2GiB（--min_free_gb）。不足时明确停止并保留原last，而非自动删除其他数据。这不能代替磁盘容量管理，也不能保证其他进程不会同时写满磁盘。
- 默认不保存best MAE权重，不用val_loss挑研究阶段。训练loss图每5轮刷新，原始epoch日志每轮保存。
- 保存invocations.jsonl中的Git版本、PyTorch、GPU及命令；input_contract.json记录gray_repeat3协议、三通道是否相同及来源；requested_config.yaml/config.yaml记录实参。

## 11. 下载哪些结果

默认本轮根目录 /root/autodl-tmp/outputs_temporal/<run_tag>，其下顶层分 result/ 和 ckpt/。下述日志、汇总、audit、图像均位于 result/；权重按相同实验名独立放在 ckpt/，包括可选全量微调权重。详细目录和旧结果迁移见 [保存与空间说明](temporal_storage_and_resume.md)。

- comparison.csv/md、representation_trajectory.png：阶段曲线与汇总。
- paired_comparisons.csv：等条件记忆/无记忆、匹配基线的EF配对比较。
- stage_times.csv：实际阶段耗时，失败阶段也记录。
- 每组logs、plots、initialization.json、config文件。
- 每个audit/epoch_xxxx下metrics.json、ef_predictions.csv、seg_predictions.csv、temporal_distances.csv、temporal_diagnostics.png。
- trajectory_*.npz保存原始诊断特征和状态，确需深入轨迹分析时再下载。
- result/<模型>/full_finetune保存可选全量微调的日志、曲线；对应权重在ckpt/<模型>/full_finetune/<任务>/。

连跑结束自动生成result/analysis.zip：包括日志/表格/图/配置，不含checkpoint权重，也不含较大的诊断特征NPZ。通常下载这个zip即可；直接下载整个result/也不会带上权重。

## 12. 验证命令与限制

```bash
python -m unittest discover -s tests -p test_temporal_research.py -v
```

测试覆盖模型模式、CPU bfloat16前后向、梯度进入主干/记忆、重置和batch独立性、遮挡像素防泄漏、跨clip因果性、mask预算、官方ROI/median/q-v偏置、NPY数据接口、epoch续训、冻结评估和逐图Dice聚合。若本地官方源码存在，还直接做官方模型数值对照；云端没有资料目录时只跳过这一项，不影响训练。

没有声称已在云端跑完9组或37组，也没有根据这些工程测试预判哪种结构效果最好。
