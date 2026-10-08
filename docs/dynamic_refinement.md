# 心超动态表征深化实验

这一轮先复用已有100轮权重，诊断局部特征、融合特征、帧展开、近期缓存和状态与医学运动的关系；然后只新增两组100轮预训练，验证已有候选的组合与一个动态分解方案。默认不做周期性任务头训练、不自动进入400/1600轮或全量微调网格。

## 运行

在云端项目目录和原来的环境中：

```bash
cd /root/autodl-tmp/EchoRVM
conda activate echocardmae
git pull
bash scripts/run_dynamic_refinement_smoke.sh
RUN_TAG=dynamic_refinement_20261008 bash scripts/run_dynamic_refinement.sh
```

冒烟是两组各2个训练step，随后跑小病例诊断、EF读出、全位置分割和80帧流式检查；输出名称自动加 `smoke_`。正式版只在最终权重做上述评估。冒烟与正式训练不会共用结果目录。

只先做已有模型的冻结诊断：

```bash
RUN_TAG=dynamic_diagnosis_20261008 bash scripts/run_dynamic_diagnostics.sh --no-endpoint
```

该命令只提取表征并拟合便宜的训练集PCA轴和ridge图像内容读出，不训练MAE，也不训练EF/分割头。默认完整版本会为所有模型做一次EF和全位置分割终点评估，以避免比较不同读出覆盖。

同一个 `RUN_TAG` 重跑原命令即可继续：完成阶段按协议检查后跳过，未完成训练自动从 `last.pt` 恢复；患者特征缓存按病例恢复。一个 `run_tag` 同时只允许一个队列进程。代码或科学参数改变导致校验失败时应使用新tag，不要删除校验文件强行混用。

## 实验清单

| 模型 | 使用方式 | 本轮新增训练 |
|---|---|---|
| baseline | 第一阶段L16空间记忆 | 无 |
| none | 第三阶段分段编码、近期缓存、无RVM | 无 |
| temporal_pool | 第三阶段时间感知压缩 | 无 |
| learned | 第二阶段可学习tubelet展开 | 无 |
| combined | learned展开 + temporal_pool，保留融合特征写入及现有门控 | 100轮 |
| factorized | combined的空间参考 + 低秩动态变化读出 | 100轮 |

以上三项旧候选全部保留并记录来源哈希；组合重新从相同VideoMAE初始化训练，不能合并两份独立训练权重作为已经验证的组合。

两组新模型的共同encoder、memory、原learned展开以及decoder参数在相同种子下保持相同初始化；新增基矩阵或query注意力的随机初始化不会推进公共参数的随机数流。后续自动微batch和重计算调优仍可能改变浮点运算与mask分组，不能宣称整个训练逐bit一致。

`tubelet1` 是可选已有权重控制，`query` 是可选新增局部时间查询方案，不进入默认队列：

```bash
# 只增加一个已有tubelet1控制，默认新训练仍然两组
bash scripts/run_dynamic_refinement.sh --controls baseline none temporal_pool learned tubelet1

# 明确选择新增查询实验时才加这第三组训练
RUN_TAG=dynamic_with_query bash scripts/run_dynamic_refinement.sh --variants combined factorized query
```

查询方案用每个空间位置的帧查询读取本地clip的tubelet序列，保留空间网格，不重新编码每个单帧。它增加参数与计算，参数量会如实记录。既有learned、factorized和query共享帧级MAE解码接口；factorized/query严格逐局部clip展开，避免64帧后处理引入跨clip未来信息。

## 权重入口

默认来源：

```text
/root/autodl-tmp/outputs_stage1/stage1_lengths_20260928/ckpt/spatial_l16/epoch_0100.pt
/root/autodl-tmp/outputs_stage3/stage3_mechanisms_20260929/ckpt/none/epoch_0100.pt
/root/autodl-tmp/outputs_stage3/stage3_mechanisms_20260929/ckpt/temporal_pool/epoch_0100.pt
/root/autodl-tmp/outputs_stage2/stage2_l16_20260929/ckpt/learned/epoch_0100.pt
ckpt/mae/videomae_vit_s.pth
```

可以用 `--stage1_root`、`--stage2_root`、`--stage3_root` 改整个来源根目录，或指定单个来源：

```bash
bash scripts/run_dynamic_refinement.sh \
  --checkpoint learned=/path/to/learned/epoch_0100.pt \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic
```

旧控制权重需包含保存的训练配置且完成100轮。队列比较科学配置，忽略worker、微batch调优等执行差异，但会核对有效batch、结构、优化器等。错误权重不会静默回退到随机初始化。诊断-only不要求初始化文件存在。

## 训练与存储

沿用第三阶段协议：L16，总64帧，112×112，灰度NPY缓存与三通道模型接口，patch8、tubelet2、mask0.75、AdamW1e-4/weight decay0.05、有效batch32、固定学习率、无增强、无早停。这样本轮组合与机制对照不会同时更换优化器、采样、增强或训练预算。

`factorized` 在每个局部clip的每个空间patch上，将可学习展开后的特征拆为完整维度的空间参考，以及rank16动态分量。写作 `y_t = reference_clip + B * phi_t`，其中 `reference_clip = mean(expanded) - B * mean(phi)`。动态坐标 `phi_t` 使用全模型共享的映射，不逐clip强制去均值，避免人为制造跨片段相位重置。参考保留空间网格；只约束所有病例共享的动态子空间，不把全部医学空间特征压成二维。单个16帧窗口本身的矩阵秩上限只有15，不能用其低秩图像单独证明该约束有效。基矩阵正交惩罚权重默认0.001，日志中分别保存重建loss、原始正交loss和加权正交loss。

这借鉴LRM-Functa的结构/变化分离，不是其复现：参考是逐clip的，编码器直接提取特征，不为每个测试视频额外优化latent。没有添加心动周期监督、未来预测或强制圆环loss。

默认输出：

```text
/root/autodl-tmp/outputs_dynamic/RUN_TAG/
  result/
    matched_manifest.json
    preserved_components.json
    current_stage.json
    stage_times.csv
    comparison.md
    dynamic_comparison.csv
    dynamic_paired.csv
    task_comparison.csv
    analysis.zip
    MODEL/{medical,ef,positions,...}
  ckpt/
    combined/{last.pt,epoch_0100.pt}
    factorized/{last.pt,epoch_0100.pt}
  cache/
```

`last.pt` 每10轮覆盖一次，最终另存权重快照；没有中间epoch快照或best，缓存完成后删除，仅清理本轮自己创建的病例缓存。恢复快照含优化器等状态，最终评估快照沿用现有轻量保存接口。默认至少保留3GB磁盘安全空间。可显式修改 `--save_last_every 20 --min_free_gb 5`。旧实验权重不复制、不删除。

## 新增评价

医学诊断默认128训练患者、128验证患者，每段192个真实连续帧，且同时包含两张官方描迹。不填充或重复凑长度，筛选记录及原始起点固定，各模型共用同一manifest。FPS、病例文件stat、原始帧索引和描迹来源均记录；不具备192帧条件时明确减小 `--audit_frames` 或病例预算。较短的视频窗口也降低可见周期数，应单独解释。

| 指标 | 回答问题 | 解读边界 |
|---|---|---|
| 时间分辨率、有效秩、二维解释方差、速度/加速度 | 哪些对象在随时间变化、是否退化 | 低秩、平滑和PCA曲线漂亮都不自动代表医学正确 |
| 0.4–1.5秒滞后内的自相关候选峰 | 是否有周期复现信号 | 是探索用范围，不是已验证心率真值 |
| ED/ES事件定位误差(ms与帧)、检出率、全病例100ms命中率 | 潜在运动与标注事件是否对应 | 无候选计漏检；MAE仅对检出病例算，必须同时读覆盖率 |
| 候选数/秒 | 低误差是否依赖过多事件候选 | 其他心动周期没有完整标签，不能把所有额外候选当假阳性 |
| 相邻两帧内容配对准确率及tie比例 | 不同输出是否携带真实帧内容 | 相同预测只得0.5；近乎相同原图对单列排除 |
| tubelet内交换图像挑战 | 表征是否响应真实图像交换 | 不是只检查learned的两个偏置不同 |
| 跨clip边界跳变与原图跳变 | 是否有分块引入的表征不连续 | 比较同类tubelet过渡，避免repeat造成的零步长干扰 |
| 亮度扰动后的动态一致性及亮度变化相关性 | 动态是否过度追随图像亮度 | 诊断信号，不直接证明结构/运动完全解耦 |
| 静止重复输入的动态能量和自相关 | 周期是否来自位置编码、分块或记忆更新本身 | 静止输入仍可有状态启动漂移，需与真实视频对照解读 |
| 全16位置共享分割头及流式检查 | 任意目标位置读出及边界 | 一个头训练覆盖所有位置，仍只验证ED/ES有标签帧 |

事件方向来自训练集：对训练运动增量拟合共享PCA轴，用训练描迹面积派生ED/ES确定一次正负方向。验证集不允许按GT交换峰谷或调阈值。因此这叫“训练标定的事件诊断”，不能冒称完全无监督ED/ES检测。

PCA与双向滤波用于离线分析，不能报告为无延迟在线事件预测。state每16帧更新，其事件分辨率天然更粗；目前用空间均值描述符，结果不证明完整状态信息是否充分。cache轨迹是最近4段描述符平均，不能代表其全部有序内容。

图像内容读出只在训练患者拟合ridge到8×8灰度描述符，验证患者做同一tubelet内正确/交换配对，去除每帧平均亮度。它是轻量内容诊断，不是高质量重建或分割的替代。原图差异极小的帧对不能给出可靠身份判断，数量会报告。

终点EF仍是512/256有效病例、400步冻结头；分割64/64患者、200步共享头，覆盖所有16个位置。它们与医学诊断是不同队列，跨模型各自保持匹配；不把冻结Dice直接与全量微调成绩比较。bootstrap只反映病例不确定性，不代表MAE训练种子波动。

## 性能调节与监控

正式预训练默认 `--autotune`，先用隔离的真实训练trial选择微batch、累积、worker和重计算，保持有效batch32及epoch样本预算。冻结提取也自动选择batch/worker，留显存余量。只调执行参数，不根据验证分数搜索结构或学习率；利用率不能保证恒定100%。

```bash
cat /root/autodl-tmp/outputs_dynamic/dynamic_refinement_20261008/result/current_stage.json
tail -f /root/autodl-tmp/outputs_dynamic/dynamic_refinement_20261008/result/combined/logs/train.log
watch -n 1 nvidia-smi
nvidia-smi dmon
htop
df -h /root/autodl-tmp
```

正式版队列未后台启动，可用已有服务器会话规则运行。断开连接的管理方式沿用云端自己的规定。整个队列顺序执行，不关机、不终止其他项目；运行时间由 `stage_times.csv` 记录。只下载 `result/` 或其中 `analysis.zip` 即可分析。

## 来源

- 已完成三阶段报告：`outputs/stage3_mechanisms_20260929_analysis/时域三阶段探索_最终汇总_20260929.md`。
- LRM-Functa：<https://arxiv.org/abs/2603.25951>，借鉴动态分解及运动轨迹分析，不移植其逐视频拟合流程。
- 前一轮明确保留的候选、帧读出覆盖结果和分割取舍写入 `preserved_components.json`，没有修改协作Excel或宣布新模型已胜出。
