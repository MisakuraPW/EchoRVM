# 现有B0权重的读出与微调配方校准

## 本次授权与范围

使用 `cr_forward_fast_v3/pretrain/epoch_1600.pt`，不新增MAE训练，不改空间网格、不改RVM/近期缓存、不重跑A。两个正式FT任务前各有一轮小数据软件冒烟。旧队列完整退出后自动接续，旧工作目录不pull、不改文件。

这是一组开发用的task适配配方改进，不是单因素消融，也不是EchoCardMAE完整复现。后续不能把收益单独归因于某一个头部模块。

## 用户提供的两个ZIP

路径均位于微信 `2026-11` 文件夹。只读取JSON和文本，没有执行包内内容。

| 用户文件 | VAL Dice（选定模型） | TEST Dice | 末轮训练Dice | 训练停止轮数 |
|---|---:|---:|---:|---:|
| pretrain with 中值滤波.zip | 0.9285 | 0.9288 | 0.956 | 51 |
| 无频率处理pretrain-baseline.zip | 0.9287 | 0.9294 | 0.948 | 56 |

第二包日志输出目录为 `outputs/without_median_filter_seg`，与用户说明一致。TEST差值0.0006，即0.06个百分点，去滤波略高。单次结果且没有源码/配置/权重指纹，不能视作严格控制所有变量的滤波消融；未提供EF日志或SSL曲线。

日志按epoch输出只有三位小数；末尾汇总是所选checkpoint的训练/验证指标，不是最后一轮训练指标。不能用三位小数的相同loss反推精确最佳epoch。基于停止规则可推测最佳约25/30轮，但缺early_stop.json，不把推测写成精确记录。

它们体现：官方配方训练Dice并非早早到99%，VAL约在前20~30轮进入92.8%附近平台。不能把其中TEST92.94%与我们VAL病例平均91.71%直接相减解释为真实掉点；我们同批预测global为92.34%，但cohort仍不同。

## SSL是否继续值得长训

本轮B0已完成的实际曲线：

| epoch | train重建loss | 固定已见病例重建诊断 | 冻结EF MAE | 冻结seg Dice |
|---|---:|---:|---:|---:|
| 400 | 0.423636 | 0.416314 | 5.2124 | 80.2746% |
| 800 | 0.398439 | 0.389818 | 5.1079 | 79.7558% |
| 1200 | 0.376097 | 0.366541 | 未测 | 未测 |
| 1600 | 0.367164 | 0.358274 | 5.1224 | 78.4580% |

不能说后期loss完全不降：400→1600训练loss还下降约13.3%；1200→1600仍下降约2.4%。真正的问题是额外重建收益没有持续转换为冻结任务收益。因此不建议把每轮观察更多帧、全数据集、更多epoch一律等同于“更接近表征上限”。目前没有单因素证据证明多clip或更多无标签数据无用，也不据此取消递归上下文。

下一轮若需要新SSL，优先考虑把算力转向14x14空间网格（patch8）；当前已经7x7（patch16）。不能改成7x7后称作提高空间分辨率。是否新训，由此次读出适配结果决定，本代码不会自动触发新SSL。

## 两项FT

| 阶段 | 相对原B0 FT的修改 | 回答的设计决定 |
|---|---|---|
| seg_decoder_transfer_ft | 192维4层3头，迁移native SSL decoder blocks/norm/encoder_to_decoder；新可学习2D位置、新像素分类器；前景+背景Dice；WD1e-4、固定LR1e-4 | 是否应恢复完整解码器与迁移路径，而不是认为当前encoder已经到极限 |
| ef_l1_ft | 保留64维attention+MLP及近期缓存；最终层零权重、bias=.556；百分数L1；WD1e-4、固定LR1e-5 | 较稳健的EF目标和初始化能否改善现有表征的适配 |

两者统一最多100epoch，patience10、min_delta1e-4；分割仍按VAL病例平均硬Dice选模型，EF按VAL MAE；原cohort、位置展开、灰度重复、输入标准化、A4增强、clip norm1和有效batch32不变。没有冒充完全采用官方crop增强/早停/单帧任务。

数据人口不变，因而可以与当前B0 FT配对比较。TEST保持不开放；用户旧TEST仅作已提供的历史报告，不用于本轮checkpoint选择。

迁移时将native separate q/v bias合并成task head的完整qkv bias，中间k bias为0；共享层严格load，尺寸不符报错，不静默跳过。沿用标准attention行为，不复制作者公开代码忽略q/v bias的实现。

### 加速、保存与恢复

- 沿用公共pipeline：每阶段单独探测microbatch、workers、CPU线程、gradient checkpointing，以有效更新吞吐选安全配置；不自动搜索LR或改变有效batch。
- BF16；原生native112；FT不能把encoder特征缓存固定后冒充全量更新。
- 相同CSV/JSON、逐步心跳、曲线、VAL预测及指标；每epoch保存last；保留最佳和final，不累计无用周期快照。
- `outputs_cr_clean/result/cr_readout_calibration_20261101/` 与 `ckpt/cr_readout_calibration_20261101/` 分离。
- 再次执行同命令可从last恢复，科学配置或代码身份改变必须新tag。

## 启动与排队

普通入口（只有在既有队列不再使用该代码目录时才能更新代码）：

```bash
python -m research_pipeline smoke --config configs/pipeline/readout_calibration.yaml
python -m research_pipeline run --config configs/pipeline/readout_calibration.yaml
```

本次服务器部署使用独立Git源码快照目录 `EchoRVM_readout_calibration_<commit>`，共享只读数据和来源checkpoint，避免正在运行的旧队列因代码身份改变而失败。

排队入口：

```bash
python -u tools/queue_readout_calibration.py \
  --wait-pid 18776 \
  --previous-result /root/autodl-tmp/outputs_cr_clean/result/cr_forward_fast_v3
```

launcher等待原队列PID及其启动身份消失，核对原队列所有选定产物（A允许声明的400轮预算暂停），再等GPU没有其他compute进程，然后冒烟和正式启动。如果原队列失败或GPU另有项目任务，不杀进程、不跳过失败工作去抢卡。

这次仅安排上述两个FT，不自动增加更多种子、无记忆对照或新的预训练。

## 验证

本地CPU：新增三项测试覆盖共享decoder迁移及encoder反向、EF初值及百分数损失、constant调度保存；公共pipeline回归19项通过。云端正式训练前另跑真实checkpoint/数据/GPU冒烟；本地通过不等于云端已通过。

### 2026-11-02 接续故障修复

旧队列33项已完成（A按400轮预算暂停是预期）。接续程序以脚本路径执行时，`chdir`没有改变Python的`sys.path`，导致等待结束后导入`research_pipeline`失败，两项校准均未启动。已显式加入脚本所在项目根目录，并新增“仓库外工作目录、真实脚本入口”的进程级回归测试；四项读出测试通过。部署只替换独立快照中的launcher，不改训练/模型代码或原队列；原失败日志保留后追加修复记录。
