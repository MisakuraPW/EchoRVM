# CR-EchoMAE 公共脚手架

版本：v1，2026-10-23。代码入口：`python -m research_pipeline`。这份文档讲公共基础设施；不是新的科研路线，也不自动授权训练或选定最终模型。

## 1. 设计边界

模型创新在 `models/`；数据/输出契约在适配器；训练、调优、恢复、缓存、指标、报告只保留公共实现。旧队列不迁移、不热替换、不重跑，原有结果继续按原有协议解释。

v1已接通原生FinalTemporalMAE的EchoNet链路：MAE、冻结线性头、冻结轻量头、全量微调EF/分割、轻量表征几何，以及原有流式/机制审计入口。不是“任意模型、任意数据集即插即用”：新数据仍要做一次经过验证的适配，但不再重写训练循环。

## 2. 一个配置，一套命令

`configs/pipeline/echo_reference.yaml` 为示例配方。阶段按声明顺序执行，`source: '@pretrain/final'` 从已完成阶段解析权重并检查SHA256。所有引用必须指向前面的阶段。

```bash
# 只显示最终解析配置，不训练
python -m research_pipeline plan --config configs/pipeline/echo_reference.yaml \
  --tag pilot --set stages.pretrain.source=ckpt/mae/native_temporal.pt

# 完整链路小预算自检，结果写 pilot_smoke，模型本身不偷换成另一个架构
python -m research_pipeline smoke --config configs/pipeline/echo_reference.yaml \
  --tag pilot --set stages.pretrain.source=ckpt/mae/native_temporal.pt

# 正式运行：不同实验使用不同tag
python -m research_pipeline run --config configs/pipeline/echo_reference.yaml \
  --tag formal_v1 --set stages.pretrain.source=ckpt/mae/native_temporal.pt
```

也可使用 `bash scripts/run_pipeline.sh run ...`，它只是同一CLI的薄包装，不包含另一套超参数。

`--set runtime.micro_batch=8`、`--set runtime.worker_candidates='[0,4,8]'`、`--set stages.ef_ft.training.patience=15` 等使用YAML值；未知字段直接报错。科研参数建议保存在版本化配方，不长期藏在shell历史里。

`--only ef_linear seg_linear` 只运行指定阶段，**不会隐式展开其他阶段**。它们引用的预训练阶段必须已在同一tag下完成；也可将对应 `source` 显式设置为已有原生checkpoint。这样不会为了测试新头而偷偷重训MAE。

参考配方从已有原生checkpoint继续做400个新epoch，不会将source历史训练轮数算入这400轮。最终科研预算和初始化暴露必须在论文中另行锁定。官方VideoMAE权重的初始化可以在新配方中用 `model.config` 构造模型，并指定 `model.init_checkpoint`；适配器复用已有映射器并拒绝编码器缺失。它不等于任意权重都能兼容。

## 3. 自动调优到底调整什么

| 类别 | 自动处理 | 不自动改动 |
| --- | --- | --- |
| 显存/计算 | micro-batch、梯度累积、gradient checkpointing候选 | 编码器大小、帧数、目标函数、观测历史 |
| CPU/IO | 实测候选worker数、spawn、持久worker、预取 | 病例集合、标签、采样定义、输入分辨率 |
| 训练预算 | 固定有效batch，尾部累积按实际样本数加权 | LR、epoch、监督比例、随机种子 |

每个候选在独立子进程中构建真实适配器，先做最大合法上下文压力更新，再做预热和真实DataLoader的完整有效batch更新。计时排除构建/导入与压力测试，保留实测样本吞吐、峰值reserved、成功更新数、失败日志。先比较batch/checkpointing，再在入选组合上比较worker；**是有预算的分阶段搜索，不是全组合穷举或全局最优保证**。

父进程在探测期间将模型移至CPU，试验权重与优化器不进入正式训练。只有OOM允许标记失败并继续；语法、shape、数据、非有限数等错误必须停止，不能伪装成“换个batch就好”。探测超时会记录，不启动正式训练去赌。

目标是端到端samples/s，而不是显存占用率越大越好。默认保留10%可用显存或至少0.5GiB；3%吞吐近似时优先较低显存与较少worker。吞吐、余量、安全性有取舍，不能承诺100%利用率。小头训练低利用率可能完全正常。

只有适配器声明逐样本目标安全，且模型不含BatchNorm，才允许搜索micro-batch。对比损失、跨样本归一化、隐式batch操作需要适配器禁用这项能力。浮点归约、dropout、随机算子仍可能随分组改变，不声称位级等价。

固定精度在科研协议中记录；不会为提速暗中由FP32切换BF16。BF16不支持时直接报错。v1没有默认 `torch.compile` 或把多卡环境当单卡偷偷放大batch；RVM动态循环的编译收益需要独立验证。

## 4. 训练、早停和恢复

- MAE必须 `patience: null`，按指定epoch完成；VAL只做固定掩码诊断，不据此早停或替换最终checkpoint。
- 冻结头/FT的早停只看VAL：EF越小越好，Dice越大越好。`patience`按验证次数计，不是保证按自然epoch计；`eval_every`改变时要一起解释。
- 原生适配器的MAE mask按record/epoch固定，VAL mask固定；采样与增强沿用现有按病例/epoch确定的逻辑。
- 累积梯度按当前有效窗口的真实样本量加权，最后不足一个batch不会被丢弃或少除一个系数。
- scheduler只在成功optimizer更新后推进。AMP若跳过更新，明确失败并保留之前恢复点，不悄悄损耗预算。
- 冻结主干设为eval且不更新，特征缓存禁用随机增强。FT保持端到端梯度，不偷换为缓存特征训练。

恢复是**完整epoch边界**，不是逐batch精确恢复。默认每10轮保存last，任何新的best、早停或结束也保存；意外退出最多重放最近未持久化的epoch。last包含模型、optimizer、scheduler、scaler、RNG、早停状态和完整已提交历史。恢复会回滚CSV中未提交的部分，不叠出重复epoch。

best先写新的带epoch名字的不可变权重，再原子提交引用它的last，之后才清理旧best。磁盘不足或保存中断不覆盖旧last。不能把weights-only的final当成完整恢复点。

完全相同的run命令可以恢复。相关代码、模型、数据清单、头、预算等变更会拒绝同tag续训；需要新tag，或以后显式实现可审计迁移，不能手删哈希强行继续。已完成阶段再次访问会核验结果和协议。

## 5. 监控与问题定位

```bash
python -m research_pipeline status --root /root/autodl-tmp/outputs_cr --tag formal_v1
watch -n 10 'python -m research_pipeline status --root /root/autodl-tmp/outputs_cr --tag formal_v1'
tail -n 30 /root/autodl-tmp/outputs_cr/result/formal_v1/ef_ft/worker.log
tensorboard --logdir /root/autodl-tmp/outputs_cr/result/formal_v1
```

`status.json`区分初始化、缓存提取、调优、训练、验证、保存、结束和异常，记录病例/样本进度、吞吐、CPU等待比例、allocated/reserved。长时间没有心跳表示需检查，不直接认定死锁，更不会自动杀进程。进程存活判断只在同一主机可靠。

队列有所有权锁，同tag同目录只允许运行一个。子任务失败会停止后续任务并保留worker.log；不盲目retry非OOM错误、不操作其他项目、不自动关机。不创建持续产生新会话的巡检任务。

确实慢时使用一次性profiler，而不是仅凭面板猜测：

```bash
python -m research_pipeline profile --config configs/pipeline/synthetic.yaml --tag profile_test --only mae
```

对真实配方可同样指定一个阶段及其source。会在一次性模型上执行三个micro-batch步骤，输出 `profile/trace.json` 和 `operators.txt`，区分取数、前向loss、反向、optimizer。带profiler开销，不当作正式速度。trace可用本地Perfetto/Chrome tracing查看。

## 6. 结果、权重、缓存

```text
<root>/result/<tag>/
  plan.json, queue.jsonl, comparison.json/csv/md
  <stage>/request.json, protocol.json, runtime.json
          worker.log, status.json, events.jsonl, metrics.csv
          observations.json, patients.json/csv, summary.json
          curves.png, previews/, tensorboard/, DONE.json
<root>/ckpt/<tag>/<stage>/
  last.pt                  完整恢复状态
  best_epoch_XXXX.pt       当前任务VAL选出的权重
  final.pt                 最后实际训练轮的权重
  epoch_XXXX.pt            周期快照，非固定指定的默认仅留最近2个
<root>/cache/<tag>/
  frozen/<identity>/       train/val的memmap特征、目标、记录、校验
  <analysis_stage>/        小规模表征数组
```

`training.save_epochs`明确指定的权重不会被滚动快照清理。它们、best、final和last之间可能重复占空间，正式大模型要估算总量；没有“任意模型都肯定不爆30GB”的保证。保存前按待写tensor容量加安全余量检查空间，旧文件写完前仍保留。v1不自动删除source、其他tag或未确认的旧研究权重。

冻结特征共享的key包含编码器权重hash、模型/数据/病例/时间窗口/出口/代码等，不含任务头名称。同一EF或分割协议的线性头与轻量头可以共享，不同出口或历史不能混用。FP32特征落盘、memmap读取，不把全数据集堆进显存或RAM；预算限制整个tag的缓存空间。空间不足显式停止，不默默降级成每轮重跑encoder。提取batch遇OOM可缩小，仅重试这一步。创建时记录SHA256；快速复用检查size/mtime，若怀疑内容损坏应按记录的SHA256全量校验。

```bash
python -m research_pipeline export --root /root/autodl-tmp/outputs_cr --tag formal_v1 \
  --destination /root/autodl-tmp/formal_v1_results.zip
```

只归档result树中的结果/图片/日志，排除权重、NPY/NPZ及嵌套ZIP。包内附文件大小和SHA256，打包后做ZIP完整性检查。可单独下载结果包，权重留云端。

## 7. 新模型怎么接

实现 `your_module:build(config, stage, source, device) -> Experiment`，在YAML改变 `adapter`。接口见 `research_pipeline/contracts.py`：

1. 提供 `model/train/val/task/metadata`；metadata必须给出完整数据身份/划分/排除信息，不只给一个文件夹名。
2. `loss(samples)`返回逐样本均值标量；有跨样本目标就令 `microbatch_safe=False`。
3. `evaluate_batch`共用一次前向返回loss和统一观察行，避免验证每个样本重复编码。
4. `features`输出确定顺序的未掩码特征；只有确定性、无随机增强、冻结的情形才声明 `cacheable=True`。
5. `calibration_samples`覆盖合法最大上下文；`set_checkpointing`明确实现或拒绝，不忽略配置。
6. `model_export`提供模型原生配置/权重；预测行的病例/原帧索引必须可追溯。

新数据集也在这层适配，不更改公共指标。CAMUS/TED的序列轴、ED/ES索引、全周期GT和病例重叠必须单独验证；当前reference adapter不会自动解释它们。外部观察行可用公共 `metrics`/`compare`命令先做同口径分析。

## 8. 验收与限制

已提供CPU单元、故障注入和文件级原生模型测试；CI运行这些契约及synthetic冒烟。实际测试结果随交付说明记录。CUDA调优/AMP、服务器真实数据吞吐仍需云端冒烟，CPU通过不能替代。

本次本地验收环境为Windows、Python 3.10、PyTorch 2.9 CPU。20项测试通过；原生文件级测试内部覆盖MAE、两任务的linear/lite/FT、冻结权重不变、缓存共享、流式和机制审计。另执行完整synthetic队列、profiler、状态查询及result-only ZIP导出检查。未连接或改动服务器任务。

v1暂不提供多卡、自动学习率搜索、强制编译、云端自动部署/杀进程、任意外域自动接入、正式TEST解封或自动生成论文结论。公共基础设施应该稳定可复用，而不是把尚未验证的能力写进功能清单。
