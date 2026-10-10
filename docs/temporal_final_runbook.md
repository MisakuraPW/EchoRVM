# 时域问题1—3：最后一轮有限实验运行说明

研究合同见 `时域最后一轮探索_Q1-Q3实验计划_20261010.md` v2。本入口只完成问题1—3；不自动拼接正式MAIN，不启动问题4或400/1600轮新预训练。

## 一、服务器启动

在服务器项目目录、原训练环境中运行：

```bash
git pull --ff-only
bash scripts/run_temporal_final_smoke.sh
RUN_TAG=temporal_final_20261010 bash scripts/run_temporal_final.sh
```

两条脚本默认使用同一真实EchoNet数据路径和已有P/C/F100权重。冒烟使用独立run tag、每split两名真实有效患者、两次适配更新及一轮任务训练，并强制走B7/B8实现分支；不会作为科研结果。正式运行先进行训练集测速、登记预算，再运行任务；不是先看验证分数再调整资源/学习率。

2026-10-10旧P权重兼容修正：旧配置省略`frame_readout`时按原`TemporalMAE`的`repeat`处理，而不是新建learned展开；源P/C/F的全部tensor在预热前核验，不允许跳过decoder或随机补帧模块。修复前已经失败的冒烟目录有旧源码协议，保留它但换新的冒烟tag运行：

```bash
git pull --ff-only
RUN_TAG=temporal_final_smoke_legacyfix_20261010 bash scripts/run_temporal_final_smoke.sh && \
RUN_TAG=temporal_final_20261010 bash scripts/run_temporal_final.sh
```

这里正式tag不变的前提是原来的`&&`已因冒烟失败而没有启动正式队列。如果正式tag已写入旧代码协议，也必须另取新tag；不要删除协议文件或绕过核验。

数据尺寸核验使用实际AVI/NPY像素形状，不把`FileList.csv`里的`FrameHeight/FrameWidth`直接当发布文件或GT坐标尺寸。服务器已核对TRAIN中5例声明768x1024/1040但实际AVI/NPY均112x112、没有分割描迹的记录；这些病例正常保留到MAE/EF人群。清单记录`declared_spatial_shape`、`dimension_metadata_mismatch`及split汇总计数，实际非112x112的文件仍拒绝，GT不因CSV声明尺寸缩放。

默认数据根：`/root/autodl-tmp/datasets/EchoNet-Dynamic`。沿用灰度NPY缓存，采样后在内存复制三通道，不创建RGB数据盘。需要 `FileList.csv`、`VolumeTracings.csv` 与对应视频/NPY；TEST不读取。缺真实数据或权重直接报错，不回退synthetic/random。

默认权重：

| 角色 | 路径 |
|---|---|
| P temporal_pool | `/root/autodl-tmp/outputs_stage3/stage3_mechanisms_20260929/ckpt/temporal_pool/epoch_0100.pt` |
| C combined | `/root/autodl-tmp/outputs_dynamic/dynamic_refinement_20261008/ckpt/combined/epoch_0100.pt` |
| F factorized | `/root/autodl-tmp/outputs_dynamic/dynamic_refinement_20261008/ckpt/factorized/epoch_0100.pt` |

如果服务器实际位置不同，显式传入，不用改代码/YAML：

```bash
RUN_TAG=temporal_final_20261010 bash scripts/run_temporal_final.sh \
  --checkpoint P=/实际路径/P.pt \
  --checkpoint C=/实际路径/C.pt \
  --checkpoint F=/实际路径/F.pt \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic
```

正式入口核对epoch100、L16/112/patch8/tubelet2/ViT-S、spatial以及角色对应的帧模块；同名文件不代表来源正确。run tag锁定代码、数据清单、源权重和科研协议，不能混用不同版本结果。

## 二、执行顺序与上限

1. 数据/权重核验、训练集测速、预算冻结。
2. P/C/F：各冻结EF/分割、各全量微调EF/分割，共12个主要校准任务；复用已完成且哈希一致的结果。
3. C/F：局部基础、融合基础、最终出口的同容量冻结读出；C的基础与最终相同直接复用，不重复拟合。EF与分割均检查。
4. B0 spatial、B1 global；分层历史、机制和表征诊断；只有内容相关互补证据才允许一次B8 mixed。
5. 在入选组织下B2 local-write、B3 late-read。记录B_org→B2及B2→B3的单因素比较，再选择完整读写配置。
6. 在共同读写背景下B4 rank32候选写入、B5 factorized、B6全维shrink。B7固定beta0.5仅在预登记条件满足时运行一次。
7. 最多一个关键对照补四个FT任务和一套head seed43；不另训MAE种子，不因区间跨零自动追加实验。
8. 最多两个条件入选模型做实际长流、FIFO、reset、目标帧身份、延迟及有限存储检查。统一输出Q1—Q3关闭表与边界；Q4仍待定。

B0—B6必须跑，B7/B8最多各一次。每个B都从同一个C100与登记的新模块初始化出发，不从已适配的胜者额外多训。正式默认1500次成功optimizer更新、有效batch32、lr1e-4、AdamW、只重建末64帧、真实H0—128可变前缀、前缀保留梯度、不做增强/早停。条件模型最多九个，不展开网格。

共同冻结头最多60epoch/patience12，FT80epoch/patience20。任务配方锁定：EF冻结1e-3、FT5e-5；分割冻结3e-4、FT1e-4。EF归一化只拟合TRAIN。FT统一A4在线增强，clip一致、image/mask同zoom；冻结特征缓存与VAL不增强。

## 三、自动运行参数选择

默认开启，逐模型/任务独立测速，不按GPU利用率截图瞎选：

- CUDA前后向测速选择有显存余量、吞吐较好的micro-batch；有效batch不变，梯度累积补足，变长H按bucket加权累积，不补帧。
- 数据加载测试在训练数据上比较有限worker候选，启用pin memory、persistent workers和prefetch4。
- AMP、梯度裁剪1、统一gradient checkpointing；不逐模型改变学习率、mask ratio、模型宽度、训练轮数或科学有效batch。
- 测速恢复权重/RNG、丢弃试训权重；真实训练的选择与试验记录写入各job的runtime/训练protocol。
- 保留显存余量，不保证100% GPU/CPU占用。冻结小头、IO、绘图和评估阶段低利用率本身不证明故障。

默认worker候选上界8，可用 `--num_workers 12` 改上界；没有必要把CPU线程或worker越加越多。任务有效batch与学习率属于研究配方，本入口不提供按模型调精度的搜索。

```bash
# 先只核验/测速并打印本机保守预算，不开始计分任务
RUN_TAG=temporal_final_20261010 bash scripts/run_temporal_final.sh --preflight_only

# 可选锁定总计划预算上界（小时），不要照抄一个未经测速的数
RUN_TAG=temporal_final_20261010 bash scripts/run_temporal_final.sh --max_gpu_hours 48
```

注意：预算参数属于run tag协议，若正式准备使用上界，`--preflight_only`也要带同一个值。预算超出时只允许统一减少B适配更新（最低750），关闭可选FT/第二head种子；不按模型分数分别减训练。最低预算仍超限则在计分前报错。估算为保守计划值，不是结束时刻保证。

## 四、存储与续跑

```text
/root/autodl-tmp/outputs_temporal_final/
  result/RUN_TAG/       # 清单、协议、日志、CSV、PNG、报告、DONE校验
  ckpt/RUN_TAG/         # 各job的权重
  cache/RUN_TAG/        # 有界一次性特征缓存
```

只下载 `result/RUN_TAG` 及其分析压缩包；不包含权重。task结束并验证DONE后删除该任务自己的缓存；不删数据、源权重、其他run或项目。适配覆盖保存last每300成功更新，末尾保存final；不堆积逐epoch历史权重。任务last每epoch、best只在更优时覆盖。默认保留5GiB空闲，不足会在保存前报错且保留上一个有效checkpoint。

### 冻结验证缓存升级

冻结骨干、无验证增强时，现在同时临时缓存训练与验证的FP32原始特征，预测/loss/Dice仍由每轮当前任务头重新计算。验证人群、全部16位置、精度、早停与训练参数不变。全量微调不跨epoch缓存特征。这里的数据集特征缓存位于CPU RAM/临时磁盘，与模型的四clip FIFO和GPU工作显存不同。

默认每job磁盘上限16GiB、全进程RAM缓存总预算4GiB；23648份验证网格与8108份训练网格的纯特征约9GiB，加入GT与文件开销接近12GiB，因此给16GiB余量。每job结束清理，不累计保存所有模型的特征。低磁盘余量时跳过写缓存并回退推理，保留5GiB给checkpoint。缓存不量化到FP16，也不缓存任务预测；首次填充仍需编码，后续epoch才省去反复编码。

已有`01daeb3`队列只允许在已完成任务边界进行一次明确的执行升级：`--adopt_validation_cache --cache_disk_gb 16`。入口严格检查原代码哈希、源权重、清单、训练预算、所有完成结果/权重哈希；数据或训练规则变化仍拒绝。升级记录保存到`operations/validation_cache_upgrade.json`，旧protocol备份保留。已完成的旧任务不重训，运行时间对比必须注明是否缓存，不能把这种加速说成架构优势。

本轮现场由`tools/handoff_temporal_validation_cache.py`等待当前P冻结分割正常完成，先做真实GPU缓存等价证书，再接续同一队列。它只结束已经暂停的调度父进程，不给当前训练worker发信号，不影响其他项目。接续日志在`operations/continued_queue.log`，状态在`operations/handoff_status.json`，证书在`operations/validation_cache_certificate.json`。不是绕过任何科学协议守卫；不需要用户再启动一份重复队列。

同一命令和RUN_TAG重新运行就是续跑：已完成job核对文件/权重哈希后跳过；适配恢复model/optimizer/scaler/RNG/采样cursor；任务恢复有效更新边界与epoch、scheduler、best及早停状态。未完成的只读诊断可以重算。源码、数据或科学参数变化要求新run tag，不能把不同实现悄悄续到一套结果里。

## 五、查看进度与结果

```bash
python tools/run_temporal_final.py --run_tag temporal_final_20261010 --status
cat /root/autodl-tmp/outputs_temporal_final/result/temporal_final_20261010/current_stage.json
tail -f /root/autodl-tmp/outputs_temporal_final/result/temporal_final_20261010/B0/adapt/logs/train.log
nvidia-smi dmon -s pucm -d 1
htop
df -h /root/autodl-tmp
```

`budget.json`是测速预算；`stage_times.csv`是真实各阶段耗时；各task有患者预测、归一化、protocol、曲线；各model的representation/mechanisms/history为配对诊断；最终`comparison.csv`、`decisions.json`与`Q1-Q3_closure.md`说明采用/不采用和未确认边界。EF pp与患者Dice分别报告，不相加为一个总分。H0仍有近期窗口内的状态，不称无记忆；首16/32/48帧可输出但不能称EF已可靠。

## 六、实现复核边界

局部CPU测试覆盖真实文件索引/掩膜、旧权重兼容、机制前后向、hidden-pixel隔离、prefix内容梯度、帧展开次数、FP32身份诊断、患者级配对、预算条件与精确续训。完整小尺寸CPU队列只验证工程流程，不验证科研性能或4090显存。服务器必须先跑真实权重/真实数据CUDA冒烟；CUDA混合精度、吞吐与实际峰值以该结果为准。

本地复核覆盖148项新检查：145项通过，3项CUDA检查因无GPU跳过；其中一处新增测试用例的变量作用域错误修正后，对应6项模型检查全部复跑通过。完整队列含条件实现分支、真实文件型测试夹具、冻结/FT、历史/机制/表征/流式诊断，以及第二次运行无新增worker的复用检查。小尺寸夹具不是临床病例成绩，不进入科研比较表。

独立复核修正了R5正负号与患者字段合并、验证窗口的种子隔离、精确源帧/位置人群哈希、详细R重复登记，以及确认失败时按对应组织/读写/候选/帧设计轴回退的问题。参数选择不是把EF和Dice相加排名；异常机制检查会阻止后续选型。源码变化也会阻止同一队列混入另一版本。
