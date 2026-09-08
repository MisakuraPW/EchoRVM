# 时域研究：保存、续训与磁盘空间

更新：2026-09-09，仅描述 scripts/run_temporal_research.sh 的新默认行为。其他历史脚本未强制切换目录；两个公共trainer支持显式 --checkpoint_dir 和 --save_last_every。

## 保存与清理

旧逻辑是每轮覆盖last.pt，保存第0/50/100/.../400轮快照，不自动删除，权重混在每个实验的结果目录中。

| 文件 | 频率 | 内容 | 自动清理条件 |
|---|---|---|---|
| last.pt | 第1轮、每10轮、评价快照轮、最终轮；下游早停时也保存 | 模型、优化器、scheduler、AMP scaler、epoch/step、RNG、早停状态 | 本实验所有请求阶段成功完成后清理；运行中保留最近一份 |
| epoch_0000.pt、epoch_0050.pt ... | 0轮及每50轮 | 模型与配置等，不含优化器 | 对应冻结评估DONE且metrics.json存在后删除；第400轮例外保留 |
| epoch_0400.pt | 预训练结束 | 轻量最终模型 | 默认不删除，供下游微调/后续使用 |
| interrupt.pt | 捕获Ctrl+C时尝试保存 | 部分epoch的紧急快照 | 完成后清理；不自动作为精确续训点 |
| 下游best.pt | 可选全量微调验证指标改善时覆盖 | 轻量最佳模型 | 默认不删除 |

400轮是默认终点；修改--epochs时，保留的是所设最终轮快照。非整50终点同样会保存和评估。

每50轮的评价口径没有减少。当前仍是每组先完成全部预训练，再逐个快照评价，因此正在训练的一组最多暂存9个阶段快照和last。不会在还没完成相应评价时删除阶段权重。

自动清理只作用于这个run_tag、指定实验ckpt目录的明确文件名；不扫描删除其他项目、旧实验、数据缓存或原始数据。清理记录在对应result/<实验>/checkpoint_cleanup.jsonl。

## 新目录

```text
/root/autodl-tmp/outputs_temporal/<run_tag>/
  result/
    comparison.csv
    comparison.md
    stage_times.csv
    analysis.zip
    echocardmae_video_port/
      requested_config.yaml
      config.yaml
      PRETRAIN_DONE
      logs/
      plots/
      tensorboard/
      audit/
        epoch_0000/
        epoch_0050/
        ...
      full_finetune/
        echonet_ef/
        echonet_seg/
    hier_global/
    ...
  ckpt/
    echocardmae_video_port/
      last.pt             # 未完成时保留
      epoch_0400.pt       # 完成后保留
      full_finetune/
        echonet_ef/best.pt
        echonet_seg/best.pt
    hier_global/
    ...
```

下载result/即可获得全部分析资料，不会携带ckpt。只需要汇总、图和日志时，下载result/analysis.zip更小；该zip还排除了诊断特征NPZ。

## 续训与保留开关

同一运行用原命令重启即可：保留run_tag、data_root、模型训练参数不变。runner跳过已完成步骤，从新ckpt目录找last。

每10轮保存意味着硬关机可能丢失最多约10轮工作；不支持精确恢复中断batch。last恢复最近保存的完整epoch、优化器和随机状态，再训练后续轮次。日志中可能保留中断前重算轮次的历史记录，分析应以epoch/step及最新记录辨别，而非把日志行数当训练轮数。

Ctrl+C的interrupt带partial_epoch标记，不能用它跳过未完成epoch继续训练；resume时指定last.pt。旧版本权重缺少早停计数时，无法凭空恢复旧计数。

显式更改频率，不改yaml：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 --save_last_every 20
```

若希望400轮后精确延长训练，或以后对中间阶段新增探针，务必事先保留相应文件：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 --keep_stage_checkpoints --keep_completed_resume
```

这两个开关增加占用，且不能恢复已经删除的文件。省空间默认只保留最终预训练模型和可选下游best，够推理及新建下游训练，不等于保留原优化器的延长训练状态。

## 旧目录迁移

停止旧进程后，使用同一run_tag、完全相同的训练参数，加--migrate_legacy_layout。以旧数据来源为RGB缓存的运行举例：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_v1_s42 --input_protocol rgb --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic-rgb --migrate_legacy_layout
```

只迁移上一版本的时域runner，不迁移历史无效baseline400实验。它将该run_tag的原实验目录拆分到result/、ckpt/，同一文件系统内移动，不额外复制权重。遇到目标同名文件拒绝覆盖；不在旧训练仍运行时操作。checkpoint内部历史路径和旧日志保留，不改写已有实验证据；后续训练使用新的显式路径。

未加迁移参数而发现旧布局会报错，不会悄悄重跑。只改存储路径/保留策略不改变训练协议；换data_root或batch等仍需新run_tag。运行中由RGB缓存切换原AVI虽然理论上像素可一致，当前不会未经核验绕过协议检查。

## 不生成RGB缓存

按用户最终选择，默认复用/root/autodl-tmp/datasets/EchoNet-Dynamic中的灰度NPY，协议gray_repeat3，采样后在内存扩展三通道，不改模型结构或初始化权重。无需RGB目录；改变模型、mask比例或clip长度通常不需重建此缓存。

可选读取/root/autodl-fs/datasets/EchoNet-Dynamic的原始AVI后转灰度；需根目录含FileList.csv、VolumeTracings.csv与Videos/。本轮推荐直接用本地灰度NPY：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 --smoke --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic
```

冒烟通过后去掉--smoke跑正式版；smoke自动用独立smoke_前缀目录。

灰度协议已启用，训练和评价统一使用。保留三通道模型的权重结构与mean/std，不把首层权重求平均改为单通道。它不是原始RGB逐像素复现，因此必须使用新run_tag，不能接续或混入原RGB实验。

原有灰度目录现在就是本轮训练数据，请保留。它也可能被其他项目引用，不能整目录删除。本轮不会生成或自动删除任何数据缓存。

用户已经删除RGB缓存也不影响新灰度实验。若另有旧RGB实验，其原续训命令不再适用；不能给同一run_tag换输入来源继续当作原实验。

保存checkpoint时额外要求2GiB空闲储备，另计即将写入的临时文件。--min_free_gb可修改储备。这是提前报错保护旧last，不是实际空间预留或无限容量保证。日志、其他项目、数据缓存也会占空间；先检查df -h与df -i。
