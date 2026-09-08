# 本轮灰度训练：直接运行

默认复用 /root/autodl-tmp/datasets/EchoNet-Dynamic/npy/ 的旧灰度缓存。已删除的 EchoNet-Dynamic-rgb 不再需要，也不会重新生成。

## 固定的输入协议

- data.input_protocol=gray_repeat3，所有9组主实验或37组完整矩阵统一使用。
- NPY仍是uint8 [T,H,W]，mmap只读，先采样clip再扩展成[T,3,H,W]并缩放到[0,1]。不会把全部视频或整个数据集复制为三通道缓存。
- 保留三通道patch embedding、decoder和预训练权重；不是恢复旧版in_chans=1并对RGB卷积权重求平均。
- 模型内部mean/std不变。复制后的三个输入通道在标准化之前相同，不额外添加颜色信息。
- MAE两个视图、初始化/阶段冻结EF和分割探针、可选全量微调全部统一灰度内容。权重记录输入协议，冻结评估自动读取。
- 名称仍是EchoCardMAE/VideoMAE算法对照，但应表述为“统一灰度输入下的适配”，不是官方原RGB的逐像素复刻。输入协议变化必须使用新run_tag。

## 服务器命令

在实际项目目录激活原训练环境，拉取代码：

```bash
git pull
```

真实灰度数据冒烟（独立smoke_目录，不污染正式实验）：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 --smoke
```

默认9组的完整400轮训练和全部阶段冻结评估：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42
```

仅在确定需要全部37组消融时：

```bash
bash scripts/run_temporal_research.sh --run_tag temporal_gray_s42 --suite full
```

同一个灰度run_tag可先跑9组、之后追加full，已完成组跳过。不要加--prepare_rgb_cache，也不要填写旧RGB实验的run_tag。数据根目录不同时显式加--data_root，首次运行和续训保持一致。

如需在最终权重上额外跑基线EF/分割全量微调，追加--anchors baselines；默认只做冻结评估，不会自动追加昂贵的全量微调。

## 启动时看什么

启动行应包含：

```text
Data=/root/autodl-tmp/datasets/EchoNet-Dynamic
prepare_rgb_cache=False
input_protocol=gray_repeat3
```

第一个MAE训练batch仍是[B,T,3,112,112]，这表示内存中的灰度三通道适配，不代表加载了已删除的RGB文件夹。来源应为旧目录下的npy。

result/input_contract.json记录input_protocol=gray_repeat3、channels_equal=true、实际source_path与video_shape。每个checkpoint配置也保存该协议；评价协议应同样为gray_repeat3。

## 输出与恢复

输出根目录：/root/autodl-tmp/outputs_temporal/temporal_gray_s42/。

- result/：各实验日志、图、评价、汇总及analysis.zip，下载此目录不带权重。
- ckpt/：各实验权重。last默认每10轮覆盖；每50轮阶段快照评价完成后删除，最终epoch_0400.pt保留。
- 原命令即可续跑尚未完成的实验。last包含完整epoch续训状态；硬关机可能重算最多约10轮，不是逐batch恢复。
- 希望保留中间阶段用于以后重新评估，提前加--keep_stage_checkpoints；希望完成后仍保留优化器续训状态，加--keep_completed_resume。

详细保留策略见 [保存与续训说明](temporal_storage_and_resume.md)。
