# Clip-Recurrent patch8 / W64 开发基线

## 决策与范围

问题：当前patch16在112图像上只有7x7空间网格。增加空间细节、将更多预算用于当前局部结构，能否改善冻结与下游任务表现？旧权重不能在不重新训练的情况下回答该配方的效果，因此实现一条新的400轮轨迹，而不是再次默认1600轮。

本版是**空间粒度、窗口长度和学习率日程的组合改进**，不是单因素patch消融，也不是官方EchoCardMAE复现。改善支持采用这套开发配方；无变化说明空间/预算调整没有解决现有瓶颈；退化则不应继续盲目延长至1600轮。单次运行不证明某个模块的独立因果收益。

## 固定设置

| 内容 | 本版 |
|---|---|
| 输入/patch/网格 | 原生112x112，8x8像素patch，14x14空间token |
| 局部编码器 | VideoMAE-S，384维、12层、6头，tubelet2，连续16帧 |
| 一个训练样本 | 每视频每轮随机取一个64帧连续窗口，内部4个clip递归 |
| 短视频 | 16~63帧取真实完整clip倍数，不补假帧；不足16帧记录排除 |
| 历史/缓存 | 额外prefix=0；保留窗口内递归全局向量及近期4个clip输出 |
| 初始权重 | 原始`videomae_vit_s.pth`，不是已训patch16 B0热启动 |
| SSL | 全split无标签，masked MSE、norm_pix_loss、ImageNet标准化、tube mask0.9 |
| 可见预算 | 每时间组保留19/196，16帧clip共152个可见token；实际mask率约90.31% |
| 优化 | AdamW，峰值LR1e-4、WD.05、betas(.9,.95)，warmup40+cosine400 |
| 批量/执行 | 有效batch64固定；实测microbatch/累积、workers、线程、checkpointing；BF16 |
| 预算 | 400固定epoch，不早停、不自动延长；每epoch覆盖last，保留0/100/200/400快照 |
| 不加入 | A中层读记忆、对比loss、ROI、滤波、频域、空间memory、学习式展开 |

空间卷积核由VideoMAE的16x16插值为8x8，复用全部兼容encoder/decoder主体；像素预测层输出尺寸改变，重新初始化并记录skipped键。不能把“允许重建head不匹配”和“encoder缺失”混淆，后者仍报错。

这不是“每视频只编码16帧”：那将没有跨clip状态训练。64帧窗口仍有4次递归步骤。状态在独立样本间重置，不跨患者传递；未证明64帧训练可无损外推到任意长历史。

当前400轮的cosine会走完日程；旧B0的400轮快照仅是1600轮日程中间点。不能把差异全归因于patch。灰度数据继续直接读取，不复制新的RGB磁盘缓存。

## 7个阶段

1. 新MAE预训练400轮，途中每25轮做固定64病例的已见人口重建诊断，不以它选择权重。
2. 初始epoch0表征诊断。
3. epoch400表征诊断。
4. epoch400冻结EF线性读取。
5. epoch400冻结分割线性读取。
6. epoch400全量分割FT：192维4层，迁移SSL decoder，Dice/WD1e-4/固定LR1e-4。
7. epoch400全量EF FT：近期attention读出，末层零权重+.556 bias，百分数L1/WD1e-4/固定LR1e-5。

监督训练上限100轮，patience10。沿用新校准头只是明确而完整的任务接口，**不宣称它已在上一轮胜出**。冻结线性任务仍用原统一损失和定义；不把不同目标的loss直接比较。

所有下游均H=0、W64。与旧B0不同的实际上下文必须披露，不能当作完全相同输入的性能对照。有效患者/原始标注保留规则未放宽；目标位置仍按16个位置聚合，数据/GT不造假，TEST标签不开放。全split SSL属于transductive，不能声称测试影像未见。

不额外安排每50轮反复训练任务头、无记忆/无缓存训练或新A候选。

## 云端运行

只在旧队列结束后更新代码。推荐在原有tmux/screen中执行：

```bash
cd /root/autodl-tmp/EchoRVM
conda activate echocardmae
git pull --ff-only
RUN_TAG=cr_patch8_w64_400_v1 bash scripts/run_clip_recurrent_patch8.sh all
```

`all`顺序是：准备独立清单→完整真实架构的小样本7阶段冒烟→正式7阶段。冒烟不偷换成小模型，但只跑少量样本、一轮。正式每阶段先自动测试安全吞吐配置，发生OOM只处理自身探测进程，不影响其他项目；不改变LR/有效batch去追逐利用率数字。

```bash
# 只冒烟；成功后run不会重复冒烟
bash scripts/run_clip_recurrent_patch8.sh smoke
bash scripts/run_clip_recurrent_patch8.sh run
# 相同tag再次run会校验已完成结果、从last恢复；不是重新训练
bash scripts/run_clip_recurrent_patch8.sh status
# 仅先跑预训练（完成后可再run接后续）
bash scripts/run_clip_recurrent_patch8.sh pretrain
```

环境变量支持`PYTHON`、`DATA_ROOT`、`INIT_CHECKPOINT`、`OUTPUT_ROOT`、`RUN_TAG`。可追加公共pipeline的`--set dotted.key=value`显式覆盖，不用改YAML；改变科学参数须新tag，不用同名续训混入旧轨迹。

## 存储与进度

默认视频仍来自`/root/autodl-tmp/datasets/EchoNet-Dynamic`，模型初始权重仍在本仓库`ckpt/mae/`。

14x14冻结分割缓存预计比7x7约大4倍。2026-11-02检查tmp仅约13GB可用、fs约110GB可用，因此默认新输出根改为 `/root/autodl-fs/outputs_cr_spatial`：

```text
manifests/dynamic_w64_h0_{ssl,tasks}.json
result/<RUN_TAG>/<stage>/   # 日志、心跳、曲线、预测、统计、配置
ckpt/<RUN_TAG>/<stage>/     # last、固定快照、best、final
cache/<RUN_TAG>/           # 28GiB上限，完成的旧特征缓存可回收
```

结果与权重分离。不同存储卷可能影响缓存/保存速度，不能提前保证与小网格相同epoch耗时。缓存不够会明确报错，不默默减病例或退回无缓存慢跑。覆盖输出位置时应预留缓存、权重和原子保存临时空间。

```bash
python tools/watch_pipeline_progress.py --root /root/autodl-fs/outputs_cr_spatial --tag cr_patch8_w64_400_v1
bash scripts/run_clip_recurrent_patch8.sh export
```

## 验证边界

新增CPU测试验证：400轮预算及smoke引用、不加载标签的全split单窗取样/短视频无补帧、真实patch16→8卷积迁移、14x14原生前向与递归梯度。真实云端GPU冒烟和执行参数必须由入口实测；CPU通过不代表已完成正式预训练或速度已知。

2026-11-02：新增3项patch8测试与4项读出回归通过。独立云端快照`EchoRVM_patch8_fe6017a`的7阶段真实4090冒烟全部完成，根目录DONE已确认；结果位于`result/cr_patch8_w64_400_v1_smoke`。这是小数据一轮软件验证，不是400轮性能结果；正式训练及完整吞吐自动探测尚未启动。启动脚本规范化初始权重为绝对路径，避免同一权重用相对/绝对路径写入同tag而产生协议冲突。
