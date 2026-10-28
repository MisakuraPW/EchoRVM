# Clip-Recurrent B0 + D + A/W 执行说明

对应研究计划：`Clip-Recurrent_前向探索与实现计划_2026-10-28.md`。本文件说明实际实现、运行边界与证据限制，不代表任何候选已取得收益。

## 默认一起跑什么

**一个完整B0 + 一个到400轮的A；W不默认跑。** 不训练无递归、无缓存、单帧或纯VideoMAE，不自动训练A+W，不做层数/宽度网格。

| 顺序 | 内容 | 默认新增预算 |
| --- | --- | --- |
| 1 | 原干净B0及既定完整评测 | 原1600轮与原评测配方不改 |
| 2 | B0@400 的D3：H/F各做EF、分割线性读取 | 4个小面板探针，TRAIN/VAL各最多64病例；公共100轮上限、patience10 |
| 3 | B0@400 的D1/D2/D4 | TRAIN最多64病例，时间信用梯度仅前2病例；无MAE参数更新 |
| 4 | B0@400 的既有可见性/未来因果/状态数值审计 | 复用已有mechanisms实现，固定32窗口，非新训练 |
| 5 | A：第6层后读取历史 | 从B0@0显式复制共有权重，按1600轮日程训练，到400预算暂停 |
| 6 | A@400 的EF/分割线性读取 | 与B0@400相同全规模TRAIN/VAL、同头/预算/种子；不是拿64病例探针对全规模成绩 |
| 7 | A@400 路径/恢复诊断和流式审计 | 固定小面板；流式2病例、最多512帧 |
| 8 | 匹配对比表 | 病例级配对bootstrap，无自动晋级 |

B0原队列含原本的400/800/1600冻结读取，默认直接复用其400结果，不额外再训一份全规模B0读取。D3小面板H/F头独立训练，只解释可读性；不能与全规模主成绩混表。

为不打断B0，默认先完成原B0全部队列，再读取其400快照做诊断和候选比较。快照仍是400，不用B0最终1600冒充400。候选的800/N阶段已声明但默认不执行。队列结束时A是 `paused_at_budget`，不是1600轮完成；整个研究run也不会伪造全阶段DONE。

## A、W与共有初始化

- A：复用同一个 `feature_fusion`，从末层norm后移到第6层后、第7层前；不保留第二个末层融合。首段无旧状态不读历史。writer读取最终F，decoder旧状态路径不变。
- W：B0读取位置不变，仅替换均值摘要为宽度64的历史条件非线性注意力。score零初始化，有效可见token初始等权；LayerNorm、tanh和状态条件投影都显式存在。仅一个全局状态，递归门控不变。
- W只在 `--with-w` 时进入执行列表。A/W分别由B0@0分叉，不互相初始化、不自动组合。
- 原VideoMAE权重只初始化B0。候选通过原生epoch0快照逐tensor复制共有参数并验证，不把“同一个seed”当共有初始化证明。新增writer模块保持单独初始化，载入不消耗公共模块/头初始化的外部RNG。
- 最终F是统一主出口。B0的 `local` 是12层norm后的无历史H；A的 `local` 已在第6层读过历史，明确标为post-read末层出口，绝不称作B0 H。

## D诊断具体测什么

### D1：时间信用

固定真实连续窗口、固定mask，**只用最后clip的重建loss，target detach**。记录每段输入、中间state、每次writer摘要以及共享writer参数的梯度连接/范数/有限性。末次写入在最后loss之后，其无梯度是正常边界，不是实现失败。共享参数有梯度不等于每次早期使用都有梯度。

分别运行正常、feature历史输入清零、decoder历史输入清零、两者清零。清零作用于cross-attention的context输入边，保留模块对当前token的计算，不把整个融合模块粗暴替换为identity。这是全序列重新展开的推理干预，不能将数值差简单相加作路径归因，也不能据此证明被阻断模块无用。

### D2：重建与保留表征

同窗口记录最后clip及全窗口masked重建loss；另外在**无mask完整输入**下记录近期F序列差异、同一个已冻结头的EF预测差/分割变化像素比例。decoder干预不应影响无mask特征：测试专门检查这一点。

固定头敏感性不等于另训头的可读性，后者单独由D3承担。诊断头必须绑定完全相同的checkpoint，且冻结头中的backbone必须与源权重逐tensor相同，否则报错。

### D3：H/F读取

TRAIN/VAL各至多64病例，相同目标位置、相同容量和日程，各训练EF/分割线性头。冻结FP32特征缓存按checkpoint、出口语义、病例清单和协议区分，不与其他出口误用。分割仍按源帧再按病例聚合。

额外的目标帧替换记录tubelet内两帧是否共享输出、替换一帧后两帧输出分别如何变化。repeat基线共享特征是已知结构限制，不会把“有目标帧索引”写成“逐帧独立编码已证实”。ED/ES/S辅助分类不是本轮主比较，未扩展为额外任务。

### D4：历史干扰与恢复

固定近期像素，比较真实prefix、重复首prefix clip；区分“仅在最后clip前清零当前state、保留此前缓存”和“近期边界reset，并重新计算整段近期缓存”。

单个黑clip后接干净clip，逐步记录state/FIFO表征差及EF固定头偏移。记录扰动片段是否已逐出缓存；视频不够长则明确没有逐出后的证据，不填假历史。人口文件保留真实长度、FPS、前缀、固定源帧、强度方差与相邻帧变化代理，以便按覆盖范围分组分析；这些代理不是临床质量真值。人工黑片段不称为真实探头切换。

主要原始文件：`temporal_credit.json`、`paths.csv`、`history_recovery.csv`、`population.json`。原mechanisms审计另保留其数值、梯度、hidden isolation与future causality证据；不再实现第二套同名检查。

## 固定日程与暂停恢复

公共配置新增：

```yaml
training:
  schedule_epochs: 1600
  run_until_epoch: 400
```

cosine总步数始终来自N，K1时学习率不会因为暂停而降到零。到预算点强制保存完整 `last.pt`（model/optimizer/scheduler/scaler/RNG/采样epoch/步数），另存模型快照、可引用的哈希清单与PAUSED状态。调整 `run_until_epoch` 是执行预算事件，不改变科研身份；改变N、数据、模型、初始化、学习率等仍被拒绝。

预算点快照永久pin，不会被rolling清理误删。模型快照可供K1探针读取，但不能代替last做原轨迹续训。暂停阶段不能解析`final/best`；只允许实际已发布且通过hash校验的 `epoch_NNNN`。未知预算或缺失依赖直接报错。

此外保留完整 `resume_epoch_0400.pt` / `resume_epoch_0800.pt`，防止之后last前进而失去旧暂停点的优化器/RNG恢复信息。同文件系统优先硬链接旧last；后续last原子替换不会覆盖旧恢复点，不支持硬链接时检查空间后保存副本。

默认首个400结束后**不会自动延长**。之后同tag显式选800可继续；选择N是长程确认，需另行决定。只调整预算不改变公共配置的总N和已有阶段定义。下游早停不变，SSL仍无loss early stopping。

## 云端指令

全新合并run，不复用已经锁定的旧run_tag：

```bash
cd /root/autodl-tmp/EchoRVM
conda activate echocardmae
git pull --ff-only

# 只看预算/执行列表；可将完整配置写出来供检查
RUN_TAG=cr_forward_v1 bash scripts/run_clip_recurrent_forward.sh plan \
  --plan-output /root/autodl-tmp/cr_forward_v1_plan.json

# 灰度数据清单 -> 真实模型小样本冒烟 -> B0完整队列 -> D -> A@400与匹配评测
RUN_TAG=cr_forward_v1 bash scripts/run_clip_recurrent_forward.sh all
```

可分开用 `smoke`、`run`。仅在明确愿意承担第二个400轮候选时，在两条指令都附加 `--with-w`。它不是默认执行项。

若B0已完成、或已在独立旧代码checkout留下所需0/K1快照，用独立新tag读取现有结果，**不重训B0**：

```bash
python -m research_pipeline.forward_suite \
  --config /root/autodl-tmp/outputs_cr_clean/result/你的B0标签/plan.json \
  --baseline-result /root/autodl-tmp/outputs_cr_clean/result/你的B0标签 \
  --tag cr_forward_from_existing --plan-only
# 检查后去掉 --plan-only 执行；建议先增加 --smoke 做真实小样本验证
```

已有B0日程从该plan读取，不能用示例1000覆盖1600。旧B0正在跑时不要原地pull换代码；等任务边界或完成后再部署，本次实现未操作服务器。旧run的软件hash锁不会被绕过；读取其不可变快照与用新代码直接续训旧run是两回事。

后续显式延长或最终确认，保持与首次相同的config、root、tag、baseline-result及候选开关：

```bash
RUN_TAG=cr_forward_v1 bash scripts/run_clip_recurrent_forward.sh run --endpoint 800
# 真正决定入选后才运行N；--confirm-final额外做最终lite/FT，不在首轮默认展开
RUN_TAG=cr_forward_v1 bash scripts/run_clip_recurrent_forward.sh run --endpoint 1600 --confirm-final
```

原基线模板当前N=1600。脚本不悄悄重写N或所有引用来生成1000轮配方；其他已锁定N的轨迹需提供一致的实际plan及保存节点。

## 结果、速度与解释边界

复用原 `result/<tag>`、`ckpt/<tag>`、`cache/<tag>` 三分目录。监控与结果无权重导出仍用公共CLI：

```bash
python -m research_pipeline status --root /root/autodl-tmp/outputs_cr_clean --tag cr_forward_v1
python -m research_pipeline export --root /root/autodl-tmp/outputs_cr_clean --tag cr_forward_v1 \
  --destination /root/autodl-tmp/cr_forward_v1_results.zip
```

新增 `forward_comparison.csv/json`：匹配checkpoint epoch、相同头/训练预算/病例窗口，记录两任务分开变化、病例级bootstrap、协议/权重指纹、帧数、更新数、训练wall-hours与头训练耗时。`hf_readability.json`单独保留D3可读性结果。缺点不拿别的epoch顶替；区间跨零不自动淘汰，loss下降不自动入选。

每个候选及头训练复用micro-batch/workers/checkpointing实测调优，不改有效batch、clip长度、mask、病例或LR。诊断是固定小面板的FP32前后向，不承诺占满GPU；流式资源审计记录的是有限病例的backbone/state/FIFO，不将其冒充整个临床系统含任意任务头的部署性能。模型有限状态不意味着训练BPTT常量显存。

本地测试包含真实文件形状的数据夹具、两候选原生前后向、探针缓存与读取、D诊断、配对表，以及固定环境连续训练与暂停后续训的参数/优化器/调度器一致性。CPU检查不证明4090吞吐；云端smoke只证明小样本可运行，不是正式ETA。正式计时应来自各模型真实病例规模的预热后epoch，不能套用“一分钟一epoch”。

数据暴露仍是全数据无标签transductive SSL；下游TRAIN训练、VAL开发，TEST标签不参与此次选择。诊断辅助定位机会，不证明递归必需、状态具备医学语义或某候选对未见外部数据有效。
