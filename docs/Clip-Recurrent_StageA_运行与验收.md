# Clip-Recurrent Stage A：固定 L16 的单因素读写对照

## 范围与对照

本轮只实现 B8 / A-R（`AR`）/ A-W（`AW`）。保持 native112、patch8、tubelet2、L16、W64/H0、一个全局状态与近期读出，不加频域、空间状态、单帧展开、不训练无记忆/无缓存组。代码不自动启动训练、组合模块或延长到1600轮。

- **B8**：读取独立400轮的原始 `plan.json`、epoch0/400哈希、训练与评估完成标记。复用已修复覆盖的分割FT、同源EF FT及EF冻结探针；只补全覆盖的冻结分割小头和有限机制面板。旧88.6265%的冻结分割不参与对照。
- **A-R**：用现有 `ForwardTemporalMAE(mid_read)`，同一个融合模块移到第6块后、经过剩下6块及最终归一化。无新参数；不把其后读出的 `local` 名称当成历史无关特征。写入、重建decoder的状态入口不改。
- **A-W**：用现有 `ForwardTemporalMAE(content_write)`，保留后融合；`ContentWriter` 在可见token上以旧S条件化的注意力汇聚，宽度64、score零初始化。起点与均值写入数值对齐，不改状态槽数或更新门。

候选从 B8 epoch0 逐张量拷贝，新增键仅允许 A-W writer，epoch0 SHA256与初始均匀权重均检查。沿用实际 B8 独立400轮方案，AdamW、LR1e-4、WD.05、有效batch64、warmup40、余弦、原tube mask与归一化像素MSE，400轮固定预算、不按loss早停。每轮续训状态及0/100/200/400快照。自动调优沿用公共引擎，只选执行分组，不搜索科研超参。SSL使用全拆分无标签影像，属于transductive协议。

## 启动

在有原始共享结果、权重及 EchoNet 数据的训练实例上执行。默认引用 `/root/autodl-fs/outputs_cr_spatial` 下 `cr_patch8_w64_400_v1` 和 `/root/autodl-fs/outputs_cr_gap` 下 `seg_gap_stage1_v1_coverage`。如需调整位置，用 `--source-root/--gap-root/--data-root`；原计划的绝对路径和哈希必须仍有效，不将搬迁副本冒充原有结果身份。先核对 GPU、磁盘与队列占用；不要在既有队列训练时更新代码。

```bash
git pull --ff-only
export PYTHON=/root/miniconda3/envs/echocardmae/bin/python
bash scripts/run_patch8_stage_a.sh plan
bash scripts/run_patch8_stage_a.sh smoke
bash scripts/run_patch8_stage_a.sh run
```

- `plan` 核验历史B8来源、同源400轮FT、分割覆盖14920真帧/8108抽样每检查及完整VAL/TEST权重绑定，打印计划，不启动训练或读新TEST标签。
- `smoke` 分别以实际模型和少量真实文件跑一轮，独立 `_smoke` tag，经过两项FT、两项冻结、机制检查，不测TEST。无卡时可加 `--cpu-smoke`，仅为软件检查。
- `run` 默认按 B8补项 → AR → AW 运行。每个候选顺序为SSL → EF FT → 分割FT → 双冻结探针 → 机制面板 → 分割完整VAL补评 → **锁定VAL最佳权重并测双任务TEST**；不看冻结成绩决定是否做FT，也不等人工选组才测TEST。可用 `--candidate AR` 或 `--candidate AW` 分开执行或接续。相同 `--root/--tag` 保持原协议，不混写历史B8训练结果。失败后先查当前stage日志；同代码、数据与运行配置下可重跑命令恢复。

主线分割FT沿用完整训练池、每检查8108抽样、原共同VAL选权重。冻结分割三组都用完整可用TRAIN标签，每标注帧一个确定窗口、无增强，仍用同一共同VAL；完整VAL补评只用于FT，不反向挑权重。EF沿用单W64窗口及同一损失/头/选择规则。两项任务各自微调，非联合模型。

## 读结果与默认 TEST

`compare` 只汇总已完成的VAL结果，不训练。核对训练/验证记录、任务头、损失、执行精度、训练配置及数据指纹；`matched_val.json` 列出患者配对差异、边界指标、参数、更新/帧数和耗时。分割另外按同一GT的较大/较小左室面积标作ED/ES代理（只有两个不等面积标注才分组）；不靠帧序猜测，排除的观察数单列。完整VAL核对同源帧、目标位置和GT面积，与共同VAL分开。

正式运行默认测本次运行的每个候选。各组在推理前分别锁定自身和B8的两个VAL最佳权重哈希。B8复用已验分割TEST、补同源400轮EF TEST；已完成且哈希匹配的评估可复用。`control/<tag>/AR_test.json` 与 `AW_test.json` 分别保存与B8同人口对比的结果。不会把1600轮EF混入，也不会根据TEST改变权重或训练参数。

`test --candidate ...` 只用于对已训练模型补测或重试推理，不重新训练；`--candidate B8` 可单独补基线。必要时显式补**一次**配对下游种子43，复用两边400轮源，不重训SSL。确认运行也默认产出自己的VAL和锁定TEST结果，保留原主结果。`AR` 只是下面的例子：

```bash
bash scripts/run_patch8_stage_a.sh compare
bash scripts/run_patch8_stage_a.sh test --candidate AR     # 仅对既有结果补测或重试
bash scripts/run_patch8_stage_a.sh confirm --candidate AR  # 可选，同种子配对确认
```

逐组锁定文件不会互相覆盖；同一组的已锁权重不能更换。确认种子不自动晋级，也不以CI跨零自动否定收益。TEST是固定配置的终点报告，不是挑配置的循环反馈。

## 机制产物与限制

同一确定性VAL病例面板（默认32个，最多128，真实W64窗口及末帧真标注），复用选中的FT头，不另训probe。逐组记录面板哈希、覆盖与运行时间；预算为该组历史/实际SSL与双FT墙钟的10%，检查周期为一个病例，当前病例算完才可停。超时只保留已做病例，不把不同已完成病例数混作配对证据。

1. `paths.csv`：最后clip和之前的状态/缓存固定，只将进入最后clip的特征融合或SSL重建decoder的S分别置零；FT报告EF误差与标注分割误差，decoder侧只影响重建，未作为FT任务出口。干预不是重新训练消融。
2. `history_recovery.csv`：首clip人工置黑、其后真实clip；记录每步EF前缀预测、状态/描述符变化、末帧真实分割误差。W64只有四个clip，扰动仍可能留在近期缓存，**不宣称测到了缓存淘汰后的恢复或真实探头漂移**。
3. `temporal_credit.json`：默认一个病例只用末clip重建loss，检查前段状态和写入摘要的连接与有限梯度；遮挡像素与未来clip反事实不应改写不该受影响的预测/状态。不以梯度范数排名，不声称clip内逐帧因果，也不训S-only EF头。

输出根目录默认 `/root/autodl-fs/outputs_cr_stage_a`，分 `result/<tag>_{B8,AR,AW}`、`ckpt/`、`cache/` 和 `control/<tag>`；完整VAL/TEST单独结果tag。缓存允许引擎回收本run已完成可重建特征，不清权重。`worker.log`、`status.json`、`metrics.csv` 可监控；首个实测epoch含训练/验证耗时后再更新ETA，不能按固定一晚估算。已有粗估每候选约8小时SSL＋约10小时FT，另有冻结与诊断开销。
