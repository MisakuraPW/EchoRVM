# 第一阶段：局部编码时间尺度

本轮决定：固定当前 spatial 记忆，在64帧总观察范围内比较局部8/16/32帧。只研究当前长短时序框架的时间分工，不增加无记忆对照，不比较 global/dual，不改变融合结构。4帧保留为结果触发的可选补充。

## 上轮诊断带来的约束

2026-09-28 二阶段提前诊断：pool 的目标位置分割 Dice 0.86561，片段平均 0.84973；spatial 对应 0.87103 与0.86295。spatial 融合前后 EF MAE 7.37649/7.47305、Dice 0.87103/0.86653，两项配对差异置信区间均跨0，不能断言融合有害或无效。两份权重的 offset head 都差于同参数量共享 head；本轮没有证据支持增加该头。旧两组的历史训练条件不完全匹配，不据此宣称记忆无效。

因此第一阶段只评估融合后普通EF与目标位置分割，不重复local/fused、state EF、偏移头、顺序任务或全量微调。保留原架构，包括decoder历史旁路与有效tubelet处理；避免长度实验同时变成结构修改实验。

## 固定协议

- 三组分别为 `spatial_l8`、`spatial_l16`、`spatial_l32`，每个训练样本总长64，连续采样stride1，tubelet2，112平方，patch8，spatial状态4x4，初始化同一VideoMAE权重。
- 每组100轮、seed42、AdamW lr1e-4、weight decay0.05、无学习率调度、无早停；有效batch32，原始microbatch8/累积4。继承现有纯时域训练协议：无在线增强、tube mask 0.75、原有输入归一化与像素目标。不是将所有预训练trick同时重新优化。
- 每个epoch的患者顺序/64帧采样规则一致。自动调速保留样本预算与有效batch；mask统计规则一致，但不同microbatch/模型执行路径不承诺逐bit相同随机轨迹。允许浮点与随机数流差异，不能把单seed当重复性结论。
- 本地长度同时改变每64帧内的状态更新次数（8/4/2）、局部注意力计算与位置编码长度。报告时间分工效果，不声称孤立识别了局部感受野的因果效应。
- 初始化后的新模块随机权重保持同一seed；预训练权重按现有严格编码器加载要求检查。全部重新开始匹配的100轮训练，不把旧400轮结果混入排序。

## 评价

每组只在100轮端点做一次：冻结主干EF岭回归（训练512/验证256病例，alpha10）与轻量分割头（训练64/验证64病例、含各自标注帧，固定200步）。不以val选择探针最优epoch。

EF读取256帧，每64帧状态重置，按完整有效tubelet聚合融合特征。分割目标统一放在64帧窗口的最后一帧index63，无目标后未来帧；读取其所在tubelet的融合特征，不使用两种offset view。保留短历史/补帧病例并记录有效性，full-context仅作为附加信息，不能偷偷排除后报主指标。index63与此前index48/49诊断不同，不横向混作同一评测。

每两档长度进行按患者配对bootstrap，EF负差更好、Dice正差更好。两任务分开，不合成任意总分；置信区间不是训练seed不确定性，验证集结果不是最终test性能。100轮只作筛选，不保证与400/1600轮排名一致。

性能统计包括训练进程耗时（含调速/启动与已记录重试）、评估耗时、参数量、轻量端点文件大小、GPU常驻输入的batch1原生64帧推理时间及峰值allocated显存。推理基准不含IO/CPU到GPU拷贝/任务头，不是实际临床端到端时延。缓冲时间单独以 `(L-1)/源视频FPS` 计算；缓存未提供可信FPS时不编造毫秒或心动周期覆盖。当前接口仍是64帧窗重置，不宣称已实现无限流式。

## 运行

```bash
conda activate echocardmae
cd /root/autodl-tmp/EchoRVM
git pull --ff-only
bash scripts/run_stage1_lengths.sh --smoke
bash scripts/run_stage1_lengths.sh
```

默认自动测速寻找较快的安全microbatch、梯度checkpointing、workers设置，固定有效batch32；不是保证GPU利用率100%或全局最优。smoke独立run_tag，只跑每组2步及极小探针，不做测速，不计入正式结果。

`--dry_run`只打印计划。可用 `--run_tag 新名称`、`--data_root 路径`、`--init_checkpoint 路径`，`--lengths 8 16 32`控制队列；`--no-autotune`关闭训练调速，`--eval_batch_size 2`强制评估batch用于OOM恢复。不要中途改epochs/seed再复用同一run_tag。

断电/中断后重复原命令，训练自动恢复last/interrupt，完成的预训练跳过；评估检查DONE与协议、使用已完成分区缓存。不同代码、初始化文件、数据清单或训练配置拒绝混用原run_tag。运行时若版本变化应先确认是否需要新run，而不是删除协议保护文件。

## 保存

```text
/root/autodl-tmp/outputs_stage1/stage1_lengths_20260928/
  result/spatial_l8/           # 配置、训练日志、曲线、endpoint评估
  result/spatial_l16/
  result/spatial_l32/
  result/current_stage.json
  result/comparison.csv
  result/paired_differences.csv
  result/analysis.zip          # 下载这个，不含权重或特征缓存
  ckpt/spatial_l8/             # 其余两组同结构
  cache/spatial_l8/            # 评估完成后清理本工具生成的四份npz
```

不保存best；初始epoch0000和最终epoch0100为轻量评估权重，last为完整续训文件，低频覆盖（每10轮及trainer要求的首末轮/中止情况），中断保存interrupt。保留最终last以便后续延长训练；不每轮新增checkpoint，不删除旧研究的任何权重。写权重要求额外保留3GiB空间。结果与ckpt分根目录。

最终判断：看8到16、16到32的收益是否饱和，或是否出现EF/分割权衡，并同时考虑计算与等待。若8不差再考虑4；若32仍明显改善，不宣称已找到最优。先不自动追加训练、seed或实验组。
