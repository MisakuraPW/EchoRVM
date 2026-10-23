# CR-EchoMAE / EchoRVM：协作者代码地图

面向心超视频自监督表征的研究代码库。当前主线：**短片段内联合时空编码、跨片段递归状态、有限近期缓存，以及同一预训练编码器的帧级几何与视频功能读取。**

这是持续演进的研究仓库，不是已经完成的论文官方复现仓库。旧实验保留原入口与协议，新实验优先使用统一 pipeline。不同协议的结果不能直接拼成公平排名。

## 文档入口

| 文档 | 用途 |
| --- | --- |
| [脚手架与运行指南](docs/pipeline_guide.md) | 启动、冒烟、自动调优、监控、恢复、存储、模型接入 |
| [评测与证据协议](docs/pipeline_evaluation.md) | 冻结/FT、病例聚合、边界、表征分析、资源口径 |
| [GitHub来源与取舍](docs/pipeline_sources_and_tradeoffs.md) | 看过哪些实现、借鉴内容、项目修改、收益与风险 |
| [上一轮时域实验](docs/temporal_final_runbook.md) | 解释历史结果和已有机制实验，不是新项目默认启动器 |
| [协作规则](AGENTS.md) | 任务范围提交，不提交数据/权重，不擅自操作服务器 |

## 目录与职责

```text
research_pipeline/   新公共执行层
  contracts.py      新模型/数据适配器的唯一接口
  echo.py           原生FinalTemporalMAE + EchoNet真实数据适配
  synthetic.py      软件自检夹具，不产生科研结论
  config.py         配置、阶段依赖、科研/运行参数边界
  runtime.py        子进程实测micro-batch、worker、gradient checkpointing
  engine.py         公共训练/评估循环、epoch边界恢复
  cache.py          磁盘映射冻结特征，按身份共享，限制总空间
  metrics.py        病例级EF、分割、边界、配对统计、表征几何
  artifacts.py      状态、协议、锁、完成校验、result-only导出
models/             ViT/VideoMAE、EchoCardMAE适配、递归核、帧展开、读取头
utils/              数据与协议、原有训练支持、机制/历史/流式审计
trainers/           原独立MAE、Hiera、下游训练器；历史实验继续使用
configs/pipeline/   新pipeline配方，不为每个新模型另写一份sh
scripts/            历史连跑脚本 + 通用薄入口run_pipeline.sh
tools/              历史协调器、数据检查/缓存、分析与报告
tests/              CPU回归测试与文件级夹具
third_party/hiera/  已纳入本仓库的Hiera依赖，保留来源和许可证
hiera_echo/         Hiera基线适配
augment/            超声增强
echo_aug_validation/ 数据IO、增强与早期代理实验；部分IO仍被当前实现复用
optim/              原有优化器；新pipeline默认AdamW，不暗改旧协议
docs/               执行文档与协作研究记录，部分本地草稿仍被Git忽略
outputs/            生成结果，默认不进Git
ckpt/               权重，默认不进Git
```

## 当前模型链路

`models/final_temporal_mae.py` 是最近一轮时域研究的原生编码器，基于 `models/temporal_mae.py`。局部编码器、记忆更新、帧展开仍在模型层，不搬进训练器。

新适配器复用 `utils/final_temporal_data.py` 的TRAIN/VAL清单和 `utils/downstream_datasets.py` 的官方EchoNet轮廓栅格化；任务读取复用 `utils/final_temporal_execution.py`。同一合法时序编码过程输出近期序列或指定帧patch，不给分割另接一个独立图像encoder。

`trainers/train_rmae.py`、`trainers/train_finetune.py` 和 `tools/run_temporal_final.py` 等保留为历史协议入口。不能因为它们名字相近，就将不同阶段的头、采样、冻结方式和预算视为完全相同。

## 环境与软件自检

Python 3.10+。先安装云环境CUDA匹配的PyTorch，再安装 `requirements.txt`；不要在已有环境里盲目重装CUDA。新循环使用 `torch.amp`。

```bash
python -m research_pipeline --help
python -m unittest discover -s tests -p 'test_research_pipeline*.py' -v
python -m research_pipeline smoke --config configs/pipeline/synthetic.yaml --tag selftest
```

synthetic只验证软件链路，不证明真实数据、CUDA吞吐或科研效果。

## 云端新实验

既有约定：代码通常在 `/root/autodl-tmp/EchoRVM`，EchoNet灰度NPY在 `/root/autodl-tmp/datasets/EchoNet-Dynamic`，CAMUS NIfTI通常在 `/root/autodl-fs/datasets/CAMUS`。这些路径可配置，不保证每个实例都已存在。

```bash
python -m research_pipeline manifest \
  --data-root /root/autodl-tmp/datasets/EchoNet-Dynamic \
  --output /root/autodl-tmp/outputs_cr/manifests/dynamic_train_val.json
```

参考配置的预训练起点故意是占位符，防止误用权重。它要求带原生 `config.model` 的时域checkpoint，不是将任意官方pth当作同一种模型：

```bash
python -m research_pipeline plan --config configs/pipeline/echo_reference.yaml \
  --set stages.pretrain.source=ckpt/mae/your_native_temporal.pt

python -m research_pipeline smoke --config configs/pipeline/echo_reference.yaml \
  --tag first_real --set stages.pretrain.source=ckpt/mae/your_native_temporal.pt

python -m research_pipeline run --config configs/pipeline/echo_reference.yaml \
  --tag first_formal --set stages.pretrain.source=ckpt/mae/your_native_temporal.pt
```

冒烟使用独立 `_smoke` tag 和小病例/轮数预算。正式阶段开始前自动实测运行参数；MAE不早停，冻结头/微调按VAL早停。参考配方不等于已经确定的最终创新点组合，也不冒充EchoCardMAE官方复现协议。

## 输出布局

```text
outputs_cr/
  result/<run_tag>/<stage>/  配置、日志、状态、CSV/JSON、图、病例观察和完成校验
  ckpt/<run_tag>/<stage>/    last、best、final、有限快照
  cache/<run_tag>/           按身份共享的冻结特征、少量分析数组
```

```bash
python -m research_pipeline status --root /root/autodl-tmp/outputs_cr --tag first_formal
python -m research_pipeline export --root /root/autodl-tmp/outputs_cr --tag first_formal \
  --destination /root/autodl-tmp/first_formal_results.zip
```

重复完全相同的run命令会核验完成项并接续未完成项。不重复启动同一队列，不终止其他项目。科研配置或相关代码改变时拒绝假装是同一次续训。

## 协作边界

新增模型优先改模型和适配器，不复制训练/评估循环。头、时间权限、划分、初始化或预算改变必须记录。新基座当前完整接通的是EchoNet参考链路；CAMUS/TED/HMC等需要经验证的数据适配器，不能复制单帧伪造视频，也不能省略患者重叠检查。

v1为单设备基座。多卡、任意模型自动编译、临床端到端速度及完整论文证据，不会因为有pipeline就自动成立。后续扩展这些能力时仍复用公共引擎，不再建另一套散装脚本。
