# GitHub实现调研、复用与项目取舍

调研日期：2026-10-23。通过用户连接的GitHub工具实际读取源码及许可证，不限于心超领域。这里区分“参考设计”“复用仓库已有实现”和“逐行复制外部代码”，不把读过一个README写成抄完一套系统。

**本次新增pipeline没有整段搬入这五个外部项目的训练框架，也没有增加Lightning/Accelerate/MONAI运行依赖。主要借鉴设计并按本项目契约独立实现；原有模型、数据、读取头和专项审计则直接复用本仓库代码。**

## 1. 实际查看的外部文件

| 项目/文件 | 查看内容 | 文件blob SHA（不是commit SHA） |
| --- | --- | --- |
| [MAE / engine_pretrain.py](https://github.com/facebookresearch/mae/blob/main/engine_pretrain.py) | AMP、梯度累积、update边界、学习率进度、日志 | `4ea0d138e3023c556b62ed01190e529c0eb89276` |
| [timm / train.py](https://github.com/huggingface/pytorch-image-models/blob/main/train.py) | loader设置、有效batch/LR关系、worker持久化的增强传播风险、训练组织 | `6e4bdf6370c370059e29e741146ac4fb479d8198` |
| [timm / checkpoint_saver.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/checkpoint_saver.py) | 临时写入替换、last/best、有限历史、恢复保存器状态 | `258f1ab238c9623c6b033f184f98f88221bd3b14` |
| [Accelerate / utils/memory.py](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py) | OOM识别、候选batch缩小、清理内存、非OOM错误透传 | `e64a4c592dc5dca3c21f25bc181d30c869fc9dc0` |
| [DINOv2 / eval/utils.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/utils.py) | eval模式、无梯度特征提取、特征身份/样本枚举、模型与评估分离 | `c50576b1940587ee64b7a422e2e96b475d60fd39` |
| [MONAI / surface_dice.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/surface_dice.py) | spacing、逐样本/逐类结果、空值聚合与边界定义差异说明 | `5f9a5f57f2530af71e7625ea541a63d14163036b` |
| [MONAI / hausdorff_distance.py](https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/hausdorff_distance.py) | 空mask、percentile、物理spacing、方向聚合的明确语义 | `200740101ffc496bb90e388e948208ecc9c7605e` |

以上URL指向当时默认分支，之后可能变化；blob SHA用于核对当时文件内容，不伪装成可定位整个版本的commit。正式复现实验还应保存环境锁及本仓库commit。

## 2. 借鉴内容、修改、收益与风险

### 2.1 MAE：有效更新与日志

落点：`research_pipeline/engine.py`、`runtime.UpdateBatchSampler`。

保留AMP、按有效batch累积、按更新推进学习率、标量日志的组织方式。没有照搬原循环每iteration全设备synchronize，也没有照搬简单 `loss /= accum_iter` 后只在整除边界更新的尾批处理。

项目修改：以实际有效窗口样本数加权；尾部样本全部使用；只有成功optimizer更新推进scheduler。MAE固定epoch，VAL只诊断；下游才早停。原生mask按病例记录与epoch确定，避免micro-batch试探改变验证掩码条件。

正向作用：执行预算可核对、不会丢尾批、减少不必要同步，避免用重建VAL最优替代研究者要求的固定轮模型。

风险：即使样本顺序/有效batch相同，随机dropout、浮点归约及跨样本操作也可能不等价。新适配器必须声明microbatch安全，不能把条件写死为所有模型安全。当前AMP跳步选择明确中止，较保守，但不会把数值失败悄悄吞掉。

### 2.2 timm：成熟训练器不是把所有开关照单全收

落点：`runtime.py`、`engine.py`、`artifacts.py`。

借鉴训练/验证职责分离、持久loader、清晰有效batch、恢复与有限权重历史。没有将timm整个分类训练脚本套进心超，也没有自动按batch改变LR。

项目修改：跨帧/不同历史样本使用列表batch和合法时间分组；epoch/sample决定采样，持久worker通过共享epoch更新；DataLoader从创建前确定spawn，避免在CUDA初始化后fork。恢复点含早停、optimizer、RNG和已提交曲线；best使用不可变epoch文件+last引用，先提交再删旧best。

正向作用：不重新写不同任务训练循环；避免乱序恢复、重复曲线、损坏last、权重无限增长。源权重与结果分离，不复制一份source进下载包。

风险：spawn有启动/序列化成本；持久worker占CPU内存、可能持有文件句柄。当前通过memmap按需读取、worker独立打开、缓存总量限制缓解，但不等于无成本。v1只有epoch边界恢复，可能重放最多一个保存间隔。固定保留的科研快照不会自动删，仍需估算总磁盘。

### 2.3 Accelerate：只对资源问题降级，不掩盖代码问题

落点：`runtime.select_runtime/probe`、`cache.prepare_cache`。

借鉴OOM与非OOM错误分离、释放试验对象再尝试候选batch的思路。没有照搬“找到第一个可执行batch即可”的目标，也没有把所有cuDNN/RuntimeError都解释为显存不足。

项目修改：独立进程压力测试，候选按真实完整有效更新吞吐选择；压力测试覆盖最大合法上下文；固定有效batch和LR，不让自动调优改变科研题目。默认留显存余量；近似速度优先省资源。冻结特征提取的OOM只缩小提取chunk，不调整训练目标。

正向作用：减少上云手调、试验污染和训练中突然OOM；shape/数据/语法错误立即暴露。试验结果、失败和时间全部留档。

风险：搜索本身花时间、结果受GPU竞争/IO热缓存/短探测噪声影响；先batch后worker不是全网格全局最优。自由显存还会被其他项目改变，因此没有OOM绝对保证，也不会为了“跑满”抢占或杀掉其他任务。候选进程超时仅标记该候选，不反复无限重试。

### 2.4 DINOv2：把表征提取与头拟合拆开

落点：`cache.py`、`contracts.Experiment.features`、`engine.prepare_features`。

借鉴冻结特征提取、样本顺序和评估模块分离。没有照搬分布式all-gather和全量特征常驻GPU；没有将面向图像CLS的提取直接当作视频目标帧几何。

项目修改：按原始病例/帧/出口/历史协议提取；FP32特征和目标以memmap落盘，线性与轻量头共享同一编码器身份的缓存；变更历史、权重或数据时缓存失效。无随机增强的冻结训练才允许缓存。

正向作用：冻结分割不再每epoch重跑全视频编码器；不用将数十GB特征装进显存。头实现与编码器创新解耦，能稳定比较不同读取能力。

风险：固定特征的probe不等于随机增强FT；FP32提取本身有一次成本；特征可能很大，磁盘IO会成为新瓶颈。缓存预算不足明确失败，不暗中把评估换成另一种协议。快速复用用size/mtime检查，SHA256在创建时记录，不能防止故意同时伪造内容与时间戳。

### 2.5 MONAI：先说明测量定义，后讨论高低

落点：`metrics.py`、`pipeline_evaluation.md`。

借鉴显式spacing、逐样本/逐类记录、空mask与聚合范围必须透明的原则。没有复制MONAI实现，也没有默认加入整个MONAI依赖；使用SciPy二维边缘距离，明确给出本项目的合并双向ASSD/HD95定义，不称其等价于MONAI全部选项。

项目修改：位置→原帧→患者等权；global Dice另列；EF窗口先聚合预测；边界失败记数，物理坐标未验证就只报pixel。公共观察行可以离线复算、配对bootstrap和分层，不需要再次训练。

正向作用：防止批大小、重复目标位置、空预测丢弃、pixel/mm混用制造假优势；保留可追溯原始观察。

风险：边界定义与别的论文可能不同，必须统一后再比；valid-only边界均值不能代表全病例；低成本表征几何和反事实不是医学解释的充分证明。bootstrap只反映病例采样，不替代预训练多seed。

## 3. 本仓库直接复用的部分

| 已有模块 | 本次复用 | 为什么不另写 |
| --- | --- | --- |
| `utils/final_temporal_data.py` | manifest、合法历史窗口、按epoch采样、目标位置交集 | 已处理数据尺寸异常、短视频、目标帧对应和split封存 |
| `utils/downstream_datasets.py` | 官方EchoNet轮廓栅格化 | 不重新发明mask生成或几何重建标准 |
| `utils/final_temporal_execution.py` | 同一次合法时序编码的目标帧/近期序列读取 | 不让EF/分割悄悄换主干或时间权限 |
| `models/final_temporal_mae.py` | 原生模型及严格加载 | 保留已探索的模型结构与checkpoint关系 |
| `models/ef_readout.py`、`models/downstream.py` | lite EF头、ViT分割头 | 公共reference固定头，而非为了新结果临时换头 |
| `utils/final_temporal_training.py` | 按record/epoch构造确定性mask的辅助函数 | 不另造一套不同掩码顺序 |
| `utils/checkpoint.py`、`utils/seed.py` | 原子torch写入、空间预检、RNG恢复 | 复用此前已修复的CPU ByteTensor/CUDA RNG处理 |
| `utils/final_temporal_streaming.py`、`final_temporal_mechanisms.py` | 有界专项审计 | 新入口只调度，不把专项研究逻辑复制进训练循环 |

没有改动这些历史文件的默认实验行为。新的训练器只承接新run，旧队列/结果不被热迁移。

## 4. 许可证与依赖边界

实际查看的MAE LICENSE为CC BY-NC 4.0；timm、Accelerate、DINOv2、MONAI所查看LICENSE为Apache-2.0。**不因代码能下载就假定任意商业使用许可**。本次不vendor这些外部代码，不更改整个仓库的许可证，不为既有第三方权重或代码做额外授权承诺。

若以后逐行移入外部函数，应保留原版权/许可证、必要NOTICE，并在这里追加明确文件及修改记录。当前借鉴通用设计与数学定义不等于可以删除模型和权重已有的许可要求。

## 5. 验收时的诚实边界

CPU回归覆盖：配置与DAG、尾批、micro-batch梯度、冻结缓存等价性、冻结主干不变、故障后恢复、早停、完成校验、spawn读缓存、结果ZIP排除权重；文件级夹具覆盖原生MAE→checkpoint→EF/分割读取与预览。

CUDA探测、真实数据吞吐、共享服务器资源竞争、外域适配和完整论文对照仍需相应验收。不会以“参考优秀框架”保证没有bug，也不把脚手架测试当成模型优越性证据。
