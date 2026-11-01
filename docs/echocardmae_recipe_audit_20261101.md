# EchoCardMAE 原仓库与当前 Clip-Recurrent 的完整训练配方审计

日期：2026-11-01。性质：源码、实际运行协议与已有结果审计，不是新增训练实验，也不是本轮最终候选排名。

## 1. 先回答核心问题

**不能把当前性能差距主要归因于 EchoCardMAE 的对比学习和 ROI，也不能据此判定 Recurrent 主线不行。当前两套系统除了这两个模块之外，还存在足以影响结果的重大差异。**

最需要优先处理的是：

1. 分割的空间 token 网格从官方 `14x14` 变成了当前 `7x7`。
2. 分割头不是同款：官方 `192维/4层`，当前实际是 `64维/2层`；官方还迁移预训练 decoder，而当前另建随机头。
3. EF 的目标函数不是同一类损失：官方在百分数上做 L1，当前在 EF/100 上做默认 SmoothL1，在正常误差范围内等价于缩放的平方误差。
4. 下游优化配方不同：EF 初始 LR 相差 10 倍，下游 weight decay 相差 500 倍，还同时改变了调度、早停、batch 和增强。
5. 当前 EF 一个固定窗口，官方另有 mean5 五窗口推理；当前分割病例/标注帧集合及 Dice 聚合方法也与官方不同。
6. 自监督阶段的采样跨度、可见 token 数、掩码的时间结构、像素目标、学习率曲线和更新次数均不同。

**所以，更准确的描述是“当前独立的流式研究配方 vs 官方完整配方”，不是“EchoCardMAE 去掉两项创新后加一个 RVM”的受控比较。**

这也意味着，之前用“都是 ViT-S、都做全量微调、都是1600轮”来暗示两者已基本可比是不够严谨的。尤其当前头部的轻量化和损失替换，必须明确披露，不能当成没有影响的工程实现细节。

## 2. 证据范围与没有获得的信息

### 2.1 已核实

- 作者仓库：`https://github.com/m1dsolo/EchoCardMAE`。
- 本次核查时 upstream `main` 提交：`37bf429d0134461e835e8665f221b05a8ea8e457`。
- 本地只读参考：`资料/EchoCardMAE-main/EchoCardMAE-main/`。
- 逐个联网核对 `pretrain.py`、`echonet/train_ef.py`、`echonet/train_seg.py`、`echonet/val_ef.py`、`model/utils.py`、`model/vit_seg.py`、`model/echocardmae_vit.py`、`utils.py`，统一换行后与 upstream 完全一致。
- 当前运行：`cr_forward_fast_v3`，不是早期单帧移植、400轮旧探索或用户另一个 EchoCardMAE 改版。
- 当前实际协议来自服务器 `result/cr_forward_fast_v3/<stage>/protocol.json`，不是只读 YAML 默认值猜测。
- 服务器当前文件的原始 SHA256 与锁定协议一致；本地源码经 CRLF→LF 后也全部匹配这次下载的协议。
- 已下载小型证据快照至 `outputs/echocardmae_recipe_audit_20261101/remote/`，没有下载权重。
- CPU 检查结果：`outputs/echocardmae_recipe_audit_20261101/audit_evidence.json`；复算脚本：同目录 `audit_local.py`。

### 2.2 尚未核实

用户“只关闭中值滤波、完整训练后接近论文效果”的那次真实配置、训练日志、初始化权重 SHA256、最终 VAL/TEST 指标，待用户补充。

因此本文的“官方”指**作者公开源码的默认行为**，不冒充已经审计了用户那次真实复现，也不把其他本地修改版 EF 头误认为作者默认实现。

没有读取 TEST 标签做新评估；没有修改、终止或重启服务器队列；没有自动排新实验。

## 3. 总览表

| 环节 | 作者公开实现 | 当前实际运行 | 是否可能显著影响结果 |
|---|---|---|---|
| 编码器规模 | ViT-S，384维，12层，6头 | 同规模局部编码器，另加全局状态与融合层 | 相近的骨干规模，不代表同一函数 |
| 原生输入 | 112x112，3通道 | 112x112，灰度重复3通道 | 通道数相同；归一化不同 |
| patch | 8x8 | 16x16 | 高，尤其分割 |
| 局部 token 数 | `8*14*14=1568` | `8*7*7=392` | 高 |
| 局部时域 | 16个采样帧，stride4，首末相差60源帧 | 16个连续帧，首末相差15源帧 | 高，尤其 EF |
| 跨片段 | 两个随机视图用于 SSL 对齐，非递归流式状态 | 近期64连续帧，外加0~128真实历史帧，按16帧递归 | 核心架构/任务定义不同 |
| SSL 数据 | 官方 TRAIN | TRAIN+VAL+TEST 无标签图像 | 协议不同，后者是 transductive |
| SSL mask | ROI 内75%，各时间 tubelet 独立采样 | 全图 tube mask90%，实际91.84%，整窗空间位置共享 | 高 |
| 每局部片段可见 token | 312 | 32 | 高，但不能说“信息量就是9.75倍” |
| 重建目标 | 数据集统计归一化像素；默认中值滤波目标 | ImageNet归一化后，每个完整tubelet patch整体标准化 | 高；关闭滤波仍不相同 |
| 对齐损失 | `0.8*MSE + 0.2*InfoNCE` | 仅 masked MSE | 属算法差异，贡献未知 |
| SSL AdamW | LR1e-4，WD0.05，betas(.9,.95) | 相同峰值LR、WD、betas | 优化器种类不是主要区别 |
| SSL 调度 | 公开代码固定LR，无warmup/clip | warmup40轮+cosine，clip norm1 | 中至高 |
| SSL 有效batch | 32个视频，每个两视图 | 64个视频窗口，micro32、累积2次 | 更新预算不同 |
| SSL 更新次数 | 按7465病例估算约372800次/1600轮 | 实测251200次/1600轮 | 相同epoch不等于相同更新量 |
| 分割输入 | 单帧2D patch embed | 视频tubelet2，再把每个tubelet特征重复给两帧 | 高，目标帧身份分辨能力不同 |
| 分割头 | 192维/4层/3头，迁移SSL decoder主体 | 64维/2层/4头，全新随机头 | 高 |
| 分割损失 | 前景+背景 soft Dice | CE+前景soft Dice | 高优先排查 |
| EF头 | 全部token均值→fc_norm→单线性层 | 先逐帧空间池化，再64维注意力聚合+MLP | 不保证更复杂就更优 |
| EF损失 | `L1(100*z, EF)` | `SmoothL1(z, EF/100, beta=1)` | 高优先排查 |
| 下游WD | 1e-4 | 0.05 | 数值相差500倍，不能直接解释成效果差500倍 |
| EF LR | 固定1e-5 | 峰值1e-4，warmup5+cosine | 高优先排查 |
| 分割LR | 固定1e-4 | 峰值1e-4，warmup5+cosine | 实际曲线不同 |
| 早停 | patience25，min25，max1000 | patience10，max100 | 正式复现与开发筛选不同 |
| EF推理 | 训练期随机单窗口；独立`val_ef.py`是mean5 | VAL每病例一个固定64帧窗口 | 不能直接对照mean5分数 |
| 分割指标 | 全局前景Dice | 主指标病例平均；另存global | 指标数值不可混用 |

## 4. 分割：当前差距首先不是“仅仅MAE没学好”

### 4.1 patch16与patch8不是只差速度

112像素输入下：

- 官方每帧空间网格 `14x14=196`。
- 当前每帧空间网格 `7x7=49`。

当前单个空间token要承担更大区域的边界描述，进入分割头的空间位置数只有官方四分之一。patch像素重排输出仍能产生112x112预测，但**输出尺寸相同不代表中间稠密表征相同**。

这不能证明patch16必然差，更不能量化它损失几个Dice点；但它是优先级很高的差异，不是可以忽略的trick。若未来改patch8，会改变计算成本和权重适配，需要新协议，不能悄悄改进当前轨迹。

### 4.2 当前分割头确实再次变轻了

真实调用位置：`research_pipeline/echo.py:19` 的默认 `hidden=64, depth=2, heads=4`，构造 `ViTPatchSegDecoder` 时显式覆盖了这个类原有的192维/4层默认值。

因此不能只看 `models/downstream.py` 中“EchoCardMAE-style”注释，就说它等同于官方头。

CPU参数实点数：

| 内容 | 官方 | 当前 |
|---|---:|---:|
| encoder之外的分割读出参数 | 1,915,136 | 157,952 |
| 相对大小 | 约12.1倍 | 1倍 |
| 位置编码 | 新建可学习参数 | 固定buffer |
| decoder层数/维度 | 4/192 | 2/64 |

官方数包括 `encoder_to_decoder`、decoder和decoder位置参数；当前数是完整task head，不含两侧主干，计数口径已对齐。

### 4.3 更重要：官方迁移decoder，不只是迁移encoder

官方 `echonet/train_seg.py` 的加载循环仅跳过3D patch卷积和 `decoder.head`：

- encoder transformer加载。
- `encoder_to_decoder`加载。
- 4层decoder transformer和decoder norm加载。
- 2D patch embed、分割类别输出层和新位置参数重新初始化。

用作者类的state_dict结构核对，有 **55个非encoder张量、1,852,800个头部参数**可从预训练直接迁移。此数包含作者声明但前向未用到的q/v bias，不应等同于全部有效计算参数。

当前虽然 `FinalTemporalMAE` 内部保留SSL decoder，但 `TemporalTask` 另外创建分割头，**不会把SSL decoder拷进task head**，而64维与192维也不能直接严格迁移。

所以官方下游利用的是“已预训练的编码器+大部分解码器”，当前是“已预训练的递归编码器+较小随机分割头”。这是明确的预训练收益传递路径差异。

### 4.4 tubelet2重复展开不等于真正的逐帧特征

当前clean baseline设定 `frame_readout=repeat`：两个相邻源帧压成同一个tubelet，再把同一空间特征重复两次。

CPU检查：同一输入窗内，tubelet内偶/奇帧特征的最大差为0。后续同一分割头对它们的预测也相同。

这不等于“完全没有目标帧信息”：tubelet仍编码两帧内容，也可以恢复共同解剖结构。但不能声称每一帧都有独立可辨别的稠密特征。官方单帧2D输入则不存在这个特定约束。

不同滑窗/位置会改变tubelet对齐和上下文，预测可以不同，不能把这个检查推广成“同一个病例所有相邻帧永远同样输出”。这也是先前可学习展开要解决的实际接口问题；本轮clean B0/A没有使用该创新。

### 4.5 损失、增强、checkpoint选择也没对齐

- 官方：前景+背景Dice，选择VAL DiceLoss最低的checkpoint。
- 当前：CE+前景Dice，选择病例平均硬Dice最高的checkpoint。
- 官方：Normalize后pad到136，再随机crop112，主要是平移增强。
- 当前：TGC、gamma/contrast、亮度、zoom、blur、speckle，几何同步但没有等价的pad/random-crop分布。

增强更多、听起来更“超声专用”，并不保证更有效。两种损失和两种checkpoint选择规则也可能选出不同训练时期，不能只归结为是否过拟合。

## 5. EF：损失、初始化和推理方式都是实质差异

### 5.1 当前SmoothL1几乎是在做MSE，不是官方MAE目标

令 `z` 是模型输出的小数EF，`y` 是百分数EF：

```text
官方：L_official = |100*z - y|
当前：L_current = SmoothL1(z - y/100, beta=1)
```

只要误差小于100个百分点，当前就在SmoothL1的二次区间：

```text
L_current = 0.5 * (z - y/100)^2
          = 0.5 * (100*z - y)^2 / 10000
```

因此差异不只是loss显示得小：平方误差更重视大误差样本；L1对不同误差幅度的梯度权重不同。二者对应的最优统计量也不一样，前者偏条件均值、后者偏条件中位数，而最终报告的主指标是MAE。

5个百分点误差的CPU自动求导例子：当前loss为0.00125、对`z`的导数约0.05；官方loss为5、导数为100。

**这不是“官方参数更新强2000倍”**：AdamW会自适应重标定，梯度裁剪、epsilon、历史二阶矩及相对样本梯度都会影响更新。正确判断是“目标和优化动力学都不同”，不能直接把梯度数值比当学习速度比。

### 5.2 官方有针对EF的保守初始化

作者默认：

- `init_scale=0`，线性EF头权重初始为0。
- bias设为0.556，初始输出55.6%。
- 全局token均值之后做`fc_norm`，再线性预测。
- LR固定1e-5。

零权重头使第一个数据梯度步骤不会通过头传入encoder；这不是长时间冻结，但会减少第一步随机头带来的主干扰动。

当前 `TemporalEFReadout` 是随机初始化投影、注意力score和两层MLP，没有55.6%先验bias，峰值LR1e-4。这种组合可能更早改变主干，也可能更快拟合训练集。

这解释的是“为什么有可能训练曲线差很多”，不是已经证明它导致当前泛化损失。原始EF加载时`encoder.norm`与下游`fc_norm`命名也不同，不能假设其最后归一化参数完整迁移；当前则原样保留native norm。

### 5.3 看起来都处理约64源帧，实际注意力范围不同

- 官方：16帧按stride4采样，索引`0,4,...,60`，这16个观测在**同一次ViT里共同注意**。
- 当前：64连续帧切4个16帧局部片段，跨片段只能通过状态/融合，再交给近期缓存读出。

在50FPS视频里，官方一个局部输入首末跨度1.20秒；当前单个局部片段是0.30秒，近期64帧首末跨度1.26秒。当前总观测可以更长，但“长范围共同注意”和“先压缩再融合”并不等价。

这属于主线算法的真实设计取舍，不是应当为了对齐而删除的特性；但它会使“相同16帧”成为误导性的比较描述。

### 5.4 官方mean5不可与当前单窗直接相比

作者 `train_ef.py` 在VAL随机取一次窗口；另一个 `val_ef.py` 在整个视频均匀选择5个起点，每窗独立预测后平均。

当前VAL每个合格病例固定取一个64连续帧窗口，源码支持按病例聚合多个预测，但本轮实际1279病例就是1279窗口，**没有实际mean5**。

多窗一般有机会降低起点不确定性、覆盖更多心动周期，但收益必须实测。应先确认用户“达到论文效果”的分数是不是mean5或TEST，不能直接把当前VAL单窗4.227与其作表征优劣比较。

## 6. 优化器：不是换了神奇优化器，而是训练配方不同

### 6.1 SSL优化器的核心参数其实相同

两边都是AdamW，峰值LR1e-4、weight decay0.05、betas(0.9,0.95)。没有发现作者默认用了Muon、额外EMA、layer-wise LR decay、Mixup或CutMix这些“隐藏秘诀”。

作者类中有`get_parameter_groups`，但实际训练脚本直接传`model.parameters()`，**不能把没有调用的分组函数当成作者使用的特殊weight-decay策略**。当前也直接传可训练参数。

真正不同的是：

- 官方公开训练脚本没有scheduler、warmup和梯度裁剪。
- 当前40轮warmup后cosine降至0，梯度范数裁剪1。
- 官方batch32，每病例两视图；当前有效batch64，且每个样本是多片段窗。
- 官方只用TRAIN；当前10030个无标签视频。

按7465个官方TRAIN病例且cache均存在估算：`floor(7465/32)*1600=372800`次update。当前实际 `ceil(10030/64)*1600=251200`次update，约为前者67.4%。

此外当前LR在训练后段逐步很小，官方则保持1e-4。不能只用“都1600轮”判断适应程度相同。但当前每窗口包含更多局部片段，不能反过来声称总帧曝光或总计算一定更少。

### 6.2 下游配方差异更大

| 项目 | 官方EF | 当前EF FT | 官方分割 | 当前分割FT |
|---|---|---|---|---|
| 优化器 | AdamW | AdamW | AdamW | AdamW |
| LR | 固定1e-5 | 峰值1e-4 | 固定1e-4 | 峰值1e-4 |
| WD | 1e-4 | 0.05 | 1e-4 | 0.05 |
| betas | .9/.999 | .9/.999 | .9/.999 | .9/.999 |
| batch | 16 | 有效32/micro8 | 64 | 有效32/micro16 |
| 调度 | 无 | warmup5+cosine100 | 无 | warmup5+cosine100 |
| 裁剪 | 未设置 | norm1 | 未设置 | norm1 |
| 精度 | FP16 autocast+GradScaler | BF16 | FP16 autocast+GradScaler | BF16 |
| 早停 | patience25/min25/max1000 | patience10/max100 | patience25/min25/max1000 | patience10/max100 |
| 选择 | VAL MAE | VAL MAE | VAL soft DiceLoss | VAL病例平均硬Dice |

WD相差500倍是配置事实，不意味着有效正则强度或掉点也相差500倍；AdamW的逐步衰减还受LR曲线和update次数影响。

当前EF实际第16轮最好、第26轮停止；cosine此时仍在高LR区。因此有必要检查保守EF微调配方，而不应直接用“已经跑到早停，所以优化没问题”排除训练侧因素。

作者的YangDL实现用`counter > patience`，当前用`>= patience`；所以同名patience也不是完全相同的停止时刻。这通常比头部、loss和LR差异小，但完整复现需记录。

## 7. 自监督目标：关闭中值滤波仍剩下多项大变化

### 7.1 输入标准化不同

- 官方mean=(.1257,.1271,.1292)，std=(.1951,.1957,.1974)。
- 当前mean=(.485,.456,.406)，std=(.229,.224,.225)。

同样灰度复制成3通道，两套变换送入卷积和注意力的数值分布不同。后续有LayerNorm也不能保证抹平输入偏移和卷积响应差异。

但这不等于ImageNet统计必然有问题：它可能更接近初始VideoMAE来源分布；官方则更贴合心超域。需要受控适配证据，不能凭直觉单方面定性。

RGB存储与灰度重复本身不是本次重点：当前已经保留3通道输入，没发现“这次仍在用单通道卷积”的问题。原始解码是否包含微小通道差异，需用户真实缓存才能量化。

### 7.2 像素目标本身不同

官方patchify标准化图像后直接做masked MSE，没有逐patch whitening。即使删除MedianBlur，它仍重建亮度、对比度等绝对信息。

当前 `norm_pix_loss=true`：对已经ImageNet标准化的整个tubelet patch，联合其时间、空间和通道维度再减均值除标准差。当前不是“官方像素MSE只删掉中值滤波”。

因此重建loss的数值不能跨两套目标比较；当前0.358与官方另一数值不表示谁重建更好。丢弃局部均值/尺度可能使某些信息更不重要，也可能有利于鲁棒结构学习，是否影响EF/分割必须验证。

### 7.3 mask难度相差很大

按作者真实ROI表计算：14x14中ROI占158个patch。

- 官方每时间tubelet可见`int(158*0.25)=39`个，8个时间组共312个。
- 当前7x7网格每组可见`int(49*0.1)=4`个，8组共32个。
- 当前实际mask率`45/49=91.84%`，不是精确90%。

官方可见token数约为当前9.75倍；因为当前patch面积大4倍，可见像素元素数的比例约2.44倍，**不能直接把token数比例叫作信息量比例**。

官方ROI mask在8个时间组分别shuffle；当前tube mask沿时间、并跨当前窗内各clip保持相同空间可见位置。官方有不同时间观测补不同空间位置的机会；当前不能简单从历史同位置已见像素补全。这是不同的训练任务难度，不是只有少量背景像素的差别。

### 7.4 ROI不是只从loss里去掉黑背景

作者 `EchoCardMAEViT.forward` 对encoder使用 `mask | ~fg_mask`，所以背景也不进入encoder；decoder再用专门background token补回背景位置，只监督ROI内被mask位置。

因此ROI同时影响encoder输入、可见token预算、重建区域和对齐特征的池化范围。不能先假定它一定贡献很小；也不能反过来把目前差距全归因于它。贡献必须来自有效受控证据。

### 7.5 对比学习也改变训练样本组织

官方每病例独立采两个stride4视图，平均两边重建loss，再以0.2系数加入双向InfoNCE；同一batch其他视频作为负例。

当前是一个连续时间窗的多clip递归重建，无成对InfoNCE目标。即使最终证实对比loss收益不大，“双视图曝光”与“连续多片段曝光”也不是同样训练数据组织。

## 8. 评估人口、标签与统计口径

### 8.1 当前分割排除了大量早期/边界标注帧

当前要求目标帧能在近期窗最后16个位置的**每一个位置**合法出现，且不补假帧。其必要条件是：

```text
target_frame >= 63
target_frame + 16 <= video_frames
```

由服务器manifest和实际protocol核实：

| 当前分割集合 | TRAIN | VAL |
|---|---:|---:|
| 满足64真实帧的视频数 | 7427 | 1279 |
| 上述视频中原始标注帧数 | 14844 | 2558 |
| 因目标位置不完整而排除的标注帧数 | 6736 | 1080 |
| 真正保留的标注帧数 | 8108 | 1478 |
| 至少保留一个标注帧的病例数 | 4975 | 902 |

此外原始FileList的7465/1288病例里，还有38/9个不足64帧的视频已在manifest层排除。

当前单epoch训练访问8108个seg records，每个标注帧选一个轮换位置；VAL把1478帧展开到16位置，因此23648个观测。**23648不是23648个独立标注帧，也不是更多独立患者。**

官方单帧分割不要求64帧历史，通常可用更多标注帧训练。当前不但评估分布不同，训练监督量也明显减少；其影响方向不确定，不能假定筛掉的一定更难或更容易。

这个约束服务的是“完整上下文、各位置公平比较”，不是原始数据坏了。对于最终应用和与官方公平比较，应区分完整上下文诊断cohort与全标注覆盖协议；不能一直把诊断cohort默认当成官方benchmark。

### 8.2 当前两个Dice本身就相差0.625个百分点

当前1600轮B0分割FT：

- 病例平均 `dice_mean=0.9171058`。
- 全局前景 `dice_global_mean=0.9233590`。

作者YangDL SegMetric累计全体像素的混淆矩阵，再算前景Dice，不含背景Dice进主指标。它更接近后一个global字段，而不是当前主字段。

同一批预测就能因聚合方式有0.625个百分点差距；但当前还有16个位置和不同病例集合，因此直接换成92.34%仍不能说已与官方同口径。需要相同病例、相同标注帧、相同输入位置/聚合规则重新计算。

### 8.3 VAL、TEST和SSL曝光不能混为一谈

本轮指标用于VAL选择，尚未提供官方TEST成绩。用户那次复现如果报告TEST，不能用它与当前VAL直接下定量结论。

当前SSL按用户要求使用全数据集无标签图像，记录为 `transductive_all_splits_unlabeled`，不是监督标签泄漏，但与作者只用TRAIN的inductive预训练协议不同。

VAL/TEST图像曝光通常不是“我们为什么更差”的直接解释，也不能因为未用标签就声称与官方具有完全同样的数据隔离。对外benchmark必须明确披露。

### 8.4 掩膜生成未见同类错误证据

官方及当前都使用VolumeTracings两侧边界、跳过首对点、拼成多边形、坐标round、`skimage.draw.polygon`栅格化，原生112坐标不按CSV原始尺寸缩放。

本次源码未发现当前重新用了此前错误的自行填充方案。但这是源码路径核查，不是逐病例对缓存GT作了全量像素一致性测试；不能把未做的全量验证说成已经通过。

## 9. 其他实现差异及已排除的错误解释

### 9.1 作者公开Attention声明了q/v bias，但标准分支没用它

`model/utils.py` 的普通Attention创建`q_bias`/`v_bias`，实际forward只调无bias的`self.qkv(x)`；`cos_attn=false`时走该分支。

CPU小张量检查：修改q/v bias后输出最大差0，两者梯度均为None。当前实现则在F.linear里真实拼入q/v bias。

这是确实存在的函数差异，尤其初始VideoMAE可能带非零bias；它不是“作者用了更强attention”的证据，也不应未经验证把疑似上游遗漏抄回来当性能优化。

若以后需要逐数值复现，应明确区分“忠实公开forward”与“标准VideoMAE兼容forward”。

### 9.2 位置编码及LayerNorm

- 两边预训练均可用flat sinusoid；当前不是此前误用的3D分离位置版本。
- 编码器均384维/12层/6头，常规MLP ratio4，无drop path。
- 当前预训练配置已设norm eps1e-6，与官方预训练一致。
- 下游官方分割新建可训练位置参数；当前固定2D sin/cos buffer。
- 官方分割默认LayerNorm eps1e-5，EF显式1e-6；当前task head默认1e-5。这不应排在patch、头迁移、loss和数据协议前面。

### 9.3 没发现“其实没加载预训练/主干被冻结”

实际初始化报告：从 `videomae_vit_s.pth` 加载218张量、24,029,376参数；`missing_encoder_keys=[]`、`skipped={}`。78个未加载native张量涉及新加模块，不能等同于78个encoder参数没加载。

从B0 epoch1600加载下游时296张量原样加载、无新增/丢弃native键。FT主干未冻结，实际观察到EF约2724万、seg约2736万参数收到梯度；native SSL decoder等不在task前向里，不应把所有`requires_grad`参数都叫作参与下游更新。

这排除了“根本是随机encoder/误冻主干”这一类简单解释，但不证明现有所有实现都完全无误。统计不能替代真实梯度路径测试。

### 9.4 运行加速不应与科学配方混淆

microbatch、workers、线程、缓存、BF16/FP16是执行侧因素；提高**有效batch**则会改变update数和噪声，属于科学配方变化。本轮扩到64已显式记录，不是数学上完全无影响的提速。

当前active-clip/indexed-mask路径已有参考loss/gradient等价测试，其作用是合并执行、不造假历史或跳过任务。不能仅因为运行快或GPU利用率变化推断MAE学坏了；也不能用执行等价去替不同的学习率、batch、loss背书。

## 10. 已有结果对原因定位的约束

本轮同口径的冻结线性结果：

| SSL轮数 | EF MAE(pp) | 分割病例平均Dice |
|---|---:|---:|
| 400 | 5.2124 | 80.2746% |
| 800 | 5.1079 | 79.7558% |
| 1600 | 5.1224 | 78.4580% |

它说明当前配方下，重建目标继续优化并没有转成持续提升的冻结分割特征。这个信号在全量FT之前就存在，因此**不能把所有问题全归咎于FT覆盖预训练**。

反过来，它也没有证明RVM造成退化：没有在相同patch、mask、目标和评估协议下隔离该因果关系。本轮也不应为此擅自扩展无记忆/无缓存训练。

当前B0最终FT是EF MAE4.2268、分割病例平均91.7106%/global92.3359%；这些和冻结分数不是同一类结果。

用户真实官方复现结果尚缺日志，因此现在不能计算两者确切掉点、说其中几成来自某一因素，也不能保证改完某项就达到论文成绩。

## 11. 原因优先级：事实与假设分开

| 优先级 | 已知差异 | 主要影响对象 | 当前判断 | 不能声称 |
|---|---|---|---|---|
| 最高 | 不同cohort/global-vs-macro/EFmean5-vs单窗/VAL-vsTEST | 分数可比性 | 必须先对齐，便宜且必要 | 对齐后必然追平 |
| 高 | patch16、tubelet repeat | 分割空间与帧身份 | 有明确容量/接口约束 | 已证实损失几个点 |
| 高 | 64维2层随机头 vs 192维4层迁移头 | 分割 | 很强的可疑来源 | 大头一定更优 |
| 高 | EF L1 vs fraction-SmoothL1、头初始化、LR | EF | 最值得先查的下游配方 | 仅改loss就解决 |
| 高 | FT WD/增强/早停与选择 | 两任务 | 当前不能算官方复现配方 | WD越小一定越好 |
| 中高 | 可见token预算与mask时间结构 | SSL与两任务 | 当前局部任务明显不同/更稀缺 | ROI/CL贡献可忽略 |
| 中 | 像素目标、输入归一化、SSL调度/更新量 | SSL | 可能解释冻结趋势，需要后续证据 | 只有多跑轮数就好 |
| 较低起查 | BF16、qv实现、LN eps、小初始化差异 | 两任务 | 完整审计需列出 | 凭空定为主要掉点原因 |

## 12. 不重跑1600轮的最小排查顺序（建议，尚未执行）

### 第一步：只重算可比指标

拿到用户官方复现的log/config和逐病例预测，确认权重确实是“去中值滤波后自训”，而不是下游脚本默认又加载了仓库自带`EchoCardMAE.pt`。

这个路径检查很重要：作者脚本默认 `CKPT_NAME='./EchoCardMAE.pt'`，指向自训 `res/pretrain/.../best.pt` 的行是注释。**这只是默认源码风险，不是指责用户那次加载错了。**

对相同病例/标注帧计算global、frame/patient mean；EF同时标注single-window与mean5、VAL与TEST。没有权重也可先对现有逐病例预测重算指标，不能完成mean5则明确缺口。

这是可比性诊断，不是新增预训练，也不涉及反复证明RVM必要性。

### 第二步：现有B0权重的有限读出/配方校准

| 候选排查 | 与当前baseline的主要变化 | 推进的决定 | 变好/不变/变差意味着什么 |
|---|---|---|---|
| EF官方式微调配方 | 保留RVM+cache，L1百分数、保守头初始化、1e-5/1e-4 WD、官方平移增强 | 当前EF瓶颈是否主要在task适配 | 变好说明配方有效；不变再查时域/表征；不能一次包改归因到单参数 |
| 分割官方规模读出 | 保留RVM+cache和已训encoder，192维4层，明确比较是否迁移SSL decoder | 是否保留完整decoder迁移接口 | 变好支持读出瓶颈；不变提高patch/目标等优先级 |

以上是有限的开发改进，不是正式单因素论文证据；暂不自动拆成完整超参网格。可先用已有冻结features测头部容量，但冻结收益不能替代FT验证。

现有证据不能直接回答这些反事实，所以少量新head/FT可能有价值；本次没有启动它们。

### 第三步：只有仍解释不了时，才动新预训练配方

再决定是否采用patch8、更接近官方的可见token预算、域归一化/像素目标及SSL调度。保留递归+bounded cache主线，改变一组有明确目标的配方，先看早期同口径冻结轨迹，不默认再开多个1600轮。

patch8和mask预算最好共同按任务可见信息/算力预算设计，不能只改patch后继续沿用所有默认值，重复制造新的混杂。

## 13. 审计结论

**用户怀疑“更像是训练配方或实现差异，而不能直接归因于论文创新点”的方向有充分源码依据。**

但现在最强的证据不是“某个神奇trick”，而是：当前分割表征和读出被压缩、预训练decoder收益没有传到下游、EF训练目标变了、下游正则和调度不同，以及benchmark口径未对齐。这些差异必须先处理，才能把后续分数解释为MAE创新是否成功。

因此建议先用现有权重恢复可比、合理的task适配，不急于再做昂贵长训；同时保留当前冻结分割随长训下降这个真实负信号，不把它全部用“FT覆盖”解释掉。

## 14. 主要源码定位

本项目：

- `configs/pipeline/clip_recurrent_global.yaml`：基础科学配方。
- `scripts/run_clip_recurrent_fast.sh`：实际有效batch64覆盖来源。
- `research_pipeline/echo.py:19`：实际task head尺寸与读出。
- `research_pipeline/echo.py:152`：EF和seg loss。
- `research_pipeline/echo.py:281`：权重来源、数据构造与metadata。
- `research_pipeline/engine.py:196`：AdamW及紧随其后的warmup/cosine。
- `research_pipeline/engine.py:357`：选择指标与早停。
- `research_pipeline/ssl_data.py:118`：all-split SSL逐视频取窗。
- `utils/final_temporal_data.py:83`：完整目标位置约束；`WindowDataset`实现训练/评估cohort。
- `utils/final_temporal_execution.py:8`：逐clip执行与近期特征/目标帧抽取。
- `models/final_temporal_mae.py:104`：repeat逐帧展开。
- `models/downstream.py:228`：patch分割decoder定义，注意其类默认值被adapter覆盖。
- `models/ef_readout.py:8`：当前EF attention head。
- `research_pipeline/metrics.py:43`：按病例与global聚合。
- `utils/pretrained_init.py:77`：VideoMAE初始化迁移。

作者公开实现（提交已记录，链接附行号便于复核）：

- [预训练配方](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/pretrain.py#L28)
- [EF训练](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/echonet/train_ef.py#L55)
- [EF mean5推理](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/echonet/val_ef.py#L106)
- [分割训练与权重迁移](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/echonet/train_seg.py#L54)
- [官方分割encoder/decoder](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/model/vit_seg.py#L161)
- [官方ROI和时序mask](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/utils.py#L199)
- [背景token及前景encoder输入](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/model/echocardmae_vit.py#L221)
- [公开Attention前向](https://github.com/m1dsolo/EchoCardMAE/blob/37bf429d0134461e835e8665f221b05a8ea8e457/model/utils.py#L168)
- YangDL指标与早停：本地只读参考 `G:/SRTP/EchoCardMAE/EchoCardMAE/yangdl/yangdl/metric/{seg,confusion_matrix}.py`、`core/{task_module,early_stop}.py`；本次未取得该依赖精确upstream提交的联网文件，未把它冒充成已完成逐文件远端校验。
