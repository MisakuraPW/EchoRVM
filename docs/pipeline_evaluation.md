# 统一评测与证据协议

本文是新pipeline的测量契约，不把脚手架等同于已完成全部论文实验。旧结果保持旧口径；新结果内部按这里统一，跨版本比较必须先核对差异。

## 1. 评测层级

| 层级 | 原生适配器实现 | 能说明什么 | 不能直接说明什么 |
| --- | --- | --- | --- |
| frozen_linear / EF | 近期序列均值后Linear；Huber拟合EF/100，输出乘100 | 指定聚合下的线性可读取性 | 不是闭式ridge；均值可能丢失顺序，弱结果不证明全部时序无用 |
| frozen_linear / seg | 目标帧patch的Linear等价1×1分类，固定双线性上采样 | 目标帧空间特征是否容易读取 | 不代表复杂头上限 |
| frozen_lite / EF | 复用TemporalEFReadout，小型有序注意力聚合 | 有限非线性/时间聚合后的可读取性 | 不能把读取头学到的关系全部归给encoder |
| frozen_lite / seg | 复用ViTPatchSegDecoder，配置中固定宽度/深度 | 固定头容量下的几何恢复能力 | 不是严格线性probe |
| finetune | 相应头+合法时序encoder全量更新 | 实际适配能力 | 不是单独的预训练表征质量证明 |
| representation | 有界样本的方差、有效秩、可选反事实变化/CKA | 坍缩、描述符变化、输入敏感性辅助证据 | 不能由高秩或变化大推导医学语义好 |
| streaming / mechanisms | 复用原有有界流式、梯度/隐藏像素/未来权限审计 | 当前实现是否遵守其明确契约 | 不代替多患者、外域或临床部署验证 |

同一source checkpoint分别训练EF和分割头，才是“共同预训练表征的双读取”。分别FT得到两套权重不是一个联合训练的多任务模型。当前reference没有偷加分割图像encoder，也没有在EF时回避记忆重新联合编码完整历史。

## 2. 固定科研协议

每项结果记录source文件SHA256、native模型配置、数据manifest、病例/记录集合及排除、训练预算、seed、目标函数、头、出口、精度和相关代码hash。

必须区分 `local_frames L`、`recent_frames W`、历史prefix、tubelet q、memory slots K、采样stride/fps。当前reference是原生112像素、稠密灰度输入在内存中扩展三通道。历史可为固定合法倍数，或 `prefix: null` 在真实可用范围采样；128仅是可配置最大评估/训练范围，不是应用前提。

输入时间权限由现有WindowDataset与encode_window保证。分割VAL覆盖最后一个局部片段的合法位置共同病例集，不是每个标注只跑一次单帧；这也是它可能比早期评测更慢的原因。当前目标帧可使用同一局部clip中的后续帧，必须承认有限clip等待，不称逐帧零延迟。

主manifest沿用现有策略：CSV内容SHA256，视频文件stat/shape/dtype，且逐次读取检查stat；不是每次读视频全文件SHA256。迁移文件、mtime改变或原始数据变化需要重建manifest，并重新记录来源，不能删校验继续。

reference只打开TRAIN/VAL，不解封TEST。少量软件夹具、冒烟、单次训练、患者bootstrap、不同head seed及独立MAE seed，分别属于不同证据等级。

## 3. EF指标

观察行需有 `patient/target/prediction`，单位均为百分数点pp。对同病例多窗口先平均预测，再计算该病例误差；病例之间等权。对于多检查/多随访数据，应在适配器中定义exam与patient层级，不能混合不同EF真值，公共代码发现同patient不同target会拒绝。

保存：MAE、RMSE、bias、R²、Pearson相关、绝对误差P90、误差≤5pp/≤10pp的比例。零方差或样本不足的相关性/R²记null，不制造一个0或1。

## 4. 分割指标

原生reference目前是EchoNet前景LV cavity二分类。观察行记录真实source frame和目标位置，保存交集、预测像素、GT像素及单观察Dice。

主Dice的顺序为：**同一原始帧的不同位置/窗口先平均 → 原始帧等权 → 病例等权**。另列每类Dice和像素总体Dice，不能混为同一个数字。保存最差10%病例的平均Dice，避免只有总体均值。

`train_batch_dice`只是增强后在线训练批次的硬Dice监控，不是上述病例级VAL主指标。用它判断拟合趋势可以，但不能将二者的差距全部解释为过拟合或bug。

有phase/view/target_position元数据时，输出 `strata.json` 分层结果。未提供ED/ES标签时不擅自依据轮廓大小命名ED/ES。扩展CAMUS多类、TED全周期或RWMA时，先在适配器明确标签定义和坐标，不能默认所有GT都是LV cavity。

### 边界与空掩膜

设置 `evaluation.boundaries: true` 才计算ASSD/HD95；平时避免每轮不必要的CPU边界距离开销。当前使用二维边缘像素的距离变换，ASSD与HD95基于合并的双向距离样本；**不是MONAI所有设置下逐方向percentile再取max的同义实现**。

有经过坐标验证的轴序spacing才标mm，否则标pixel。不能拿112重采样后的距离冒充原始物理距离。若做resize/crop，外域适配器必须恢复原图测量或明确换算后的spacing。

两者都空时Dice=1并记录空标记；仅预测为空而GT非空时Dice=0，边界记失败/null。报告失败数量，不能让null被聚合悄悄排除后只剩好病例。有限边界均值明确叫 `assd_valid_only/hd95_valid_only`，不是全病例无失败的均值。

## 5. 冻结缓存

冻结模型在eval模式用FP32提取确定性特征，训练头可使用配置精度。一个epoch不重复编码同样的冻结数据。缓存key包含真实病例记录、历史条件、出口、源权重及代码；改变历史不能复用旧特征。

当前被冻结训练集每条记录采用固定平衡位置，VAL为所有合法位置；这是明确的probe协议，不是FT的随机增强/跨epoch位置变化。比较线性/轻量头时共享相同特征，不混入更强数据增强。缓存提取和训练头耗时分别记录，不能把头训练秒数称为整个评估成本。

## 6. 轻量表征与既有机制工具

`representation`不训练新的MAE，也不训练新的任务头。取固定顺序VAL记录，有界数量，池化描述符报告方差/有效秩，可选：

- repeat_pivot：重复目标/中间帧，查看对时间内容变化的敏感性。
- target_substitution：分割读取时只替换目标帧，保留其余上下文。
- repeat_history_only：有真实prefix时仅替换历史，保持近期窗口。

输出配对样本、L2变化和CKA。这些是合成反事实，不是新的真实标签或临床失败样本；敏感性越大不自动越好。某干预并非覆盖全部记录时保留逐记录结果，不冒充同总体CKA。

更细的patch几何/时间身份、错误历史/恢复、梯度和流式一致性已有 `utils/final_temporal_representation.py`、`final_temporal_history.py`、`final_temporal_mechanisms.py`、`final_temporal_streaming.py`。新入口直接暴露streaming/mechanisms；详细历史和解剖probe仍沿用明确的专项工具，不在默认run里自动展开。新模型需满足对应特征/流式接口后才接这些专项分析，不能假设任意工厂都等价于FinalTemporalMAE。

## 7. 资源、统计和可视化

训练报告实际已完成的训练/验证秒数、当前调用wall time、samples/s、有效更新数、模型总参数、声明可训练参数、实际观察到梯度的参数。完整适配器可能仍包含不参与下游的MAE decoder，总参数和文件体积不能直接当部署encoder大小。

`summary.checkpoints`给出实际文件字节数，不把AdamW恢复文件当纯模型文件。表征诊断的descriptor_forward时间包括传输及CPU取回，不包括视频解码/采集；流式audit的核心速度也不是临床端到端速度。临床延迟还包括clip采集等待和实际部署头。

训练曲线保留真实CSV，不平滑、不插值未测轮数。原生适配器固定前两个VAL记录保存分割轮廓或MAE target/visible/reconstruction图，不挑最好病例；归一化像素无法合法反解时明确跳过。MAE图固定mask、共同显示范围，不能把可见像素拷回效果称为全部像素重建精度。

```bash
# 任意适配器生成统一观察行后，都能复算，无需重训
python -m research_pipeline metrics --task ef --observations observations.json --output recomputed.json

# 只能在病例严格匹配时配对；field可用EF的absolute_error或分割dice
python -m research_pipeline compare --reference ref/patients.json --candidate ours/patients.json \
  --field dice --output paired.json --repetitions 2000 --seed 42
```

bootstrap是病例采样不确定性，不是MAE多次独立训练稳定性。不会因为区间跨零就自动否决方法，也不会因为点估计向好就自动晋升为正式结论。`comparison.md`仅汇总，不自动挑选“胜者”；跨配方公平性仍需实验设计保证。

## 8. 与新研究计划的对应

E1复用冻结双读取；E2通过配置不同模型/可见性/更新组织；E3用目标帧读取、分层/边界与专项身份probe；E4用历史配对和流式资源审计；E5用FT及经验证外域适配器；E6用同一引擎承载不同历史条件目标与必要2×2。

公共引擎不替研究者自动决定M1/M2/M3/M4架构、把所有对照跑一遍、选择新的mask/loss、打开TEST或认定创新成立。完整基座的意义是这些实验共享同一执行与测量基础，而不是省略它们的科学定义。
