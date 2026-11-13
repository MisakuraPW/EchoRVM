# Clip-Recurrent Stage A2

固定 native112、patch8、tubelet2、L16、最多 W64，保留单全局 S 和近期缓存。目标是在 A-R 基础上提高 EF，同时保住分割。本轮不做可变 clip、额外状态、组合搜索或 800/1600 轮预训练。

| 路线 | 唯一改动 | 判断 |
| --- | --- | --- |
| H：读出诊断 | 复用 A-R 400，原 TemporalEFReadout 池化前加一个 64 维、4 头、FFN256 的时序残差块。旧头与新头共享相同的 64 个帧描述符、缓存、L1、训练和验证预算。 | 冻结 VAL EF MAE 不比旧头差超过 0.05 pp，才做一次正式 EF 全量微调和锁定 TEST。只改善冻结读出不能证明最终精度提高。 |
| R-T：先建模时序再写入 | 对 8 个 tubelet 时间组分别汇聚有效空间 token，增加有位置编码的 96 维时序关系块；其零初始化残差加到按有效 token 数加权的原均值上。其余 A-R 不变。 | EF 改善且分割不明显退化才值得确认。倒序写入对照只检验顺序依赖，不等同于验证心动生理机制。 |
| R-M：重新分配遮挡 | A-R 网络不变。四 clip 窗口按 source/target/source/target 共享同一随机空间顺序，分别保留 28/10/28/10 个空间位置；总预算等于 4×19。短窗口沿用原对称遮挡并保留全部病例。 | 比较性能而非不同遮挡条件下的训练 loss；共同对称掩码的固定面板只作诊断。匹配的是可见 token 数，不是 FLOPs。 |

两条 R 路线都从同一 B8 epoch0 初始化，使用 A-R 的独立 400 轮日程、AdamW、有效 batch64、LR 1e-4、WD .05、warmup40、相同训练人口、每视频每轮一个窗口和原 masked pixel objective。自动调优只改变执行参数。SSL 采用包含 VAL/TEST 无标签影像的 transductive 协议；监督训练和选权重不接触 TEST 标签。保存 0/100/200/400 和最新可恢复训练状态。

两组都用原任务头完成 400 轮终点的冻结双任务、双任务 FT、完整分割 VAL 和锁定双任务 TEST；冻结探针不拦截 FT。分割保持已修正的完整 TRAIN 监督池和共同 VAL 选权重协议。补 A-R/R-T/R-M 的 200 轮冻结 EF 对照；这是早期可读性，不是 200 轮 FT 性能。开发门槛预定为共同 VAL 的 EF MAE 改善至少 0.10 pp，患者 Dice 退化不超过 0.10 pp。门槛不是显著性或等效性证明。没有达标也不自动重训或追加候选。R+H 仅是可选的独立下游确认，不属于自动队列。

机制评估复用 A-R 固定 VAL 面板，使用实际 FT 任务头。固定旧缓存，仅阻断末 clip 对 S 的读取；R-T 另反转倒数第二个 clip 的 writer 时间组，保持前三个缓存描述符不变。后段重建对过去状态和 writer 的梯度、隐藏像素与未来 clip 的隔离用于正确性核查。历史同一 1600 轮轨迹中，400/800/1600 的重建误差下降，但冻结 EF 和分割都变差；目前只支持“重建更好不保证下游更好”，不证明时空任务必有取舍。

执行：`python -u tools/run_patch8_stage_a2.py plan` 检查基线与方案；`smoke` 使用真实数据小样本且不打开 TEST 标签；`run` 执行正式队列。同 tag 重入时核对方案、产物和恢复身份。默认持久结果与权重在 `/root/autodl-fs/outputs_cr_stage_a2/{result,ckpt,control}`，可重建缓存单放 `/root/autodl-tmp/outputs_cr_stage_a2_cache`。已有 Stage A 产物不覆盖。

方法背景：[RVM](https://arxiv.org/html/2512.13684v2) 已使用非对称遮挡，本实验不声称发明这一策略；[EchoJEPA](https://arxiv.org/html/2602.02603v4) 支持检查训练目标与迁移任务的关系；[DISCOVR](https://arxiv.org/html/2506.11777v2) 是联合时空表征的参考。这些工作不是本项目性能或机制的证据。
