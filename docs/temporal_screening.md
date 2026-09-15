# 时域 MAE：快速筛选与按需补测

## 结论与依据

不必每 50 epoch 重复全部探针。新实验默认筛选节点为 **0 / 100 / 200 / 400**；每个节点只做冻结编码器的正常顺序 EF 读出。0 是迁移初始化基线，不是随机网络。快速入口不改变 MAE 的训练损失、学习率或训练轮数，也不会自动根据探针早停。

旧全量入口保留原行为；新增 scripts/run_temporal_screen.sh 使用快速评估。明确指定 --audit_profile quick 时同样生效。

依据上一轮 temporal_gray_20260909 的 400 轮结果：

| 模型 | 正常读出 EF MAE，越低越好 | 状态读出 EF MAE，仅诊断 |
|---|---:|---:|
| clip_mae_pool64 | 7.2782 | 7.2875 |
| hier_global | 7.0366 | 7.6775 |
| hier_spatial | 7.4960 | 6.8716 |
| hier_dual | 7.3287 | 6.5657 |

正常 EF 在早期有更明显差异，例如 dual 的 100 轮为 6.3779，对照为 6.7612；但优势未持续到 400 轮。400 轮三种记忆模型相对对照的正常 EF 配对差值区间均跨零，尚不能宣布谁稳定更好。状态读出有信息，却与正常读出排名不同。因此不能选每个模型的较优读出进行排名，也不能仅因某个诊断指标差异大就当成表征质量标准。

数值来源：outputs/temporal_gray_20260909_final_analysis 下的 final_scores.csv、trajectory.csv 和 paired_gains_10000.csv。当前经验只支持阶段性筛选，尚未证明这套指标能预测最终全量微调排名。

## 固定筛选口径

1. 主指标：正常顺序视频特征的冻结 EF Ridge 回归，MAE、RMSE、相关系数及逐病例预测。
2. 诊断：同一次编码获得的状态特征 EF，不额外遍历编码器；无状态模型会缺省。保留便宜的有效视频长度回归控制。
3. 固定 TRAIN 512、VAL 256 病例，seed=42，256 帧连续上下文，沿用既有采样与灰度输入协议。各原生模型窗口之间仍重置状态。
4. Ridge 沿用原评估的标准化与 alpha=10，闭式求解，不是训练若干 epoch 的 EF 全量微调。
5. 不运行 Dice、反序、打乱、记忆干预、时序距离、相位探针和推理计时。它们按需要再执行，而不是放在每个节点。
6. 重建 loss 可以保留作数值稳定性监控，但不作为最终排名依据。

新脚本按既有队列方式，先完成一个模型的训练，再评估其保存节点；不是训练到 100 轮自动决定是否继续。想先只做小预算筛选可显式 --epochs 100，并使用独立 run_tag；这也会改变学习率调度长度，不能当成 400 轮轨迹的前 100 轮直接混比。

## 如何判定要不要补测

- 比较同节点、同患者、同读出的正常 EF。自动输出配对病例 bootstrap 差值区间，差值为候选减对照，负数有利于候选。
- 区间跨零时标记“未分出差别”，不要去多个指标里挑一个最有利的数。优先扩大冻结 EF 的验证样本，或补独立随机种子；对照与候选必须一起补。
- 如果创新明确针对空间定位，即使 EF 接近，也应在最终节点补一次相同协议的 Dice；EF 不能替代所有空间能力评价。
- 如果状态 EF 改善但主 EF 没改善，记录为“状态读出有信息”，可研究特征融合，不能宣称整体表征更好。
- 仅对有可信收益或明确机制疑问的候选补全量微调、反序/重置等昂贵评估。统计区间未经多重比较校正，多节点探索产生的结论需要独立确认。

## 新实验一键入口

在云端项目根目录：

```bash
RUN_TAG=temporal_screen_v1 bash scripts/run_temporal_screen.sh --autotune
```

默认依次跑 clip_mae_pool64、hier_global、hier_spatial、hier_dual。训练超参数沿用现有 runner，自动调参仍只优化执行参数。预演不启动训练：

```bash
bash scripts/run_temporal_screen.sh --dry_run
```

快速模式强制保留筛选节点的模型权重以及最后的续训权重，不再评完一个节点就删除。result 和 ckpt 仍分开；不会额外每 50 轮保存一份。last 的保存周期沿用 --save_last_every，默认值以 --help 为准。

```text
/root/autodl-tmp/outputs_temporal/<run_tag>/
  ckpt/<model>/epoch_0000.pt, epoch_0100.pt, epoch_0200.pt, epoch_0400.pt, last.pt
  result/<model>/quick_audit/epoch_XXXX/
    metrics.json, ef_predictions.csv, screen_protocol.json, features.npz, DONE
  result/quick_comparison.csv
  result/quick_comparison.md
  result/quick_paired_comparisons.csv
  result/analysis.zip
```

阶段文件为模型快照，不含完整优化器；last 用于断点续训。实际体积依架构而定，保留 4 个节点比原先 9 个少，但不保证磁盘永不满。下载 analysis.zip 即可分析，权重和特征 NPZ 不进入压缩包。仅有一组结果时配对比较文件尚不会生成。

## 已完成的实验：不要重训，单独补测

旧 run 的训练协议与保存节点不同，不要用新训练入口覆盖它。原有正常 EF 已存在时可直接分析旧文件，不必为了新表格重复跑一遍。旧中间权重如果已被自动删除，不能凭日志补测，只能用仍存在的快照。

下面以现存 400 轮 hier_dual 为例。替换模型名可测对照组。两个命令不启动 MAE 训练。

```bash
ROOT=/root/autodl-tmp/outputs_temporal/temporal_gray_20260909
MODEL=hier_dual
python tools/evaluate_temporal_screen.py \
  --checkpoint "$ROOT/ckpt/$MODEL/epoch_0400.pt" \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic \
  --output_dir "$ROOT/result/$MODEL/quick_audit/epoch_0400" \
  --batch_size 4 --num_workers 8
```

只补 Dice，不顺带重跑 EF：

```bash
python tools/evaluate_temporal_screen.py --profile seg \
  --checkpoint "$ROOT/ckpt/$MODEL/epoch_0400.pt" \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic \
  --output_dir "$ROOT/result/$MODEL/seg_history_v1/epoch_0400" \
  --batch_size 4 --num_workers 8
```

Dice 补测默认 T=64、标注帧位置=48（从0计数），使 64 帧记忆模型的目标不落在原生窗口刚重置的第一段。保留 64/64 患者和200步冻结分割头口径，不做额外相位/状态分割训练。前文不足用有效性标记处理，不循环复制。并非每个病例都有完整真实历史；局部16帧编码还可以看同段后续帧，不是严格因果评估。

旧 Dice 使用 T=256、目标128，恰好位于64帧模型重置点；它能测局部特征，但不能据此判断在线历史记忆收益。新 Dice 必须连同对照一起重测，不能把新旧数字当同一指标拼接。

## 复用缓存与扩展 EF

缓存绑定权重 SHA256、FileList SHA256、患者选择和评估协议；不匹配就报错，不会静默复用。底层 NPY 改动未逐文件散列，此时必须换输出和缓存目录。不同模型不共享缓存。

在相同特征上补 64/256/512 标注预算，只增加 Ridge 求解：

```bash
python tools/evaluate_temporal_screen.py \
  --checkpoint "$ROOT/ckpt/$MODEL/epoch_0400.pt" \
  --data_root /root/autodl-tmp/datasets/EchoNet-Dynamic \
  --output_dir "$ROOT/result/$MODEL/ef_budgets/epoch_0400" \
  --feature_cache "$ROOT/result/$MODEL/quick_audit/epoch_0400/features.npz" \
  --ef_budgets 64 256 512 --batch_size 4 --num_workers 8
```

扩大验证集时例如 --ef_val_cases 1288，需新输出目录且不要引用旧缓存；患者数由当前 FileList 决定。新增样本需要重新编码，不能靠旧256病例的缓存得到全验证集结果。完成后汇总：

```bash
python tools/run_temporal_research.py --run_tag temporal_gray_20260909 --summarize_only
```

## 速度和验证边界

减少的是编码遍历、探针种类和评估节点；不承诺固定加速倍数。状态读出复用正常编码结果，不额外读视频。缓存命中仍要加载模型和校验快照，但无需再次提取特征。

CPU 测试覆盖目标帧与标注对齐、记忆重置边界、EF不依赖分割标注、真实NPY到冻结EF及缓存复用、独立Dice入口和新旧runner预演。实际4090吞吐需云端测量；不会因为新增脚本就自动停止或启动服务器任务。
