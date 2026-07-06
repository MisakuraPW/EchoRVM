# Recurrent Echo MAE

本项目用于开发面向超声心动图的 Recurrent MAE：以 EchoCardMAE 为基础，优先支持 EchoNet-Dynamic 与 CAMUS，后续在 EchoRisk 权限可用后接入多中心风险预测任务。

当前阶段包含 AutoDL 起步脚本、超声特化数据增强、以及用于筛选增强策略的小模型验证流程。

## 快速开始

在 AutoDL 新实例中，建议代码放在 `/root/autodl-tmp/MAE`，数据放在 `/root/autodl-fs/datasets`：

```bash
cd /root/autodl-tmp/MAE
bash scripts/setup_autodl.sh
python tools/check_env.py --strict
python tools/check_dataset.py --dataset echonet --root /root/autodl-fs/datasets/EchoNet-Dynamic
python tools/check_dataset.py --dataset camus --root /root/autodl-fs/datasets/CAMUS
```

运行数据增强验证 debug：

```bash
bash scripts/run_aug_validation_debug.sh
```

运行增强策略筛选：

```bash
AUGS="A0_no_aug A4_tgc_zoom_speckle A6_per_frame_random A7_clip_consistent" \
SEEDS="0" \
bash scripts/run_aug_screening.sh
```

正式长任务建议在 tmux 中运行，避免 SSH 断开导致任务中断。

## 目录说明

```text
configs/       训练与增强验证配置
scripts/       AutoDL 初始化、debug、正式训练脚本
tools/         环境检查与数据集检查工具
augment/       超声特化数据增强
echo_aug_validation/  数据增强验证代理实验
```

## 私有文档

`docs/` 和 `资料/` 是本地研究资料与实验说明，默认不上传 GitHub。

## 数据路径约定

AutoDL 默认使用：

```text
/root/autodl-fs/datasets/EchoNet-Dynamic
/root/autodl-fs/datasets/CAMUS
/root/autodl-fs/datasets/EchoRisk
/root/autodl-tmp/outputs
/root/autodl-tmp/logs
```

所有训练脚本应从 YAML 配置读取路径，不在代码中写死本地 Windows 路径。

## 离线数据增强

EchoNet-Dynamic：

```bash
bash scripts/run_augment_echonet.sh \
  /root/autodl-fs/datasets/EchoNet-Dynamic \
  /root/autodl-fs/augmented/EchoNet-Dynamic
```

CAMUS：

```bash
bash scripts/run_augment_camus.sh \
  /root/autodl-fs/datasets/CAMUS \
  /root/autodl-fs/augmented/CAMUS
```

如训练时发现 `/root/autodl-fs` 读取慢，可同步增强缓存到高速盘：

```bash
bash scripts/sync_augmented_to_tmp.sh /root/autodl-fs/augmented /root/autodl-tmp/augmented
```
