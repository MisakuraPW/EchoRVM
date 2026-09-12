# MAE 运行参数自动调优

## 功能边界

这是启动时的有限候选实测，不是按 GPU 利用率闭环实时改训练，也不保证 GPU 永远 100%。目标是提高真实样本吞吐，同时保留显存余量。当前接入 train_rmae.py 和 temporal research 的预训练阶段；冻结审计、Ridge、分割探针、全量微调暂不自动调参。截图不能单独确定当前阶段。

自动搜索：micro batch、相应的梯度累积、gradient checkpointing 开关、DataLoader num_workers。prefetch_factor、AMP、线程数等沿用用户配置。不自动修改帧数、分辨率、模型宽深、mask、增强、学习率、优化器或有效批量。

## 云端运行

代码同步到云端之后，在项目根目录运行：

    python tools/run_temporal_research.py \\
      --run_tag temporal_gray_20260909 \\
      --input_protocol gray_repeat3 \\
      --only clip_mae_pool64 hier_global hier_spatial hier_dual \\
      --autotune

单独的 MAE trainer 也支持：

    python trainers/train_rmae.py --config <原配置.yaml> --output_dir <原结果目录> --autotune

不要与现有队列同时启动。新参数不会热更新正在运行的 Python 进程。优先在 epoch checkpoint 保存后停止旧队列，再用相同命令和 run_tag 加此开关重启。保留原 checkpoint；按既有逻辑从可用断点恢复，断点之后未保存的训练可能重跑。

已完成的模型不会为了调优重新训练。未完成的每个模型首次进入预训练时单独调优；调优结果存在各自目录中。后续即使省略 --autotune，trainer 发现该目录的 runtime.json 仍会恢复同一选择，避免无意退回旧微批量。

## 实测方法与保护

- 每个候选启动独立子进程，读取真实训练数据并执行前向、反向和 optimizer.step。试跑模型和优化器完全丢弃，不保存训练 checkpoint。
- 完成两次成功的优化器更新用于预热，再计时三个更新窗口。CUDA 同步后计算 samples/s，包含取数、传输、前向、反向和更新。若 AMP 一直跳步，候选不能胜出。
- 先试原配置，再尝试更大的有效批量整除因子；原批量失败时尝试更小值。随后搜索重计算开关与 worker 数。这是分阶段有限搜索，不是穷举所有组合。
- 预留至少 15% 总显存且至少 2 GiB，并计入其他进程占用。OOM/显存余量不足的候选不采用；其他代码或数据错误立即报错，不当作 OOM 吞掉。
- 单候选上限 240 秒，超时终止子进程树并报错。复杂模型可能需要单独配置更长试跑策略，当前不会无限等待。
- 吞吐差距在 3% 以内优先选择显存占用更小的方案。几次短测有噪声，不能保证长时间训练速度或显存峰值；调优日志保留所有候选供核查。
- 不与其他 GPU 任务同时测。其负载改变、文件缓存、功耗限制都可能影响结果。

## 研究可比性

有效 batch 固定，例如原 8×4=32，可以选择 16×2 或 32×1，不能变成 64×1。对比学习的负样本依赖微批量，EchoCardMAE 双视图等模型锁定微批量，只搜索其他运行项。微批量扩大目前只允许已知的 temporal_v1 数据流和无对比目标的 temporal_mae/echo_videomae。

改变微批量时，额外 sampler 保留原 drop_last 决定的每轮样本预算及排列；不会因新的批量大小多丢尾部样本。最后一个不足有效批量的窗口按实际样本数加权。优化器更新窗口数量不变。

这不保证逐位复现：随机 mask 的调用分组、GPU 浮点归约顺序可能变化；可变有效帧的 batch 内重建归一化与梯度累积也不意味着所有分组完全数值等价。需要逐位续训或严格固定微批量消融时，不应中途启用。请在结果中记录 autotune 选项与开始使用的 checkpoint。

调优配置保存在实际 config.yaml 和 checkpoint 的配置中；研究队列的 requested_config.yaml 与协议哈希保留原请求，独立 runtime.json 记录运行差异。不会把性能调优伪装成模型算法变化。

## 日志、阶段与恢复

结果目录：

    /root/autodl-tmp/outputs_temporal/<run_tag>/result/
      current_stage.json
      <实验名>/autotune/runtime.json
      <实验名>/autotune/trial_XX.json
      <实验名>/autotune/trial_XX_result.json
      <实验名>/autotune/trial_XX.log
      <实验名>/config.yaml
      <实验名>/logs/train.log

查看阶段：

    cat /root/autodl-tmp/outputs_temporal/temporal_gray_20260909/result/current_stage.json

查看选择（将实验名换成当前模型）：

    cat /root/autodl-tmp/outputs_temporal/temporal_gray_20260909/result/hier_global/autotune/runtime.json

current_stage 的 pretrain 表示该模型预训练子进程（包含开始时的调优）；audit 表示冻结表征审计。文件记录最后启动/结束状态，不是心跳，机器突然关机后 running 可能是旧状态，还需核对进程是否存在。旧版运行进程不会自动生成这个新文件。

runtime.json 绑定请求配置、GPU 型号/总显存、PyTorch/CUDA 版本。相同配置重新启动复用，不反复试跑。签名不同会拒绝静默覆盖，保留旧报告并使用新输出目录；本版本不提供强制覆盖已跑实验的选项。

没有 GPU 时跳过调优，按原配置执行。启用 torch_compile 的配置目前显式拒绝自动调优，避免把编译时间计入几步短测。审计阶段即使显存很少，也不会因此增大探针训练 batch 或改变其优化轨迹。

## 验证范围

本地为 PyTorch CPU 环境。已测试候选约束、配置隔离、OOM 候选筛选、吞吐选择、尾批样本顺序、梯度累积尾批和 CPU 回退；CUDA OOM、实际显存余量和云端加速比例需要真实 GPU 首次校准验证。不要把本地测试通过理解成已经在服务器跑过。

