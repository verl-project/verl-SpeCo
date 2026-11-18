# Ascend 训练状态与显存生命周期

## 目前的根本问题

同卡交替执行要求保留 engine、Adam 动量和 scheduler，同时把大量训练显存交还给
rollout。直接把参数 `.to("cpu")` 再移回来可能破坏 vLLM 的参数别名与地址；单纯
保留 engine 又不能释放其显存。CaMem 提供地址不变的暂存机制，但其常用场景是
权重/KV 初始化，不能直接假定训练过程中所有流、所有 Tensor 都归它管理。

原诊断代码把“优化器属性中的每个 Tensor 都必须在池内”当成完整性条件，把真正的
训练状态与派生统计量混在一起。这是本次检查失败的原因。此前的 C 回调引用修复
仍有必要，但它不足以证明整个生命周期正确。此前用 CPU 通过作为继续完整模型
重跑的主要依据，验证层级不够，应先验证真实 NPU 分配器与小模型生命周期。

## 最新失败的事实

目录：`/models/z00829645/speco_online_rounds_20261103_160030`。

- 第一轮 optimizer step=1、暂存、revision 1 的 16-worker 更新及新特征采集通过。
- `train_round_02.json` 记录 optimizer step=2、loss=1.1315062046、
  optimizer_probe_delta=0.000244140625，revision 2 的 runtime shards 已写入。
- 失败发生在第二轮保存后的暂存检查：DP0 的 `_global_grad_norm` 位于池外。
- 第二轮训练完成不等于两轮闭环通过；revision 2 的推理验收尚未执行。

服务器 DeepSpeed `runtime/zero/stage_1_and_2.py` 每步从
`scaled_global_grad_norm / prev_scale` 重新产生 `_global_grad_norm`；engine 保留
它用于报告。日志证实它可能位于池外；具体分配流尚无 profiler 证据，不能把流来源
当成已经证明的事实。小模型诊断主动制造其他流上的同类分配来验证覆盖规则。

## 明确的状态契约

| 状态 | 保留与检查规则 |
| --- | --- |
| 模型/teacher 参数和 buffer、FP32 master、Adam 状态 | NPU 数据必须在托管池内；暂存/唤醒检查地址、形状、dtype、有限值采样及对象身份 |
| CPU optimizer step、scheduler | 保留原对象和进度，不因轮次重建 |
| ZeRO 的结构性 meta 描述 | 仅已核对的布局字段允许 meta，检查 schema，不伪造数据 |
| root optimizer `_global_grad_norm` | 可留在池外；仅单元素浮点、无 autograd、有限、底层 storage ≤64 字节；池外统计量总预算4 KiB，仍检查原地址和值 |
| 其他池外 Tensor | 收集完整清单并拒绝暂存；小型 Adam step/moment 也不能豁免 |
| HCCL/运行时资源 | 暂存不释放通信器；沿用角色端口隔离与暂存后实际 NPU 剩余容量检查 |

池外统计量保持原值、原设备和地址，不移动、不丢弃，不修改梯度范数计算或裁剪算法。
`allocation_inventory` 写入每轮结果和失败报告，列出所有未知状态；不再只报告第一个。
失败汇总保留原 rank 报告，避免在多轮 all-gather 中递归复制完整异常列表。

## 分层验证，减少完整模型重跑

1. CPU：状态分类正反例、存储别名、大 storage 的标量视图、NaN、未知 scalar、
   Adam step/moment、回调 GC、地址和值变化；同时回归原生 PyTorch 路径。
2. 小模型 NPU：下面的诊断使用真实 CaMem、DeepSpeed ZeRO-1、AdamW、scheduler
   和 TP8/DP2 进程组，连续4轮更新/暂存/唤醒；池外制造跨流梯度范数，并在暂存时
   使用默认分配器。无需 target checkpoint、features、vLLM 服务或 RL。
3. 小模型通过后，再用新的 ONLINE_ROOT 跑既有完整 DeepSeek 两轮闭环。
4. 最后验证完整 RL：复用原有准入、激活/清理、发布和 target 版本接口。

```bash
cd /models/z00829645/verl-SpeCo
set -o pipefail
export ALLOC_ROOT=/models/z00829645/speco_allocator_$(date +%Y%m%d_%H%M%S)
python3 -m verl_speco.probe_ascend_state_parking \
  --tensor-parallel-size 8 --data-parallel-size 2 --rounds 4 \
  --output "$ALLOC_ROOT" 2>&1 | tee "${ALLOC_ROOT}.log"
```

检查 `$ALLOC_ROOT/report.json`：status=passed，16 个 worker 各有4轮，累计 optimizer
step=4、wake_verified=true，保留 engine/optimizer/scheduler；未知池外状态为空。
这个小模型并未实现 DeepSeek TP 数学计算，只复用同样的 TP/DP 进程组接口；它验证
分配器与 ZeRO 生命周期，不能替代 DeepSeek 前反向或 rollout 验收。

## 生产接入与边界

已提供 `training.vllm_ascend.state_parking=camem` 的生产激活/清理钩子，默认值
`none` 保留现有行为；原生 PyTorch 路径不启用该组件。CaMem 使用同一个
`AscendStateParking`，由 `AscendTrainingLifecycle` 协调 CPU/Gloo 控制、按主机
累计的暂存预算及恢复前设备容量检查。

- worker 初始化时创建 CPU 控制组，避免分计划 preflight 分支改变建组顺序。
- 初次构造与每次训练在托管池内分配；激活恢复原 engine/Adam/scheduler。
- 沿用 worker 的“缓存独立 CPU 发布快照→清理→后续 IPC 发布”顺序。
- 清理先结束所有 rank 的设备工作，再审计状态、核对主机内存并暂存；重复清理
  不读取已经取消映射的张量。预热后的 `release_training_memory_after_activation`
  使用相同释放逻辑。
- 发布导出、检查点保存需要读取设备状态时，先检查设备容量、唤醒，完成后恢复
  原来的暂存状态；使用 Gloo 传递控制和 RNG 元数据。检查点必须处于 actor/rollout
  已释放设备资源的边界，容量不足时提前拒绝，不隐式抢占其他角色。
- 使用 verl 现有 `worker_env` 在 drafter 进程导入前传入分配器配置；推理 actor
  保留原 Ray setup hook，并接收独立 HCCL 范围。推理 worker 在 native CaMem 首次
  注册前也保留绑定回调，支持后续 sleep/wake 的引用生命周期。

实际同卡 RL 启用时，在已有可运行的 DeepSeek RL 配置上追加：

```text
actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend
actor_rollout_ref.rollout.drafter.training.vllm_ascend.state_parking=camem
actor_rollout_ref.rollout.enable_sleep_mode=true
```

生产仍要求 PP=1、rollout engine DP=1，由已有 rollout replicas 提供训练 DP。
生产 DP2 engine 路由没有解禁；TP8/DP2 仅在已有无 RL 诊断的私有拓扑入口验证。
原生 FSDP offload 参数仍不适用于此后端。rollout 必须按既有调度释放设备内存；
NPU 原生 sleep level 1 的整模型恢复和完整 RL 尚未验收。


## 2026-11-04：诊断入口统一负责子进程环境

`speco_allocator_20261103_171939.log` 显示所有 worker 在环境预检退出，未进入
NPU 训练。旧错误未记录实际变量值，无法从历史日志确定当时为何不匹配。
服务器纯 CPU 复现确认，显式设置 `PYTORCH_NPU_ALLOC_CONF` 后，torch 导入与
单 worker torchrun 均保留 `expandable_segments:False`。

现在由上述 Python 入口直接启动 torchrun，在新解释器启动前明确传入分配器配置；
不再依赖手工 shell 多行环境前缀。默认 TP8/DP2 启动16进程，缺省可见设备为0–15，
HCCL 范围为61000–61099和61100–61199；已指定的设备和端口设置继续保留。
启动日志记录命令及这四项环境值，worker 预检报告实际值并核对 WORLD_SIZE，
成功报告包含各 rank 观察到的环境。`--dry-run` 只打印计划，不启动进程或创建目录。
旧式 torchrun 的全新 worker 若缺少该配置，会在导入 torch/torch-npu 前补齐同样的默认值。
已导入 runtime 时缺少配置，或显式设置了冲突值，会拒绝继续，避免事后改变分配器。
优先使用上面的统一入口。


`speco_allocator_20261104_100040.log` 再次显示外层 torchrun 路径缺少配置，
worker 实际值为 None。新增兼容分支在首次 torch 导入之前完成 worker 环境初始化，
并记录 `configuration_source=worker_bootstrap`；统一启动器传入的配置记录为 inherited。
用 `--check-environment-only` 可经过真实的16进程入口检查环境并退出，不创建输出目录、
不调用训练、不在 worker 中导入 torch/torch-npu。该检查分别覆盖直接 Python 入口
和未设置分配器环境的外层 torchrun 入口。


## 2026-11-04：已验证的阶段与下一轮验证

- `speco_allocator_20261104_103114/report.json`：真实 NPU、TP8/DP2、16 rank、
  四轮 step/scheduler=1→4，暂存/唤醒通过，池外统计量每 rank 4 字节。
- `speco_online_rounds_20261104_104627/online_rounds_report.json`：完整 DeepSeek
  两轮训练及 runtime revision 0→1→2 的 IPC/推理诊断通过；同一 trainer/Adam/
  scheduler 复用，target 固定版本0，rollout 服务轮间重启。
- 上述成功发生在生产钩子接入前。新增组件的 CPU 回归不代替新的 NPU 验收。

现在小模型入口已复用生产生命周期组件，先用上方小模型命令再次验证四轮。
通过后，在既有完整两轮命令中增加 `--production-lifecycle`：

```bash
python3 -m verl_speco.probe_deepseek_v4_online_rounds \
  --model /models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16 \
  --trajectories /models/z00829645/speco_minimal_20261029_143417/one_trajectory.jsonl \
  --output "$ONLINE_ROOT" --tensor-parallel-size 8 --data-parallel-size 2 \
  --max-model-len 4096 --response-tokens 64 --production-lifecycle \
  2>&1 | tee "${ONLINE_ROOT}.log"
```

`ONLINE_ROOT` 必须选择新目录。此模式不再用外层诊断代码包住训练和清理，而是
调用生产 `activate_training_model` / `cleanup_training` 钩子。每个 worker 必须返回
`state_lifecycle=production_hooks`，成功总报告必须包含
`production_lifecycle_hooks_validated=true`，仍然不代表真实 RL 已验收。


## 1TB 容器下的文件暂存（新增，NPU 待验证）

`speco_online_rounds_20261104_152138` 的两轮训练暂存总量分别为623.14、715.47 GiB；
推理日志报告权重约39.12 GiB/卡，16卡约625.92 GiB。原生 level-1 sleep 把权重复制
到 CPU，因而双方备份可能重叠约1341 GiB。当前 cgroup 上限为1,000,000,000,000字节
（约931.32 GiB），原有轮间重启不会暴露这个重叠峰值。

保留1TB配额时，可选 `training.vllm_ascend.state_parking_storage=file`，默认仍为
`memory`。文件模式保留所有分配的完整字节，使用有界 pinned CPU 缓冲区分块写入
共享工作目录，不改变参数地址、dtype、mask、Adam 或 scheduler：

- 每块写入完成后同步并建议释放文件页缓存，避免累积大量 dirty pages。
- 所有 rank 的文件均完成后，才提交设备取消映射；任一写入失败时关闭临时文件并拒绝交接。
- 唤醒按原地址映射，用同样大小的 staging buffer 读回；逐分配校验 SHA-256，之后
  继续校验参数地址、采样值、optimizer step 和 scheduler。
- 主存预算计入分块缓冲区，另对共享盘累计完整状态文件容量进行预检。
- 文件是匿名临时 scratch，正常唤醒后关闭，进程退出时由 OS 释放；它不是恢复检查点。

该模式增加文件 I/O，不能据此承诺吞吐提升。原生推理 sleep/wake 及常驻完整闭环
仍需后续验证。先运行四轮小模型的文件复制读回：

```bash
cd /models/z00829645/verl-SpeCo
set -o pipefail
ALLOC_ROOT="/models/z00829645/speco_allocator_$(date +%Y%m%d_%H%M%S)"
python3 -m verl_speco.probe_ascend_state_parking \
  --output "$ALLOC_ROOT" --tensor-parallel-size 8 --data-parallel-size 2 \
  --rounds 4 --state-storage file --chunk-mib 128 \
  2>&1 | tee "${ALLOC_ROOT}.log"
```

报告须同时满足 status=passed、state_storage=file、16个rank各4轮、step/scheduler连续到4、
wake_verified=true、未知池外状态为空。各轮记录 host_snapshot_bytes 和 scratch_disk_bytes，
前者不超过128 MiB/rank。真实 DeepSeek 文件暂存及常驻推理流程先不直接扩大运行。
