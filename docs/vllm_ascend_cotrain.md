# vllm_ascend 随 verl 自动训练与回传权重

Last updated: 2026-11-11

rollout 直接 HS 采集已回退。当前在线来源为 actor old-logprob，配置与实测边界见
[actor 特征说明](deepseek_v4_actor_features.md)。完整 actor→训练→发布闭环尚未验收。

暂不接入完整 RL 时，先按 [无 RL 的 NPU 冒烟](deepseek_v4_no_rl_smoke.md) 验证独立采集、训练、恢复和导出后的推理。

## 目标与接入方式

目标是 DeepSeek V4 0731 DSpark 随 RL 自动更新 Drafter，不需要手动启动离线训练任务：

```text
verl actor old-logprob 前向 → hidden states + pre-HC → drafter_sample
  → 现有 collection 事务 → 同一步数据和 teacher 版本预检
  → 现有 scheduler 按 interval 触发 SpecoWorker
  → 持续存在的 DeepSpeed / SpeCo 内置 DSpark 前向、反向及 optimizer.step
  → 原有发布快照、Ray 引用、IPC 传输与失败回滚
  → rollout 对应 TP rank 原地加载 → 后续 rollout 使用更新后的 Drafter
```

`training.backend=vllm_ascend`、`training.mode=online`、`enable_drafter_training=true` 选择这条路径。
同时设置 `collect_hidden_states_from_old_logprob=true`、`collect_hidden_states_from_rollout=false`。
特征复用 actor 计算 old-logprob 的前向，通过现有内存通道交付，不要求 feature store。
离线训练的数据可通过独立轨迹 replay 获取，不恢复 rollout 直接采集。

数值部分复用已接入的 `VllmAscendDSparkEngine` 和包内 `trainer/deepseek_v4_training.py` 的 `run_dspark_training`；buffer、collection 事务、训练计划、preflight、发布缓存和回滚复用 SpeCo。新实现主要在 `trainer/vllm_ascend_online.py` 与 `integration/deepseek_v4_online.py`。

## 版本、数据及权重约束

- 当前适配成对版本 vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1、0.26.0 / 0.26.0rc1 和 0.27.1 / 0.27.1rc1（支持 +empty 本地标签）；DeepSeek V4 0731 BF16 DSpark；不开放量化权重、MTP 算法或 confidence head。
- rollout PP=1、engine DP=1、无 EP；复用现有 SpeCo mesh，`sp` 组用于训练 TP，rollout 副本维度用于训练 DP。训练和推理的 TP 度必须相同。
- 仅使用本 RL 步特征，mask 统一为 `1=监督、0=忽略`。同步完整 HC、norm、LM-head teacher 和 embedding；训练前从同一 actor 版本更新冻结的共享 embedding/head，并检查版本。
- 每次触发保留模型、DeepSpeed 优化器、scheduler、累积 optimizer step 和独立 RNG 状态。`batch_size_per_gpu` 必须等于后端数值配置的 microbatch × accumulation；只训练完整批次，每个本地批次在同一轮触发中至多消费一次。
- 参数以已处理的运行时布局发布，并标明 TP rank；接收端检查名字、形状、dtype、有限值、TP 度和完整覆盖，原地拷贝并检查参数地址不变。HC 的 FP32 不被统一转成 BF16。`weight_update_mode=auto` 在这条路径选择 compat；不能强制 native 传输协议。
- 关闭两个仅训练侧的可选投影重分片优化：`MTP_DS4_TP_SHARD_EH_PROJECTIONS=0` 和 `MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS=0`。它们会改变训练/推理的运行时参数布局。
- 初始化时也通过同一发布通道同步 Drafter，并保留初始回滚快照。恢复 RL 时加载配对 Drafter checkpoint，再发布到 rollout。

## 资源与配置

当前继续使用 SpeCo 已有的 worker 资源池布局，没有新增专用物理卡的资源池分配。DeepSpeed 实例常驻，未接入 FSDP 参数/优化器 offload；须为 actor、rollout 与 Drafter 实际驻留配置容量。这里的“在线”描述自动训练闭环，与物理同卡或分卡不是同一个概念。

训练实现已迁入 SpeCo 的 `models/`、`ops/`、`backends/`、`trainer/` 和 `data/`。默认加载包内 `config/deepseek_v4_dspark_backend.yaml`；`config_path` 可指定自己的配置，`overrides` 调整数值选项。无需 AIRDrafter checkout；旧 `source_root` 参数仅为兼容保留，不读取外部目录。迁入范围及来源见 [实现来源说明](deepseek_v4_backend_provenance.md)。`work_dir` 必须是所有训练 worker 可访问的共享路径，用于临时 TP 参数分片；每次启动分配独立子目录，成功组装快照后清理分片。它不是训练特征落盘目录。

通过 `total_optimizer_steps` 设置跨 RL 触发的 optimizer 总步数上限和学习率调度周期。不要用一次触发的 batch 数替代它。在线窗口不再经过离线分段，设后端 `train.segment_length=null`、`train.max_len=null`，并使 `training.max_seq_len`、rollout 长度、训练 runtime cache 容量相容。

## 可选的同卡状态暂存

`training.vllm_ascend.state_parking=camem` 将经过诊断验证的状态契约接入现有激活、
清理和检查点接口；默认 `none`。启用时同时设置 `rollout.enable_sleep_mode=true`。
训练 engine 和 Adam/scheduler 对象持续保留，NPU 存储在清理时按原地址暂存到 CPU；
发布仍使用清理前已缓存的 CPU 快照。新的生产钩子与 native rollout sleep 组合尚待
NPU 验收，步骤及检查点资源边界见 [状态生命周期](ascend_state_lifecycle.md)。

## NPU 两轮冒烟

先准备已经能运行该目标模型的 RL 启动脚本；脚本末尾须将 `"$@"` 转交给 `python -m verl_speco.main`。包装脚本只追加后端参数，不提供未经验证的 DeepSeek V4 actor 训练配置。

```bash
export RL_SCRIPT=/path/to/working_deepseek_v4_rl.sh
export MODEL_PATH=/path/to/DeepSeek-V4-0731-bf16
export DRAFT_WORK_DIR=/shared/speco/online-work
export DRAFTER_CKPTS=/shared/speco/drafter-checkpoints
export MTP_DS4_TP_SHARD_EH_PROJECTIONS=0
export MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS=0

bash examples/run_deepseek-v4-0731_drafter_dspark_vllm_npu.sh
```

脚本从模型配置读取 block size，设置两轮 RL、每轮一批训练、同步发布和短输出。actor 的 batch、数据、TP 数及设备数沿用基础脚本；确保每个 rollout 副本至少产生一个有效完整训练样本。可在命令后追加配置覆盖。

验收时检查：两轮 `drafter/collected_samples>0`、`drafter/trained=1`、`drafter/published=1`，optimizer step 连续增加；worker 日志出现递增的 `committed online drafter revision`。还应对比接收端参数校验和及后续推理结果，确认新参数实际参与运行。仅日志成功不足以证明 NPU 数值一致性。先完成这项短测，再扩大序列和训练批量。

## Checkpoint、回退与当前验证边界

现有 RL checkpoint 调度调用新 trainer 的 DeepSpeed save，保存模型、优化器、scheduler、进度、各 rank RNG，完成后写 `draft_step_N/speco_online.json`。自动恢复要求该边界与 actor 的 RL step 相同；实现、模型配置、数值设置、runtime 及训练拓扑不一致时拒绝恢复。也可通过 `training.vllm_ascend.resume_from` 显式指定。恢复后的初始化发布用于同步运行中的 Drafter。

切回 `training.backend=pytorch` 可选择原有训练实现；同时使用该后端支持的模型与采集配置。关闭自动训练并设置 `mode=offline`，可使用[轨迹补采](deepseek_v4_trajectory_features.md)和独立训练路径。

CPU 测试覆盖连续两轮真实 AdamW 更新、已有 worker 生命周期转交、正向 mask、HC/norm/vocab teacher 同步、TP 分片接收和恢复边界校验。Ray、DeepSpeed/NPU 实例创建、真实 TP/DP 通信、设备端发布及更新后推理仍需上述设备测试；当前不宣称已完成 NPU 闭环验收。

首批在线适配记录：新增在线测试 12 项通过；包含此前采集和后端测试的扩展回归共 485 项通过、5 项跳过、1 项因缺少 verl 未执行。16 个相关源文件 mypy 通过；新增实现 Ruff、示例命名、文档日期和脚本语法检查通过。包装脚本已用模拟 RL 启动器验证参数转发，不代表 NPU 冒烟已运行。

0.26 适配范围、环境检查和真实 runtime 源码测试见 [Ascend 0.26 兼容说明](ascend_026_compatibility.md)。verl 按当前约定使用 0.9.0，本轮没有扩展其版本兼容范围。

本轮包内迁入与独立安装的验证结果见 [实现来源说明](deepseek_v4_backend_provenance.md)。当前无需外部训练源码目录。
