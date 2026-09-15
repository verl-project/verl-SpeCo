# verl-SpeCo ROCm / MI355X 适配说明

本文档说明 `rocm-enablement` 分支上的所有改动：每处改了什么、解决了什么问题。目标是让
verl-SpeCo（构建在 verl `release/v0.8.0` 之上的 DFlash drafter 协同训练 overlay）在 AMD ROCm /
MI355X 上、以 SGLang 为后端跑通端到端流程，并且完整触发一次 drafter 的
**采集 → 训练 → 发布（collect → train → publish）** 循环。

**硬性约束：** verl-SpeCo 是对 verl 的 *仅 import 的 overlay*。**绝不能修改 verl 源码**。这里只
改动 `verl_speco/` 源码以及示例运行脚本 / 依赖约束文件。凡是能用启动参数表达的，都放在运行脚本里
保持"纯配置"，不写进源码。

## 改动总览

| 文件 | 类型 | 解决的问题 |
| --- | --- | --- |
| `verl_speco/main.py` | 源码 (+34) | Ray **driver** 上的 HIP/CUDA 可见性冲突 abort |
| `verl_speco/integration/sglang_runtime.py` | 源码 (+103) | spawn 出的 **SGLang TP** 子进程 HIP/CUDA abort；`rocminfo` 显存探测崩溃 |
| `verl_speco/integration/sglang_patch.py` | 源码 (+86) | SGLang 流式输出重构导致的 `IndexError` |
| `verl_speco/trainer/base_trainer.py` | 源码 (+8/-1) | drafter 训练+发布永不触发（样本全被丢弃） |
| `examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh` | 新增 | ROCm 冒烟运行配置（attention 后端、all-reduce、GPU 隔离） |
| `rocm_constraints.txt` | 新增 | 锁定的 ROCm 依赖集合 |

## 源码改动（verl_speco）

### 1. `verl_speco/main.py` — `_neutralize_rocm_hip_visible_devices()`（driver 侧）

**问题。** vLLM 的 ROCm 平台模块（`vllm/platforms/rocm.py`）在 import 时会执行一次性的
`_sync_hip_cuda_env_vars()`，把 `CUDA_VISIBLE_DEVICES` 复制进 `HIP_VISIBLE_DEVICES`。当 Ray 初始化
时若 `HIP_VISIBLE_DEVICES` 已存在，Ray 的 AMD 加速器管理器会切换到 HIP-keyword 模式，只按 GPU
worker 重写 HIP 变量，而每个 worker 的 `CUDA_VISIBLE_DEVICES` 仍是完整的旧 mask。ROCm 运行时随后
会在每个 GPU worker 的 `import torch` 处 abort，报
*"Conflicting visibility ... between HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES"*。

**修复。** 在 Hydra `main()` 的最开头，先主动触发一次 vLLM 的同步（让其模块体被缓存、无法再次执
行），然后 pop 掉 `HIP_VISIBLE_DEVICES`。driver 只剩 `CUDA_VISIBLE_DEVICES`，于是 Ray 保持
CUDA-native，为每个 worker 分配单块物理 GPU，且永不引入冲突的 HIP mask。在非 ROCm 构建上会
no-op（自动跳过）。

### 2. `verl_speco/integration/sglang_runtime.py` — 两个 SGLang actor 补丁

两者都在 `install_sglang_server_actor_runtime()` 中安装。

**2a. `_neutralize_rocm_hip_visible_devices()`（server-actor 侧）。** verl 的异步 SGLang server 会用
`multiprocessing` 的 `spawn` 启动方式派生 TP scheduler 子进程。由于 SGLang 跑在 Ray worker 内部，
每个被 spawn 的子进程都会重新执行 Ray 的 `default_worker.py`，它会在任何 SGLang 代码运行**之前**
import Ray 并调用 `AMDGPUAcceleratorManager.get_visible_accelerator_ids_env_var()`。当两个环境变量取
值不一致时，该管理器会抛出
`ValueError: Inconsistent values ... HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES`。这种不一致同样来自
在该 actor 内运行的 vLLM `_sync_hip_cuda_env_vars()`——在 SGLang 把 `CUDA_VISIBLE_DEVICES` 收窄到每个
TP 子进程一块物理 GPU 之后。这与 driver 侧修复（#1）是同一类 bug，但必须在同步会重新运行的 actor
内部再打一遍。

**2b. `_install_amdgpu_memory_capacity_patch()`。** SGLang 的 `get_amdgpu_memory_capacity`
（`sglang/srt/utils/common.py`）会 shell 调用 `rocminfo | grep ... | awk ...` 并把 stdout 按 float
解析。在某些 server-actor 进程里，该管道偶尔会返回空 stdout 且 returncode 为 0（末尾的 `awk` 掩盖了
上游失败），从而在 `ServerArgs.__post_init__` 里产生
`ValueError: could not convert string to float: ''`，直接杀掉整个任务。此补丁包装该函数：当原探测
抛异常或返回空时，回退到 torch 的设备总显存（取可见设备的最小值，单位 MiB）。

### 3. `verl_speco/integration/sglang_patch.py` — `patch_sglang_stream_accumulator_hidden_states()`

**问题。** 这个版本的 SGLang 把流式输出从 `scheduler_output_processor_mixin`（SPECO 现有的
hidden-state 补丁所针对的模块）重构进了
`scheduler_components/output_streamer._GenerationStreamAccumulator.accept`。旧补丁的 import 静默失败
（`logger.debug` 后 no-op），于是未打补丁的 `accept` **只**为设置了 `return_hidden_states` 的请求追加
`output_hidden_states`。tokenizer manager 随后按全 batch 位置索引 `output_hidden_states[i]`，因此
部分（仅 drafter）采集会得到一个比 `rids` 短的列表，触发
`IndexError: list index out of range`。

**修复。** 用源码改写补丁（regex + `inspect.getsource` + `exec`）替换那个条件式追加，使未采集的位置
用 `[]` 补齐。这样列表长度始终与 `rids` 一致；或者保持为空 → `None`，此时消费端的
`if getattr(...):` 守卫会短路。该补丁被接入 `patch_sglang_hidden_states_tensor_output()`，且具备
幂等性——若找不到目标代码块只告警而不崩溃。

### 4. `verl_speco/trainer/base_trainer.py` — DFlash positions 豁免

**问题。** drafter 每步都采集到样本（`collected_samples=16`），但训练从不启动：每一步都是
`schedule_reason=9 (no_trainable_batch)` 且 `train_no_trainable_batch=1`，这是结构性的（在 35+ 步上
确认——加长步数并没有用）。根因链条：

1. DFlash-aux 的 hidden states **按设计就不带 positions**——`sglang_patch.py` 的 dflash-aux 路径设置了
   `logits_output.hidden_states` 和 `_verl_dflash_aux_hidden_states` 标志，但从不设置 positions 张量；
   而 `sglang_runtime.py` 对 `uses_dflash_aux_hidden` 会**刻意跳过**它自己的 positions fail-closed 检查。
2. DFlash 训练后端（`backends/dflash_trainer_backend.py`）**根本不读** `hidden_positions`。
3. 但 `base_trainer.py` 只要 `collect_hidden_states_from_sgl=True` 就无条件丢弃任何缺 positions 的样本，
   且没有 DFlash 豁免。于是 16 个样本全被丢弃 → `trainable_batches=0` → 调度器
   （`training_trigger.py`）从不启动训练 → 什么都发布不了。

**修复。** 让 positions 要求依赖于后端类型：

```python
require_sglang_positions = bool(
    self.config.rollout.drafter.training.get("collect_hidden_states_from_sgl", False)
) and not self._is_block_drafter_backend()
```

`_is_block_drafter_backend()` 对 block 类 drafter（`dflash`、`dspark`、`domino`）返回 true。这与
`sglang_runtime.py` 中已有的豁免保持一致。DFlash 样本（`hidden_positions = None`）现在会走到已有的
`else` 传统回退窗口构建分支，该分支本就能处理 `None` 情况。EAGLE 路径（本就携带 positions）不受影响。

## 纯配置改动（运行脚本）

`examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh` 是 ROCm 冒烟启动脚本。其中与 ROCm 相关的部分都
以参数形式表达——**无需改动源码**：

- `+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=aiter` —— 当前使用的 verl
  （`verl-08`）对 `sglang>=0.5.12` 硬编码了 flashinfer 且无 ROCm 分支；此 passthrough 改为选择 AITER
  后端。
- `+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True` —— 规避
  `AiterCustomAllreduce` 的 `hipIpcGetMemHandle` abort。
- `export CUDA_VISIBLE_DEVICES=0,1` 且**从不**设置任何 HIP 变量 —— 保持 Ray CUDA-native（完整的
  Ray 2.56 AMD 加速器管理器原理见脚本头部注释）。
- 输出和 Ray 临时目录都放到 `/dev/shm`，因为宿主机磁盘已满。
- 开启 FSDP `param_offload` / `optimizer_offload`，让冒烟能在 2 卡上放下。
- `trainer.total_training_steps=${SPECO_TOTAL_TRAINING_STEPS:-3}` —— 步数可覆盖，便于跑更长的验证；
  冒烟默认仍为 3。

`rocm_constraints.txt` 锁定了经过验证的 ROCm 依赖集合：
`torch==2.11.0+rocm7.2`、`numpy==2.2.6`、`transformers==5.8.1`、`sglang==0.5.14`。

## 验证结果

在 MI355X（GPU 0,1）上以 8 步运行（`SPECO_TOTAL_TRAINING_STEPS=8`）验证：

- 全部 8 个 GRPO + DFlash 步完成（`Training Progress: 100%|██████████| 8/8`）。
- 每一步：`schedule_launch=1`、`schedule_reason=10 (training_ready)`、`trained=1`、`published=1`、
  `train_no_trainable_batch=0`。
- 零条 `Drop drafter sample: missing or mismatched SGLang hidden positions` 告警。
- 每步都有真实的 draft 权重更新回推到 SGLang。
- 退出时末尾的 `DataLoader worker ... killed by signal` traceback 是无害的收尾噪声（它出现在
  `Final validation metrics: None` **之后**，即运行已经结束之后）。

## 复现

```bash
# 在仓库根目录、verl-speco-rocm 容器内
CUDA_VISIBLE_DEVICES=0,1 SPECO_TOTAL_TRAINING_STEPS=8 \
  bash examples/run_qwen3-8b_drafter_dflash_sglang_rocm.sh
```

先用 `rocm-smi --showmemuse` 检查 GPU 占用；本机 GPU 为共享，占用是动态变化的。
