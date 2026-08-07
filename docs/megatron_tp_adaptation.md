# EAGLE3 草稿模型在 Megatron TP 后端的适配指南

> 本文档记录了在 Ascend NPU (8×Ascend910) 上使用 Megatron 后端 (MindSpeed)
> 训练 EAGLE3 草稿模型的完整适配方案，涵盖隐藏状态收集、ObjectRef 内存优化、
> NPU 兼容性修复等关键问题。

## 1. 环境概览

| 项目 | 值 |
|------|-----|
| 硬件 | 8× Ascend910 (65GB/卡) |
| Actor 模型 | Qwen3-8B (Megatron TP=8, PP=1) |
| 草稿模型 | Qwen3-8B_eagle3_AngelSlim (FSDP) |
| 推理引擎 | vLLM-Ascend (TP=2) |
| 参考策略 | Megatron (TP=8) |
| 训练框架 | verl-SpeCo (GRPO) |

## 2. 核心架构

### 2.1 数据流

```
Task Runner (SpecoTaskRunner)
  │
  ├─ rollout (vLLM 生成 + speculation)
  │
  ├─ compute_old_log_prob (Megatron forward)
  │    ├─ 构建 collect_plan (collect_mask, hidden_positions)
  │    ├─ 将 collect_plan 注入 batch (非张量字段)
  │    ├─ dispatch 到 WorkerDict
  │    │    ├─ Megatron forward_step → 前向 hooks 捕获隐藏状态
  │    │    ├─ postprocess_micro_batch → 消费捕获, 生成 ObjectRef
  │    │    ├─ postprocess_batch_func (aggregate) → 合并跨微批 ObjectRef
  │    │    └─ _postprocess_output → 提取 SPECO 键到 final_output
  │    └─ _speco_collect_oldlogprob_features → 从 output 提取隐藏状态
  │         → 通过 RPC 发送到 SpecoWorker
  │
  ├─ update_actor (PPO 更新)
  │
  └─ train_drafter (每 N 步触发)
       ├─ prepare_training_batch (从 collected_data 采样)
       ├─ forward → backward → optimizer.step (20 次训练步)
       └─ publish (权重推送到 vLLM)
```

### 2.2 关键文件

| 文件 | 职责 |
|------|------|
| `verl_speco/integration/oldlogprob_runtime.py` | 隐藏状态捕获 hooks, ObjectRef 管理, 聚合 patch |
| `verl_speco/trainer/speco_ray_trainer.py` | Task runner: collect_plan 构建, 特征收集, compute_old_log_prob patch |
| `verl_speco/trainer/base_trainer.py` | Drafter trainer: training_step, 数据对齐, checkpoint |
| `verl_speco/workers/speco_worker.py` | SpecoWorker: ObjectRef 解析, drafter 训练触发, 权重发布 |
| `examples/run_qwen3-8b_drafter_eagle3_megatron_vllm_npu_epoch.sh` | 启动脚本 |

## 3. 适配要点

### 3.1 禁用 Sequence Parallel

MindSpeed 的 `repatch()` 会强制设置 `sequence_parallel=True`，覆盖用户配置。
mbridge 的 `LLMBridge._build_base_config()` 也硬编码了
`"sequence_parallel": self.mpu.tp_size > 1`。

**解决方案**: 使用 `+` 前缀添加新键来覆盖 mbridge 的硬编码:

```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False
```

同时在 batch 中传递 SP 禁用标记:

```python
# speco_ray_trainer.py
tu.assign_non_tensor_data(
    batch_td, "speco_oldlogprob_sp_disabled",
    not _user_seq_parallel,
)
```

### 3.2 隐藏状态存储路径: model_output

隐藏状态需要经过 `postprocess_batch_func` 的聚合管道
(`unbind → as_nested_tensor → restore_dynamic_batch`)。

**关键**: 必须将隐藏状态张量存储在 `model_output` 字典中（而非 `output_dict`
顶层），这样原始聚合函数会自动处理:

```python
# oldlogprob_runtime.py — speco_megatron_postprocess
model_output = output_dict.setdefault("model_output", {})
if torch.is_tensor(v):
    model_output[k] = v  # 张量 → model_output (经过聚合)
elif _is_sparse_selected(v):
    output_dict[k] = v   # 稀疏 → 顶层
```

**ObjectRef 键** (`OLD_LOGPROB_HIDDEN_REFS_KEY` 等) 存储在 `output_dict` 顶层，
由 `speco_aggregate_output` patch 合并。

### 3.3 aggregate_output Patch 的模块绑定问题

`transformer_impl.py` 通过 `from ..utils import postprocess_batch_func`
直接导入函数对象，创建了一个**本地绑定**。仅 patch `utils_module` 的属性
无法传播到已导入的引用。

**解决方案**: 同时 patch 所有 `transformer_impl` 模块的本地引用:

```python
for mod_name in (
    "verl.workers.engine.megatron.transformer_impl",
    "verl.workers.engine.fsdp.transformer_impl",
    "verl.workers.engine.automodel.transformer_impl",
    "verl.workers.engine.torchtitan.transformer_impl",
    "verl.workers.engine.veomni.transformer_impl",
):
    impl_mod = importlib.import_module(mod_name)
    if getattr(impl_mod, "postprocess_batch_func", None) is original_aggregate:
        impl_mod.postprocess_batch_func = speco_aggregate_output
```

### 3.4 ObjectRef 模式: 跨微批合并

当 `sp_size <= 1` 时启用 ObjectRef 模式，将隐藏状态存储为 Ray ObjectRef
而非内联张量，避免内存耗尽。

`speco_aggregate_output` 需要正确合并跨微批的 ObjectRef:

```python
# refs/metas: 按 local batch_idx 索引 → 用 indices 重映射到全批索引
for local_idx, full_idx in enumerate(mb_indices):
    refs_merged[full_idx] = refs[local_idx]

# chunk_refs: 直接 extend (chunk 跨微批独立)
# chunk_meta: extend + 重映射 sample_indices 到全批索引
```

### 3.5 compact_selected 禁用

`compact_selected = bool(object_ref_enabled and sp_size <= 1)` 在
`sp_size <= 1` + ObjectRef 时为 True，会导致 `output_batch_size=0`
（因为 `collect_mask_sum=0` 的微批不产生任何输出）。

**解决方案**: `sp_size <= 1` 时强制 `compact_selected = False`，使 refs 按
简单本地索引存储:

```python
if sp_size <= 1:
    compact_selected = False
```

### 3.6 TORCHDYNAMO_DISABLE: NPU 兼容性

EAGLE3 模型使用 `@torch.compile(dynamic=True)` 装饰器，生成的 Triton kernel
在 NPU 上触发 `aivec error`（向量核心指令错误）。

**解决方案**: 在启动脚本中设置:

```bash
export TORCHDYNAMO_DISABLE=1
```

这使 `torch.compile` 回退为 no-op，模型使用 eager 模式运行，避免 Triton
kernel 兼容性问题。

### 3.7 actual_batch_size 修复

当 `use_remove_padding=True` 时，`hidden_positions` 等非张量字段不会被
微批分割，每个微批仍持有全批 (320 行) 的数据。但 `input_ids.offsets()`
反映的是当前微批的实际样本数。

```python
actual_batch_size = int(offsets.size(0) - 1)  # 微批实际样本数 (如 2)
if full_batch_size > actual_batch_size:
    hidden_positions = hidden_positions[:actual_batch_size]
    collect_mask = collect_mask[:actual_batch_size]
```

## 4. 启动脚本配置

### 4.1 关键环境变量

```bash
export TORCHDYNAMO_DISABLE=1          # 禁用 torch.compile, 避免 NPU aivec error
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_TMPDIR=/home/model/tmp/ray_tmpdir  # 避免 /tmp 磁盘空间不足
```

### 4.2 关键训练参数

```bash
# Actor (Megatron)
actor_rollout_ref.actor.strategy=megatron
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=8
actor_rollout_ref.actor.megatron.sequence_parallel=False
+actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=False

# Reference (Megatron)
actor_rollout_ref.ref.strategy=megatron
actor_rollout_ref.ref.megatron.param_offload=True

# Drafter
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True
actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook
actor_rollout_ref.rollout.drafter.training.use_logits=False
actor_rollout_ref.rollout.drafter.training.collect_interval_steps=10
actor_rollout_ref.rollout.drafter.training.training_interval_steps=10
```

## 5. 性能指标

### 5.1 训练开销

| 项目 | 耗时 |
|------|------|
| 正常 step (无 drafter) | ~629s |
| Drafter 训练触发开销 | ~15s |
| 隐藏状态收集开销 | ~2s |
| 权重推送开销 | ~2s |
| 单次总开销 | ~19s |
| 占比 (单次) | ~3.0% |
| 摊销占比 (每 10 步) | ~0.3%/步 |

### 5.2 训练效果

| 指标 | 初始 | Step 10 | Step 20 | 趋势 |
|------|------|---------|---------|------|
| Drafter loss | - | 12.28→11.36 | 11.33→10.97 | 下降 |
| Reward | -0.619 | +0.112 | - | 上升 |
| Val Acc | - | 0.297 | - | 提升 |
| Accept Length | 2.036 | 2.051 | 2.080 | 上升 |
| Checkpoint MD5 | - | 6d3ea02d... | 0bd709c2... | 不同 (权重更新) |

## 6. 已知问题与限制

1. **NPU OOM**: 长序列 (max_response_length=8192) 在 actor update 阶段
   可能 OOM。可减小 `max_response_length` 或 `ppo_mini_batch_size`。

2. **vLLM WorkerProc 初始化失败**: 间歇性 NPU 问题，清理后重试通常可解决。

3. **NPU 端口冲突**: HCCL 默认使用端口 65536，多进程场景需设置
   `HCCL_NPU_SOCKET_PORT_RANGE`。

4. **Triton kernel 不兼容**: NPU 不支持 `torch.compile` 生成的 Triton
   kernel，必须设置 `TORCHDYNAMO_DISABLE=1`。

5. **Ray 对象存储溢出**: 大量隐藏状态 (320×9×16384×2B ≈ 94MB/次)
   可能导致 Ray 溢出。ObjectRef 模式将数据存储在 Ray 对象存储中，
   避免在 TensorDict 管道中内联传输。
