# rollout 直接采集已回退

2026-11-11：按用户要求移除 verl rollout worker 的直接 HS 采集接入。普通 generate 不再执行采集 RPC/特征拷贝，也不再为采集关闭 prefix caching 或强制 eager。`collect_hidden_states_from_rollout=true` 在 Ascend 启动时明确拒绝。在线训练改用 actor old-logprob 采集；独立轨迹补采保留。旧两轮 rollout 采集探针已停用，以下为历史实现和验证记录。

---

# verl rollout 内直接采集训练特征

Last updated: 11/03/2026

## 路径与实现范围

采集入口是 **verl 训练中的 rollout worker**。复用当前 target 前向产生的特征，与生成结果一起交付；不会为采集再调用 generate、prefill 或 actor replay。独立 Producer 的 generate→prefill 流程保持原样。

```text
verl RL rollout target forward
  → worker 原生执行状态 / 模型 hook + sampler 边界
  → 请求级确认、连续性检查、异步 CPU 拷贝
  → 原有 HTTP server / agent-loop 返回通道
      ├─ vllm_ascend online：drafter_sample → 原有 scheduler → 自动训练与发布
      └─ vllm_ascend offline：按 target 版本写 SpeCo TorchShardFeatureStore → 离线训练
```

新增实现位于 `integration/rollout_feature_capture.py` 和 `integration/vllm_rollout_capture.py`；现有文件只增加连接与来源选择。新增收集器仅由 `training.backend=vllm_ascend` 启用，DeepSeek V4 单独验证选层特征与 pre-HC 状态。未配置 backend 或选择 `pytorch` 时，保持原有 old-logprob / SGLang 采集路径；即使设置 `collect_hidden_states_from_rollout=true`，也不会启用新增采集器。

sampler 接口接受已审查的 **vLLM 0.25.1 / vLLM-Ascend 0.25.1rc1** 或 **vLLM 0.26.0（含 +empty）/ vLLM-Ascend 0.26.0rc1**，要求 V1 model runner、禁用 FlashComm1、单级 pipeline、单请求单 completion。DeepSeek V4 使用模型实际返回的 DSpark aux 与 `get_mtp_target_hidden_states()` 的 pre-HC 状态。其他 runtime 版本会明确拒绝，CUDA/MRV2 不在本次已接通范围内。

`rollout.n` 可大于 1，由 agent-loop 拆成独立请求；限制的是每次 server generate 调用的 completion 数。

开启后使用 eager，关闭 prefix caching，确保每个特征行都经过当前前向。DeepSeek V4 使用 checkpoint 内嵌的原生 `method=dspark`，并完整加载初始 checkpoint；不经过 Qwen/DFlash 的配置、独立 draft 重载或 LM-head 同步逻辑。其 level-2 sleep 恢复尚未验收，明确拒绝；使用 level 1 或关闭 sleep。

自动训练与权重回传见 [vllm_ascend 在线闭环](vllm_ascend_cotrain.md)。下文 DeepSeek V4 离线路径是保留的独立消费方式，不是新后端唯一的训练方式。

## 数据约束

- mask 始终为 SpeCo 的 `1=监督、0=忽略`，不反转。
- 保留 chunked prefill 与普通 decode 行；投机验证只保留已确认输入对应的行，剔除 rejected draft。
- replacement/bonus token 只有实际进入后续前向才有同位置特征。末尾刚采样 token 的特征不补造；EOS/stop 截断按最终轨迹裁剪。
- 检查 token 链、绝对位置、完整覆盖、有限值和特征布局。前向缓冲区先获得独立所有权，再异步等待 DtoH 完成。
- 请求数、在途字节与行数有界；取消或失败不交付部分训练数据。验证和 probe 的 skip-collection 提示沿用现有逻辑。

## Backend 隔离

只有 `actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend` 才启用新增的直接采集、pre-HC 数据转交、目标 HC/LM-head 同步、内嵌 Drafter 权重接收、SHM 桶自动扩容和版本提交接口。采集还须显式开启 `collect_hidden_states_from_rollout=true`。

`pytorch` 与缺省 backend 保留原有训练实现、采集来源、请求准入类及权重更新顺序；V1 原有 old-logprob 配置校验继续生效。切回 PyTorch 时使用其原有模型和采集配置，不能沿用 Ascend 专用模型配置。基础共享入口保留后端选择分支，不执行新增后端逻辑。

## DeepSeek V4：RL 采集后离线训练

在已能运行 DeepSeek V4 的 RL 启动参数中加入以下 Hydra 覆盖。`model.path` 与 `drafter.model_path` 均选择同一完整 DSpark checkpoint；`spec_verify_tokens` 必须等于其 `config.json` 的 `dspark_block_size`：

```text
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.drafter.enable=true
actor_rollout_ref.rollout.drafter.enable_drafter_training=false
actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK
actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend
actor_rollout_ref.rollout.drafter.training.mode=offline
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_rollout=true
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=false
actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false
actor_rollout_ref.rollout.drafter.training.feature_store.path=/new/path/rl_features
actor_rollout_ref.rollout.enable_sleep_mode=false
```

RL 继续正常训练 actor；此离线示例主动关闭 Drafter 自动训练。特征直接写到 `/new/path/rl_features/target_step_000000NN/`，并返回 `_speco_rollout_feature_store`、`_speco_rollout_feature_key`、`_speco_rollout_feature_rows`。每个 RL run 使用新的输出根目录。缺少 serving target 版本或请求期间 target 版本改变时拒绝交付。

消费时复用[离线冒烟脚本](deepseek_v4_0731_draft_smoke.md)：

```bash
export FEATURE_STORE=/new/path/rl_features/target_step_000000NN
export MODEL_PATH=/path/to/the-matching-target-snapshot
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh preflight \
  draft_engine.target_runtime_version=NN
```

`MODEL_PATH` 必须包含该版本完整 target/teacher 权重；本实现不自动导出大型 target checkpoint。架构配置相同不能证明权重相同。预检校验声明的 runtime 版本与 store 一致，真实权重身份仍需用所选 RL checkpoint 核对。通过后再运行同样版本声明的 `train`。

## 本地验证与回退

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_training_backend_isolation.py \
  tests/unit/test_rollout_feature_capture.py \
  tests/integration/test_vllm_runtime_contract.py \
  tests/unit/test_collection_adapter.py tests/unit/test_drafter_scheduler.py
```

CPU 验证覆盖确认前缀、拒绝/替换/bonus、截断、缓冲区复用、容量失败、原有采样返回值、单次生成、版本分区存储及 V1 收集转交。2026-11-03 在服务器 verl 0.9.0 / Ray 2.48.0 环境中，后端隔离及相关回归共 351 项通过，包含原生 worker、V1、调度器与权重更新检查；日志位于 `/models/z00829645/speco_validation_archive/backend_isolation_cpu_20261103.log`。默认或 PyTorch 即使误开新采集开关，也不触发新增 RPC、特征字段或桶扩容。此轮只运行 CPU 测试；完整 RL 与 NPU 数值回归仍需单独验收。

当前记录：新增采集测试 20 项通过；扩展回归共 473 项通过、5 项跳过，另有 1 项因缺少 verl 而未执行。14 个相关源文件的 mypy 检查通过。已有文件中的 Ruff 基线问题未作批量整理，新增实现独立检查通过，连接改动没有增加 Ruff 诊断。

回退时设 `collect_hidden_states_from_rollout=false`，恢复原有采集来源配置并重新启动 worker；现有独立 Producer 和 replay 仍可用于补采、离线数据准备与一致性对照。

0.26 适配范围、环境检查和真实 runtime 源码测试见 [Ascend 0.26 兼容说明](ascend_026_compatibility.md)。verl 按当前约定使用 0.9.0，本轮没有扩展其版本兼容范围。

## DeepSeek V4 原生执行状态采集

10/29 服务器在 TP=8、DP=2、EP 开启后已成功加载并运行 target：单设备权重约 39.12 GiB，warmup 后 Torch 已分配约 47.22 GiB。后续失败为 `Rollout forward did not capture auxiliary and pre-HC states`，不是 OOM。

原因是 DeepSeek V4 的编译包装器覆写 `__call__`，直接调用 `forward()`，即使 eager 模式也绕过 `nn.Module` forward hook。DeepSeek 路径现在直接读取当前 `ExecuteModelState.aux_hidden_states`，pre-HC 仍取原生 `get_mtp_target_hidden_states()`；两者在 sampler/proposer 重用缓冲区前获得独立存储。缺失或错形状的原生 aux 会明确拒收，不用旧缓存或普通 last hidden states 代替。

DP 空批次、无已注册请求的批次，以及注册请求当步未被调度时，不读取或提交特征。原有其他模型的 layer/norm hook、正向 mask 和数据存储格式保持不变。新增 CPU 回归模拟原生 `__call__` 绕过 hook、执行状态清除、采样器重用缓冲区和 DP 空批次；实际 NPU 特征抓取仍需重试。

## RPC 的 Tensor 类型恢复

10/29 的后续 NPU 重试已经在 worker 端完成特征收集，接收端报 `AttributeError: list object has no attribute tolist`。实际 vLLM 0.26 utility result 是非强类型返回；默认设置下嵌套 Tensor 被编码成 `[dtype, shape, buffer]` 描述列表，接收端不会自动重建 Tensor，不能将其当作普通 token 数组转换。

采集 RPC 现在使用版本化的显式封装：复用 SpeCo 的 tensor-tree manifest，把 CPU tensor 字段通过 safetensors 编成二进制，接收端恢复原 dtype、shape 和嵌套结构。BF16 HS、FP32 pre-HC、int64 IDs/positions 均保持原值。独立采集和 verl rollout 采集共享该编码；Feature Store、mask、模型训练及发布协议不变，不修改 vLLM 全局 RPC 设置。

直接/in-process 返回的 Tensor 字典仍受支持；旧的非强类型描述列表会明确拒收，避免猜测 dtype。新增集成测试直接调用已安装 vLLM 的 `MsgpackEncoder/MsgpackDecoder(UtilityResult)`，先复现描述列表，再验证新的封装可恢复 Tensor；本地缺少 vLLM 时只跳过该安装环境测试。
