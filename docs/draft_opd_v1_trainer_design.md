# Draft-OPD 适配 verl V1 Trainer 设计文档

Last updated: 2026-11-19

状态：草案 v0.1。范围：把已在 legacy trainer（`SpecoRayPPOTrainer`）上交付的 Draft-OPD L1（真实验证轨迹 + CE/E2E-TV，可叠加 VAT）接入 `--config-name=speco_v1_trainer` 的 V1 trainer（上游 `release/v0.9.0` 的 `verl.trainer.ppo.v1`）。
前置文档：`docs/draft_opd_vat_spark_design.md`（算法设计）、`docs/drafter_opd_gpu_implementation.md`（legacy L1 实现）、`docs/bubble_time_v1.md`（V1 生命周期）。

本文所有结论来自静态阅读本仓库和 `.pytest_cache/review-ci/upstream-verl`（上游 V1 源码快照），没有运行 GPU/NPU 实验。

---

## 1. 目标与非目标

目标：

1. V1 `trainer_mode=sync` 下，Draft-OPD（DFlash、标准 anchor-as-first DSpark，GPU）与 legacy 行为一致：同一批 rollout 得到同样的 trace、anchor plan、loss 与指标。
2. V1 `colocate_async` / `separate_async` 下，正确处理 partial rollout 续跑与策略滞后，不把过期证据当作 on-policy 证据训练。
3. 所有不支持的组合在 worker 创建前 fail-closed 报错，不静默退化成随机 anchor。

非目标（沿用 legacy 交付边界）：L2 被拒 token/logprob、suffix-k3、bonus-anchor DSpark、SPARK、SGLang；NPU 动态验证另行验收。

---

## 2. 现状：V1 已复用了大部分链路

V1 adapter 的设计是"上游 V1 掌管 PPO 主循环，SpeCo 只在生命周期 hook 上挂载"。drafter 相关的采集、训练和发布由 `SpecoV1Mixin.__getattr__`（`verl_speco/trainer/v1/speco_mixin.py:206`）懒绑定到 legacy `SpecoRayPPOTrainer` 的同名方法上。因此 Draft-OPD 大部分代码在 V1 上**已经被调用**：

| 阶段 | 位置 | 是否与 trainer 无关 | V1 现状 |
| --- | --- | --- | --- |
| scheduler 采集 L1 轨迹 | `integration/vllm_runtime.py:3103-3134`，`integration/verify_trace.py:14-64` | 是（vLLM 进程内） | 直接可用 |
| server 侧附加 `prefix_digest` / revision | `integration/vllm_runtime.py:4382-4398` | 是 | 直接可用（V1 freeze probe 已依赖同一 extra_fields 通道） |
| 启动期容量校验 | `configure_vllm_runtime_from_config`（`vllm_runtime.py:4914-4939`），V1 在 `_setup` 中调用（`speco_mixin.py:966`） | 是 | 直接可用 |
| agent loop 按轨迹收集/合并多 turn | `agent_loop_runtime.py:282-308, 365-378, 690`，V1 通过 `install_v1_agent_loop_worker_bridge`（`agent_loop_runtime.py:409`，`task_runner.py:263-265`）替换 `AgentLoopWorkerTQ` | 是 | 单输出 agent loop 可用；**list 输出不可用**（G2） |
| 轨迹进入 TQ | 上游 `AgentLoopOutput.as_dict` 保留整个 `extra_fields`（`agent_loop_tq.py:194`） | 上游 | 可用 |
| TQ → legacy DataProto | `_speco_v1_batch_data`（`speco_mixin.py:1471-1533`）+ `to_legacy_padded_batch`（`trainer/v1/batch_adapter.py:21`） | V1 专有 | 已读取 `_speco_verify_trace`、request id、revision；`attention_mask` 覆盖全部 response（含 tool token），`response_mask` 是 assistant mask |
| old-logprob 采集计划与样本构造 | `_speco_build_oldlogprob_collect_plan`（`speco_ray_trainer.py:5008`）、`_speco_collect_oldlogprob_features`（`speco_ray_trainer.py:5551`，含 5751-5806 的 OPD 分支与缺失 trace 报错），V1 在 `speco_mixin.py:1652, 1706` 调用 | 共享 | 已被调用 |
| hidden layout 强制带 final hidden | `integration/oldlogprob_layer_ids.py:112-115` | 共享 | 已生效 |
| worker 白名单、`token_offset`、loss mask | `workers/speco_worker.py:915-918, 1034-1048, 1454` | 共享 | 已生效 |
| feature store / batch padding / packing 拒绝 | `trainer/base_trainer.py:4713-4779, 5712-5731, 6185-6187, 6615-6620` | 共享 | 已生效 |
| anchor planner 与 loss | `backends/drafter_opd.py`，两个 `*_trainer_backend.py` | 共享 | 已生效 |
| 训练指标白名单 | `trainer/scheduler/training_outcome.py`；V1 `_update_actor` 合并 `_speco_train_drafter` 返回值（`speco_mixin.py:2181-2193`） | 共享 | 预期可用，未有测试 |

结论：**V1 适配不需要重写算法或采集协议**，工作集中在 V1 特有的边界：TQ 行语义、上游 agent loop 的 list 输出、async 模式的续跑分段与策略滞后、配置校验与测试。`trainer/v1/feature_adapter.py` 是未接入的占位视图，本设计不使用它。

---

## 3. 差距清单

| 编号 | 问题 | 影响模式 | 严重度 |
| --- | --- | --- | --- |
| G1 | V1 没有任何 Draft-OPD 配置校验（`_speco_validate_v1_training_config`，`speco_mixin.py:835-914`）。async、Bubble、`parameter_sync_step>1`、data buffer 等组合都会被放行，错误只在 rollout 或训练时暴露 | 全部 | 高 |
| G2 | V1 `AgentLoopWorkerTQ._agent_loop_postprocess` 接受 `AgentLoopOutput | list[AgentLoopOutput]`（上游 `agent_loop_tq.py:150-156`），而 `merge_turn_traces(output)` 直接访问 `output.prompt_ids`（`verify_trace.py:120`）。多输出 agent loop 会抛 `AttributeError`；即使逐个处理，所有 turn 也会被塞给每个输出，导致 digest 校验失败 | 全部 | 高（仅多输出 loop） |
| G3 | 上游 padding 行（`padding_utils.py:94-123`，`is_padding=True`，prompt/response 各 1 token）在 V1 中 trace 被置为 `None`（`speco_mixin.py:1528`）。当前仅因 `response_len < hidden_rows` 被"短样本"规则跳过，从而没有触发缺失 trace 报错（`speco_ray_trainer.py:5789`）。这是隐式依赖：`hidden_rows` 配置得很小或上游改变 padding 形状时会直接报错 | 全部 | 中 |
| G4 | async 模式下 `FullyAsyncLLMServerClient.generate` 在 abort 后用 `prompt + 已生成 token` 重新发请求（上游 `llm_server.py:404-456`）。每段都会经过 SpeCo 打补丁的基类 `generate`，被 `collect_turn_trace` 当作一个 turn。坐标在机制上是连续的，但没有测试；被 abort 段的 trace 与返回 token 数是否一致（`vllm_runtime.py:4392` 的严格检查）未验证 | colocate/separate async | 高 |
| G5 | 策略滞后没有度量与门控。async 下 rollout 的 target 版本（tag `min_global_steps`）早于 old-logprob 使用的 actor；drafter 也可能在请求中途或段间被发布。现有 trace 只在 server 记录 `drafter_revision_min/max`，而该 revision 只在 freeze/Bubble 路径下才被更新（`vllm_runtime.py:4161-4164, 4243`），普通训练恒为 0 | async，及 sync + `parameter_sync_step>1` | 高 |
| G6 | V1 没有 Draft-OPD 示例与测试；`tests/v1/` 中没有 trace 相关用例 | 全部 | 中 |

已核对、**不是**问题的点：

- tool token：V1 不删除 tool token，`response_mask` 对 tool token 为 0（上游 `agent_loop.py:96-98`），与 legacy OPD "保留 tool token、单独传 assistant mask" 的语义一致，无需移植 legacy 中的 tool token 保留逻辑。
- `request_ids`：V1 已从 `extra_fields` 填 `_speco_vllm_request_id`，缺失 trace 的报错可带出 request id。
- separate_async 的 π_old：采集 forward 已复用上游 CPU 快照槽位（`speco_mixin.py:1844-1875`）。
- agent loop 配置：`_v1_agent_loop_config` 会删除 `drafter` 节点（`task_runner.py:205-214`），但 `collect_turn_trace` 不读配置，不受影响；vLLM 侧开关通过 `SPECO_DRAFTER_CONFIG_ENV` 与 `_ensure_vllm_server_drafter_env` 恢复。

---

## 4. 分期

| 阶段 | 范围 | 验收门槛 |
| --- | --- | --- |
| **P1** | V1 `sync` + `scheduler.execution.strategy=sync`，`parameter_sync_step=1`；单输出 agent loop（单轮与多轮 tool 调用）；G1/G2/G3/G6 | CPU 契约测试 + 与 legacy 同输入的 trace/plan 一致性测试 + GPU 冒烟 |
| **P2** | `colocate_async`、`separate_async`；G4/G5 | 分段坐标测试、staleness 门控测试、GPU async 冒烟 |
| **P3** | V1 Bubble（`rollout_idle_worker`）与 freeze/ROI 组合；多输出 agent loop 正式支持 | 另立验收 |

P1 不支持的组合一律在 `_setup` 早期报错，报错信息写明要改哪个配置项。

---

## 5. P1 设计（V1 sync）

### 5.1 配置校验（G1）

在 `_speco_validate_v1_training_config` 中新增 `_speco_validate_v1_drafter_opd_config(config)`，仅当 `training.drafter_opd.enable=true` 时执行。它在 worker 和 rollout server 创建前运行，与已有 Bubble 校验同一位置：

| 规则 | 理由 |
| --- | --- |
| `capture_verify_trace=true` | 否则所有样本 trace 为 `None`，首次采集即报错 |
| `speculative_algorithm ∈ {DFLASH, DSPARK}`，loss mode 为 `full_vocab` | 与 legacy 约束一致，提前到 V1 启动期 |
| `training.packing.enable=false` | legacy 在训练时才拒绝（`base_trainer.py:6615-6620`） |
| P1：`trainer.v1.trainer_mode=sync`，`execution.strategy=sync` | async 与 Bubble 留到 P2/P3 |
| P1：`trainer.v1.sync.parameter_sync_step=1` | 上游 sync 在每个 local batch 后更新 actor，而 rollout 只在 step 结束时同步。local batch k>0 的 old-logprob 使用已更新 k 次的 actor，target hidden 与 rollout 时的 target 不一致 |
| `use_data_buffer=false` 或 `sample_last_n_steps=1` | 与 legacy OPD 示例一致；跨步 replay 会训练过期 trace，P2 用 staleness 门控后才放开 |
| 与 actor OPD（`validate_sync_opd_cotrain_config`）的已有约束不变 | — |

DSpark bonus-anchor checkpoint 沿用 backend 初始化时的拒绝（`dspark_trainer_backend.py:189-193`），不在 V1 校验中重复。

### 5.2 agent loop：支持 list 输出（G2）

修改 `_speco_worker_agent_loop_postprocess`（`agent_loop_runtime.py:365`），并同步修改 legacy 补丁版本（763 行）：

```python
outputs = output if isinstance(output, list) else [output]
merge_turn_traces_for_outputs(outputs)   # 新函数，放在 verify_trace.py
for item in outputs:
    _speco_default_agent_loop_extra_fields(item)
```

`merge_turn_traces_for_outputs` 的规则：

- 单个输出：行为与现有 `merge_turn_traces` 完全相同（同样做全量 digest 校验，失败报错）。
- 多个输出：对每个 turn，按 `prefix_digest` 与 `prompt_length + response_length` 找到 token 前缀与之匹配的输出（`ids[:end]` 的 digest 相等），归入该输出。一个 turn 匹配零个输出时报错（证据丢失）。匹配多个输出时，归入最后一个匹配的输出，且不允许重复使用。
- P1 中多输出仍通过配置项 `drafter_opd.allow_multi_output_agent_loop=false` 默认拒绝：检测到 list 长度大于 1 时报错，说明需要 P3。匹配逻辑先实现并测试，打开开关前需要用真实多输出 loop 验证。

这样单输出路径零行为变化，多输出路径从 `AttributeError` 变为明确报错。

### 5.3 TQ 行语义：显式排除 padding 行（G3）

- `_speco_v1_batch_data` 已经计算了 `padding` 列表（`speco_mixin.py:1510-1513`）。新增 `result.non_tensor_batch["_speco_v1_is_padding"] = np.asarray(padding)`。
- `_speco_build_oldlogprob_collect_plan` 读取该列（不存在时视为全 False，legacy 不受影响）。padding 行不进入候选，并计入新指标 `drafter/oldlogprob_padding_skipped`，不再借用 `short_response_skipped`。
- 这条规则对非 OPD 也成立：padding 行不应作为 drafter 训练样本。

缺失 trace 的报错（`speco_ray_trainer.py:5789`）保持不变，只对真实行生效。

### 5.4 训练与发布

P1 不改 backend、loss 和 worker。V1 sync 的边界顺序是：old-logprob（采集）→ `_update_actor`（before hook → actor 更新 → drafter 训练）→ `on_step_end`（actor 同步 → drafter 发布），与 legacy sync 等价：trace 由版本 v 的 drafter 生成，训练从版本 v 开始，发布后成为 v+1。

### 5.5 示例

新增两个 V1 sync 示例，命名符合 `tests/special_sanity/check_example_naming.py`（`v1` 放在 model 段）：

- `examples/run_qwen3-8b-opd_v1_drafter_dflash_vllm.sh`
- `examples/run_qwen3-8b-opd_v1_drafter_dspark_vllm.sh`

内容基于现有 legacy OPD 示例，增加 `--config-name=speco_v1_trainer`、`trainer.v1.trainer_mode=sync`、`transfer_queue.enable=True`。保留 `full_vocab`、`drafter_opd.*`、`publish_async=false`、`use_data_buffer=false`、`sample_last_n_steps=1`、DFlash `spec_verify_tokens = block_size - 1`、TV 温度与 rollout 温度一致。同步更新 `tests/examples/test_example_scripts.py` 与 `ci/manual_examples.json`。

> 注意：工作区中 legacy 的四个 OPD/VAT 示例与 `tests/examples/test_example_scripts.py` 目前有未提交改动，实现时不要覆盖。

### 5.6 P1 测试

全部为 CPU 契约测试，放在 `tests/v1/`，并遵守 `SPECO_REQUIRE_V1=1` 时不得跳过的约定：

1. **配置校验**：5.1 每条规则一正一反；`drafter_opd.enable=false` 时现有 V1 配置不受影响。
2. **TQ → DataProto**：构造含 `_speco_verify_trace`（单轮、多轮 `turns`）和 padding 行的 TQ 记录，断言 `_speco_v1_batch_data` 输出的 trace、`_speco_v1_is_padding`，以及 `attention_mask`/`response_mask` 的 tool token 语义。
3. **采集计划**：padding 行即使 response 很长也被跳过；真实行缺失 trace 时仍报原错误（带 step、batch_idx、request_id）。
4. **与 legacy 一致性**：同一组 prompt/response/trace，分别走 legacy DataProto 和 V1 TQ → `to_legacy_padded_batch`，再进入 `_speco_collect_oldlogprob_features` 与 backend `preprocess_trace`，断言 sample 的 `input_ids`、`response_mask`、`token_offset` 以及 anchor plan 逐项相等（DFlash、DSpark 各一组）。
5. **agent loop**：在替换后的 `AgentLoopWorkerTQ` 上分别传单个 output 和 list output；单输出与旧行为一致，list 在开关关闭时报明确错误、打开时按 digest 正确分配。
6. **指标**：V1 `_update_actor` 的 metrics 中出现 `<算法>/opd/*`。

GPU 冒烟：两个新示例各跑 3–5 步，成功标准是 `<算法>/opd/trace_blocks > 0`、`drafter/train_optimizer_step_max` 递增，且发布成功。另用同一 seed、同一数据，与 legacy 示例对比首步 `opd/visible_tokens` 的量级。

---

## 6. P2 设计（V1 async）

### 6.1 partial rollout 分段（G4）

现状机制：abort 后的续跑是一个新的 vLLM 请求，prompt 等于原 prompt 加已生成 token。scheduler 为它新建 trace，其 `prompt_length` 等于续跑时的完整前缀长度，anchor 为绝对坐标；`collect_turn_trace` 把每一段追加到 `TURN_TRACES`；`merge_turn_traces` 用每段的 `prefix_digest` 与最终 token 序列核对；`preprocess_trace` 只拒绝重叠段。因此**分段在设计上等价于多轮**，不需要新的数据结构。

需要补的内容：

1. **abort 段的长度一致性**：验证被 abort 的 vLLM 请求返回的 `token_ids` 与 scheduler 已提交 token 数一致。如果 async scheduling 存在一步输出滞后，`vllm_runtime.py:4392` 会报错。处理原则：不放宽检查；如确有滞后，则在 scheduler 侧以返回给前端的 token 数截断最后一步（标记 `truncated`），并补测试。
2. **零 token 段**：被 abort 时尚未产出 token 的请求，trace 可能不存在或 `steps` 为空。前者在 `vllm_runtime.py:4390` 用 `extra_fields[TRACE_KEY]` 直接取值，scheduler 没有为该请求记录任何步时可能抛 `KeyError`（需用测试确认），建议改为"没有 trace 且 `token_ids` 为空时跳过"，其他情况仍报错。
3. **续跑首步**：续跑请求的第一步是 prefill，产出 1 个 token 且 `drafts=0`，它只推进坐标，不产生 block。测试需要覆盖"abort 前最后一个 verify 步"与"续跑 prefill 步"的坐标衔接。

### 6.2 策略滞后的度量与门控（G5）

定义两个按样本计算的滞后量，在 `_speco_v1_batch_data` 中从 TQ tag 与 trace 读出并写入 `non_tensor_batch`：

- **target 滞后** `policy_lag = (global_steps - 1) - min_global_steps`。`min_global_steps` 是上游 tag（`agent_loop_tq.py:216`），表示生成该轨迹最早使用的权重版本。old-logprob 在 sync/colocate_async 使用 step 开始时的 actor，在 separate_async 使用快照槽 0，三者都对应版本 `global_steps - 1`。实现时要用测试固定这个对应关系：V1 sync 下 `policy_lag` 恒为 0。
- **drafter 滞后** `drafter_lag = 当前 drafter 版本 - min(各段 drafter_revision_min)`。需要先让 serving revision 在普通训练中也有意义：drafter_opd 开启时，在 `_speco_finalize_v1_drafter_publish` 成功发布后调用已有的 `speco_set_serving_drafter_revision`（freeze 路径已经这样做，`speco_mixin.py:304-313`）。某段 `min != max` 时，说明请求中途换了 drafter，按 `max` 侧的滞后保守计算，并计入 `opd/mixed_revision_turns`。

新增配置（`training.drafter_opd`）：

```yaml
max_policy_staleness: 0     # 允许的 target 滞后；sync 下恒为 0
max_drafter_staleness: 0    # 允许的 drafter 滞后
```

门控规则：超过阈值的**样本**在采集计划中被排除，与 padding 一样不进入候选，并计入 `opd/stale_samples_skipped_{policy,drafter}`。**不**退化为 random anchor：legacy 的原则是 trace 缺失或无效就报错、不捏造证据；过期证据属于预期状态，丢弃比降级更不容易混淆实验。async 模式下默认阈值仍为 0，用户显式调大后才训练过期样本，并在指标中看到比例。

P2 放开的配置：`trainer_mode ∈ {colocate_async, separate_async}`；`use_data_buffer=true` 仅在设置了 staleness 阈值时允许，且 replay 时按同一规则过滤（复用 `data_status_policy` 按 version 淘汰）。

### 6.3 发布时序

- colocate_async：`on_sample_end` 会 abort 并 sleep 服务，drafter 在 `on_step_end` 发布，进行中的请求只会在段边界看到新 drafter。每段的 revision 是确定的。
- separate_async：standalone replica 持续服务，drafter 发布直接打到 standalone worker（`speco_mixin.py:990-1022`），请求可能在中途换 drafter，由 6.2 的 mixed revision 规则处理。

### 6.4 P2 测试

- 用 fake 的 `FullyAsyncLLMServerClient` 产生 2–3 段（含零 token 段、abort 后续跑 prefill），断言合并后的 `turns`、`preprocess_trace` 输出的 anchor 与"一次性完整生成"的结果一致。
- `policy_lag` 和 `drafter_lag` 在三种模式下的计算，以及阈值门控、指标。
- serving revision 在非 freeze 训练中随发布递增。
- GPU：`colocate_async` DSpark 冒烟，检查 `opd/stale_samples_skipped_*` 与 `trace_blocks`。

---

## 7. P3 预留（Bubble / freeze / 多输出）

- Bubble：训练在下一轮 rollout 的空闲窗口执行，并可能跨步补齐 quota（例如 step 11 补齐 step 8 的 quota），训练时的 drafter 版本可能晚于生成 trace 的版本。需要把 6.2 的 drafter 滞后接入 quota 计划，或者要求 quota 数据版本等于当前 drafter 版本。
- freeze/ROI 探针请求直接调用 serving RPC，不进入 TQ，不影响 trace；但探针请求也会产生 trace 计算开销，可在探针路径关闭。
- 多输出 agent loop：打开 5.2 的开关，并用真实多输出 loop 做 GPU 验证。

---

## 8. 改动清单

| 文件 | 改动 | 阶段 |
| --- | --- | --- |
| `verl_speco/trainer/v1/speco_mixin.py` | `_speco_validate_v1_drafter_opd_config`；`_speco_v1_batch_data` 输出 `_speco_v1_is_padding`（P1）和 `policy_lag`/`drafter_lag`（P2）；发布后更新 serving revision（P2） | P1/P2 |
| `verl_speco/trainer/speco_ray_trainer.py` | 采集计划显式跳过 padding 行（P1）与过期样本（P2），新增对应指标 | P1/P2 |
| `verl_speco/integration/verify_trace.py` | `merge_turn_traces_for_outputs` | P1 |
| `verl_speco/integration/agent_loop_runtime.py` | 两处 `_agent_loop_postprocess` 包装支持 list | P1 |
| `verl_speco/integration/vllm_runtime.py` | 零 token 段不再 `KeyError`；abort 段长度一致性（视验证结果） | P2 |
| `verl_speco/config/speco_base.yaml` | `drafter_opd.allow_multi_output_agent_loop`、`max_policy_staleness`、`max_drafter_staleness` | P1/P2 |
| `examples/run_qwen3-8b-opd_v1_drafter_{dflash,dspark}_vllm.sh` | 新增 V1 sync 示例 | P1 |
| `tests/v1/test_drafter_opd_v1.py`（新增）、`tests/examples/test_example_scripts.py`、`ci/manual_examples.json` | 5.6 与 6.4 的测试 | P1/P2 |
| `docs/drafter_opd_gpu_implementation.md` | 增加 V1 小节，链接本文 | P1 |

不改动：两个 backend、`drafter_opd.py`、worker 白名单、feature store、vLLM fork。

---

## 9. 风险与待确认

1. **target hidden 与 rollout target 的数值差异**：FSDP old-logprob 与 vLLM 采样的 logits 本来就有差异，legacy 同样存在。V1 没有引入新的差异，但 async 会叠加策略滞后，应先看 6.2 的指标再解读 AL 变化。
2. **abort 段长度**（6.1-1）需要在 GPU 上确认，结论可能改变 P2 中 vLLM 侧的改动量。
3. **`min_global_steps` 与 old-logprob 版本的对应关系**（6.2）依赖上游 vLLM server 的 `global_steps` 打标时机，实现前用 V1 sync 测试固定。
4. **过期样本的处理方式**：本文选择丢弃并计数。如果实验需要"过期 trace 退化为 random anchor"，应作为显式的 `stale_policy: drop | random` 配置，而不是默认行为。
5. **NPU**：legacy 已接受 `trainer.device=npu`。V1 NPU 示例启用了 adaptive verification，`drafts_t` 可能变长，L1 已逐步记录，但需单独做契约测试。

待确认：

- P1 是否需要覆盖多输出 agent loop？如果不需要，5.2 只做"list 长度大于 1 时报错"，匹配逻辑推迟到 P3。
- async 模式下的默认 staleness 阈值（本文建议为 0，即只训练完全 on-policy 的样本，这在 colocate_async 下可能导致大部分样本被丢弃）。
