# Draft-OPD GPU L1 实现说明

Last updated: 11/16/2026

日期：2026-11-04。代码范围：仅 verl-SpeCo。支持 DFlash 和标准 anchor-as-first DSpark。

本次交付设计中的 P1 L1：记录真实验证计数，在真实验证起点训练 CE 或 E2E TV，并可组合已实现的 VAT。没有修改 Draft-OPD 参考仓库或 vLLM fork。L2 的被拒 token/logprob、suffix-k3、bonus-anchor、SPARK 和 NPU 不在本次交付中。

## 修改及原因

| 文件 | 修改 | 原因 |
| --- | --- | --- |
| `integration/verify_trace.py`（新增） | scheduler 快照、轨迹校验、turn 收集、token 摘要和 padding 偏移 | 区分 stop 前后的计数，保持多轮/续跑坐标 |
| `integration/vllm_runtime.py` | 扩展已有 scheduler patch 与 request stats 通道，最终输出携带轨迹和 serving revision 范围 | L1 不需要改变 proposer、采样或 vLLM 数据结构 |
| `integration/agent_loop_runtime.py` | 每条 agent trajectory 独立收集所有请求的轨迹；postprocess 前核对 token 前缀 | 防止 tool turn 或 partial rollout 只保留最后一个 request |
| `trainer/speco_ray_trainer.py` | Draft-OPD 采集时保留 tool token，单独传递 assistant mask 与轨迹 | 删除 tool token 会改变后续 anchor 和 hidden 坐标 |
| `workers/speco_worker.py`、`trainer/v1/speco_mixin.py` | 补齐 worker、V1 TQ 白名单与 feature metadata | 防止轨迹在采集或传输时丢失 |
| `backends/drafter_opd.py`（新增） | 窗口坐标映射、固定预算 anchor planner、首拒权重、分块 E2E TV | 两种算法共享 draft 索引语义 |
| 两个 `*_trainer_backend.py` | 接入 planner、可见前缀 mask、loss 与指标；DSpark 使用含 Markov bias 的分布 | 与各自标签布局一致，避免训练未访问的拒绝后缀 |
| `backends/verification_aware.py` | trace block 的 VAT 改用观测接受数和接受标签 | 避免用模拟标签覆盖真实验证证据 |
| `trainer/base_trainer.py` | 轨迹 batch padding、左填充偏移、assistant mask 与指标聚合 | 窗口裁剪和 batch padding 后仍按原 token 定位 |
| `integration/oldlogprob_layer_ids.py`、`config/speco_base.yaml` | 开关默认关闭；启用时收集 final hidden | 全词表 TV 需要 target final hidden |
| 两个 VAT 示例 | 显式设置 `full_vocab` | 修正上一批脚本继承基础示例 `restricted_ce` 的冲突 |

feature store 已逐项保留 metadata，TQ 使用同一 feature payload，因此没有另建轨迹存储协议。`trainer/v1/feature_adapter.py` 是未接入本训练路径的简化视图，本次没有修改。

## 轨迹与坐标

每个 request 的 `_speco_verify_trace` 记录 `layout`、`num_spec_tokens`、`prompt_length`、`response_length` 和 `steps`。每步包括：

- `anchor`：本步前最后一个已提交 token 的绝对位置。
- `drafts`：扣除 grammar 无效项后的有效 draft 数。
- `raw_generated`：scheduler stop 截断前返回的生成 token 数。
- `accepted`：投机步的 `raw_generated - 1`，不含纠正/bonus token。
- `commit`：stop 截断后真正追加到 request 的 token 数。
- `truncated`：`commit < raw_generated`。

非投机步仍记录 commit，维持下一步的坐标。没有输出的 chunked prefill、重算或失败 KV load 不产生虚假的验证步。快照保留 request 对象，因此结束时 scheduler 删除 request 不会丢失最后一步。

校验要求：各步 anchor 连续、commit 总和等于 response 长度、`0 <= accepted <= drafts <= capacity`，且 layout/capacity 与训练一致。DFlash 的 capacity 为 `B-1`，DSpark 为 `B`。丢失或不一致的轨迹直接报错，不自动换成随机 anchor。

多轮和续跑保留独立 request 的绝对坐标；tool token 在最终序列中占据真实位置，loss mask 为零。agent postprocess 用 token 前缀摘要检查请求序列与最终 trajectory 是否一致；发生重分词或截断而改变前缀时直接报错。本实现不猜测新的 token 映射。

采集时把 prompt 左填充长度存为 `token_offset`，预处理再扣除 `feature_start + window_start`。窗口外 anchor 被丢弃；窗口内 block 的监督长度裁剪到可用标签。沿用现有 first-response 窗口策略，没有实现 `max_anchor_density`。

## Anchor 和 loss 定义

配置位于 `actor_rollout_ref.rollout.drafter.training.drafter_opd`：

```yaml
enable: false
capture_verify_trace: false
anchor_planner: mixed
random_anchor_ratio: 0.25
loss: e2e_tv
first_reject_weight: 2.0
chunk_size: 1024
```

每个样本的 block 预算仍由对应算法的 `num_anchors` 决定：

- `random`：全部使用现有随机 sampler；保留轨迹采集，便于相同采集条件下消融。
- `trace`：从当前窗口的有效验证起点随机无放回选择。
- `mixed`：随机配额为 `ceil(num_anchors * random_anchor_ratio)`，其余配额用于 trace。trace 不足时对应 block 保持无效，不补随机 block；两类 anchor 允许重合。

trace block 只监督实际可见的已接受前缀及首拒位置。全接受块只监督有效 draft，不把额外 bonus 当作 draft；EOS 截断只监督已提交位置，不捏造拒绝标签。slot 映射为 DFlash 的 `j+1` 和 DSpark 的 `j`。

`loss=ce` 使用现有主损失：DFlash CE（以及显式配置的 LK），DSpark CE/L1/LK。trace 前缀不使用原来的固定位置衰减，真实可见首拒位乘 `first_reject_weight`；random block 保留固定衰减。DFlash 原有 front-position 权重仍有效，示例将其设为 1，避免混入额外位置加权。

`loss=e2e_tv` 替代上述 CE/L1/LK 主损失。每块在有效连续前缀上计算：

```text
TV_j = 0.5 * sum_v |softmax(target_logits / T) - softmax(draft_logits / T)|
survival_j = product_{i<=j}(1 - TV_i)
loss_block = sum_j w_j * (1 - survival_j) / num_valid_positions
```

`T` 使用现有 `lk_temperature`。先对块内位置计算，再对有效块求均值；空块不参与。teacher scoring 请求不进入 student 轨迹收集。random block 使用其完整有效前缀。

`w_j` 使用 backend 的最终位置权重：可见首拒位乘 `first_reject_weight`，random block 保留固定位置衰减；组合 VAT 时使用其返回的位置权重，DFlash front-position 权重也保留。权重作用在累计前缀失败损失上，不乘进 TV 或 survival 概率链。分母保留有效位置数，不按权重和归一化，因此只有一个可见首拒 token 时倍权仍有效；加权 loss 可以大于 1，不再解释为失败概率。所有有效位置权重为 1 时等价于旧公式。EOS/窗口截断隐藏的首拒位不加权，DFlash anchor 和不连续前缀后的无效位置不参与。

DFlash 设置 `front_position_weight=0` 时，E2E-TV 仍计算完整可见前缀的 TV；零权重位置没有自身的加权损失项，但会通过后续 survival 影响 loss 和梯度。CE/VAT 的原有样本选择保留。额外投影仅用于该零权重配置，默认正权重路径继续复用已有 logits。

2026-11-10 起，两个 backend 显式传入 `position_weights`，补齐上述 E2E-TV 权重支持；此前版本仅对 CE 应用这些权重。CE 仍用于原有诊断。训练器沿用原有 token 权重归约，不另改分布式归约协议。

该修正的 CPU 验证：`test_drafter_opd.py`、`test_verify_trace_runtime.py` 和 `test_dspark_trainer_backend.py` 共 **163 passed、1 skipped**（缺少 CUDA/NPU）。新增测试核对带权 dense loss/梯度、单个首拒位置的倍权，以及两种 backend 组合 VAT 的实际前后向路径；将 Objective 临时恢复为忽略权重时，四个 backend 回归用例均失败。Ruff 和 `git diff --check` 通过。GPU 接受长度、吞吐及训练稳定性需要重新实验；旧日志不能用作修正后的性能证据。

DSpark 的 draft logits 包含 Markov bias，prev token 取真实 response 前缀。TV 按行分块并使用 activation checkpoint；不复用旧 L1 helper，以免同时改变旧 helper 的标量返回契约。TV 模式不再计算被替代的 L1/LK 分布损失。

启用对应算法的 `per_position_loss_weight=vat` 可组合 VAT。trace block 用观测接受数，random block 保留模拟接受数；KD、head BCE、可选 self-coverage 作为增量项保留。DSpark 使用现有 confidence head。非 VAT 的原 confidence loss 不接受此组合，初始化时报错。

`vat/sim_vs_obs_visible_accept_gap` 只比较可见前缀上的模拟接受数与 `min(观测接受数, 可见长度)`，不是完整 draft 块的差值。`opd/loss`、`opd/trace_blocks`、`opd/random_blocks`、`opd/visible_tokens` 按 rank/microbatch 诊断次数求均值。

## 示例

从仓库根目录运行，先设置基础示例使用的 `MODEL_PATH`、`DRAFTER_PATH`、`TRAIN_FILE`、`TEST_FILE`、`CKPTS_DIR`：

```bash
bash examples/run_qwen3-8b-drafter_opd_dflash_vllm.sh
bash examples/run_qwen3-8b-drafter_opd_dspark_vllm.sh
```

脚本设置 GPU MRv2、full_vocab、mixed + E2E TV、同步发布、单步样本保留和禁用 data buffer。TV 温度 0.6 与基础 rollout 温度一致。GPU 数和 TP 沿用基础示例，需按实际资源覆盖。

消融与组合通过末尾参数覆盖：

```bash
# Trace-only CE + 首拒倍权
bash examples/run_qwen3-8b-drafter_opd_dflash_vllm.sh \
  actor_rollout_ref.rollout.drafter.training.drafter_opd.anchor_planner=trace \
  actor_rollout_ref.rollout.drafter.training.drafter_opd.loss=ce

# DSpark Draft-OPD + VAT
bash examples/run_qwen3-8b-drafter_opd_dspark_vllm.sh \
  actor_rollout_ref.rollout.drafter.training.dspark_per_position_loss_weight=vat \
  actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=1.0
```

## 验证与交付边界

成功标准：数值/梯度与稠密公式一致；截断、grammar、prefill、窗口和左填充坐标正确；DFlash/DSpark 的 CE/TV、VAT 组合和空块可反向；关闭开关时原有 runtime/采集测试通过；示例的实际参数符合配置约束。

本机使用 PyTorch CPU；测试导入只读引用本地 Draft-OPD/verl，没有修改该参考仓库。验证命令和最终结果在本节末尾记录。测试覆盖真实 backend、scheduler patch、request stats 传输、agent loop、采集和 feature-store batch 路径；scheduler 使用 fake 输出，不等于 vLLM GPU 实机验证。

以下边界明确保留：

- 仅 GPU、DFlash/标准 DSpark、full_vocab；packing 显式拒绝。未实现设计中的密度选窗、staleness 淘汰和 L2/k3。
- serving revision 范围随轨迹保留，但训练不据此宣称严格 on-policy；重复优化同一轨迹、异步更新及旧 target hidden 都可能造成策略差异。
- TV 来自原始 LM logits，经指定温度归一化，没有重放 top-p/top-k、penalty 或 grammar processor。即使温度一致，也不把 TV 链称为 greedy draft 场景的真实 rollout 期望接受长度。
- 多轮/续跑有坐标与 token 一致性测试；GPU async scheduling、preemption、完整训练 checkpoint/optimizer 恢复、FSDP2 空 rank collective、vLLM 热发布和 AL/耗时/显存收益仍需实机验证。没有 GPU 收益结论。

本次最终验证结果：**380 passed，3 skipped**；Ruff、Python compileall、git diff --check 通过。跳过项未计为通过。Windows 沙箱内的 asyncio socketpair 阻塞后，相关测试在沙箱外重跑完成；没有安装或更换依赖。

```bash
python -m pytest \
  tests/unit/test_drafter_opd.py \
  tests/integration/test_verify_trace_runtime.py \
  tests/integration/test_dspark_trainer_backend.py \
  tests/unit/test_vat_loss.py tests/unit/test_dpace_loss.py \
  tests/integration/test_collection_transport.py \
  tests/integration/test_drafter_runtime_control_contract.py \
  tests/integration/test_agent_loop_runtime_contract.py \
  tests/integration/test_vllm_runtime_contract.py \
  tests/unit/test_draft_feature_store.py tests/v1/test_feature_adapter.py \
  tests/integration/test_selective_publish_state.py -q
```

四个 VAT/Draft-OPD 包装脚本分别通过 Git Bash `-n`。另以仅打印参数的 `python3` 替身执行脚本，再合并实际 Hydra 参数，确认 full_vocab、Draft-OPD 开关、同步发布和温度；没有启动训练。

### 2026-11-06：vLLM 远程 worker 的轨迹传递修正

收到的报错是训练预处理缺少 `_speco_verify_trace`。检查的独立脚本已设置 `drafter_opd.enable=true` 和 `capture_verify_trace=true`，不能用再加一次开关解决。

代码中 `_configure_speco_agent_loop_manager_instance` 原来只为 SGLang 绑定显式远程 worker。vLLM 依赖主进程对上游类的猴子补丁；Ray 子进程重新导入上游类时，这些补丁不能保证保留。轨迹收集和合并依赖这些补丁，因而存在轨迹未进入最终训练样本的路径。

本次让 vLLM 也绑定现有 `SpecoAgentLoopWorker`，由其初始化在远程进程安装补丁；vLLM 继续使用原来的 rollout replica。没有修改损失、采样、脚本或缺失轨迹时的报错策略。DFlash 和 DSpark 共用此修正。

新增回归测试从未安装补丁的上游测试类开始，验证 manager 绑定、worker 初始化安装、生成结果轨迹合并及 `preprocess_trace` 的坐标结果。修正前测试在 vLLM worker 未被替换处失败；修正后，两种布局均通过。Ray 绑定在测试中使用替身，不是跨进程 GPU 验证。

本次运行：

```bash
python -m pytest tests/integration/test_agent_loop_runtime_contract.py \
  tests/integration/test_verify_trace_runtime.py tests/unit/test_drafter_opd.py \
  tests/integration/test_collection_transport.py \
  tests/integration/test_drafter_runtime_control_contract.py -q
```

结果为 **81 passed、1 skipped**；两个修改的 Python 文件通过 Ruff，`git diff --check` 通过。服务器是否走到了该缺口尚未实测：同步修改后，需要退出旧作业并新建本次作业的 Ray actors，再用原脚本复跑。成功标准是采集样本携带真实轨迹并完成首轮 Draft-OPD 训练；本地验证不能代替这一检查。

### 2026-11-09：复跑仍缺失轨迹，继续定位

用户反馈前一修正后仍在 `preprocess_trace` 报同样错误。因此，前一节的 worker 绑定修正不能视为本次服务器故障的已验证解法。

本次增加真实 CPU 测试，直接调用 `collect_online_data` 入库，再调用 DFlash backend 预处理；输入真实格式的验证轨迹时，轨迹及坐标保留正确。这排除了该测试输入下的入库字段丢失，没有证明服务器采集端产生了轨迹。

在 old-logprob 采集构造跨 worker payload 的边界，增加对必需轨迹的检查。缺少整列或某行值为 `None` 时，立即报告 step、batch_idx、request_id 和列是否存在，停止发送该批次。该改动用于定位和阻止无效样本进入训练，不会补造轨迹、跳过样本或修改目标函数。若新报错发生在此处，轨迹在 RPC 发出前已经缺失，下一步应检查 scheduler capture 与 agent-loop 合并；若仍在原预处理处报错，需要核对服务器实际代码及采集路径。

新增入库到预处理测试单独运行通过。缺失轨迹的采集边界用例受本机 trainer 导入依赖限制而随模块跳过，未计为通过。服务器根因及首轮训练完成仍待验证。

### 同日：修正同步模式下的 vLLM 配置恢复

新服务器报错确认：old-logprob 采集前轨迹列存在，但第 0 行为 `None`；问题已发生在训练 worker 的 RPC 之前。

随后在 `_ensure_vllm_server_drafter_env` 复现了配置恢复错误：原逻辑仅在候选配置携带 idle event bus 时选用嵌入的 launcher 配置。示例使用 `strategy=sync`，不会创建该 bus；远程 server 和环境配置均为空时，函数会忽略有效的嵌入配置而返回空配置。server 保留删减配置时，也会丢掉嵌入配置中的 `training.drafter_opd`。这样 server/scheduler 可能读不到 `capture_verify_trace=true`，请求仍正常完成，训练侧最终收到空轨迹。该错误与新日志吻合，但日志本身没有给出服务器当时的配置内容，不能据此宣称已完成 GPU 故障复现。

本次修改：

- server 和环境配置均为空时，恢复嵌入的完整配置，不再要求存在 idle bus。
- 选用其他配置来源时，从嵌入配置恢复 `training.drafter_opd`，保留原有 event bus 的选择以及其他 server 配置。
- sidecar 控制合并时基于已恢复的配置继续合并，避免再次覆盖 Draft-OPD 字段。

新增两个配置恢复用例，修正前均失败。另将 scheduler 传输测试扩展为从环境或嵌入配置恢复，再执行真实 patch 的采集和 finished-request 传输；没有用替身替换配置读取函数。仍使用测试 scheduler，不涉及真实 GPU 解码。

```bash
python -m pytest tests/integration/test_vllm_runtime_contract.py \
  tests/integration/test_verify_trace_runtime.py \
  tests/integration/test_acceptance_sidecar_control.py -q
```

结果：**160 passed**。同步本次 `verl_speco/integration/vllm_runtime.py` 后，需要重新创建本次作业的 Ray actors，再使用原脚本验证。成功标准仍是携带真实轨迹并完成首轮 Draft-OPD 训练；本地验证尚未覆盖这一 GPU 实机标准。

### 同日：处理 vLLM 合法的空无效 token 计数

新日志显示 EngineCore 已进入 `snapshot_verify_step`，说明轨迹采集分支已经启用。当前崩溃是直接对 `scheduled.num_invalid_spec_tokens=None` 调用 `.get()`。

本地 vLLM 参考源码中，`SchedulerOutput.num_invalid_spec_tokens` 定义为 `dict[str, int] | None = None`；`Scheduler.make_spec_decoding_stats` 仅在该字段非空时扣减无效 draft 数。因此 `None` 是框架支持的正常状态，并非需吞掉的异常。

本次只修改 `verl_speco/integration/verify_trace.py` 的计数：该字段非空时扣减，`None` 或空字典时保留已调度 draft 数。没有异常捕获或轨迹回退。DFlash、DSpark 共用该逻辑。

finished-request 传输测试增加两种布局和三种计数输入（`None`、空字典、请求含 1 个无效 token）。修正前，开启采集的四个 `None` 用例复现同一 AttributeError；修正后用例验证有效 draft 数、接受数、截断信息和输出轨迹。

```bash
python -m pytest tests/integration/test_verify_trace_runtime.py \
  tests/unit/test_drafter_opd.py tests/integration/test_vllm_runtime_contract.py \
  tests/integration/test_acceptance_sidecar_control.py -q
```

结果：**204 passed**。本次代码改动只需同步 `verify_trace.py`，然后重新启动本次作业；此前的配置恢复修改也需保留。完整 GPU 训练仍待实机验证。

### 同日：统一 DFlash 的训练和验证容量

后续日志显示轨迹已到达训练预处理，但布局或容量校验失败。原错误消息没有给出实际值；核对用户指定的 `scripts/examples_20261106/run_qwen3-8b-drafter_opd_dflash_vllm.sh`，其中 `spec_verify_tokens=16`，训练 `dflash_block_size` 则默认 16，存在确定的容量差一位问题。

vLLM 将 `spec_verify_tokens` 直接映射为 `num_speculative_tokens`，scheduler 将该容量写入轨迹。DFlash 训练 block 的位置 0 是 anchor，实际 draft 容量为 `dflash_block_size - 1`；DSpark 的容量则等于 `dspark_block_size`。因此原 DFlash 示例会记录容量 16，而预处理要求 15。该配置问题与日志吻合；新错误消息也增加实际和期望的布局、容量，以便识别额外的服务器配置差异。

本次将仓库的 Draft-OPD DFlash 包装示例和用户指定的独立脚本明确设为 `dflash_block_size=16`、`spec_verify_tokens=15`。没有修改基础示例或其他实验组。此改动减少了每轮最多 draft 数，和其他组比较接受长度或速度时，应使用相同验证长度。

同时在 `configure_vllm_runtime_from_config` 的 Draft-OPD 用户配置边界提前检查容量一致性，在启动 rollout 前报告参数和值。轨迹校验仍要求一致，不截断或改写观测容量。

新增四个启动检查用例：DFlash 16/16 和 DSpark 7/6 在修正前未被拒绝；修正后均提前报错，DFlash 16/15 和 DSpark 7/7 正常通过。相关回归结果 **208 passed**；Ruff 通过，两个修改的脚本通过 Git Bash `-n`。未验证完整 GPU 训练。

### 同日：补齐 Draft-OPD 指标上报

backend 和 worker 已产生 Draft-OPD 指标，但 `TrainingOutcome` 的白名单未转发这些字段，导致训练汇总丢失诊断信息。本次仅在 `trainer/scheduler/training_outcome.py` 增加 DFlash、DSpark 四个指标的明确白名单，沿用现有 worker 均值汇总规则，不改变损失、梯度或训练调度。

控制台及已配置的日志后端可在训练事件中收到下列指标，`<算法>` 为 `dflash` 或 `dspark`：

| 指标 | 含义及检查方法 |
| --- | --- |
| `<算法>/opd/loss` | Draft-OPD 主损失，检查是否有限；`e2e_tv` 模式下为 TV 主损失，不含额外 VAT 项 |
| `<算法>/opd/trace_blocks` | 使用真实验证 anchor 的有效块数；多次训练事件中应大于 0 |
| `<算法>/opd/random_blocks` | 使用随机 anchor 的有效块数；mixed 模式下不要求每次严格符合配置比例 |
| `<算法>/opd/visible_tokens` | 有效监督位置数，应大于 0 |

worker 先按诊断批次求均值，训练汇总再对提供该字段的 worker 求均值；这些数量不是整个训练事件的总和。内部 `opd/batch_count` 不对外转发。未启用 Draft-OPD 时不会补零或生成这些指标。判断是否完成更新还需结合 `drafter/train_successful_steps_max` 和持续增加的 `drafter/train_optimizer_step_max`。

新增测试覆盖两种算法、多 worker 均值、零值保留、未知字段排除，以及无 OPD 数据时不生成指标。修正前两种算法均复现指标缺失；修正后运行：

```bash
python -m pytest tests/unit/test_training_outcome_metrics.py tests/unit/test_drafter_opd.py -q
```

结果 **36 passed**；两个 Python 文件通过 Ruff，`git diff --check` 通过。未验证 GPU 实际日志；同步修改并重新启动作业后观察上述字段。


## Ray transport fixes and Ascend validation (2026-11-16)

The HTTP bridge now owns forwarding methods on its serialized subclass, so a
fresh Ray worker importing the unpatched upstream base retains collection and
configuration restoration. Reinstalling an early bridge refreshes its resolved
configuration instead of retaining an empty snapshot. The launcher passes the
resolved drafter configuration through the Ray job, TaskRunner and HTTP actor
runtime environments, including when upstream supplies actor-specific options.

Draft-OPD now accepts `trainer.device=npu` as well as CUDA. In the development
environment, actual small-model DFlash/DSpark CE and E2E-TV NPU backward checks
passed; a DSpark two-step RL debug run collected eight samples per step, advanced
the drafter optimizer to steps 1 and 2 and published both updates. This bounded
run does not establish model quality, performance or support for every runtime.
The baseline old-logprob feature collector still requires remove-padding;
short-run response/window lengths must leave enough rows for collection.

The transfer preserves this branch's VAT and weighted E2E-TV behavior. It does
not add SpecOPD collection or tree scoring. Local transport/objective regression:
267 passed, 7 skipped (upstream verl unavailable). Seven new regression cases
failed before the fix, including fresh-interpreter cloudpickle, late config,
actor runtime environment propagation and NPU launch checks.


Validation command:

```bash
python -m pytest -q -rs tests/integration/test_vllm_runtime_contract.py \
  tests/integration/test_native_bypass_contract.py \
  tests/integration/test_verify_trace_runtime.py \
  tests/integration/test_drafter_config_env.py \
  tests/integration/test_agent_loop_runtime_contract.py \
  tests/unit/test_drafter_opd.py tests/unit/test_vat_loss.py \
  tests/unit/test_training_outcome_metrics.py
```

Ruff checks, license, device-boundary and diff-whitespace checks pass. The
repository-wide documentation-date check still reports three unchanged files:
`draft_opd_vat_spark_design.md`, `draft_opd_vat_spark_design_review.md` and
`dspark_vat_publish_fix.md`. They already lack `Last updated` on the source branch.
