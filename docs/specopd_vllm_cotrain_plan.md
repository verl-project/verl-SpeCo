# vLLM 草稿采集与 Actor 联合前向协同训练方案

方案主线（2026-11-16 更新）：**在后训练原本必需的 actor old-logprob 前向中，共享正常轨迹主干，同时获得正常 response 的概率与草稿分支监督。** 独立分支评分仅作为正确性和性能对照，不作为默认上线流程。

Last updated: 11/16/2026

状态：已在 `.44` 的 `verl-speco-090` 容器完成 **Qwen3-8B + DSpark、Ascend NPU、vLLM MRV2、legacy 同步训练**的 Draft-OPD / SpecOPD 两步短测、更新/发布、AIME 小验证和数值探针。随后两种方法均完成 **V1 sync + TransferQueue** 两步短测、更新/发布与 AIME 小验证，退出码均为 0。验收仅覆盖这些组合与调试预算，不代表质量或性能提升。旧版结果见 [原验收结果](debug44_results.md)，V1 结果及修复见 [V1 台账](debug44_v1_validation.md)，评分核心见 [联合评分说明](specopd_joint_scoring.md)，`.242` 脚本见 [部署说明](deploy242_opd_v1.md)。

最新扩展：Ulysses 已支持当前静态 RoPE 的全注意力模型；Qwen3-8B V1 sync actor/ref SP=4、每 5 轮训练 20 步的 10-step 短测均退出 0。非采集轮使用正常 old-logprob，不采集分支也不训练草稿。数值、DP/SP 分组、发布计数及范围见 [Ulysses 台账](debug44_ulysses_validation.md)。

本地代码基线：`054683f15550af4bf28e6e6a024df15bf455b64e`。复用来源：[vx120/verl-SpeCo PR #1](https://github.com/vx120/verl-SpeCo/pull/1)，固定提交 `7611fd559f3c5a8d89034683eacb436bf0a377b7`；只合入该 PR 的改动，不引入其父分支的无关改动。

## 0. 方法选择及与基线的边界

统一入口是 `training.drafter_opd.enable` 和 `training.drafter_opd.method`：

| 选择 | 数据和 actor 评分 | drafter 训练 |
|---|---|---|
| `enable=false` | 原有 SpeCo | 原训练行为，旧 `specopd.enable` 也必须为 false |
| `enable=true, method=draft_opd` | PR 原有验证 trace，正常最终 response 前向和特征采集 | 接受前缀及首拒绝位置；CE 或加权 E2E-TV；random/trace/mixed 锚点 |
| `enable=true, method=spec_opd` | 预设窗口、完整原始 proposal，正常主干与全部有效分支一次联合 actor 前向 | 固定采集锚点、纯分支 E2E-TV、固定采样组内按 training.step 训练；当前部署每 5 轮采集、20 步后发布一次 |

`opd_method.py` 在入口、server、backend 和训练钩子中统一分派。默认方法为 `draft_opd`，因此 PR 原配置仍有效；原 `specopd.enable=true` 作为兼容入口保留，与已启用的 Draft-OPD 同时配置则报错。SpecOPD 模式忽略基线 trace 采集、随机锚点比例和首拒绝加权，不能同时跑两套采集/训练钩子。

已复用：PR 的 Draft-OPD 目标和 planner、验证 trace、远端 agent worker 补丁绑定、私有配置恢复、特征元数据传递、回归测试；两种方法共用 prefix-survival 数学函数和原有模型/LM-head/optimizer/权重发布设施。SpecOPD 自行实现完整 proposal 协议、预采集窗口、树形联合评分和一次消费调度。PR 的 trace 仅含位置和计数，不能替代完整 proposal。

以下 §1–§13 中关于“只保存 token、完整拒绝尾部、联合前向、一次消费事务”的要求专指 **SpecOPD 路径**，不强加给 Draft-OPD 基线。两种方法已通过 .44 上真实小模型 NPU 反向验证，因此 Draft-OPD 入口已放开 NPU；两者完整 serving/RL 短跑验收仍进行中，见 [实机记录](debug44_validation.md)。

## 1. 目标与已确定的边界

在现有 SpeCo 后训练流程中，复用 vLLM rollout 实际生成的草稿 proposal，将它们挂接到对应的正常轨迹主干，在 actor 更新前执行联合无梯度前向，输出正常 response 的 old-logprob 与分支监督，再以 SpecOPD 的 E2E-TV 目标训练 drafter。

- 主要开发后端为 vLLM。第一阶段选择 Qwen3-8B 全注意力模型和 DFlash，随后适配 DSpark、DFlash2。
- rollout 仅保存 token、窗口、请求映射、版本及恢复 proposal 分布所需的元数据，不保存 target hidden states/logits/概率监督。
- 每个选中的 proposal 必须包含首次拒绝之后已经生成的尾部。不能从最终 response 推测尾部。
- actor 在 old-logprob 阶段执行联合前向，正常轨迹主干只计算一次，同时产生两类输出。评分 target 是实际验证草稿的 rollout actor 对应快照，不是 PPO reference model 或外部 OPD teacher。
- actor 仍使用正常 rollout response 计算奖励、优势及 PPO/GRPO loss；分支不进入 RL loss。
- drafter 使用分支 E2E-TV，不混入正常主干 CE/L1/LK/D-PACE loss。第一阶段也关闭 confidence 辅助 loss。
- 主干用于提供 drafter 上下文及 target 评分前缀，主干 token 本身不单独产生 drafter loss。
- 第一阶段同步执行、明确版本边界、一次采集形成固定组，在 actor 更新前完成配置数量的 optimizer 更新，最后发布一次（当前配置为每 5 轮采集、20 步训练），直接以全注意力树形联合评分为目标；暂不启用异步多版本 replay、Bubble Time 或混合/线性注意力状态重放。
- 不修改已安装的 upstream verl 源码；继续使用 `verl_speco` overlay 和运行时适配。

## 2. 原始基线检查与实际扩展

本节所述“已有/当前”指原始代码检查，用于解释新增实现的必要性。现在 SpecOPD 已新增 `specopd_window.py`、`specopd_vllm.py` 和独立 proposal 协议：窗口在请求进入 rollout 前确定，raw proposal 在 scheduler 消耗前复制，正常 response 不用于重建拒绝尾部。真实运行时对齐仍待实机验证。

### 2.1 已有窗口能力

`verl_speco/config/speco_base.yaml` 的 `actor_rollout_ref.rollout.drafter.training` 已包含：

- `hidden_state_window_mode: front`；
- `hidden_state_window_tokens_per_sample: 512`；
- `hidden_state_window_min_rows: 512`；
- old-logprob hidden-state 采集相关开关。

`verl_speco/trainer/speco_ray_trainer.py`：

- `_speco_oldlogprob_window_mode` 识别 front/random；
- `_speco_build_oldlogprob_collect_plan` 在拿到最终 rollout batch 后选择窗口；
- 内部 `add_candidate` 根据实际 response 长度计算偏移，构造 `hidden_positions`、mask、owner rank；
- 随机模式使用 `sample_key:window` 的确定性散列；起点以 `prompt_len - 1` 为基准，属于 hidden-state 预测行坐标。

结论：现有策略可复用，但 actor 侧窗口是在 rollout 完成后确定的，不能原封不动作为 rollout 前的 proposal 采集计划。需要把窗口计划提前下发，并统一 token 与 hidden-row 坐标。

### 2.2 vLLM 已有原始 proposal 的候选入口

`verl_speco/integration/vllm_runtime.py`：

- `_record_vllm_worker_spec_decode_output` 从 `scheduler_output.scheduled_spec_decode_tokens[request_id]` 读取该轮草稿 token；
- 从 `model_runner_output.sampled_token_ids` 读取实际输出，目前用于接受率统计；
- `_observe_vllm_worker_output` 会复制 `scheduled_spec_decode_tokens` 和 invalid-token 元数据，避免异步 output 与下一轮 scheduler 状态串扰；
- 已有 request acceptance stats 的 staging、输出携带与 extra fields 适配路径。

目前这里保存的是计数，不是带上下文和版本的完整训练分支。`sampled_token_ids` 是接受 token 加 target 纠正/bonus 输出，不能当作原始 proposal。

`scheduled_spec_decode_tokens` 是优先验证的采集点，不直接宣称它在所有 vLLM/GPU/NPU 版本里都包含完整、有效、未裁剪的 proposal。实施时必须用一个中途拒绝的真实轮次确认：token 顺序、anchor 是否包含在数组中、invalid token 语义、动态长度和异步生命周期。若 scheduler 已提前裁剪，需前移至 drafter 输出/model runner 边界。

### 2.3 “拒绝后尾部不在现有样本中”的证据与范围

现有 old-logprob 路径评分的是最终 response，协议没有独立的 proposal 分支结构。因此仅靠该路径无法找回被丢弃的 proposal token。

作为交叉检查，SGLang 的 `integration/sglang_patch.py` 中 `_reconstruct_sglang_accepted_indices_from_lens` 和 `_filter_sglang_drafter_last_hidden_from_verify_result` 明确按接受结果选行。但这是 SGLang 的 hidden-state 行为，不能用它推定 vLLM 内部所有版本的 token 裁剪位置。

`DraftFeatureSample` 及 `transport/drafter_sample_protocol.py` 当前描述 ids、mask、hidden states、target 等线性样本，没有原始 proposal、拒绝点、branch predecessor 映射和 serving drafter revision。

### 2.4 一个额外的前置条件：草稿采样模式

`vllm_runtime.py` 注释明确保留 greedy draft 默认行为，并允许概率重叠目标显式选择概率采样。SpecOPD 需要训练概率与实际采样 proposal 一致。

不能用 greedy rollout 的 token，配上另一个温度下的 softmax，就声称是该 softmax 下的 on-policy 样本。已有明确配置键 `drafter.vllm.draft_sample_method: probabilistic`，由 config builder 传入 speculative config；但目标 vLLM 版本是否实际执行该模式仍需检查。第一阶段应启用并验证受支持的随机 proposal 模式；若运行时仅支持 greedy，则报告不支持严格 SpecOPD，不能静默降级。保持 verifier 的精确拒绝采样规则。

## 3. 总体数据流与执行时序

```text
构造 rollout 请求
  └─ 提前生成窗口计划、预算、actor/drafter revision
        ↓
vLLM 正常 speculative rollout
  ├─ 正常验证与纠正 → 最终 response → RL 奖励/优势
  └─ 窗口命中的实际轮次 → 复制原始 proposal + anchor 元数据
        ↓
response 与 proposal sidecar 按稳定 sample/request ID 汇合
        ↓
actor_t 的 old-logprob 阶段
  └─ 编译正常主干 + 全部已采草稿分支
        ↓
     同一 actor 快照、同一联合无梯度前向
        ├─ 主干读出 → 原 response 的 old logprob / 所需统计 → RL
        └─ 主干辅助层特征 + 分支 predecessor 读出 → drafter 监督
        ↓
drafter_k：重建对应 proposal 概率，计算分支 E2E-TV
        ↓
每个 optimizer step 累积整组 microbatch；完成 20 步 → 发布一次 drafter_k+1
        ↓
正常 actor 更新 → actor_t+1
        ↓
发布 actor 与 drafter，确认后进入下一轮；下一轮训练前同步对应 target head
```

**联合前向必须替换本次对应的正常 old-logprob 前向，而不是在旧前向完成后再调用一次联合评分。** 同一条主干及其分支在同一个执行单元内共享计算；一个大 rollout batch 可以拆成多个 microbatch，因此“一次前向”指每个联合 microbatch 内只调用一次模型，不要求整轮 RL 的所有请求都塞进一次调用。

预算约束优先在采集前确定，保证该主干及其所有已采有效分支能进入一个联合执行单元。采集后超预算则显式报错，不能静默丢掉分支，也不能复制主干到多个 microbatch 后再宣称主干仅计算一次。独立分支前向保留为测试 oracle 和消融基线，不作为本方案的生产降级路径。

## 4. 预设窗口与完整尾部采集规则

### 4.1 坐标约定

- token 序列使用不含 padding 的实际模型输入，含 chat template 和工具历史；不得 decode/re-tokenize 恢复。
- `prefix_len = n` 表示本轮 proposal 开始前共有 n 个已确定 token。
- anchor 是 `prefix_ids[n-1]`，anchor token 下标为 `n-1`。
- `proposal_ids[0]` 是 anchor 后的第一个未知草稿 token，不含 anchor，不含纠正 token，不含 bonus token。
- response offset 定义为 `prefix_len - request_prompt_len`，offset 0 表示从请求 prompt 末尾开始 proposal。
- 窗口 `[start_offset, end_offset)` 使用上述 response offset，而不是 hidden-state 行号。与旧窗口的 `prompt_len-1` 起点转换放在统一 helper 中。

以上是协议的归一化坐标，不是对 vLLM 内部数组的假设。vLLM 可能先采一个 target token，再以它作为下一 proposal 的 anchor；异步调度的 `num_computed_tokens` 也未必等于逻辑 prefix 长度。采集适配器必须记录真正生成该 proposal 时的逻辑前缀，不能从统计回调的当前 request 长度反推。若 anchor 是本轮新采出的 target token，应将它纳入 prefix，并校验它确实进入相应已确定历史。

### 4.2 窗口计划

建议新增 `integration/specopd_window.py`，将 front/random 的策略抽成纯函数，返回窗口计划，再由 vLLM 请求路径下发。

front 从 offset 0 开始；random 在请求的生成长度预算和可配置最大偏移内确定性选择起点。随机 seed 使用稳定 sample ID、turn ID 和 collection epoch；不依赖可能重启变化的 server request ID 单独决定。

与现有 actor 后选窗口的区别：提前选择只能依据生成长度预算，实际回答可能提前 EOS，导致窗口未触达。此时记录 `window_not_reached` 并跳过，不伪造样本、不临时改成只收集已知长回答。后续如需改善覆盖率，应单独设计并说明采样偏差。

### 4.3 轮次命中与尾部保存

仅当本轮开始的 response offset 位于窗口内，且未超过预算时采集该轮完整 proposal。不是从窗口中逐 token 随意拼接分支。

- 命中后，即使 proposal 越过窗口右边界，也保存该轮所有有效尾部。
- 首次拒绝、target 纠正、随后正式 response，都不改变已保存的原始 proposal。
- 对 invalid/padding/scheduler 占位 token 单独提供 valid length/mask；它们与“有效但被拒绝的 token”不同。
- 第一阶段只接受线性、完整可解释的 proposal；树候选需要明确选中的路径，不能把候选表展平成一条序列。
- EOS 仅保留到该分支第一个有效 EOS（含 EOS）；后面的 token 不纳入 loss。
- 动态长度可保留前缀有效 mask，或者第一阶段禁用动态草稿长度。不能根据接受长度裁 loss mask。
- 每请求设置最大分支数、总 proposal token 和 sidecar 字节预算，超额使用明确的确定性丢弃规则。

多轮 agent/tool 场景必须记录每个生成请求的精确输入 token 或其不可变引用；最终扁平 response 不一定能还原该轮工具上下文。首阶段可限制为单轮纯文本。

## 5. 已实现的数据结构与传输契约

已新增 `transport/specopd_protocol.py`，区分尚未评分的 ProposalRecord 与已评分样本，不把未经评分的记录伪装成已有 DraftFeatureSample。

### 5.1 实际 rollout wire：bundle 与 ProposalRecord

`SamplingParams.extra_args.specopd_plan` 携带预设窗口、预算、prompt 长度、请求 ID、版本、温度和 EOS 集合。完成结果通过 `trace_headers["specopd_proposals"]` 回传，最终保存在 agent extra field `_specopd_proposals` 中。wire 只含 JSON 元数据，最大 1 MiB。

| 实际字段 | 所属结构 / 用途 |
|---|---|
| `schema` | bundle 协议版本，当前 1 |
| `request_id` | bundle；稳定外部请求 ID，合并 RL batch 后必须唯一 |
| `actor_revision`、`drafter_revision` | bundle；实际 serving actor step 和已确认发布的 drafter optimizer revision |
| `proposal_temperature` | bundle；必须与本次训练和 rollout 温度一致 |
| `records` | bundle；该请求所有被预设窗口及预算选中的完整 proposal |
| `branch_id`、`round_id` | record；请求内解码轮次和分支唯一标识 |
| `prefix_len`、`prefix_digest` | record；精确已确定前缀的长度和 SHA256 |
| `proposal_ids` | record；完整有效原始草稿，不含 anchor/bonus，保留至首次 EOS（含 EOS），长度即有效长度 |
| `accepted_count` | record；诊断信息，不裁剪训练 mask |

请求完成后，单轮场景从最终 `prompt + response` 的 `trunk[:prefix_len]` 重建前缀并检查 digest；不符合就报错，不用文本或最终 response 推测拒绝尾部。多轮工具状态和停止字符串可能改变 token 历史，当前入口不支持。

当前没有单独传输 `run_id/collection_id/turn_id/window_plan_id/target_dependency_revision` 或候选表：单轮同步 group 由 driver 生命周期管理，target 依赖绑定当前 actor，同版本的 selector 候选按模型确定性重建。DFlash2 会检查已采 token 是否属于重建支持，但这不能代替实际 serving-q 数值探针。候选规则不确定、动态草稿或 sampler 处理无法恢复的部署不能宣称严格 on-policy。

### 5.2 实际评分与 drafter 数据

- `JointScoringRequest` 持有 `sample_id`、精确 `trunk_ids`、`response_start` 和固定 `DraftBranch`。
- `PackedJointBatch.branch_rows` 和 `position_ids` 在 actor 内部维护 predecessor 与逻辑位置映射，不经线性 label roll。
- `JointScores` 按主干保存一次所选层 `trunk_features`，另外返回正常 response 的 logprob/可选 entropy。
- `BranchScores` 持有 `branch_id`、predecessor `target_hidden` 和 `target_logz`。完整 `target_logits` 为测试/诊断可选项，driver 默认不传输。
- `SpecOPDCycle` 持有本组版本、消费状态和 `(request, scores)`；worker 在训练前校验同版本完整 target LM head。
- `build_specopd_training_batch` 构造固定 `anchor_positions`、`proposal_ids`、有效前缀 mask、共享主干特征及目标 hidden。首阶段需要完整词表 head，不启用子词表 head 同步。

E2E-TV 用完整分布重叠，不能只保存已采 token 的 target logprob。当前没有持久化 scored replay；在更新前完成评分和训练，失败组停止，恢复后重新采集。

### 5.3 生命周期与预算

`specopd_vllm.py` 在 scheduler `schedule` 前复制原始 proposal，在 `update_from_output` 前读取对应输出，在请求释放后只使用不可变快照；通过显式 trace header、OutputProcessor/RequestState 及 frontend registry 回传，不依赖跨进程 Python setattr。

frontend 完成缓存按请求消费，设置 TTL 和总条数上限；生成失败清理对应结果。异步 placeholder 调度直接拒绝。每请求有 request/branch/proposal-token/wire 预算；无分支请求仍回传显式空 bundle。合并 batch 后再次检查身份、版本、前缀和预算。真实取消、抢占、EOS、多副本与 IPC 行为仍需目标环境验收。

## 6. Old-logprob 中的主干与草稿分支联合前向

### 6.1 硬性执行要求

**对于同一条轨迹，正常主干和纳入采集预算的所有有效草稿分支，必须在 old-logprob 阶段通过一次 actor 模型前向完成。** 不允许先前向正常主干，再对每条草稿分支独立前向；也不把同一主干复制成多条“完整前缀 + 草稿”作为默认实现。

执行单位是 `JointTrajectory = trunk + branches[]`。数据准备可以有多个步骤，分布式模型内部可以有多个 kernel/collective，但一次执行只调用一次 actor 模型 forward；多个不同轨迹可以组成一个联合 microbatch。old-logprob 阶段的所有输出均使用 actor 更新前同一个参数快照。

分支采集预算应在 rollout 前根据联合执行的长度/显存上限设置。采集完成后，对全部已采有效分支先做完整预检；若一条轨迹无法整体进入一次前向，明确报超预算，不能静默拆开重复主干或丢弃有效分支。下一轮可调整采集预算；本轮验收“一次前向”覆盖当前完整的有效分支集合。

### 6.2 将一条轨迹编译成一个树形输入

示例正常轨迹为 `[A,B,C,D,E,F]`；在 C 后采到 `[u,v,w]`，在 E 后采到 `[x,y,z]`：

```text
逻辑结构：
正常路径：A → B → C → D → E → F
分支 1：       C → u → v → w
分支 2：               E → x → y → z

物理输入：     [A B C D E F | u v | x y]
物理行号：      0 1 2 3 4 5 | 6 7 | 8 9
逻辑位置：      0 1 2 3 4 5 | 3 4 | 5 6
```

末尾 w/z 不必作为输入，因为分别由 v/y 的输出评分。物理排列不是普通的因果序列，不能使用默认下三角 mask。定义 `parent_map` 并构造 attention 可见性：每个节点仅可见自己及其祖先。

| 查询行 | 可见上下文 |
| --- | --- |
| 主干 D | A、B、C、D |
| 分支 u | A、B、C、u |
| 分支 v | A、B、C、u、v |
| 分支 x | A、B、C、D、E、x |
| 分支 y | A、B、C、D、E、x、y |

主干不能看到任何分支；分支不能看到 anchor 后的主干未来 token 或兄弟分支。RoPE 使用各节点沿路径的逻辑位置，不能使用拼接后的物理 offset。不同轨迹在 batch 或 block-diagonal mask 上彼此隔离，padding 行不得污染任何有效节点。

### 6.3 同一次输出的两套读出映射

联合前向返回一份 hidden/logits，再根据映射分流，而不是额外调用模型：

| 用途 | 读出位置 → 被评分 token | 去向 |
| --- | --- | --- |
| 正常轨迹概率 | 主干 A→B、B→C、C→D、D→E、E→F；按 response mask 选取 | 正常 old-logprob 返回值 |
| 草稿分支 1 | 主干 C→u、分支 u→v、分支 v→w | drafter 分支监督 |
| 草稿分支 2 | 主干 E→x、分支 x→y、分支 y→z | drafter 分支监督 |

建议联合执行计划显式携带：`trunk_row_map`、`response_prediction_rows`、`response_label_ids`、`branch_predecessor_rows`、`branch_label_ids`、`position_ids`、`parent_map`、`branch_valid_mask` 以及请求/版本映射。

正常 response 的第一个 token 使用 prompt 最后一行评分；分支的第一个 token 使用对应 anchor 行评分。同一行 logits 可以同时用于正常主干的真实 next-token logprob 和分支首 token 的分布监督，不需要重复投影。后续分支 token 必须读自己的前驱行，不能读主干分叉后的行，也不能读该 token 自身的行。

正常输出恢复为现有 `old_log_probs` 的 response 形状、顺序和 padding 约定；如算法需要 entropy、其他概率统计或已有蒸馏读出，也必须从这次联合输出的主干行获得。不能为了补算这些字段再次前向主干。首次不支持的输出组合在入口报错，不悄悄走额外普通前向。

草稿监督保持独立的 payload，含 predecessor 最后层 hidden、同版本 head/归一化信息及有效分支位置。分支不追加到 PPO 的 response、reward、advantage 或 loss mask 中。

### 6.4 Drafter 上下文特征一并取得

同一次联合前向按需捕获主干辅助层特征，只保存一次，通过 `trunk_ref + anchor + visibility` 供不同分支引用。target 最后层监督按分支 predecessor 行选取；不能将重复的主干特征按每个分支全量复制传输作为默认协议。

drafter 输入按服务语义构造：DFlash 使用主干上下文、anchor 与 masks；DSpark 使用实际采样前驱的 Markov 条件；DFlash2 保留候选和 selector 语义。草稿 token 可以作为 target 因果评分的输入，但不得直接暴露给双向 draft backbone 作为答案。主干虽然提供上下文和正常 RL 概率，本方案仍只在草稿分支上计算 drafter E2E-TV，不增加主干 drafter loss。

### 6.5 接入现有 old-logprob 路径

在 `SpecoRayPPOTrainer` 的 old-logprob 入口检查是否开启联合模式：

1. 关联正常轨迹与 proposal sidecar，进行版本、前缀 digest、anchor 和预算校验。
2. 以完整 `JointTrajectory` 分配 microbatch/owner，编译 mask、位置及两套读出映射。
3. 调用 actor 的联合计算接口 `compute_joint_old_log_prob`；worker 在自身 engine 推理上下文中只执行一次模型 forward。
4. 主干读出走正常 old-logprob postprocess/返回链路；分支读出提交 drafter collection adapter。
5. 正常 `_compute_old_log_prob` 不再对同一批轨迹重复执行；原有 selected-hidden 采集也不能另外触发一遍主干前向。

没有分支的轨迹可进入同一接口并退化为普通主干因果计算，也可在调度前分配到正常 old-logprob 路径，但每条轨迹只能选择其中一路。联合接口失败不能在一次部分执行后自动调用旧接口并掩盖重复计算；按全组失败策略处理。

联合路径直接读取标准 HF hidden-state tuple 和最终归一化 hidden，复用现有辅助层 ID 解析并显式转换 decoder/HF tuple 坐标；由独立 predecessor 映射恢复概率，使 tree 行不进入原线性序列的 roll/shift-label 逻辑。原 `oldlogprob_runtime.py` 路径继续用于基线，不把它的线性行选择直接套在树上。仅在普通输入末尾追加分支，再复用原来的 `torch.roll` 标签对齐是错误的。

RL RPC 挂接实际 actor-rollout worker，参考 `integration/task_runner.py` 的 worker 包装及 `DraftWeightPublishMixin`。不要改用 SFT 专用 `SpecoTrainingWorker`。API 名称及已有参考实现见文档开头链接，最终接入以保持现有 RL 输出契约为准。

### 6.6 模型后端与数值约束

第一阶段为全注意力 Qwen3 + FSDP2 方向，先完成单进程正确性，再验证实际多卡执行。eager/SDPA 的显式 4D mask 可作为首个实现；现有 FlashAttention、remove-padding、fused logprob 或 SP 路径必须证明支持树结构才可启用，不能丢弃 mask 后继续跑。VeOmni、Megatron、线性/混合注意力及未验证 NPU kernel 组合明确报不支持。

全树一次调用时，主干 KV 在同次计算中自然共享；另一种“先单独 prefill 主干，再调模型计算分支”的实现虽然也能共享缓存，但不满足本方案当前要求的一次模型前向，应作为其他执行模式单独命名。

`torch.no_grad()`、无 optimizer 更新、关闭 dropout 或使用确定推理模式，并恢复 engine 状态。共享 actor revision 不意味着 vLLM 和 actor kernel 位级一致；首阶段固定 BF16/非量化及相同位置、上下文语义，量化实际误差。drafter 服务缓存无法由主干恢复时，需要非监督状态元数据或限制模式，不能仅凭版本号认为概率可重建。

### 6.7 节省的工作与不能省掉的工作

设主干长度为 L、分支 b 的前缀长度为 P_b、有效草稿宽度为 W_b：

```text
原 old-logprob + 独立分支评分的输入行数：L + sum_b(P_b + W_b - 1)
联合树形前向的输入行数：              L + sum_b(W_b - 1)
避免的重复前缀投影/MLP 行数：             sum_b(P_b)
```

正常 RL 必需的主干计算仍执行；新增分支前驱行、分支对前缀的注意力、mask 编译和监督传输仍有开销。若使用稠密树 mask，attention 按打包总长做计算，可能抵消部分前缀复用收益。因此“一次调用”和“吞吐更高”是两个不同验收项，不能用调用次数替代端到端测量。

## 7. Drafter 概率与 E2E-TV

### 7.1 概率重建

训练时以 proposal 采样时相同的 drafter 权重和 target 依赖参数重建 q，采样 token IDs 和候选 IDs 固定，但 q 的概率保持可微。关闭与 serving 不一致的 dropout，并验证相同上下文、候选集和采样温度下的分布对齐。

- DFlash：从并行 block 的 proposal 分布取 q。
- DSpark：使用已采样前驱，重新计算 Markov 条件概率，不重新抽前驱。
- DFlash2：使用实际 selector 候选及条件偏置；不能只优化 unary logits。
- 所有 temperature/top-k/p/约束处理必须说明其有效分布。首阶段可限制为正温度、全支持、无额外 logits processor，以缩小一致性范围。
- target p 的温度与约束语义显式定义，并与所选择的评分目标一致。训练使用 T=1 时，不宣称它直接等于 greedy serving 的接受目标。

原 forward 的 `_sample_anchor_positions` 和线性 label gather 保留给基线。实际新增 `specopd_batch` 参数和 `backends/specopd_forward.proposal_loss`：在 wrapped model forward 内部提前分派，以固定 anchor、原始 proposal 和评分 hidden 构造可微 q，保留 FSDP hooks。DFlash/DSpark 不重抽锚点；DSpark 使用实际采样前驱，DFlash2 使用候选集合和 selector。

### 7.2 目标

对 branch b 的有效位置 i：

```text
alpha[b,i] = sum_v min(p[b,i,v], q[b,i,v])
survival[b,j] = product_{i=0..j} alpha[b,i]
loss_numerator = sum_{b,j valid} (1 - survival[b,j])
loss_denominator = sum_{b,j} valid[b,j]
loss = loss_numerator / loss_denominator
```

有效位置必须构成每条分支的连续前缀。padding 用中性乘子 1，且不计入分子/分母；不能将拒绝后的有效位置 mask 掉。

FP32 softmax、overlap、cumprod 和归约；对 q 及前缀乘积求导，p detach。现有 `lk_loss.py` 的 `-log(alpha)` 和 `dpace_loss.py` 的 detached 权重不等于此 loss，建议新增 `backends/specopd_loss.py`。

候选支持 S 之外 q=0；target 必须保持全词表归一化，不能在 S 内重归一化。若使用 `target_logz`，它必须来自同一快照、同一温度和同一目标分布语义。

DP/梯度累积按全局有效位置数归一化，适配框架已有梯度平均，不能平均各 rank 的局部 mean。例如 world size 为 R、一次更新的全局有效位置总数为 N，若 DDP 在最后平均梯度，每个 rank 累积各 microbatch 的 numerator 后应等效贡献 `R * local_numerator / N`；不能再额外除一次 accumulation steps。若框架已经实现此缩放，复用而不是叠加。首阶段固定分支长度可减少歧义；若可变长度，采用上述按位置加权并明确与固定宽度论文公式的关系。

### 7.3 一次采集、一次更新

采集时记录 actor/drafter 来源版本；汇总本轮 rollout branches 成一个固定组。每个 optimizer step 对该组微批次累积梯度，当前配置连续执行 20 步，期间 actor/teacher 不更新；随后一次发布并消费/释放整组，不跨采集周期 replay。

默认现有 training.step/epoch、last_n_steps、replay、自动补 quota 等逻辑可能重复消费数据，必须给 SpecOPD 独立约束，不能仅设置一个 loss flag。

同组 20 步中使用固定采样历史、重算当前 q；这是用户要求与 Draft-OPD 对齐的多步复用采样数据近似。采样来源 q 随训练逐步偏离当前 q，不能宣称每个 optimizer step 都重新从当前 q 采样。原单步设置仍可用 training.step=1 选择。

同样不能“本轮采集不够，等 actor 下一次更新后再补”：即使 drafter 权重没变，其共享 target head/embedding 和上下文特征分布可能已变。第一阶段在当前 actor/drafter revision 内形成一个 update group；不足时本轮不训练并丢弃/释放该组，不跨 revision 合并。

## 8. 实际版本与调度

初始/恢复时先同步 target head，从实际恢复的 drafter optimizer step 获取 revision，发布并等待 acknowledgement；不假定服务上的 drafter 与本地 checkpoint 自然一致。

实际选择独立 `SpecOPDDriver` / `SpecOPDCycle`，没有新增原 scheduler 的 `on_branch_scoring_ready` 事件。原 `DrafterScheduler` 和 PR 训练钩子保留给基线；SpecOPD 入口整套替换旧在线采集/训练钩子，避免其 after-actor 事件再次训练同一组。

```text
ready → collecting → scored → updating → awaiting_publish → ready
```

非采集轮走正常 RL 前向，actor 同步后仅刷新 actor/drafter ACK；不增加 drafter revision。仅第 5、10 等采集轮进入上面的 cycle。

rollout 携带已确认 drafter revision，server 记录自身实际 actor revision；整组必须一致。同步执行期间 actor 不会在 rollout 与联合评分之间更新。评分后、actor 更新前同步当前完整 target head 并训练 drafter，每一步均完成组内微批次累积，严格检查 optimizer 累计增加 training.step（当前 20），草稿逻辑服务 revision 只在本周期发布后增加 1。worker 拒绝错 revision、重复 step 或 target-head step 不符的数据。

actor 更新并发布后，driver 发布新 drafter 并等待确认；下一轮训练前再次同步那一轮的 actor head。DSpark 等依赖 target head/embedding 的架构，必须实测 serving 与训练重建是否一致；不能把版本号相同当作数值一致的证明。target 更新导致的接受率漂移需要单独观察。

数据不足时本轮不更新 drafter，消费/释放该组，不跨 actor revision 补齐。评分、协议、候选、训练或发布异常向上抛出并停止本次执行，不回退到旧 old-logprob 重算，也不静默退回主干 CE。分布式调用先进行共享 schedule/错误预检；空 rank 仍执行 dummy forward。执行中 collective 损坏的恢复不是本地契约测试能保证的能力，需要实机故障测试。

checkpoint 不保存 pending proposal/scored group；恢复后从已提交权重和 optimizer 重新 bootstrap。这样不会把半完成组当作可 replay 数据；中途故障与多步恢复还需环境验证。

## 9. 实际修改位置

除 examples 外，路径均相对 `verl_speco/`。表中为已存在的本轮实现，不再使用拟议模块名。

| 路径 | 实际职责 |
|---|---|
| `opd_method.py`、`specopd_config.py`、`config/speco_base.yaml` | 方法选择、旧开关兼容、预算及互斥/能力校验 |
| `integration/task_runner.py` | 启动时执行能力和配置校验 |
| `main.py` | 在 Ray 启动前显式传递解析后的 drafter 配置，防止远端配置为空 |
| `integration/vllm_runtime.py` | PR 验证 trace 保留；按方法安装补丁、恢复配置、下发窗口及版本、回收结果 |
| `integration/specopd_window.py`、`specopd_vllm.py` | 预设 front/random 窗口、完整 proposal 快照、结束回传及请求预算 |
| `integration/specopd_vllm.py` | MRV2 真实 draft ID 回传：SpecOPD 下替代普通请求的 `-1` 占位符；NPU 为少量 ID 的同步 CPU 拷贝 |
| `transport/specopd_protocol.py` | 有界 JSON、token digest、坐标/版本和完整 proposal；不含 rollout 监督 tensor |
| `integration/verify_trace.py`、`agent_loop_runtime.py`、`specopd_agent.py` | 复用 PR 的 trace 与远端绑定；通过 trajectory-local ContextVar 把单次 LLM 输出的完整 proposal wire 传至 agent 输出，并传递 revision |
| `joint_scoring.py`、`joint_actor.py` | 4D 树 mask、逻辑 RoPE、双路读出、actor engine 生命周期、整树分 rank 和空 rank dummy |
| `integration/rollout_publish.py` | 真实 actor 上的 `compute_joint_old_log_prob` RPC |
| `integration/specopd_driver.py`、`trainer/speco_ray_trainer.py` | 自动替换 old-logprob；按方法选择整套训练钩子，恢复正常 RL 输出 |
| `integration/specopd_driver.py`、`integration/vllm_runtime.py` | actor/drafter 版本分离、逐 frontend 发布 ACK、请求版本检查和同一步发布指标 |
| `trainer/scheduler/specopd_cycle.py`、`workers/speco_worker.py` | bootstrap、同版本 scored group、一次消费和同步发布 |
| `trainer/base_trainer.py` | 复用 optimizer/梯度累积/发布机制；新增固定分支组的配置步数更新入口 |
| `backends/drafter_opd.py`、`opd_math.py` | PR 基线目标保留；TV 和前缀 survival 公共计算 |
| `backends/specopd_forward.py`、`specopd_loss.py` | 固定草稿历史的 forward、纯分支 E2E-TV、按有效位置归一化及 DFlash2 候选外质量 |
| `backends/dflash_trainer_backend.py`、`dspark_trainer_backend.py` | 保留原模式/PR 基线，增加 SpecOPD wrapped forward 分派；DFlash2 继承对应入口 |
| `integration/oldlogprob_layer_ids.py`、`trainer/v1/speco_mixin.py`、`trainer/v1/specopd_adapter.py` | PR 基线最后层采集和 trace 传递；V1 sync 中绑定请求 revision、读取 TQ 完整 wire、联合 old-logprob 写回 TQ、周期内多步更新及 actor 同步后的单次草稿发布/ACK；跳过旧 replay hooks |
| `examples/run_qwen3-8b-draftopd_drafter_{dflash,dspark}_vllm.sh` | 保留 PR 基线参数，按仓库规则调整脚本名 |
| `examples/run_qwen3-8b-specopd_drafter_dflash_vllm.sh` | 小预算同步双方法模板，`OPD_METHOD` 选择方法，默认 SpecOPD |
| `examples/run_qwen3-8b-opd-debug_drafter_dspark_vllm_npu.sh` | 本次 NPU 两方法短测入口，固定 16/4 数据子集、2 步；按方法设置 padding/attention |

本轮不修改 upstream verl 源码。V1 sync 已接入并短测；保留原生 TQ worker，避免通用 manager 初始化覆盖它。独立 SFT worker、离线 TQ 数据管道、V1 异步 SpecOPD、VeOmni/Megatron、异步/Bubble 和高性能树 attention kernel 均不属于已验收范围。V1 SpecOPD 强制 sync、parameter_sync_step=1，并拒绝 group-filter refill。

## 10. 实际配置与启动入口

以下为已实现配置入口。通用模板见 `examples/run_qwen3-8b-specopd_drafter_dflash_vllm.sh`；设置 `OPD_METHOD=draft_opd` 可选择基线。本次已验收的 NPU/DSpark 短测脚本为 `examples/run_qwen3-8b-opd-debug_drafter_dspark_vllm_npu.sh`，运行环境与边界见 `debug44_results.md`；其他部署组合仍需单独验证。

```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: sync
transfer_queue:
  enable: true
actor_rollout_ref:
  model:
    use_remove_padding: true
    use_fused_kernels: false
    override_config:
      attn_implementation: flash_attention_2
  actor:
    strategy: fsdp2
    ulysses_sequence_parallel_size: 4
  ref:
    ulysses_sequence_parallel_size: 4
  rollout:
    name: vllm
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    drafter:
      enable: true
      enable_drafter_training: true
      speculative_algorithm: DSPARK
      vllm:
        draft_sample_method: probabilistic
        allow_lossy_speculative_sampling: false
      rollout:
        spec_verify_tokens: 7
      training:
        mode: online
        collect_hidden_states_from_sgl: false
        collect_hidden_states_from_old_logprob: true
        collect_interval_steps: 5
        training_interval_steps: 5
        step: 20
        publish_async: false
        target_lm_head_row_restricted_sync: false
        dspark_block_size: 7
        drafter_opd:
          enable: true
          method: spec_opd
        specopd:
          enable: false # 兼容旧入口；新入口不需要开启
          collection_source: rollout_proposals
          window_mode: front
          window_response_tokens: 512
          max_branches_per_request: 8
          max_proposal_tokens_per_request: 56
          proposal_width: 7
          capture_rejected_tail: true
          scoring_phase: before_actor_update
          scorer: actor
          scoring_impl: joint_tree_old_logprob
          replace_normal_old_logprob_forward: true
          one_forward_per_trajectory: true
          independent_branches_for_validation_only: true
          scoring_token_budget: 9216
          max_dense_mask_elements: 84934656
          objective: e2e_tv
          objective_weight: 1.0
          mix_trunk_loss: false
          confidence_loss_weight: 0.0
          strict_versions: true
          max_drafter_revision_lag: 0
          optimizer_updates_per_collection: 20 # 默认由 training.step 推导；显式配置需一致
          replay: false
          allow_bubble_training: false
```

8 个分支、宽度 7 只是小预算起点，并非论文复现参数。target scoring budget 需要计算完整前缀，不仅计算 proposal token。模型最大长度约束在采集/评分前分别检查，不允许无声明裁掉前缀。

原 `hidden_state_window_min_rows=512` 是旧轨迹样本门槛，不能套到仅有几十个 proposal 位置的 branch 批次；SpecOPD 使用独立的 `min_scored_positions_per_update`，并在新数据状态策略中分派。旧 sample/window/token quota 的单位也必须区分 request、branch、proposal position 和重算 prefix token，禁止混算。

已实现的 SpecOPD 消费语义中，`step: 20` 表示当前固定 update group 完成 20 次 optimizer.step；每次均可包含多个 microbatch 累积，而不是每个 microbatch 都 step。整个周期只消费一次并在结束后发布一次。需关闭/拒绝 adaptive schedule、replay、旧 hard-selection/冻结调度或 quota top-up 对该组的重复执行；collection interval 未命中时应同时跳过新采集和对应训练。

联合评分预算按每条轨迹的 `L + sum_b(valid_width_b - 1)` 及 microbatch 合计控制；稠密 mask 另按 `batch_size * max_tree_length^2` 控制。完整轨迹单元无法容纳时，按 §6.1 预检失败处理，不能偷偷左截断、丢弃已采有效分支或复制主干分批重复评分。当前未实现单独的特征字节配置上限，主要由整树 token/mask 预算约束；环境验收要实测 host/Ray 对象峰值，再决定是否增加字节门槛。BF16 主干辅助层拼接特征近似为 `2 * context_rows * hidden_dim * num_aux_layers` 字节/轨迹，按 trunk 保存一次，另加各分支 predecessor hidden 和 head 缓存。token 预算不能替代显存/host/Ray 对象预算。

新增配置统一由 specopd 校验器解析；当开启时，旧 CE/L1/LK/D-PACE 不应被默认非零参数带入。可以选择强校验并提示冲突，或显式 loss mode 分派；无论哪种方式，都打印实际启用的目标。

## 11. 验证与可观测性

### 11.1 协议和单元验证

1. 窗口前置选择的确定性、坐标转换、短回答未触达、跨窗口尾部保留。
2. 首 token 拒绝、中途拒绝、全接受：采集 proposal 都保持原值，bonus/纠正 token 不混入。
3. 异步两轮交错、batch 重排、多副本相同 request ID、取消：归属正确且缓存释放。
4. prefix 与 score-row 小例子的精确断言；改变未来主干或 sibling 不改变目标分支监督。
5. 同版本接受，错 target/head/drafter revision 拒绝；已消费样本不能再次更新。
6. E2E-TV 手算值、有限差分梯度、padding/变长 mask、候选外 target 概率质量、全局归一化。
7. DSpark 固定 sampled predecessor、DFlash2 固定候选的概率重建对齐。
8. 真实 PPO actor 的评分 RPC 可达，SFT worker 不参与；空分支 rank、分支数不齐和一侧预检失败不会死锁。
9. 更新前事件确实只训练一次，after-actor 不重跑；checkpoint 中断后不会重复消费。
10. 比较一个大 batch 与不等长多 microbatch、多 rank 的梯度，确认归一化没有重复缩放。
11. 对同一轨迹多个锚点注册模型 forward 计数器，确认正常 old-logprob 和全部分支监督只触发一次调用；确认旧 `compute_log_prob` 不再重复调用。
12. 联合结果的正常 response logprob、所需 entropy/统计、主干辅助层和分支监督分别与独立计算对齐；padding 和多个轨迹的输出恢复不串位。

已实现 `tests/unit/test_opd_method.py`、`test_specopd_contracts.py`、`test_joint_scoring.py`、`test_joint_actor.py`；集成测试为 `tests/integration/test_specopd_lifecycle.py`、`test_specopd_forward.py`、`test_opd_launch_config.py`，并保留 PR 的 trace/objective/runtime-control 测试。上列验收项包含 CPU 已覆盖项和实机待测项，具体证据以开发记录为准，不全部标记通过。

### 11.2 真实 GPU 集成验证

- 固定权重、prompt 和采样配置；在实际 vLLM 轮次制造拒绝，核对导出的原始 proposal。不能只用 mock 确认尾部存在。
- 用独立完整因果 forward 作为评分正确性基准，对齐 branch target hidden/logits，误差阈值按 dtype 与 backend 记录。
- 核对 serving 与训练 q；实际有效 proposal 分布、候选支持、head、Markov 参数不同即阻止训练。
- 一次完整同步更新，检查 finite loss/grad、仅 drafter 参数更新、正常 RL batch 内容不变、actor 参数在评分中不变。
- 打开/关闭功能分别检查旧 old-logprob、接受率统计、发布及请求取消路径。

### 11.3 关键指标

采集：窗口命中率、未触达率、每请求 branches、完整 proposal token、拒绝后有效 token、预算丢弃、重复记录、各类版本不匹配、tail 不完整原因。

评分：branch 数、tree 实际/填充行数、避免的重复 prefix 行、正常/联合/独立模型调用数、mask 元素数、编译/联合前向/读出/传输时间、峰值显存、hidden/head 缓存字节。模型调用数按轨迹执行单元记录，避免 microbatch 数变化导致误判。

训练：overlap 分位置值、prefix survival、E2E-TV、有效位置数、梯度范数、消费量、丢弃旧版本量、真实 optimizer 更新数。

端到端：rollout elapsed、联合 oldlogprob 总耗时及编译/模型/双路读出分项、drafter critical-path time、RL step elapsed、实际接受长度、吞吐和奖励/任务指标。独立评分只在对照实验中单独计时，不与联合模式重复相加。

## 12. 实施进度与环境验收标准

代码阶段已完成方法分派、PR 基线接入、完整 proposal 旁路、联合 old-logprob、DFlash/DSpark/DFlash2 固定分支反向和同步消费/发布钩子。本地已验证 CPU 模型等价性、梯度、无重抽锚点、单次 actor 调用及生命周期顺序。本次还完成了上述 NPU/DSpark 组合的短闭环、真实 Qwen3-8B 独立因果 oracle 和原生服务 q 探针，证据见 [验收结果](debug44_results.md)。以下 A–E 是验证/扩展路线图；DFlash/DFlash2 服务、其他部署、长序列、质量和 kernel 性能等未覆盖项仍保留，不能用已通过的 DSpark 短测代替。


### A. 完整 proposal 可采性验证

先独立运行已实现的 vLLM 捕获及 token sidecar，用固定 prompt 验证拒绝尾部和 anchor。确认 window 前置下发、进程间通道、请求映射和随机 proposal 开关确实生效。先从目标运行环境记录准确的 verl/vLLM/PyTorch 版本、硬件、模型 revision、dtype/量化及 sampler 行为，形成支持矩阵。若 scheduler 点不完整，明确前移 hook；不能进入“用最终 response 拼 proposal”的替代路径。还应在实现完整训练前做一次服务 q 与训练重建 q 的数值探针；若仅保存 token/元数据无法恢复 serving 状态，该组合暂不进入阶段 B/C。

### B. Old-logprob 联合前向闭环

在真实 actor 上先运行已实现的树形编译、一次模型调用和双路读出，临时关闭 drafter optimizer 作为验收探针。用独立主干及逐分支因果 forward 作为 oracle；同时检查数值与调用计数。再替换真实 old-logprob 入口，恢复原 RL 输出契约，完成同版本 head 配套。验收要求没有额外普通 old-logprob 调用，不能仅展示一个单独可调用的联合评分 RPC 就宣称闭环完成。

### C. DFlash 同步 E2E-TV

支持配置数量的 optimizer updates（当前 20）、无主干混合 loss、单次消费事务校验与版本约束。先冻结 target 验证接受率，再接入更新 actor 的短 RL run。

### D. DSpark / DFlash2 与更多部署组合

已完成三个 backend 的 CPU 模型反向；DSpark 的真实 proposal 分布、NPU/MRV2 及 V1 sync 已按各自台账验收。其他 backend 的 V1、异步和 Bubble 扩展仍需独立适配与验收。默认严格单版本策略在各组合验证前不放开。

### E. 联合前向的 kernel 与内存优化

共享主干和树评分已是阶段 B 的主路线；此阶段根据 profile 将稠密 mask 替换为支持树结构的稀疏/分块 kernel，减少全量 logits/hidden 临时占用，优化分布式执行与 NPU 适配。仍不保存 rollout target 监督。混合 GDN 或稀疏 target 需要状态恢复/索引一致性，不能只加树 mask；也不能在扩展后默认把“一次前向”变成独立主干 prefill 加多次分支模型调用。

对照至少包含：当前 SpeCo；保留的 Draft-OPD 基线；相同分支数据及 loss 但采用“普通 old-logprob + 独立分支评分”；共享主干联合 old-logprob。后两者保持监督数据、actor/drafter 快照及更新预算一致，用于隔离联合执行带来的收益。再报告等墙钟时间下的训练效果。是否上线以实际 RL step 时间及质量为准，不以训练 loss、AL 或模型调用数单指标决定。

## 13. 与论文的关系及依据

论文：本机 `C:/Users/c00954340/Downloads/specopd-paper.pdf`，SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding。方法见 §2、条件概率与代理目标边界见附录 C、模型特定概率与更新步骤见附录 D.1–D.2；27B 的 pure SpecOPD 配置见 B.1。

本方案复用论文的“在草稿自己生成的历史上评分 + E2E-TV”和树形共享前缀机制；锚点来自真实 vLLM 解码轮次，不是固定参考数据上任意抽取的锚点，target 随 RL 更新。因此这是针对在线后训练的改造，不是论文吞吐数字的直接复现。

### 13.1 本方案重点发展的方向

**核心方向：将 RL 原本必需的正常轨迹 old-logprob 与 drafter 分支监督编译为一次共享主干的 actor 前向，并将同一输出按两套映射分别交付 RL 与 drafter。**

| 维度 | 论文 SpecOPD | 本方案主路线 |
| --- | --- | --- |
| 主干来源 | 预先收集的参考回答 | 本轮后训练的真实已验证 rollout |
| 草稿来源 | 在参考锚点额外进行草稿采样 | 复用 rollout 中实际产生的完整 proposal |
| target 任务 | 为草稿训练提供监督 | 同时提供正常 RL old-logprob 和分支监督 |
| 主干计算的复用对象 | 不同草稿分支之间共享 | 正常 RL 概率任务与所有分支监督共同共享 |
| 参数生命周期 | 实验中固定 target | actor 更新前同一快照评分，之后受控更新/发布 |

树形 mask、E2E-TV、接受与拒绝 proposal 训练等不能单独当作新增贡献。值得进一步评估的技术点是联合执行计划、正常概率与分支监督的双路读出、保持 RL 输出契约的 old-logprob 替换，以及它们与版本边界的组合。上述差异是设计定位，不等于已经确认专利新颖性/创造性；需进一步检索并用实现和实验支撑。

### 13.2 需要证明的技术效果

- 相对普通 old-logprob 后再评分分支，避免重复的主干投影和前缀计算。
- 同一联合单元只调用一次 actor 模型，且主干概率与分支监督同时满足独立前向等价性。
- 不保存 rollout 监督张量，不额外生成参考回答，也不改变原 RL 样本和优势计算。
- 在相同分支训练预算下，测量新增监督的边际耗时、显存及 RL step 时间；稠密 mask 若更慢，应如实报告并定位 kernel 成本，不预设一定加速。

论文的监督 token 吞吐收益相对顺序 OPD，不等于当前 SpeCo 的预期收益。正常 RL rollout 本来就必须执行，额外分支评分和传输能否被后续解码节省覆盖，需要上述端到端实验回答。

### .44 调试补充：Ray 运行时配置传递（2026-11-16）

实测发现，HTTP 的 generate 补丁可以正常执行，但服务进程可能收到空的 drafter 配置，从而没有启用验证轨迹或 proposal 采集。不能仅凭 rollout 能生成或补丁类名存在判断链路已接通。

实际代码在 `verl_speco/main.py` 增加了启动前的配置传递：把解析后的 drafter JSON 写入 Ray job 的 `runtime_env.env_vars.VERL_SPECO_DRAFTER_CONFIG`，并交给 TaskRunner。`vllm_runtime.py` 同时保留 HTTP 类快照、重复安装时刷新快照，以及 HTTP actor 自身的 env 传递。同步 OPD 不依赖训练器稍后修改的父进程环境变量；有运行时更新的配置仍按服务侧恢复规则处理。

该变更正在 .44 验证；当前尚不能据此宣称两种方法均完成草稿更新。每轮日志、失败原因和未验证项见 `debug44_validation.md`。

### .44 实测补充：MRV2 草稿 ID 回传

当前 vLLM 0.30 开发版的 `DraftTokensHandler` 对普通请求仅向同步调度器发送 `-1` 占位符。仅在 scheduler 中复制 `request.spec_token_ids` 不能获得真实 proposal；拒绝后尾部也无法从该占位符恢复。

`specopd_vllm.py` 现在为 SpecOPD 安装独立的 token transport hook：CUDA 复用原生异步拷贝通路；NPU 明确将本批草稿 token ID 拷回 CPU，通过原生 `DraftTokenIds` 返回。普通 Draft-OPD 保持原路径。NPU 当前是同步的小整数元数据拷贝，仍需计入采集开销；它不增加模型前向，也不保存 rollout 的 target hidden/logits 监督。完整 proposal 仍在调度器截断前保存，负值/宽度/版本检查保持严格。

原生 NPU transport 小测试已通过；完整 SpecOPD 训练与发布正在验证，不能以该小测试替代端到端结果。

### .44 实测补充：主模型与草稿版本必须分别确认

草稿专用发布不能调用主模型的 `set_global_steps` 来标记草稿训练步数；SpecOPD 分支已取消这一混用，普通 Draft-OPD 路径保持既有行为。`SpecOPDDriver` 在初始/恢复发布及每轮提交后，向各 rollout frontend 设置并收集草稿 revision ACK，ACK 同时返回该服务的原生 actor revision。所有副本需要一致，old-logprob 的 bundle 也需要与 ACK 匹配。服务在生成前检查请求草稿 revision 与实际已安装版本，不能把请求自己声明的版本直接当成发布证据。

发布完成的指标会补入上游即将记录的同一个 actor_output，避免日志滞后一轮。该严格边界正在最终短测前验证，详细进展以 `debug44_validation.md` 为准。

## Ulysses 联合评分执行方式

`joint_ulysses.py` 在一次评分调用内覆盖 attention 实例方法：输入沿物理 token 行切分，QKV all-to-all 汇总序列并切分 heads，按完整树 mask 计算注意力，再逆向 all-to-all 恢复本地 token shard。GQA 在需要时仅复制足够的 KV heads。逻辑 RoPE 位置随 token 一起切分，不用物理行号替代分支位置。

DP 按 SP 组分配完整轨迹；组内协同计算，只由 leader 输出。空 DP 组运行 dummy，防止 FSDP collective 失配。仅归并所需 response 概率、分支 readout 和辅助 hidden，不 all-gather 整个序列×词表 logits。

普通 actor/ref 的 PPO 前向仍使用 upstream unpadding + FlashAttention/Ulysses；联合评分结束后恢复实例方法。SP=1 保留原 dense eager/SDPA 路径、remove_padding=false。SP>1 使用 remove_padding=true；本次 SP=4。动态/adaptive RoPE、LoRA、滑窗/混合模型仍拒绝。当前真实模型长序列/16 die、性能和 checkpoint 恢复不由短测覆盖。


### 固定组的执行效率（2026-11-16）

联合old-logprob把完整微批计划一次提交给actor，保持一次engine驻留；各轨迹仍按既定微批依次forward，不增加单个树mask峰值。草稿按实际DP组分配整条轨迹，SP组内共享；组内每个optimizer step都覆盖完整组数据的全局token均值，20步后统一发布。非均匀分配使用零loss dummy保持collective次数，dummy不参与有效token归一化。固定输入仅在本组内有界缓存（64MiB设备上限），训练context和q始终重新前向。耗时指标、原始数据与验收边界见 [性能记录](specopd_performance_optimization.md)。
