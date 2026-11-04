# verl-SpeCo 接入 Draft-OPD / VAT / SPARK 设计文档

状态：草案 v0.2（2026-11-04）

v0.2 相对 v0.1 的变更：
- Draft-OPD 的 rollout 侧**只使用 vLLM**：基于 fork `vllm-main-20260702-dspark` 的 MRv2（`vllm/v1/worker/gpu`）路径，SGLang 不在本期范围内。
- VAT 和 Draft-OPD 都**同时支持 DFlash 和 DSpark**，增加了两种算法的统一坐标约定（§4.1）以及各自的落地细节（§5.6、§6.5）。

范围：verl-SpeCo 中的 DFlash 与 DSpark drafter（DFlash2 继承 DFlash，基本可以直接复用）。

参考代码：
- Draft-OPD：`Draft-OPD/verl`（verl fork）+ `Draft-OPD/sglang-dflash`，只参考算法，不复用它的推理栈
- VAT：`VAT/specforge/core/vat.py`，`VAT/scripts/train_dflash.py`
- SPARK：arXiv 2608.04962，没有开源代码
- vLLM：`vllm-main-20260702-dspark`（下文简称 `vllm/`）
- verl-SpeCo：`verl_speco/backends/{dflash,dspark}_trainer_backend.py` 等

---

## 1. 背景与目标

三种方法的目标都是**提高 drafter 的接受长度（AL）**：

| 方法 | 核心思想 | 改动位置 | 推理侧是否改动 |
| --- | --- | --- | --- |
| **VAT** | 训练时用 target argmax **模拟**顺序验证，得到首个拒绝位置 `k*`；以 `k*` 为起点重新计算 position decay，再加 KD 项和 accept head（BCE） | 只改训练 loss | 否 |
| **Draft-OPD** | 用 rollout 中**真实**的验证结果做监督：只在真实的 verify 起点放 anchor，并对首个拒绝位置做 on-policy 蒸馏（k3 / E2E TV） | vLLM 导出验证轨迹 + 训练 loss | 否，只做观测 |
| **SPARK** | rollout 期间对 draft logits 做 prompt-group 级别的低秩在线修正 `z + B_g A^T h` | 推理引擎 proposer + 训练 | **是** |

目标：
1. **P0 – VAT**（DFlash → DSpark）：只改训练端。
2. **P1 – Draft-OPD**（DFlash + DSpark，vLLM）：vLLM 侧采集验证轨迹，训练侧按轨迹规划 anchor 并加 on-policy loss。复用 P0 的公共组件。
3. **P2 – SPARK**：视时间而定，先做可行性 spike。

不在本期范围：Draft-OPD 的 "composed student" 训练架构；SGLang rollout 路径。

---

## 2. 参考实现要点

### 2.1 VAT（`VAT/specforge/core/vat.py`）

```
teacher_logits = lm_head(target_last_hidden[label_idx - 1])           # no_grad
token_match    = argmax(draft) == argmax(teacher); match[...,0] = True   # DFlash 第 0 位是 anchor
accept_len     = cumprod(token_match).sum(-1)                         # 含 anchor
w              = weight_mask * exp(-clamp(k - accept_len, 0) / gamma) # k* 之前（含 k*）权重为 1
loss = CE_w + kd_w * KL_T(teacher||draft)·T²_w
     + acc_w * BCE(accept_head(h), k < accept_len)
     + cov_w * self_cov
```

accept head 只用于训练，不进入导出的 draft 权重。默认超参：`gamma=7, kd_w=1, T=2, acc_w=1, self_cov_w=0.6`。

### 2.2 Draft-OPD

- rollout（sglang-dflash）在每轮 verify 时记录：`reject_token_mask`（每轮提交的最后一个 token）、被拒 draft 后缀的 `{anchor, offset, token_id, teacher_logprob}`。
- 训练：
  - anchor 取 `[prompt_len-1] + reject positions`，segment 截断到真实拒绝点；
  - response stream 用 k3 sampled KL；
  - rejected-draft stream 权重 ×2.0，并做 `0.9^(offset-1)` 衰减；
  - E2E TV：`1 - Σ_j Π_{i≤j}(1-TV_i)/n`，README 中效果最好的配置是 TV + first-reject-only。
- first-reject-only 时，**所有被监督位置的上下文都等于 response 前缀**，所以 teacher 分布直接取自真实轨迹上的 target hidden；只有更深的被拒后缀才需要用 branch 重跑主模型。

### 2.3 SPARK

- `B_g ∈ R^{V×r}`：在 prompt group 内共享，每次 rollout 重置为 0；`A ∈ R^{d×r}`：用 lm_head 的若干行做 QR 初始化，并跟随 drafter 一起训练。
- 每轮 verify 之后闭式更新 `B ← B - η Σ w δ u^T`，其中 `δ = (q-p)` 只保留 TopK(p) ∪ TopK(q) 上的分量。block 型 drafter 只用已接受前缀和首个被拒 token 来更新。
- 超参 `r=8, k=16, η=0.01`。

---

## 3. 与 verl-SpeCo / vLLM 的映射

### 3.1 关键观察

1. **SpeCo 中 drafter 的 teacher 就是 target policy 本身。** 在 `dflash_aux_plus_last` layout 下，DFlash 和 DSpark 的 backend 都已经拿到了 `target_last_hidden_states`：
   - DFlash：`DFlashTrainingModel.forward` 中的 `aligned_target_hidden`；
   - DSpark：`DSparkTrainingModel.forward` 中的 `_gather_aligned_target_hidden`，被 l1/LK/confidence 使用。

   ⇒ 真实轨迹上任意位置的**全词表 teacher 分布**都可以在训练时重新算出来。所以 Draft-OPD 的 first-reject-only 方案**只需要 vLLM 导出"每轮提交了几个 token"**。
2. **VAT 和 Draft-OPD 共用一套流程："block 级接受长度 → 位置权重 / 监督位置 / accept 标签"**。两者的区别只有两点：
   - 接受长度的来源：VAT 是模拟的，Draft-OPD 是观测的；
   - anchor 的来源：VAT 随机采样，Draft-OPD 用真实 verify 起点。
3. **vLLM 中验证信息的分布：**
   - **Scheduler（CPU）**：`Scheduler.update_from_output`（`vllm/v1/core/sched/scheduler.py:1556-1622`）里每个 request 每步都能拿到 `num_draft_tokens = len(scheduled_spec_token_ids)`、`num_accepted = len(generated) - num_sampled`，以及经过 stop 截断后真正追加的 token 数（通过 `num_output_tokens_before` 前后的差值算出）。SpeCo 已经 patch 了这个函数（`vllm_runtime.patch_vllm_request_acceptance_stats`），并且有一条现成的通道：请求 finished 时通过 trace header 把数据带回 `RequestOutput`。
   - **draft token id 在 MRv2 下通常不会回传到 scheduler**：`DraftTokensHandler.get_draft_tokens` 只在有 structured output 时才拷回；其余情况 scheduler 拿到的是 `-1` 占位（`vllm/v1/worker/gpu/spec_decode/utils.py:44-51`）。因此**被拒 token id 和 target logprob 只能在 worker 侧获取**，入口是 `RejectionSampler.__call__`，那里同时有 `processed_logits`、`draft_sampled`、`num_sampled`（`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:101-160`）。
   - **DSpark 在 vLLM 中是固定长度 drafting**：confidence head 还没有接入推理（`qwen3_dspark.py:173` 加载时直接跳过它的权重），每轮固定产出 `num_speculative_tokens` 个 draft。
4. **SpeCo 已有的挂钩点：**
   - per-sample key 白名单（`speco_worker.py` 约 `L1006`、`L1414`）；
   - `DraftFeatureSample.metadata` 支持嵌套 tensor；
   - DFlash 的 `_auxiliary_loss` hook；
   - 两个 backend 都有 `per_position_loss_weight`（`fixed_exp_decay | dpace`）；
   - HF 导出只导出 `draft_model.*`（`base_trainer._get_pretrained_export_model`），而热发布（`export_fsdp2_publish_state` → vLLM `translate_name`）会把参数全部带上，需要显式过滤。

### 3.2 命名约定

SpeCo 已有的 `integration/opd_cotrain.py` 指的是对 **actor** 做 OPD，与 Draft-OPD 无关。新功能统一放在 **`drafter_opd`** 命名空间下。Draft-OPD 中 drafter 的 teacher 是 student policy（也就是 target），这符合 `ensure_student_only_feature_payload` 的约束；**禁止**读取 actor OPD 中 teacher 的 `teacher_ids/teacher_logprobs`。

---

## 4. 公共组件（P0 落地，P1 复用）

新增 `verl_speco/backends/verification_aware.py`，纯 torch 实现，**与算法无关**。

### 4.1 统一坐标约定（DFlash / DSpark）

以一轮 verify 为单位：设 `p` 为本轮开始前最后一个已提交 token 的绝对位置（也就是 anchor），drafter 产出 `N` 个 draft，记为 `d_0..d_{N-1}`，其中 `d_j` 预测位置 `p+1+j` 的 token。target 接受 `a` 个（`0 ≤ a ≤ N`），本轮提交 `c = a+1` 个 token（被 EOS 或 max_len 截断时可能更少）。`d_a` 是**首个被拒的 draft**（`a < N` 时存在），对应位置上真正提交的是 target 的纠正 token。

| | DFlash | DSpark（anchor-as-first，默认） | DSpark（`dspark_bonus_anchor=True`） |
| --- | --- | --- | --- |
| 训练 block 长度 `B` | `dflash_block_size`（16） | `dspark_block_size`（7） | `B` |
| vLLM `num_speculative_tokens` `N` | `B-1` | `B` | `B-1` |
| draft 索引 `j` → 训练 block slot | `slot = j+1`（slot 0 是 anchor 本身，不计 loss） | `slot = j` | `slot = j+1` |
| slot 的标签 | `input_ids[anchor + slot]` | `input_ids[anchor + slot + 1]` | 同 DFlash |
| 首拒 slot `s*` | `a+1` | `a` | `a+1` |
| Markov 依赖 | 无 | slot `j` 的 prev 为 `j-1` 位置的 token | 有 |

公共函数一律用**draft 索引 `j`** 表示位置，再由 `SlotLayout(kind, block_size)` 转成各 backend 的 slot。这样 VAT 和 Draft-OPD 的权重、mask、TV 链都只需要写一份。

### 4.2 接口

```python
@dataclass
class SlotLayout:            # dflash | dspark | dspark_bonus_anchor
    kind: str; block_size: int
    def draft_to_slot(self, j) -> int
    def num_drafts(self) -> int

@dataclass
class BlockPlan:
    anchor_positions: Tensor            # [B, N] long，= 4.1 中的 p
    block_keep_mask: Tensor             # [B, N] bool
    observed_accept: Tensor | None      # [B, N] long，= a；None 表示随机 anchor
    commit_len: Tensor | None           # [B, N] long，= c（监督长度，含纠正 token）
    source: Tensor                      # [B, N] int8：0=random，1=trace

def simulate_greedy_accept(draft_top1, teacher_top1, valid, layout) -> Tensor        # VAT：返回 a
def simulate_prob_accept(draft_probs, teacher_probs, valid, layout) -> Tensor        # 采样验证下的期望值
def reanchored_decay(base_mask, accept, gamma, layout) -> Tensor   # w_j = mask * exp(-max(0, j - a)/gamma)
def first_reject_mask(accept, layout) -> Tensor
def trace_supervision_mask(commit_len, layout) -> Tensor           # j < c 的 slot
def chunked_teacher_terms(draft_h, target_h, lm_head_w, kinds, T, chunk)  # argmax / KL_T / TV
def e2e_tv_block_loss(tv, block_index, block_count) -> (loss, expected_al)
class AcceptHead(nn.Module)          # 只用于训练（DFlash）；DSpark 改用 confidence head（§5.6）
```

`reanchored_decay` 按 draft 索引计算：`j ≤ a` 的位置权重为 1，之后按 `exp(-(j-a)/gamma)` 衰减。对 DFlash 而言，这与 VAT 原式 `exp(-(k-accept_len)/gamma)` 完全等价，因为 `k = j+1`、`accept_len = a+1`。

### 4.3 Anchor Planner

从两个 backend 的 `_sample_anchor_positions` 中抽出 planner：

| planner | 行为 |
| --- | --- |
| `random`（默认，即现有行为） | 按 loss_mask 随机采样 |
| `trace` | anchor 取轨迹中所有 `draft_len > 0` 的轮次的 `p` |
| `mixed` | trace anchor，再按 `random_anchor_ratio` 补充随机 anchor，并去重 |

DSpark 的 `_sample_anchor_positions` 还要考虑 `next_token_loss_mask`，所以 planner 接口需要额外接收这个参数。没有轨迹、或者轨迹校验失败时，退回到 `random`，并通过 `drafter_opd/trace_fallback_ratio` 记录比例。

### 4.4 全词表 teacher 项

复用 LK 已有的 chunk + checkpoint 实现：DFlash 的 `_lk_chunk_terms`，DSpark 的 l1/LK chunk 循环。在同一次循环中算出 teacher argmax（VAT 用）、`KL_T`（VAT KD 用）和 TV（Draft-OPD 用；DSpark 的 `l1_dist = 2·TV` 已经是现成的）。

VAT 需要整块网格上的 argmax 来做 cumprod：先在 active rows 上算，再 scatter 回 `[B, N, slots]`，非 active 位置视为 mismatch，链路在该处中断。

### 4.5 训练专用参数的发布过滤

- `export_fsdp2_publish_state` 增加由 backend 提供的 `is_training_only(name)` 判断。DFlash 的 `accept_head.*` 必须被过滤掉，否则 vLLM `load_weights` 遇到未知参数名会报错。
- DFlash 的 accept head 挂在训练 wrapper 上（不挂在 `draft_model` 上），这样 HF 导出会自动排除它，resume 时由全量 state dict 恢复。
- FSDP 一致性：和 DFlash2 selector 的处理方式一样，对无效位置**用零权重而不是切片**，保证每个 rank 每一步都给 head 产生梯度。

---

## 5. P0：VAT

### 5.1 配置

```yaml
# rollout.drafter.training
dflash_per_position_loss_weight: vat     # 新增枚举值；DSpark 对应 dspark_per_position_loss_weight
vat:
  kd_weight: 1.0
  kd_temperature: 2.0
  accept_head_weight: 1.0          # DFlash：独立的 accept head；DSpark：confidence head 的 VAT 标签权重
  self_cov_weight: 0.0             # 论文建议取 0.4~0.6
  self_cov_floor: 1.0e-6
  accept_mode: greedy              # greedy | probabilistic（§5.4）
```

校验规则（fail-closed）：
- hidden layout 必须是 `dflash_aux_plus_last`；
- `loss_mode` 必须是 `full_vocab`；
- `gamma > 0`；
- `vat` 与 `dpace` 互斥。

### 5.2 DFlash 前向改动（`DFlashTrainingModel.forward`）

1. 设置了 `vat` 时，跳过 `_cached_decay_weights`；
2. 用 `chunked_teacher_terms(kinds={argmax, kl_T})` 得到 teacher argmax，再用 `simulate_greedy_accept` 算出 `a`；
3. 计算 `w = reanchored_decay(base_loss_mask, a)`，然后更新 `active_weights`；
4. `ce = Σw·CE/Σw`，`kd = Σw·KL_T·T²/Σw`；
5. 在 `_auxiliary_loss` 中加入 accept head 的 BCE：标签为 `j ≤ a-1`（即"该 draft 被接受"），权重取 `base_loss_mask`；
6. 指标：`vat/{ce,kd,accept,self_cov}_loss` 和 `vat/mean_accepted`。`mean_accepted` 可以和 rollout 中的 `_verl_request_mean_accept_len` 做对照。

### 5.3 开销

已经开启 LK 时，额外只多一次 argmax。未开启时，需要额外做一次 chunked 的 `[rows, V]` teacher logits 计算（1024 行一个 chunk，fp32 约 0.6 GB）。

### 5.4 RL 采样场景（`accept_mode=probabilistic`）

RL rollout 通常是 `T=1` 采样，verify 走的是 rejection sampling。此时：
- 逐位置接受率 `α_j = Σ min(p,q)`（复用 LK 中 acceptance 的实现），期望 `E[a] = Σ_j Π_{i≤j} α_i`；
- 权重的重锚定点使用 `E[a]`，accept 标签改为软标签 `Π_{i≤j} α_i`。

这一部分在 P0 主体之后做小实验，不阻塞 P0。

### 5.5 测试

- 与 `OnlineVATDFlashModel` 做数值对齐（loss、各项分量、梯度，误差 < 1e-5，CPU + SDPA）；
- accept head 不出现在 HF 导出和 publish 结果中，resume 后保持一致；
- FSDP2 下某个 rank 没有 active row 时不会 hang。

### 5.6 DSpark 适配

DSpark 与 DFlash 的差异，以及 VAT 在 DSpark 上的处理：

| 方面 | DSpark 现状（`dspark_trainer_backend.py`） | VAT 改法 |
| --- | --- | --- |
| slot 布局 | slot `j` 的标签是 anchor+j+1，所有 slot 都计 loss；`eval_mask` 做了 cumprod（`L327-330`） | 使用 `SlotLayout(dspark)`，`a` 的取值范围为 `0..B` |
| draft 预测 | `base_logits + markov_bias(prev)`，训练时 prev 用 teacher forcing（`_build_label_tensors` 中的 `prev_token_ids`） | 模拟接受时，draft argmax **必须包含 markov bias**。对 `j ≤ a` 的位置，teacher forcing 下的 prev 就是已被接受的 draft，与推理时完全一致，所以这一段的模拟是精确的 |
| 位置权重 | 固定的 `exp(-j/gamma)`（`L848-854`），对 CE/l1/LK/confidence 共用 | 换成 `reanchored_decay`，作用于所有分项。注释中提到的 "l1 保持固定衰减" 这一约束改为可配置：`vat_reweight_aux: true` |
| 分布项 | 已有 `ce_loss_alpha=0.1` + `l1_loss_alpha=0.9`（`l1 = 2TV`） | 默认**不额外新增 KD**，而是让 VAT 的权重作用在已有的 l1 上；`kd_weight>0` 时再加 `KL_T` |
| accept head | 已有 `confidence_head`，学习的是**逐位置条件接受率**（greedy 时为 `p_target[argmax q]`，`L586-608`） | **不新建 head**，改为对 confidence head 施加 VAT 式监督：标签为 `match_j`（greedy）或 `α_j`，**只对 `j ≤ a` 的位置计 loss**，因为首拒之后的条件不会在推理中出现。这样它与 VAT 的"生存概率"是同一个分解：`P(survive j) = Π_{i≤j} conf_i`，而且保持了 confidence head 在推理端的含义，便于以后接入 confidence-scheduled verification |
| 开关 | `dspark_confidence_head_alpha / loss_alpha` | `vat.accept_head_weight` 映射为 confidence loss 的权重；若 checkpoint 中没有 confidence head，按现有逻辑报错 |

DSpark 的 VAT 单测：构造只有 markov bias 能改变 argmax 的样例，验证模拟出的 `a` 考虑了 bias；同时覆盖 `a=0`（首个 draft 就被拒）和 `a=B`（整块全部接受）这两个边界。

---

## 6. P1：Draft-OPD（vLLM）

### 6.1 vLLM 侧：验证轨迹采集

分两层，第一层是必须的，第二层可选：

**L1：计数轨迹（必需，纯 scheduler patch，MRv1/MRv2 以及 vllm-ascend 通用）**

每个 request 记录每一步实际追加 token 的情况：

```python
_speco_verify_trace = {
  "commit": list[int],   # 本步真正追加进 output 的 token 数（已经过 stop 截断）
  "drafts": list[int],   # 本步被验证的 draft 数；0 表示 prefill/重算/非投机步
  "drafter_version": int,
  "layout": "dflash" | "dspark" | "dspark_bonus_anchor",
  "num_spec_tokens": int,
}
```

实现上扩展 SpeCo 现有的 `patched_update_from_output`：
- 调用原函数之前，对 `scheduler_output.num_scheduled_tokens` 中的每个 request 做一次快照：`(request 对象, len(request._output_token_ids), len(scheduled_spec_decode_tokens.get(rid, ())))`。保留 request 对象本身，因为 finished 的请求可能已经从 `self.requests` 里删掉了。
- 调用原函数之后，用 output 长度的差值作为 `commit`；`drafts` 直接取快照值。
- 有 `num_invalid_spec_tokens`（grammar 截断）时，`drafts` 要减去无效的部分。
- request 结束时，沿用 `_attach_vllm_request_stats_to_trace_headers` 的通道，把轨迹带回 `RequestOutput`，再写入 agent loop 的 `extra_fields`。

推导：`p_t = prompt_len + Σ_{s<t} commit_s - 1`，并且只对 `drafts_t > 0` 的步生成 block，`a_t = commit_t - 1`（如果因 stop 截断导致 `commit < a+1`，就把该步标记为 truncated，只监督 `j < commit` 的位置）。

这一层能覆盖 **VAT 组合方案、Draft-OPD 的 (a)(b)(c) 三项 loss**（§6.4），两种算法都适用。

**L2：被拒 draft 详情（可选，需要改 fork 的 worker 侧，仅 MRv2）**

在 `RejectionSampler.__call__` 中、开关打开时：
- 从 `draft_sampled`（即 `input_ids[logits_indices]`）和 `num_sampled` 中取出每个 request 的被拒后缀 token id `d_a..d_{N-1}`；
- 计算 `log_softmax(processed_logits)[pos, d_j]`，得到 target 在**draft 前缀条件下**对这些 token 的 logprob。这与 Draft-OPD 的 `rejected_draft_teacher_logprobs` 语义一致，温度和 top-p 都已经包含在 processed logits 中。
- 把结果作为新字段加到 `SamplerOutput`，经 `AsyncOutput` 异步做 D2H 拷贝，写入 `ModelRunnerOutput.spec_verify_detail`（新增，按 `req_ids` 对齐）。scheduler 在 `update_from_output` 中把它追加到 `_speco_verify_trace`。

需要改动的 fork 文件：
- `v1/worker/gpu/spec_decode/rejection_sampler.py`
- `v1/worker/gpu/sample/output.py`
- `v1/worker/gpu/async_utils.py`（AsyncOutput）
- `v1/outputs.py`（ModelRunnerOutput）
- `v1/core/sched/scheduler.py`

开关挂在 `speculative_config` 新增的 `capture_verify_detail: bool` 上，默认关闭；关闭时零开销，所有新逻辑都放在 `if` 分支内。为了 CUDA graph 安全，这段逻辑放在 rejection sampling 之后（实现前需要确认 MRv2 的 sampler 不在 CUDA graph 捕获范围内）。

建议 L2 **直接改 fork**，而不是走 SpeCo 的 monkeypatch：worker 侧的数据结构跨越了多个进程边界，用 monkeypatch 太脆弱。L1 维持现有的 monkeypatch 方式，这样不依赖 fork 也能运行。

**校验**（在 agent loop 收集端执行，失败时退回 random）：
- `Σ commit == response_len`；
- 所有 `drafts_t ≤ num_spec_tokens`；
- `layout` 和 `num_spec_tokens` 与训练配置一致：DFlash 要求 `N = B-1`，DSpark 要求 `N = B`。

**多轮 / partial rollout**：每轮 tool 调用都是独立的 request，轨迹按 turn 拼接，turn 之间插入的 tool token 记作 `drafts=0` 的步（`commit` = tool token 数），这样 cumsum 仍然对齐。partial rollout 续跑时同样做拼接。复用 `acceptance_metrics.bridge_request_acceptance_counters` 的跨 turn 累计位置。

**NPU**：L1 属于 scheduler 层，vllm-ascend 可以直接复用；L2 需要另外适配 vllm-ascend 的 rejection sampler，不在本期范围。

### 6.2 样本透传

- 在 `speco_worker.py` 的两处白名单和 `metadata` 中加入 `_speco_verify_trace`；同步修改 `trainer/v1/feature_adapter.py`、TQ 和 feature store。
- **窗口裁剪**：DFlash 的 `preprocess_individual_items`（`dflash_max_window=512`）和 DSpark 对应的逻辑，要把轨迹坐标减去 `feature_start + start`，并丢弃窗口外的 block。新增 `drafter_opd.window_policy: first_response | max_anchor_density`，默认 `first_response`，与现状一致。
- DSpark 的 `next_token_ids` 已经和 context row 对齐，trace 中的 `p` 直接对应 context row 的索引。

### 6.3 训练侧：TraceAnchorPlanner

```
for 每一步 t（drafts_t > 0）:
    anchor   = p_t
    accept   = a_t = commit_t - 1
    监督 slot = layout.draft_to_slot(j)，j ∈ [0, commit_t)   # 已接受的 draft + 首拒位（纠正 token）
```

- 首拒位 `j = a` 的标签就是 response 中实际提交的纠正 token，它的上下文是 response 前缀，所以 target hidden 是**精确**的。
- DSpark 在 `j ≤ a` 范围内，teacher forcing 下的 prev token 等于推理时的 prev（都是已接受的 token），因此 student 端也是精确复现。

### 6.4 Loss（DFlash 和 DSpark 通用）

| 项 | 内容 | teacher 来源 | 轨迹层级 | 默认 |
| --- | --- | --- | --- | --- |
| (a) on-policy CE/分布 | 在 `j < commit` 范围内计算 CE（DSpark 再加已有的 l1），权重为 `reanchored_decay(·, a_obs)` 或硬截断 | target_last_hidden | L1 | 开 |
| (b) 首拒加权 | `j = a` 的位置乘以 `first_reject_weight`（对应 Draft-OPD 的 2.0） | 同上 | L1 | 2.0 |
| (c) **E2E TV first-reject-only** | 每个 block 取 `j=0..a`，计算 `1 - Σ_j Π(1-TV_i)/(a+1)` | 同上，全词表精确值 | L1 | `loss: e2e_tv` |
| (d) 首拒 token sampled-k3 | `k3(log q(d_a), log p(d_a))` | `lm_head(target_h)[d_a]`，或取 L2 的值 | L2 | 0 |
| (e) 深层被拒后缀 sampled-k3 | `j > a` 的 `d_j`，权重 `decay^(j-a)` | **只能来自 L2**（target 在 draft 前缀条件下的 logprob） | L2 | 0 |

几点说明：
- (c) 在 DSpark 上直接复用 l1 chunk 的 TV（`tv = l1/2`）。DFlash 则在 `chunked_teacher_terms` 中新增 TV 计算。
- (e) 在两种算法上 **student 侧都可以精确复现**：
  - DFlash 的 slot 之间是并行的，`j > a` 的 slot 不依赖前面的 draft token；
  - DSpark 只要在 `j > a` 处把 prev 换成 L2 记录的被拒 token（`d_{j-1}`）做 teacher forcing 即可。

  这一项对应 Draft-OPD 的完整 rejected stream，作为 P1.3 的消融项。
- 有全词表 teacher 时，(a)(c) 已经覆盖了首拒位置的概率质量，所以 (d) 只作为消融。

配置：

```yaml
drafter_opd:
  enable: false
  capture_verify_trace: false        # L1
  capture_verify_detail: false       # L2，需要 fork 支持
  anchor_planner: mixed              # random | trace | mixed
  random_anchor_ratio: 0.25
  loss: e2e_tv                       # ce | e2e_tv
  first_reject_weight: 2.0
  position_decay: reanchored         # reanchored | hard_truncate | none
  rejected_k3_weight: 0.0            # (d)
  rejected_suffix_k3_weight: 0.0     # (e)
  rejected_suffix_decay: 0.9
  max_trace_staleness: 2             # drafter 版本落后超过这个值时，该样本退回 random
  window_policy: first_response
```

### 6.5 DFlash / DSpark 差异汇总

| 项 | DFlash | DSpark |
| --- | --- | --- |
| `N` 与 `B` 的关系 | `N = B-1` | `N = B`（bonus-anchor 格式为 `B-1`） |
| block 中 slot 0 | anchor，不计 loss | 预测 `d_0`，计 loss |
| `a = N`（整块全接受） | 监督 `j=0..N-1`，外加 bonus 位置（超出 block 的部分不监督） | 同左 |
| (c) 的 TV | 在 `chunked_teacher_terms` 中新增 | 复用 l1 |
| (e) 的 student 复现 | 直接复现（并行 slot） | 需要把 prev 换成被拒 token |
| accept/confidence head 标签 | wrapper 上的 AcceptHead，标签取观测到的 `j < a` | confidence head，标签取观测到的 `match_j`，只用 `j ≤ a` |
| vLLM 实现 | `v1/worker/gpu/spec_decode/dflash/speculator.py` | `.../dspark/speculator.py`（继承 DFlash，Markov 顺序采样） |

### 6.6 on-policy 程度与温度

- sync 模式下，轨迹严格 on-policy。bubble/async 模式下存在 drafter 版本滞后，用 `max_trace_staleness` 控制，并按 `drafter_version` 设定 replay 的淘汰优先级（复用 `data_status_policy`）。
- RL 采样时 `T=1`，拒绝点带有随机性，但都是真实访问过的状态。TV 计算使用的温度 `lk_temperature` 必须和 `rollout.temperature` 一致（沿用现有约定）。
- 现有的 `dflash_hard_sample_ratio` 和 `dspark_hard_sample_ratio` 依据 `_verl_request_mean_accept_len` 挑选难样本，可以和轨迹配合使用。

### 6.7 测试

- 坐标单测：轨迹 → BlockPlan，覆盖 DFlash、DSpark、bonus-anchor 三种 layout，以及 prefill 首 token、stop 截断、grammar 无效 token、preemption 后的重算步、多轮 tool、窗口裁剪；
- 与 Draft-OPD 对齐：把 `_build_opd_anchor_plan` 和 `_compute_e2e_tv_block_outputs` 的用例（`tests/workers/test_dflash_opd_on_cpu.py`）移植到 DFlash layout 上，做数值对比；
- vLLM L1：用 fake 的 `SchedulerOutput`/`ModelRunnerOutput` 驱动 `patched_update_from_output`，断言 `Σcommit == len(output)`；
- vLLM L2（fork）：在 `tests/v1/spec_decode` 下新增用例。以 greedy 和小模型构造确定性样例，检查被拒 token id 与 `draft_tokens` 一致，并且 logprob 与离线 target forward 的误差在 1e-3 以内；开关关闭时输出与原版逐位一致；
- GPU e2e：分别在 Qwen3-8B + DFlash 和 Qwen3-8B + DSpark（`examples/dspark_mrv2`）上比较 random / trace / mixed。

---

## 7. VAT × Draft-OPD 组合

两者共用 `BlockPlan` 和 `SlotLayout`：
- trace block 使用观测值 `a_obs`，random block 使用模拟值 `a_sim`；
- accept/confidence head 在 trace block 上使用**真实**接受标签；
- 在 trace block 上额外计算 `a_sim` 并输出 `vat/sim_vs_obs_accept_gap`，用来量化 VAT 的模拟偏差（尤其是 RL 采样场景下）。

实验矩阵（DFlash、DSpark 各做一组；成对比较，沿用 `docs/experiments/opd-cotrain` 的审计规范）：

| # | anchor | 权重 / loss | 目的 |
| --- | --- | --- | --- |
| B0 | random | 现状（DFlash：CE；DSpark：CE + l1） | 基线 |
| V1 | random | VAT | P0 |
| V2 | random | VAT + self_cov 0.6 | P0 |
| O1 | mixed | e2e_tv first-reject | P1 |
| O2 | mixed | (a) + first_reject_weight | P1 消融 |
| O3 | mixed | O1 + (e) suffix-k3 | P1.3，需要 L2 |
| VO | mixed | VAT 权重（trace 用观测值）+ e2e_tv | 组合 |

主要指标：rollout 真实 AL（`drafter/spec_decode/*`）、生成时间、单步总时间、drafter 训练耗时。

---

## 8. P2：SPARK 可行性（视时间而定）

在 vLLM MRv2 中需要做的事：

| 子任务 | 位置 | 难点 |
| --- | --- | --- |
| logits 修正 `z + B_g(A^T h)` | DFlash：`dflash/speculator.py` 的采样处；DSpark：`_sample_sequential` 中 `logits_i = base_logits[:, i] + bias` 这一行（`dspark/speculator.py:123`），天然可以在这里再加一项 | `B_g` 需要按 request→group 做 gather，并且要进入 CUDA graph（固定 buffer：`[max_groups, V, r]`，fp32 下 r=8 时每组约 4.8 MB）；TP 切分 |
| verify 后更新 `B_g` | `RejectionSampler.__call__` 之后（复用 L2 的位置） | 需要 target 的 TopK(p)，而 processed_logits 已经就在这里 |
| group 标识 | verl agent loop 的 `uid` → `SamplingParams.extra_args` | DP 多个 replica 之间无法共享，只能在 replica 内共享 |
| `A` 的梯度回传 | 在 worker 中累积 `h (B^T δ)^T`，rollout 结束后取回，并入 drafter 的 optimizer step | 是一条新的跨进程数据通道 |
| NPU | vllm-ascend | 另行适配 |

建议先做 **SPARK-lite spike**：只支持 DSpark/DFlash、GPU、TP=1、eager 模式；`A` 用 QR 初始化后冻结（论文 Table 2 中 frozen A 的 AL 降到 2.88，低于完整版；相对无 SPARK 的基线是否还有收益，需要对照原表确认）。先看 RL 场景下 AL 有没有提升，再决定是否打通 `A` 的训练通道和 CUDA graph。SPARK 与 VAT/Draft-OPD 正交，可以叠加；它复用 L2 的 verify hook 和 uid 透传。

---

## 9. 改动清单

| 模块 | 文件 | VAT | Draft-OPD | SPARK |
| --- | --- | :-: | :-: | :-: |
| 公共组件 | `backends/verification_aware.py`（新增） | ✔ | ✔ | |
| DFlash loss | `backends/dflash_trainer_backend.py` | ✔ | ✔ | |
| DSpark loss | `backends/dspark_trainer_backend.py`（权重、confidence 标签、trace planner、suffix 的 prev 替换） | ✔ | ✔ | |
| 配置与校验 | `config/speco_base.yaml` | ✔ | ✔ | ✔ |
| 发布过滤 | `trainer/publish_state.py`，`integration/vllm_runtime.py` 中的 `translate_name` | ✔ | | ✔ |
| L1 轨迹 | `integration/vllm_runtime.py`（`patch_vllm_request_acceptance_stats` / `patched_update_from_output`） | | ✔ | |
| L2 轨迹（fork） | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`、`sample/output.py`、`async_utils.py`、`v1/outputs.py`、`v1/core/sched/scheduler.py`、`config/speculative.py` | | P1.3 | ✔ |
| agent loop 收集 | `integration/agent_loop_runtime.py`、`trainer/speco_ray_trainer.py` 的 key 列表 | | ✔ | ✔ |
| 样本透传 | `workers/speco_worker.py`、`trainer/v1/feature_adapter.py`、feature store/TQ | | ✔ | |
| 窗口裁剪 | 两个 backend 的 `preprocess_individual_items` | | ✔ | |
| proposer 修正（fork） | `dflash/speculator.py`、`dspark/speculator.py` | | | ✔ |
| 测试 | `tests/unit/test_vat_{dflash,dspark}.py`、`test_drafter_opd_{trace,loss}.py`；fork 中的 `tests/v1/spec_decode/test_verify_detail.py` | ✔ | ✔ | |
| 示例 | `examples/run_qwen3-8b-vat_drafter_{dflash,dspark}_vllm.sh`、`..._drafter_opd_...` | ✔ | ✔ | |

实施顺序（S/M/L 表示相对工作量）：

1. [M] `verification_aware.py`（含 `SlotLayout`）+ DFlash VAT + 与参考实现的数值对齐；
2. [S] accept head 的发布过滤、resume、FSDP2 冒烟测试；
3. [M] DSpark VAT（markov 感知的模拟、confidence head 的 VAT 标签、重锚定权重）；
4. [S] VAT 的 GPU e2e（DFlash 和 DSpark 各一组 B0/V1/V2）；
5. [M] vLLM L1 轨迹采集、透传、校验（两种 layout）；
6. [M] TraceAnchorPlanner + E2E TV + 窗口坐标映射，DFlash/DSpark 同时实现；
7. [S] Draft-OPD e2e（O1/O2/VO）；
8. [M] fork L2 + (d)(e) 两项 loss（P1.3）；
9. [L] SPARK-lite spike（视时间而定）。

---

## 10. 风险与待确认

1. **VAT 模拟与真实接受的偏差**（RL 采样场景）：用 §5.4 的 probabilistic 模式，配合 §7 的 gap 监控。
2. **L1 轨迹的完整性**：async scheduling、preemption、grammar 截断都可能导致轨迹不完整。校验失败时退回 random，并对 `trace_fallback_ratio` 设置告警。
3. **窗口裁剪**：`max_window=512` 远小于 RL 的 response 长度 → 使用 `max_anchor_density` 策略，或增大窗口、开启 packing。
4. **DSpark 推理端没有 confidence 调度**：目前 `N` 固定。如果以后 fork 接入了 confidence-scheduled verification，`drafts_t` 会变成变长；L1 已经按步记录 `drafts_t`，这种情况不受影响。
5. **fork 维护成本**：L2 和 SPARK 都需要修改 fork 中的 MRv2 文件，rebase 到上游时容易冲突。因此把改动收敛在 `speculative_config.capture_verify_detail` 开关之后，并保持关闭时逐位等价。
6. **与 actor OPD 同时开启**：`validate_sync_opd_cotrain_config` 的约束仍然有效；需要用单测确保 drafter_opd 不会读到 teacher 张量。

待确认：
- DSpark 推理时是 greedy drafting 还是 probabilistic drafting？这决定了 VAT 和 confidence 标签默认用 `greedy` 还是 `probabilistic`。目前 vLLM 中 `draft_logits` 非空时走 probabilistic。
- 是否需要 bonus-anchor（speculators 格式）的 DSpark checkpoint？如果不需要，`SlotLayout` 只实现两种即可。
- P1 是否需要覆盖 NPU？只用 L1 的话可以复用；L2 需要额外排期。
