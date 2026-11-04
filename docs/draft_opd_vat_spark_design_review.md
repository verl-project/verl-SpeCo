# Draft-OPD / VAT / SPARK 设计审阅

日期：2026-11-04。基于本地 verl-SpeCo、vllm-main-20260702-dspark、VAT 源码和现有示例的静态核对；没有运行 GPU/NPU 实验，也未核验 SPARK 论文结论。

## 原文三个待确认项

1. **DSpark drafting 默认已可确认：greedy，可配置 probabilistic。**
   `verl_speco/config/speco_base.yaml:165` 默认 greedy；`integration/vllm_runtime.py:1965` 解析并写入 speculative config；fork `vllm/config/speculative.py:269` 同样默认 greedy。`spec_decode/speculator.py:134` 仅在 probabilistic 时分配 draft_logits，DSpark `_sample_sequential` 据此选择 Gumbel 或 argmax（`dspark/speculator.py:124`）。不能从“draft_logits 非空时 probabilistic”反推默认模式。
2. **bonus-anchor 属于范围选择，现有训练端不能直接宣称支持。**
   fork `dspark/speculator.py:45` 已识别 checkpoint 的 dspark_bonus_anchor，但 SpeCo `_build_label_tensors` 固定 `anchor+slot+1`（`dspark_trainer_backend.py:285` 起），本次检索未找到训练端对该标志的处理。建议第一期明确仅支持 anchor-as-first，遇 bonus checkpoint 显式报错；若需要兼容，必须同步改 query 构造、标签、target hidden 对齐、配置校验和导出，不能只增加 SlotLayout。
3. **NPU 是否交付仍需产品范围决定，但不能认定 L1 无需适配即可复用。**
   `examples/dspark_mrv2/run_qwen3-8b_drafter_dspark_vllm_npu.sh:107` 已启用 enable_adaptive_verification，并在 122–124 行训练 Markov-fused confidence head。`dspark_trainer_backend.py:1408` 也有 Ascend 动态验证约束。当前 GPU fork 固定 N 的判断不能推广到 NPU；需要实际 Ascend scheduler、动态有效 draft 数、异步输出协议的契约测试。建议 GPU 为首期验收门槛，NPU L1 单独列为适配验收项，L2 暂不承诺。

## 开发前应修正的高优先级问题

### 1. 接受模拟必须区分 target 与 draft 的采样模式

§5.4 的 `sum min(p,q)` 适用于按 q 采样的 proposal。greedy draft + 随机 target 的逐位置接受率是 `p_target[argmax(q)]`，现有 confidence loss 已正确区分（`dspark_trainer_backend.py:586–610`）。target 也 greedy 时才是 argmax 相等的确定性标签。

建议分别记录 target 温度和 draft_sample_method。VAT 的 argmax 模拟可以保留为训练代理目标，但不能称为真实接受率；DSpark confidence 默认保持现有条件接受率语义，二值 VAT 标签作为显式消融。对随机接受，response teacher-forcing 前缀不必等于 draft 前缀，不能沿用“j≤a 模拟精确”的论断。固定 teacher-forcing 网格上的接受率乘积也只是代理估计，不能无条件称为实际 rollout 的 E[a]。

### 2. L1 的 commit 不足以识别首拒位置

§6.1 用 `a=commit-1` 仅对未截断、包含纠正/bonus token 的正常 verify 步成立。举例：原本接受 5 个 draft，但第 2 个是 EOS，提交 2 个；推导会误把第二个已接受 token 标成首拒。仅有 commit/drafts 也无法确定是否被 stop 截断。

建议 L1 同时保留原始输出长度、截断前 accepted 数、最终 commit、有效 drafts 和 truncated 标志。scheduler 已在 `scheduler.py:1584–1602` 算出 num_accepted，在 1616 行记录截断前 output 长度，可扩展现有统计 hook。truncated 块只监督实际可见位置，不生成虚假的负接受标签/首拒加权；全接受块 `a=N` 没有首拒，所有 mask 与 E2E TV 分母都应按实际有效 draft 数计算，不能直接使用 `a+1`。

### 3. 训练重算分布不自动等于 rollout 的实际分布

全词表 target hidden 能恢复原始 lm_head logits，但不自动恢复 temperature、top-p/top-k、penalty、grammar 等处理，也不保证 actor 更新后仍是原 rollout policy。基础 GPU 示例 temperature=0.6（`run_qwen3-8b_drafter_dspark_vllm.sh:33`），MRv2 示例为 1；不能全局假定 T=1。

DSpark 原有 l1/confidence 用原始 softmax，只有 LK 分支使用 lk_temperature（`dspark_trainer_backend.py:540–565`）。因此直接 `l1/2` 复用 TV 并不能通过设置 lk_temperature 自动对齐 rollout。建议显式区分原始分布蒸馏和 rollout 分布匹配；后者第一期限制支持的 sampling processors，并记录 actor/drafter 版本。sync 仅保证调度顺序，不足以证明训练 student/teacher 就是采集时版本。

### 4. DSpark bonus 与变长验证的校验尚不一致

§6.1 校验只列 N=B，遗漏 bonus 的 B-1；动态验证时还应区分最大预算与当步有效 draft 数，验证 `0≤accepted≤effective_drafts≤capacity`。block 中无有效标签或 a=N 时必须正确屏蔽首拒位。

## 其他风险与接口缺口

- **窗口风险应分算法表述。** DFlash 默认 max_window=512，DSpark 默认 0（无限制，`speco_base.yaml:382`，backend:1689）。DSpark 无窗口时的风险反而是长序列内存。max_anchor_density 可能引入样本选择偏差，需要同时统计覆盖率。packing 本身不能解决轨迹坐标问题，必须保证每个样本的边界与位置映射。
- **多轮轨迹要保留 turn 的 prompt/response 映射。** tool 文本、模板和 tokenizer 拼接不能仅凭接受统计累计器假定对齐；应按实际 token 序列生成区间映射，断言窗口后的 anchor、标签和 response mask 一致。
- **VAT 默认配置矛盾。** 公共 vat.kd_weight=1，但 DSpark 章节说默认不新增 KD，需设置算法默认值或统一含义；vat_reweight_aux 在正文出现但 YAML 缺失。self_cov 必须补上算法与归一化定义；参考 `VAT/specforge/core/vat.py:203–225`，否则 V2 实验尚不可实现/复现。
- **E2E TV 的组合规则不完整。** 明确 loss=e2e_tv 是替代 CE/l1 还是增量项，first_reject_weight 是否作用 TV 链，以及 mixed 中 random block 的监督范围和归一化方式。random_anchor_ratio 也应定义为最终占比还是相对 trace 数的补充比例，并设置固定 block 预算以控制成本。
- **FSDP/发布需端到端验证。** head 放 wrapper 上有利于 HF 过滤，但零权重必须仍执行 head forward；不能被“无 active rows”提前返回绕过。需要验证 optimizer/resume、热发布参数过滤及各 rank collective 顺序。
- **L2 与 SPARK 暂保持独立阶段。** L2 必须按 request 的 cu_num_logits 对齐 draft id、a、bonus 和 padding，处理 raw/processed logprob 的数值差异；采样器是否在 graph 外应以实际 runner 调用链与设备实验确认。SPARK 还缺 group 生命周期、调度顺序、跨 replica 语义和内存上限，论文数值不作为本次审阅已确认事实。

## 建议落实顺序

1. 修订模式语义、L1 schema、全接受/截断边界和 loss 组合规则。
2. GPU、标准 DSpark layout 实现公共组件与 VAT；参考 VAT 数值对齐和发布/resume 验证。
3. L1 先验证同步单轮，再补 async/preemption/多轮/窗口测试，最后接入 Draft-OPD loss。
4. NPU 动态验证、bonus checkpoint、L2 和 SPARK 各自独立验收，避免影响基础路径交付。
