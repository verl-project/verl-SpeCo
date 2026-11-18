# AIRDrafter 接入 verl-SpeCo 进展

更新日期：2026-11-12。已恢复其余训练适配与验证；跨运行时特征对齐暂搁置。

后续授权补充：已完成独立的 [actor 与 vLLM-Ascend 对齐诊断](deepseek_v4_actor_native_alignment.md)；
默认配置下辅助 HS 存在约 10%–14% 的相对 L2 差异，
两条短输入的平均预测分布 TV 分别为 11.57% 和 7.35%。参数精度控制后仍有差异，不能默认视为等价输入。

目标：将 AIRDrafter 的训练能力迁入 SpeCo，以 `backend=vllm_ascend` 支持
DeepSeek V4 Flash 0731 DSpark，复用 SpeCo 的数据、调度与权重发布接口，保留原有 PyTorch 功能。

**当前结论：核心训练及 actor 特征接口已接通，完整闭环尚未验收，不能视为稳定生产可用。**

## 已完成

| 内容 | 当前进展 |
| --- | --- |
| 代码迁入与后端选择 | 实现放入 SpeCo 现有 `models/`、`ops/`、`backends/`、`trainer/` 等目录，不再依赖外部 AIRDrafter checkout；通过 backend 参数区分训练实现。 |
| 核心训练 | 接入 DSpark 前向、反向、优化器及 TP/DP 执行；TP8/DP2 实机训练冒烟已跑通。 |
| 数据与 mask | 统一采用 SpeCo 的 `1=监督、0=忽略` 规则，显式保留辅助特征和 pre-HC teacher 特征。 |
| 特征获取 | 保留离线轨迹重放；接通在线 actor old-logprob HS 获取。按要求撤回 rollout 直接采集，不恢复该路径。 |
| 真实 actor 验证 | 完整 284.33B 目标模型在 16 卡 FSDP2 上完成短输入验证；HS、HC 特征及 logprob 与同一 HF actor 内部的独立 hook/无采集前向对照最大误差均为 0，并导出训练传输数据。这不是与 vLLM-Ascend 的对照。 |
| actor → Drafter 训练端权重同步 | 接通 actor 输出层及共享 token 权重的同步接口，用于刷新训练 teacher 和 Drafter 的冻结共享层；含义和验证范围见下节。 |
| 权重接口 | 实现标准 PyTorch/safetensors 导出、原生冷加载及运行时发布接口；本轮标准 PyTorch 导出与训练 TP 布局的冷加载数值验收通过，真实 HTTP/Ray server 的 TP8/DP2/EP IPC 热更新探针退出 0。 |
| 版本适配 | 保留旧版运行时分支，补充 vLLM/vLLM-Ascend 0.27.1 适配。当前实机证据主要来自 0.27.1 环境。 |

## actor 权重同步的具体含义

同步方向是 **当前 actor → Drafter 训练端**，只传递以下权重：

| 权重 | 在 Drafter 训练端的用途 |
| --- | --- |
| 最终 HC head 的 `hc_head_fn`、`hc_head_base`、`hc_head_scale` | 合并多路 pre-HC 特征，作为 teacher 输出计算的第一步；不是同步所有 decoder 层的 HC 参数。 |
| 最终 norm 的 `norm_weight` | 对合并后的 teacher 特征做归一化。 |
| `lm_head.weight` | 将 teacher 特征转换为词表预测分数，并刷新 Drafter 冻结的输出 LM-head 分片。 |
| `embed_tokens.weight` | 刷新 Drafter 冻结的输入 embedding 分片。 |

teacher 是由这些输出层组成的计算模块：使用采集到的 actor pre-HC 特征，计算
监督 Drafter 的预测分布。它不是另一份完整 actor。Drafter 的共享 embedding 和
LM-head 不参与其优化器更新，只从对应 actor 版本刷新。

同一次训练使用的特征、teacher 权重和共享 token 权重需要属于同一 actor 版本，
避免混用新特征与旧权重。接收接口先暂存带版本的权重，实际应用在训练激活后、
执行训练前完成；训练端检查 target 与特征版本。

**这与 Drafter → rollout 的热更新是两个方向：** 前者为训练提供当前 actor 的
监督与共享权重；后者把训练好的 Drafter 参数发布给推理端。

当前已验证真实完整 actor 的六项权重导出，以及生产接收接口的版本/字段契约。
逐项应用有单元及 NPU 分项测试；真实 actor 数据驱动的连续训练与发布整链仍未验收，
不能把“同步接口已接通”理解为在线 RL 闭环已完成。

## 已定位并处理的问题

- **metadata 初始化顺序：** 最小测试复现了首次 HCCL 通信后 metadata 调用失败；增加一次性通信预热后，完整模型重复前向不再出现该错误。
- **TP 副本分叉：** 普通通信配置下，初始参数和输入一致，所查梯度及更新参数仍在 rank 间分叉。启用确定性通信后，14 组 TP 对照中的所查参数均逐字节一致，完整模型重复前向也完全一致。
- **导出检查遗漏：** 补齐复制参数检查，拒绝将存在副本差异的 checkpoint 静默合并为合格权重。此前生成文件和读回成功，不能替代完整 TP 等价验证。

确定性通信设置已纳入恢复身份记录；在线只向 Ascend Drafter worker 注入。
本轮没有替换注意力或 MoE 前向公式，也没有开启 PyTorch 全局确定性。

## 尚未完成

actor 特征与实际 vLLM-Ascend 推理特征的数值相容性尚未通过验收；独立诊断已完成，剩余差异尚未修复。
此前内部采集对照误差为 0，不能替代这项跨运行时检查。

1. **离线后续验证：** 固定短数据下的“保存 → 恢复续训 → 严格 PyTorch 导出 → 冷加载”已通过。16 rank 的 step-1 恢复与 step-2 续训完整状态逐字节一致；全部参数/cache/layout 冷加载一致，HS 和完整词表 logits 差值为 0。长数据、多步与性能尚未验收。
2. **推理端后续验证：** 新 checkpoint 的 TP 复制参数检查通过；真实 verl HTTP server 的 TP8/DP2/EP、IPC 热更新和版本提交探针已正常退出，检查更新前后目标贪心 token 一致。多轮自动发布和多 target 版本仍待验收。
3. **在线整链验收：** 真实 actor 数据驱动连续训练、自动回传权重、接收端加载及后续推理尚未端到端验收；相关两轮诊断辅助代码仍未完成验证。
4. **真实 RL 与性能：** 尚未完成完整 RL 任务、长时间运行、吞吐和资源占用评估。短输入验证不代表训练质量或长期稳定性。

## 记录位置与恢复工作顺序

服务器记录集中在 `/data/zzc/speco_validation_archive/`：

- `actor_fsdp_retry_20261111/results/`：完整 actor 特征及 target 导出报告。
- `offline_recovery_20261111/`：恢复、导出、通信及梯度对照；最新整链状态为 `strict_validation_status.json`。
- `offline_completion_20261112/`：显式确定性通信配置下的新离线整链；`status.json` 记录各阶段命令与退出状态。启动环境及后端隔离回归为 **124 passed**，见 `launch_policy_tests.log`。
- 同根目录的 `server_update_status.json`：真实服务热更新退出 0，运行时间 14:36:49–14:42:44；完整服务报告保存在 `server_update/server_update_report.json`。

已完成本轮全部参数、恢复、标准 PyTorch 格式和一次真实权重发布验证；使用既有原生重放特征，不把 actor 特征当成已对齐输入。真实 actor 驱动的数值整链验收保留为待办。
详细过程见 [离线恢复记录](deepseek_v4_offline_recovery.md) 和 [actor 特征说明](deepseek_v4_actor_features.md)。

2026-11-13 已重新核对服务器实际报告，并同步两端记录。核对范围与验证脚本合并说明见
[文件一致性记录](source_consistency_20261113.md)。
