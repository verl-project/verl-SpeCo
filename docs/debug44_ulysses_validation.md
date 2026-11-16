# SpecOPD Ulysses 接入与验收

2026-11-16，用户要求适配 Ulysses，并以 `.242` 指定的
`/efs_rl/z00886395/shell/speco-runping/run_qwen3-8b_drafter_dspark_drafter_with_train.sh`
（`train_sp=4`）作为配置参照。此前 V1 sync 短测通过只覆盖 SpecOPD SP=1。

## 阶段

- [x] SP-00：确认 upstream remove-padding/SP forward 与通信接口；给出作用域隔离的联合评分实现。
- [x] SP-01：CPU 多进程 SP=2/4 与完整树前向对照，覆盖非整除长度、GQA、逻辑 RoPE、不同分支位置、hidden/logprob 与隔离；该完整树参考实现已有独立因果 oracle。
- [x] SP-02：真实 NPU 小模型/通信检查，SP=4、DP=2/SP=2、leader 和 dummy 组无重复/遗漏。
- [x] SP-03：`.44` V1 sync + actor/ref SP=4、10 个 RL step 短测，覆盖第 5/10 轮，完成正常 actor/ref 与草稿更新、评分/发布 ACK。
- [ ] SP-04：参数、方案和 `.242` 代码/脚本同步；短测范围和生产长序列范围分别说明。

## SP-00 进行中

- `.44` 实际 verl FSDP engine 明确要求 SP>1 时 `use_remove_padding=true`；当前联合评分既拒绝 remove-padding，也拒绝 SP。
- 不能删校验后把树输入交给原 causal FlashAttention。需保留普通 PPO 的 upstream SP 路径，并在联合评分调用中使用支持树 mask 的真实 Ulysses QKV 通信。
- 未启动新的 NPU 训练。

## SP-00/01 初步通过

- 实现独立作用域的 Ulysses QKV all-to-all 与逆向交换；按 SP 组分配 DP 轨迹，只有组 leader 回传一份结果。普通 PPO/ref 仍使用 upstream unpadding + FlashAttention，不修改其全局 attention 实现。
- CPU Gloo 的 SP=2/4、GQA（KV heads 少于 SP）、Qwen3/Qwen2/Llama、非整除长度、分支/future-trunk 隔离、logprob/entropy/hidden/readout 对照，4 项多进程测试通过。其他相关回归 87 passed / 2 skipped。
- 用户明确调整调度：**每 5 个 RL step 采集一次，固定该组数据训练 20 个 optimizer step，结束后统一发布一次**。这替代此前单次 optimizer 更新的保守约束；20 步间复用采样分支属于近似的多步训练，不声称每步都有重新从当前 q 采样。
- 新增调度合约：非更新轮保留正常 PPO old-logprob 和 actor 更新、不采集分支；第 5/10 轮分别训练 20 步，逻辑草稿发布 revision 各增加 1，actor ACK 每轮更新。相关 39 项检查通过。
- 短测新脚本计划使用 SP=4、KL/ref 启用、10 个 RL step（覆盖第 5/10 轮），两种方法都用 5×20 调度；尚未作为 NPU 通过结果。

## SP-02：NPU 数值与分组通过

- 首次远端 CPU Gloo / NPU rendezvous 因主机名解析失败而未进入数学检查；Gloo 设 `GLOO_SOCKET_IFNAME=lo`，torchrun 使用显式 `127.0.0.1` 静态 rendezvous 后通过。未改变公共网络/认证配置。
- `ulysses-cpu-02`：4 个 CPU 多进程对照通过，补充 DP=2/SP=2 和 dummy/leader 检查在本地通过。
- `ulysses-npu-02`：4 die、真实 HCCL，随机 Qwen3 两层模型。SP=4 的 FP32/BF16 response logprob、branch logits、branch hidden 最大差异均为 0；退出 0。
- `ulysses-npu-dp-03`：同一 NPU oracle 再覆盖 DP=2/SP=2，仅 rank 0 回传真实轨迹，其余 rank（含 dummy 组）不重复回传；FP32 LP 最大差异 4.77e-7，BF16 为 0；退出 0。
- 阈值预设为 FP32 5e-5 / BF16 0.03，仅针对该小探针，不能当作所有大模型/长上下文的通用误差界。
- 暂拒绝 adaptive/dynamic/longrope 的 SP 评分，避免各序列 shard 按各自 position 最大值生成不同 RoPE 频率；部署用 Qwen3-8B 为静态 RoPE。普通 LoRA、滑窗、混合注意力等原有范围限制不放开。

## SP-03：真实 Qwen3-8B V1 / SP=4 / 5×20 调度通过

源码快照 `/home/c00954340/verl-SpeCo-ulysses-debug`，与旧 V1 SP=1 验收隔离。
两方法各使用 4 die，actor/ref SP=4、rollout TP=2、KL/ref 开启；16 个 17k 来源训练 prompt、4 个 AIME 测试 prompt，batch=4、n=2、response 上限 128、10 个 RL step、最多 3 epoch、1800 秒上限。没有运行 200 步正式训练。

| 项目 | Draft-OPD | SpecOPD |
|---|---|---|
| die | 8–11 | 4–7 |
| 日志前缀 | `ulysses-draft_opd-01` | `ulysses-spec_opd-01` |
| 退出码 | 0 | 0 |
| 草稿训练轮次 | 仅 5、10 | 仅 5、10 |
| optimizer 累计 | 20 → 40 | 20 → 40 |
| 草稿发布 | 第 5/10 轮各一次 | 第 5/10 轮各一次；逻辑 revision 1 → 2 |
| 其余 8 轮 | 正常 RL、无草稿训练 | 正常 old-logprob/actor/ref；无 proposal 采集、无草稿训练/发布 |
| 实际 actor SP | 配置 SP=4，普通 upstream 路径 | 联合前向 RPC 直接报告 `actor_sp_size=4` |
| 每个采集轮 | 基线旧 trace 采集 | 8 次联合 forward、112 个 proposal 位置 |
| 来源 actor/drafter → ACK | 原基线版本协议 | 4/0 → 5/1；9/1 → 10/2 |
| AIME 小验证 | 完成，acc@1=0 | 完成，acc@1=0 |

两方法 actor grad_norm 均为 0；本次验证控制流和数值/更新/发布，不证明 RL 学习收益。SpecOPD 同批数据内 20 步复用的损失降低也不能作为泛化质量提升证据。
SpecOPD 两个采集轮 logical/independent 行数分别 2526/5388、2426/5092，仅为结构计数，不是加速比。

结束清理中出现一条 `EngineDeadError`。完整日志显示：最终 validation 与 100% 进度已经完成；对应 Ray core-worker 日志在 11:49:53 记录 owner `INTENDED_SYSTEM_EXIT`，随后主动杀死子进程，11:49:54 vLLM 才打印 shutdown/output-handler 错误。该条为退出时的 worker 回收日志，不是训练过程中的 EngineCore 失败；原始日志保留，不屏蔽。

本地日志在 `tmp/debug44-artifacts/ulysses-*`，远端在 `/home/c00954340/specopd-debug-logs/ulysses-*`。多 optimizer 测试 1/20 两种配置均在真实容器依赖下通过；每组只在全部更新完成后生成一次发布快照。
