# .44 Draft-OPD / SpecOPD 短测结果

Last updated: 11/16/2026

已完成用户要求的短步调试：保留 Draft-OPD 基线，同时可通过内部方法开关运行 SpecOPD。两条路径均执行真实采集、草稿更新、权重发布及 AIME 小验证，退出码为 0。未进行长跑，未据此宣称质量或速度提升。

## 环境与数据

- 主机：`7.208.150.44`，容器：`verl-speco-090`。
- 代码：`/home/c00954340/verl-SpeCo-debug`；本地：`D:/CodexProject/specopd/verl-SpeCo`。
- target：`/data/c00954340/Qwen3-8B`；drafter：`/data/c00954340/qwen3_8b_dspark`，5 层、K=7、vanilla Markov rank 256、全词表 151936。
- Python 3.12.13；torch/torch-npu 2.10；verl 0.9.1；vLLM `0.30.1.dev0+gced6857af.d20261013`；vllm-ascend `0.19.1rc2.dev2631+g5f9382a66`；Transformers 5.10.4。
- 从用户 17k/AIME parquet 中分别选取 16/4 个不同 prompt，保存到 `/data/c00954340/specopd-debug/{train16,aime4}.parquet`。原始数据未改写。
- 两步 RL，batch=4、rollout n=2、最多生成 128 tokens；每个任务 4 个 die，vLLM TP=2。初始并行使用 0–3 / 4–7；其他数值探针使用独立 die。

## 直接证据

| 项目 | Draft-OPD attempt 10 | SpecOPD attempt 15 |
|---|---|---|
| 退出码 | 0 | 0 |
| 每步采集/评分 | 8 个样本 | 8 条轨迹 × 2 分支 × 7 token = 112 个位置 |
| 草稿 optimizer step | 1 → 2 | 1 → 2 |
| 草稿实际 loss | 0.6675 / 0.8236 | 0.8023 / 0.7994 |
| 更新与发布 | 每步 trained=1、published=1 | 每步 trained=1、published=1；草稿 ACK 1 → 2 |
| Actor 评分 | 原基线 old-logprob | 每步 8 次联合模型调用，对应 8 条轨迹 |
| 正常 RL 数据 | 保留 | 正常 response old-logprob 与分支监督从同一次联合前向读出 |
| AIME | 4 个 prompt，acc@1=0 | 4 个 prompt，acc@1=0 |

SpecOPD 第一步逻辑行数为 2334，逐分支独立计算对应 4807；第二步为 2302 / 4714。它们是结构/调用计数，不能当作实测加速比。日志可见完整 proposal wire 及拒绝后尾部；没有从最终接受回答反推尾部。

严格版本路径已实测：第一轮 actor/drafter 来源 revision 为 0/0，提交后服务 ACK 为 1/1；第二轮来源为 1/1，提交后 ACK 为 2/2。ACK 代表完成的发布边界，不代表 actor 参数一定产生非零变化。

## 数值验证及边界

1. 真实 Qwen3-8B NPU 联合评分，与不含 sibling/future-trunk 的独立普通因果 forward 对照。对齐物理 padding/KV 位置并保留正确逻辑 RoPE 后，BF16、FP32 的分支 logits/hidden 与主干辅助 hidden 差异均为 0，正常 response logprob 最大差异分别约 1.43e-6 / 1.91e-6；两项测试通过。
2. 同时保留不同物理布局的诊断值：该合成 fixture 上 BF16 的普通 padding logprob 最大差异 0.352724、紧凑独立分支 TV 最大 0.028218；FP32 相应约 3.15e-5 / 6.81e-6。不能把对齐布局后的结果理解成所有推理/训练布局逐位相同。
3. 原生 AscendDSparkSpeculator：固定 96-token 上下文，捕获第一组真实 7-position 条件 logits/采样 IDs，再用训练代码、同一上下文及固定草稿历史重建 q。T=1，最大 TV=0.0243364、KL=0.0020278、实际采样 token 的 logprob 最大差异=0.101965，通过探针事先设置的 TV≤0.03、LP≤0.125 检查。该阈值仅用于此次小探针，不是通用质量保证。
4. 两个 TP rank 的原生草稿 IDs 与条件 logits 一致；drafter checkpoint 的自有 embedding/head 与 target checkpoint 在有效 BF16 下相等。当前模型的自有 embedding 保持冻结；已有 DSpark target 更新钩子会同步 Actor LM head，并保留在线训练得到的其他 drafter 权重。

这些验证排除了已覆盖配置中的布局编译/读出错误，并量化了一组训练/推理路径差异；没有逐算子分解全部误差。Draft-OPD 也存在同类低精度路径差异，但本次未单独量化其 q 差值。长序列、更多 prompt、其他量化/backend 的数值范围，以及接受率/收敛影响仍属于后续正式实验。

此次短回答奖励均为 0，Actor 梯度范数为 0；它验证了正常 RL 调度、计算及发布接口，不证明 Actor 学习收益。SpecOPD 通用日志中的分类 accuracy 为 N/A/NaN（未提供分类诊断计数），实际 E2E-TV loss 有限且 optimizer 正常执行。另有 NPU jagged→padded CPU fallback，未做吞吐、host/Ray 峰值或长期质量验收。

## 修复位置

- `verl_speco/main.py`：在 Ray 启动前传递解析后的 drafter 配置。
- `integration/vllm_runtime.py`：跨进程保留 HTTP 补丁及配置；SpecOPD 的草稿发布不改写 Actor global_steps；frontend ACK 与请求版本检查。
- `integration/specopd_vllm.py`：MRV2 对普通请求默认只回传占位符，SpecOPD 显式回传真实 ID。NPU 当前同步复制小量整数 ID，不增加模型 forward；完整 proposal 的宽度/负值检查保留。
- `integration/specopd_driver.py`：联合 old-logprob 替换、版本确认、一次消费/更新/发布及同一步发布指标。
- `integration/agent_loop_runtime.py`、`specopd_agent.py`：将完整 wire 从 LLM 输出传到轨迹 batch。
- NPU 短脚本：Draft-OPD 使用原 unpadded actor 路径；SpecOPD 使用 dense eager 树评分。因此本次两种方法的耗时不可作为公平性能对照。

训练数据通道不保存 rollout target 监督张量。q 探针的临时上下文/logits 捕获仅用于验证，没有进入训练数据。

## 复现短测

先选择空闲的 4 个 die；以下在容器内顺序执行，输出目录应换成新名称：

```bash
cd /home/c00954340/verl-SpeCo-debug
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
OPD_METHOD=draft_opd DEBUG_OUTPUT=/home/c00954340/specopd-debug-logs/recheck-draft \
  timeout --signal=TERM --kill-after=30 900 bash examples/run_qwen3-8b-opd-debug_drafter_dspark_vllm_npu.sh
OPD_METHOD=spec_opd DEBUG_OUTPUT=/home/c00954340/specopd-debug-logs/recheck-spec \
  timeout --signal=TERM --kill-after=30 900 bash examples/run_qwen3-8b-opd-debug_drafter_dspark_vllm_npu.sh
```

本地最终回归：368 passed / 15 skipped。真实 NPU 小模型数值检查 12 passed、原生 ID transport 2 passed、真实 checkpoint oracle 2 passed；后两项有独立仓库测试。完整日志/config/exit、数值捕获和 JSON 位于本地 `D:/CodexProject/specopd/tmp/debug44-artifacts` 与远端 `/home/c00954340/specopd-debug-logs`。逐轮失败、修复和未覆盖项见 [测试台账](debug44_validation.md)。
