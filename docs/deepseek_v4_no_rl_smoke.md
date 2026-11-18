# DeepSeek V4 0731 无 RL 的 NPU 冒烟

Last updated: 10/29/2026

如果已有完整轨迹，使用 [轨迹重放采集](deepseek_v4_trajectory_features.md)，可直接获取其 HS/pre-HC，无需重新生成回答。

## 验证范围

先验证独立串行流程：真实 vLLM-Ascend 推理 → HS/pre-HC → SpeCo Feature Store → 独立 DSpark 训练 → checkpoint 恢复 → 导出 → 新进程加载导出模型。无需 RL_SCRIPT、reward、PPO 或 actor 更新，也不需要外部 AIRDrafter 目录。原有 RL 在线路径保留。

采集复用 `AscendRolloutCapture`、`RolloutFeatureCollector` 和数据契约；训练、恢复和导出复用已有 standalone 入口。各阶段分别退出，推理与训练无需同时驻留。此流程不验证 verl worker 生命周期、在线调度、teacher 动态同步或原地热更新。

新入口为 `python -m verl_speco.collect_deepseek_v4_features`，其请求提交、step 和 RPC 接口核对了 [vLLM 0.26.0 的 LLMEngine 源码](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/llm_engine.py)。实际设备前向仍待验收。

## 1. 环境和目录

先同步最新 SpeCo 本地修改到服务器，在已配套的 Ascend 环境中按 [服务器操作手册](blue_zone_machine_runbook.md) 安装本仓库并检查版本。以下均在仓库根目录执行，模型必须为所选 0731 BF16 完整 checkpoint。

```bash
export MODEL_PATH=/实际路径/DeepSeek-V4-0731-bf16
export CARD_COUNT=8
export TRAIN_TP=8
export SMOKE_ROOT=/models/z00829645/speco_no_rl_$(date +%Y%m%d_%H%M%S)
export FEATURE_STORE=$SMOKE_ROOT/features
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export MTP_DS4_TP_SHARD_EH_PROJECTIONS=0
export MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS=0
mkdir -p "$SMOKE_ROOT"
set -o pipefail
```

卡数和 TP 按模型容量调整，`CARD_COUNT` 必须能被 `TRAIN_TP` 整除。首轮设两者相等，关闭 EP，保持与采集的 TP 度一致。

## 2. 真实 rollout 采集

```bash
python -m verl_speco.collect_deepseek_v4_features \
  --model "$MODEL_PATH" \
  --output "$FEATURE_STORE" \
  --tensor-parallel-size "$TRAIN_TP" \
  --max-model-len 512 --max-tokens 128 \
  2>&1 | tee "$SMOKE_ROOT/capture.log"
```

默认使用两条内置短 prompt，每条生成 128 token。为获得足够监督行，冒烟使用 greedy、忽略 EOS；它不衡量回答质量。可通过 `--prompts /path/prompts.jsonl` 替换，文件每行为 `{"prompt":"..."}` 或 `{"prompt_token_ids":[1,2,3]}`，文本按原文编码，不自动套 chat template。

检查 `$FEATURE_STORE/capture_report.json`：`status=passed`，每条样本的 `supervised_rows>0`。报告保留生成 token、真实前向行数及模型数据契约。末尾最后一个采样 token 尚无前向特征，因此不伪造其 HS。Feature Store 的 mask 统一为 prompt=0、有效生成前向行=1。

输出目录必须不存在；失败输出保留供排查，重新运行须换目录。此阶段完全退出后再启动训练。

## 3. 预检和一个优化器更新

```bash
RUN_DIR="$SMOKE_ROOT/preflight" \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh preflight \
  2>&1 | tee "$SMOKE_ROOT/preflight.log"

RUN_DIR="$SMOKE_ROOT/step1" MAX_STEPS=1 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh train \
  2>&1 | tee "$SMOKE_ROOT/train1.log"
```

预检应通过，`training_report.json` 应记录 global step=1，并生成 `step1/checkpoints/step_00000001/`。使用采集时完全相同的 target checkpoint，不能用另一个相同 config 的模型替代 teacher。若预检报告有效 anchor 不足，应增加采集输出长度并换新 Feature Store，而非绕过校验。

## 4. 新进程恢复到第二步

```bash
RESUME_FROM="$SMOKE_ROOT/step1/checkpoints/step_00000001" \
RUN_DIR="$SMOKE_ROOT/step2" MAX_STEPS=2 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh resume \
  2>&1 | tee "$SMOKE_ROOT/train2.log"
```

MAX_STEPS 为累计步数。恢复应保留优化器、scheduler、数据游标和 RNG，报告累计 global step=2。模型、特征目录、数值选项和 TP/DP 拓扑保持一致。

## 5. 导出并重新推理

```bash
export EXPORT_DIR=$SMOKE_ROOT/exported_model
CHECKPOINT="$SMOKE_ROOT/step2/checkpoints/step_00000002" \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh export \
  2>&1 | tee "$SMOKE_ROOT/export.log"

python -m verl_speco.collect_deepseek_v4_features \
  --model "$EXPORT_DIR" \
  --output "$SMOKE_ROOT/features_after_export" \
  --tensor-parallel-size "$TRAIN_TP" \
  --max-model-len 512 --max-tokens 128 \
  2>&1 | tee "$SMOKE_ROOT/inference_after_export.log"
```

导出完整模型需要额外磁盘空间，输出路径不得覆盖原模型。检查 `speco_export_report.json` 中参数验证通过，以及新进程能够完成生成和特征采集。输出文本未变化不意味着权重未更新；teacher/target 固定时，greedy 的最终输出可能相同。参数转换校验、真实权重变化和推理加载应分别核对。

通过本流程后，再进行 [完整 RL 在线闭环](vllm_ascend_cotrain.md)。保留 capture/preflight/training/export 报告和各阶段完整日志。

## 本地验证边界

CPU 测试使用真实 confirmed-row collector 和 Feature Store，覆盖请求先注册后提交、每次生成只执行一次模拟前向、terminal token 排除、SpeCo mask、失败清理和重复 TP payload 拒收。CPU 结果不证明 vLLM/NPU 实际运行或训练数值已通过。
