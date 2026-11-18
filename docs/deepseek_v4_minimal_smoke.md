# DeepSeek V4 最小特征与一步训练冒烟

Last updated: 10/29/2026

目标仅是打通真实前反向与 checkpoint：抓取一条完整短轨迹，再训练一个 optimizer step。无需完整 RL、全量特征或导出整套模型。

服务器预处理目录为 `/models/z00829645/speco_trajectory_prepared_20261029T033650Z`。已确认最短样本 2,608 token、159 个监督 token，在当前 512 长度、64 overlap、4 anchors、间距 8 的配方下，只有一个有效训练片段，满足 4 个 anchors。BF16 的 HS/pre-HC 张量合计约 142.6 MiB，文件及临时张量另计。抓取短轨迹减少特征量，但仍须加载真实目标模型；设备数按模型容量设置。

## 服务器代码状态

10/29 已将最新 SpeCo 更新至 `/models/z00829645/verl-SpeCo`，包内采集和训练入口可导入，服务器针对性 CPU 回归 147 项通过。替换前备份位于 `/models/z00829645/verl-SpeCo-backup-20261029T035818Z`。无需重复上传，进入现有 Ascend 环境后从以下样本选择步骤开始；真实 NPU 抓取与训练尚待执行。

## 选择一条短轨迹

进入已配套的 Ascend Python 环境，在服务器仓库根目录执行：

```bash
cd /models/z00829645/verl-SpeCo
uv pip install --no-deps -e .
export MODEL_PATH=/models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16
export RUN_ROOT=/models/z00829645/speco_minimal_$(date +%Y%m%d_%H%M%S)
export FEATURE_STORE=$RUN_ROOT/features
export SMOKE_TRAJECTORY=$RUN_ROOT/one_trajectory.jsonl
export CARD_COUNT=16
export TRAIN_TP=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
mkdir -p "$RUN_ROOT"
set -o pipefail

python - <<'PY'
import json, os
source = '/models/z00829645/speco_trajectory_prepared_20261029T033650Z/trajectories.jsonl'
with open(source) as handle:
    row = min((json.loads(line) for line in handle if line.strip()), key=lambda r: len(r['input_ids']))
with open(os.environ['SMOKE_TRAJECTORY'], 'x') as handle:
    handle.write(json.dumps(row) + '\n')
print('selected:', row['sample_id'], 'tokens:', len(row['input_ids']), 'supervised:', int(sum(row['loss_mask'])))
PY
```

该选择使用完整原始 token/mask，不截断上下文；原文件保持只读。预期选到 `gpt-5.6-terra_20260926-123021-52983e15_claude_add_rms_norm.jsonl:1`。

## 抓取一条并训练一步

```bash
python -m verl_speco.replay_deepseek_v4_features \
  --model "$MODEL_PATH" --trajectories "$SMOKE_TRAJECTORY" \
  --output "$FEATURE_STORE" --tensor-parallel-size "$TRAIN_TP" \
  --data-parallel-size 2 --enable-expert-parallel \
  --max-model-len 4096 --max-num-batched-tokens 512 \
  --max-samples 1 --capture-max-bytes 536870912 \
  2>&1 | tee "$RUN_ROOT/capture.log"

RUN_DIR="$RUN_ROOT/preflight" \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh preflight \
  2>&1 | tee "$RUN_ROOT/preflight.log"

RUN_DIR="$RUN_ROOT/train_step1" MAX_STEPS=1 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh train \
  2>&1 | tee "$RUN_ROOT/train.log"
```

每一步成功后再执行下一步。采集结束后退出推理进程，再启动训练；此配方 batch=1、梯度累积=1、最多一个 optimizer step。4 个 anchors 是该训练片段的内部采样点，不会为此重新抓取四条特征。训练使用 TP=8、DP=2（CARD_COUNT=16），反向仍不启用 EP。现有 sampler 会为这个最小样本补齐 DP batch，因此两副本各消费同一个训练片段，特征只保存一份。

## 验收

- `features/replay_report.json`：`status=passed`，一个样本、2,608 行特征、159 行监督。
- `preflight/preflight.json`：`status=passed`，`source_samples=1`、`segments=1`。
- `train_step1/training_report.json`：`status=passed`、`progress.global_step=1`。
- `train_step1/checkpoints/step_00000001/`：checkpoint 完成。

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
capture = json.loads((root/'features/replay_report.json').read_text())
preflight = json.loads((root/'preflight/preflight.json').read_text())
training = json.loads((root/'train_step1/training_report.json').read_text())
assert capture['status'] == preflight['status'] == training['status'] == 'passed'
assert len(capture['samples']) == preflight['source_samples'] == 1
assert preflight['segments'] == 1
assert training['progress']['global_step'] == 1
print('Single-trajectory capture and one-step training passed')
PY
```

这只验收独立抓取与一次训练，不证明恢复连续性、导出后推理或 RL 在线热更新已经通过。保留 RUN_ROOT 下报告和完整日志；失败时查看对应阶段错误及 `failure_rank_*.json`。

## 本机 OOM 与并行配置

10/29 服务器日志确认旧抓取配置 TP=8、DP=1、EP 关闭时，在 FusedMoE 权重张量构造阶段耗尽约 61.28 GiB 的单设备容量，尚未分配 KV cache 或采集特征。checkpoint 约 608 GB，不能按 8 个 64 GB 逻辑 NPU 配置加载完整 BF16 target。

参照用户提供的 `configs/hs_data_dspark.yaml` 和显式 DP 启动逻辑，采集使用 TP=8、DP=2、EP=true，总计 16 个逻辑 NPU。两个独立前端通过 VLLM_DP_* 建立 DP/EP 拓扑，各 TP worker 使用完整可见设备列表并由原生 local DP rank 选择设备切片。不能直接给单进程 LLM 传 data_parallel_size=2，也不能只把数据复制为两个无 EP 副本。

独立采集入口新增 `--data-parallel-size`、`--enable-expert-parallel`。两个前端执行同一条小轨迹以保持 EP collective 调度一致，只有 DP0 注册抓取并回传 HS/pre-HC，所以存储仍是一条约 143 MiB 的 feature。默认 DP=1、EP=false 行为保留，RL 在线训练的 no-EP 限制及现有数值反向未改变。CPU spawn/协议回归不代表实际 NPU DP/EP 已验收。

## 特征抓取失败后的重试目录

模型初始化完成后发生抓取错误时，features 目录可能已有 metadata。保留失败现场，给重试选择新目录，并让后续预检和训练继续使用同一 FEATURE_STORE：

```bash
export FEATURE_STORE=$RUN_ROOT/features_native_aux_$(date +%Y%m%d_%H%M%S)
```

10/29 的 TP8/DP2/EP 实测已通过 target 权重加载和初始化；当前修正了 DeepSeek 原生编译包装器绕过 forward hook 的采集问题，改用执行状态的 aux 与原生 pre-HC getter。仍需真实 NPU 重试验收。

## 一步训练后的 checkpoint 主机内存限制

10/30 首次 NPU 一步训练已执行前向、反向和参数更新，loss=0.456684；保存 checkpoint 时 rank 10 收到 SIGKILL。容器主机内存限额约 93.1 GiB，历史峰值触及限额且存在 OOM kill 记录，尚无内核 PID 对应记录。部分文件不是完整 checkpoint，不可恢复或导出。

保存路径现对同步 TorchCheckpointEngine 的本地序列化使用共享文件锁，限制为一次一个写入，并使用 ZIP 序列化。所有 rank 仍集体进入 DeepSpeed 保存流程；模型、优化器、调度器和 RNG 状态不裁剪。额外模型权重的 CPU 拷贝同样限流。锁文件位于 checkpoint 根目录，不要在保存过程中删除；共享文件系统必须支持 flock。保存会比并行写入更慢，真实 NPU 重试仍待验证。

复用已采集的 FEATURE_STORE，改用新的 RUN_DIR 重新运行 train；不要从失败的 step_00000001 恢复。
