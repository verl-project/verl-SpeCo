# DeepSeek V4 0731 离线训练

本目录集中维护 `backend=vllm_ascend` 的离线运行入口。复用 SpeCo 已有原生轨迹重放、
Feature Store、训练、恢复和导出实现，不依赖外部 AIRDrafter，也不需要完整 RL。
原有 `examples/` 入口保留；本次只整理脚本，不修改前向、反向或数据契约。

本页默认介绍 DSpark。原生 MTP 训练使用相同脚本并设置 `DRAFT_ALGORITHM=MTP`，
其检查点、特征和训练边界见 [MTP 离线训练说明](README_MTP.md)。不能混用两种特征或草稿权重。

| 文件 | 用途 |
| --- | --- |
| `collect_hs.sh` | 对已有轨迹执行原生 vLLM-Ascend prefill，保存辅助 HS 和 pre-HC |
| `train.sh` | 预检、训练、完整 checkpoint 恢复、标准权重导出；支持旧捕获格式导入 |

## 1. 环境和目录

在已经配套的 Ascend 环境中使用。本轮验收环境为 verl 0.9.0、vLLM 0.27.1、
vLLM-Ascend 0.27.1rc1、torch/torch-npu 2.10.0、DeepSpeed 0.19.8。
旧版本兼容分支仍保留，不能随意混搭运行时版本。

以下路径对应当前服务器。换数据或开始新训练时，修改相应变量；路径建议使用绝对路径。
所有命令在 Bash 中执行，脚本会自动进入仓库根目录。

```bash
cd /data/zzc/verl-SpeCo
set -euo pipefail
export PYTHON_BIN=python3
export MODEL_PATH=/data/share/deepseek-ai/DeepSeek-V4-Flash-0731-bf16
export TRAJECTORIES=/data/zzc/speco_minimal_20261029_143417/one_trajectory.jsonl
export RUN_ROOT=/data/zzc/speco_offline_$(date +%Y%m%d_%H%M%S)
export FEATURE_STORE="$RUN_ROOT/features"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CAPTURE_TP=8 CAPTURE_DP=2
export CARD_COUNT=16 TRAIN_TP=8
mkdir -p "$RUN_ROOT"
```

采集使用 **TP8/DP2/EP** 以容纳完整 BF16 target；训练使用 **TP8/DP2、无训练 EP**。
单条轨迹会按现有 sampler 补齐两个训练 DP 的 batch，不会重复保存两份特征。
完整 checkpoint 含模型、Adam 等状态约 354 GiB，标准草稿导出约 37 GiB；预留对应磁盘和主存。
已完成的完整模型验收使用约 1.85 TiB 主存限额，不能按本说明保证 1 TiB 配额可用。

## 2. 获取 HS/pre-HC

```bash
bash offline_training/collect_hs.sh 2>&1 | tee "$RUN_ROOT/capture.log"
```

默认只抓 **1 条**，最大模型长度 4096、prefill token budget 512、采集预算 512 MiB。
不会重新生成轨迹回答；完成 prefill 所需的额外一个采样 token 会丢弃。
输入 token、绝对位置和 SpeCo `loss_mask` 保留，`1=监督、0=忽略`。

推荐输入每行包含原始 token 和显式监督范围：

```json
{"sample_id":"sample-1","input_ids":[10,11,12,13,14],"loss_mask":[0,0,1,1,1]}
```

该短示例只说明字段，实际训练需要足够的有效监督位置和 anchors。
也支持 prompt/response token、文本、messages 和已适配的轨迹目录，详见[数据格式说明](../docs/deepseek_v4_trajectory_features.md)。
每条轨迹长度必须小于 `MAX_MODEL_LEN`，为额外 token 留一个位置；不静默截断。
`MAX_SAMPLES=1` 读取输入顺序的第一条，不负责选择最短样本；首次运行建议使用已准备的短轨迹。

需要扩大抓取时显式调整，例如：

```bash
MAX_SAMPLES=all MAX_MODEL_LEN=36864 CAPTURE_MAX_BYTES=4294967296 \
  bash offline_training/collect_hs.sh 2>&1 | tee "$RUN_ROOT/capture.log"
```

该命令使用新的 `FEATURE_STORE`，不能接着覆盖已经生成的目录。
文本原始轨迹可先只准备 token/mask：

```bash
FEATURE_STORE="$RUN_ROOT/prepared" MAX_SAMPLES=all MAX_MODEL_LEN=36864 \
  bash offline_training/collect_hs.sh --prepare-only
export TRAJECTORIES="$RUN_ROOT/prepared/trajectories.jsonl"
```

准备阶段不生成 HS，随后再用原来的 `FEATURE_STORE` 执行真正采集。
`features/replay_report.json` 应为 `status=passed`，`feature_rows` 等于输入 token 数，监督行数大于 0。
只有 metadata 或部分 shard 不能作为完成依据。

已有相同 target 的完整特征时可以直接跳过抓取：

```bash
export FEATURE_STORE=/data/zzc/speco_restore_20261109_151846/features
```

## 3. 预检并训练一步

采集进程完全退出、设备释放后，再依次运行：

```bash
RUN_DIR="$RUN_ROOT/preflight" bash offline_training/train.sh preflight \
  2>&1 | tee "$RUN_ROOT/preflight.log"

RUN_DIR="$RUN_ROOT/train" MAX_STEPS=1 bash offline_training/train.sh train \
  2>&1 | tee "$RUN_ROOT/train.log"
```

默认配方与已验证短测一致：epochs=2、segment=512、overlap=64、4 anchors、spacing=8、
microbatch=1、gradient accumulation=1。训练入口设置确定性通信所需变量，保持 TP 复制参数一致。
这是默认冒烟配方，正式训练须显式设置步数、epochs、数据量和数值选项。

后续参数原样转交 Hydra，例如：

```bash
RUN_DIR="$RUN_ROOT/train_longer" MAX_STEPS=10 bash offline_training/train.sh train \
  ++draft_engine.overrides.train.epochs=10
```

检查 `training_report.json` 为 `status=passed`、`progress.global_step=1`，
并确认 `train/checkpoints/step_00000001/speco_ds4.json` 存在。
预检和训练必须使用不同输出目录；日志写在父目录，避免 `tee` 提前创建训练输出目录。
有效 anchors 不足时应检查样本和配方，不跳过数据校验。

## 4. 从完整 checkpoint 继续到第二步

```bash
export RESUME_FROM="$RUN_ROOT/train/checkpoints/step_00000001"
RUN_DIR="$RUN_ROOT/resume" MAX_STEPS=2 bash offline_training/train.sh resume \
  2>&1 | tee "$RUN_ROOT/resume.log"
```

`MAX_STEPS` 是累计上限，从 step 1 恢复到 2 只再更新一步。
恢复时保持模型、特征、数值实现/配方、TP/DP 和运行时身份一致；不能从标准导出目录恢复 Adam。
如初次训练用了自定义 Hydra 覆盖，恢复时传入相同覆盖。

## 5. 导出标准草稿权重

```bash
export CHECKPOINT="$RUN_ROOT/resume/checkpoints/step_00000002"
export EXPORT_DIR="$RUN_ROOT/draft_pytorch"
bash offline_training/train.sh export 2>&1 | tee "$RUN_ROOT/export.log"
```

此入口默认 `--format draft-pytorch`，先严格检查 TP 复制参数，再合并并逐张量读回。
需要 safetensors 时，换用新的 `EXPORT_DIR`，追加 `--format draft-safetensors`。
标准草稿含 `mtp.*` 权重，不包含完整 target，也不能让任意 AutoModel 直接推理。
初始加载仍需匹配的原始 target；完整训练恢复使用 DeepSpeed checkpoint。

## 6. 输出与检查

| 输出 | 验收内容 |
| --- | --- |
| `features/replay_report.json` | 原生采集完成、输入和监督行数匹配 |
| `preflight/preflight.json` | 模型、特征契约及 TP/DP batch 检查通过 |
| `train/training_report.json` | 完成 step 1，观察到有效优化器更新 |
| `resume/training_report.json` | 恢复后完成累计 step 2 |
| `*/checkpoints/step_*/speco_ds4.json` | 完整 checkpoint sidecar；不能以部分文件代替完成 |
| `draft_pytorch/metadata.json`、分片及索引 | 标准草稿导出完成 |

采集、训练和导出目录均使用新路径；失败时先保留日志和 `failure_rank_*.json`。
运行 `bash offline_training/collect_hs.sh --help` 或 `bash offline_training/train.sh --help` 可查看入口参数。
更多实机验收范围见[离线恢复与标准权重记录](../docs/deepseek_v4_offline_recovery.md)。
