# DeepSeek V4 原生 MTP 离线训练

使用 `backend=vllm_ascend`、`speculative_algorithm=MTP`、`draft_engine.variant=ds4_mtp`。
独立 PyTorch MTP 基线尚待用户提供；本入口是 AIRDrafter 原生 Ascend MTP 训练路径的迁入。
复用 SpeCo 的 Feature Store、分段/DP sampler、分布式启动、DeepSpeed 构造、
teacher、进度和完整 checkpoint 保存/恢复，不改变已有 DSpark/PyTorch 数值实现。

## 模型与数据

- 使用真正包含原生单层 MTP 的 DeepSeek V4 BF16 checkpoint。
  当前服务器使用 `/data/share/deepseek-ai/DeepSeek-V4-Flash-BF16`。
- **不能使用 `DeepSeek-V4-Flash-0731-bf16` 的 DSpark 参数冒充 MTP**：即使配置同时有
  `num_nextn_predict_layers=1`，其中的 `dspark_*` 和草稿层布局仍不同，预检会拒绝。
- MTP 特征是最终 HC head/norm 前的完整残差，宽度 `hidden_size × hc_mult`；
  本目标为 16384。它不是 DSpark 的三层辅助 HS，也不是压缩后的 4096 维输出。
- 现有 DSpark 特征目录不自动转换为 MTP。特征必须来自本次指定的真实 MTP target。
- mask 采用 SpeCo `1=监督、0=忽略`，绝对位置保持不变。第 k 次递推的 token/teacher
  分布向前偏移 k 行，位置偏移 k−1 行；新增尾部 mask 为 0。

## 顺序运行

在配套的 Ascend 环境和仓库根目录运行：

```bash
cd /data/zzc/verl-SpeCo
set -euo pipefail
export PYTHON_BIN=python3 DRAFT_ALGORITHM=MTP
export MODEL_PATH=/data/share/deepseek-ai/DeepSeek-V4-Flash-BF16
export TRAJECTORIES=/data/zzc/speco_minimal_20261029_143417/one_trajectory.jsonl
export RUN_ROOT=/data/zzc/speco_mtp_$(date +%Y%m%d_%H%M%S)
export FEATURE_STORE="$RUN_ROOT/features"
export CAPTURE_TP=8 CAPTURE_DP=2 CARD_COUNT=16 TRAIN_TP=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
mkdir -p "$RUN_ROOT"

# 默认一条轨迹；独立原生重放，不在 verl rollout 中安装 HS hook。
bash offline_training/collect_hs.sh 2>&1 | tee "$RUN_ROOT/capture.log"

# 等采集进程退出、设备释放后执行。
RUN_DIR="$RUN_ROOT/preflight" bash offline_training/train.sh preflight \
  2>&1 | tee "$RUN_ROOT/preflight.log"

RUN_DIR="$RUN_ROOT/train" MAX_STEPS=1 bash offline_training/train.sh train \
  2>&1 | tee "$RUN_ROOT/train.log"

export RESUME_FROM="$RUN_ROOT/train/checkpoints/step_00000001"
RUN_DIR="$RUN_ROOT/resume" MAX_STEPS=2 bash offline_training/train.sh resume \
  2>&1 | tee "$RUN_ROOT/resume.log"

export CHECKPOINT="$RUN_ROOT/resume/checkpoints/step_00000002"
export EXPORT_DIR="$RUN_ROOT/draft_pytorch"
bash offline_training/train.sh export 2>&1 | tee "$RUN_ROOT/export.log"
```

采集为 TP8/DP2/EP；训练为 TP8/DP2、非 EP、ZeRO-1。默认 microbatch=1、
累积=1、512-token segment、overlap=64、epochs=2、学习率 1e-5、递推 2 步。
`MAX_STEPS` 是累计优化器步数，与递推长度不同。改变递推长度可追加：

```text
++draft_engine.overrides.train.ttt_length=4
```

恢复时保持相同数值配置、特征、模型和 TP/DP；只是提高累计步数上限。
完整 checkpoint 包含模型、Adam、进度/游标和 RNG。新运行使用新目录，不覆盖旧训练。
已有 AIR MTP 捕获文件可使用 `train.sh import`，必须另外提供正向 SpeCo mask；
不读取或反转 AIR 的 prompt mask。

## 损失与梯度

首期支持 AIR 的 **Eagle KL**：按完整词表分布蒸馏、递推步权重 `0.8^step`，
teacher 冻结；默认保持递推 hidden-state 图相连，microbatch 内一次 backward。
复用原生 MTP HC/attention/MoE/输出头和既有算子反向适配；不使用 Qwen 草稿替换 V4。
跨 DP 和梯度累积按每个预测步的全局有效 token 数归一化，而不是平均各样本的局部均值。
TP 只保存局部词表分布，避免为 KL 聚合完整 logits。

目前显式拒绝其他 loss、量化训练、多个原生 MTP 层、EP 训练和在线 RL MTP。
保留 AIR 的源码来源 SHA256，见 `verl_speco/config/deepseek_v4_mtp_sources.json`。
MTP 使用单独数值身份；新增功能不改变已有 DSpark 的数值指纹或恢复条件。

## 导出与初始化

`train.sh export` 默认标准 PyTorch 草稿，可追加 `--format draft-safetensors`。
两者使用独立 MTP layout 标识，复用现有 MTP TP 合并/矩阵转换，先检查复制参数，
再逐张量读回。不默认复制完整 target。原始共享 LM-head 从匹配 target 读取。

以标准草稿重新初始化训练可追加：

```text
++draft_engine.overrides.model.initial_draft_artifact=/absolute/path/to/draft_pytorch
```

它新建优化器；完整续训使用 `RESUME_FROM`。导出文件可读、CPU 数学测试和真实 NPU
训练/恢复/冷加载是不同级别的验收；实测记录见
[接入与验证记录](../docs/deepseek_v4_mtp_training.md)，不能套用 DSpark 的通过结论。

## 无模型 CPU 数值检查

在项目 Python 环境中运行：

```bash
python3 -m pytest -q \
  tests/unit/test_deepseek_v4_mtp_training.py \
  tests/unit/test_deepseek_v4_mtp_artifact.py \
  tests/unit/test_deepseek_v4_mtp_runtime.py \
  tests/unit/test_deepseek_v4_mtp_probe.py

SPECO_RUN_DISTRIBUTED_TESTS=1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m pytest -q \
  tests/integration/test_deepseek_v4_mtp_loss_distributed.py
```

CPU 检查包含独立稠密数学参考，不能代替 Ascend 原生算子、完整 checkpoint 和实际加载器验证。
