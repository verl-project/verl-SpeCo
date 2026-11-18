# DeepSeek V4 0731 后端与冒烟测试

Last updated: 10/29/2026

暂不接入完整 RL 时，先按 [无 RL 的 NPU 冒烟](deepseek_v4_no_rl_smoke.md) 验证独立采集、训练、恢复和导出后的推理。

## 实现范围与解耦

当前实现支持独立离线 DSpark 训练，以及 [verl rollout 内直接采集](rollout_feature_capture.md)、随 RL 自动训练和回传权重的在线路径。必要数值实现已迁入 SpeCo，不需要外部训练框架路径。MTP 未开放；真实 NPU 前反向、采集及发布仍待设备验收。资源布局沿用现有 SpeCo worker 池。

训练统一通过 `actor_rollout_ref.rollout.drafter.training.backend` 选择，缺省为 `pytorch`，新增实现为 `vllm_ascend`。后端名称区分前反向执行路径；`vllm_ascend` 的反向仍使用 PyTorch autograd 和原有 DeepSpeed 实现。

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        backend: pytorch  # 或 vllm_ascend
```

原有启动命令和默认行为保留。只在公共入口、Ray Worker、默认配置和 native 模型工厂加入集中分派或防误用检查，不修改已有前反向循环。实现按现有职责布局：

| 位置 | 职责 |
| --- | --- |
| `backends/training_backend.py` | 统一后端选择、延迟导入与执行边界 |
| `integration/vllm_ascend_config.py`、`vllm_ascend_implementation.py` | 包内配置绑定与实现指纹 |
| `backends/vllm_ascend_engine.py`、`vllm_ascend_runtime.py` | 有限训练周期与 Ascend 模型/teacher/DeepSpeed 构造 |
| `trainer/deepseek_v4_contract.py`、`deepseek_v4_dataset.py` | DS4 数据契约、导入与原生采样器复用 |
| `trainer/vllm_ascend_mask.py` | 统一 SpeCo mask 语义的分段、padding 和模型输入准备 |
| `trainer/vllm_ascend_checkpoint.py` | 模型/优化器/scheduler、数据游标和各 rank RNG 恢复 |
| `trainer/vllm_ascend_training_loop.py`、`export_deepseek_v4_draft.py` | 新后端执行流程与非破坏性导出 |
| `config/deepseek_v4_0731_draft_trainer.yaml` | 独立 Hydra 配置，不依赖 PPO 配置组合 |

文件按职责命名：`vllm_ascend_*` 表示后端执行与恢复，`deepseek_v4_*` 表示模型专属数据契约、导入和导出。当前后端限定 DeepSeek V4 0731 DSpark；本文件描述独立离线训练，随 RL 自动训练及回传权重见 [在线闭环说明](vllm_ascend_cotrain.md)。命名不代表其他模型已经可用。

为兼容已生成的数据和 checkpoint，保留序列化标识 `ds4_contract`、`ds4_draft`、`speco_ds4_v1` 和 `speco_ds4.json`。数值配置中的 `ds4_dspark`、`ds4_layout` 和已有 MTP_* 环境变量保留兼容名称；正式实现按现有目录分工组织，见 [实现来源说明](deepseek_v4_backend_provenance.md)。

两个后端共用 `verl_speco.draft_train_launcher` 和 `verl_speco.draft_train`，不再增加专用训练入口：

```bash
# 原有模型与配置：可省略 backend，默认 pytorch。
python -m verl_speco.draft_train_launcher \
  actor_rollout_ref.rollout.drafter.training.backend=pytorch \
  ...

# DeepSeek V4 0731：模型、数据和引擎配置仍需匹配。
python -m verl_speco.draft_train_launcher \
  --config-name=deepseek_v4_0731_draft_trainer \
  actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend \
  draft_engine.action=train \
  ...
```

选择 `pytorch` 时，后端分派不引入外部引擎或 DS4 训练模块；rollout 仍可独立使用 vLLM-Ascend。选择 `vllm_ascend` 不进入现有 PyTorch 训练循环。`speculative_algorithm` 仍表示模型算法，`strategy` 仍表示 native 并行策略，均不替代 `backend`。本独立训练入口只消费离线 DS4 DSpark 数据；RL 在线训练使用新接入的 SpecoWorker 适配器，独立 scheduled TQ 入口仍会拒绝。

停用新后端只需使用 `backend=pytorch`。若彻底移除新增源码，需要同时还原公共入口、Worker、默认配置和模型工厂的少量连接改动。

pre-HC 特征存放在 `DraftFeatureSample.metadata.ds4_draft.target_head_hidden_states`，数据契约放在同一命名空间的 `contract`。已有 TorchShardFeatureStore 与 TQ 的 tensor-tree codec 能无损保存这些字段，因此本阶段没有修改通用数据结构或协议。不得将其作为普通 `last_hidden_states` 使用。

mask 全链路统一为 SpeCo 的 `1=参与监督，0=忽略`，不进行反转。导入、存储、传输、分段、padding 和模型目标构造均遵循此规则；padding 与仅提供上下文的 overlap 均为 0。迁入后的 DSpark 模型直接以正向 mask 构造训练目标；`vllm_ascend_mask.py` 负责分段和 batch 组装。损失及自定义反向实现维护在包内，不注入外部命名空间。原生训练函数的 `raw_mask` 参数名作为调用兼容别名保留，其值同样采用 SpeCo 规则。

## 本地 CPU 验证

按仓库要求用 uv 创建 Python 3.12 环境。这里的 CPU 环境用于接口测试，不作为 Ascend 依赖版本建议。

```bash
uv venv --python 3.12
uv pip install pytest hydra-core torch numpy safetensors tqdm ruff
.venv/bin/python -m pytest -q \
  tests/unit/test_deepseek_v4_native_backend.py \
  tests/unit/test_deepseek_v4_hc_backward.py \
  tests/unit/test_deepseek_v4_dsa_backward.py \
  tests/unit/test_deepseek_v4_export.py \
  tests/unit/test_training_backend.py \
  tests/unit/test_deepseek_v4_contract.py \
  tests/unit/test_vllm_ascend_engine.py \
  tests/unit/test_vllm_ascend_checkpoint.py \
  tests/integration/test_vllm_ascend_source_contract.py
```

包内数值、导出、独立导入和 CLI 预检测试均不需要外部源码。仅可选的 `test_vllm_ascend_source_contract.py` 对照测试读取开发者提供的原始 checkout；可通过 `DRAFT_ENGINE_SOURCE_ROOT` 指定。未设置时该对照跳过，不影响训练。

CPU 数值测试使用小模型和真实 autograd 验证训练周期委托、累积尾批和恢复，不代表 DS4 NPU 前反向已验证。checkpoint 测试的 CPU 引擎也不代表真实 DeepSpeed 分布式恢复已通过。

首批适配阶段的本地验证记录（本轮迁入验证见 [实现来源说明](deepseek_v4_backend_provenance.md)）：

- 后端选择、新增适配及相关 factory、scheduler、TQ、恢复、launcher、native draft update 回归：261 passed，3 skipped；包括 SpeCo mask 的分段、零 padding、anchor 容量和真实目标构造对照。
- 后端实现与公共入口的 12 个源码文件通过 mypy；本次新增与分派代码通过 Ruff 检查。
- AIR 原有 `offline_training/validation/check_ds4_dspark_export.py` 合成 TP/DP 导出检查通过。
- Shell 语法、示例命名、文档日期与链接检查通过。
- 尚未运行真实 DeepSeek V4 0731 NPU 前反向、DeepSpeed 分布式恢复或推理加载验收。

## NPU 环境准备

在已配套的 Ascend 环境中，从本仓库根目录执行。保留该环境原有的 torch_npu、DeepSpeed、Transformers 和算子依赖，避免直接用 CPU 测试依赖覆盖它们。

```bash
uv pip install --no-deps -e .

export MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731-DSpark
export CAPTURE_DIR=/path/to/capture/mtp_dataset
export MASKS_PATH=/path/to/the-exact-capture/speco_masks.jsonl
export FEATURE_STORE=/path/to/new/speco_ds4_0731_features
export CARD_COUNT=8
export TRAIN_TP=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_IF_BASE_PORT=36000
```

脚本使用当前 `python`；需要时设置 `PYTHON_BIN`。多机 rendezvous 参数可透传给现有 SpeCo launcher，但首轮先验证单机。

运行时门控支持成对的 vLLM `0.25.1` / vLLM-Ascend `0.25.1rc1`，以及 vLLM `0.26.0`（含 `+empty`）/ vLLM-Ascend `0.26.0rc1`。不同系列不能混配；其他版本仍需先审阅接口。模型配置必须为 `deepseek_v4` 且包含合法的 DSpark 配置；`target_revision=0731` 是调用者对所选权重的声明，配置指纹不能证明权重文件确实来自 0731。必须使用与捕获完全匹配的 checkpoint。

## 1. 导入真实捕获数据

```bash
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh import
```

要求每个请求包含 input IDs、绝对 positions、选定层 hidden states、pre-HC teacher hidden states，以及对应的二值 `loss_mask`。mask JSONL 每行格式如下，长度须与该请求捕获的 token 序列完全一致：

```json
{"sample_idx": "request_000000", "loss_mask": [0, 0, 1, 1]}
```

导入器拒绝仅含 `prompt_generation_mask` 的旧 AIR mask 文件，不自动猜测或反转语义。此前已正确存入 SpeCo 的 Feature Store 无需反转或重写。

自定义 `sample_idx` 需要按顺序匹配 `request_000000` 时，可显式追加 `--allow-ordinal`；必须确认 chats.jsonl 次序与捕获输入完全相同。重复 ID、缺失字段、非有限值、错误形状或不连续位置会拒收。原始输入和已存在的输出不会被覆盖；失败 staging 目录保留用于排查。

首批 store 按请求保存一个 shard，便于检查和限制缓存。大量小请求的 I/O 优化留到正确性验收之后。

## 2. 环境、特征和 TP/DP 预检

```bash
RUN_DIR=outputs/ds4_0731_preflight \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh preflight
```

检查生成的 `preflight.json`。预检读取全部样本并构造每个 DP 副本的真实 DSpark batch，确认 TP 组内的样本分配规则及 DP batch 数。它不加载完整 DS4 模型。

当前 store loader 限制 `num_workers=0`，保持现有数据访问和恢复边界。包内实现不再解压源码或修改导入路径。训练 TP 与 SpeCo world size 必须整除；全局 micro batch 按 DP 副本数计算。

## 3. 单个优化器更新

```bash
RUN_DIR=outputs/ds4_0731_step1 MAX_STEPS=1 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh train
```

每次训练使用新的 `RUN_DIR`。脚本默认采用有限 smoke 配置：512 token 分段、64 overlap、4 anchors、micro batch 1、梯度累积 1、噪声关闭。需要验证 AIR 原始训练配置时，另行通过 Hydra 覆盖参数，并在后续恢复命令中保持一致。

检查：

- `training_report.json` 中 global step 为 1。
- loss 与梯度审计有限，非零 LR 下观察到参数更新。
- `checkpoints/step_00000001/` 有 DeepSpeed checkpoint、各 rank 模型状态和 `speco_ds4.json`。
- 出错时查看 `failure_rank_*.json` 和 torchrun 全部 rank 日志。

建议保留完整 stdout/stderr。训练只委托包内 `run_dspark_training`，没有再次执行 SpeCo native 的 backward 或 optimizer step。

## 4. 新进程恢复并更新到第 2 步

```bash
RESUME_FROM=outputs/ds4_0731_step1/checkpoints/step_00000001 \
RUN_DIR=outputs/ds4_0731_step2 MAX_STEPS=2 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh resume
```

`MAX_STEPS` 是恢复后累计总步数。必须保持相同源码、模型配置、数值设置、数据和 TP/DP 拓扑。恢复检查 SpeCo 实现指纹、数值配置指纹、SpeCo store manifest 游标、DeepSpeed 步数和 rank RNG。

为了验证数值连续性，另用新目录从零训练到第 2 步，与恢复后的模型参数、优化器和 scheduler 对照。BF16 容差应先依据 AIR 原始实现的重复运行误差确定。

```bash
RUN_DIR=outputs/ds4_0731_continuous MAX_STEPS=2 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh train
```

本阶段要求输入 store 在训练期间保持不变；尚未接入在线 TQ 消费确认。

## 5. 导出并校验推理参数

```bash
CHECKPOINT=outputs/ds4_0731_step2/checkpoints/step_00000002 \
EXPORT_DIR=/path/to/new/DeepSeek-V4-0731-DSpark-trained \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh export
```

复用迁入 SpeCo 的导出器执行 DP 副本选择、TP 合并、复制参数一致性检查及 DS4 参数布局转换，并逐张量验证输出。默认执行完整复制，不覆盖原模型，不放宽 Markov replica 容差。导出完整目标 checkpoint 可能需要较多磁盘空间。

`speco_export_report.json` 的 `weights_verified=true` 只表示参数转换与落盘校验通过，`inference_validated=false` 表示还需使用已验证的 DeepSeek V4 部署/评估脚本加载导出目录，执行真实生成与接受长度测试。在线热更新使用独立的 RL 发布流程，见 [在线闭环说明](vllm_ascend_cotrain.md)。

## 回传结果

请提供以下报告与对应错误栈，避免传输权重：

- `preflight.json`、`resolved_air_config.yaml`。
- 第 1 步与恢复后第 2 步的 `training_report.json`。
- 失败时的 `failure_rank_*.json` 和完整 rank 日志。
- 导出报告，以及真实生成的结果。

完整后续路线见 [迁移计划](airdrafter_migration_plan.md)。

0.26 适配范围、环境检查和真实 runtime 源码测试见 [Ascend 0.26 兼容说明](ascend_026_compatibility.md)。verl 按当前约定使用 0.9.0，本轮没有扩展其版本兼容范围。
