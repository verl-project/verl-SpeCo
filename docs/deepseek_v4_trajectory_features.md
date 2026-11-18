# 从已有轨迹采集 DeepSeek V4 特征

Last updated: 10/29/2026

只为跑通训练时，使用 [一条短轨迹、一步训练](deepseek_v4_minimal_smoke.md)，特征张量约 143 MiB。

## 用途与来源

已有轨迹无需重新生成回答。`verl_speco.replay_deepseek_v4_features` 将完整轨迹作为 vLLM-Ascend prefill，抓取选定 target 层 HS 和 pre-HC，直接写入 SpeCo TorchShardFeatureStore，供现有独立 DSpark 训练入口读取。无需 RL_SCRIPT、reward、PPO 或外部 AIRDrafter 路径。

参考用户提供的 `data_preparation/vllm_ascend_hs_dataset_creator.py` 中 `_iter_capture_batches`、`_patch_deepseek_v4_dspark_target_capture` 和 `_patch_dspark_model_runner_chunk_capture`：完整序列作为输入，aux states 来自 target backbone，pre-HC 来自 `get_mtp_target_hidden_states()`，按绝对位置对齐。实际抓取复用 SpeCo 已有的 `AscendRolloutCapture` 和 confirmed-row collector，不复制 AIR 的全局 hook、DP 控制平面或 mask 规则。

为完成 vLLM 请求，额外采样一个 token 并丢弃；保存的 token 和所有 HS 行严格对应原轨迹，包括原轨迹最后一个 token 的前向状态。长轨迹可分块 prefill，不截断输入。该离线重放不代表已验证 RL 在线热更新。

## 支持的 JSONL 格式

推荐使用原始 token IDs，避免文本重新编码改变 token 边界。每行选择一种格式。

完整 token 序列及 SpeCo mask，适合多轮任意监督范围：

```json
{"sample_id":"t1","input_ids":[10,11,12,13,14],"loss_mask":[0,0,1,1,1]}
```

分开的 prompt/response token IDs，默认 prompt=0、response=1：

```json
{"sample_id":"t2","prompt_token_ids":[10,11],"response_token_ids":[12,13,14]}
```

已完成的文本轨迹：

```json
{"sample_id":"t3","prompt":"Question: ...\nAnswer: ","response":"Recorded answer..."}
{"sample_id":"t4","messages":[{"role":"user","content":"Question..."},{"role":"assistant","content":"Recorded answer..."}]}
```

文本和 messages 复用 SpeCo Producer 的解析、tokenization 和监督边界；也支持其 `conversation`/`conversations` 格式。多轮 messages 默认只监督最后一条 assistant 回答，之前回合作为上下文；若需要监督多个回合，使用显式 `input_ids + loss_mask`。文本重新编码不保证恢复原始采样 token IDs。

AIRDrafter 风格完整序列化文本也可输入，但必须提供对应的 SpeCo 正向 mask：

```json
{"sample_id":"t5","chat":"完整序列化对话文本","loss_mask":[0,0,1,1]}
```

`chat` 使用 target tokenizer 的 `add_special_tokens=True`，mask 长度必须匹配其实际 token 数，示例 mask 仅说明结构。不接受只有 `prompt_generation_mask` 的记录，不猜测或反转语义。输入缺少回答、重复 ID、token 越界、mask 不合法或长度超限时明确报错，并注明文件行号。

## NPU 命令

先同步最新 SpeCo 代码，在已配套的 Ascend 环境中安装本仓库。模型使用目标 DeepSeek V4 0731 BF16 checkpoint，TP 按可用设备和容量调整：

```bash
cd /models/z00829645/verl-SpeCo
export MODEL_PATH=/实际路径/DeepSeek-V4-0731-bf16
export TRAJECTORIES=/实际路径/trajectories.jsonl
export FEATURE_STORE=/实际路径/新的特征目录

set -o pipefail
python -m verl_speco.replay_deepseek_v4_features \
  --model "$MODEL_PATH" \
  --trajectories "$TRAJECTORIES" \
  --output "$FEATURE_STORE" \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-samples 2 \
  2>&1 | tee trajectory_capture.log
```

`--max-samples 2` 先验证两条；正式抓取时删除该参数。每条轨迹需在 `max-model-len` 内留出一个采样位置；超限时提高配置或在上游明确准备新样本，不静默截断。分块 token budget 小于上下文长度时启用 chunked prefill。脚本固定 V1、eager、关闭 prefix caching/FlashComm1/EP，并复用现有 worker 扩展。

输出目录必须不存在。每条样本单独 flush，避免长轨迹积攒为大批 CPU tensor。失败输出保留，不生成成功报告；再次运行换新目录，不覆盖原轨迹或已有特征。

## 验收与训练

检查 `$FEATURE_STORE/replay_report.json`：`status=passed`，样本 ID 对应输入，`feature_rows` 等于完整轨迹 token 数，`supervised_rows>0`。目录内还包括 metadata、manifest 和 tensor shards；HS/pre-HC 已经按现有训练契约保存，无需再转换。

随后直接复用现有预检和训练命令：

```bash
export CARD_COUNT=8
export TRAIN_TP=8
RUN_DIR=outputs/trajectory_preflight \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh preflight

RUN_DIR=outputs/trajectory_step1 MAX_STEPS=1 \
bash examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh train
```

预检若提示有效 anchor 不足，需要更长的监督片段或明确调整训练配置，不能跳过校验。后续恢复、导出与新进程推理见 [无 RL 的 NPU 冒烟](deepseek_v4_no_rl_smoke.md)。

CPU 测试覆盖原 token/mask 保留、文本与消息边界、错误行和重复 ID 拒收、丢弃额外采样 token，以及真实 collector 输出可被现有 StoreDataset 直接读取。真实 NPU 抓取与数值仍需设备验收。

本轮本地相关回归：131 项通过，2 项可选 Parquet 测试因缺少 pyarrow 跳过；三个相关源文件 mypy 与 Ruff 通过，新增模块已验证纳入 wheel。该记录不包含真实 NPU 执行。

## 当前服务器的 Alpaca 工具轨迹

已只读检查 `/models/z00829645/dsv4_sft/traj_alpaca_32k`：139 个 JSONL、139 条记录，结构为 `instruction/input/output/history/system/messages/metadata`，schema 为 `alpaca-role-preserving-v1`，全部以 tool 消息结束。`messages` 含原始角色、内容、`tool_calls` 和 `tool_call_id`，是采集的权威输入；不从空的 `output` 字段重建回答。

该 schema 使用全回合监督：所有 assistant 回合的内容和工具调用为 1，system/user/tool 上下文为 0；保留最终 tool 消息和历史 reasoning，原消息不做角色清洗或字段裁剪。模板使用 `drop_thinking=False`，监督边界复用 SpeCo 的 `boundary_from_renders`；若模板重写历史导致不能对齐则拒绝处理。

目标模型为 `/models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16/`。已用其实际 vLLM `deepseek_v4` tokenizer 在 CPU 上检查第一条 DeepSeek 轨迹：27,981 token，9 个 assistant 回合均满足历史/完整序列前缀对齐。模型目录本身没有 chat_template，不能用默认 AutoTokenizer 模板替代原生 tokenizer。

目录混有多个模型生成的文本轨迹，长度元数据使用的 tokenizer 不同，因此要先用目标 tokenizer 编码，不能直接把原 `metadata.token_count` 当成目标长度。目录输入按文件名排序，默认 sample ID 为 `文件名:行号`，`--max-samples` 限制整个目录的总样本数。

先在 CPU 上准备 token/mask，不加载 NPU 模型：

```bash
export MODEL_PATH=/models/l00809589/ckpt/DeepSeek-V4-Flash-0731-bf16
export TRAJECTORIES=/models/z00829645/dsv4_sft/traj_alpaca_32k
export RUN_ROOT=/models/z00829645/speco_trajectory_$(date +%Y%m%d_%H%M%S)

python -m verl_speco.replay_deepseek_v4_features \
  --model "$MODEL_PATH" --trajectories "$TRAJECTORIES" \
  --prepare-only --output "$RUN_ROOT/prepared" \
  --max-model-len 131072
```

输出 `prepared/trajectories.jsonl` 和 `prepare_report.json`。报告含样本数、每条 token/监督数及最大长度；`npu_forward_validated=false`。再按实测最大长度选择 NPU 的 `--max-model-len`，至少比最长序列多一个 token，避免静默截断。

例如最长轨迹在 65,535 token 内时，先抓取两条：

```bash
export FEATURE_STORE=$RUN_ROOT/features
python -m verl_speco.replay_deepseek_v4_features \
  --model "$MODEL_PATH" \
  --trajectories "$RUN_ROOT/prepared/trajectories.jsonl" \
  --output "$FEATURE_STORE" \
  --tensor-parallel-size 8 --max-model-len 65536 \
  --max-num-batched-tokens 1024 --capture-max-bytes 4294967296 \
  --max-samples 2
```

这个目标的 HS 宽度为 12,288，pre-HC 宽度为 16,384，BF16 时 32K token 的两类特征约 1.75 GiB，另有 token 和临时张量。轨迹重放默认单请求采集容量为 4 GiB，可用 `--capture-max-bytes` 调整；特征本身仍需对应的主机内存和磁盘空间。准备 token/mask 不会生成 HS/pre-HC。

用户明确授权后，已在服务器 CPU 完成全部 139 条轨迹预处理，输出目录为 `/models/z00829645/speco_trajectory_prepared_20261029T033650Z/`，包含 `trajectories.jsonl` 和 `prepare_report.json`。目标 tokenizer 实测长度 2,608～32,772 token，每条均有有效监督（最少 159 token）；所有 assistant 边界均通过前缀校验。后续抓取建议 `--max-model-len 36864`，为最长轨迹和额外一个采样 token 留足空间。原始目录仅只读访问，尚未运行 NPU 抓取；预处理完成后，10/29 已将最新 SpeCo 代码和采集入口同步至服务器仓库，147 项服务器针对性 CPU 测试通过；后续可以直接执行最小 NPU 冒烟。

当前 16 个 64 GB 逻辑 NPU 的完整 BF16 target 采集使用 `--tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel`，并让全部 16 个逻辑设备可见。此前仅 TP=8、DP=1 的示例在该 checkpoint 上已实测权重构造 OOM。重试配置及采集/训练并行区别见 [最小冒烟指南](deepseek_v4_minimal_smoke.md)。
