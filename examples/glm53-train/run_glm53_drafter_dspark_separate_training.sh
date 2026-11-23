#!/usr/bin/env bash
# Train a GLM-5.3 DSpark drafter against an existing hidden-state vLLM service.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
cd "${repo_root}"

# ===================== 路径和服务 =====================

export MODEL_PATH="/path/to/GLM-5.3"
export TRAIN_FILE="/path/to/train.jsonl"
export DRAFT_CKPTS_DIR="/path/to/glm53_dspark_checkpoints"

# 留空初始化新 drafter；加载已有 drafter 时填写其目录。
export DRAFTER_PATH=""

export TRAIN_DEVICES="0,1,2,3,4,5,6,7"
export TRAIN_GPUS=8

# 只填写节点 0 的 API 地址，不填写 headless 节点 1。
# 服务需接受 MODEL_PATH 作为模型名称，训练机需能读取服务返回的共享文件路径。
export SPECO_VLLM_ENDPOINTS='[http://NODE0_IP:9000/v1]'

export PROJECT_NAME="verl_glm53_dspark"
export EXP_NAME="glm53_dspark_separate_training"
export TRAIN_LOGGERS='["console","tensorboard"]'

# ===================== 训练和保存 =====================

export MAX_STEPS=10000
export BATCH_SIZE_PER_GPU=1

export SAVE_INTERVAL_STEPS=2000
export SAVE_FINAL_CHECKPOINT=true

export LEARNING_RATE=6e-4
export LR_WARMUP_STEPS=100
export LR_SCHEDULER_TYPE="global_cosine"
export LR_DECAY_STEPS="${MAX_STEPS}"
export MIN_LR_RATIO=0.1

export PARAM_OFFLOAD=false
export OPTIMIZER_OFFLOAD=false

# ===================== DSpark 结构 =====================

# 对齐服务抓取的 [4,22,41,59,77,78]：
# 前五个为辅助层，78 为最终 hidden-state 层。
export VLLM_AUX_HIDDEN_STATE_LAYER_IDS='[4,22,41,59,77]'
export DSPARK_NUM_TARGET_LAYERS=5

# Drafter 自身的 Transformer 层数。
export DSPARK_NUM_HIDDEN_LAYERS=5

export DSPARK_BLOCK_SIZE=8
export DSPARK_NUM_ANCHORS=512
export DSPARK_MARKOV_RANK=256

# 0 表示不额外裁剪特征序列。
export DSPARK_MAX_WINDOW=0

# mask token ID 和 FFN 宽度不覆盖，使用代码自动推导。

# ===================== Loss 配置 =====================

export DSPARK_LOSS_MODE="full_vocab"
export DSPARK_LOSS_DECAY_GAMMA=4

export DSPARK_CE_LOSS_ALPHA=0.1
export DSPARK_L1_LOSS_ALPHA=0.45

export DSPARK_DISTRIBUTION_LOSS_IMPL="auto"
export DSPARK_L1_CHUNK_SIZE=0

# Confidence head。
export DSPARK_CONFIDENCE_HEAD_ALPHA=1.0
export DSPARK_CONFIDENCE_HEAD_WITH_MARKOV=true
export DSPARK_CONFIDENCE_LOSS_ALPHA=1.0

# ===================== 数据顺序 =====================

export PRODUCER_SHUFFLE=true
export PRODUCER_SHUFFLE_SEED=42

# ===================== 启动 =====================

bash examples/run_qwen3-8b_drafter_dspark_separate_training.sh \
    speco.draft_training.data_transport=ray_reference \
    actor_rollout_ref.rollout.drafter.training.strategy=fsdp2 \
    "$@"
