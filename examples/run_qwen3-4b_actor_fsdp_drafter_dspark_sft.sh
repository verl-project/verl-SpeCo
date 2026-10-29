#!/usr/bin/env bash
# SFT Co-Train: Qwen3-4B + DSPARK drafter (从预训练 drafter 继续训练)
#
# 用法:
#   bash examples/run_qwen3-4b_actor_fsdp_drafter_dspark_sft.sh
#
# 可通过环境变量覆盖默认值，例如:
#   N_GPUS_PER_NODE=4 TOTAL_TRAINING_STEPS=500 bash examples/run_qwen3-4b_actor_fsdp_drafter_dspark_sft.sh
set -euo pipefail
set -x

project_name=${PROJECT_NAME:-verl_sft_cotrain}
exp_name=${EXP_NAME:-qwen3_4b_dspark_sft_cotrain}

# ===== 路径配置 =====
VERL_ROOT=${VERL_ROOT:-/path/to/verl}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-4B}
# 预训练好的 DSPARK drafter（block_size=7）
DRAFTER_MODEL_PATH=${DRAFTER_MODEL_PATH:-/path/to/pretrained_dspark_drafter}
# 训练/验证数据（parquet 格式，含 messages 列）
TRAIN_FILE=${TRAIN_FILE:-/path/to/train.parquet}
TEST_FILE=${TEST_FILE:-/path/to/test.parquet}
# SFT 模型 checkpoint 保存目录（含 hf_model 时会同时导出 HF 格式到子目录 huggingface/）
CKPTS_DIR=${CKPTS_DIR:-/path/to/checkpoint}
# drafter 模型保存目录
DRAFTER_CKPT_DIR=${DRAFTER_CKPT_DIR:-${CKPTS_DIR}/drafter}

# ===== 学习率 =====
# SFT 学习率：默认 2e-5。verl 默认 1e-3 过大，会导致 target 模型 hidden states 分布
# 快速偏移，使预训练 drafter 失效（接受长度从 ~3 骤降到 ~1.0）。
SFT_LR=${SFT_LR:-2e-5}
# drafter 学习率（继续训练用较小 lr）
DRAFTER_LR=${DRAFTER_LR:-1e-5}

# ===== 实验模式 =====
# "cotrain" = SFT + drafter 联合训练; "sft_only" = 仅 SFT（不训练 drafter）
EXPERIMENT_TYPE=${EXPERIMENT_TYPE:-cotrain}

# ===== 训练规模 =====
IGNORE_INPUT_IDS_MISMATCH=${IGNORE_INPUT_IDS_MISMATCH:-true}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
N_NODES=${N_NODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-4096}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1500}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-50}

# ===== Drafter 训练参数 =====
# 每隔 DRAFTER_TRAIN_INTERVAL 步 SFT，采集一次 hidden states 并训练 drafter
DRAFTER_TRAIN_INTERVAL=${DRAFTER_TRAIN_INTERVAL:-5}
# 每次 drafter 训练的步数
DRAFTER_TRAINING_STEPS=${DRAFTER_TRAINING_STEPS:-10}

# ===== DSPARK 架构参数（需与预训练 drafter 一致）=====
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-7}
DSPARK_NUM_ANCHORS=${DSPARK_NUM_ANCHORS:-512}
DSPARK_NUM_TARGET_LAYERS=${DSPARK_NUM_TARGET_LAYERS:-5}
DSPARK_NUM_HIDDEN_LAYERS=${DSPARK_NUM_HIDDEN_LAYERS:-5}
DSPARK_MARKOV_RANK=${DSPARK_MARKOV_RANK:-256}
DSPARK_MARKOV_HEAD_TYPE=${DSPARK_MARKOV_HEAD_TYPE:-vanilla}
DSPARK_LOSS_MODE=${DSPARK_LOSS_MODE:-full_vocab}
DSPARK_CE_LOSS_ALPHA=${DSPARK_CE_LOSS_ALPHA:-0.1}
DSPARK_L1_LOSS_ALPHA=${DSPARK_L1_LOSS_ALPHA:-0.9}
DSPARK_LOSS_DECAY_GAMMA=${DSPARK_LOSS_DECAY_GAMMA:-7}
DSPARK_MAX_WINDOW=${DSPARK_MAX_WINDOW:-512}
DSPARK_MASK_TOKEN_ID=${DSPARK_MASK_TOKEN_ID:-151669}

# ===== 路径校验 =====
if [[ "${MODEL_PATH}" == /path/to/* || "${TRAIN_FILE}" == /path/to/* ]]; then
    echo "Set MODEL_PATH and TRAIN_FILE before starting training." >&2
    exit 2
fi

# 如果设置了预训练 drafter 路径，则通过 actor_rollout_ref.rollout.drafter.model_path 传入
DRAFTER_PATH_ARG=""
if [[ -n "${DRAFTER_MODEL_PATH}" && "${DRAFTER_MODEL_PATH}" != /path/to/* ]]; then
    DRAFTER_PATH_ARG="actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_MODEL_PATH}"
fi

# ===== 配置回显 =====
echo "========================================"
echo "  SFT Co-Train Configuration (DSPARK)"
echo "========================================"
echo "  MODEL_PATH         = ${MODEL_PATH}"
echo "  DRAFTER_MODEL_PATH = ${DRAFTER_MODEL_PATH}"
echo "  SFT_LR             = ${SFT_LR}"
echo "  DRAFTER_LR         = ${DRAFTER_LR}"
echo "  CKPTS_DIR (SFT)    = ${CKPTS_DIR}"
echo "  DRAFTER_CKPT_DIR   = ${DRAFTER_CKPT_DIR}"
echo "  SAVE_FREQ          = ${SAVE_FREQ}"
echo "  N_GPUS_PER_NODE    = ${N_GPUS_PER_NODE}"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export VERL_ROOT
export PYTHONPATH="${VERL_ROOT}:${SPECO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${CKPTS_DIR}"
mkdir -p "${DRAFTER_CKPT_DIR}"

# optim.lr: SFT 学习率
# actor_rollout_ref.rollout.drafter.training.lr: drafter 学习率
# checkpoint.save_contents 含 hf_model：SFT 模型同时保存为 HF 格式
#   （在 ${CKPTS_DIR}/global_step_XXX/huggingface/），便于后续评估脚本直接加载。
PYTHONUNBUFFERED=1 python3 "${SPECO_ROOT}/verl_speco/run_sft_co_train.py" \
    experiment_type=${EXPERIMENT_TYPE} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    model.path=${MODEL_PATH} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    data.ignore_input_ids_mismatch=${IGNORE_INPUT_IDS_MISMATCH} \
    optim.lr=${SFT_LR} \
    actor_rollout_ref.rollout.drafter.training.lr=${DRAFTER_LR} \
    actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.drafter.training.lr_scheduler_type=constant \
    actor_rollout_ref.rollout.drafter.training.use_logits=false \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.default_local_dir=${CKPTS_DIR} \
    checkpoint.save_contents=["model","optimizer","extra","hf_model"] \
    speco.sft_specific.checkpoint_save_dir=${DRAFTER_CKPT_DIR} \
    speco.sft_specific.drafter_train_interval=${DRAFTER_TRAIN_INTERVAL} \
    ${DRAFTER_PATH_ARG} \
    actor_rollout_ref.rollout.drafter.training.step=${DRAFTER_TRAINING_STEPS} \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=${DSPARK_BLOCK_SIZE} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=${DSPARK_NUM_ANCHORS} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=${DSPARK_NUM_TARGET_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=${DSPARK_NUM_HIDDEN_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=${DSPARK_MARKOV_RANK} \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=${DSPARK_MARKOV_HEAD_TYPE} \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=${DSPARK_LOSS_MODE} \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=${DSPARK_CE_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=${DSPARK_L1_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=${DSPARK_LOSS_DECAY_GAMMA} \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=${DSPARK_MAX_WINDOW} \
    actor_rollout_ref.rollout.drafter.training.dspark_mask_token_id=${DSPARK_MASK_TOKEN_ID} \
    "$@"