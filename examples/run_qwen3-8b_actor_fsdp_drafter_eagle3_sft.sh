#!/usr/bin/env bash

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
set -x

# SFT Co-Train: train the SFT actor and the drafter in the same process.
# Hidden states are collected from the SFT old-logprob forward pass (no
# vllm/sglang rollout is involved). The drafter is trained periodically on
# the collected hidden states.
#
# Supported algorithms: EAGLE3 (default), DSPARK, DFLASH.
# Set SPECULATIVE_ALGORITHM to switch.

project_name=${PROJECT_NAME:-verl_sft_cotrain}
exp_name=${EXP_NAME:-qwen3_8b_eagle3_sft_cotrain}

# ===== Paths =====
VERL_ROOT=${VERL_ROOT:-/path/to/verl}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
DRAFTER_MODEL_PATH=${DRAFTER_MODEL_PATH:-/path/to/pretrained_drafter}
TRAIN_FILE=${TRAIN_FILE:-/path/to/train.parquet}
TEST_FILE=${TEST_FILE:-/path/to/test.parquet}
# SFT 模型保存目录（verl checkpoint 格式；含 hf_model 时会同时导出 HF 格式到子目录 huggingface/）
CKPTS_DIR=${CKPTS_DIR:-/path/to/checkpoint}
# drafter 模型保存目录
DRAFTER_CKPT_DIR=${DRAFTER_CKPT_DIR:-${CKPTS_DIR}/drafter}

# ===== Learning rates =====
# SFT 学习率：默认 1e-5。verl 默认 1e-3 过大，会导致 target 模型 hidden states 分布
# 快速偏移，使预训练 drafter 失效（接受长度从 ~3 骤降到 ~1.0）。
SFT_LR=${SFT_LR:-1e-5}
# Drafter 学习率：默认 2e-5（与 speco_sft_trainer.yaml 默认值一致）。
DRAFTER_LR=${DRAFTER_LR:-2e-5}

# ===== Experiment type =====
# "cotrain" = SFT + drafter co-train; "sft_only" = SFT only (no drafter)
EXPERIMENT_TYPE=${EXPERIMENT_TYPE:-cotrain}
SPECULATIVE_ALGORITHM=${SPECULATIVE_ALGORITHM:-EAGLE3}

# ===== Training scale =====
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
N_NODES=${N_NODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-4096}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}

# ===== Drafter training interval =====
# Collect hidden states every COLLECT_INTERVAL steps, train drafter every
# TRAIN_INTERVAL steps.
DRAFTER_TRAIN_INTERVAL=${DRAFTER_TRAIN_INTERVAL:-5}
DRAFTER_TRAINING_STEPS=${DRAFTER_TRAINING_STEPS:-10}

if [[ "${MODEL_PATH}" == /path/to/* || "${TRAIN_FILE}" == /path/to/* ]]; then
    echo "Set MODEL_PATH and TRAIN_FILE before starting training." >&2
    exit 2
fi

# 如果设置了预训练 drafter 路径，则通过 actor_rollout_ref.rollout.drafter.model_path 传入
DRAFTER_PATH_ARG=""
if [[ -n "${DRAFTER_MODEL_PATH}" && "${DRAFTER_MODEL_PATH}" != /path/to/* ]]; then
    DRAFTER_PATH_ARG="actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_MODEL_PATH}"
fi

echo "========================================"
echo "  SFT Co-Train Configuration"
echo "========================================"
echo "  MODEL_PATH         = ${MODEL_PATH}"
echo "  DRAFTER_MODEL_PATH = ${DRAFTER_MODEL_PATH}"
echo "  SFT_LR             = ${SFT_LR}"
echo "  DRAFTER_LR         = ${DRAFTER_LR}"
echo "  CKPTS_DIR (SFT)    = ${CKPTS_DIR}"
echo "  DRAFTER_CKPT_DIR   = ${DRAFTER_CKPT_DIR}"
echo "  SAVE_FREQ          = ${SAVE_FREQ}"
echo "========================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export VERL_ROOT
export PYTHONPATH="${VERL_ROOT}:${SPECO_ROOT}:${PYTHONPATH:-}"

# optim.lr: SFT 学习率
# actor_rollout_ref.rollout.drafter.training.lr: drafter 学习率
# checkpoint.save_contents 含 hf_model：SFT 模型同时保存为 HF 格式
#   （在 ${CKPTS_DIR}/global_step_XXX/huggingface/），便于后续评估脚本直接加载。
PYTHONUNBUFFERED=1 python3 "${SPECO_ROOT}/verl_speco/run_sft_co_train.py" \
    experiment_type=${EXPERIMENT_TYPE} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=${SPECULATIVE_ALGORITHM} \
    model.path=${MODEL_PATH} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    optim.lr=${SFT_LR} \
    actor_rollout_ref.rollout.drafter.training.lr=${DRAFTER_LR} \
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
    actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFTER_CKPT_DIR} \
    speco.sft_specific.drafter_train_interval=${DRAFTER_TRAIN_INTERVAL} \
    actor_rollout_ref.rollout.drafter.training.step=${DRAFTER_TRAINING_STEPS} \
    actor_rollout_ref.rollout.drafter.training.total_training_steps=${DRAFTER_TRAINING_STEPS} \
    ${DRAFTER_PATH_ARG} \
    "$@"