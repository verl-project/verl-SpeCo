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
TRAIN_FILE=${TRAIN_FILE:-/path/to/train.parquet}
TEST_FILE=${TEST_FILE:-/path/to/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-/path/to/checkpoint}

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export VERL_ROOT
export PYTHONPATH="${VERL_ROOT}:${SPECO_ROOT}:${PYTHONPATH:-}"

PYTHONUNBUFFERED=1 python3 "${SPECO_ROOT}/run_sft_co_train.py" \
    experiment_type=${EXPERIMENT_TYPE} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=${SPECULATIVE_ALGORITHM} \
    model.path=${MODEL_PATH} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${N_NODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.default_local_dir=${CKPTS_DIR} \
    speco.sft_specific.drafter_train_interval=${DRAFTER_TRAIN_INTERVAL} \
    actor_rollout_ref.rollout.drafter.training.step=${DRAFTER_TRAINING_STEPS} \
    actor_rollout_ref.rollout.drafter.training.total_training_steps=${DRAFTER_TRAINING_STEPS} \
    "$@"
