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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Standalone DSpark draft-model training from a token-replay / feature store
# (the non-TQ data path). Unlike run_qwen3-8b_drafter_dspark_separate_training.sh
# (which streams hidden states through TransferQueue), the Consumer reads the
# store directly and materializes target hidden states on the fly from a
# running target-model vLLM. Start tools/run_qwen3-8b_drafter_hidden_state_vllm.sh
# first, or point feature_store.path at a store that already has hidden states.

project_name=${PROJECT_NAME:-verl_dspark_drafter}
exp_name=${EXP_NAME:-qwen3_8b_dspark_feature_store_training}

draft_train_gpus_per_node=${TRAIN_GPUS:-2}

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
# Token-replay input: JSONL rows with `conversations` (ShareGPT) or already
# tokenized `input_ids`/`loss_mask`.
TRAIN_FILE=${TRAIN_FILE:-/path/to/train_file.jsonl}
# Optional. Leave empty to initialize DSpark from the target-model/config
# fallback; set it only when loading or resuming an existing drafter.
DRAFTER_PATH=${DRAFTER_PATH:-}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/path/to/dspark_draft_checkpoints}

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE_ENV=${DEVICE_ENV:-ASCEND_RT_VISIBLE_DEVICES}
TRAIN_DEVICES=${TRAIN_DEVICES:-2,3}
SPECO_VLLM_ENDPOINTS=${SPECO_VLLM_ENDPOINTS:-'[http://127.0.0.1:8000/v1]'}
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-120}

# Standalone trainer.
MAX_STEPS=${MAX_STEPS:-10}
SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-5}
SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-true}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
LR_DECAY_STEPS=${LR_DECAY_STEPS:-100}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-true}

# Feature store (non-TQ) read-side filters and batching.
FEATURE_STORE_MAX_SEQ_LEN=${FEATURE_STORE_MAX_SEQ_LEN:-512}
FEATURE_STORE_WINDOW_MODE=${FEATURE_STORE_WINDOW_MODE:-loss}
FEATURE_STORE_TRAIN_ON=${FEATURE_STORE_TRAIN_ON:-last_assistant}
FEATURE_STORE_ON_ERROR=${FEATURE_STORE_ON_ERROR:-skip}
FEATURE_STORE_MAX_CONSECUTIVE_ERRORS=${FEATURE_STORE_MAX_CONSECUTIVE_ERRORS:-20}
FEATURE_STORE_MIN_SUPERVISED_TOKENS=${FEATURE_STORE_MIN_SUPERVISED_TOKENS:-1}
FEATURE_STORE_STRICT_TOKEN_ALIGNMENT=${FEATURE_STORE_STRICT_TOKEN_ALIGNMENT:-warn}
GROUP_BY_LENGTH=${GROUP_BY_LENGTH:-false}
GROUP_BY_LENGTH_MEGABATCH=${GROUP_BY_LENGTH_MEGABATCH:-8}

# Target-feature replay (materialize hidden states from the target vLLM).
TARGET_FEATURE_REPLAY_BACKEND=${TARGET_FEATURE_REPLAY_BACKEND:-vllm_file}
VLLM_REQUEST_TIMEOUT=${VLLM_REQUEST_TIMEOUT:-120}
VLLM_MAX_RETRIES=${VLLM_MAX_RETRIES:-3}

# DSpark architecture, sampling and losses. TARGET_LAYER_IDS must match the
# auxiliary layers exposed by the hidden-state vLLM service.
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-7}
DSPARK_NUM_ANCHORS=${DSPARK_NUM_ANCHORS:-32}
DSPARK_MAX_WINDOW=${DSPARK_MAX_WINDOW:-512}
DSPARK_LOSS_MODE=${DSPARK_LOSS_MODE:-full_vocab}
DSPARK_SAMPLED_CE_NEGATIVES=${DSPARK_SAMPLED_CE_NEGATIVES:-0}
DSPARK_LOSS_DECAY_GAMMA=${DSPARK_LOSS_DECAY_GAMMA:-7}
DSPARK_NUM_TARGET_LAYERS=${DSPARK_NUM_TARGET_LAYERS:-5}
DSPARK_NUM_HIDDEN_LAYERS=${DSPARK_NUM_HIDDEN_LAYERS:-5}
DSPARK_TARGET_LAYER_IDS=${DSPARK_TARGET_LAYER_IDS:-'[1,9,17,25,33]'}
DSPARK_MARKOV_RANK=${DSPARK_MARKOV_RANK:-256}
DSPARK_MARKOV_HEAD_TYPE=${DSPARK_MARKOV_HEAD_TYPE:-vanilla}
DSPARK_CE_LOSS_ALPHA=${DSPARK_CE_LOSS_ALPHA:-0.1}
DSPARK_L1_LOSS_ALPHA=${DSPARK_L1_LOSS_ALPHA:-0.45}
DSPARK_L1_CHUNK_SIZE=${DSPARK_L1_CHUNK_SIZE:-0}
# Confidence head training (speculators-style): the head learns each draft
# slot's analytical acceptance rate alpha = sum_v min(p_v, q_v) = 1 - TV from the
# target last hidden state. Set DSPARK_CONFIDENCE_LOSS_ALPHA > 0 to train it; the
# head is created automatically and the target final hidden state is requested.
DSPARK_CONFIDENCE_HEAD_ALPHA=${DSPARK_CONFIDENCE_HEAD_ALPHA:-0.0}
DSPARK_CONFIDENCE_HEAD_WITH_MARKOV=${DSPARK_CONFIDENCE_HEAD_WITH_MARKOV:-true}
DSPARK_CONFIDENCE_LOSS_ALPHA=${DSPARK_CONFIDENCE_LOSS_ALPHA:-0.0}
DSPARK_DEBUG_LOG=${DSPARK_DEBUG_LOG:-false}
DSPARK_DEBUG_LOG_FIRST_N=${DSPARK_DEBUG_LOG_FIRST_N:-2}
DSPARK_DEBUG_LOG_INTERVAL=${DSPARK_DEBUG_LOG_INTERVAL:-100}

export "${DEVICE_ENV}=${TRAIN_DEVICES}"
export SPECO_VLLM_ENDPOINTS

# Fail before entering the launcher when the separately managed vLLM is absent.
# With backend=vllm_file the Consumer needs it to materialize hidden states.
if ! "${PYTHON_BIN}" tools/wait_for_vllm_endpoints.py \
    --endpoints "${SPECO_VLLM_ENDPOINTS}" \
    --timeout-seconds "${VLLM_READY_TIMEOUT_SECONDS}"; then
    echo "Start tools/run_qwen3-8b_drafter_hidden_state_vllm.sh first" >&2
    exit 1
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl_speco.draft_train_launcher \
    speco.draft_training.num_gpus_per_node=${draft_train_gpus_per_node} \
    speco.draft_training.nnodes=1 \
    speco.draft_training.standalone=True \
    data.train_files=${TRAIN_FILE} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    actor_rollout_ref.rollout.drafter.training.mode=offline \
    actor_rollout_ref.rollout.drafter.training.max_steps=${MAX_STEPS} \
    actor_rollout_ref.rollout.drafter.training.save_interval_steps=${SAVE_INTERVAL_STEPS} \
    actor_rollout_ref.rollout.drafter.training.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT} \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.drafter.training.lr=${LEARNING_RATE} \
    actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.rollout.drafter.training.lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    actor_rollout_ref.rollout.drafter.training.lr_decay_steps=${LR_DECAY_STEPS} \
    actor_rollout_ref.rollout.drafter.training.min_lr_ratio=${MIN_LR_RATIO} \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=${DSPARK_BLOCK_SIZE} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=${DSPARK_NUM_ANCHORS} \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=${DSPARK_MAX_WINDOW} \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=${DSPARK_LOSS_MODE} \
    actor_rollout_ref.rollout.drafter.training.dspark_sampled_ce_negatives=${DSPARK_SAMPLED_CE_NEGATIVES} \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=${DSPARK_LOSS_DECAY_GAMMA} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=${DSPARK_NUM_TARGET_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=${DSPARK_NUM_HIDDEN_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.dspark_target_layer_ids=${DSPARK_TARGET_LAYER_IDS} \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=${DSPARK_MARKOV_RANK} \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=${DSPARK_MARKOV_HEAD_TYPE} \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=${DSPARK_CE_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=${DSPARK_L1_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_chunk_size=${DSPARK_L1_CHUNK_SIZE} \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=${DSPARK_CONFIDENCE_HEAD_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_with_markov=${DSPARK_CONFIDENCE_HEAD_WITH_MARKOV} \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=${DSPARK_CONFIDENCE_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log=${DSPARK_DEBUG_LOG} \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_first_n=${DSPARK_DEBUG_LOG_FIRST_N} \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_interval=${DSPARK_DEBUG_LOG_INTERVAL} \
    actor_rollout_ref.rollout.drafter.training.feature_store.type=jsonl_token_replay \
    actor_rollout_ref.rollout.drafter.training.feature_store.path=${TRAIN_FILE} \
    actor_rollout_ref.rollout.drafter.training.feature_store.tokenizer_path=${MODEL_PATH} \
    actor_rollout_ref.rollout.drafter.training.feature_store.max_seq_len=${FEATURE_STORE_MAX_SEQ_LEN} \
    actor_rollout_ref.rollout.drafter.training.feature_store.window_mode=${FEATURE_STORE_WINDOW_MODE} \
    actor_rollout_ref.rollout.drafter.training.feature_store.train_on=${FEATURE_STORE_TRAIN_ON} \
    actor_rollout_ref.rollout.drafter.training.feature_store.on_error=${FEATURE_STORE_ON_ERROR} \
    actor_rollout_ref.rollout.drafter.training.feature_store.max_consecutive_errors=${FEATURE_STORE_MAX_CONSECUTIVE_ERRORS} \
    actor_rollout_ref.rollout.drafter.training.feature_store.min_supervised_tokens=${FEATURE_STORE_MIN_SUPERVISED_TOKENS} \
    actor_rollout_ref.rollout.drafter.training.feature_store.strict_token_alignment=${FEATURE_STORE_STRICT_TOKEN_ALIGNMENT} \
    actor_rollout_ref.rollout.drafter.training.group_by_length=${GROUP_BY_LENGTH} \
    actor_rollout_ref.rollout.drafter.training.group_by_length_megabatch=${GROUP_BY_LENGTH_MEGABATCH} \
    actor_rollout_ref.rollout.drafter.training.target_feature_replay.backend=${TARGET_FEATURE_REPLAY_BACKEND} \
    actor_rollout_ref.rollout.drafter.training.target_feature_replay.vllm_endpoints=${SPECO_VLLM_ENDPOINTS} \
    actor_rollout_ref.rollout.drafter.training.target_feature_replay.vllm_model=${MODEL_PATH} \
    actor_rollout_ref.rollout.drafter.training.target_feature_replay.on_generate=delete \
    actor_rollout_ref.rollout.drafter.training.target_feature_replay.request_timeout=${VLLM_REQUEST_TIMEOUT} \
    actor_rollout_ref.rollout.drafter.training.target_feature_replay.max_retries=${VLLM_MAX_RETRIES} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    "$@"
