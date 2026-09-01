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

# Standalone DSpark draft-model training using an already-running hidden-state
# vLLM. Start tools/run_qwen3-8b_drafter_hidden_state_vllm.sh in another terminal
# first. This process owns Ray/TQ, Producer and Consumer, but it must not own
# the target vLLM so inference and training can use different accelerators.

project_name=${PROJECT_NAME:-verl_dspark_drafter}
exp_name=${EXP_NAME:-qwen3_8b_dspark_separate_training}

draft_train_gpus_per_node=${TRAIN_GPUS:-2}

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
# Ordinary verl prompt Parquet is supported; target vLLM generates responses.
TRAIN_FILE=${TRAIN_FILE:-/path/to/train_file.parquet}
# Optional. Leave empty to initialize DSpark from the target-model/config
# fallback; set it only when loading or resuming an existing drafter.
DRAFTER_PATH=${DRAFTER_PATH:-}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/path/to/dspark_draft_checkpoints}

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE_ENV=${DEVICE_ENV:-ASCEND_RT_VISIBLE_DEVICES}
TRAIN_DEVICES=${TRAIN_DEVICES:-2,3}
SPECO_VLLM_ENDPOINTS=${SPECO_VLLM_ENDPOINTS:-'[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]'}
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-120}

# Producer -> vLLM concurrency and bounded queues. MAX_INFLIGHT_REQUESTS is the
# process-wide request limit; PER_ENDPOINT_CONCURRENCY applies independently to
# every URL in SPECO_VLLM_ENDPOINTS.
VLLM_REQUEST_TIMEOUT=${VLLM_REQUEST_TIMEOUT:-120}
VLLM_MAX_INFLIGHT_REQUESTS=${VLLM_MAX_INFLIGHT_REQUESTS:-16}
VLLM_PER_ENDPOINT_CONCURRENCY=${VLLM_PER_ENDPOINT_CONCURRENCY:-4}
PRODUCER_INPUT_QUEUE_SIZE=${PRODUCER_INPUT_QUEUE_SIZE:-32}
PRODUCER_PUBLISH_QUEUE_SIZE=${PRODUCER_PUBLISH_QUEUE_SIZE:-16}
PRODUCER_MAX_PENDING_SAMPLES=${PRODUCER_MAX_PENDING_SAMPLES:-1024}
PRODUCER_PENDING_POLL_INTERVAL=${PRODUCER_PENDING_POLL_INTERVAL:-0.5}
PRODUCER_MAX_SEQUENCE_LENGTH=${PRODUCER_MAX_SEQUENCE_LENGTH:-8192}
PRODUCER_MAX_FEATURE_LENGTH=${PRODUCER_MAX_FEATURE_LENGTH:-512}
PRODUCER_GENERATION_MAX_TOKENS=${PRODUCER_GENERATION_MAX_TOKENS:-512}

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

# DSpark architecture, sampling and losses. TARGET_LAYER_IDS must match the
# auxiliary layers exposed by both hidden-state vLLM services.
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
# The current DSpark trainer rejects nonzero confidence loss because target
# acceptance labels are not part of the standalone feature protocol yet.
DSPARK_CONFIDENCE_LOSS_ALPHA=${DSPARK_CONFIDENCE_LOSS_ALPHA:-0.0}
DSPARK_DEBUG_LOG=${DSPARK_DEBUG_LOG:-false}
DSPARK_DEBUG_LOG_FIRST_N=${DSPARK_DEBUG_LOG_FIRST_N:-2}
DSPARK_DEBUG_LOG_INTERVAL=${DSPARK_DEBUG_LOG_INTERVAL:-100}

export "${DEVICE_ENV}=${TRAIN_DEVICES}"
export SPECO_VLLM_ENDPOINTS

# Fail before entering the unified launcher when the separately managed vLLM
# is absent. Otherwise a localhost endpoint would make the launcher start its
# fallback vLLM inside the training process and on the training devices.
if ! "${PYTHON_BIN}" tools/wait_for_vllm_endpoints.py \
    --endpoints "${SPECO_VLLM_ENDPOINTS}" \
    --timeout-seconds "${VLLM_READY_TIMEOUT_SECONDS}"; then
    echo "Start tools/run_qwen3-8b_drafter_hidden_state_vllm.sh first" >&2
    exit 1
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl_speco.standalone_tq_training_launcher \
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
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=${DSPARK_CONFIDENCE_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log=${DSPARK_DEBUG_LOG} \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_first_n=${DSPARK_DEBUG_LOG_FIRST_N} \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_interval=${DSPARK_DEBUG_LOG_INTERVAL} \
    speco.standalone_tq_producer.request_timeout=${VLLM_REQUEST_TIMEOUT} \
    speco.standalone_tq_producer.max_inflight_requests=${VLLM_MAX_INFLIGHT_REQUESTS} \
    speco.standalone_tq_producer.per_endpoint_concurrency=${VLLM_PER_ENDPOINT_CONCURRENCY} \
    speco.standalone_tq_producer.input_queue_size=${PRODUCER_INPUT_QUEUE_SIZE} \
    speco.standalone_tq_producer.publish_queue_size=${PRODUCER_PUBLISH_QUEUE_SIZE} \
    speco.standalone_tq_producer.max_pending_samples=${PRODUCER_MAX_PENDING_SAMPLES} \
    speco.standalone_tq_producer.pending_poll_interval_seconds=${PRODUCER_PENDING_POLL_INTERVAL} \
    speco.standalone_tq_producer.max_sequence_length=${PRODUCER_MAX_SEQUENCE_LENGTH} \
    speco.standalone_tq_producer.max_feature_length=${PRODUCER_MAX_FEATURE_LENGTH} \
    speco.standalone_tq_producer.generation_max_tokens=${PRODUCER_GENERATION_MAX_TOKENS} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    "$@"
