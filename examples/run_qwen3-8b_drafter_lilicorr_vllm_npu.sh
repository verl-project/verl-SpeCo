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

# Standalone LiLiCorr (arXiv:2608.20530) drafter training for Qwen3-8B against
# an already-running hidden-state vLLM. Start
# tools/run_qwen3-8b_drafter_hidden_state_vllm.sh in another terminal first.
#
# The recipe follows speculators' lilicorr_qwen3_8b_online.sh: block 16
# (15 speculative tokens), K=8 candidates, 5 draft layers fed by verifier
# layers 2/10/18/26/34, DFlash decay beta 7.0 (w_s = exp(-(s-1)/beta), exactly
# what this overlay's fixed_exp_decay implements), full vocabulary,
# loss = DFlash + 0.25*(CE + 1.0*penalty) per paper Eq. (6) / Table 4 (no
# margin hinge; the paper's "margin" wording is about marginals). The lattice
# head needs the target's final
# hidden state, so hidden collection uses dflash_aux_plus_last.

project_name=${PROJECT_NAME:-verl_lilicorr_drafter}
exp_name=${EXP_NAME:-qwen3_8b_lilicorr_separate_training}
TRAIN_LOGGERS=${TRAIN_LOGGERS:-'["console"]'}

draft_train_gpus_per_node=${TRAIN_GPUS:-2}

DATA_SOURCE=${DATA_SOURCE:-online} # online | saved_hidden
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
TRAIN_FILE=${TRAIN_FILE:-/path/to/train_file.jsonl}
DRAFTER_PATH=${DRAFTER_PATH:-}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/path/to/lilicorr_draft_checkpoints}

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE_ENV=${DEVICE_ENV:-ASCEND_RT_VISIBLE_DEVICES}
TRAIN_DEVICES=${TRAIN_DEVICES:-2,3}
SPECO_VLLM_ENDPOINTS=${SPECO_VLLM_ENDPOINTS:-'[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]'}
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-120}
export SPECO_STANDALONE_LOG_LEVEL=${SPECO_STANDALONE_LOG_LEVEL:-INFO}

VLLM_REQUEST_TIMEOUT=${VLLM_REQUEST_TIMEOUT:-120}
VLLM_MAX_INFLIGHT_REQUESTS=${VLLM_MAX_INFLIGHT_REQUESTS:-64}
VLLM_PER_ENDPOINT_CONCURRENCY=${VLLM_PER_ENDPOINT_CONCURRENCY:-32}
VLLM_SUCCESS_LOG_INTERVAL=${VLLM_SUCCESS_LOG_INTERVAL:-100}
PRODUCER_INPUT_QUEUE_SIZE=${PRODUCER_INPUT_QUEUE_SIZE:-32}
PRODUCER_PUBLISH_QUEUE_SIZE=${PRODUCER_PUBLISH_QUEUE_SIZE:-16}
PRODUCER_PUBLISH_WORKERS=${PRODUCER_PUBLISH_WORKERS:-4}
PRODUCER_PUBLISH_MAX_ATTEMPTS=${PRODUCER_PUBLISH_MAX_ATTEMPTS:-3}
PRODUCER_PUBLISH_RETRY_BACKOFF_SECONDS=${PRODUCER_PUBLISH_RETRY_BACKOFF_SECONDS:-0.5}
PRODUCER_MAX_PENDING_SAMPLES=${PRODUCER_MAX_PENDING_SAMPLES:-1024}
PRODUCER_PENDING_POLL_INTERVAL=${PRODUCER_PENDING_POLL_INTERVAL:-0.5}
PRODUCER_MAX_SEQUENCE_LENGTH=${PRODUCER_MAX_SEQUENCE_LENGTH:-3120}  # paper Table 4
PRODUCER_MAX_FEATURE_LENGTH=${PRODUCER_MAX_FEATURE_LENGTH:-0}
PRODUCER_GENERATION_MAX_TOKENS=${PRODUCER_GENERATION_MAX_TOKENS:-3120}
if [ -z "${PRODUCER_SHUFFLE:-}" ]; then
    case "${TRAIN_FILE,,}" in
        *.parquet|*.pq) PRODUCER_SHUFFLE=false ;;
        *) PRODUCER_SHUFFLE=true ;;
    esac
fi
PRODUCER_SHUFFLE_SEED=${PRODUCER_SHUFFLE_SEED:-42}

SCHEDULER_LOW_WATERMARK_SAMPLES=${SCHEDULER_LOW_WATERMARK_SAMPLES:-$((PRODUCER_MAX_PENDING_SAMPLES / 2))}
SCHEDULER_HIGH_WATERMARK_SAMPLES=${SCHEDULER_HIGH_WATERMARK_SAMPLES:-${PRODUCER_MAX_PENDING_SAMPLES}}

# Paper: 130,968 steps at global batch 64 x seq 3120 over 1.397M conversations
# (6 epochs). Keep a small default for smoke runs and raise it for a real run.
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-2000}
SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-true}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-4}
LEARNING_RATE=${LEARNING_RATE:-6e-4}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-$((MAX_STEPS * 4 / 100))}  # paper: 4% warmup
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}  # paper Table 4
LR_DECAY_STEPS=${LR_DECAY_STEPS:-10000}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}  # paper Table 4
PARAM_OFFLOAD=${PARAM_OFFLOAD:-false}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}

# LiLiCorr geometry/loss. TARGET_LAYER_IDS are vLLM aux output ids; the
# launcher stores them minus one in dflash_config.target_layer_ids
# ([2,10,18,26,34] -> [1,9,17,25,33]).
LILICORR_BLOCK_SIZE=${LILICORR_BLOCK_SIZE:-16}
LILICORR_NUM_ANCHORS=${LILICORR_NUM_ANCHORS:-512}
LILICORR_MAX_WINDOW=${LILICORR_MAX_WINDOW:-3120}
LILICORR_LOSS_DECAY_GAMMA=${LILICORR_LOSS_DECAY_GAMMA:-7.0}
LILICORR_NUM_TARGET_LAYERS=${LILICORR_NUM_TARGET_LAYERS:-5}
LILICORR_NUM_HIDDEN_LAYERS=${LILICORR_NUM_HIDDEN_LAYERS:-5}
VLLM_AUX_HIDDEN_STATE_LAYER_IDS=${VLLM_AUX_HIDDEN_STATE_LAYER_IDS:-'[2,10,18,26,34]'}
LILICORR_MASK_TOKEN_ID=${LILICORR_MASK_TOKEN_ID:-151669}
LILICORR_CANDIDATE_TOPK=${LILICORR_CANDIDATE_TOPK:-8}
LILICORR_HEAD_HIDDEN_SIZE=${LILICORR_HEAD_HIDDEN_SIZE:-1024}
LILICORR_FACTOR_DIM=${LILICORR_FACTOR_DIM:-1024}
LILICORR_NUM_LAYERS=${LILICORR_NUM_LAYERS:-2}
LILICORR_NUM_HEADS=${LILICORR_NUM_HEADS:-8}
LILICORR_MLP_RATIO=${LILICORR_MLP_RATIO:-2.0}
LILICORR_LOGIT_SCALE=${LILICORR_LOGIT_SCALE:-8.0}
LILICORR_VECTOR_EPS=${LILICORR_VECTOR_EPS:-1.0e-4}
LILICORR_W_CE=${LILICORR_W_CE:-0.25}
LILICORR_W_PEN=${LILICORR_W_PEN:-0.25}
LILICORR_LOSS_CHUNK_ANCHORS=${LILICORR_LOSS_CHUNK_ANCHORS:-128}

export "${DEVICE_ENV}=${TRAIN_DEVICES}"
export SPECO_VLLM_ENDPOINTS

if [[ "${DATA_SOURCE}" == "online" ]]; then
    if ! "${PYTHON_BIN}" tools/wait_for_vllm_endpoints.py \
        --endpoints "${SPECO_VLLM_ENDPOINTS}" \
        --timeout-seconds "${VLLM_READY_TIMEOUT_SECONDS}"; then
        echo "Start tools/run_qwen3-8b_drafter_hidden_state_vllm.sh first" >&2
        exit 1
    fi
fi

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl_speco.standalone_tq_training_launcher \
    speco.draft_training.runtime_backend=ray \
    speco.draft_training.num_gpus_per_node=${draft_train_gpus_per_node} \
    speco.draft_training.nnodes=1 \
    speco.draft_training.standalone=True \
    speco.draft_training.data_source=${DATA_SOURCE} \
    speco.draft_training.scheduler.low_watermark_samples=${SCHEDULER_LOW_WATERMARK_SAMPLES} \
    speco.draft_training.scheduler.high_watermark_samples=${SCHEDULER_HIGH_WATERMARK_SAMPLES} \
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
    actor_rollout_ref.rollout.drafter.speculative_algorithm=LILICORR \
    actor_rollout_ref.rollout.drafter.training.mode=offline \
    actor_rollout_ref.rollout.drafter.training.strategy=ddp \
    actor_rollout_ref.rollout.drafter.training.max_steps=${MAX_STEPS} \
    actor_rollout_ref.rollout.drafter.training.save_interval_steps=${SAVE_INTERVAL_STEPS} \
    actor_rollout_ref.rollout.drafter.training.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT} \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.drafter.training.packing.enable=true \
    actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_batch_workers=12 \
    actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_prefetch_batches=12 \
    'actor_rollout_ref.rollout.drafter.training.adamw_betas=[0.9,0.999]' \
    actor_rollout_ref.rollout.drafter.training.lr=${LEARNING_RATE} \
    actor_rollout_ref.rollout.drafter.training.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.rollout.drafter.training.lr_scheduler_type=${LR_SCHEDULER_TYPE} \
    actor_rollout_ref.rollout.drafter.training.lr_decay_steps=${LR_DECAY_STEPS} \
    actor_rollout_ref.rollout.drafter.training.min_lr_ratio=${MIN_LR_RATIO} \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.lilicorr_block_size=${LILICORR_BLOCK_SIZE} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_num_anchors=${LILICORR_NUM_ANCHORS} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_max_window=${LILICORR_MAX_WINDOW} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_loss_decay_gamma=${LILICORR_LOSS_DECAY_GAMMA} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_num_target_layers=${LILICORR_NUM_TARGET_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_num_hidden_layers=${LILICORR_NUM_HIDDEN_LAYERS} \
    speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids=${VLLM_AUX_HIDDEN_STATE_LAYER_IDS} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_mask_token_id=${LILICORR_MASK_TOKEN_ID} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_candidate_topk=${LILICORR_CANDIDATE_TOPK} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_head_hidden_size=${LILICORR_HEAD_HIDDEN_SIZE} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_factor_dim=${LILICORR_FACTOR_DIM} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_num_layers=${LILICORR_NUM_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_num_heads=${LILICORR_NUM_HEADS} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_mlp_ratio=${LILICORR_MLP_RATIO} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_logit_scale=${LILICORR_LOGIT_SCALE} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_vector_eps=${LILICORR_VECTOR_EPS} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_w_ce=${LILICORR_W_CE} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_w_pen=${LILICORR_W_PEN} \
    actor_rollout_ref.rollout.drafter.training.lilicorr_loss_chunk_anchors=${LILICORR_LOSS_CHUNK_ANCHORS} \
    speco.standalone_tq_producer.request_timeout=${VLLM_REQUEST_TIMEOUT} \
    speco.standalone_tq_producer.max_inflight_requests=${VLLM_MAX_INFLIGHT_REQUESTS} \
    speco.standalone_tq_producer.per_endpoint_concurrency=${VLLM_PER_ENDPOINT_CONCURRENCY} \
    speco.standalone_tq_producer.vllm_success_log_interval=${VLLM_SUCCESS_LOG_INTERVAL} \
    speco.standalone_tq_producer.input_queue_size=${PRODUCER_INPUT_QUEUE_SIZE} \
    speco.standalone_tq_producer.publish_queue_size=${PRODUCER_PUBLISH_QUEUE_SIZE} \
    speco.standalone_tq_producer.publish_workers=${PRODUCER_PUBLISH_WORKERS} \
    speco.standalone_tq_producer.publish_max_attempts=${PRODUCER_PUBLISH_MAX_ATTEMPTS} \
    speco.standalone_tq_producer.publish_retry_backoff_seconds=${PRODUCER_PUBLISH_RETRY_BACKOFF_SECONDS} \
    speco.standalone_tq_producer.max_pending_samples=${PRODUCER_MAX_PENDING_SAMPLES} \
    speco.standalone_tq_producer.pending_poll_interval_seconds=${PRODUCER_PENDING_POLL_INTERVAL} \
    speco.standalone_tq_producer.max_sequence_length=${PRODUCER_MAX_SEQUENCE_LENGTH} \
    speco.standalone_tq_producer.max_feature_length=${PRODUCER_MAX_FEATURE_LENGTH} \
    speco.standalone_tq_producer.generation_max_tokens=${PRODUCER_GENERATION_MAX_TOKENS} \
    speco.standalone_tq_producer.shuffle=${PRODUCER_SHUFFLE} \
    speco.standalone_tq_producer.shuffle_seed=${PRODUCER_SHUFFLE_SEED} \
    trainer.project_name=${project_name} \
    trainer.logger="${TRAIN_LOGGERS}" \
    trainer.experiment_name=${exp_name} \
    "$@"
