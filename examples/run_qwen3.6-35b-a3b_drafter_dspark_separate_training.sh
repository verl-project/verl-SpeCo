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

# Standalone DSpark draft-model training for Qwen3.6-35B-A3B (MoE, 40 layers)
# using the TransferQueue data path. Best-measured card split (uses all 8 cards):
# the target runs as two DP replicas (DP=2 x TP=2) on cards 0-3; the drafter
# trains on cards 4-7.
#
# Hyperparameters mirror the released RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
# config as used by speculators/examples/train/dspark_qwen3_6_35B_redhat.sh:
#   aux layers [2,10,20,30,37], block_size=8, max_anchors=512, markov_rank=256,
#   lr=3e-4, ce=0.1, tv/l1=0.9, mask_token_id=248077.
#
# This script starts two target-model hidden-state vLLM replicas (DP=2 x TP=2)
# on cards 0-1 and 2-3, each exposing the DSpark aux layers plus the verifier's
# final layer (40) on its own endpoint (ports 8000/8001), then trains the drafter
# on cards 4-7. Graph mode is on by default (no --enforce-eager). Chunked prefill
# is enabled for the 35B prefill. Set SPECO_VLLM_ENDPOINTS to reuse an
# already-running target service instead of starting one.
#
# Measured (300-step A/B on 8x910B3): the end-to-end rate is ~15-16 samples/s and
# is set by the producer/engine host-side supply, not by the card split, batch
# size, TQ backend, hidden-state transport (file vs Mooncake handle), consumer
# prefetch, or max_pending_samples. This script keeps the best-measured defaults
# (4 engine + 4 training cards, bs=8, file hidden-state handoff, TQ MooncakeStore).
#
# VLLM_AUX_HIDDEN_STATE_LAYER_IDS must match the aux layers served by vLLM. The
# training-side list holds only the aux layers; the appended final layer is used
# as the verifier's last hidden state.

project_name=${PROJECT_NAME:-verl_dspark_drafter}
exp_name=${EXP_NAME:-qwen3_6_35b_a3b_dspark_separate_training}

draft_train_gpus_per_node=${TRAIN_GPUS:-4}

# Save all stdout/stderr and TensorBoard event files under .logs/.
LOG_DIR=${LOG_DIR:-${repo_root}/.logs}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}"
export TENSORBOARD_DIR
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

MODEL_PATH=${MODEL_PATH:-/home/model/Qwen/Qwen3.6-35B-A3B}
# Ordinary verl prompt Parquet is supported; target vLLM generates responses.
TRAIN_FILE=${TRAIN_FILE:-/home/dataset/open-perfectblend/data/train-00000-of-00006.parquet}
# Optional. Leave empty to initialize DSpark from the target-model/config
# fallback; set it to the RedHatAI speculator snapshot to use its 5-layer backbone.
DRAFTER_PATH=${DRAFTER_PATH:-}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/home/dataset/dspark_draft_checkpoints}

# Target-model hidden-state vLLM: one replica per VLLM_DEVICES_* entry, each
# TP=VLLM_TP cards and its own port. A third replica is optional (DP=3 x TP=2);
# TRAIN_DEVICES below must not overlap them.
VLLM_DEVICES_0=${VLLM_DEVICES_0:-0,1}
VLLM_DEVICES_1=${VLLM_DEVICES_1:-2,3}
VLLM_DEVICES_2=${VLLM_DEVICES_2:-}
VLLM_TP=${VLLM_TP:-2}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8000}
VLLM_BASE_PORT_1=${VLLM_BASE_PORT_1:-8001}
VLLM_BASE_PORT_2=${VLLM_BASE_PORT_2:-8002}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
# Frontend API servers for the target vLLM (raises HTTP concurrency; matches the
# reference DSpark setup). Set to 0 to omit. Renderer workers must stay at 1 for
# this multimodal model: >1 conflicts with the multimodal processor cache.
VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT:-9}
VLLM_RENDERER_NUM_WORKERS=${VLLM_RENDERER_NUM_WORKERS:-1}
# Auxiliary training layers followed by the target's final hidden-state layer.
VLLM_HIDDEN_STATE_LAYER_IDS=${VLLM_HIDDEN_STATE_LAYER_IDS:-'[2,10,20,30,37,40]'}
HIDDEN_STATES_DIR=${HIDDEN_STATES_DIR:-/dev/shm/speco-vllm-hidden-states}
# Hidden-state transfer backend: "file" (safetensors handoff under
# HIDDEN_STATES_DIR) or "mooncake" (handle-based store read via hs_connectors).
# The unified launcher builds the Producer-side store from the same setting.
HIDDEN_STATES_STORE=${HIDDEN_STATES_STORE:-file}
export SPECO_VLLM_HIDDEN_STATES_STORE="${HIDDEN_STATES_STORE}"
# Mooncake settings fall back to the TQ Mooncake variables. The launch probes
# the master before starting vLLM; set ..._SKIP_PRECHECK=true to bypass.
SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER=${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER:-${SPECO_TQ_MOONCAKE_MASTER:-127.0.0.1:50051}}
SPECO_VLLM_HIDDEN_STATES_MOONCAKE_PROTOCOL=${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_PROTOCOL:-${SPECO_TQ_MOONCAKE_PROTOCOL:-tcp}}
export SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER
export SPECO_VLLM_HIDDEN_STATES_MOONCAKE_PROTOCOL

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE_ENV=${DEVICE_ENV:-ASCEND_RT_VISIBLE_DEVICES}
# Draft training; the target replicas own the other cards.
TRAIN_DEVICES=${TRAIN_DEVICES:-4,5,6,7}
_default_vllm_endpoints="[http://${VLLM_HOST}:${VLLM_BASE_PORT}/v1,http://${VLLM_HOST}:${VLLM_BASE_PORT_1}/v1"
if [[ -n "${VLLM_DEVICES_2}" ]]; then
    _default_vllm_endpoints+=",http://${VLLM_HOST}:${VLLM_BASE_PORT_2}/v1"
fi
_default_vllm_endpoints+="]"
SPECO_VLLM_ENDPOINTS=${SPECO_VLLM_ENDPOINTS:-"${_default_vllm_endpoints}"}
# A 35B MoE target takes longer to load than an 8B model: weight loading alone
# takes ~7 min on FUSE.OBSFS and the multimodal warmup adds ~40 s, so leave
# generous headroom over the previous 600 s.
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-1800}
# Reuse an already-running target vLLM instead of launching new replicas: skips
# the vLLM start below and only waits for the endpoints. Set to true and point
# SPECO_VLLM_ENDPOINTS at the running service (e.g. one left up on cards 0-3).
REUSE_EXTERNAL_VLLM=${REUSE_EXTERNAL_VLLM:-false}

# Producer -> vLLM concurrency and bounded queues. MAX_INFLIGHT_REQUESTS is the
# process-wide request limit and the number of producer request workers;
# PER_ENDPOINT_CONCURRENCY applies independently to every URL in
# SPECO_VLLM_ENDPOINTS. The target engine has plenty of spare KV cache
# (a few % used), so run enough concurrent requests to actually batch decode.
VLLM_REQUEST_TIMEOUT=${VLLM_REQUEST_TIMEOUT:-900}
VLLM_MAX_INFLIGHT_REQUESTS=${VLLM_MAX_INFLIGHT_REQUESTS:-64}
VLLM_PER_ENDPOINT_CONCURRENCY=${VLLM_PER_ENDPOINT_CONCURRENCY:-16}
PRODUCER_INPUT_QUEUE_SIZE=${PRODUCER_INPUT_QUEUE_SIZE:-64}
PRODUCER_PUBLISH_QUEUE_SIZE=${PRODUCER_PUBLISH_QUEUE_SIZE:-64}
PRODUCER_MAX_PENDING_SAMPLES=${PRODUCER_MAX_PENDING_SAMPLES:-1024}
PRODUCER_PENDING_POLL_INTERVAL=${PRODUCER_PENDING_POLL_INTERVAL:-0.5}
PRODUCER_MAX_SEQUENCE_LENGTH=${PRODUCER_MAX_SEQUENCE_LENGTH:-8192}
# Cap the target context; the +1 leaves room for the appended verifier token.
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$((PRODUCER_MAX_SEQUENCE_LENGTH + 1))}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
# Graph mode: allow torch.compile / NPU graphs instead of eager. Graph capture
# makes the engine take much longer to become ready, so the engine-ready timeout
# below is raised accordingly.
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}
# vLLM aborts the API server if the engine core is not ready within this window;
# graph capture on this Ascend build needs well over the 600 s default.
VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-2400}
export VLLM_ENGINE_READY_TIMEOUT_S
PRODUCER_MAX_FEATURE_LENGTH=${PRODUCER_MAX_FEATURE_LENGTH:-512}
PRODUCER_GENERATION_MAX_TOKENS=${PRODUCER_GENERATION_MAX_TOKENS:-512}

# Standalone trainer.
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-10000}
SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-true}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-8}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-2}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
# Decay over the full run by default; otherwise the LR reaches MIN_LR_RATIO
# long before max_steps and the remaining steps train nothing.
LR_DECAY_STEPS=${LR_DECAY_STEPS:-${MAX_STEPS}}
MIN_LR_RATIO=${MIN_LR_RATIO:-1e-7}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-true}
# Drafter parallelism. DDP (replicated full params + gradient all-reduce) is
# fastest for the small drafter; set fsdp2 to shard instead.
DRAFTER_STRATEGY=${DRAFTER_STRATEGY:-ddp}
# FSDP2-only shard dimension: 1 replicates, >1 shards over that many ranks.
FSDP_SHARD_SIZE=${FSDP_SHARD_SIZE:-1}

# Matches the speculators RedHat recipe: Muon on 2D hidden weights, AdamW else.
OPTIMIZER=${OPTIMIZER:-muon}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
# null lets the trainer derive muon_lr = 10 * lr.
MUON_LR=${MUON_LR:-null}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_NESTEROV=${MUON_NESTEROV:-true}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.1}
MUON_NS_STEPS=${MUON_NS_STEPS:-5}
MUON_ADJUST_LR_FN=${MUON_ADJUST_LR_FN:-match_rms_adamw}

# DSpark architecture, sampling and losses. VLLM auxiliary IDs must match the
# auxiliary layers exposed by the hidden-state vLLM service. For Qwen3.6-35B-A3B
# (40 layers) this is the RedHatAI 5-aux-layer recipe, with final layer 40
# appended by the vLLM service.
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-8}
DSPARK_NUM_ANCHORS=${DSPARK_NUM_ANCHORS:-512}
DSPARK_MAX_WINDOW=${DSPARK_MAX_WINDOW:-512}
DSPARK_LOSS_MODE=${DSPARK_LOSS_MODE:-full_vocab}
DSPARK_SAMPLED_CE_NEGATIVES=${DSPARK_SAMPLED_CE_NEGATIVES:-0}
DSPARK_LOSS_DECAY_GAMMA=${DSPARK_LOSS_DECAY_GAMMA:-7}
DSPARK_NUM_TARGET_LAYERS=${DSPARK_NUM_TARGET_LAYERS:-5}
DSPARK_NUM_HIDDEN_LAYERS=${DSPARK_NUM_HIDDEN_LAYERS:-5}
VLLM_AUX_HIDDEN_STATE_LAYER_IDS=${VLLM_AUX_HIDDEN_STATE_LAYER_IDS:-'[2,10,20,30,37]'}
# Draft FFN width for the target-derived fallback backbone.
DSPARK_INTERMEDIATE_SIZE=${DSPARK_INTERMEDIATE_SIZE:-6144}
# Sliding-window attention for all draft layers (null = full attention).
DSPARK_SLIDING_WINDOW=${DSPARK_SLIDING_WINDOW:-2048}
# Document-aware packing: concatenate the batch's samples into one row and
# isolate each document via document_ids. Disabled by default.
PACKING_ENABLE=${PACKING_ENABLE:-true}
PACKING_MAX_LEN=${PACKING_MAX_LEN:-0}
DSPARK_MASK_TOKEN_ID=${DSPARK_MASK_TOKEN_ID:-248077}
DSPARK_MARKOV_RANK=${DSPARK_MARKOV_RANK:-256}
DSPARK_MARKOV_HEAD_TYPE=${DSPARK_MARKOV_HEAD_TYPE:-vanilla}
DSPARK_CE_LOSS_ALPHA=${DSPARK_CE_LOSS_ALPHA:-0.1}
DSPARK_L1_LOSS_ALPHA=${DSPARK_L1_LOSS_ALPHA:-0.9}
DSPARK_L1_CHUNK_SIZE=${DSPARK_L1_CHUNK_SIZE:-512}
# Fused distribution loss backend: auto | fused | eager (see tools/benchmark_fused_losses.py).
DSPARK_DISTRIBUTION_LOSS_IMPL=${DSPARK_DISTRIBUTION_LOSS_IMPL:-fused}
# Confidence head training (speculators-style): the head learns each draft
# slot's analytical acceptance rate alpha = sum_v min(p_v, q_v) = 1 - TV from the
# target last hidden state. Set DSPARK_CONFIDENCE_LOSS_ALPHA > 0 to train it; the
# head is created automatically and the target final hidden state is requested.
# Keep it at 0 for fixed-length verification.
DSPARK_CONFIDENCE_HEAD_ALPHA=${DSPARK_CONFIDENCE_HEAD_ALPHA:-1.0}
DSPARK_CONFIDENCE_HEAD_WITH_MARKOV=${DSPARK_CONFIDENCE_HEAD_WITH_MARKOV:-true}
DSPARK_CONFIDENCE_LOSS_ALPHA=${DSPARK_CONFIDENCE_LOSS_ALPHA:-1.0}
DSPARK_DEBUG_LOG=${DSPARK_DEBUG_LOG:-false}
DSPARK_DEBUG_LOG_FIRST_N=${DSPARK_DEBUG_LOG_FIRST_N:-2}
DSPARK_DEBUG_LOG_INTERVAL=${DSPARK_DEBUG_LOG_INTERVAL:-100}

# Start the target-model hidden-state vLLM replicas (DP=2 x TP=2) in the
# background. VLLM_HIDDEN_STATE_LAYER_IDS ends with the target model's final
# layer, which the DSpark L1/confidence losses consume as the verifier state.
SPECULATIVE_CONFIG=$(printf '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":%s}}}' "${VLLM_HIDDEN_STATE_LAYER_IDS}")

VLLM_PIDS=()
cleanup_vllm() {
    for pid in "${VLLM_PIDS[@]}"; do
        if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            continue
        fi
        # vLLM is started via `setsid`, so it leads its own process group;
        # signalling the negative PID tears down the engine core and all
        # tensor-parallel workers together. Escalate to SIGKILL so a
        # SIGTERM-ignoring server cannot leave this script blocked forever in
        # `wait`.
        kill -TERM -"${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL -"${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        fi
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup_vllm EXIT INT TERM

if [[ "${REUSE_EXTERNAL_VLLM}" == "true" ]]; then
    echo "REUSE_EXTERNAL_VLLM=true: reusing ${SPECO_VLLM_ENDPOINTS}; not starting vLLM"
else
vllm_extra_args=()
if [[ "${VLLM_ENFORCE_EAGER}" == "true" ]]; then
    vllm_extra_args+=(--enforce-eager)
fi
if [[ -n "${VLLM_API_SERVER_COUNT}" && "${VLLM_API_SERVER_COUNT}" != "0" ]]; then
    vllm_extra_args+=(--api-server-count "${VLLM_API_SERVER_COUNT}")
fi
if [[ -n "${VLLM_RENDERER_NUM_WORKERS}" && "${VLLM_RENDERER_NUM_WORKERS}" != "0" ]]; then
    vllm_extra_args+=(--renderer-num-workers "${VLLM_RENDERER_NUM_WORKERS}")
fi

VLLM_REPLICA_DEVICES=("${VLLM_DEVICES_0}" "${VLLM_DEVICES_1}")
VLLM_REPLICA_PORTS=("${VLLM_BASE_PORT}" "${VLLM_BASE_PORT_1}")
if [[ -n "${VLLM_DEVICES_2}" ]]; then
    VLLM_REPLICA_DEVICES+=("${VLLM_DEVICES_2}")
    VLLM_REPLICA_PORTS+=("${VLLM_BASE_PORT_2}")
fi
for replica in "${!VLLM_REPLICA_DEVICES[@]}"; do
    replica_devices="${VLLM_REPLICA_DEVICES[${replica}]}"
    replica_port="${VLLM_REPLICA_PORTS[${replica}]}"
    # Each replica needs its own hidden-state handoff directory.
    service_hidden_states_dir="${HIDDEN_STATES_DIR}/service-${replica}"
    mkdir -p "${service_hidden_states_dir}"
    if [[ "${HIDDEN_STATES_STORE}" == "mooncake" ]]; then
        KV_TRANSFER_CONFIG=$(printf '{"kv_connector":"SpecoMooncakeHiddenStatesConnector","kv_connector_module_path":"verl_speco.mooncake_hidden_states_connector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake":{"local_hostname":"%s","metadata_server":"%s","master_server_address":"%s","global_segment_size":%s,"local_buffer_size":%s,"protocol":"%s","device_name":"%s","num_writer_threads":%s}}}' \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_LOCAL_HOSTNAME:-$(hostname -I 2>/dev/null | awk '{print $1}')}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_GLOBAL_SEGMENT_BYTES:-4294967296}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_LOCAL_BUFFER_BYTES:-2147483648}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_PROTOCOL}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_DEVICE_NAME:-${SPECO_TQ_MOONCAKE_DEVICE_NAME:-}}" \
            "${SPECO_VLLM_HIDDEN_STATES_MOONCAKE_WRITER_THREADS:-8}")
    else
        KV_TRANSFER_CONFIG=$(printf '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"%s","use_synchronization_lock":true}}' "${service_hidden_states_dir}")
    fi
    setsid env "${DEVICE_ENV}=${replica_devices}" vllm serve "${MODEL_PATH}" \
        --host "${VLLM_HOST}" \
        --port "${replica_port}" \
        --tensor-parallel-size "${VLLM_TP}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
        "${vllm_extra_args[@]}" \
        --speculative-config "${SPECULATIVE_CONFIG}" \
        --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
        --enable-chunked-prefill \
        &
    VLLM_PIDS+=($!)
    echo "HIDDEN_STATE_VLLM_STARTED replica=${replica} pid=${VLLM_PIDS[${replica}]} devices=${replica_devices} tp=${VLLM_TP} endpoint=http://${VLLM_HOST}:${replica_port}/v1"
done
fi

export "${DEVICE_ENV}=${TRAIN_DEVICES}"
export SPECO_VLLM_ENDPOINTS

# Reduce NPU allocator fragmentation for the drafter training workers. The 35B
# target vLLM sets this for itself, but the training process does not inherit it.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# Requires a mooncake_master on SPECO_TQ_MOONCAKE_MASTER; set
# TQ_STORAGE_BACKEND=SimpleStorage to use the in-memory backend instead.
TQ_STORAGE_BACKEND=${TQ_STORAGE_BACKEND:-MooncakeStore}
SPECO_TQ_MOONCAKE_MASTER=${SPECO_TQ_MOONCAKE_MASTER:-127.0.0.1:50051}
export SPECO_TQ_STORAGE_BACKEND="${TQ_STORAGE_BACKEND}"
export SPECO_TQ_MOONCAKE_MASTER
# Keep the producer's Mooncake segment mounted until a slow consumer drains it.
export SPECO_TQ_PRODUCER_LINGER_SECONDS=${SPECO_TQ_PRODUCER_LINGER_SECONDS:-600}

# Wait for the target vLLM before entering the unified launcher. Otherwise a
# localhost endpoint would make the launcher start its fallback vLLM inside the
# training process and on the training devices.
if ! "${PYTHON_BIN}" tools/wait_for_vllm_endpoints.py \
    --endpoints "${SPECO_VLLM_ENDPOINTS}" \
    --timeout-seconds "${VLLM_READY_TIMEOUT_SECONDS}"; then
    echo "Timed out waiting for the hidden-state vLLM replicas on cards 2-5 (DP=2 x TP=2)" >&2
    exit 1
fi

# TransferQueue runtime backend. "subprocess" runs the producer/owner/consumer
# as plain processes that inherit the driver's ASCEND_RT_VISIBLE_DEVICES, so the
# CPU-only producer keeps a valid NPU context for Mooncake's Ascend transport.
# With the default "ray" backend, Ray clears that variable for the CPU-only
# producer actor and Mooncake setup fails (error code -1). Override with
# RUNTIME_BACKEND=ray only once the producer actor device env is handled.
RUNTIME_BACKEND=${RUNTIME_BACKEND:-subprocess}

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl_speco.standalone_tq_training_launcher \
    speco.draft_training.num_gpus_per_node=${draft_train_gpus_per_node} \
    speco.draft_training.nnodes=1 \
    speco.draft_training.standalone=True \
    speco.draft_training.runtime_backend=${RUNTIME_BACKEND} \
    data.train_files=${TRAIN_FILE} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=${DRAFTER_STRATEGY} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.drafter.training.fsdp_shard_size=${FSDP_SHARD_SIZE} \
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
    actor_rollout_ref.rollout.drafter.training.optimizer=${OPTIMIZER} \
    actor_rollout_ref.rollout.drafter.training.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.rollout.drafter.training.muon_lr=${MUON_LR} \
    actor_rollout_ref.rollout.drafter.training.muon_momentum=${MUON_MOMENTUM} \
    actor_rollout_ref.rollout.drafter.training.muon_nesterov=${MUON_NESTEROV} \
    actor_rollout_ref.rollout.drafter.training.muon_weight_decay=${MUON_WEIGHT_DECAY} \
    actor_rollout_ref.rollout.drafter.training.muon_ns_steps=${MUON_NS_STEPS} \
    actor_rollout_ref.rollout.drafter.training.muon_adjust_lr_fn=${MUON_ADJUST_LR_FN} \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=${DSPARK_BLOCK_SIZE} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=${DSPARK_NUM_ANCHORS} \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=${DSPARK_MAX_WINDOW} \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=${DSPARK_LOSS_MODE} \
    actor_rollout_ref.rollout.drafter.training.dspark_sampled_ce_negatives=${DSPARK_SAMPLED_CE_NEGATIVES} \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=${DSPARK_LOSS_DECAY_GAMMA} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=${DSPARK_NUM_TARGET_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=${DSPARK_NUM_HIDDEN_LAYERS} \
    actor_rollout_ref.rollout.drafter.training.dspark_intermediate_size=${DSPARK_INTERMEDIATE_SIZE} \
    speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids=${VLLM_AUX_HIDDEN_STATE_LAYER_IDS} \
    actor_rollout_ref.rollout.drafter.training.dspark_sliding_window=${DSPARK_SLIDING_WINDOW} \
    actor_rollout_ref.rollout.drafter.training.packing.enable=${PACKING_ENABLE} \
    actor_rollout_ref.rollout.drafter.training.packing.max_packed_len=${PACKING_MAX_LEN} \
    actor_rollout_ref.rollout.drafter.training.dspark_mask_token_id=${DSPARK_MASK_TOKEN_ID} \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=${DSPARK_MARKOV_RANK} \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=${DSPARK_MARKOV_HEAD_TYPE} \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=${DSPARK_CE_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=${DSPARK_L1_LOSS_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_chunk_size=${DSPARK_L1_CHUNK_SIZE} \
    actor_rollout_ref.rollout.drafter.training.dspark_distribution_loss_impl=${DSPARK_DISTRIBUTION_LOSS_IMPL} \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=${DSPARK_CONFIDENCE_HEAD_ALPHA} \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_with_markov=${DSPARK_CONFIDENCE_HEAD_WITH_MARKOV} \
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
    trainer.logger='["console","tensorboard"]' \
    "$@"
