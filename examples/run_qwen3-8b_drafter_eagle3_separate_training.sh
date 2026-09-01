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

# Standalone EAGLE3 drafter training.  Start
# examples/run_qwen3-8b_drafter_hidden_state_vllm.sh in another terminal first.
# The target-model vLLM and the drafter trainer can therefore use disjoint GPUs.
#
# EAGLE3 can initialize its drafter structure from the target model config, so
# no pre-initialized drafter directory is required. Set model_path only when
# loading an existing drafter checkpoint/config is desired.
#
# The vLLM hidden-state layer IDs must be EAGLE3_TARGET_LAYER_IDS followed by
# the target model's final layer.  For Qwen3-8B the default is:
#   [1,9,17,25,33,36]
# The EAGLE3 drafter config must have the same number (five) of aux states.

project_name=${PROJECT_NAME:-verl_eagle3_drafter}
exp_name=${EXP_NAME:-qwen3_8b_eagle3_separate_training}

draft_train_gpus_per_node=${TRAIN_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-4B}
TRAIN_FILE=${TRAIN_FILE:-/path/to/data}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/path/to/ckpt}

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE_ENV=${DEVICE_ENV:-CUDA_VISIBLE_DEVICES}
TRAIN_DEVICES=${TRAIN_DEVICES:-6,7}
SPECO_VLLM_ENDPOINTS=${SPECO_VLLM_ENDPOINTS:-'[http://127.0.0.1:8000/v1]'}
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-120}

# These IDs must equal the auxiliary prefix of VLLM_HIDDEN_STATE_LAYER_IDS in
# run_qwen3-8b_drafter_hidden_state_vllm.sh.  Do not include the final layer.
EAGLE3_TARGET_LAYER_IDS=${EAGLE3_TARGET_LAYER_IDS:-'[1,9,17,25,33]'}

# Producer throughput and bounded queues.
VLLM_REQUEST_TIMEOUT=${VLLM_REQUEST_TIMEOUT:-120}
VLLM_MAX_INFLIGHT_REQUESTS=${VLLM_MAX_INFLIGHT_REQUESTS:-16}
VLLM_PER_ENDPOINT_CONCURRENCY=${VLLM_PER_ENDPOINT_CONCURRENCY:-4}
PRODUCER_INPUT_QUEUE_SIZE=${PRODUCER_INPUT_QUEUE_SIZE:-32}
PRODUCER_PUBLISH_QUEUE_SIZE=${PRODUCER_PUBLISH_QUEUE_SIZE:-16}
PRODUCER_MAX_PENDING_SAMPLES=${PRODUCER_MAX_PENDING_SAMPLES:-1024}
PRODUCER_PENDING_POLL_INTERVAL=${PRODUCER_PENDING_POLL_INTERVAL:-0.5}
PRODUCER_MAX_SEQUENCE_LENGTH=${PRODUCER_MAX_SEQUENCE_LENGTH:-8192}
PRODUCER_MAX_FEATURE_LENGTH=${PRODUCER_MAX_FEATURE_LENGTH:-512}
PRODUCER_GENERATION_MAX_TOKENS=${PRODUCER_GENERATION_MAX_TOKENS:-511}

# Standalone EAGLE3 trainer settings.
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_INTERVAL_STEPS=${SAVE_INTERVAL_STEPS:-100}
SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT:-true}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-2}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
LR_DECAY_STEPS=${LR_DECAY_STEPS:-1000}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-true}

if [[ "${MODEL_PATH}" == /path/to/* || "${TRAIN_FILE}" == /path/to/* ]]; then
    echo "Set MODEL_PATH and TRAIN_FILE before starting training." >&2
    exit 2
fi

export "${DEVICE_ENV}=${TRAIN_DEVICES}"
export SPECO_VLLM_ENDPOINTS

# Avoid the launcher's localhost fallback vLLM: this job must consume the
# separately managed hidden-state services, which keep target inference off the
# training devices.
"${PYTHON_BIN}" - "${SPECO_VLLM_ENDPOINTS}" "${VLLM_READY_TIMEOUT_SECONDS}" <<'PY'
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

raw_endpoints = sys.argv[1].strip()
if not (raw_endpoints.startswith("[") and raw_endpoints.endswith("]")):
    raise SystemExit("SPECO_VLLM_ENDPOINTS must use [url0,url1] syntax")
endpoints = [
    item.strip().strip("'\"").rstrip("/")
    for item in raw_endpoints[1:-1].split(",")
    if item.strip()
]
if not endpoints:
    raise SystemExit("SPECO_VLLM_ENDPOINTS must contain at least one URL")
deadline = time.monotonic() + float(sys.argv[2])
pending = set(endpoints)
while pending:
    for endpoint in list(pending):
        try:
            with urlopen(f"{endpoint}/models", timeout=2) as response:
                if 200 <= response.status < 300:
                    print(f"EXTERNAL_VLLM_READY endpoint={endpoint}", flush=True)
                    pending.remove(endpoint)
        except (OSError, URLError):
            pass
    if pending and time.monotonic() >= deadline:
        raise SystemExit(
            "external hidden-state vLLM is not ready at: "
            + ", ".join(sorted(pending))
            + "; start examples/run_qwen3-8b_drafter_hidden_state_vllm.sh first"
        )
    if pending:
        time.sleep(1)
PY

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
    actor_rollout_ref.rollout.drafter.enable=true \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=true \
    actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=EAGLE3 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=3 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=4 \
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
    actor_rollout_ref.rollout.drafter.training.use_logits=false \
    actor_rollout_ref.rollout.drafter.training.eagle3_target_layer_ids=${EAGLE3_TARGET_LAYER_IDS} \
    speco.standalone_tq_producer.target_layer_ids=${EAGLE3_TARGET_LAYER_IDS} \
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
