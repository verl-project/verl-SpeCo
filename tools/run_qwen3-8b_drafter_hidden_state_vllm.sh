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

# Start only the target-model vLLM used by the standalone TQ Producer.
# Run this script in its own terminal before starting the training script.
#
# Ascend example:
#   Set DEVICE_ENV=ASCEND_RT_VISIBLE_DEVICES, VLLM_DEVICES and VLLM_TP below,
#   then run: bash tools/run_qwen3-8b_drafter_hidden_state_vllm.sh
# CUDA example:
#   Set DEVICE_ENV=CUDA_VISIBLE_DEVICES, VLLM_DEVICES and VLLM_TP below,
#   then run: bash tools/run_qwen3-8b_drafter_hidden_state_vllm.sh

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
DEVICE_ENV=${DEVICE_ENV:-ASCEND_RT_VISIBLE_DEVICES}
# Devices assigned to target-model vLLM. The script splits this list into
# consecutive groups of VLLM_TP devices and starts one service per group.
VLLM_DEVICES=${VLLM_DEVICES:-0,1,2,3,4,5}
VLLM_TP=${VLLM_TP:-1}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8000}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
# Auxiliary training layers followed by the target model's final hidden-state
# layer. Keep the auxiliary prefix aligned with DSPARK_TARGET_LAYER_IDS in the
# standalone training script. DSpark L1 loss consumes the final entry.
VLLM_HIDDEN_STATE_LAYER_IDS=${VLLM_HIDDEN_STATE_LAYER_IDS:-'[1,9,17,25,33,36]'}
HIDDEN_STATES_DIR=${HIDDEN_STATES_DIR:-/tmp/speco-vllm-hidden-states}

if ! [[ "${VLLM_TP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "VLLM_TP must be a positive integer, got: ${VLLM_TP}" >&2
    exit 2
fi
if ! [[ "${VLLM_BASE_PORT}" =~ ^[0-9]+$ ]]; then
    echo "VLLM_BASE_PORT must be an integer, got: ${VLLM_BASE_PORT}" >&2
    exit 2
fi

visible_devices=${VLLM_DEVICES}

IFS=',' read -r -a DEVICE_IDS <<< "${visible_devices}"
for index in "${!DEVICE_IDS[@]}"; do
    DEVICE_IDS[index]=${DEVICE_IDS[index]//[[:space:]]/}
    if [[ -z "${DEVICE_IDS[index]}" ]]; then
        echo "Visible device list contains an empty item: ${visible_devices}" >&2
        exit 2
    fi
done

device_count=${#DEVICE_IDS[@]}
if (( device_count % VLLM_TP != 0 )); then
    echo "Visible device count (${device_count}) must be divisible by VLLM_TP (${VLLM_TP}): ${visible_devices}" >&2
    exit 2
fi
service_count=$((device_count / VLLM_TP))

SPECULATIVE_CONFIG=$(printf '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":%s}}}' "${VLLM_HIDDEN_STATE_LAYER_IDS}")

start_vllm() {
    local devices=$1
    local port=$2
    local hidden_states_dir=$3
    shift 3
    local kv_transfer_config
    kv_transfer_config=$(printf '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"%s","use_synchronization_lock":true}}' "${hidden_states_dir}")
    env "${DEVICE_ENV}=${devices}" vllm serve "${MODEL_PATH}" \
        --host "${VLLM_HOST}" \
        --port "${port}" \
        --tensor-parallel-size "${VLLM_TP}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --speculative-config "${SPECULATIVE_CONFIG}" \
        --kv-transfer-config "${kv_transfer_config}" \
        --no-enable-chunked-prefill \
        "$@" &
    STARTED_PID=$!
}

PIDS=()
ENDPOINTS=()
cleanup() {
    if (( ${#PIDS[@]} > 0 )); then
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for ((service_index = 0; service_index < service_count; service_index++)); do
    first_device=$((service_index * VLLM_TP))
    service_devices=${DEVICE_IDS[first_device]}
    for ((tp_index = 1; tp_index < VLLM_TP; tp_index++)); do
        service_devices+=",${DEVICE_IDS[first_device + tp_index]}"
    done

    service_port=$((VLLM_BASE_PORT + service_index))
    service_hidden_states_dir="${HIDDEN_STATES_DIR}/service-${service_index}"
    mkdir -p "${service_hidden_states_dir}"
    start_vllm \
        "${service_devices}" \
        "${service_port}" \
        "${service_hidden_states_dir}" \
        "$@"
    PIDS+=("${STARTED_PID}")
    ENDPOINTS+=("http://${VLLM_HOST}:${service_port}/v1")
done

endpoint_list=$(IFS=,; echo "[${ENDPOINTS[*]}]")
echo "VLLM_SERVICES_STARTED count=${service_count} tp=${VLLM_TP} devices=${visible_devices} endpoints=${endpoint_list} pids=${PIDS[*]}"
echo "Use this for standalone training: SPECO_VLLM_ENDPOINTS='${endpoint_list}'"
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
exit "${status}"
