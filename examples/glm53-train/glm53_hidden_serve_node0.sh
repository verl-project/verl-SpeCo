#!/usr/bin/env bash
# Launch node 0 of the GLM-5.3 hidden-state producer service.
# vLLM version: 0.30.0rc2 (use the same version on both nodes).
set -euo pipefail

# ======================== 本地配置 ========================
# 在这里填写，或通过同名环境变量传入；不包含机器/个人默认路径。
NIC_NAME="${NIC_NAME:-}"
LOCAL_IP="${LOCAL_IP:-}"
NODE0_IP="${NODE0_IP:-${LOCAL_IP}}"
MODEL_PATH="${MODEL_PATH:-}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-}"
# ==========================================================

: "${NIC_NAME:?请配置通信网卡 NIC_NAME}"
: "${LOCAL_IP:?请配置本节点 IP：LOCAL_IP}"
: "${NODE0_IP:?请配置节点 0 IP：NODE0_IP}"
: "${MODEL_PATH:?请配置模型目录 MODEL_PATH}"
: "${HIDDEN_STATES_PATH:?请配置共享 hidden states 目录 HIDDEN_STATES_PATH}"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_ENGINE_READY_TIMEOUT_S=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_USE_V2_MODEL_RUNNER=1

# 用 JSON 编码路径，避免路径中的空格、引号或反斜杠破坏配置。
kv_transfer_config=$(python -c '
import json
import sys

print(json.dumps({
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {"shared_storage_path": sys.argv[1]},
}))
' "${HIDDEN_STATES_PATH}")

exec vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 9000 \
    --api-server-count 1 \
    --safetensors-load-strategy prefetch \
    --data-parallel-size 8 \
    --data-parallel-size-local 4 \
    --data-parallel-address "${NODE0_IP}" \
    --data-parallel-rpc-port 12980 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm \
    --tool-call-parser glm47 \
    --reasoning-parser glm47 \
    --enable-auto-tool-choice \
    --max-num-seqs 64 \
    --max-model-len 16384 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --quantization ascend \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --attention_config.indexer_kv_dtype int8 \
    --additional-config '{"recompute_scheduler_enable": false, "multistream_overlap_shared_expert": true, "enable_dsa_cp": false, "enable_sparse_sfa_c8": false, "enable_sparse_li_c8": true, "c8_enable_reshape_optim": false}' \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [4,22,41,59,77,78]}}}' \
    --kv_transfer_config "${kv_transfer_config}"
