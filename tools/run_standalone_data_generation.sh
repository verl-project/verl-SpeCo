#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
set -euo pipefail

# Edit this section, then run: bash tools/run_standalone_data_generation.sh
# vLLM services must already be running. No training or vLLM process is launched.
# Every setting can also be overridden through the environment.

# ---------- Task and files ----------
PYTHON_BIN=${PYTHON_BIN:-python3}
# Same model path as tools/run_qwen3-8b_drafter_hidden_state_vllm.sh and the
# Qwen3-8B standalone DSpark training example.
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
TASK=${TASK:-responses} # responses | hidden
INPUT_PATH=${INPUT_PATH:-/data/source.jsonl}
# responses: a JSONL file; hidden: a feature-store directory.
OUTPUT_PATH=${OUTPUT_PATH:-/data/responses.jsonl}
RESUME=${RESUME:-true}
ON_ERROR=${ON_ERROR:-raise} # raise | skip (write failed requests to errors.jsonl)
MAX_SAMPLES=${MAX_SAMPLES:-0} # newly processed records; 0 = all remaining
LOG_INTERVAL=${LOG_INTERVAL:-20} # timing report interval; 1 = every record
LOG_INTERVAL_SECONDS=${LOG_INTERVAL_SECONDS:-10} # status heartbeat while requests/files are pending

# ---------- External vLLM services ----------
# YAML/Hydra list. All addresses must serve the same model and chat template.
SPECO_VLLM_ENDPOINTS=${SPECO_VLLM_ENDPOINTS:-'[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]'}
VLLM_MODEL=${VLLM_MODEL:-${MODEL_PATH}} # served model name
VLLM_API_KEY=${VLLM_API_KEY:-EMPTY} # responses only; hidden uses Producer auth
VLLM_MAX_INFLIGHT_REQUESTS=${VLLM_MAX_INFLIGHT_REQUESTS:-16}
VLLM_PER_ENDPOINT_CONCURRENCY=${VLLM_PER_ENDPOINT_CONCURRENCY:-4}
VLLM_REQUEST_TIMEOUT=${VLLM_REQUEST_TIMEOUT:-120}
MAX_RETRIES=${MAX_RETRIES:-3} # responses; hidden uses Producer retry policy
ENDPOINT_COOLDOWN=${ENDPOINT_COOLDOWN:-2} # responses only, seconds

# ---------- Response generation (TASK=responses) ----------
# Inline or separate reasoning is saved as a complete <think>...</think> block.
PRODUCER_GENERATION_MAX_TOKENS=${PRODUCER_GENERATION_MAX_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
SEED=${SEED:-null} # integer or null

# ---------- Hidden extraction (TASK=hidden) ----------
# Input must already contain answers, e.g. the response JSONL from the first task.
# Set OUTPUT_PATH to a directory, such as /data/hidden.
# MODEL_PATH is also used for tokenizer/model-config loading.
TOKENIZER_PATH=${TOKENIZER_PATH:-null} # null = use MODEL_PATH
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
VLLM_AUX_HIDDEN_STATE_LAYER_IDS=${VLLM_AUX_HIDDEN_STATE_LAYER_IDS:-'[1,9,17,25,33]'}
INCLUDE_FINAL_HIDDEN=${INCLUDE_FINAL_HIDDEN:-true} # service also captures final layer (Qwen3-8B: 36)
VALIDATE_HIDDEN_OUTPUTS=${VALIDATE_HIDDEN_OUTPUTS:-false} # optional full NaN/Inf scan
HIDDEN_FILE_CONCURRENCY=${HIDDEN_FILE_CONCURRENCY:-4} # file I/O workers, independent of requests
# Match the service's connector capture plan, including the final layer when needed.
RENDER_BOUNDARY_ENABLED=${RENDER_BOUNDARY_ENABLED:-false}
RENDER_BOUNDARY_TIMEOUT=${RENDER_BOUNDARY_TIMEOUT:-10}
PARSER_STRICT_ROLES=${PARSER_STRICT_ROLES:-false}
MAX_CONSECUTIVE_ERRORS=${MAX_CONSECUTIVE_ERRORS:-20}
PRODUCER_INPUT_QUEUE_SIZE=${PRODUCER_INPUT_QUEUE_SIZE:-32}
PRODUCER_PUBLISH_QUEUE_SIZE=${PRODUCER_PUBLISH_QUEUE_SIZE:-16}
PRODUCER_MAX_SEQUENCE_LENGTH=${PRODUCER_MAX_SEQUENCE_LENGTH:-8192}
# File connector output must be accessible from this process. Other existing
# backends can use a complete mapping here, including backend-specific options.
HIDDEN_STATES_STORE=${HIDDEN_STATES_STORE:-file} # same selector as training
HIDDEN_STATES_STORE_CONFIG=${HIDDEN_STATES_STORE_CONFIG:-"{backend:${HIDDEN_STATES_STORE}}"}

# ---------- Launch (no configuration below this line) ----------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "${SCRIPT_DIR}/.."

# Quote literal strings for Hydra, including paths with spaces or commas.
hydra_string() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '"%s"' "${value}"
}

hydra_optional_string() {
    if [[ "$1" == null ]]; then
        printf 'null'
    else
        hydra_string "$1"
    fi
}

config=standalone_data_generation
args=(
    "${config}.task=${TASK}"
    "${config}.log_interval=${LOG_INTERVAL}"
    "${config}.log_interval_seconds=${LOG_INTERVAL_SECONDS}"
    "${config}.input_path=$(hydra_string "${INPUT_PATH}")"
    "${config}.output_path=$(hydra_string "${OUTPUT_PATH}")"
    "${config}.resume=${RESUME}"
    "${config}.on_error=${ON_ERROR}"
    "${config}.max_samples=${MAX_SAMPLES}"
    "${config}.vllm.endpoints=${SPECO_VLLM_ENDPOINTS}"
    "${config}.vllm.model=$(hydra_string "${VLLM_MODEL}")"
    "${config}.vllm.api_key=$(hydra_string "${VLLM_API_KEY}")"
    "${config}.vllm.concurrency=${VLLM_MAX_INFLIGHT_REQUESTS}"
    "${config}.vllm.per_endpoint_concurrency=${VLLM_PER_ENDPOINT_CONCURRENCY}"
    "${config}.vllm.request_timeout=${VLLM_REQUEST_TIMEOUT}"
    "${config}.vllm.max_retries=${MAX_RETRIES}"
    "${config}.vllm.endpoint_cooldown=${ENDPOINT_COOLDOWN}"
    "${config}.responses.max_tokens=${PRODUCER_GENERATION_MAX_TOKENS}"
    "${config}.responses.temperature=${TEMPERATURE}"
    "${config}.responses.top_p=${TOP_P}"
    "${config}.responses.seed=${SEED}"
    "${config}.hidden.target_model_path=$(hydra_string "${MODEL_PATH}")"
    "${config}.hidden.tokenizer_path=$(hydra_optional_string "${TOKENIZER_PATH}")"
    "${config}.hidden.trust_remote_code=${TRUST_REMOTE_CODE}"
    "${config}.hidden.vllm_aux_hidden_state_layer_ids=${VLLM_AUX_HIDDEN_STATE_LAYER_IDS}"
    "${config}.hidden.include_final_hidden=${INCLUDE_FINAL_HIDDEN}"
    "${config}.hidden.validate_outputs=${VALIDATE_HIDDEN_OUTPUTS}"
    "${config}.hidden.file_concurrency=${HIDDEN_FILE_CONCURRENCY}"
    "${config}.hidden.render_boundary.enabled=${RENDER_BOUNDARY_ENABLED}"
    "${config}.hidden.render_boundary.timeout=${RENDER_BOUNDARY_TIMEOUT}"
    "${config}.parser_strict_roles=${PARSER_STRICT_ROLES}"
    "${config}.max_consecutive_errors=${MAX_CONSECUTIVE_ERRORS}"
    "${config}.input_queue_size=${PRODUCER_INPUT_QUEUE_SIZE}"
    "${config}.publish_queue_size=${PRODUCER_PUBLISH_QUEUE_SIZE}"
    "${config}.hidden.max_sequence_length=${PRODUCER_MAX_SEQUENCE_LENGTH}"
    "++${config}.hidden.hidden_states_store=${HIDDEN_STATES_STORE_CONFIG}"
)
exec "${PYTHON_BIN}" -m verl_speco.standalone_data_generate "${args[@]}" "$@"
