#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(dirname "$0")/.."
: "${SPECO_TARGET_MODEL:?set SPECO_TARGET_MODEL}"
: "${SPECO_DSPARK_DRAFT_MODEL:?set SPECO_DSPARK_DRAFT_MODEL}"
: "${SPECO_TRAIN_FILE:?set SPECO_TRAIN_FILE}"
steps=${SPECO_STANDALONE_STEPS:-5}
[[ "$steps" =~ ^[1-9][0-9]*$ ]] || { echo 'steps must be a positive integer' >&2; exit 2; }
test -d "$SPECO_TARGET_MODEL"
test -d "$SPECO_DSPARK_DRAFT_MODEL"
test -f "$SPECO_TRAIN_FILE"
run_dir=${SPECO_STANDALONE_RUN_DIR:-$(mktemp -d)}
# Never reuse previous outputs, including checkpoints and hidden-state files.
mkdir -p "$run_dir"
test -z "$(ls -A "$run_dir")" || { echo "Output directory is not empty: $run_dir" >&2; exit 2; }
run_dir=$(cd "$run_dir" && pwd)
exec > >(tee "$run_dir/runner.log") 2>&1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export VERL_LOGGING_LEVEL=INFO
export SPECO_STANDALONE_LOG_LEVEL=INFO
# File hidden-state transfer and SimpleStorage TQ need no external Mooncake service.
export SPECO_VLLM_HIDDEN_STATES_STORE=file
export SPECO_TQ_STORAGE_BACKEND=SimpleStorage
export SPECO_VLLM_ENDPOINTS='[http://127.0.0.1:8000/v1]'

npu-smi info
python - <<'PY'
import importlib.metadata
import socket
import torch
import torch_npu
import vllm
assert torch.npu.is_available() and torch.npu.device_count() >= 4
assert importlib.metadata.version('TransferQueue') == '0.1.10'
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 8000))
print('vllm', vllm.__version__, 'NPU count', torch.npu.device_count())
PY

python ci/verify_dspark_standalone_checkpoint.py prepare \
  --target "$SPECO_TARGET_MODEL" --initial "$SPECO_DSPARK_DRAFT_MODEL" \
  --baseline "$run_dir/baseline.json" --layer-env "$run_dir/layers.env"
source "$run_dir/layers.env"
compilation_config=${SPECO_STANDALONE_COMPILATION_CONFIG:-'{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'}
printf 'target=%q\ndrafter=%q\ntrain_file=%q\nsteps=%q\nbatch_size_per_gpu=2\nvllm_devices=0,1\ntrain_devices=2,3\ncompilation_config=%q\n' \
  "$SPECO_TARGET_MODEL" "$SPECO_DSPARK_DRAFT_MODEL" "$SPECO_TRAIN_FILE" "$steps" "$compilation_config" \
  >"$run_dir/settings.env"
vllm_pid=''
cleanup() {
  if [[ -n "$vllm_pid" ]]; then
    # The service script and its TP workers belong to this dedicated session.
    kill -TERM -- "-$vllm_pid" 2>/dev/null || true
    for ((i=0; i<20; i++)); do
      kill -0 -- "-$vllm_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
MODEL_PATH="$SPECO_TARGET_MODEL" DEVICE_ENV=ASCEND_RT_VISIBLE_DEVICES \
VLLM_DEVICES=0,1 VLLM_TP=2 VLLM_MAX_NUM_SEQS=8 \
VLLM_HIDDEN_STATE_LAYER_IDS="$CAPTURE_LAYER_IDS" HIDDEN_STATES_DIR="$run_dir/hidden_states" \
setsid bash tools/run_qwen3-8b_drafter_hidden_state_vllm.sh \
  --compilation-config "$compilation_config" --block-size 128 \
  --no-enable-prefix-caching >"$run_dir/vllm.log" 2>&1 &
vllm_pid=$!
ready=false
for ((i=0; i<120; i++)); do
  kill -0 "$vllm_pid" 2>/dev/null || { tail -100 "$run_dir/vllm.log"; exit 1; }
  if curl --fail --silent --max-time 2 http://127.0.0.1:8000/health >/dev/null; then
    ready=true
    break
  fi
  sleep 5
done
[[ "$ready" == true ]] || { echo 'vLLM readiness timeout' >&2; exit 1; }

MODEL_PATH="$SPECO_TARGET_MODEL" DRAFTER_PATH="$SPECO_DSPARK_DRAFT_MODEL" \
TRAIN_FILE="$SPECO_TRAIN_FILE" DEVICE_ENV=ASCEND_RT_VISIBLE_DEVICES \
TRAIN_DEVICES=2,3 TRAIN_GPUS=2 MAX_STEPS="$steps" BATCH_SIZE_PER_GPU=2 \
SAVE_INTERVAL_STEPS="$steps" SAVE_FINAL_CHECKPOINT=true \
DRAFT_CKPTS_DIR="$run_dir/checkpoints" TRAIN_LOGGERS='["console"]' \
LEARNING_RATE=1e-5 DSPARK_BLOCK_SIZE=7 DSPARK_NUM_ANCHORS=8 DSPARK_MAX_WINDOW=64 \
DSPARK_LOSS_MODE=restricted_ce DSPARK_L1_LOSS_ALPHA=0.9 \
VLLM_AUX_HIDDEN_STATE_LAYER_IDS="$AUX_LAYER_IDS" \
VLLM_MAX_INFLIGHT_REQUESTS=4 VLLM_PER_ENDPOINT_CONCURRENCY=4 \
PRODUCER_INPUT_QUEUE_SIZE=8 PRODUCER_PUBLISH_QUEUE_SIZE=8 PRODUCER_PUBLISH_WORKERS=2 \
PRODUCER_MAX_PENDING_SAMPLES=32 SCHEDULER_LOW_WATERMARK_SAMPLES=8 \
SCHEDULER_HIGH_WATERMARK_SAMPLES=16 PRODUCER_MAX_FEATURE_LENGTH=64 \
PRODUCER_GENERATION_MAX_TOKENS=512 \
bash examples/run_qwen3-8b_drafter_dspark_separate_training.sh \
  speco.standalone_tq_producer.on_missing_response=generate \
  actor_rollout_ref.rollout.drafter.training.gradient_accumulation_steps=1 \
  actor_rollout_ref.rollout.drafter.training.dspark_target_layer_ids="$TARGET_LAYER_IDS" \
  2>&1 | tee "$run_dir/training.log"

cleanup
vllm_pid=''
ASCEND_RT_VISIBLE_DEVICES=2 python ci/verify_dspark_standalone_checkpoint.py verify \
  --target "$SPECO_TARGET_MODEL" --initial "$SPECO_DSPARK_DRAFT_MODEL" \
  --baseline "$run_dir/baseline.json" --checkpoint "$run_dir/checkpoints/draft_step_$steps" \
  --steps "$steps" --training-log "$run_dir/training.log" --report "$run_dir/verification.json"
