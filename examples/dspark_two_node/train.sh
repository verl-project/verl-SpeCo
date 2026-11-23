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
set -eo pipefail
source ./job.env
phase=${1:?train or resume}
export TRAIN_FILE=${2:?training dataset}
export MODEL_PATH="$MODEL" TRAIN_LOGGERS='["console"]' PYTHON_BIN=python3
export TRAIN_GPUS=1 TRAIN_DEVICES="$CONSUMER_NPU" BATCH_SIZE_PER_GPU=2
export LEARNING_RATE=1e-6 LR_WARMUP_STEPS=0 LR_SCHEDULER_TYPE=constant LR_DECAY_STEPS=100
export PARAM_OFFLOAD=true OPTIMIZER_OFFLOAD=true
export DSPARK_BLOCK_SIZE=7 DSPARK_NUM_ANCHORS=32 DSPARK_MAX_WINDOW=512
export DSPARK_LOSS_MODE=restricted_ce DSPARK_LOSS_DECAY_GAMMA=7 DSPARK_L1_LOSS_ALPHA=0.9
export DSPARK_CONFIDENCE_HEAD_ALPHA=0 DSPARK_CONFIDENCE_LOSS_ALPHA=0
export PRODUCER_SHUFFLE=false VLLM_MAX_INFLIGHT_REQUESTS=1 VLLM_PER_ENDPOINT_CONCURRENCY=2
export PRODUCER_MAX_PENDING_SAMPLES=12 SCHEDULER_LOW_WATERMARK_SAMPLES=2 SCHEDULER_HIGH_WATERMARK_SAMPLES=4
export VLLM_READY_TIMEOUT_SECONDS=20
export RAY_TMPDIR="$(cat "$RUN_DIR/ray.tmp")"
case "$phase" in
  train)  export MAX_STEPS=25 SAVE_INTERVAL_STEPS=25 SAVE_FINAL_CHECKPOINT=true ;;
  resume) export DRAFTER_PATH=${3:?checkpoint} MAX_STEPS=50 SAVE_INTERVAL_STEPS=0 SAVE_FINAL_CHECKPOINT=false ;;
  *) exit 2 ;;
esac
cd "$REPO"
exec bash examples/run_qwen3-8b_drafter_dspark_separate_training.sh \
  speco.draft_training.nnodes=1 "speco.draft_training.producer_node_ip=$PRODUCER_IP" speco.draft_training.placement_timeout_seconds=180 \
  speco.draft_training.data_transport=ray_reference \
  speco.draft_training.scheduler.per_rank_reference_ack=true \
  speco.standalone_tq_producer.max_published_references=2 \
  speco.standalone_tq_producer.max_reference_execution_slots=2 \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_batch_workers=0 \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.consumer_prefetch_batches=4
