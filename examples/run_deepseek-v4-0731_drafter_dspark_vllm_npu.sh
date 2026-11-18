#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Append SpeCo online-backend overrides to an already working DS4 RL launcher.
set -euo pipefail

# The capture/backward adapter uses the reviewed V1, full-token-row boundary.
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

: "${RL_SCRIPT:?Set RL_SCRIPT to a working DeepSeek V4 RL script that forwards its arguments}"
: "${MODEL_PATH:?Set MODEL_PATH to the full DeepSeek V4 0731 BF16 checkpoint}"
: "${DRAFT_WORK_DIR:?Set DRAFT_WORK_DIR to a shared worker directory}"
: "${DRAFTER_CKPTS:?Set DRAFTER_CKPTS to a shared drafter checkpoint directory}"

speco_python=${PYTHON:-python3}
speco_block_size=$("$speco_python" -c 'import json, pathlib, sys; print(int(json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())["dspark_block_size"]))' "$MODEL_PATH")

exec bash "$RL_SCRIPT" \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "actor_rollout_ref.rollout.drafter.model_path=$MODEL_PATH" \
  "actor_rollout_ref.rollout.drafter.checkpoint_path=$DRAFTER_CKPTS" \
  actor_rollout_ref.rollout.drafter.enable=true \
  actor_rollout_ref.rollout.drafter.enable_drafter_training=true \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
  "actor_rollout_ref.rollout.drafter.spec_verify_tokens=$speco_block_size" \
  actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend \
  actor_rollout_ref.rollout.drafter.training.mode=online \
  actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_rollout=false \
  actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=true \
  actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=false \
  actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
  actor_rollout_ref.rollout.drafter.training.use_logits=false \
  actor_rollout_ref.rollout.drafter.training.use_data_buffer=false \
  actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
  actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
  actor_rollout_ref.rollout.drafter.training.train_batches_per_cycle=1 \
  actor_rollout_ref.rollout.drafter.training.publish_interval_steps=1 \
  actor_rollout_ref.rollout.drafter.training.publish_async=false \
  actor_rollout_ref.rollout.drafter.training.is_offload_param=false \
  actor_rollout_ref.rollout.drafter.training.is_offload_optimizer=false \
  actor_rollout_ref.rollout.drafter.training.max_seq_len=1024 \
  "actor_rollout_ref.rollout.drafter.training.vllm_ascend.work_dir=$DRAFT_WORK_DIR" \
  actor_rollout_ref.rollout.drafter.training.vllm_ascend.total_optimizer_steps=4 \
  ++actor_rollout_ref.rollout.drafter.training.vllm_ascend.overrides.train.segment_length=null \
  ++actor_rollout_ref.rollout.drafter.training.vllm_ascend.overrides.train.max_len=null \
  ++actor_rollout_ref.rollout.drafter.training.vllm_ascend.overrides.deepspeed.train_micro_batch_size_per_gpu=1 \
  ++actor_rollout_ref.rollout.drafter.training.vllm_ascend.overrides.deepspeed.gradient_accumulation_steps=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.enable_sleep_mode=false \
  actor_rollout_ref.model.use_remove_padding=false \
  ++actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  data.max_prompt_length=256 \
  data.max_response_length=32 \
  trainer.total_training_steps=2 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  speco.runtime.weight_update_mode=compat \
  "$@"
