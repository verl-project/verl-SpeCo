#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
set -euo pipefail

# GPU quality-alignment recipe: 4 GPUs, TP=2, two rollout replicas.
# Reuse the regular DSpark launch; do not change the existing Sync example.
# See docs/bubble_time_drafter_training.md for setup and interpretation.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${SPECO_TRAINING_MODE:-bubble}
case "${mode}" in
    bubble) execution_strategy=rollout_idle_worker; quota_enabled=true ;;
    sync) execution_strategy=sync; quota_enabled=false ;;
    *) printf 'SPECO_TRAINING_MODE must be bubble or sync\n' >&2; exit 2 ;;
esac

MODEL_PATH=${MODEL_PATH:-/path/to/model}
DRAFTER_PATH=${DRAFTER_PATH:-/path/to/vllm-compatible-dspark-drafter}
TRAIN_FILE=${TRAIN_FILE:-/path/to/train_file}
TEST_FILE=${TEST_FILE:-/path/to/test_file}
CKPTS_DIR=${CKPTS_DIR:-./checkpoints/qwen3-8b_dspark_${mode}}
export MODEL_PATH DRAFTER_PATH TRAIN_FILE TEST_FILE CKPTS_DIR

# Memory-safe default: park trainer state before the main workload resumes.
# With measured memory headroom, enable SPECO_KEEP_HOT=true on Bubble runs.
# It does not change the quota, optimizer ownership or publication boundary.
keep_hot=${SPECO_KEEP_HOT:-false}
# Example-only default: trim unused cache when a Bubble plan retains hot state.
# Preserve explicit false; do not change the Sync environment or global default.
if [[ "${mode}" == bubble ]]; then
    export VERL_DRAFTER_KEEP_HOT_TRIM_CACHE=${VERL_DRAFTER_KEEP_HOT_TRIM_CACHE:-true}
fi

# Arguments supplied by the caller are last and override this recipe.
# Run both modes from the same initial checkpoints and use separate outputs.
exec bash "${script_dir}/../run_qwen3-8b_drafter_dspark_vllm.sh" \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.drafter.training.strategy=fsdp2 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.drafter.training.lr=1e-5 \
    actor_rollout_ref.rollout.drafter.training.step=20 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=4 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=4 \
    actor_rollout_ref.rollout.drafter.training.collection_sample_rate=1.0 \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=16 \
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=16384 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log=False \
    actor_rollout_ref.rollout.drafter.training.publish_async=True \
    actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=${execution_strategy} \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.enable=${quota_enabled} \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.trigger_mode=interval \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.target_steps=20 \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.max_accumulated_debt=20 \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.max_debt_age_steps=3 \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.max_completion_lag_steps=3 \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.max_sync_topup_steps=20 \
    actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.allow_critical_path_fallback=true \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.keep_hot_between_plans=${keep_hot} \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.historical_window_readmission=true \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.writer_migration_enable=false \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.full_collective_fallback=false \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.early_publish_staging=false \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.head_reference_transport=true \
    actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.calibrated_budgets=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.experiment_name=qwen3_8b_dspark_${mode} \
    trainer.total_training_steps=40 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@"
