#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export OPD_METHOD=${OPD_METHOD:-spec_opd}
sp_size=${SPECO_SP_SIZE:-4}
bash examples/run_qwen3-8b-opd-v1-debug_drafter_dspark_vllm_npu.sh \
    actor_rollout_ref.model.use_remove_padding=true \
    ++actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$sp_size" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="$sp_size" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.drafter.training.step=20 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=5 \
    trainer.total_training_steps=10 trainer.total_epochs=3 trainer.test_freq=10 \
    "$@"
