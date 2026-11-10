#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with the model/data environment variables from the base example.
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
bash examples/run_qwen3-8b_drafter_dspark_vllm.sh \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dspark_per_position_loss_weight=fixed_exp_decay \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.enable=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.capture_verify_trace=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.anchor_planner=mixed \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.random_anchor_ratio=0.25 \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.loss=e2e_tv \
    actor_rollout_ref.rollout.drafter.training.lk_temperature=0.6 \
    actor_rollout_ref.rollout.drafter.training.packing.enable=false \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=false \
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_async=false \
    actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=sync \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=false \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0 \
    trainer.experiment_name=qwen3_8b_dspark_draft_opd_gpu "$@"
