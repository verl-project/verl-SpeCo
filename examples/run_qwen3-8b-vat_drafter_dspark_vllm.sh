#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; configure model/data paths in the base example.
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
bash examples/run_qwen3-8b_drafter_dspark_vllm.sh \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dspark_per_position_loss_weight=vat \
    actor_rollout_ref.rollout.drafter.training.vat.accept_mode=greedy \
    actor_rollout_ref.rollout.drafter.training.vat.self_cov_weight=0.0 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=1.0 \
    trainer.experiment_name=qwen3_8b_dspark_vat_gpu "$@"
