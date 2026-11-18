#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; configure model/data paths in the base example.
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
bash examples/run_qwen3-8b_drafter_dflash_vllm.sh \
    actor_rollout_ref.rollout.drafter.training.dflash_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dflash_per_position_loss_weight=vat \
    actor_rollout_ref.rollout.drafter.training.vat.accept_mode=greedy \
    actor_rollout_ref.rollout.drafter.training.vat.self_cov_weight=0.0 \
    trainer.experiment_name=qwen3_8b_dflash_vat_gpu "$@"
