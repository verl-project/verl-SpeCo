#!/usr/bin/env bash
set -euo pipefail

# Small dual-method template, default SpecOPD. Supply the same MODEL_PATH, DRAFTER_PATH,
# TRAIN_FILE, TEST_FILE and CKPTS_DIR as the existing DFlash example.
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
num_devices=${N_GPUS:-8}
rollout_tp=${ROLLOUT_TP:-1}
opd_method=${OPD_METHOD:-spec_opd}
remove_padding=false
attention=eager
if [ "$opd_method" = draft_opd ]; then
    remove_padding=true
    attention=flash_attention_2
fi

bash examples/run_qwen3-8b_drafter_dflash_vllm.sh \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.enable=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.method=${opd_method} \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.capture_verify_trace=true \
    actor_rollout_ref.rollout.drafter.training.specopd.enable=false \
    actor_rollout_ref.model.use_remove_padding=${remove_padding} \
    actor_rollout_ref.model.use_fused_kernels=false \
    +actor_rollout_ref.model.override_config.attn_implementation=${attention} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.drafter.world_size=${num_devices} \
    actor_rollout_ref.rollout.drafter.vllm.draft_sample_method=probabilistic \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7 \
    actor_rollout_ref.rollout.drafter.training.dflash_block_size=8 \
    actor_rollout_ref.rollout.drafter.training.dflash_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dflash_per_position_loss_weight=fixed_exp_decay \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_weight=1.0 \
    actor_rollout_ref.rollout.drafter.training.lk_temperature=0.6 \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=false \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=false \
    actor_rollout_ref.rollout.drafter.training.packing.enable=false \
    actor_rollout_ref.rollout.drafter.training.step=1 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_async=false \
    actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=sync \
    actor_rollout_ref.rollout.drafter.training.specopd.actor_temperature=0.6 \
    actor_rollout_ref.rollout.drafter.training.specopd.proposal_temperature=0.6 \
    actor_rollout_ref.rollout.drafter.training.specopd.target_temperature=0.6 \
    actor_rollout_ref.rollout.drafter.training.specopd.proposal_width=7 \
    actor_rollout_ref.rollout.drafter.training.specopd.max_branches_per_request=8 \
    actor_rollout_ref.rollout.drafter.training.specopd.max_proposal_tokens_per_request=56 \
    actor_rollout_ref.rollout.drafter.training.specopd.microbatch_trajectories=1 \
    data.train_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    trainer.use_v1=false \
    trainer.n_gpus_per_node=${num_devices} \
    trainer.total_training_steps=5 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger='[console]' \
    trainer.experiment_name=qwen3_8b_dflash_specopd_vllm \
    "$@"
