#!/usr/bin/env bash
set -euo pipefail
method=${OPD_METHOD:-draft_opd}
case "$method" in draft_opd|spec_opd) ;; *) exit 2 ;; esac
remove_padding=false
attention=eager
if [ "$method" = draft_opd ]; then
    remove_padding=true
    attention=flash_attention_2
fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V1=1 VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export VERL_SPECO_DIAGNOSTIC_TRANSPORT=1
# This single-node development host has a hostname that Gloo cannot use.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
unset RAY_ADDRESS
out=${DEBUG_OUTPUT:-/home/c00954340/specopd-debug-logs/${method}-01}
mkdir -p "$out"
python -m verl_speco.main --config-name="${SPECO_CONFIG_NAME:-speco_trainer}" \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/c00954340/specopd-debug/train16.parquet \
    data.val_files=/data/c00954340/specopd-debug/aime4.parquet \
    data.train_batch_size=4 data.val_batch_size=2 \
    data.max_prompt_length=1024 data.max_response_length=128 \
    data.filter_overlong_prompts=true data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=/data/c00954340/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding="$remove_padding" \
    actor_rollout_ref.model.use_fused_kernels=false \
    +actor_rollout_ref.model.override_config.attn_implementation="$attention" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.calculate_entropy=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.drafter.enable=true \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=true \
    actor_rollout_ref.rollout.drafter.model_path=/data/c00954340/qwen3_8b_dspark \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    actor_rollout_ref.rollout.drafter.world_size=4 \
    actor_rollout_ref.rollout.drafter.vllm.draft_sample_method=probabilistic \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7 \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.enable=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.method="$method" \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.capture_verify_trace=true \
    actor_rollout_ref.rollout.drafter.training.drafter_opd.loss=e2e_tv \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=true \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dspark_per_position_loss_weight=fixed_exp_decay \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=4 \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=false \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=1 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=64 \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=false \
    actor_rollout_ref.rollout.drafter.training.step=1 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_async=false \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.drafter.training.specopd.max_branches_per_request=2 \
    actor_rollout_ref.rollout.drafter.training.specopd.max_proposal_tokens_per_request=14 \
    trainer.device=npu trainer.use_v1=false trainer.n_gpus_per_node=4 trainer.nnodes=1 \
    trainer.total_training_steps=2 trainer.total_epochs=1 \
    trainer.val_before_train=false trainer.test_freq=2 trainer.save_freq=-1 \
    trainer.logger='[console]' trainer.default_local_dir="$out" \
    trainer.resume_mode=disable \
    ray_kwargs.ray_init.num_cpus=16 \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_SPECO_DIAGNOSTIC_TRANSPORT='1'" \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.resources.NPU=4 \
    +ray_kwargs.ray_init.include_dashboard=false \
    +ray_kwargs.ray_init._temp_dir="/tmp/speco44-${method}" \
    "$@"
