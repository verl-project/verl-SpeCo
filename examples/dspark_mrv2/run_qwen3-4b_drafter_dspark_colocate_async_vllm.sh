#!/usr/bin/env bash
# Two-GPU native MRV2 DSpark smoke test, derived from the NPU example.
# Requires a CUDA vLLM build with native MRV2 DSpark support.
# The sync script retains the original legacy SPECO trainer; async uses V1/TQ.
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1

project_name='verl_grpo_example_dspark_drafter'
exp_name='dspark_mrv2_colocate_async_vllm_gpu_smoke'
gen_tp=2
train_sp=1
ppo_gpus_per_node=2
rollout_gpus_per_node=2
ray_num_cpus=${SPECO_RAY_NUM_CPUS:-64}
ray_worker_soft_limit=${SPECO_RAY_WORKER_SOFT_LIMIT:-8}
spec_verify_tokens=7
rollout_max_model_len=2048
# vLLM instance memory budget; actor/drafter training needs additional memory.
gpu_memory_utilization=${SPECO_GPU_MEMORY_UTILIZATION:-0.4}

MODEL_PATH=/path/to/model
TRAIN_FILE=/path/to/train_file
TEST_FILE=/path/to/test_file
DRAFTER_PATH=/path/to/vllm-compatible-dspark-drafter
CKPTS_DIR=/path/to/checkpoint

# Three global steps: collect/train/publish every two steps, two draft updates
# per training cycle. Short responses and micro-batches limit smoke-test memory.
PYTHONUNBUFFERED=1 python3 -m verl_speco.main --config-name=speco_v1_trainer \
    algorithm.adv_estimator=grpo \
    transfer_queue.enable=True \
    ray_kwargs.ray_init.num_cpus=${ray_num_cpus} \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=${ray_worker_soft_limit} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=8 \
    data.val_batch_size=4 \
    data.truncation='error' \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n_gpus_per_node=${rollout_gpus_per_node} \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=True \
    actor_rollout_ref.rollout.drafter.training.data_buffer_max_size=64 \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=8 \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=restricted_ce \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=256 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=vanilla \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=False \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1 \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.9 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${spec_verify_tokens} \
    actor_rollout_ref.rollout.drafter.training.step=2 \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=4 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=2048 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=2 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=2 \
    actor_rollout_ref.rollout.drafter.training.publish_interval_steps=2 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.drafter.training.is_offload_param=True \
    actor_rollout_ref.rollout.drafter.training.is_offload_optimizer=True \
    actor_rollout_ref.rollout.drafter.training.publish_async=False \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=128 \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.load_format="auto" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.v1.trainer_mode=colocate_async \
    ++trainer.v1.colocate_async.num_warmup_batches=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${ppo_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.total_training_steps=3 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=6 "$@"
