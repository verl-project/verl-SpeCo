set -x
export ASCEND_RT_VISIBLE_DEVICES="0"
case "${LD_PRELOAD:-}" in
    *libjemalloc*) ;;
    *)
        if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        elif [ -f /usr/lib64/libjemalloc.so.2 ]; then
            export LD_PRELOAD="/usr/lib64/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
        ;;
esac
export MALLOC_CONF="${MALLOC_CONF:-narenas:8,thp:never,metadata_thp:disabled,dirty_decay_ms:0,muzzy_decay_ms:0}"
export SPECO_JEMALLOC_RECLAIM_MODE="${SPECO_JEMALLOC_RECLAIM_MODE:-purge}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

# NPU example for the native DFlash proposer in vLLM.
project_name='verl_grpo_adaptive_schedule_dflash_drafter'
exp_name='qwen3_4b_lora_dflash_drafter_adaptive_schedule_vllm_npu'

gen_tp=1
train_sp=1
ppo_gpus_per_node=${SPECO_ACCELERATOR_COUNT:-1}
ray_num_cpus=${SPECO_RAY_NUM_CPUS:-4}
ray_worker_soft_limit=${SPECO_RAY_WORKER_SOFT_LIMIT:-2}

MODEL_PATH=/path/to/model
CKPTS_DIR=/path/to/checkpoint
TRAIN_FILE=/path/to/train_file
TEST_FILE=/path/to/test_file
DRAFTER_PATH=/path/to/drafter

PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    transfer_queue.enable=False \
    ray_kwargs.ray_init.num_cpus=${ray_num_cpus} \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=${ray_worker_soft_limit} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=8 \
    data.prompt_key="source_prompt" \
    data.max_prompt_length=256 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.truncation='error' \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=512 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
    actor_rollout_ref.rollout.drafter.training.adaptive_schedule.enable=True \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.dflash_block_size=8 \
    actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=64 \
    actor_rollout_ref.rollout.drafter.training.dflash_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dflash_loss_mode=restricted_ce \
    actor_rollout_ref.rollout.drafter.training.dflash_loss_decay_gamma=7 \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_weight=2.0 \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_count=3 \
    actor_rollout_ref.rollout.drafter.training.dflash_hard_sample_ratio=0.3 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7 \
    actor_rollout_ref.rollout.drafter.training.step=20 \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=16 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.publish_async=True \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.load_format="safetensors" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${ppo_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.total_training_steps=100 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@