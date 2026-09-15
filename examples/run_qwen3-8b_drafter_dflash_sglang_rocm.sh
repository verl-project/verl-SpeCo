set -x
# Minimal ROCm/MI355X smoke: GRPO + DFlash drafter co-training on SGLang.
# Small sizes to complete a couple of real global steps (incl. a drafter
# collect+train+publish) quickly. Outputs go to /dev/shm because host disk is full.

project_name='verl_speco_rocm_smoke'
exp_name='qwen3_8b_dflash_drafter_rocm'

gen_tp=2
train_sp=1
ngpus=2
ray_num_cpus=${SPECO_RAY_NUM_CPUS:-32}
ray_worker_soft_limit=${SPECO_RAY_WORKER_SOFT_LIMIT:-8}

MODEL_PATH=${MODEL_PATH:-/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen3-8B}
DRAFTER_PATH=${DRAFTER_PATH:-/home/zhenchen/.cache/huggingface/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4}
TRAIN_FILE=${TRAIN_FILE:-/home/zhenchen/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-/home/zhenchen/data/gsm8k/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-/dev/shm/verl_speco_ckpt}

# ROCm + Ray GPU isolation. Set ONLY CUDA_VISIBLE_DEVICES (no HIP var, no NOSET flags):
#  - Ray 2.56's AMD accelerator manager (amd_gpu.py) uses CUDA_VISIBLE_DEVICES as its
#    keyword when HIP_VISIBLE_DEVICES is unset, so this mask restricts Ray's pool to
#    exactly these 4 GPUs and Ray becomes fully CUDA-native.
#  - With NOSET off, Ray assigns each worker a single physical GPU and rewrites that
#    worker's CUDA_VISIBLE_DEVICES to just that id (torch sees 1 device, ordinal 0).
#    verl's Worker sees is_ray_noset_visible_devices=False and SKIPS set_device(local_rank),
#    avoiding the "invalid device ordinal" that a non-contiguous HIP mask + NOSET causes
#    (Ray returns physical ids like 4/5/6 but torch only has ordinals 0..3).
#  - Crucially, HIP_VISIBLE_DEVICES is NEVER set by anyone. The SGLang http server
#    (verl async_sglang_server.py) sets CUDA_VISIBLE_DEVICES to its aggregated TP devices;
#    with no HIP var present there is no "Conflicting visibility between HIP and CUDA" abort
#    that occurs when Ray injects HIP while the server sets CUDA.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    ray_kwargs.ray_init.num_cpus=${ray_num_cpus} \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=${ray_worker_soft_limit} \
    +ray_kwargs.ray_init._temp_dir=/dev/shm/ray \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=aiter \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=True \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=64 \
    actor_rollout_ref.rollout.drafter.training.dflash_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dflash_loss_decay_gamma=7 \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_weight=2.0 \
    actor_rollout_ref.rollout.drafter.training.dflash_front_position_count=3 \
    actor_rollout_ref.rollout.drafter.training.dflash_hard_sample_ratio=0.3 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=16 \
    actor_rollout_ref.rollout.drafter.training.step=4 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_async=True \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.load_format="auto" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${ngpus} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=${SPECO_TOTAL_TRAINING_STEPS:-3} \
    trainer.total_epochs=1 $@
