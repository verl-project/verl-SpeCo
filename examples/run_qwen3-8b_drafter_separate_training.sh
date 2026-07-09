set -x

# GPU smoke example for separate EAGLE3 draft model training.
#
# Stage 1 collects draft-training features from a short PPO/vLLM run without
# training the drafter inside PPO. Stage 2 launches independent multi-GPU draft
# training with python -m verl_speco.draft_train_launcher, which internally
# starts torch.distributed.run.
#
# Usage:
#   bash examples/run_qwen3-8b_drafter_eagle3_vllm_separate_multigpu.sh
#   RUN_STAGE=collect bash examples/run_qwen3-8b_drafter_eagle3_vllm_separate_multigpu.sh
#   RUN_STAGE=train bash examples/run_qwen3-8b_drafter_eagle3_vllm_separate_multigpu.sh

project_name='verl_grpo_example_eagle3_drafter'
exp_name='qwen3_8b_eagle3_separate_drafter_vllm_gpu'

gen_tp=2
train_sp=1
ppo_gpus_per_node=8
draft_train_gpus_per_node=8

MODEL_PATH=/path/to/model
CKPTS_DIR=/path/to/checkpoint
TRAIN_FILE=/path/to/train_file
TEST_FILE=/path/to/test_file
DRAFTER_PATH=/path/to/vllm-compatible-eagle3-drafter
FEATURE_STORE_DIR=/path/to/speco/eagle3_features
DRAFT_CKPTS_DIR=/path/to/speco/eagle3_draft_ckpts

RUN_STAGE=${RUN_STAGE:-both}

if [ "${RUN_STAGE}" = "both" ] || [ "${RUN_STAGE}" = "collect" ]; then
PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=256 \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm="EAGLE3" \
    actor_rollout_ref.rollout.drafter.training.mode=collect_only \
    actor_rollout_ref.rollout.drafter.training.feature_store.type=torch_shard \
    actor_rollout_ref.rollout.drafter.training.feature_store.path=${FEATURE_STORE_DIR} \
    actor_rollout_ref.rollout.drafter.training.feature_store.max_samples_per_shard=256 \
    actor_rollout_ref.rollout.drafter.training.feature_store.flush_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=True \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=3 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=4 \
    actor_rollout_ref.rollout.drafter.training.step=20 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.publish_interval_steps=0 \
    actor_rollout_ref.rollout.drafter.training.publish_async=False \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.load_format="auto" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name}_collect \
    trainer.n_gpus_per_node=${ppo_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@
fi

if [ "${RUN_STAGE}" = "both" ] || [ "${RUN_STAGE}" = "train" ]; then
PYTHONUNBUFFERED=1 python3 -m verl_speco.draft_train_launcher \
    speco.draft_training.num_gpus_per_node=${draft_train_gpus_per_node} \
    speco.draft_training.nnodes=1 \
    speco.draft_training.standalone=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm="EAGLE3" \
    actor_rollout_ref.rollout.drafter.training.mode=offline \
    actor_rollout_ref.rollout.drafter.training.max_steps=10 \
    actor_rollout_ref.rollout.drafter.training.save_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.drafter.training.lr=1e-6 \
    actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.drafter.training.warmup_style=constant \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    actor_rollout_ref.rollout.drafter.training.feature_store.type=torch_shard \
    actor_rollout_ref.rollout.drafter.training.feature_store.path=${FEATURE_STORE_DIR} \
    actor_rollout_ref.rollout.drafter.training.feature_store.shuffle=True \
    actor_rollout_ref.rollout.drafter.training.feature_store.repeat=True \
    actor_rollout_ref.rollout.drafter.training.feature_store.strict_schema=True $@
fi
