#!/usr/bin/env bash
# Synchronous OPD co-train | DSpark drafter | vLLM rollout | FSDP2 | NVIDIA GPUs
# vLLM rollout only: teacher prefill timing is not collected on the SGLang path.
# Defaults mirror the reported 8B-student / 32B-teacher experiment (6 GPUs).

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export PYTHONUNBUFFERED=1

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/path/to/Qwen3-8B}
TEACHER_MODEL=${TEACHER_MODEL:-/path/to/Qwen3-32B}
# Must be a vLLM-compatible DSpark checkpoint built for STUDENT_MODEL whose
# config.json matches the DSPARK_* settings below (block_size, layers, ...).
DRAFTER_MODEL=${DRAFTER_MODEL:-/path/to/dspark_qwen3_8b}

# Student training/rollout share 4 GPUs; the OPD teacher uses another 2 GPUs.
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.85}

drafter_lr=${DRAFTER_LR:-6e-6}
drafter_train_batch_size=${DRAFTER_TRAIN_BATCH_SIZE:-8}
drafter_train_steps=${DRAFTER_TRAIN_STEPS:-20}
drafter_collect_interval=${DRAFTER_COLLECT_INTERVAL:-4}
drafter_training_interval=${DRAFTER_TRAINING_INTERVAL:-4}
drafter_publish_interval=${DRAFTER_PUBLISH_INTERVAL:-${drafter_training_interval}}
drafter_sample_last_n_steps=${DRAFTER_SAMPLE_LAST_N_STEPS:-4}
drafter_window_tokens=${DRAFTER_WINDOW_TOKENS:-256}
drafter_max_collect_samples=${DRAFTER_MAX_COLLECT_SAMPLES:-64}
drafter_max_collect_tokens=${DRAFTER_MAX_COLLECT_TOKENS:-32768}
# Architecture settings must match DRAFTER_MODEL/config.json.
dspark_block_size=${DSPARK_BLOCK_SIZE:-4}
dspark_num_target_layers=${DSPARK_NUM_TARGET_LAYERS:-5}
dspark_num_hidden_layers=${DSPARK_NUM_HIDDEN_LAYERS:-5}
dspark_markov_rank=${DSPARK_MARKOV_RANK:-256}
dspark_markov_head_type=${DSPARK_MARKOV_HEAD_TYPE:-vanilla}
dspark_num_anchors=${DSPARK_NUM_ANCHORS:-64}

total_epochs=${TOTAL_EPOCHS:-6}
total_training_steps=${TOTAL_TRAINING_STEPS:-200}
save_freq=${SAVE_FREQ:-500}
test_freq=${TEST_FREQ:-25}
project_name=${PROJECT_NAME:-verl_opd_cotrain_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_from_qwen3_32b_dspark_vllm_fsdp2}
# ---- end user-adjustable ----

gsm8k_train=/path/to/datasets/gsm8k/train.parquet
gsm8k_test=/path/to/datasets/gsm8k/test.parquet
math_train=/path/to/datasets/math/train.parquet
math_test=/path/to/datasets/math/test.parquet

train_files=${TRAIN_FILES:-"['$gsm8k_train', '$math_train']"}
val_files=${VAL_FILES:-"['$gsm8k_test', '$math_test']"}
max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # Co-train needs the Student old-logprob forward to collect drafter features.
    algorithm.rollout_correction.bypass_mode=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

DRAFTER=(
    actor_rollout_ref.rollout.drafter.enable=True
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True
    actor_rollout_ref.rollout.drafter.model_path="$DRAFTER_MODEL"
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK
    actor_rollout_ref.rollout.drafter.training.mode=online
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook
    actor_rollout_ref.rollout.drafter.training.use_logits=False
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=True
    actor_rollout_ref.rollout.drafter.training.lr=${drafter_lr}
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${drafter_train_batch_size}
    actor_rollout_ref.rollout.drafter.training.step=${drafter_train_steps}
    actor_rollout_ref.rollout.drafter.training.collection_sample_rate=1.0
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=${drafter_collect_interval}
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=${drafter_training_interval}
    actor_rollout_ref.rollout.drafter.training.publish_interval_steps=${drafter_publish_interval}
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps=${drafter_sample_last_n_steps}
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_mode=random
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=${drafter_window_tokens}
    actor_rollout_ref.rollout.drafter.training.hidden_state_random_seed_by_step=True
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=${drafter_max_collect_samples}
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=${drafter_max_collect_tokens}
    actor_rollout_ref.rollout.drafter.training.max_seq_len=${max_num_tokens}
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=${dspark_block_size}
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=${dspark_num_anchors}
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=${drafter_window_tokens}
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=${dspark_num_target_layers}
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=${dspark_num_hidden_layers}
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=${dspark_markov_rank}
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=${dspark_markov_head_type}
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=full_vocab
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=4
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.45
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0
    # Publish updated drafter weights synchronously between rollout batches.
    actor_rollout_ref.rollout.drafter.training.publish_async=False
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=True
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${dspark_block_size}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.resume_mode=disable
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -m verl_speco.main \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DRAFTER[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
