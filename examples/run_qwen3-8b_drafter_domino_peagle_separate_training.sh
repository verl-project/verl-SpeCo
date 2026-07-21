set -euo pipefail
set -x

# GPU smoke example for separate Domino / P-EAGLE draft model training.
#
# Neither family is an engine-level speculative algorithm: engines serve Domino
# as a DFlash projector sub-mode (dflash_config.projector_type="domino"), and
# P-EAGLE needs the parallel-drafting runtime. Both are therefore trained
# offline and served separately, which is exactly the two-stage separate
# training workflow:
#
#   stage 1 (collect): rollout writes features with the engine algorithm whose
#                      hidden-state layout the drafter consumes
#                        Domino  <- DFLASH (dflash_aux)
#                        P-EAGLE <- EAGLE3 (eagle3_aux_plus_last)
#   stage 2 (train):   the standalone trainer reads that same feature store with
#                      speculative_algorithm=DOMINO or PEAGLE
#
# Usage:
#   DRAFT_ALGO=domino bash examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh
#   DRAFT_ALGO=peagle RUN_STAGE=collect bash examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh
#   DRAFT_ALGO=peagle RUN_STAGE=train bash examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh

DRAFT_ALGO=${DRAFT_ALGO:-domino}
RUN_STAGE=${RUN_STAGE:-both}

gen_tp=2
train_sp=1
ppo_gpus_per_node=8
draft_train_gpus_per_node=8

MODEL_PATH=/path/to/model
CKPTS_DIR=/path/to/checkpoint
TRAIN_FILE=/path/to/train_file
TEST_FILE=/path/to/test_file

case "${DRAFT_ALGO}" in
  domino)
    COLLECT_ALGO="DFLASH"
    TRAIN_ALGO="DOMINO"
    # Stage-1 drafter: any engine-servable DFlash drafter for the rollout.
    COLLECT_DRAFTER_PATH=/path/to/vllm-compatible-dflash-drafter
    # Stage-2 drafter: an existing Domino checkpoint, or a path that does not
    # exist yet to cold-start the draft from the target config.
    DRAFT_INIT_PATH=/path/to/domino-drafter-or-new-dir
    ALGO_TRAIN_ARGS=(
      actor_rollout_ref.rollout.drafter.training.domino_block_size=16
      actor_rollout_ref.rollout.drafter.training.domino_num_anchors=512
      actor_rollout_ref.rollout.drafter.training.domino_num_target_layers=5
      actor_rollout_ref.rollout.drafter.training.domino_emb_dim=256
      actor_rollout_ref.rollout.drafter.training.domino_gru_hidden_dim=1024
      actor_rollout_ref.rollout.drafter.training.domino_lambda_base_start=1.0
      actor_rollout_ref.rollout.drafter.training.domino_lambda_base_decay_steps=2000
    )
    ;;
  peagle)
    COLLECT_ALGO="EAGLE3"
    TRAIN_ALGO="PEAGLE"
    COLLECT_DRAFTER_PATH=/path/to/vllm-compatible-eagle3-drafter
    DRAFT_INIT_PATH=/path/to/peagle-drafter-or-new-dir
    ALGO_TRAIN_ARGS=(
      actor_rollout_ref.rollout.drafter.training.peagle_num_draft_layers=4
      actor_rollout_ref.rollout.drafter.training.peagle_num_aux_hidden_states=3
      actor_rollout_ref.rollout.drafter.training.peagle_num_depths=8
      actor_rollout_ref.rollout.drafter.training.peagle_down_sample_ratio=0.7
      actor_rollout_ref.rollout.drafter.training.peagle_down_sample_ratio_min=0.2
    )
    ;;
  *)
    echo "Unsupported DRAFT_ALGO=${DRAFT_ALGO}; expected domino or peagle" >&2
    exit 1
    ;;
esac

project_name="verl_grpo_example_${DRAFT_ALGO}_drafter"
exp_name="qwen3_8b_${DRAFT_ALGO}_separate_drafter_vllm_gpu"
FEATURE_STORE_DIR=/path/to/speco/${DRAFT_ALGO}_features
DRAFT_CKPTS_DIR=/path/to/speco/${DRAFT_ALGO}_draft_ckpts

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
    actor_rollout_ref.rollout.drafter.model_path=${COLLECT_DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm="${COLLECT_ALGO}" \
    actor_rollout_ref.rollout.drafter.training.mode=collect_only \
    actor_rollout_ref.rollout.drafter.training.feature_store.type=torch_shard \
    actor_rollout_ref.rollout.drafter.training.feature_store.path=${FEATURE_STORE_DIR} \
    actor_rollout_ref.rollout.drafter.training.feature_store.max_samples_per_shard=256 \
    actor_rollout_ref.rollout.drafter.training.feature_store.flush_interval_steps=1 \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False \
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
    actor_rollout_ref.rollout.drafter.model_path=${DRAFT_INIT_PATH} \
    actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm="${TRAIN_ALGO}" \
    actor_rollout_ref.rollout.drafter.training.mode=offline \
    actor_rollout_ref.rollout.drafter.training.max_steps=10 \
    actor_rollout_ref.rollout.drafter.training.save_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.drafter.training.lr=1e-4 \
    actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.drafter.training.warmup_style=constant \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    "${ALGO_TRAIN_ARGS[@]}" \
    actor_rollout_ref.rollout.drafter.training.feature_store.type=torch_shard \
    actor_rollout_ref.rollout.drafter.training.feature_store.path=${FEATURE_STORE_DIR} \
    actor_rollout_ref.rollout.drafter.training.feature_store.shuffle=True \
    actor_rollout_ref.rollout.drafter.training.feature_store.repeat=True \
    actor_rollout_ref.rollout.drafter.training.feature_store.strict_schema=True $@
fi
