#!/usr/bin/env bash
# Native online RL run for the SpeCo C5 retention audit on the RTX 5090 host (GPUs 1 and 3).
#
# Usage: run_native_online.sh <smoke|eagle|dflash> [extra hydra overrides...]
set -euo pipefail
ALGORITHM="${1:-smoke}"
shift || true

ROOT="${SPECO_ROOT:-$HOME/0z5a-work/speco-retention}"
SRC="${SPECO_SRC:-$ROOT/src/current}"
PY="${SPECO_PYTHON:-$HOME/0z5a/bin/python}"
GPUS="${SPECO_GPUS:-1,3}"
STEPS="${SPECO_STEPS:-20}"
RUN_TAG="${SPECO_RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
EVIDENCE="$ROOT/evidence/full-e2e"
MODELS="${SPECO_MODELS:-$ROOT/models}"

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0
export RAY_DEDUP_LOGS=0
export RAY_worker_niceness=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export RAY_TMPDIR="/tmp/speco-retention-ray"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
export HYDRA_FULL_ERROR=1
export PYTHONSAFEPATH=1
PROBE_OVERLAY="${SPECO_PROBE_OVERLAY:-$SRC/src/probe-overlay}"
[ -d "$PROBE_OVERLAY" ] && export PYTHONPATH="$PROBE_OVERLAY:$SRC/padding-overlay:$SRC" || export PYTHONPATH="$SRC/padding-overlay:$SRC"
mkdir -p "$EVIDENCE" "$RAY_TMPDIR" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

"$PY" "$SRC/experiment/l40s/prepare_data.py" "$EVIDENCE"

COMMON=(
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False
  ray_kwargs.ray_init.num_cpus=16
  data.train_files="$EVIDENCE/train.parquet"
  data.val_files="$EVIDENCE/val.parquet"
  data.train_batch_size=2 data.max_prompt_length=128 data.max_response_length=64
  data.filter_overlong_prompts_workers=1 data.truncation=error
  actor_rollout_ref.model.path="$MODELS/target-qwen3-4b"
  actor_rollout_ref.model.use_remove_padding=True
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=2
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.calculate_entropy=False
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.load_format=auto
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.gpu_memory_utilization=0.10
  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=9216
  actor_rollout_ref.rollout.max_model_len=256
  actor_rollout_ref.rollout.max_num_seqs=4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.agent.num_workers=2
  actor_rollout_ref.rollout.drafter.enable=True
  actor_rollout_ref.rollout.drafter.enable_drafter_training=True
  actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False
  actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True
  actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook
  actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=32
  actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=2
  actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1
  actor_rollout_ref.rollout.drafter.training.step=1
  actor_rollout_ref.rollout.drafter.training.collect_interval_steps=1
  actor_rollout_ref.rollout.drafter.training.training_interval_steps=1
  actor_rollout_ref.rollout.drafter.training.publish_interval_steps=1
  actor_rollout_ref.rollout.drafter.training.publish_async=True
  actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=256
  actor_rollout_ref.rollout.drafter.training.draft_update_use_shm=True
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=134217728
  actor_rollout_ref.rollout.max_num_batched_tokens=256
  trainer.n_gpus_per_node=2 trainer.nnodes=1
  trainer.val_before_train=False trainer.logger='[console]'
  custom_reward_function.path="$SRC/experiment/l40s/integration_reward.py"
  custom_reward_function.name=compute_score
  +data.apply_chat_template_kwargs.enable_thinking=False
  trainer.save_freq=10 trainer.test_freq=-1
  trainer.total_epochs=3 trainer.project_name=speco-retention
)

case "$ALGORITHM" in
  smoke)
    ARMS=(eagle)
    STEPS="${SPECO_STEPS:-2}"
    ;;
  eagle)
    ARMS=(eagle)
    ;;
  dflash)
    ARMS=(dflash)
    ;;
  both)
    ARMS=(eagle dflash)
    ;;
  *)
    echo "usage: $0 <smoke|eagle|dflash|both> [overrides...]" >&2
    exit 2
    ;;
esac

status=0
for arm in "${ARMS[@]}"; do
  log="$EVIDENCE/online-$arm-$RUN_TAG.log"
  echo "[run] arm=$arm steps=$STEPS gpus=$GPUS log=$log"
  arm_status=0
  if [ "$arm" = eagle ]; then
    "$PY" -m verl_speco.main "${COMMON[@]}" \
      actor_rollout_ref.rollout.drafter.model_path="$ROOT/models/eagle-training-alias" \
      +actor_rollout_ref.rollout.drafter.vllm.speculative_config_overrides.model="$MODELS/eagle3-qwen3-4b" \
      actor_rollout_ref.rollout.drafter.speculative_algorithm=EAGLE3 \
      trainer.experiment_name=eagle3-retention \
      +trainer.total_training_steps="$STEPS" \
      trainer.default_local_dir="$EVIDENCE/checkpoints-eagle-$RUN_TAG" "$@" > "$log" 2>&1 || arm_status=$?
  else
    "$PY" -m verl_speco.main "${COMMON[@]}" \
      actor_rollout_ref.rollout.drafter.model_path="$MODELS/dflash-qwen3-4b-b16" \
      actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
      actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=16 \
      actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=8 \
      actor_rollout_ref.rollout.drafter.training.dflash_max_window=32 \
      trainer.experiment_name=dflash-retention \
      +trainer.total_training_steps="$STEPS" \
      trainer.default_local_dir="$EVIDENCE/checkpoints-dflash-$RUN_TAG" "$@" > "$log" 2>&1 || arm_status=$?
  fi
  echo "$arm_status" > "$EVIDENCE/online-$arm-$RUN_TAG.exit"
  echo "[run] arm=$arm exit=$arm_status"
  [ "$arm_status" -ne 0 ] && status=$arm_status
done
exit "$status"
