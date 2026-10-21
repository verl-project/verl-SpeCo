#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
# Fast integration smoke: V1 sync + hard sampling + paired freeze/checkpoint.
# These permissive freeze thresholds validate lifecycle wiring, NOT convergence.
# Override SPECO_* model/data/output paths for your environment.
set -euo pipefail
export SPECO_RUN_NAME="${SPECO_RUN_NAME:-v1-hard-freeze-smoke}"
export SPECO_TRAINER_MODE="${SPECO_TRAINER_MODE:-sync}"
export SPECO_TRAINER_GPUS="${SPECO_TRAINER_GPUS:-16}"
export SPECO_ROLLOUT_GPUS="${SPECO_ROLLOUT_GPUS:-16}"
export SPECO_TOTAL_STEPS="${SPECO_TOTAL_STEPS:-5}"
export SPECO_TRAIN_BATCH_SIZE="${SPECO_TRAIN_BATCH_SIZE:-8}"
export SPECO_ROLLOUT_N="${SPECO_ROLLOUT_N:-2}"
export SPECO_PPO_MINI_BATCH_SIZE="${SPECO_PPO_MINI_BATCH_SIZE:-16}"
export SPECO_TRAIN_SP="${SPECO_TRAIN_SP:-4}"
export SPECO_GEN_TP="${SPECO_GEN_TP:-2}"
export SPECO_MAX_RESPONSE_LENGTH="${SPECO_MAX_RESPONSE_LENGTH:-512}"
export SPECO_ROLLOUT_MEMORY_UTILIZATION="${SPECO_ROLLOUT_MEMORY_UTILIZATION:-0.6}"
export SPECO_CUDAGRAPH_CAPTURE_SIZES="${SPECO_CUDAGRAPH_CAPTURE_SIZES:-[1, 2, 4, 8, 16, 24, 32]}"
export SPECO_MAX_CUDAGRAPH_CAPTURE_SIZE="${SPECO_MAX_CUDAGRAPH_CAPTURE_SIZE:-32}"
export SPECO_DRAFTER_ENABLE="${SPECO_DRAFTER_ENABLE:-true}"
export SPECO_DRAFTER_TRAINING_ENABLE="${SPECO_DRAFTER_TRAINING_ENABLE:-true}"
export SPECO_DRAFTER_STEPS_PER_TRIGGER="${SPECO_DRAFTER_STEPS_PER_TRIGGER:-2}"
export SPECO_DRAFTER_BATCH_SIZE="${SPECO_DRAFTER_BATCH_SIZE:-2}"
export SPECO_DRAFTER_WINDOW_TOKENS="${SPECO_DRAFTER_WINDOW_TOKENS:-128}"
export SPECO_DRAFTER_MAX_COLLECT_TOKENS="${SPECO_DRAFTER_MAX_COLLECT_TOKENS:-4096}"
export SPECO_DRAFTER_COLLECT_INTERVAL="${SPECO_DRAFTER_COLLECT_INTERVAL:-1}"
export SPECO_DRAFTER_TRAINING_INTERVAL="${SPECO_DRAFTER_TRAINING_INTERVAL:-1}"
export SPECO_DRAFTER_PUBLISH_INTERVAL="${SPECO_DRAFTER_PUBLISH_INTERVAL:-1}"
export SPECO_PUBLISH_ASYNC="${SPECO_PUBLISH_ASYNC:-false}"
export SPECO_DRAFT_WEIGHT_BUCKET_MB="${SPECO_DRAFT_WEIGHT_BUCKET_MB:-512}"
export SPECO_WEIGHT_BUCKET_MB="${SPECO_WEIGHT_BUCKET_MB:-3072}"
export SPECO_DSPARK_TARGET_LM_HEAD_ROW_RESTRICTED_SYNC="${SPECO_DSPARK_TARGET_LM_HEAD_ROW_RESTRICTED_SYNC:-false}"
export SPECO_SAVE_FREQ="${SPECO_SAVE_FREQ:--1}"
export SPECO_TEST_FREQ="${SPECO_TEST_FREQ:--1}"

export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export VERL_SPECO_VLLM_NPU_STAGING="${VERL_SPECO_VLLM_NPU_STAGING:-0}"
export SPECO_JEMALLOC_RECLAIM_MODE="${SPECO_JEMALLOC_RECLAIM_MODE:-purge}"
export MALLOC_CONF="${MALLOC_CONF:-narenas:8,thp:never,metadata_thp:disabled,dirty_decay_ms:0,muzzy_decay_ms:0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

case "${LD_PRELOAD:-}" in
    *libjemalloc*) ;;
    *)
        if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
        elif [[ -f /usr/lib64/libjemalloc.so.2 ]]; then
            export LD_PRELOAD="/usr/lib64/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
        fi
        ;;
esac

: "${SPECO_RUN_NAME:?wrapper must set SPECO_RUN_NAME}"
: "${SPECO_TRAINER_MODE:?wrapper must set SPECO_TRAINER_MODE}"
: "${SPECO_TRAINER_GPUS:?wrapper must set SPECO_TRAINER_GPUS}"
: "${SPECO_ROLLOUT_GPUS:?wrapper must set SPECO_ROLLOUT_GPUS}"

project_name="${SPECO_PROJECT_NAME:-verl_speco_v1_mrv2_longrun}"
output_root="${SPECO_OUTPUT_ROOT:-/data/c00954340/speco-v1-longruns}"
run_dir="${output_root}/${SPECO_RUN_NAME}"
log_dir="${run_dir}/logs"
mkdir -p "${log_dir}"

model_path="${SPECO_MODEL_PATH:-/data/c00954340/qwen3-8b}"
drafter_path="${SPECO_DRAFTER_PATH:-/data/c00954340/qwen3-8b-dspark}"
train_file="${SPECO_TRAIN_FILE:-/data/c00954340/DAPO-Math-17k/dapo-math-17k.parquet}"
test_file="${SPECO_TEST_FILE:-/data/c00954340/AIME-2024/aime-2024.parquet}"
feature_store_path="${SPECO_FEATURE_STORE_PATH:-}"

total_steps="${SPECO_TOTAL_STEPS:-500}"
save_freq="${SPECO_SAVE_FREQ:--1}"
test_freq="${SPECO_TEST_FREQ:--1}"
resume_mode="${SPECO_RESUME_MODE:-disable}"
gen_tp="${SPECO_GEN_TP:-2}"
train_sp="${SPECO_TRAIN_SP:-8}"
capture_sizes="${SPECO_CUDAGRAPH_CAPTURE_SIZES:-[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]}"
max_capture_size="${SPECO_MAX_CUDAGRAPH_CAPTURE_SIZE:-512}"
cudagraph_mode="${SPECO_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
rollout_memory_utilization="${SPECO_ROLLOUT_MEMORY_UTILIZATION:-0.5}"
draft_load_format="${SPECO_DRAFT_LOAD_FORMAT:-auto}"
drafter_enable="${SPECO_DRAFTER_ENABLE:-true}"
drafter_training_enable="${SPECO_DRAFTER_TRAINING_ENABLE:-true}"
data_shuffle="${SPECO_DATA_SHUFFLE:-true}"
feature_store_shuffle="${SPECO_FEATURE_STORE_SHUFFLE:-true}"
# SpeCo's V1 acceptance-length bridge consumes the vLLM scheduler statistics.
# verl disables those statistics by default, which leaves the bridge with zero
# counters even though speculative decoding is active. Keep this override
# configurable for throughput-only runs.
rollout_disable_log_stats="${SPECO_ROLLOUT_DISABLE_LOG_STATS:-false}"
#ray_num_cpus="${SPECO_RAY_NUM_CPUS:-128}"
ray_worker_soft_limit="${SPECO_RAY_WORKER_SOFT_LIMIT:-64}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/${SPECO_RUN_NAME}_${timestamp}.log"

# V1 standalone vLLM servers are Ray actors created after the trainer. Their
# process config only has upstream RolloutConfig fields, so carry the DSpark
# launch essentials in Ray's inherited environment too. Otherwise they fall
# back to speco_base.yaml's placeholder drafter path.
export VERL_SPECO_DRAFTER_CONFIG="{\"enable\":${drafter_enable},\"enable_drafter_training\":${drafter_training_enable},\"speculative_algorithm\":\"DSPARK\",\"model_path\":\"${drafter_path}\",\"rollout\":{\"spec_steps\":${SPECO_DRAFTER_SPEC_STEPS:-1},\"spec_topk\":${SPECO_DRAFTER_SPEC_TOPK:-1},\"spec_verify_tokens\":${SPECO_DRAFTER_SPEC_VERIFY_TOKENS:-7}}}"

extra_args=()
if [[ -n "${feature_store_path}" ]]; then
    extra_args+=(
        "actor_rollout_ref.rollout.drafter.training.feature_store.type=torch_shard"
        "actor_rollout_ref.rollout.drafter.training.feature_store.path=${feature_store_path}"
        "actor_rollout_ref.rollout.drafter.training.feature_store.max_samples_per_shard=${SPECO_FEATURE_STORE_SHARD_SAMPLES:-256}"
        "actor_rollout_ref.rollout.drafter.training.feature_store.flush_interval_steps=1"
    )
fi

# Qwen3-8B's fp32 embedding tensor alone is about 2374 MiB during the
# standalone V1 initial publish, so the upstream 1200 MiB default cannot hold
# a single tensor. Keep this overridable for smaller models.
PYTHONUNBUFFERED=1 python3 -m verl_speco.main --config-name=speco_v1_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_file}" \
    data.val_files="${test_file}" \
    data.train_batch_size="${SPECO_TRAIN_BATCH_SIZE:-64}" \
    data.shuffle="${data_shuffle}" \
    data.max_prompt_length="${SPECO_MAX_PROMPT_LENGTH:-512}" \
    data.max_response_length="${SPECO_MAX_RESPONSE_LENGTH:-1024}" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr="${SPECO_ACTOR_LR:-1e-6}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${SPECO_PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${SPECO_PPO_MICRO_BATCH_SIZE:-2}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${train_sp}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${SPECO_LOGPROB_MICRO_BATCH_SIZE:-10}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${train_sp}" \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="${capture_sizes}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size="${max_capture_size}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="${cudagraph_mode}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${SPECO_ROLLOUT_GPUS}" \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${SPECO_LOGPROB_MICRO_BATCH_SIZE:-2}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_memory_utilization}" \
    actor_rollout_ref.rollout.max_model_len="${SPECO_ROLLOUT_MAX_MODEL_LEN:-null}" \
    actor_rollout_ref.rollout.disable_log_stats="${rollout_disable_log_stats}" \
    actor_rollout_ref.rollout.max_num_seqs="${SPECO_MAX_NUM_SEQS:-64}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${SPECO_MAX_BATCHED_TOKENS:-8192}" \
    actor_rollout_ref.rollout.n="${SPECO_ROLLOUT_N:-1}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${SPECO_WEIGHT_BUCKET_MB:-2500}" \
    actor_rollout_ref.rollout.drafter.enable="${drafter_enable}" \
    actor_rollout_ref.rollout.drafter.enable_drafter_training="${drafter_training_enable}" \
    actor_rollout_ref.rollout.drafter.model_path="${drafter_path}" \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    +actor_rollout_ref.rollout.drafter.vllm.speculative_config_overrides.draft_load_config.load_format="${draft_load_format}" \
    actor_rollout_ref.rollout.drafter.training.mode="${SPECO_DRAFTER_TRAINING_MODE:-online}" \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer="${SPECO_USE_DATA_BUFFER:-true}" \
    actor_rollout_ref.rollout.drafter.training.data_buffer_max_size="${SPECO_DATA_BUFFER_MAX_SIZE:-2048}" \
    actor_rollout_ref.rollout.drafter.training.feature_store.shuffle="${feature_store_shuffle}" \
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps="${SPECO_SAMPLE_LAST_N_STEPS:-2}" \
    actor_rollout_ref.rollout.drafter.training.step="${SPECO_DRAFTER_STEPS_PER_TRIGGER:-20}" \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu="${SPECO_DRAFTER_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica="${SPECO_DRAFTER_MAX_COLLECT_SAMPLES:-16}" \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample="${SPECO_DRAFTER_WINDOW_TOKENS:-512}" \
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica="${SPECO_DRAFTER_MAX_COLLECT_TOKENS:-16384}" \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps="${SPECO_DRAFTER_COLLECT_INTERVAL:-1}" \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps="${SPECO_DRAFTER_TRAINING_INTERVAL:-1}" \
    actor_rollout_ref.rollout.drafter.training.publish_interval_steps="${SPECO_DRAFTER_PUBLISH_INTERVAL:-1}" \
    actor_rollout_ref.rollout.drafter.training.publish_async="${SPECO_PUBLISH_ASYNC:-true}" \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes="${SPECO_DRAFT_WEIGHT_BUCKET_MB:-1200}" \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation="${SPECO_DRAFT_UPDATE_PAUSE_GENERATION:-true}" \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before="${SPECO_DRAFT_UPDATE_FLUSH_BEFORE:-true}" \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after="${SPECO_DRAFT_UPDATE_FLUSH_AFTER:-true}" \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=32 \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=restricted_ce \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=256 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=vanilla \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync="${SPECO_DSPARK_TARGET_LM_HEAD_ROW_RESTRICTED_SYNC:-true}" \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1 \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.9 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens="${SPECO_DSPARK_VERIFY_TOKENS:-7}" \
    trainer.v1.trainer_mode="${SPECO_TRAINER_MODE}" \
    trainer.v1.separate_async.parameter_sync_step="${SPECO_PARAMETER_SYNC_STEP:-1}" \
    trainer.n_gpus_per_node="${SPECO_TRAINER_GPUS}" \
    trainer.nnodes=1 \
    trainer.default_local_dir="${run_dir}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.total_training_steps="${total_steps}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${SPECO_RUN_NAME}" \
    actor_rollout_ref.rollout.drafter.training.dspark_hard_candidate_ratio=0.4 \
    actor_rollout_ref.rollout.drafter.training.dspark_hard_sample_ratio=0.1 \
    actor_rollout_ref.rollout.drafter.training.dspark_collect_samples_per_step=16 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.enabled=true \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.mode=active \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.branch_checkpoint.enabled=true \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.observation.min_requests_per_version=8 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.observation.min_rollout_steps_per_version=1 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.window_updates=1 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.patience_updates=1 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.bootstrap_samples=40 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.practical_gain_all=1.0 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.practical_gain_hard=1.0 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.freeze_guard_drop_all=1.0 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.gain.freeze_guard_drop_hard=1.0 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.probe.enabled=true \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.probe.interval_updates=1 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.probe.prompt_count=16 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.probe.batch_size=16 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.probe.max_new_tokens=64 \
    actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.probe.min_timing_pairs=8 \
    trainer.total_epochs=100 \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1='"1"' \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER='"1"' \
    "${extra_args[@]}" "$@" \
    > "${log_file}" 2>&1

printf 'exit=0\n' > "${run_dir}/exit_code"
ln -sfn "${log_file}" "${run_dir}/run.log"
