#!/usr/bin/env bash
set -euo pipefail

platform="${1:-}"
backend="${2:-}"
drafter="${3:-}"

case "${platform}/${backend}/${drafter}" in
  gpu/vllm/eagle3)
    example="examples/run_qwen3-8b_drafter_eagle3_vllm.sh"
    ;;
  gpu/vllm/dflash)
    example="examples/run_qwen3-8b_drafter_dflash_vllm.sh"
    ;;
  gpu/vllm/dspark)
    example="examples/run_qwen3-8b_drafter_dspark_vllm.sh"
    ;;
  gpu/sglang/eagle3)
    example="examples/run_qwen3-8b_drafter_eagle3_sglang.sh"
    ;;
  gpu/sglang/dflash)
    example="examples/run_qwen3-8b_drafter_dflash_sglang.sh"
    ;;
  npu/vllm/eagle3)
    example="examples/run_qwen3-8b_drafter_eagle3_vllm_npu.sh"
    ;;
  npu/vllm/dflash)
    example="examples/run_qwen3-8b_drafter_dflash_vllm_npu.sh"
    ;;
  npu/vllm/dspark)
    example="examples/run_qwen3-8b_drafter_dspark_vllm_npu.sh"
    ;;
  npu/sglang/eagle3)
    example="examples/run_qwen3-8b_drafter_eagle3_sglang_npu.sh"
    ;;
  npu/sglang/dflash)
    example="examples/run_qwen3-8b_drafter_dflash_sglang.sh"
    ;;
  *)
    echo "usage: $0 {gpu|npu} {vllm|sglang} {eagle3|dflash|dspark}" >&2
    exit 2
    ;;
esac

required_vars=(
  SPECO_TARGET_MODEL
  SPECO_TRAIN_FILE
  SPECO_TEST_FILE
  SPECO_CKPT_DIR
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "required environment variable ${name} is not set" >&2
    exit 2
  fi
done

case "${drafter}" in
  eagle3)
    draft_model="${SPECO_EAGLE3_DRAFT_MODEL:-}"
    draft_algorithm="EAGLE3"
    ;;
  dflash)
    draft_model="${SPECO_DFLASH_DRAFT_MODEL:-}"
    draft_algorithm="DFLASH"
    ;;
  dspark)
    draft_model="${SPECO_DSPARK_DRAFT_MODEL:-}"
    draft_algorithm="DSPARK"
    ;;
esac
if [[ -z "${draft_model}" ]]; then
  echo "required ${drafter} draft model environment variable is not set" >&2
  exit 2
fi

accelerator_count="${SPECO_ACCELERATOR_COUNT:-1}"
tensor_parallel_size="${SPECO_TENSOR_PARALLEL_SIZE:-1}"
sequence_parallel_size="${SPECO_SEQUENCE_PARALLEL_SIZE:-1}"

if [[ "${platform}" == "npu" ]]; then
  if [[ "${SPECO_DRY_RUN:-false}" != "true" ]]; then
    physical_npu_count="$(python - <<'PY'
import torch
import torch_npu
print(torch.npu.device_count())
PY
)"
    if (( accelerator_count > physical_npu_count )); then
      echo "SPECO_ACCELERATOR_COUNT=${accelerator_count} exceeds physical NPU count ${physical_npu_count}" >&2
      exit 2
    fi
  fi
  if (( accelerator_count < 1 )); then
    echo "SPECO_ACCELERATOR_COUNT must be >= 1, got ${accelerator_count}" >&2
    exit 2
  fi
  if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    visible_devices=""
    for ((device_index = 0; device_index < accelerator_count; device_index++)); do
      if [[ -n "${visible_devices}" ]]; then
        visible_devices+=","
      fi
      visible_devices+="${device_index}"
    done
    export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
  else
    visible_count=1
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]]; then
      visible_count="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"
    fi
    if (( accelerator_count > visible_count )); then
      echo "SPECO_ACCELERATOR_COUNT=${accelerator_count} exceeds visible NPU count ${visible_count} from ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" >&2
      exit 2
    fi
  fi
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
  export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
  export HCCL_OP_EXPANSION_MOD="${HCCL_OP_EXPANSION_MOD:-AIV}"
  if [[ "${backend}" == "sglang" ]]; then
    export SGLANG_DEEPEP_BF16_DISPATCH="${SGLANG_DEEPEP_BF16_DISPATCH:-1}"
    export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
  fi
fi

enable_training="${SPECO_ENABLE_TRAINING:-true}"
total_epochs="${SPECO_TOTAL_EPOCHS:-1}"
if [[ "${enable_training}" != "true" ]]; then
  total_epochs="${SPECO_GENERATION_ONLY_EPOCHS:-1}"
fi

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

overrides=(
  "actor_rollout_ref.model.path=${SPECO_TARGET_MODEL}"
  "actor_rollout_ref.rollout.drafter.model_path=${draft_model}"
  "actor_rollout_ref.rollout.drafter.speculative_algorithm=${draft_algorithm}"
  "actor_rollout_ref.rollout.drafter.enable=True"
  "actor_rollout_ref.rollout.drafter.enable_drafter_training=${enable_training}"
  "data.train_files=${SPECO_TRAIN_FILE}"
  "data.val_files=${SPECO_TEST_FILE}"
  "trainer.default_local_dir=${SPECO_CKPT_DIR}"
  "trainer.n_gpus_per_node=${accelerator_count}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_parallel_size}"
  "actor_rollout_ref.rollout.drafter.vllm.draft_tensor_parallel_size=${SPECO_DRAFT_TENSOR_PARALLEL_SIZE:-${tensor_parallel_size}}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sequence_parallel_size}"
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sequence_parallel_size}"
  "data.train_batch_size=${SPECO_TRAIN_BATCH_SIZE:-1}"
  "data.max_prompt_length=${SPECO_MAX_PROMPT_LENGTH:-256}"
  "data.max_response_length=${SPECO_MAX_RESPONSE_LENGTH:-64}"
  "actor_rollout_ref.rollout.n=${SPECO_ROLLOUT_N:-1}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${SPECO_PPO_MINI_BATCH_SIZE:-1}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${SPECO_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${SPECO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${SPECO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.rollout.drafter.rollout.spec_steps=${SPECO_SPEC_STEPS:-3}"
  "actor_rollout_ref.rollout.drafter.rollout.spec_topk=${SPECO_SPEC_TOPK:-1}"
  "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPECO_SPEC_VERIFY_TOKENS:-4}"
  "actor_rollout_ref.rollout.drafter.training.step=${SPECO_DRAFTER_TRAINING_STEPS:-1}"
  "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${SPECO_DRAFTER_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.rollout.drafter.training.collect_interval_steps=${SPECO_COLLECT_INTERVAL_STEPS:-1}"
  "actor_rollout_ref.rollout.drafter.training.training_interval_steps=${SPECO_TRAINING_INTERVAL_STEPS:-1}"
  "actor_rollout_ref.rollout.drafter.training.publish_interval_steps=${SPECO_PUBLISH_INTERVAL_STEPS:-1}"
  "actor_rollout_ref.rollout.drafter.training.train_batches_per_cycle=${SPECO_TRAIN_BATCHES_PER_CYCLE:-1}"
  "actor_rollout_ref.rollout.drafter.training.publish_async=${SPECO_PUBLISH_ASYNC:-False}"
  "actor_rollout_ref.rollout.drafter.training.publish_dtype=${SPECO_PUBLISH_DTYPE:-bf16}"
  "actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=${SPECO_DRAFT_UPDATE_FLUSH_BEFORE:-True}"
  "actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=${SPECO_DRAFT_UPDATE_FLUSH_AFTER:-True}"
  "trainer.logger=${SPECO_TRAINER_LOGGER:-[\"console\"]}"
  "trainer.val_before_train=${SPECO_VAL_BEFORE_TRAIN:-False}"
  "trainer.save_freq=${SPECO_SAVE_FREQ:--1}"
  "trainer.test_freq=${SPECO_TEST_FREQ:--1}"
  "trainer.total_epochs=${total_epochs}"
  "trainer.total_training_steps=${SPECO_TOTAL_TRAINING_STEPS:-2}"
  "data.train_max_samples=${SPECO_TRAIN_MAX_SAMPLES:-1}"
  "data.val_max_samples=${SPECO_VAL_MAX_SAMPLES:-1}"
  "data.dataloader_num_workers=${SPECO_DATALOADER_NUM_WORKERS:-0}"
)

if [[ "${drafter}" == "dflash" ]]; then
  overrides+=(
    "actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=${SPECO_HIDDEN_STATE_WINDOW_MIN_ROWS:-1}"
    "actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=${SPECO_HIDDEN_STATE_WINDOW_TOKENS_PER_SAMPLE:-64}"
    "actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=${SPECO_DFLASH_NUM_ANCHORS:-8}"
    "actor_rollout_ref.rollout.drafter.training.dflash_max_window=${SPECO_DFLASH_MAX_WINDOW:-64}"
    "actor_rollout_ref.rollout.drafter.training.dflash_loss_decay_gamma=${SPECO_DFLASH_LOSS_DECAY_GAMMA:-7}"
    "actor_rollout_ref.rollout.drafter.training.dflash_front_position_weight=${SPECO_DFLASH_FRONT_POSITION_WEIGHT:-2.0}"
    "actor_rollout_ref.rollout.drafter.training.dflash_front_position_count=${SPECO_DFLASH_FRONT_POSITION_COUNT:-3}"
    "actor_rollout_ref.rollout.drafter.training.dflash_hard_sample_ratio=${SPECO_DFLASH_HARD_SAMPLE_RATIO:-0.3}"
  )
fi

if [[ "${drafter}" == "dspark" ]]; then
  overrides+=(
    "actor_rollout_ref.rollout.drafter.rollout.spec_steps=${SPECO_DSPARK_SPEC_STEPS:-1}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPECO_DSPARK_SPEC_VERIFY_TOKENS:-7}"
    "actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=${SPECO_HIDDEN_STATE_WINDOW_MIN_ROWS:-1}"
    "actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=${SPECO_HIDDEN_STATE_WINDOW_TOKENS_PER_SAMPLE:-64}"
    "actor_rollout_ref.rollout.drafter.training.dspark_block_size=${SPECO_DSPARK_BLOCK_SIZE:-7}"
    "actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=${SPECO_DSPARK_NUM_ANCHORS:-8}"
    "actor_rollout_ref.rollout.drafter.training.dspark_max_window=${SPECO_DSPARK_MAX_WINDOW:-64}"
  )
fi

if [[ -n "${SPECO_EXTRA_HYDRA_ARGS:-}" ]]; then
  while IFS= read -r extra_arg; do
    [[ -z "${extra_arg}" ]] && continue
    overrides+=("${extra_arg}")
  done <<< "${SPECO_EXTRA_HYDRA_ARGS}"
fi

if [[ "${SPECO_DRY_RUN:-false}" == "true" ]]; then
  echo "platform=${platform}"
  echo "backend=${backend}"
  echo "drafter=${drafter}"
  echo "example=${example}"
  echo "draft_algorithm=${draft_algorithm}"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'Hydra overrides:\n'
  printf '  %q\n' "${overrides[@]}"
  exit 0
fi

bash "${example}" "${overrides[@]}"
