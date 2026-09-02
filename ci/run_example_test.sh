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
  gpu/vllm/dflash2)
    example="examples/run_qwen3-8b_drafter_dflash2_vllm.sh"
    ;;
  gpu/vllm/peagle|gpu/vllm/domino)
    example="examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh"
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
  npu/vllm/megatron-eagle3)
    example="examples/run_qwen3-4b_actor_megatron_drafter_eagle3_vllm_npu.sh"
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
    echo "usage: $0 {gpu|npu} {vllm|sglang} {eagle3|megatron-eagle3|dflash|dflash2|dspark|peagle|domino}" >&2
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

# Domino and P-EAGLE are not engine-level speculative algorithms, so neither can
# be trained inside the rollout loop. They use the two-stage separate-training
# workflow instead: stage 1 rolls out with the engine algorithm whose
# hidden-state layout the drafter consumes and writes a feature store, stage 2
# trains the drafter offline from that same store.
# The example owns the collect-to-train algorithm pairing; the runner only needs
# the collect side of it to pick the matching cached draft model.
separate_training="false"

case "${drafter}" in
  eagle3|megatron-eagle3)
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
  dflash2)
    draft_model="${SPECO_DFLASH2_DRAFT_MODEL:-}"
    draft_algorithm="DFLASH2"
    ;;
  peagle)
    draft_model="${SPECO_EAGLE3_DRAFT_MODEL:-}"
    draft_algorithm="EAGLE3"
    separate_training="true"
    ;;
  domino)
    draft_model="${SPECO_DFLASH_DRAFT_MODEL:-}"
    draft_algorithm="DFLASH"
    separate_training="true"
    ;;
esac
if [[ -z "${draft_model}" ]]; then
  echo "required ${drafter} collect-stage draft model environment variable is not set" >&2
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
      visible_commas="${ASCEND_RT_VISIBLE_DEVICES//[^,]/}"
      visible_count=$(( ${#visible_commas} + 1 ))
    fi
    if (( accelerator_count > visible_count )); then
      echo "SPECO_ACCELERATOR_COUNT=${accelerator_count} exceeds visible NPU count ${visible_count} from ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" >&2
      exit 2
    fi
  fi
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
  export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"
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

# The CI runner uses a short feature window for every drafter. A one-step
# rollout smoke test cannot populate the production 512-token window.
hidden_state_window_min_rows="${SPECO_HIDDEN_STATE_WINDOW_MIN_ROWS:-32}"
hidden_state_window_tokens_per_sample="${SPECO_HIDDEN_STATE_WINDOW_TOKENS_PER_SAMPLE:-32}"

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
  "actor_rollout_ref.rollout.drafter.training.hidden_state_window_min_rows=${hidden_state_window_min_rows}"
  "actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=${hidden_state_window_tokens_per_sample}"
)

if [[ "${drafter}" != "megatron-eagle3" ]]; then
  overrides+=(
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sequence_parallel_size}"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sequence_parallel_size}"
  )
fi

if [[ "${drafter}" == "dflash" ]]; then
  overrides+=(
    "actor_rollout_ref.rollout.drafter.training.dflash_num_anchors=${SPECO_DFLASH_NUM_ANCHORS:-8}"
    "actor_rollout_ref.rollout.drafter.training.dflash_max_window=${SPECO_DFLASH_MAX_WINDOW:-64}"
    "actor_rollout_ref.rollout.drafter.training.dflash_loss_decay_gamma=${SPECO_DFLASH_LOSS_DECAY_GAMMA:-7}"
    "actor_rollout_ref.rollout.drafter.training.dflash_front_position_weight=${SPECO_DFLASH_FRONT_POSITION_WEIGHT:-2.0}"
    "actor_rollout_ref.rollout.drafter.training.dflash_front_position_count=${SPECO_DFLASH_FRONT_POSITION_COUNT:-3}"
    "actor_rollout_ref.rollout.drafter.training.dflash_hard_sample_ratio=${SPECO_DFLASH_HARD_SAMPLE_RATIO:-0.3}"
  )
fi

if [[ "${drafter}" == "dflash2" ]]; then
  overrides+=(
    # vLLM sizes the DFlash2 convolution block as 1 + spec_verify_tokens, and
    # the trainer folds by dflash2_block_size (default 8), so the two must agree.
    "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPECO_DFLASH2_SPEC_VERIFY_TOKENS:-7}"
    "actor_rollout_ref.rollout.drafter.training.dflash2_block_size=${SPECO_DFLASH2_BLOCK_SIZE:-8}"
    "actor_rollout_ref.rollout.drafter.training.dflash2_num_anchors=${SPECO_DFLASH2_NUM_ANCHORS:-8}"
    "actor_rollout_ref.rollout.drafter.training.dflash2_loss_decay_gamma=${SPECO_DFLASH2_LOSS_DECAY_GAMMA:-7}"
    "actor_rollout_ref.rollout.drafter.training.dflash_max_window=${SPECO_DFLASH_MAX_WINDOW:-64}"
  )
fi

if [[ "${drafter}" == "dspark" ]]; then
  overrides+=(
    "actor_rollout_ref.rollout.drafter.rollout.spec_steps=${SPECO_DSPARK_SPEC_STEPS:-1}"
    "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${SPECO_DSPARK_SPEC_VERIFY_TOKENS:-7}"
    "actor_rollout_ref.rollout.drafter.training.dspark_block_size=${SPECO_DSPARK_BLOCK_SIZE:-7}"
    "actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=${SPECO_DSPARK_NUM_ANCHORS:-8}"
    "actor_rollout_ref.rollout.drafter.training.dspark_max_window=${SPECO_DSPARK_MAX_WINDOW:-64}"
  )
fi

train_overrides=()
if [[ "${separate_training}" == "true" ]]; then
  feature_store_dir="${SPECO_CKPT_DIR}/${drafter}_features"
  # A draft init path that does not exist yet cold-starts the drafter from the
  # target config, so this lane needs no extra model in the CI cache.
  draft_init_path="${SPECO_CKPT_DIR}/${drafter}_draft_init"

  # The example already sets the two-stage shape (collect_only/offline modes,
  # the standalone launcher, the feature-store flags and the per-algorithm
  # hyperparameters). Override only what CI has to control: where the stages
  # meet on disk, and the sizes that keep a smoke run cheap.
  # A configured SPECO_CKPT_DIR persists between jobs on a self-hosted runner,
  # so drop any earlier shards: otherwise a collect stage that produced nothing
  # still trains, and the job goes green on stale features.
  rm -rf "${feature_store_dir}" "${draft_init_path}"

  overrides+=(
    "actor_rollout_ref.rollout.drafter.training.feature_store.path=${feature_store_dir}"
    "actor_rollout_ref.rollout.drafter.training.feature_store.max_samples_per_shard=32"
  )

  train_overrides=(
    "actor_rollout_ref.model.path=${SPECO_TARGET_MODEL}"
    "actor_rollout_ref.rollout.drafter.model_path=${draft_init_path}"
    "actor_rollout_ref.rollout.drafter.checkpoint_path=${SPECO_CKPT_DIR}/${drafter}_draft_ckpts"
    "actor_rollout_ref.rollout.drafter.training.max_steps=1"
    "actor_rollout_ref.rollout.drafter.training.save_interval_steps=1"
    "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${SPECO_DRAFTER_BATCH_SIZE_PER_GPU:-1}"
    "actor_rollout_ref.rollout.drafter.training.feature_store.path=${feature_store_dir}"
  )

  case "${drafter}" in
    peagle)
      train_overrides+=(
        "actor_rollout_ref.rollout.drafter.training.peagle_num_draft_layers=1"
        "actor_rollout_ref.rollout.drafter.training.peagle_num_depths=2"
      )
      ;;
    domino)
      train_overrides+=(
        "actor_rollout_ref.rollout.drafter.training.domino_block_size=4"
        "actor_rollout_ref.rollout.drafter.training.domino_num_anchors=8"
        "actor_rollout_ref.rollout.drafter.training.domino_max_window=64"
        "actor_rollout_ref.rollout.drafter.training.domino_emb_dim=64"
        "actor_rollout_ref.rollout.drafter.training.domino_gru_hidden_dim=128"
        "actor_rollout_ref.rollout.drafter.training.domino_lambda_base_decay_steps=10"
      )
      ;;
  esac
fi

if [[ -n "${SPECO_EXTRA_HYDRA_ARGS:-}" ]]; then
  while IFS= read -r extra_arg; do
    [[ -z "${extra_arg}" ]] && continue
    # Stage 2 runs a different entrypoint with a different config tree, so these
    # rollout-oriented overrides stay on the collect stage.
    overrides+=("${extra_arg}")
  done <<< "${SPECO_EXTRA_HYDRA_ARGS}"
fi

if [[ "${SPECO_DRY_RUN:-false}" == "true" ]]; then
  echo "platform=${platform}"
  echo "backend=${backend}"
  echo "drafter=${drafter}"
  echo "example=${example}"
  echo "draft_algorithm=${draft_algorithm}"
  echo "separate_training=${separate_training}"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-}"
  printf 'Hydra overrides:\n'
  printf '  %q\n' "${overrides[@]}"
  if [[ "${separate_training}" == "true" ]]; then
    printf 'Hydra train overrides:\n'
    printf '  %q\n' "${train_overrides[@]}"
  fi
  exit 0
fi

if [[ "${separate_training}" == "true" ]]; then
  DRAFT_ALGO="${drafter}" RUN_STAGE=collect bash "${example}" "${overrides[@]}"
  if [[ "${enable_training}" == "true" ]]; then
    DRAFT_ALGO="${drafter}" RUN_STAGE=train DRAFT_TRAIN_GPUS_PER_NODE="${accelerator_count}" bash "${example}" "${train_overrides[@]}"
  else
    # Generation-only runs capture no hidden states, so the feature store the
    # offline trainer would read is empty.
    echo "SPECO_ENABLE_TRAINING is not true; skipping the offline train stage"
  fi
else
  bash "${example}" "${overrides[@]}"
fi
