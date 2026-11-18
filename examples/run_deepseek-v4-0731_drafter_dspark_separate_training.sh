#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Run from the verl-SpeCo repository root in the matching Ascend environment.
# Stages: import | preflight | train | resume | export
STAGE=${1:-preflight}
shift "$(( $# > 0 ? 1 : 0 ))"
: "${MODEL_PATH:?Set MODEL_PATH to the exact DeepSeek V4 0731 DSpark checkpoint}"
: "${FEATURE_STORE:?Set FEATURE_STORE to a new/imported SpeCo store path}"
CARD_COUNT=${CARD_COUNT:-8}
TRAIN_TP=${TRAIN_TP:-8}
RUN_DIR=${RUN_DIR:-outputs/ds4_draft_0731_smoke}
PYTHON_BIN=${PYTHON_BIN:-python}

# Keep direct preflight and distributed training on the same required policy.
export HCCL_DETERMINISTIC=strict
export LCCL_DETERMINISTIC=1

COMMON=(
    "--config-name=deepseek_v4_0731_draft_trainer"
    "actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.rollout.drafter.training.feature_store.path=${FEATURE_STORE}"
    "speco.draft_training.nproc_per_node=${CARD_COUNT}"
    "draft_engine.output_dir=${RUN_DIR}"
    "++draft_engine.overrides.model.vllm_tensor_parallel_size=${TRAIN_TP}"
    # A bounded smoke recipe. The same overrides must be used when resuming.
    "++draft_engine.overrides.train.epochs=2"
    "++draft_engine.overrides.train.segment_length=512"
    "++draft_engine.overrides.train.segment_overlap=64"
    "++draft_engine.overrides.train.dspark_num_anchors=4"
    "++draft_engine.overrides.train.dspark_anchor_min_spacing=8"
    "++draft_engine.overrides.deepspeed.train_micro_batch_size_per_gpu=1"
    "++draft_engine.overrides.deepspeed.gradient_accumulation_steps=1"
    "++draft_engine.overrides.optimizations.dspark.anchor_micro_batch_size=4"
)

case "$STAGE" in
    import)
        : "${CAPTURE_DIR:?Set CAPTURE_DIR to the AIR mtp_dataset directory}"
        : "${MASKS_PATH:?Set MASKS_PATH to the exact capture SpeCo loss_mask JSONL}"
        "$PYTHON_BIN" -m verl_speco.trainer.deepseek_v4_dataset \
            --model "$MODEL_PATH" --capture "$CAPTURE_DIR" \
            --masks "$MASKS_PATH" --output "$FEATURE_STORE" "$@"
        ;;
    preflight)
        "$PYTHON_BIN" -m verl_speco.draft_train "${COMMON[@]}" \
            draft_engine.action=preflight draft_engine.require_runtime=true "$@"
        ;;
    train|resume)
        if [[ "$STAGE" == resume ]]; then
            : "${RESUME_FROM:?Set RESUME_FROM to the previous step checkpoint directory}"
            COMMON+=("draft_engine.resume_from=${RESUME_FROM}")
        fi
        "$PYTHON_BIN" -m verl_speco.draft_train_launcher \
            "${COMMON[@]}" \
            draft_engine.action=train \
            "actor_rollout_ref.rollout.drafter.training.max_steps=${MAX_STEPS:-1}" "$@"
        ;;
    export)
        : "${CHECKPOINT:?Set CHECKPOINT to a completed SpeCo AIR step directory}"
        : "${EXPORT_DIR:?Set EXPORT_DIR to a new serving checkpoint directory}"
        "$PYTHON_BIN" -m verl_speco.export_deepseek_v4_draft \
            --checkpoint "$CHECKPOINT" \
            --model "$MODEL_PATH" --output "$EXPORT_DIR" "$@"
        ;;
    *)
        echo "Unknown stage: $STAGE (import | preflight | train | resume | export)" >&2
        exit 2
        ;;
esac
