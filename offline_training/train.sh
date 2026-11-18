#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash offline_training/train.sh [preflight|train|resume|export|import] [options]
Required environment: MODEL_PATH, FEATURE_STORE.
Defaults: CARD_COUNT=16 TRAIN_TP=8 MAX_STEPS=1 PYTHON_BIN=python3
          RUN_DIR=outputs/deepseek_v4_offline/<stage>
resume: set RESUME_FROM to a full step checkpoint; MAX_STEPS is cumulative.
export: set CHECKPOINT and EXPORT_DIR; default format is draft-pytorch.
import: set CAPTURE_DIR and MASKS_PATH for existing legacy capture files.
Set DRAFT_ALGORITHM=MTP for native MTP training (default: DSPARK).
Training/preflight options are Hydra overrides; export/import options are CLI flags.
Reuses examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh.
HELP
    exit 0
fi

STAGE=${1:-preflight}
shift "$(( $# > 0 ? 1 : 0 ))"
case "$STAGE" in
    preflight|train|resume|export|import) ;;
    *) echo "Unknown stage: $STAGE (preflight | train | resume | export | import)" >&2; exit 2 ;;
esac
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$REPO_ROOT"

export CARD_COUNT=${CARD_COUNT:-16}
export TRAIN_TP=${TRAIN_TP:-8}
export PYTHON_BIN=${PYTHON_BIN:-python3}
case ${DRAFT_ALGORITHM:-DSPARK} in
    DSPARK|dspark) ;;
    MTP|mtp)
        : "${MODEL_PATH:?Set MODEL_PATH to a native DeepSeek V4 MTP checkpoint}"
        : "${FEATURE_STORE:?Set FEATURE_STORE to the matching MTP feature store}"
        export RUN_DIR=${RUN_DIR:-outputs/deepseek_v4_offline/mtp/$STAGE}
        export HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1
        COMMON=(
            --config-name=deepseek_v4_mtp_draft_trainer
            actor_rollout_ref.rollout.drafter.training.backend=vllm_ascend
            "actor_rollout_ref.model.path=$MODEL_PATH"
            "actor_rollout_ref.rollout.drafter.training.feature_store.path=$FEATURE_STORE"
            "speco.draft_training.nproc_per_node=$CARD_COUNT"
            "draft_engine.output_dir=$RUN_DIR"
            "++draft_engine.overrides.model.vllm_tensor_parallel_size=$TRAIN_TP"
        )
        case "$STAGE" in
            preflight)
                exec "$PYTHON_BIN" -m verl_speco.draft_train "${COMMON[@]}" draft_engine.action=preflight draft_engine.require_runtime=true "$@"
                ;;
            train|resume)
                if [[ $STAGE == resume ]]; then
                    : "${RESUME_FROM:?Set RESUME_FROM to a full MTP step checkpoint}"
                    COMMON+=("draft_engine.resume_from=$RESUME_FROM")
                fi
                exec "$PYTHON_BIN" -m verl_speco.draft_train_launcher "${COMMON[@]}" draft_engine.action=train "actor_rollout_ref.rollout.drafter.training.max_steps=${MAX_STEPS:-1}" "$@"
                ;;
            export)
                : "${CHECKPOINT:?Set CHECKPOINT to a full MTP step checkpoint}"
                : "${EXPORT_DIR:?Set EXPORT_DIR to a new artifact directory}"
                exec "$PYTHON_BIN" -m verl_speco.export_deepseek_v4_draft --checkpoint "$CHECKPOINT" --model "$MODEL_PATH" --output "$EXPORT_DIR" --format draft-pytorch "$@"
                ;;
            import)
                : "${CAPTURE_DIR:?Set CAPTURE_DIR to the existing MTP capture directory}"
                : "${MASKS_PATH:?Set MASKS_PATH to explicit SpeCo loss masks}"
                exec "$PYTHON_BIN" -m verl_speco.trainer.deepseek_v4_dataset --model "$MODEL_PATH" --capture "$CAPTURE_DIR" --masks "$MASKS_PATH" --output "$FEATURE_STORE" --algorithm MTP "$@"
                ;;
        esac
        ;;
    *) echo "DRAFT_ALGORITHM must be DSPARK or MTP" >&2; exit 2 ;;
esac
export RUN_DIR=${RUN_DIR:-outputs/deepseek_v4_offline/$STAGE}

# The existing exporter defaults to a full patched target. This offline entry
# defaults to the small standard draft artifact verified by the current run.
if [[ $STAGE == export ]]; then
    set -- --format draft-pytorch "$@"
fi
exec bash "$REPO_ROOT/examples/run_deepseek-v4-0731_drafter_dspark_separate_training.sh" "$STAGE" "$@"
