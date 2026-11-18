#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash offline_training/collect_hs.sh [replay CLI options]
Required environment: MODEL_PATH, TRAJECTORIES, FEATURE_STORE (new directory).
Defaults: CAPTURE_TP=8 CAPTURE_DP=2 MAX_MODEL_LEN=4096
          MAX_NUM_BATCHED_TOKENS=512 MAX_SAMPLES=1
          CAPTURE_MAX_BYTES=536870912 PYTHON_BIN=python3
Set MAX_SAMPLES=all to capture the entire input. DP>1 enables expert parallel.
Set DRAFT_ALGORITHM=MTP for a native MTP checkpoint (default: DSPARK).
Pass --prepare-only to prepare token/mask data without loading the NPU model.
Additional arguments are forwarded to verl_speco.replay_deepseek_v4_features.
HELP
    exit 0
fi

: "${MODEL_PATH:?Set MODEL_PATH to the exact DeepSeek V4 0731 BF16 checkpoint}"
: "${TRAJECTORIES:?Set TRAJECTORIES to existing trajectory JSONL or a directory}"
: "${FEATURE_STORE:?Set FEATURE_STORE to a new output directory}"
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$REPO_ROOT"

ARGS=(
    --model "$MODEL_PATH"
    --trajectories "$TRAJECTORIES"
    --output "$FEATURE_STORE"
    --tensor-parallel-size "${CAPTURE_TP:-8}"
    --data-parallel-size "${CAPTURE_DP:-2}"
    --max-model-len "${MAX_MODEL_LEN:-4096}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-512}"
    --capture-max-bytes "${CAPTURE_MAX_BYTES:-536870912}"
)
case ${DRAFT_ALGORITHM:-DSPARK} in
    DSPARK|dspark) MODULE=verl_speco.replay_deepseek_v4_features ;;
    MTP|mtp) MODULE=verl_speco.replay_deepseek_v4_mtp_features ;;
    *) echo "DRAFT_ALGORITHM must be DSPARK or MTP" >&2; exit 2 ;;
esac
if [[ ${CAPTURE_DP:-2} != 1 ]]; then
    ARGS+=(--enable-expert-parallel)
fi
if [[ ${MAX_SAMPLES:-1} != all ]]; then
    ARGS+=(--max-samples "${MAX_SAMPLES:-1}")
fi
exec "${PYTHON_BIN:-python3}" -m "$MODULE" "${ARGS[@]}" "$@"
