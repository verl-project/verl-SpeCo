#!/usr/bin/env bash
# Launch the SpeCo C5 retention E2E on the GPU host (GPUs 1 and 3).
#
#   run_retention.sh <smoke|eagle|dflash|both> [SPECO_STEPS=N]
set -euo pipefail
ALGORITHM="${1:-smoke}"
ROOT="$HOME/0z5a-work/speco-retention"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"

SPECO_ROOT="$ROOT" \
SPECO_SRC="$ROOT/src/current" \
SPECO_PYTHON="$HOME/0z5a/bin/python" \
SPECO_GPUS="${SPECO_GPUS:-1,3}" \
SPECO_STEPS="${SPECO_STEPS:-20}" \
SPECO_RUN_TAG="$RUN_TAG" \
  bash "$ROOT/src/current/experiment/l40s/run_native_online.sh" "$ALGORITHM"

echo "run tag: $RUN_TAG"
ls -la "$ROOT/evidence/full-e2e/" | tail -6
