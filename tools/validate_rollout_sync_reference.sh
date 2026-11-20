#!/usr/bin/env bash
set -uo pipefail
: "${SPECO_CAPTURE_TARGET:?Set target checkpoint}"
: "${SPECO_CAPTURE_DRAFT:?Set draft checkpoint}"
python -c 'import torch, vllm; assert torch.cuda.is_available(), "GPU validation requires CUDA"' || exit 2
result=0
export SPECO_CAPTURE_GPU_FILTER=0
export SPECO_CAPTURE_BENCH_MAX_TOKENS=512
export SPECO_CAPTURE_BENCH_REPEATS=5
export SPECO_CAPTURE_PACK_CHUNKS=0
export SPECO_CAPTURE_TOKEN_FIXTURE="${SPECO_CAPTURE_TOKEN_FIXTURE:-$PWD/sync-reference-tokens.json}"
export SPECO_CAPTURE_FORWARD_AUDIT=0
export SPECO_CAPTURE_VERIFY_TRANSPORT=0
export SPECO_CAPTURE_DIAGNOSTICS=0
for mode in inline reference; do
  if [[ "$mode" == reference ]]; then
    export SPECO_CAPTURE_TRAINING_JSON='{"hidden_state_window_tokens_per_sample":512,"hidden_state_window_min_rows":512,"vllm_capture_reference":true}'
  else
    export SPECO_CAPTURE_TRAINING_JSON='{"hidden_state_window_tokens_per_sample":512,"hidden_state_window_min_rows":512}'
  fi
  export SPECO_CAPTURE_EXPORT_ROOT="${SPECO_CAPTURE_TEST_ROOT:-$PWD/capture-sync-validation}/$mode"
  mkdir -p "$SPECO_CAPTURE_EXPORT_ROOT"
  python -m pytest 'tests/heavy/test_vllm_rollout_capture_gpu.py::test_rollout_features_match_reference[True-True]' -v -s 2>&1 | tee "sync-$mode.log"
  status=${PIPESTATUS[0]}
  printf 'mode=%s pytest_exit=%s\n' "$mode" "$status"
  if (( status != 0 )); then result=$status; fi
done
# These exercise the exporter/store directly. Actual V1/TQ on/off and actor
# baseline must also run with equal sample selection and training quota.

exit "$result"
