#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
# Historical V1 sync online dimensions, with hard0.4/0.1 and normal freeze policy.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPECO_RUN_NAME="${SPECO_RUN_NAME:-v1-hard-freeze-longrun}"
export SPECO_TOTAL_STEPS="${SPECO_TOTAL_STEPS:-100}"
export SPECO_TRAIN_BATCH_SIZE=64
export SPECO_ROLLOUT_N=5
export SPECO_MAX_RESPONSE_LENGTH=8192
export SPECO_PPO_MINI_BATCH_SIZE=16
export SPECO_ROLLOUT_MEMORY_UTILIZATION="${SPECO_ROLLOUT_MEMORY_UTILIZATION:-0.7}"
export SPECO_CUDAGRAPH_CAPTURE_SIZES='[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]'
export SPECO_MAX_CUDAGRAPH_CAPTURE_SIZE=512
export SPECO_DRAFTER_STEPS_PER_TRIGGER=20
export SPECO_DRAFTER_BATCH_SIZE=4
export SPECO_DRAFTER_WINDOW_TOKENS=512
export SPECO_DRAFTER_MAX_COLLECT_TOKENS=16384
export SPECO_DRAFTER_COLLECT_INTERVAL=5
export SPECO_DRAFTER_TRAINING_INTERVAL=5
export SPECO_DRAFTER_PUBLISH_INTERVAL=0
export SPECO_SAMPLE_LAST_N_STEPS=20
export SPECO_USE_DATA_BUFFER=true
export SPECO_DATA_BUFFER_MAX_SIZE=1024
export SPECO_FEATURE_STORE_SHUFFLE=false
freeze=actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze
exec bash "${script_dir}/run_qwen3-8b_v1_hard_freeze_smoke_drafter_dspark_vllm_npu.sh" \
    actor_rollout_ref.rollout.drafter.training.dspark_collect_samples_per_step=80 \
    "${freeze}.observation.min_requests_per_version=256" \
    "${freeze}.observation.min_rollout_steps_per_version=2" \
    "${freeze}.gain.window_updates=3" \
    "${freeze}.gain.patience_updates=3" \
    "${freeze}.gain.bootstrap_samples=300" \
    "${freeze}.gain.practical_gain_all=0.005" \
    "${freeze}.gain.practical_gain_hard=0.005" \
    "${freeze}.gain.freeze_guard_drop_all=0.02" \
    "${freeze}.gain.freeze_guard_drop_hard=0.03" \
    "${freeze}.probe.interval_updates=3" \
    "${freeze}.probe.prompt_count=256" \
    "${freeze}.probe.batch_size=64" \
    "${freeze}.probe.max_new_tokens=2048" \
    "${freeze}.probe.min_timing_pairs=16" \
    "$@"
