#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# Launch new paired runs; these do not recreate the historical random state.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
output_dir=${OUTPUT_DIR:-"$repo_root/outputs/opd-cotrain-reproduction"}
read -r -a seeds <<< "${SEEDS:-42 43 44}"
mkdir -p -- "$output_dir"
for seed in "${seeds[@]}"; do
    for mode in fixed_drafter co_train; do
        train=False
        if [[ "$mode" == co_train ]]; then train=True; fi
        run_dir="$output_dir/seed-$seed/$mode"
        mkdir -p -- "$run_dir"
        bash "$repo_root/examples/run_qwen3-8b_opd_actor_fsdp2_drafter_dspark_vllm.sh" \
            "$@" \
            "data.seed=$seed" \
            "actor_rollout_ref.actor.data_loader_seed=$seed" \
            "actor_rollout_ref.rollout.drafter.enable_drafter_training=$train" \
            "trainer.experiment_name=opd_${mode}_seed_${seed}" \
            "trainer.default_local_dir=$run_dir/checkpoints" \
            "trainer.resume_mode=disable" \
            2>&1 | tee "$run_dir/console.log"
    done
done
