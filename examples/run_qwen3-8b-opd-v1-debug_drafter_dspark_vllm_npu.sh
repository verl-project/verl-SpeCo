#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export SPECO_CONFIG_NAME=speco_v1_trainer
bash examples/run_qwen3-8b-opd-debug_drafter_dspark_vllm_npu.sh \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=true \
    ray_kwargs.ray_init.num_cpus=64 \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=16 \
    "$@"
