#!/usr/bin/env bash
set -euo pipefail
cd /experiment
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export PYTHONPATH=/experiment/online-deps
export HF_HOME=/experiment/.cache/huggingface
out=/experiment/evidence/l20-20260920/c2-head-pruning
mkdir -p "$out"
for arm in A0 O0 P0 P1 O1 A1; do
  source=/experiment/variants/partition-before
  partitions=2
  if [[ "$arm" == A* ]]; then partitions=1; fi
  if [[ "$arm" == P* ]]; then source=/experiment/variants/partition-e2e; fi
  timeout -k 15 900 .venv-clean/bin/python benchmark_training.py \
    --source "$source" --target /experiment/models/target --tokens 1024 \
    --partitions "$partitions" --output "$out/$arm.json" > "$out/$arm.log" 2>&1
done
