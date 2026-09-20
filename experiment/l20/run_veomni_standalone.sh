#!/usr/bin/env bash
set -euo pipefail
engine=${1:?fsdp or veomni}
label=${2:?output label}
shift 2
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
source=${SPECO_SOURCE:-/experiment/variants/partition-e2e}
export PYTHONPATH=/experiment/veomni-deps:/experiment/online-deps:$source
cd "$source"
/experiment/.venv-clean/bin/python -m torch.distributed.run \
  --nproc-per-node=2 --master-port=29592 experiment/l20/seeded_train.py \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.model.path=/experiment/tiny-peagle/target \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.drafter.model_path=/experiment/tiny-peagle/draft-original \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=PEAGLE \
  actor_rollout_ref.rollout.drafter.checkpoint_path=/experiment/evidence/l20-20260919/"$label" \
  actor_rollout_ref.rollout.drafter.training.engine="$engine" \
  actor_rollout_ref.rollout.drafter.training.max_steps=6 \
  actor_rollout_ref.rollout.drafter.training.save_interval_steps=3 \
  actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.drafter.training.feature_store.path=/experiment/tiny-peagle/features-packed \
  actor_rollout_ref.rollout.drafter.training.feature_store.shuffle=False \
  actor_rollout_ref.rollout.drafter.training.seed=7 "$@"
