#!/usr/bin/env bash
set -euo pipefail
cd /experiment/variants/partition-e2e
evidence=/experiment/evidence/l20-20260919
TIMEFORMAT='%R'
for arm in A0 P0 P1 A1; do
    partitions=1
    if [[ "$arm" == P* ]]; then partitions=2; fi
    { time timeout 600 bash experiment/l20/run_veomni_standalone.sh fsdp "partition-$arm" \
      actor_rollout_ref.rollout.drafter.training.peagle_sequence_partitions="$partitions" \
      > "$evidence/partition-$arm.log" 2>&1; } 2> "$evidence/partition-$arm.seconds"
done
