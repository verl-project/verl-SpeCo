#!/usr/bin/env bash
set -euo pipefail
out=/experiment/evidence/l20-20260920/c2-head-pruning
trap 'echo $? > "$out/standalone.exit"' EXIT
TIMEFORMAT='%R'
for arm in A0 P0 P1 A1; do
  source=/experiment/variants/partition-before
  if [[ "$arm" == P* ]]; then source=/experiment/variants/partition-e2e; fi
  { time SPECO_SOURCE="$source" timeout -k 15 900 bash /experiment/run_partition_standalone.sh fsdp "c2-pruned-$arm" actor_rollout_ref.rollout.drafter.training.peagle_sequence_partitions=2 > "$out/standalone-$arm.log" 2>&1; } 2> "$out/standalone-$arm.seconds"
done
