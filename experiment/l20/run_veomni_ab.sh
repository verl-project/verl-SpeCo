#!/usr/bin/env bash
set -euo pipefail
cd /experiment/variants/veomni
evidence=/experiment/evidence/l20-20260919
TIMEFORMAT='%R'
for arm in A0 P0 P1 A1; do
    engine=fsdp
    if [[ "$arm" == P* ]]; then engine=veomni; fi
    { time timeout 600 bash experiment/l20/run_veomni_standalone.sh "$engine" "veomni-$arm" \
      > "$evidence/veomni-$arm.log" 2>&1; } 2> "$evidence/veomni-$arm.seconds"
done
timeout 600 bash experiment/l20/run_veomni_standalone.sh veomni veomni-resume \
  actor_rollout_ref.rollout.drafter.model_path="$evidence/veomni-P1/draft_step_6" \
  actor_rollout_ref.rollout.drafter.training.max_steps=8 \
  > "$evidence/veomni-resume.log" 2>&1
