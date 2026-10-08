# Bubble Time Drafter Training

Last updated: 09/30/2026

Bubble Time is the rollout-idle execution path for online drafter co-training.
Its goal is to keep the same drafter-training quality target as synchronous
training, while moving as much drafter optimizer work as possible into rollout
idle windows.

The intended behavior is:

1. Collect training features using the same data policy as the synchronous path.
2. Register a drafter-training quota for the collected data.
3. Train the drafter on an idle rollout worker group when a safe idle window is
   available.
4. Accumulate partial progress across idle windows instead of requiring every
   idle launch to finish the full quota.
5. Publish the completed drafter snapshot to all rollout workers.
6. Allow the next quota to switch to another writer group only after that group
   has acknowledged the latest published drafter version.

This keeps Bubble Time from producing an artificial speedup by simply doing
less drafter training. In quality-alignment runs, use the same optimizer-step
quota, batch size, effective global batch, learning rate, and sampling order as
the synchronous configuration.

## Execution Flow

```text
rollout / old_log_prob feature collection
        |
        v
training quota registered for one data snapshot
        |
        v
idle rollout worker group selected
        |
        v
partial drafter training during rollout idle windows
        |
        v
quota fully repaid
        |
        v
publish staged drafter snapshot to rollout workers
        |
        v
record per-worker publish acknowledgement and local drafter version
```

Only one writer group owns a quota at a time. Other rollout workers receive the
published drafter weights, but optimizer state is not migrated between groups.
After a quota is finished and published, the scheduler may choose a different
writer group for the next quota if that group has acknowledged the latest
published drafter version. If a candidate group is stale, the scheduler keeps
using the current hot writer or waits for a safe candidate.

## Recommended Configuration

The main switches live under
`actor_rollout_ref.rollout.drafter.training.scheduler`.

```bash
actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=rollout_idle_worker \
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.enable=true \
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.target_steps=20 \
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.trigger_mode=interval \
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.allow_critical_path_fallback=true \
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.group_mode=auto \
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.group_size=null \
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.training_groups=[] \
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.full_collective_fallback=false
```

For strict Sync/Bubble quality comparison, keep `target_steps` equal to the
synchronous drafter `step` setting and keep the same collection interval,
batching, and data sampling settings. For production runs, Bubble Time can still
use interval triggering, or it can refresh based on acceptance-length and loss
signals through `trigger_mode`; the scheduler should continue an unfinished
quota on the same data snapshot rather than repeatedly collecting new data that
will not be trained.

`allow_critical_path_fallback=true` lets the scheduler top up unfinished quota
on the critical path when idle windows are not enough. This is useful for
quality-alignment and stability testing. For pure production throughput tests,
turning it off makes the resource-isolation contract stricter, but stale or
insufficiently trained drafters may take longer to refresh.

## Key Logs and Metrics

Use these logs to confirm that Bubble Time is actually using rollout idle time:

- `training_quota_registered`: a new quota was created for a data snapshot.
- `idle_writer_elected`: an idle worker group was selected for drafter training.
- `training_quota_repaid`: idle or top-up training reduced the quota debt.
- `rollout_drafter_versions_updated`: publish completed and rollout-worker
  drafter versions were recorded.
- `idle_group_stale_drafter_version`: a candidate writer group was rejected
  because it had not acknowledged the latest published drafter version.

Useful metrics:

- `timing_s/step`: end-to-end step time.
- `timing_s/drafter`: drafter training time left on the critical path.
- `timing_s/gen`: rollout generation time.
- `drafter/spec_decode/mean_acceptance_length`: speculative-decoding quality.
- `bubble/training_quota_debt_steps`: unfinished drafter-training quota.
- `drafter/train_successful_steps_max`: successful drafter optimizer steps.

When comparing with synchronous drafter training, exclude checkpoint-save steps
from the speed calculation because checkpoint I/O is orthogonal to Bubble Time.

## Observed GPU Result

On a Qwen3-4B DFlash GPU run, Bubble Time kept the drafter-training quota
aligned with the synchronous path and moved most drafter work off the main
critical path. Excluding checkpoint-save steps, the observed result was:

| Metric | Sync drafter training | Bubble Time | Change |
| --- | ---: | ---: | ---: |
| End-to-end step time | 123.52 s | 115.23 s | 6.7% faster |
| Drafter time on the critical path | 8.33 s | 2.08 s | 75.0% lower |
| Generation time | 49.47 s | 47.43 s | 4.1% faster |
| Mean acceptance length | 3.30 | 3.25 | Similar |

The expected healthy pattern is that Bubble Time retains the speculative
decoding gain from the trained drafter, while reducing the amount of drafter
training that blocks the PPO critical path.
