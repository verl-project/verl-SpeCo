# Bubble Time Drafter Training

Last updated: 10/15/2026

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
6. Retain the same optimizer owner across quotas; inference-weight
   acknowledgements alone do not transfer optimizer or scheduler state.

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
published drafter weights; optimizer state stays with the writer by default.
After a quota is finished and published, the writer remains authoritative even
when its tensors are parked on CPU. Writer migration is experimental and defaults
to disabled (`idle_worker.writer_migration_enable=false`). Keep this default until
real subgroup handoff, checkpoint save/resume and next-update consistency have
been validated on the target GPU/NPU stack. To opt in after validation:

```bash
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.writer_migration_enable=true
```

When enabled, at the start of a new quota another eligible
idle group can receive the complete CPU training state: corresponding parameter
shards, Adam state, learning-rate scheduler, optimizer counters, replay data,
target-head snapshots and training RNG. All destination shard layouts and
optimizer parameter ordering are validated before applying the transfer.
Only equal-size, parked CPU shard layouts are supported; DTensor migration and
keep-hot handoffs retain the existing writer. No mid-quota transfer is allowed.

The remaining idle window must cover transfer, restore, a complete optimizer
step and parking. The initial transfer estimate uses 2 GiB/s and is subsequently
bounded by measured migration time. Short windows retain the owner rather than
paying an unproductive transfer. This cost model needs device-host validation;
more idle utilization is not guaranteed when transfer itself is too expensive.

Replica-local trainers also use that writer for blocking top-ups, with gradient
accumulation adjusted to preserve the Sync effective global batch. Dispatching a
Sync plan to every rank does not turn independent replica-local FSDP groups into
one shared optimizer. The full-collective fallback is only valid for an actual
full-collective topology with no private writer history.

## Recommended Configuration

`idle_worker.keep_hot_between_plans=false` is the default. Drafter tensors can
stay on the device between idle leases within a rollout, but are parked on CPU
before PPO computation and before rollout weight updates/wake-up. Parking also
applies to a writer whose training task has already completed. It preserves the
optimizer state, optimizer-step count, replay snapshot, and remaining quota.

Set the following only when memory is sufficient for the main workload and the
resident drafter trainer together:

```bash
actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker.keep_hot_between_plans=true
```

This opts out of boundary offload; it does not permit training during PPO.
The trainer builds and parks its models before timed rollouts, and measures a
second restore/park and a discarded CPU weight snapshot without optimizer
updates to calibrate group admission and final staging cost.
Subsequent activation restores CPU tensors rather than rereading the checkpoint. First
backward/optimizer initialization can still be more expensive than steady-state
steps. Confirmed runtime idle events and training resource leases remain required
for safe overlap; historical windows bound the number of planned steps after
bootstrap. Generation completion does not launch another Bubble task using an
artificial post-rollout window.

When a plan carries a deadline and keep-hot is disabled,
the worker parks its training state as part of the plan's reserved tail, before
returning the idle lease. Historical windows use estimated execution deadlines,
not authoritative runtime promises; post-generation reclaim remains necessary.
If a final snapshot cannot fit and keep-hot is disabled, quota plans stay partial and leave at least one
optimizer step pending; the completed snapshot waits for a larger window or
configured blocking completion. No quota steps are discarded.

Measure activation, training, and boundary parking costs separately. In particular,
`timing_s/drafter_memory_park_before_post_rollout` and
`timing_s/drafter_memory_park_before_update_weights` expose residual main-path
offload overhead. Short windows may be skipped when restoration and a complete
optimizer step cannot fit. Use quota completion fallback when bounded training
freshness is required; neither overlap nor a speedup is guaranteed for every workload.

Each data version can register only one quota. Discovering a retained buffer at
a later step does not create another quota for already-trained data. Restore
costs are learned per writer group; measured device parking is charged to the
tail reserve. Publication and in-flight-step waiting are not charged again as
the deadline guard. Bubble FSDP2 full-head export materializes only the actor
projection, avoiding full actor parameter enumeration while preserving all rows.
Unsupported actor layouts and Sync retain their existing export path.

Workers report individual optimizer-step durations. Only participating ranks
with successful optimizer steps contribute; empty results from other rollout
groups cannot suppress timing updates. The first cold optimizer
update is excluded from steady-state estimates, and writer-local synchronous
top-ups also refresh the estimate. FSDP1 optimizer checkpoint export/load use
the writer's explicit process group during optimizer state-dict conversion as
well as during the outer DCP operation. This explicit subgroup conversion is
limited to replica-local Bubble trainers; Sync retains its original DCP
optimizer conversion and checkpoint process-group behavior.

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

With quota enabled and `trigger_mode=interval`, old-logprob collection first
selects samples using Sync's logical-owner sample/token budgets, hard-sample
policy, and hidden-state windows. It then routes the selected set to the current
writer without selecting again. Thus a two-owner `16 + 16` budget stays at 32
samples both before and after writer election, subject to the same candidate
and token limits as Sync. `drafter/selection/sync_aligned_collection=1` identifies
this path. Adaptive collection retains its existing writer-local
minimum and budget policy. This aligns collection, not bitwise optimizer updates
or end-to-end quality; those still need device validation.

For the first nonempty interval-mode old-logprob collection, the scheduler
selects the bootstrap writer before dispatch and sends the complete selected
pool directly to that group. It does not first distribute the pool and then
gather it again. Empty or skipped collection does not elect a writer. Selecting
this data destination does not mark the optimizer initialized or the writer hot;
the existing first-quota bootstrap and real-idle admission checks still apply.
Until that bootstrap finishes, another group cannot take over its data snapshot.

Initial writer election uses recent, recency-weighted observations with a
confidence discount for sparse history instead of selecting an old p75 spike.
This changes only the initial election score, not training admission deadlines
or the ownership of an already-trained writer. Completed training reports
`bubble/training_quota_debt_before` and `bubble/training_quota_debt_after`;
`bubble/training_quota_debt_steps` in its result is the post-training value.

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

## Diagnosing keep-hot device memory

### Window and training-budget timelines

`idle_event_timeline` reports request completion callback, event emission, resource
lease release, bus receipt and driver receipt timestamps when supplied by the
runtime. `idle_window_timeline` adds the usable idle start and window closure.
Request completion callbacks are not device kernel completion timestamps; no
extra device synchronization is introduced to measure them. Lease release only
proves the existing runtime resource-ownership contract, not that all accelerator
allocations have been freed. Missing timestamps remain absent, rather than being
invented. Cross-node delivery lag measurements require synchronized host clocks.

Per-replica histories now separately report `generation_start_closed_p75_s` and
`generation_end_closed_p75_s` with sample counts. These are close-event cohorts:
they help distinguish request gaps and rollout tails but do not predict which
cohort a currently open window will belong to. Admission therefore retains the
existing mixed-history safety bound until a runtime supplies that classification.

`optimizer_plan_timing` splits the first optimizer step from subsequent steps.
Only subsequent non-warmup steps calibrate steady throughput; a one-step plan does
not overwrite steady throughput with its first-step allocation cost. Measured
first-step excess is reserved once per plan as `first_step_extra_s` within the
startup budget. A hot writer entering a new target-head version also reserves its
measured head preparation; repeated plans for the same version do not pay that
reserve again. `training_activation_breakdown` reports host head apply, model and
optimizer restore, head movement and device head apply separately. These are wall
times without added device barriers; asynchronous device waits may appear in the
first optimizer step instead. GPU/NPU testing must validate the resulting budget.

`weight_update_breakdown` and `timing_s/upstream_weight_update_only` separate the
original checkpoint-manager weight-update call from the external drafter hook.
The upstream call can still include runtime sleep/wake and other built-in work;
it is not a pure actor tensor-copy measurement. These new fields overlap existing
`timing_s/update_weights` and publish metrics and must not be added to them again.

### Validation isolation

Validation does not start the idle event thread, launch drafter training or
collect drafter samples. Bubble drains outstanding training/head export and
publication before validation; any locally deferred publish is completed before
validation generation admits requests. The validation guard survives mutation of
batch metadata and is restored even on exceptions. Validation runtime events are
discarded rather than entering training buffers/window history; late events with
timestamps through the validation exit boundary are also ignored. Look for
`validation_isolated` and `bubble/training_blocked_by_validation`. A quota collected
at step 10 therefore resumes during training rollout 11, not validation at step
10. Validation idle capacity is intentionally excluded from performance accounting.
Sync retains its original wait-before-validation path.

### Completing a hot-writer quota

For replica-local DSPARK/FSDP1 hot publication, the exporter now exposes full
parameters with `writeback=False` and copies only publishable parameters and
persistent floating buffers on the writer leader. Frozen target-shared weights,
nonpersistent buffers and entries excluded by publish-name patterns are not copied
to CPU. Publish precision is applied before host transfer. This still requires
FSDP1 parameter gathering; it does not avoid frozen weights in the same flat
buffer, and GPU full-parameter materialization can change temporary memory peaks.
Sync, checkpoint export and other drafter backends retain the original path.
Measure `publish_trainable_only_export` bytes/time and device headroom before long
runs; subgroup gathering and publish payload equivalence need GPU validation.

When a keep-hot writer has a measured historical window large enough for another
optimizer step, soft critical-path topups are postponed until further idle leases
or the existing hard completion deadline. `quota_soft_topup_deferred` explains the
decision. The exact quota, data snapshot, hard freshness deadline and blocking
completion at that deadline are preserved. This prediction does not guarantee a
future window; if it fails, remaining work is still completed synchronously at the
configured hard boundary. Acceptance length must be evaluated with the changed
within-deadline update timing.

If optimizer work fits an idle window but final snapshot export does not, a
keep-hot writer can now finish the full quota with `publish_snapshot_deferred`.
It normally exports the snapshot at the existing safe publish point; if training
finishes early enough, all writer ranks may agree to prepare it in the remaining
idle budget instead. The writer rejects new training until staging finishes and verifies its
optimizer step is unchanged. Collection remains blocked until publish completion.
Cold/offloaded writers retain the original final-tail admission rule. This avoids
leaving one optimizer step solely to pay the export cost; it does not make snapshot
export free or guarantee its cost is hidden. Watch `deferred_snapshot_staged` and
publish success before treating the cycle as complete.

### Version-safe target-head prefetch

Bubble submits its head export after old-logprob collection, allowing eligible
FSDP2 head-only export to overlap the following reference stage. Before actor
update, an explicit export fence waits for any unfinished export; only subsequent
host dispatch may overlap the actor update. `timing_s/drafter_target_lm_head_export_fence`
reports the remaining critical-path wait. Unsupported layouts keep synchronous
export, and the async path refuses fallback to a full-actor reload. Sync uses the
unchanged synchronous getter. Measure reference-stage time and memory as well as
submit/fence time: overlapping an export can still cause resource contention.
Async selection first probes every participating actor rank without gathering
weights, and caches the capability result. The worker probe and exporter share
the same candidate selection, including the wrapped-class drafter overlay used
by verl 0.8 when rollout config fields are stripped. If any rank cannot export
head-only (or probing is unavailable), the driver chooses the original synchronous
getter before submitting export. Check `target_head_export_capability` for each
rank's reason; the async full-actor-reload guard remains enabled.

For first-stage scheduling diagnosis, distinguish `bubble/idle_optimizer_steps_total`
from `bubble/critical_path_topup_optimizer_steps_total`. These are cumulative
counters since process startup (not checkpoint-restored training counters); take
their differences over the comparison interval. Writer-local synchronous topups
are not counted as `drafter/idle_trained`, even though they share the Bubble
execution strategy. An idle-path launch alone does not prove all its work was
hidden from the main critical path.

Window rejection logs include `min_required_s`, `budget_deficit_s`, startup/tail
reserves and the selected writer. A writer with a short window is reported as
`window_too_small`, rather than being masked by another group's migration
requirement. `idle_overhead_updated` logs separate activation, park and snapshot
costs; successful writer-local topups also calibrate their shared overhead, while
full-collective Sync plans do not change replica-local estimates.
Worker results measure non-park cleanup separately before per-rank aggregation,
so subtracting maxima from different ranks cannot distort the budget. Snapshot
cost contributes only to the final tail, not partial training admission. Inspect
`memory_park_breakdown` and `training_memory_parked` for model, optimizer, target
head, synchronization and cache-release costs; the required measured park reserve
is not removed merely to admit more work.

Set `VERL_DRAFTER_MEMORY_DIAGNOSTICS=true` on training workers to emit
`[DrafterMemory]` logs at activation, keep-hot cleanup and parking. Values are
bytes: local target-head, model, buffers, gradients, optimizer, replay and publish
storage, plus allocator allocated/reserved/cache totals. Shared storage is counted
once; no FSDP all-gather or device-to-host copy is performed. Unattributed allocated
memory includes tensors outside these categories. Peak allocation is the allocator
peak since its last reset, not necessarily this training window. Process memory
reported by an OOM also includes non-PyTorch allocations, so these numbers need not
sum to the process total. Diagnostics are best-effort and disabled by default.
The SPECO entrypoint explicitly forwards the two diagnostic switches to its Ray
TaskRunner. Check each training worker's `worker_environment` log to confirm the
effective values before trusting missing diagnostics. Existing actors must be
restarted when changing environment variables.
Opt-in memory reports use WARNING level so default worker log filtering does not
hide them. Reports include `revision=3` and the trainer's loaded source path; an
initial `replica_local_initialized` report confirms the implementation was loaded
before any training starts. This initial report is not the peak training footprint.
Revision 3 additionally counts retained replay sessions and local model/handle/
flat-parameter attributes, plus direct trainer/backend tensor containers. Shared
storage is still counted only once. `extra_storage` lists the largest additional
storage holders to help identify live memory left after parking. Inspection is
bounded and read-only; it does not traverse autograd graphs, run full-parameter
collectives or free private FSDP buffers. Unknown allocations can still remain.
On CUDA, large unidentified memory after parking/cache trimming additionally logs
`unattributed_allocator_blocks`, showing active block addresses/sizes and pending
free states from allocator metadata. This does not resolve Python ownership or
automatically free any blocks, and is disabled with memory diagnostics.

With keep-hot enabled, optionally set `VERL_DRAFTER_KEEP_HOT_TRIM_CACHE=true` to
return unused allocator blocks after a training lease. Live model, optimizer and
target-head tensors remain resident, and training progress is preserved. This
does not solve OOM caused by live tensors; cache reuse can be faster, so the option
is disabled by default. Compare before/after cache-trim logs and end-to-end time
before enabling it in production. For insufficient live-memory headroom, use
`idle_worker.keep_hot_between_plans=false` instead. Sync cleanup is unchanged.
`keep_hot_cache_trim` reports elapsed time, released reserved bytes and allocated
bytes before/after. Use this to verify that cache was actually released without
offloading live tensors. Run an initial diagnostic short test across two quota
cycles and actor updates; do not assume a successful trim guarantees OOM safety.
When keep-hot is enabled, idle tail admission excludes historical offload cost
because main-workload parking is disabled in that mode. Non-park cleanup (including
optional cache trimming) and final snapshot costs remain in their respective
budgets. Disabling keep-hot retains the measured park reserve for OOM safety.

## Boundary and budget verification

The rollout-return timestamp is captured before event-thread shutdown and
training reclaim. Historical idle windows end at that timestamp; reclaim wait
must not become a larger future idle budget. Historical windows also carry an
execution deadline, not just a maximum batch count. A deadline is a conservative
estimate, not a guarantee: an in-flight optimizer step cannot be preempted.

Model prewarming does not prove optimizer initialization. With quota fallback
enabled, the first quota bootstraps real optimizer state at the first safe point
where collected data and its immutable target head are ready, rather than waiting
for the hard freshness deadline. Before any state is trained, the bootstrap
prefers measured idle capacity instead of metadata order. These are real quota
optimizer steps, not extra dummy updates. This adds a one-time critical-path
cost and must not be counted as hidden Bubble training.
Successful writer-local training then enables ordinary idle admission, even if
keep-hot is disabled. Disabling critical-path fallback cannot guarantee progress
when no sufficiently long authoritative window exists.

With keep-hot enabled, a new target-head version is applied to the resident writer
after PPO finishes, before subsequent idle admission. This only applies the
cached frozen head: it does not restore model/optimizer, train extra steps, change
the replay snapshot or activate other replicas. Every writer rank must acknowledge
the exact version after its device copy completes. The elapsed time is reported
as `timing_s/drafter_target_head_prepare_safe_point`; `hot_writer_head_safe_point`
and `hot_target_head_prepared` report completion. Once prepared, its cost is not
reserved again in the following idle window. Cold activation observations are
not relabelled as measured hot-head preparation. This preparation is not free;
compare its one-time-per-version cost against the recovered idle work.

Only genuine idle plans yield between complete optimizer steps and agree on stop
requests through the trainer's explicit writer SP/DP groups. An accumulation is
atomic; stopping does not consume a partial accumulation or repay quota debt.
Sync and synchronous quota top-ups retain their previous accumulation path.
`timing_s/drafter_idle_control` measures this small-message control overhead;
it is not a zero-cost operation and does not involve other rollout replicas.

A completed deferred-snapshot plan may prepare its immutable CPU snapshot before
cleanup if every writer rank has enough remaining time for the measured final
tail. Otherwise it retains the existing safe-point export. This optimization
does not publish an incomplete quota and does not move device export into PPO.
It advances snapshot preparation only; engine staging/commit and ACK still have
cost and remain protected by the publish safe point.

For the first device test, keep the existing topology/configuration and run at
least three complete quota cycles. Check `rollout_boundary_timing`,
`quota_optimizer_bootstrap_planned`, `hot_writer_head_safe_point`, `optimizer_plan_timing`,
`training_quota_repaid`, and `publish_succeeded`. The optional
`publish_snapshot_staged_in_idle` log appears only when enough budget remains.
Compare `timing_s/bubble_rollout_rpc`,
`timing_s/bubble_rollout_return_to_ready`, and
`timing_s/drafter_reclaim_wait`; all are wall-time measurements, not sums of GPU
kernel durations. `bubble/weight_update_source_step` identifies the source step
of weight-update metrics, which may be flushed in the following step's row.

After functional verification, run fresh Sync and Bubble baselines on identical
devices, model/data seeds, collection interval, actual sample policy, effective
global batch, optimizer quota and learning rate. Configuration similarity alone
does not prove identical training snapshots. Disable periodic validation and
checkpointing for the performance run (then test both separately). Report both
whole-run wall time including bootstrap/final drain and steady-state step time,
plus accepted length and response length. Compare several runs; do not infer a
10–15% speedup or guaranteed quality equivalence from successful idle launches.

### Diagnosing exposed training costs

Keep topology and writer ownership unchanged when measuring reclaim overhead.
`reclaim_sent`, `reclaim_requested`, `training_stop_boundary`, and
`training_cleanup_boundary` report driver send, worker receive, stop, and cleanup
completion timestamps. Cross-host wall clocks must be synchronized before using
send-to-receive differences to estimate RPC queue delay. Worker-local durations
`timing_s/drafter_reclaim_receive_to_stop` and
`timing_s/drafter_reclaim_receive_to_cleanup` use a monotonic clock and report the
maximum across participating ranks; they are overlapping intervals, not additive
costs. They exclude waiting before the reclaim RPC reaches a worker. A peer may
stop the group before its own local reclaim RPC arrives, so a zero local interval
does not prove zero driver wait.

Completed idle-control timers are carried into the step metrics and are no longer
overwritten by a subsequent no-launch/top-up outcome. Use
`timing_s/bubble_event_thread_stop` and `timing_s/bubble_event_drain` to identify
delay before reclaim dispatch. `training_data_probe` measures the existing data
inspection RPC plus aggregation without adding a probe RPC.
`timing_s/drafter_schedule_probe` covers the last scheduling/preflight call and
`timing_s/drafter_target_head_poll_last` covers the last prefetch poll, including
dispatch if that poll advances the state machine. These can overlap other named
timers and must not be summed blindly or subtracted again from step time.

Genuine idle training uses a short timer yield (requested 1 ms) before each
optimizer-step control agreement, so queued event-loop I/O has a chance to run.
Unlike `sleep(0)`, this is not just another immediately-ready continuation.
It does not preempt an accumulation or guarantee that Ray's RPC dispatcher has
delivered a stop request. Actual timer scheduling may take longer than 1 ms;
`timing_s/drafter_idle_control_yield_elapsed` reports the maximum accumulated
yield time across participating ranks. Sync and blocking top-ups do not use
this yield. Verify driver-send to worker-receive latency on the real Ray setup;
if requests still arrive only after training ends, a separate control-delivery
path is needed rather than increasing the idle deadline.

Only genuine idle plans calibrate hot idle first-step overhead. Blocking top-ups
can still calibrate steady optimizer throughput and their cleanup/export costs,
but their first step no longer inflates idle restart reserve. Real measured idle
first-step and cleanup costs remain budgeted; keep-hot does not make them free.
Step-level `drafter/trained` and `drafter/idle_trained` preserve completed idle
outcomes when the actor hook launches nothing. `train_successful_steps_max` is
the maximum per-plan count across completed outcomes in the step, not the sum of
idle and top-up updates; use quota repayment and cumulative optimizer counters
to verify the complete training amount.

The rollout event loop consumes proven-ready training outcomes instead of only
releasing their resource lease. This repays quota once, retires `RUNNING`, and
allows a completed small plan to be followed by another plan without requiring a
new idle event. `idle_training_completion_consumed` identifies this transition;
`bubble/idle_optimizer_steps` and training timers accumulate across those plans.
The continuation keeps the writer and quota snapshot, and caps admission by the
previous absolute deadline. It does not renew a historical prediction or bypass
first-step/cleanup reserves. Reclaim, deadline stops, and unsuccessful outcomes
close continuation for that rollout; subsequent rollouts start fresh admission.

A completed final quota queues its publication until the existing safe point;
the event loop cannot start a new quota over that frozen snapshot. Publication
callback failures retain the queue, and failed training futures are surfaced at
the PPO boundary. Generation return closes admission before joining the event
thread, including a check after a potentially slow preflight. Sync remains on
its original execution/publication path. Continuations can incur another
preflight, first-step warmup, and cleanup; verify net end-to-end savings, not just
increased optimizer counts. A long idle duration observed after the fact is not
permission to extend an estimated deadline.

First verify at least three complete quotas and matching publish ACKs on the
existing device configuration. Then compare an aligned Sync/Bubble run, including
bootstrap and final drain, excluding checkpoint/validation steps consistently.
Do not switch FSDP implementations or migrate writers in the same experiment:
those require separate subgroup state/optimizer/checkpoint consistency tests.

## Independent reclaim control and experimental writer migration

Bubble writers receive the lease-bus name in their plan. A CPU-only watcher polls
the independent actor every 20 ms; it never runs model, optimizer or collective
operations. Driver reclaim writes a plan-scoped flag there instead of queuing
behind training. Waiting rollout requests set the flag directly, without waiting
for the driver's lifecycle-event poll. Writers still stop group-wide between
complete optimizer steps. Late flags cannot stop a new plan; pending generation
requests take priority over continuation. Watcher failure requests a safe stop;
unavailable control submission falls back to the worker RPC. Restart all actors
when deploying. Check `reclaim_received` with `source=lease_bus` and compare
delivery timestamps and reclaim wait. There is small CPU/control RPC overhead;
this cannot preempt the in-flight optimizer step or cleanup.

Migration is off by default. Under
`actor_rollout_ref.rollout.drafter.training.scheduler.idle_worker`:

```yaml
writer_migration_enable: false
writer_migration_history_steps: 4
writer_migration_min_idle_gain_sec: 1.0
writer_migration_cost_margin: 2.0
```

Completed rollout windows are paired by group. Candidates need the exact latest
published-version ACK, equal shard topology, and at least a 1-second idle advantage
in 75% of the last four rollouts. Decisions are recorded after publication ACK,
even with migration disabled: `writer_switch_decision` and metrics
`bubble/writer_switch_current_idle_mean_s`,
`bubble/writer_switch_candidate_idle_mean_s`,
`bubble/writer_switch_missed_idle_s`, and
`bubble/writer_switch_estimated_gain_s`. Missed idle is the positive observed
window difference, not a measured speedup or proof that all of it was trainable.
Prospective gain over a bounded history-sized horizon must exceed estimated
transfer, activation and cleanup cost times the margin. Actual window admission
and resource leases must also succeed. Cost estimates require device measurement.
Only the boundary after publication ACK and before the next quota's first update
allows migration; no mid-quota switching is allowed.

The handoff copies CPU parameter/buffer shards, optimizer/LR-scheduler states,
training/optimizer/checkpoint counters, replay and target-head snapshots, and
each rank's canonical post-training RNG. Validate layouts before applying shards;
change ownership only after all destination ACKs. A hot source must have enough
confirmed idle for parking and transfer, never offload it during its rollout.
The transfer also requires another active serving group to hide its cost. With
only two groups, a resident source and an idle destination leave no serving
overlap, so that handoff is skipped; a parked source does not need a device lease.
Enabling migration with keep-hot therefore does not guarantee a switch. Partial
apply/ACK failure is fatal. FSDP2/DTensor and unequal-topology handoffs are not
enabled by this CPU-shard protocol.

A prewarmed destination is parked under its idle resource lease before import;
the switch budget includes parking and a cold destination restart, even with
keep-hot enabled. Unsupported/uninitialized state or a layout-validation failure
is retried only after the quota/version or candidate changes, not on every event
poll. Pre-apply rejection keeps the source as owner; partial apply or missing
destination ACKs remain fatal. Logits-based training does not require a target-head
prefetch ACK to migrate.

First test reclaim with migration disabled. On the Linux training host:

```bash
python -m pytest tests/integration/test_rollout_idle_events.py tests/unit/test_training_worker_preflight.py tests/unit/test_bubble_training_boundaries.py
```

Coverage includes real Ray delivery to a busy async actor, CPU model/Adam/LR
handoff, RNG-dependent next-loss/update equality and checkpoint serialization
roundtrip. Skips do not count as validation. Before real migration, separately
validate actual writer subgroups, GPU/NPU restoration and distributed checkpoint
save/resume. Then compare migration off/on with aligned data and full optimizer
quota, including handoff cost in whole-run timing. CPU tests alone do not justify
enabling migration by default.

### Window capacity and first-step calibration

Idle batch caps are tracked per writer group. Early rollout reclaim ends the
current plan but does not collapse the next window's capacity to the number of
steps completed in that short opportunity. Deadline exhaustion still reduces
the cap, and the existing generation-slowdown guard remains active. Successful
window-limited plans progressively recover the cap. Historical windows still
bound planned work; continuation never renews the original absolute deadline.

First-step calibration uses the last four observations for each group and
restart mode: cold, hot, or hot continuation in the same rollout window.
Cold optimizer initialization and synchronous top-ups do not contaminate hot
restart estimates. A parked writer still pays cold restore costs, even for a
continuation. No quota, data snapshot, effective batch, or Sync training behavior
is changed by these admission-budget adjustments.

Independent reclaim delivery uses the CPU lease-bus watcher described above.
The control acknowledgement has a bounded timeout and falls back to the worker
RPC when unavailable; neither path interrupts a partially executed optimizer
step. Validate notify-to-receive delay, cleanup wait, idle optimizer steps, and
whole-run timing on GPU/NPU before claiming a performance improvement.

### Collection failure and configuration defaults

A collection is successful only after every owner acknowledges finalize with
the same transaction ID, worker incarnation, and source step. A finalize RPC or
ACK failure raises `CollectionFinalizeError` with reason
`collection_finalize_failed`, `collected=false`, and
`committed_unfinalized=true`. Background idle collection retains this error and
stops admission; the PPO boundary raises it instead of silently training on an
ambiguous dataset. Recovery requires restarting from a consistent checkpoint.

Do not blindly roll back after partial finalize: an owner may have already
discarded its rollback journal while another still retains one. Rolling back
only the latter creates divergent datasets. Failures before finalize retain
their existing abort/rollback behavior.

Python constructors, mapping-based configuration, and YAML now share production
defaults: `training_quota.trigger_mode=adaptive` and
`training_quota.allow_critical_path_fallback=false`. Strict Sync-equivalence runs
must explicitly select `interval` and, when blocking completion is required,
`allow_critical_path_fallback=true`.

Validation isolation is enforced by the trainer's validation guard and event
cutoff, not an unused sampling-parameter flag. The obsolete idle-event marker is
no longer emitted. The runtime still maintains resource leases; validation
events cannot admit training or enter the training-window history.

### Review safeguards for training quality and topology

Quota replay uses the same random/hard-sample policy as ordinary training,
restricted to the immutable reserved snapshot. The replay cursor retains the
sampling seed across short plans. DSPARK fractional hard-sample draws use the
optimizer counter and accumulation micro-index, so rounding does not restart
at every quota. This preserves the sampling policy, not bitwise equivalence
between different distributed topologies or identical acceptance lengths.

Replica-local FSDP1 clips the full writer-group gradient norm through FSDP's
subgroup-aware API. Sync FSDP2/DDP retain the ordinary clipping path. Cold Bubble
cleanup must successfully park model, optimizer, and target-head state before
returning training resources; it preserves training counters. Keep-hot remains
an explicit memory/performance trade-off and bypasses this parking path.

Batch readiness and recoverable preparation failures are agreed within the
writer group, never across unrelated rollout groups. The scalar readiness
collective replaces the former DP-only check; its Bubble cost is reported in
`timing_s/drafter_batch_readiness`. Accumulation prepares and releases one device
micro-batch at a time while retaining deterministic replay order, distinct
samples within the optimizer step, and combined valid-token normalization.
Deadline/reclaim decisions still stop coordinated Bubble training only at a
complete optimizer boundary. Quota size and idle-window admission are unchanged.
Sync retains main's cross-step pool and passes sample bounds through inspection
and batch preparation; Bubble alone pins the quota's target version.

Explicit `training_groups` must match complete runtime training collectives.
Metadata discovery or lease-group configuration failure is fatal in Bubble,
rather than falling back to guessed group sizes. Independent replica-local
optimizers cannot become one full-card optimizer by enabling
`full_collective_fallback`; that configuration is rejected. All rollout actors
must restart after deploying protocol changes.

Bubble checks rank coverage, phase, and version for staging and live apply ACKs.
Missing, duplicate, stale, or mixed-phase ACKs cannot release a quota or authorize
writer migration. An inconsistent training result after optimizer progress also
stops the run instead of silently retrying divergent shards.
When convergence freeze is enabled, a Bubble publication advances its version
clock exactly once after validated live apply ACKs, never at submission/staging.
Failed publication does not advance the clock. Ordinary Sync retains its
existing outer-hook freeze observer and batch-preparation timing boundary.

Generation lease waits have request-scoped cancellation cleanup. A cancelled or
timed-out wait cannot leave a stale serving-priority marker that blocks future
Bubble reservations; cleanup does not release another request's serving lease.
The lease timeout bounds the RPC wait itself as well as polling, so a missing RPC
reply cannot bypass the configured timeout.

Actor timing is deducted only once when deferred-publish metrics are merged.
Generation slowdown feedback is counted once per original global step, so one
rollout cannot trigger multiple cap reductions or the cooldown patience by
repeated metric merges. Sync keeps its current-step collection guards, optional
lightweight cleanup, and sample-bound metadata; CPU target-head version history
is allocated only for Bubble.

These safeguards need aligned GPU/NPU runs for performance and quality claims.
New idle launches require a live serving lease on another group. The driver
skips launch preparation when every known rollout group has released serving;
the lease bus checks live ownership atomically again before granting training.
When the last serving request releases or cancels its lease, the bus also marks
active overlap plans for reclaim through the existing independent watcher,
without waiting for driver event draining or adding another RPC.
This prevents late event delivery from starting work after generation has ended,
but cannot preempt an optimizer step already in flight. Restart all actors after
deploying this lease-protocol update.

Historical window estimates are replica-specific. Once three real
`generation_completed` tails exist, their recent median caps the older p75;
an old long window cannot indefinitely override consistently short recent tails.
The first optimizer step of each idle plan logs
`optimizer_first_step_breakdown`, including accumulated micro-batch preparation
and existing forward/reduction/backward/optimizer host timings. These diagnostics
add no device synchronization or collective; asynchronous device execution means
they are not a precise kernel-time breakdown.

Keep writer migration disabled initially, complete at least three quotas, verify
all-rank publish ACKs, compare optimizer steps/loss/acceptance length, and include
all startup, parking, reclaim, and publication costs in end-to-end timing.

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
