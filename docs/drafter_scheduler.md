# Drafter Scheduler Implementation and Strategy Integration Guide

Last updated: 08/19/2026

## Design Goals

The verl_speco.trainer.scheduler package is the single decision point for
drafter data collection, training, and weight publication. The outer Trainer
only supplies lifecycle events and runtime RPC adapters; Workers only validate
and execute generated plans.

The boundary is:

~~~text
SpecoRayPPOTrainer
  ├─ Builds contexts and binds Ray WorkerGroup RPCs
  └─ Calls Scheduler lifecycle events
            │
            ▼
DrafterScheduler (single decision Facade)
  ├─ CollectionPlan / TrainingPlan / PublishPlan
  ├─ Trigger / Budget / Strategy
  └─ Executor ports
            │
            ▼
SpecoWorker / rollout runtime
  └─ Validates plans and executes collection, training, and publish RPCs
~~~

External code should import public contracts only from the
verl_speco.trainer.scheduler package entry point, for example:

~~~python
from verl_speco.trainer.scheduler import (
    DrafterScheduleConfig,
    DrafterScheduler,
    TrainingPlan,
)
~~~

Do not make Trainer, Worker, or feature code depend directly on Scheduler
implementation modules such as training_trigger.py, training_budget.py, or
execution_strategy.py.

## Current Synchronous Path

### Initialization and Infrastructure Binding

SpecoRayPPOTrainer.attach_speco_worker_group() creates and binds three
Executors:

| Executor | Responsibility | Existing RPC boundary |
| --- | --- | --- |
| DrafterWorkerExecutor | Data status, training preparation, preflight, and training submission | SpecoWorkerGroup |
| DrafterCollectionExecutor | Collection stage, commit, rollback, and finalize | SpecoWorkerGroup |
| DrafterPublishExecutor | Fetch a training snapshot and hot-update the rollout drafter | Drafter WorkerGroup and rollout runtime |

The Scheduler depends only on these Protocols, not on Ray. Ray ObjectRef
submission, waiting, and result parsing are handled by Callback Executor
adapters.

### Full Online Training Sequence

~~~text
rollout / compute_old_log_prob produces features
  │
  ├─ Trainer: plan_collection(context, config)
  │     └─ CollectionPlan: whether to collect, source, sampling/window budget,
  │        and collection_id
  │
  ├─ Trainer: prepare_collection_payload(...)
  │     └─ SGLang adapter buckets by replica_rank
  │        old-log-prob adapter buckets by explicit owner
  │
  └─ Scheduler: on_collection_ready(plan, payload)
        └─ SyncCollectionStrategy
             ├─ set_global_step
             ├─ stage
             ├─ validate collection_id, routing, versions, and Worker identity
             ├─ commit
             └─ finalize; abort / rollback on failure

Before the PPO actor update
  │
  └─ Scheduler: on_before_actor_update(context)
        ├─ prepare_training_plan()
        │    ├─ Return a skip plan for cheap collect_only / pending / interval conditions
        │    ├─ Query TrainingDataStatus from all Workers only when training is possible
        │    ├─ TrainingTrigger decides whether training should start
        │    ├─ TrainingBudget computes max_batches / min_batches and related limits
        │    └─ TrainingPlan freezes versions, budget, and execution strategy
        └─ prepare_training_execution(plan)
             ├─ set Worker global_step
             └─ synchronize target lm_head when use_logits=False

PPO actor update
  │
  └─ Existing PPO update logic is unchanged

After the PPO actor update
  │
  └─ Scheduler: on_after_actor_update(plan, runtime_state)
        └─ SyncExecutionStrategy
             ├─ preflight all Workers
             │    └─ validate plan_id, step, buffer/data/target versions,
             │       incarnation, and batch availability
             ├─ any Worker is not ready: abort all; do not submit distributed training
             ├─ all Workers ready: submit and wait for train_drafter
             └─ TrainingOutcome aggregates results; inconsistency sets trained=false
                and prevents publication

At the rollout-safe point after the main-model weight update
  │
  └─ Scheduler: on_after_weight_update(context)
        ├─ plan_publish()
        └─ PublishExecutionStrategy
             ├─ wait for the previous asynchronous publish, if any
             ├─ fetch the training snapshot from the single publish leader
             └─ update rollout drafter weights synchronously or asynchronously
~~~

### Current Synchronous Strategy Rules

- The default TrainingTrigger is IntervalAndBufferTrigger. It requires a
  matching training interval, no pending training, consistent Worker data
  versions, and at least min_trainable_batches trainable batches after Worker
  aggregation.
- SyncTrainingBudgetPolicy uses the configured training.step as max_batches.
  Data availability only determines whether training can start; it no longer
  silently reduces the configured optimizer-step count to the number of
  distinct batches.
- min_batches is the minimum trainable-batch requirement before launch. When
  max_batches is less than min_batches, the Scheduler returns an
  insufficient_training_budget skip plan.
- With use_logits=False, required_target_version equals the current
  main-training global_step and every Worker must use that target version.
- With use_logits=True, targets are already stored in data.
  required_target_version=None means that the current Worker target version is
  unconstrained; it does not require that value to be None.
- Publication occurs only when TrainingOutcome.trained=True and
  TrainingPlan.publish_after_success=True. Only the unique publish leader must
  provide a usable publish snapshot.

## Core Objects and Responsibilities

| Object | Responsibility | Must not be responsible for |
| --- | --- | --- |
| DrafterScheduleContext | Read-only facts: current step, mode, and pending state | Training budget or Worker execution details |
| TrainingDataStatus | Aggregated snapshot of trainable data on all Workers | Triggering training or invoking training RPCs |
| TrainingPlan | Immutable decision: launch, strategy, budget, versions, and plan_id | Recomputing trigger conditions in Workers |
| CollectionPlan | Collection decision and static sampling budget | Parsing source-specific feature formats |
| PublishPlan | Publication decision | Fetching or transferring weights |
| TrainingOutcome | Aggregates multi-Worker results and consistency to decide publication | Recomputing training budget |
| Executor | Connects Scheduler code to Ray, Workers, and rollout runtime | Trigger, budget, or publication-interval policy |

plan_id, collection_id, data_version, buffer_version, and worker_incarnation
prevent cross-process state drift. When data changes after planning, a Worker
restarts, or messages are mismatched, the path must fail closed rather than
allow only part of a distributed job to enter training.

## Adding a Strategy: What to Change

Choose the smallest applicable change. Do not add another set of interval,
buffer, or publish decisions in SpecoRayPPOTrainer or SpecoWorker.

### 1. Change Only Whether Training Is Triggered

Examples include using acceptance rate, loss, sample age, or an external SFT
event to decide whether to train.

1. Implement TrainingTriggerPolicy.should_train(...) in
   scheduler/training_trigger.py.
2. Return TriggerDecision(should_train, reason), and add a stable numeric
   reason code to TrainingPlan._REASON_CODES.
3. Inject or select the trigger policy in DrafterScheduler.
4. Preserve prepare_training_plan() cheap-skip behavior so conditions that do
   not need Worker data return before Worker RPCs.
5. Add pure unit tests for each trigger branch.

This type of strategy must not modify SyncExecutionStrategy, Worker preflight,
or the publication call path.

### 2. Change Only How Many Steps Run or When They Stop

Examples include using buffer size, token budget, deadline, or SFT quota to
compute the number of training steps.

1. Implement TrainingBudgetPolicy.make_budget(...) in
   scheduler/training_budget.py.
2. Return TrainingBudget(max_batches, min_batches, deadline_ts,
   require_full_batch, sample_last_n_steps, reason).
3. Ensure max_batches is at least min_batches. Otherwise,
   DrafterScheduler.plan_training() returns an insufficient_training_budget
   skip plan.
4. For a new budget field, extend TrainingBudget and TrainingPlan, update
   TrainingPlan.to_worker_payload(), then make Workers consume only that
   serialized field.
5. Test zero budget, minimum budget, caps, and deadline branches.

### 3. Add a Training Execution Strategy

Examples include Bubble Time, rollout idle workers, asynchronous queues, or
SFT co-training. ROLLOUT_IDLE_WORKER is reserved in the current enum but is
not implemented yet.

1. Add or enable an enum value in DrafterExecutionStrategy in
   schedule_types.py, and update the strategy code in TrainingPlan.metrics().
2. Implement DrafterTrainingExecutionStrategy.execute(plan, executor,
   runtime_state) in execution_strategy.py.
3. Select the strategy in DrafterScheduler.plan_training(). Reuse the trigger
   and budget policies unless the new semantics require different policies.
4. Route the strategy in DrafterScheduler.execute_training_plan(). Unknown
   strategies must fail explicitly, not silently fall back to sync execution.
5. If resource queries, non-blocking polling, cancellation, or deadline
   execution require new RPCs, add explicit methods to DrafterWorkerExecutor
   and adapt them in CallbackDrafterWorkerExecutor.
6. Preserve preflight, version validation, result aggregation, and publish
   gating. An asynchronous strategy may change waiting behavior but must not
   bypass these consistency constraints.

Bubble Time should usually change only when training is submitted, whether the
result is awaited, and which idle resources are selected. TrainingPlan remains
the single execution input; Workers must not decide for themselves whether to
train.

### 4. Add a Collection Source or Collection Strategy

Examples include SFT data, a third rollout engine, or a new feature format.

1. Add a source value to DrafterCollectionSource and a source code in
   CollectionPlan.metrics().
2. Implement DrafterCollectionAdapter to convert source-specific samples into
   CollectionPayload and bucket them by owner.
3. Register the adapter through DrafterScheduler.register_collection_adapter().
   Do not hand-write another payload-bucketing and bucket-alignment path in
   the Trainer.
4. If collection transaction semantics differ, implement a new
   DrafterCollectionStrategy. Otherwise reuse SyncCollectionStrategy stage,
   commit, rollback, and finalize behavior.
5. If external RPCs are required, extend DrafterCollectionExecutor and
   implement its callback adapter and Worker handler together.
6. Carry one collection_id through stage, commit, rollback, and finalize. A
   successful collection must update a verifiable data version.

### 5. Change Publication Behavior

Examples include different rollout engines, asynchronous hot updates, or a
versioned-weight store.

1. Prefer to reuse PublishPlan and PublishExecutionStrategy.
2. Replace or extend DrafterPublishExecutor only when the RPC boundary differs.
3. Put new publication decision rules in DrafterScheduler.plan_publish(). Do
   not check the publication interval in both Trainer and Worker.
4. Publication input must come from the snapshot of the single publish leader
   after successful training.

## Development Checklist

When adding a strategy or Executor, verify all of the following:

- The outer Trainer calls only the public DrafterScheduler Facade, not an
  internal policy.
- Workers execute only TrainingPlan.to_worker_payload() and do not introduce
  new trigger, interval, or publication decisions.
- All participating Workers pass preflight before training; one failure skips
  all Workers.
- data_version must be consistent. Check target_version only when
  required_target_version is not None.
- An asynchronous strategy has explicit pending, completed, failed, and safe
  publication states; it does not publish unfinished training.
- Failed collection uses abort or rollback and never treats partially committed
  data as trainable.
- New reason, source, and strategy metric codes are stable, and plans and
  outcomes have observable logs.
- Add unit tests for the normal path, skip path, version mismatch,
  duplicate/missing Worker results, and exception cleanup.

Existing test commands:

~~~bash
python -m pytest -q tests/unit tests/integration/test_drafter_runtime_control_contract.py
python -m ruff check verl_speco/trainer/scheduler tests/unit
~~~
