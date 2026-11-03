# Adaptive drafter scheduling and training budget

Last updated: 09/30/2026

This document describes the optional adaptive extension to the default scheduler
documented in [Drafter Scheduler Implementation and Strategy Integration Guide](drafter_scheduler.md).
The extension adds startup training/collection opportunities and a feedback-driven
training budget while retaining the existing trigger, execution and publication
contracts.

The controller consumes the current training rollout's vLLM
`drafter/spec_decode/mean_acceptance_length`, including the bonus token.
It never uses acceptance rate or divides accepted tokens by drafted tokens.
The trainer aggregates runtime accepted-token and verification-round counts
before computing the existing mean-length metric, so worker means are weighted
by verification rounds. Validation rollouts are excluded.

## Configuration

All settings remain under `actor_rollout_ref.rollout.drafter.training.adaptive_schedule`:

```yaml
enable: false
warmup_train_steps: 20
warmup_max_steps: 20
warmup_window_size: 5
warmup_min_improvement: 0.1
warmup_patience: 3
min_train_steps: 0
max_train_steps: 25
max_step_change: 20
ema_fast_alpha: 0.3
ema_slow_alpha: 0.05
trend_tolerance: 0.02
min_feedback_samples: 32
```

The YAML above shows the defaults in `verl_speco/config/speco_base.yaml`.
The Python config fallback also defaults to `enable: false`. Set `enable: true`
to opt in; the adaptive example launch script does this explicitly.
With the policy disabled, collection and training use their configured intervals,
and `SyncTrainingBudgetPolicy` uses `training.step` as the training budget.
Explicit `training.step=0` continues to disable training.

## Scheduler integration and extension points

The adaptive policy separates frequency decisions from permission to execute:

- The pure `training_opportunity()` function in
  `verl_speco/trainer/scheduler/training_opportunity.py` evaluates configured
  intervals and the startup override. Collection and training share this decision
  so their startup behavior agrees.
- `IntervalAndBufferTrigger` remains the default trigger policy. It retains
  collect-only, pending-training, current-step
  sample, Worker data/target version and minimum trainable-batch guards.
- `AdaptiveTrainingBudgetPolicy` supplies the session budget when enabled,
  preserving the explicit `training.step=0` disable switch. The budget must still
  satisfy the plan's minimum-batch requirement. `TrainingPlan.launch` remains the
  final execution gate.

The generic `TrainingTriggerPolicy.should_train()` extension point described in
the scheduler guide remains available for acceptance, loss or external-event
conditions. Changes that must coordinate collection and training frequency should
use the shared `training_opportunity()` function. Preserve the existing execution
and data guards, and keep `prepare_training_plan()` cheap skips before Worker
inspection RPCs. Preserve skip reasons and add stable numeric codes to
`TrainingPlan._REASON_CODES` for new reasons.

This extension reuses `SyncExecutionStrategy`, Worker preflight and the existing
publication execution path. Adaptive state records successful training and
completed publication through scheduler lifecycle events; it does not bypass
version validation or successful-training publication gating. The outer Trainer
supplies acceptance feedback and checkpoint persistence without making trigger
or budget decisions.

## Warmup

Warmup provides training/collection opportunities every RL step and always uses
`warmup_train_steps`, independently of the adaptive bounds. It ends at absolute
`global_step >= warmup_max_steps`, including when feedback is unavailable.
It may also end early: keep the latest `warmup_window_size` valid mean lengths;
once full, compare latest minus earliest to `warmup_min_improvement`. Consecutive
low-gain windows increment a counter; improvement resets it. After
`warmup_patience` low-gain windows, warmup ends on the next step so collection and
training agree within the current step. With constant lengths and the defaults,
observation 7 triggers early exit.

EMAs learn during warmup, but the dynamic budget is not adjusted. At the first
observation call after warmup ends, both EMAs reset to the mean of the recent
warmup acceptance-length window and the interval trend history is cleared. The
transition step establishes this baseline (zero relative trend); its feedback
does not advance either EMA. Subsequent valid feedback updates the EMAs normally.
If no warmup history exists, the first valid length initializes both EMAs. The post-warmup
budget starts from `warmup_train_steps` clamped to the adaptive bounds. There is
no separate initial-budget setting. Configured training/collection intervals
resume after warmup; budget changes never modify these intervals.

## Trend and budget

After transition, each valid feedback updates the fast and slow EMA of mean acceptance length:

```text
ema = alpha * length + (1 - alpha) * previous_ema
relative_trend = (ema_fast - ema_slow) / max(abs(ema_slow), 1e-8)
```

The interval mean below `-trend_tolerance` is a decline; above
`trend_tolerance` is an improvement.
Each complete valid interval acts immediately on its mean trend, without
cross-interval confirmation or directional counters. Missing, invalid,
unpublished, duplicate and stale feedback are excluded. The normalization
and integer rounding are explicit:

```text
interval_trend = mean(valid relative_trends in this training interval)
severity = min(abs(interval_trend) / 0.2, 1.0)
delta_steps = ceil(severity * max_step_change)
budget += delta_steps  # decline
budget -= delta_steps  # improvement
budget = clamp(budget, min_train_steps, max_train_steps)
```

The internal normalization scale is 0.2 (20%); it is not a configuration option.
With max change 20, relative-trend magnitudes 0.03, 0.05, 0.15 and 0.2 produce
changes of 3, 5, 15 and 20 steps, respectively. Larger changes are capped at 20.
A budget of 5 can become 20 for a 15% decline or 25 for a 20% decline. Inside the tolerance band (including its boundaries),
the budget is unchanged. No minimum-step-change setting is needed.
Observation updates EMAs, sample counts and timestamped interval trend history only; it never changes
budget or the decision reason. The scheduler calls `decide_budget()` only at an
existing training opportunity, respecting collect-only and pending-training
guards.
Each opportunity makes at most one decision, including repeated planning of the
same RL step. The window size is the existing `training_interval_steps`, with no new setting.
A decision requires a full interval of valid trends. For interval 5 the window
ending at step 10 contains steps 6 through 10. Warmup/transition, duplicate,
stale, unpublished, missing and invalid feedback do not populate this history.
An incomplete interval holds the budget.
Every decision clears the history, including holds and invalid feedback. Old
intervals also expire if execution guards prevented a decision. The first partial
interval after warmup therefore holds its budget. Outside training
opportunities, the metrics retain the last budget and decision reason.
No target acceptance threshold or fixed
step increment is used.

A zero budget skips optimizer training for that cycle. Feedback continues to be
consumed, including when no new weights have been published, allowing recovery
from zero. Pending trained weights must still finish publication before new
feedback is consumed. Normal source, data, version and execution guards remain.

`min_feedback_samples` counts vLLM verification rounds contributing to the
current RL step's mean length, not requests or optimizer steps. If there are too
few samples, or length/counts are invalid, neither EMA nor budget is updated.

## Metrics and checkpoints

`drafter/adaptive_budget_steps` reports the effective budget (fixed during
warmup). `adaptive_acceptance_fast` and `adaptive_acceptance_slow` now contain
length EMAs; `adaptive_acceptance_trend` contains the single-step relative trend.
`adaptive_interval_trend` is emitted at valid interval decisions and contains
the mean actually used for budget adjustment. Successful-training INFO logs
label it `interval_trend`; incomplete intervals log `interval_trend=n/a`. Existing
charts must account for this change of units.

Warmup reports `adaptive_warmup_active`, `adaptive_warmup_early_exit`,
`adaptive_warmup_acceptance_length`, `adaptive_warmup_gain`, and
`adaptive_warmup_no_improvement_count`. All use the `drafter/` prefix.

`adaptive_reason`: 0 initial, 1 missing feedback, 2 invalid/insufficient feedback,
3 awaiting publication, 4 hold, 5 length decline, 6 length improvement,
7 fixed warmup budget, 8 restored state.

Checkpoint schema 5 saves the dynamic budget (including zero), length EMAs,
interval trend history, warmup history/counter, transition and last-decision steps,
and publication state. Older schemas
or changed configurations recalibrate instead of interpreting rate EMAs as
length EMAs. The absolute warmup cap still applies on resume.

## Verification

Tests cover decline recovery, severity-dependent adjustment, reduction to zero,
zero-budget skip and subsequent recovery without publishing new weights, fixed
warmup budget, sample filtering, warmup plateau/cap, publication waits,
and checkpoint restoration.
Tests also cover shared collection/training startup frequency, existing trigger
policy vetoes, cheap skips without Worker RPCs, and legacy decisions with adaptive
scheduling disabled.
