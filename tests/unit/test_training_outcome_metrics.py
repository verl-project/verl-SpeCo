# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from verl_speco.trainer.scheduler import (
    DrafterExecutionStrategy,
    DrafterRuntimeState,
    TrainingPlan,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome
from verl_speco.trainer.scheduler.training_outcome import (
    TrainingOutcome,
    _forward_block_metrics,
)


def _plan() -> TrainingPlan:
    return TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=4,
        max_batches=2,
        publish_after_success=False,
        plan_id="plan-4",
    )


def _running_state(plan: TrainingPlan) -> DrafterRuntimeState:
    state = DrafterRuntimeState()
    state.submit(plan, started_at=0.0)
    state.mark_running()
    return state


def _result(**extra: object) -> dict[str, object]:
    result: dict[str, object] = {
        "triggered": True,
        "trained": True,
        "attempted_steps": 1,
        "successful_steps": 1,
        "elapsed_sec": 1.0,
        "reason": "trained",
    }
    result.update(extra)
    return result


def test_from_execution_forwards_confidence_metrics() -> None:
    """Co-train confidence scalars must survive the outcome whitelist."""
    plan = _plan()
    raw = [
        _result(
            **{
                "dspark/confidence_loss": 0.5,
                "dspark/confidence_accept_rate": 0.66,
                "dspark/confidence_pred_mean": 0.67,
                "dspark/confidence_weighted_token_count": 128.0,
                "dspark/ce_loss": 1.5,
                "dspark/l1_loss": 0.6,
                "dspark/accuracy": 0.7,
                "dspark/valid_token_count": 999.0,  # not whitelisted
                "unrelated/metric": 3.0,  # not whitelisted
            }
        )
    ]

    outcome = TrainingOutcome.from_execution(
        ExecutionOutcome(raw_results=raw, elapsed_sec=1.0),
        runtime_state=_running_state(plan),
        plan=plan,
    )

    assert outcome.metrics["dspark/confidence_loss"] == 0.5
    assert outcome.metrics["dspark/confidence_accept_rate"] == 0.66
    assert outcome.metrics["dspark/confidence_pred_mean"] == 0.67
    assert outcome.metrics["dspark/confidence_weighted_token_count"] == 128.0
    assert outcome.metrics["dspark/ce_loss"] == 1.5
    assert outcome.metrics["dspark/l1_loss"] == 0.6
    assert outcome.metrics["dspark/accuracy"] == 0.7
    assert "dspark/valid_token_count" not in outcome.metrics
    assert "unrelated/metric" not in outcome.metrics


def test_from_execution_keeps_summary_metrics() -> None:
    """Forwarding block scalars must not clobber the drafter/* summary keys."""
    plan = _plan()
    raw = [_result(**{"dspark/ce_loss": 1.0})]

    outcome = TrainingOutcome.from_execution(
        ExecutionOutcome(raw_results=raw, elapsed_sec=1.0),
        runtime_state=_running_state(plan),
        plan=plan,
    )

    assert outcome.metrics["drafter/trained"] == 1
    assert outcome.metrics["drafter/train_successful_steps_max"] == 1
    assert outcome.metrics["dspark/ce_loss"] == 1.0


def test_forward_block_metrics_averages_across_workers() -> None:
    metrics = _forward_block_metrics(
        [
            {"dspark/confidence_loss": 0.4, "dspark/confdience_typo": 1.0},
            {"dspark/confidence_loss": 0.6},
        ]
    )
    assert metrics == {"dspark/confidence_loss": 0.5}
    assert "dspark/confdience_typo" not in metrics


def test_forward_block_metrics_skips_non_numeric_and_unprefixed() -> None:
    metrics = _forward_block_metrics(
        [
            {
                "dspark/ce_loss": "not-a-number",
                "ce_loss": 1.0,  # no block prefix -> ignored
                "dspark/l1_loss": 0.2,
            },
            None,
        ]
    )
    assert metrics == {"dspark/l1_loss": 0.2}
