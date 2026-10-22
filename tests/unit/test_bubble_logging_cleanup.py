# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Keep Bubble diagnostics visible without duplicate console writes."""

import ast
import logging
import re
from pathlib import Path
from io import StringIO

import pytest

from test_idle_window_readmission import _live_episode_plan
from verl_speco.trainer.scheduler import DrafterScheduler, DrafterScheduleConfig
from verl_speco.integration.bubble_logging import configure_bubble_logger


_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = (
    "verl_speco/trainer/scheduler/drafter_scheduler.py",
    "verl_speco/trainer/speco_ray_trainer.py",
    "verl_speco/workers/speco_worker.py",
)


def _bubble_event(expression):
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        prefix = expression.value
    elif isinstance(expression, ast.JoinedStr):
        parts = []
        for part in expression.values:
            if not isinstance(part, ast.Constant):
                break
            parts.append(part.value)
        prefix = "".join(parts)
    else:
        return None
    match = re.search(r"\[BubbleTime\]\s+([A-Za-z_][A-Za-z0-9_]*)", prefix)
    return match.group(1) if match else None


@pytest.mark.parametrize(
    "source",
    [
        '"[BubbleTime] publish_version: value"',
        'f"[BubbleTime] publish_version: {version}"',
    ],
)
def test_bubble_event_preserves_distinct_publish_records(source):
    event = _bubble_event(ast.parse(source, mode="eval").body)
    assert event == "publish_version"
    assert event != "publish_succeeded"


@pytest.mark.parametrize("source", _SOURCES)
def test_bubble_records_have_valid_format_arguments_and_no_duplicate_adjacent_print(
    source,
):
    tree = ast.parse((_ROOT / source).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements):
                if not isinstance(statement, ast.Expr):
                    continue
                call = statement.value
                if (
                    not isinstance(call, ast.Call)
                    or not isinstance(call.func, ast.Attribute)
                    or not isinstance(call.func.value, ast.Name)
                    or call.func.value.id != "logger"
                    or not call.args
                    or not isinstance(call.args[0], ast.Constant)
                    or not isinstance(call.args[0].value, str)
                    or "[BubbleTime]" not in call.args[0].value
                ):
                    continue
                if len(call.args) > 1 and not any(
                    isinstance(arg, ast.Starred) for arg in call.args
                ):
                    # Python's formatter validates placeholder count and types
                    # without evaluating diagnostic arguments or touching Ray.
                    call.args[0].value % tuple(0 for _ in call.args[1:])
                if index + 1 < len(statements):
                    following = statements[index + 1]
                    assert not (
                        isinstance(following, ast.Expr)
                        and isinstance(following.value, ast.Call)
                        and isinstance(following.value.func, ast.Name)
                        and following.value.func.id == "print"
                        and following.value.args
                        and _bubble_event(call.args[0]) is not None
                        and _bubble_event(following.value.args[0])
                        == _bubble_event(call.args[0])
                    ), (source, statement.lineno)


def test_idle_budget_is_visible_once_at_default_warning_level(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        _, _, _, _, plan, _ = _live_episode_plan(monkeypatch)
    rows = [
        record.getMessage() for record in caplog.records if "idle_budget:" in record.msg
    ]
    assert len(rows) == 1
    assert f"planned_batches={plan.max_batches}" in rows[0]
    for field in (
        "first_step_extra_s=",
        "window_source=",
        "calibrated_budgets=",
        "writer_steady_samples=",
        "writer_partial_tail_samples=",
    ):
        assert field in rows[0]


def test_real_idle_window_has_one_complete_timeline_record(caplog):
    scheduler = DrafterScheduler()
    scheduler._replica_idle_started_at[0] = 100.0
    scheduler._replica_idle_event_delivery_lag_sec[0] = 0.02
    with caplog.at_level(logging.WARNING):
        window = scheduler._record_observed_replica_idle_window(
            0, 105.0, source="generation_completed"
        )
    assert window == 5.0
    rows = [record.getMessage() for record in caplog.records]
    assert len(rows) == 1
    assert "idle_window_timeline:" in rows[0]
    assert "observed_window_s=5.0000" in rows[0]
    assert "conservative_window_s=5.000" in rows[0]
    assert "samples=1 event_delivery_lag_s=0.020" in rows[0]


def test_retired_multiplier_is_not_a_scheduler_or_default_yaml_option():
    assert "idle_worker_speculative_window_multiplier" not in (
        DrafterScheduleConfig.__dataclass_fields__
    )
    assert "speculative_window_multiplier" not in (
        _ROOT / "verl_speco/config/speco_base.yaml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("parent_level", [logging.CRITICAL, logging.INFO])
def test_bubble_route_survives_disabled_logger_and_root_filtering(parent_level, capsys):
    subject = logging.Logger("bubble-driver", logging.NOTSET)
    parent = logging.Logger("filtered-parent", parent_level)
    root_output = StringIO()
    parent.addHandler(logging.StreamHandler(root_output))
    subject.parent = parent
    subject.disabled = True
    configure_bubble_logger(subject)
    configure_bubble_logger(subject)
    subject.warning("[BubbleTime] idle_budget: planned_batches=5")
    subject.info("hidden detail")
    emitted = capsys.readouterr().err
    assert emitted.count("idle_budget:") == 1
    assert "hidden detail" not in emitted
    assert not root_output.getvalue()
    assert parent.level == parent_level
    assert len(subject.handlers) == 1
    assert not subject.propagate
    assert not subject.disabled
    for handler in subject.handlers:
        handler.close()


def test_bubble_logging_configuration_is_skipped_for_sync(monkeypatch):
    from test_bubble_review_regressions import _methods
    from verl_speco.integration import bubble_logging

    monkeypatch.setattr(
        bubble_logging,
        "configure_bubble_logger",
        lambda logger: pytest.fail("Sync must not reconfigure Bubble loggers"),
    )
    monkeypatch.delenv("SPECO_ROLLOUT_IDLE_EVENT_BUS", raising=False)
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_configure_rollout_idle_event_bus"},
        os=__import__("os"),
        SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV="SPECO_ROLLOUT_IDLE_EVENT_BUS",
    )
    driver = cls()
    driver._speco_rollout_idle_worker_enabled = lambda: False
    assert driver._speco_configure_rollout_idle_event_bus() is None
