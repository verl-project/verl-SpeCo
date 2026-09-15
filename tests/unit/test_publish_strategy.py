# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from verl_speco.trainer.scheduler import (
    CallbackDrafterPublishExecutor,
    DrafterScheduler,
    PublishPlan,
)


def _plan(*, publish: bool = True, asynchronous: bool = False) -> PublishPlan:
    return PublishPlan(
        publish=publish,
        reason="publish_interval_reached",
        interval_matched=True,
        source_global_step=8,
        asynchronous=asynchronous,
    )


def test_publish_execution_waits_fetches_and_updates_synchronously() -> None:
    events = []
    executor = CallbackDrafterPublishExecutor(
        wait=lambda: events.append("wait") or 1,
        fetch=lambda: events.append("fetch") or {"wrapped": {"weights": 1}},
        update=lambda payload, step, asynchronous: events.append(
            ("update", payload, step, asynchronous)
        ),
        normalize_payload=lambda value: value["wrapped"],
    )

    outcome = DrafterScheduler(publish_executor=executor).execute_publish_plan(
        _plan()
    )

    assert events == ["wait", "fetch", ("update", {"weights": 1}, 8, False)]
    assert outcome.attempted
    assert outcome.published
    assert outcome.waited_submissions == 1


def test_publish_execution_forwards_async_mode() -> None:
    updates = []
    executor = CallbackDrafterPublishExecutor(
        wait=lambda: 0,
        fetch=lambda: {"weights": 1},
        update=lambda payload, step, asynchronous: updates.append(asynchronous),
        normalize_payload=lambda value: value,
    )

    DrafterScheduler(publish_executor=executor).execute_publish_plan(
        _plan(asynchronous=True)
    )

    assert updates == [True]


def test_inactive_publish_plan_does_not_call_executor() -> None:
    executor = CallbackDrafterPublishExecutor(
        wait=lambda: (_ for _ in ()).throw(AssertionError("unexpected wait")),
        fetch=lambda: None,
        update=lambda payload, step, asynchronous: None,
        normalize_payload=lambda value: value,
    )

    outcome = DrafterScheduler(publish_executor=executor).execute_publish_plan(
        _plan(publish=False)
    )

    assert not outcome.attempted
    assert not outcome.published


def test_missing_snapshot_is_attempted_but_not_published() -> None:
    executor = CallbackDrafterPublishExecutor(
        wait=lambda: 0,
        fetch=lambda: None,
        update=lambda payload, step, asynchronous: (_ for _ in ()).throw(
            AssertionError("unexpected update")
        ),
        normalize_payload=lambda value: value,
    )

    outcome = DrafterScheduler(publish_executor=executor).execute_publish_plan(_plan())

    assert outcome.attempted
    assert not outcome.published
