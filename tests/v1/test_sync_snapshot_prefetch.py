# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only snapshot overlap preserves publication barriers and versions."""

import ast
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


def fetch_method():
    source = Path("verl_speco/trainer/speco_ray_trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_speco_get_published_drafter_weights"
    )
    namespace = {"time": time, "logger": logging.getLogger(__name__)}
    exec(  # noqa: S102 - execute only the repository method under test
        compile(ast.Module(body=[method], type_ignores=[]), "production", "exec"),
        namespace,
    )
    return namespace[method.name]


def harness(
    *, device="cuda", bubble=False, asynchronous=True, trained=True, prefetch=True
):
    calls = []
    trainer = SimpleNamespace(
        device_name=device,
        _speco_drafter_training_config=lambda: {
            "gpu_publish_snapshot_prefetch": prefetch
        },
        global_steps=12,
        _speco_v1_pending_training=trained,
        _speco_rollout_idle_worker_enabled=lambda: bubble,
        _speco_drafter_schedule_config=lambda: SimpleNamespace(
            publish_async=asynchronous
        ),
        _require_speco_worker_group=lambda: SimpleNamespace(
            maybe_publish_async=lambda **kw: calls.append(kw) or ["ref"]
        ),
        _ray_get_if_needed=lambda refs: ["snapshot"],
        _first_non_null=lambda values: next((v for v in values if v is not None), None),
        speco_maybe_publish=lambda: calls.append("fallback") or ["fallback"],
    )
    return trainer, calls


@pytest.mark.parametrize(
    "options",
    [
        {"device": "npu"},
        {"bubble": True},
        {"asynchronous": False},
        {"trained": False},
        {"prefetch": False},
    ],
)
def test_ineligible_paths_unchanged(options):
    trainer, calls = harness(**options)
    SpecoV1Mixin._speco_prefetch_sync_cpu_snapshot(trainer)
    assert not calls


def test_single_snapshot_consumed_without_second_export():
    trainer, calls = harness()
    SpecoV1Mixin._speco_prefetch_sync_cpu_snapshot(trainer)
    assert calls == [{"source_step": 12, "sync_cpu_only": True}]
    with pytest.raises(RuntimeError, match="not consumed"):
        SpecoV1Mixin._speco_prefetch_sync_cpu_snapshot(trainer)
    assert fetch_method()(trainer) == "snapshot"
    assert trainer._speco_publish_snapshot_prefetch is None
    assert (
        trainer._speco_publish_snapshot_prefetch_metrics[
            "drafter/sync_snapshot_prefetch_hit"
        ]
        == 1
    )
    assert len(calls) == 1


def test_cache_miss_keeps_original_safe_point_export():
    trainer, calls = harness()
    SpecoV1Mixin._speco_prefetch_sync_cpu_snapshot(trainer)
    trainer._ray_get_if_needed = lambda value: (
        value if value == ["fallback"] else [None]
    )
    assert fetch_method()(trainer) == "fallback"
    assert calls[-1] == "fallback"


def test_version_mismatch_is_fatal_before_consumption():
    trainer, _ = harness()
    SpecoV1Mixin._speco_prefetch_sync_cpu_snapshot(trainer)
    trainer.global_steps += 1
    with pytest.raises(RuntimeError, match="source version"):
        fetch_method()(trainer)


def test_upstream_sync_stays_between_prefetch_and_apply():
    events = []

    class Parent:
        def on_step_end(self):
            events.append("actor_sync")

    class Trainer(SpecoV1Mixin, Parent):
        _speco_colocate_admission_closed = False

        def _speco_prefetch_sync_cpu_snapshot(self):
            events.append("host_prefetch")

        def _speco_finalize_v1_drafter_publish(self):
            events.append("apply_and_ack")
            return {}

    Trainer().on_step_end()
    assert events == ["host_prefetch", "actor_sync", "apply_and_ack"]


@pytest.mark.parametrize(
    "bubble,sync,expected",
    [(False, True, True), (True, False, True), (False, False, False)],
)
def test_worker_prefetch_only_requests_cached_host_snapshot(bubble, sync, expected):
    tree = ast.parse(
        Path("verl_speco/workers/speco_worker.py").read_text(encoding="utf-8")
    )
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "maybe_publish_async"
    )
    method.decorator_list = []
    namespace = {"time": time, "logger": logging.getLogger(__name__)}
    exec(  # noqa: S102 - execute only the repository method under test
        compile(ast.Module(body=[method], type_ignores=[]), "worker", "exec"), namespace
    )
    calls = []
    worker = SimpleNamespace(
        enable_drafter=True,
        in_drafter_train_group=True,
        trainer=SimpleNamespace(_bubble_time_enabled=bubble),
        last_global_step=12,
        rank=0,
        maybe_publish=lambda **kw: calls.append(kw) or "host_snapshot",
    )
    result = namespace["maybe_publish_async"](
        worker, source_step=12, sync_cpu_only=sync
    )
    assert calls == ([{"cached_snapshot_only": True}] if expected else [])
    assert result == ("host_snapshot" if expected else None)
    calls.clear()
    assert (
        namespace["maybe_publish_async"](worker, source_step=13, sync_cpu_only=sync)
        is None
    )
    assert not calls


def test_missing_prefetch_option_defaults_to_disabled():
    trainer, calls = harness()
    trainer._speco_drafter_training_config = dict
    SpecoV1Mixin._speco_prefetch_sync_cpu_snapshot(trainer)
    assert calls == []


def test_lifecycle_spans_exclude_each_other_and_preserve_result(monkeypatch):
    from verl_speco.trainer.v1 import speco_mixin

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        speco_mixin, "time", SimpleNamespace(perf_counter=lambda: clock.now)
    )

    class Parent:
        def on_step_end(self):
            clock.now += 8
            return "original"

    class Trainer(SpecoV1Mixin, Parent):
        _speco_colocate_admission_closed = False

        def _speco_prefetch_sync_cpu_snapshot(self):
            clock.now += 0.01

        def _speco_finalize_v1_drafter_publish(self):
            clock.now += 3
            return {"drafter/published": 1}

    trainer = Trainer()
    trainer.device_name = "cuda"
    trainer._pending_sync_metrics = {"existing": 2}
    assert trainer.on_step_end() == "original"
    assert trainer._pending_sync_metrics[
        "timing_s/drafter_v1_upstream_step_end"
    ] == pytest.approx(8)
    assert trainer._pending_sync_metrics[
        "timing_s/drafter_v1_step_end_total"
    ] == pytest.approx(11.01)
    assert trainer._pending_sync_metrics["existing"] == 2
