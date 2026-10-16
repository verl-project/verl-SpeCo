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
"""CPU-light contract tests for the optional RL-Insight bridge."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

import verl_speco.rl_insight_bridge as bridge


class _FakeRLInsight:
    def __init__(self, enabled: bool = True) -> None:
        self.init_calls = 0
        self.init_kwargs = []
        self.events = []
        self.trace_calls = []
        self.exits = []
        self._STATE = SimpleNamespace(enabled=enabled)

    def init(self, *args, **kwargs) -> None:
        self.init_calls += 1
        self.init_kwargs.append(kwargs)

    def metric_gauge(self, name, value, **labels) -> None:
        self.events.append(("gauge", name, value, labels))

    def trace_state(self, state_name, *, state_lane_id=None, **labels):
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                outer.trace_calls.append(("enter", state_name, state_lane_id, labels))
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                outer.trace_calls.append(("exit", state_name))
                outer.exits.append(exc_type)
                return False

        return _Ctx()


@pytest.fixture(autouse=True)
def _reset_bridge():
    bridge.reset_for_tests()
    yield
    bridge.reset_for_tests()


def _enable(monkeypatch, fake):
    monkeypatch.setenv(bridge.ENABLE_ENV, "1")
    monkeypatch.setattr(bridge, "_module", fake)
    return fake


def test_sanitize_metric_name():
    assert bridge._sanitize_metric_name("eagle3/loss") == "eagle3_loss"
    assert (
        bridge._sanitize_metric_name("drafter/spec_decode/mean-acceptance-length")
        == "drafter_spec_decode_mean_acceptance_length"
    )


def test_sanitize_metric_name_folds_other_invalid_chars():
    assert bridge._sanitize_metric_name("a.b c:d") == "a_b_c_d"
    assert bridge._sanitize_metric_name("1leading") == "_1leading"
    assert bridge._sanitize_metric_name("") == "_"


def test_init_forwards_project_and_experiment_from_env(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())
    monkeypatch.setenv(bridge.PROJECT_ENV, "my-project")
    monkeypatch.setenv(bridge.EXPERIMENT_ENV, "my-experiment")

    bridge.log_scalar_metrics("p_", {"a/b": 1.0})

    assert fake.init_kwargs == [
        {"project": "my-project", "experiment_name": "my-experiment"}
    ]


def test_init_without_server_url_marks_unavailable(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight(enabled=False))

    bridge.log_scalar_metrics("p_", {"a": 1.0})

    assert fake.init_calls == 1
    assert fake.events == []
    assert bridge._init_state == "unavailable"


def test_log_scalar_metrics_sanitizes_prefix(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    bridge.log_scalar_metrics("bad prefix/", {"eagle3/loss": 0.5})

    assert fake.events == [("gauge", "bad_prefix_eagle3_loss", 0.5, {})]


def test_disabled_emits_nothing_and_does_not_init(monkeypatch):
    monkeypatch.delenv(bridge.ENABLE_ENV, raising=False)
    fake = _FakeRLInsight()
    monkeypatch.setattr(bridge, "_module", fake)

    bridge.log_scalar_metrics("p_", {"a/b": 1.0})
    with bridge.trace_state("drafter/train", state_lane_id="lane"):
        pass

    assert fake.init_calls == 0
    assert fake.events == []
    assert fake.trace_calls == []


def test_enabled_emits_sanitized_gauge_with_labels(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    bridge.log_scalar_metrics("speco_drafter_worker_", {"eagle3/loss": 0.5}, rank=3)

    assert fake.init_calls == 1
    assert fake.events == [("gauge", "speco_drafter_worker_eagle3_loss", 0.5, {"rank": 3})]


def test_init_runs_once_per_process(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    bridge.log_scalar_metrics("p_", {"a": 1.0})
    bridge.log_scalar_metrics("p_", {"b": 2.0})

    assert fake.init_calls == 1


def test_non_finite_and_non_scalar_values_are_skipped(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    bridge.log_scalar_metrics(
        "p_",
        {
            "a": float("nan"),
            "b": float("inf"),
            "c": float("-inf"),
            "d": None,
            "e": True,
            "f": "not-a-number",
        },
    )

    assert fake.events == []


def test_log_scalar_metrics_keeps_only_finite_scalars(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    bridge.log_scalar_metrics(
        "speco_drafter_worker_",
        {"eagle3/loss": 0.25, "eagle3/top1_acc": float("nan"), "eagle3/steps": 7},
        worker_rank=1,
    )

    assert fake.events == [
        ("gauge", "speco_drafter_worker_eagle3_loss", 0.25, {"worker_rank": 1}),
        ("gauge", "speco_drafter_worker_eagle3_steps", 7.0, {"worker_rank": 1}),
    ]


def test_log_scalar_metrics_ignores_non_mapping(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    bridge.log_scalar_metrics("p_", ["a", 1.0])  # type: ignore[arg-type]

    assert fake.events == []


def test_trace_state_enters_and_exits_with_labels(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    with bridge.trace_state(
        "drafter/train", state_lane_id="drafter_worker_2", global_step=7
    ):
        ran = True

    assert ran
    assert fake.trace_calls == [
        ("enter", "drafter/train", "drafter_worker_2", {"global_step": 7}),
        ("exit", "drafter/train"),
    ]


def test_trace_state_propagates_body_exception(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    with pytest.raises(ValueError):
        with bridge.trace_state("drafter/train", state_lane_id="lane"):
            raise ValueError("boom")

    assert fake.trace_calls[-1] == ("exit", "drafter/train")


def test_trace_state_forwards_body_exception_to_monitor(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    with pytest.raises(ValueError):
        with bridge.trace_state("drafter/train", state_lane_id="lane"):
            raise ValueError("boom")

    assert fake.exits == [ValueError]


def test_trace_state_exit_uses_none_when_body_succeeds(monkeypatch):
    fake = _enable(monkeypatch, _FakeRLInsight())

    with bridge.trace_state("drafter/train", state_lane_id="lane"):
        pass

    assert fake.exits == [None]


def test_trace_state_setup_failure_still_runs_body(monkeypatch):
    class _Exploding:
        def init(self, *args, **kwargs):
            return None

        def trace_state(self, *args, **kwargs):
            raise RuntimeError("no trace")

    monkeypatch.setenv(bridge.ENABLE_ENV, "1")
    monkeypatch.setattr(bridge, "_module", _Exploding())

    with bridge.trace_state("drafter/train", state_lane_id="lane"):
        ran = True

    assert ran


def test_init_failure_degrades_to_unavailable(monkeypatch):
    class _BadInit:
        def init(self, *args, **kwargs):
            raise RuntimeError("no init")

    monkeypatch.setenv(bridge.ENABLE_ENV, "1")
    monkeypatch.setattr(bridge, "_module", _BadInit())

    bridge.log_scalar_metrics("p_", {"eagle3/loss": 0.1})

    assert bridge._init_state == "unavailable"


def test_import_does_not_eagerly_import_rl_insight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import verl_speco.rl_insight_bridge as b; "
            "assert 'rl_insight' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
