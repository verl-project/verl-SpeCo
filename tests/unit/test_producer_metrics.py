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
from __future__ import annotations

import pytest
import sys
from types import SimpleNamespace

from verl_speco.producer.metrics import producer_window_metrics
from verl_speco.trainer.standalone_tracking import _log_producer_tracking_metrics
from verl_speco.trainer.standalone_tracking import (
    _build_standalone_tracking, _finish_standalone_tracking,
    _log_standalone_tracking_metrics,
)


def test_window_uses_wall_time_for_throughput_and_per_sample_residual():
    metrics = producer_window_metrics([
        {"timing": {"e2e": 10, "prefill": 3, "publish_queue_wait": 1, "generate": 2}},
        {"timing": {"e2e": 8, "prefill": 4, "publish_queue_wait": 2}},
    ], elapsed=4)
    assert metrics == {
        "producer/samples_per_second": 0.5,
        "producer/sample_total_time": 9.0,
        "producer/prefill_time": 3.5,
        "producer/publish_queue_wait_time": 1.5,
        "producer/other_time": 3.0,
        "producer/generation_time": 2.0,
    }


def test_existing_responses_do_not_emit_generation_time():
    assert "producer/generation_time" not in producer_window_metrics([
        {"timing": {"e2e": 1, "prefill": 0.5}}
    ], elapsed=1)
    assert producer_window_metrics([], elapsed=1) == {}


def test_partial_window_uses_actual_sample_count():
    metrics = producer_window_metrics([{"timing": {"e2e": 2}}] * 37, elapsed=10)
    assert metrics["producer/samples_per_second"] == pytest.approx(3.7)


def test_metadata_metrics_report_per_sample_latency_and_real_batch_size():
    rows = [{"timing": {
        "e2e": 6.0, "publish_queue": 2.0, "publish_queue_wait": 0.5,
        "publish_transport": 1.0, "publish_retry_wait": 0.2,
        "metadata_batch_size": 4, "metadata_submit": 1.2,
    }}] * 4
    metrics = producer_window_metrics(rows, elapsed=3)
    assert metrics["producer/samples_per_second"] == pytest.approx(4 / 3)
    assert metrics["producer/publish_queue_time"] == 2.0
    assert metrics["producer/publish_queue_wait_time"] == 0.5
    assert metrics["producer/publish_transport_time"] == 1.0
    assert metrics["producer/publish_retry_wait_time"] == 0.2
    assert metrics["producer/metadata_batch_size"] == 4
    assert metrics["producer/metadata_submit_time"] == 1.2
    assert not any("tq" in key for key in metrics)


def test_producer_logging_uses_published_count_and_independent_step_backends():
    calls = []
    class Tracker:
        def log(self, **kwargs):
            calls.append(kwargs)
    metrics = {"producer/samples_per_second": 2.0, "tq/ready_samples": 4.0}
    _log_producer_tracking_metrics([Tracker()], metrics, published_total=137)
    assert calls == [{"data": metrics, "step": 137, "backend": ["swanlab", "tensorboard"]}]


def test_single_tracker_accepts_both_streams_and_records_safe_config(monkeypatch):
    instances = []
    class Tracking:
        supported_backend = ["swanlab"]

        def __init__(self, **kwargs):
            self.config = kwargs["config"]
            self.calls = []
            self.finished = False
            instances.append(self)

        def log(self, **kwargs):
            self.calls.append(kwargs)

        def finish(self):
            self.finished = True

    monkeypatch.setitem(sys.modules, "verl.utils.tracking", SimpleNamespace(Tracking=Tracking))
    config = {
        "trainer": {"logger": ["console", "swanlab"], "project_name": "test"},
        "actor_rollout_ref": {"rollout": {"drafter": {
            "speculative_algorithm": "DSPARK", "training": {"lr": 1e-6, "api_key": "secret"},
        }}},
    }
    trackers = _build_standalone_tracking(config, rank=0)
    assert _build_standalone_tracking(config, rank=1) == []
    _log_producer_tracking_metrics(trackers, {"producer/prefill_time": 1.0}, published_total=100)
    _log_standalone_tracking_metrics(trackers, {"dspark/loss": 0.5}, step=3)
    _finish_standalone_tracking(trackers, rank=0)
    assert len(instances) == 1
    tracker = instances[0]
    assert tracker.finished
    assert tracker.config["standalone_training"] == {"lr": 1e-6, "algorithm": "DSPARK"}
    assert tracker.config["trainer"]["project_name"] == "test"
    assert [call["step"] for call in tracker.calls] == [100, 3]


def test_http_occupancy_is_time_weighted_and_survives_window_reset(monkeypatch):
    import verl_speco.producer.metrics as module
    clock = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    stats = module.ProducerDiagnostics()
    clock[0] = 1
    stats.http_start()
    clock[0] = 3
    stats.http_start()
    clock[0] = 4
    stats.http_finish()
    stats.loop_lag(0.2)
    stats.loop_lag(0.4)
    clock[0] = 5
    first = stats.snapshot()
    assert first["http_inflight_avg"] == 1.0
    assert first["http_inflight_peak"] == 2
    assert first["http_inflight_now"] == 1
    assert first["http_calls"] == 2
    assert first["loop_lag_avg"] == pytest.approx(0.3)
    assert first["loop_lag_max"] == 0.4
    clock[0] = 7
    stats.http_finish()
    clock[0] = 8
    second = stats.snapshot()
    assert second["http_inflight_avg"] == pytest.approx(2 / 3)
    assert second["http_inflight_peak"] == 1
    assert second["http_inflight_now"] == second["http_calls"] == 0
    assert second["loop_lag_max"] == 0


@pytest.mark.parametrize("initialized,node_id,expected", [
    (True, "node-a", "node-a"), (False, "node-a", None), (True, "", None),
])
def test_clock_identity_requires_initialized_ray(monkeypatch, initialized, node_id, expected):
    from verl_speco.producer.metrics import current_ray_node_id

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        is_initialized=lambda: initialized,
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: node_id),
    ))
    assert current_ray_node_id() == expected
