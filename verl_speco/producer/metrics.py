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
"""Scalar metrics for successful Producer publish windows."""

from __future__ import annotations

import time
from typing import Any


def producer_window_metrics(
    rows: list[dict[str, Any]], elapsed: float
) -> dict[str, float]:
    """Average successful-sample latencies; throughput uses window wall time.

    E2E starts at input preparation and ends after publication/temporary cleanup.
    Generation is averaged only over samples that actually generated a response.
    Publish queue wait measures enqueue blocking, not residence in that queue.
    """
    if not rows:
        return {}
    timings = [row["timing"] for row in rows]
    count = len(timings)

    def average(key: str) -> float:
        return sum(float(t.get(key, 0.0)) for t in timings) / count

    metrics = {
        "producer/samples_per_second": count / max(elapsed, 1e-9),
        "producer/sample_total_time": average("e2e"),
        "producer/prefill_time": average("prefill"),
        "producer/publish_queue_wait_time": average("publish_queue_wait"),
        "producer/other_time": sum(
            max(
                0.0,
                float(t["e2e"])
                - float(t.get("prefill", 0.0))
                - float(t.get("publish_queue_wait", 0.0))
                - float(t.get("generate", 0.0)),
            )
            for t in timings
        )
        / count,
    }
    generated = [float(t["generate"]) for t in timings if "generate" in t]
    if generated:
        metrics["producer/generation_time"] = sum(generated) / len(generated)
    for key in (
        "publish_queue",
        "publish_encode",
        "publish_transport",
        "publish_retry_wait",
        "input_dispatch",
        "input_get_wait",
        "prefill_global_wait",
        "prefill_endpoint_wait",
        "prefill_http",
        "prefill_retry_backoff",
        "metadata_build",
        "publish_enqueue",
        "publish_residence",
        "publish_batch_wait",
        "metadata_ready_delay",
    ):
        if any(key in t for t in timings):
            metrics[f"producer/{key}_time"] = average(key)
    if any("metadata_batch_size" in t for t in timings):
        # Sample-weighted: each sample experiences the latency of its whole RPC.
        metrics["producer/metadata_batch_size"] = average("metadata_batch_size")
        metrics["producer/metadata_submit_time"] = average("metadata_submit")
    return metrics


class ProducerDiagnostics:
    """Event-loop-local counters; HTTP occupancy is weighted by wall time."""

    def __init__(self):
        self._started = self._changed = time.monotonic()
        self.http_inflight = 0
        self._area = 0.0
        self._peak = 0
        self._calls = 0
        self._lag_sum = self._lag_max = 0.0
        self._lag_count = 0

    def http_start(self):
        self._update_area()
        self.http_inflight += 1
        self._peak = max(self._peak, self.http_inflight)
        self._calls += 1

    def http_finish(self):
        self._update_area()
        self.http_inflight -= 1

    def _update_area(self):
        now = time.monotonic()
        self._area += (now - self._changed) * self.http_inflight
        self._changed = now

    def loop_lag(self, seconds):
        self._lag_sum += seconds
        self._lag_max = max(self._lag_max, seconds)
        self._lag_count += 1

    def snapshot(self, *, reset=True):
        now = time.monotonic()
        area = self._area + (now - self._changed) * self.http_inflight
        result = {
            "http_inflight_avg": area / max(now - self._started, 1e-9),
            "http_inflight_peak": self._peak,
            "http_inflight_now": self.http_inflight,
            "http_calls": self._calls,
            "loop_lag_avg": self._lag_sum / max(self._lag_count, 1),
            "loop_lag_max": self._lag_max,
        }
        if reset:
            self._started = self._changed = now
            self._area = 0.0
            self._peak = self.http_inflight
            self._calls = self._lag_count = 0
            self._lag_sum = self._lag_max = 0.0
        return result
