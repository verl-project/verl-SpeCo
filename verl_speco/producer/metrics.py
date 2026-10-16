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

from typing import Any


def producer_window_metrics(
    rows: list[dict[str, Any]], elapsed: float
) -> dict[str, float]:
    """Average successful-sample latencies; throughput uses window wall time.

    E2E starts at input preparation and ends after TQ publish/temporary cleanup.
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
    return metrics
