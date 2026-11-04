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
"""Bound deferred producer heap reclamation without retaining payloads."""

from pathlib import Path


class PublishReclaimPolicy:
    def __init__(self, max_bytes=8 << 30, max_releases=8, rss_growth_kib=1 << 20):
        self.max_bytes = max_bytes
        self.max_releases = max_releases
        self.rss_growth_kib = rss_growth_kib
        self.pending_bytes = 0
        self.releases = 0
        self.baseline_rss = None

    def decide(self, released_bytes, snapshot, *, cgroup_pressure=False):
        self.pending_bytes += max(int(released_bytes), 0)
        self.releases += 1
        rss = snapshot.get("rss_gib")  # Snapshot values are KiB despite their labels.
        available = snapshot.get("available_gib")
        unavailable = snapshot.get("node_unavailable_gib")
        if rss is None or available is None:
            return "unknown_memory"
        if self.baseline_rss is None:
            self.baseline_rss = rss
        if (
            cgroup_pressure
            or available < (1 << 20)
            or (
                unavailable is not None and available < (available + unavailable) * 0.05
            )
        ):
            return "memory_pressure"
        if rss - self.baseline_rss >= self.rss_growth_kib:
            return "rss_growth"
        if self.pending_bytes >= self.max_bytes:
            return "released_bytes_limit"
        if self.releases >= self.max_releases:
            return "release_count_limit"
        return "deferred"

    def reset(self, rss=None):
        self.pending_bytes = 0
        self.releases = 0
        self.baseline_rss = rss


def cgroup_memory_pressure():
    for used_path, limit_path in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    ):
        try:
            limit = Path(limit_path).read_text().strip()
            if limit == "max":
                return False
            used = int(Path(used_path).read_text().strip())
            limit = int(limit)
            return limit > 0 and used >= limit * 0.9
        except (OSError, ValueError):
            continue
    return False
