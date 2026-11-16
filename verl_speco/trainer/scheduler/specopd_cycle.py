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
"""Single-use revision-fenced SpecOPD update cycle, independent of replay quotas."""

from dataclasses import dataclass


@dataclass
class SpecOPDCycle:
    actor_revision: int = 0
    drafter_revision: int = 0
    state: str = "ready"
    step: int | None = None
    pending: object = None

    def begin(self, step):
        if self.state != "ready":
            raise RuntimeError(f"cannot collect SpecOPD in state {self.state}")
        self.step, self.state = int(step), "collecting"

    def scored(self, payload, actor_revision, drafter_revision):
        if self.state != "collecting" or (actor_revision, drafter_revision) != (
            self.actor_revision,
            self.drafter_revision,
        ):
            raise RuntimeError("SpecOPD scoring state/revision mismatch")
        self.pending, self.state = payload, "scored"

    def begin_update(self):
        if self.state != "scored":
            raise RuntimeError(
                "SpecOPD group is not scored or has already been consumed"
            )
        self.state = "updating"
        return self.pending

    def updated(self, trained):
        if self.state != "updating":
            raise RuntimeError("SpecOPD update result without an active group")
        self.drafter_revision += int(trained)
        self.pending, self.state = None, "awaiting_publish"

    def published(self):
        if self.state != "awaiting_publish":
            raise RuntimeError("SpecOPD publish outside update boundary")
        self.actor_revision += 1
        self.state = "ready"
