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

"""Small, dependency-light V1 feature representation.

This is intentionally limited to ordinary V1 PPO batches in Phase 1.  Agent
request/turn fields and context-plus-assistant slicing are added in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class V1DrafterBatch:
    """Explicit names for the fields needed by future V1 collection hooks."""

    input_ids: Any
    attention_mask: Any
    response_mask: Any | None = None
    hidden_states: Any | None = None
    target_logprobs: Any | None = None
    global_step: int | None = None


def from_transfer_queue_batch(
    batch: Any, *, global_step: int | None = None
) -> V1DrafterBatch:
    """Build a feature view without assuming a legacy upstream batch container.

    ``batch`` may be a TensorDict-like object or a mapping. Missing optional
    fields stay ``None``; no padding or token-window policy is applied here.
    """

    def get(name: str, default=None):
        if hasattr(batch, "get"):
            return batch.get(name, default)
        try:
            return batch[name]
        except (KeyError, TypeError):
            return default

    return V1DrafterBatch(
        input_ids=get("input_ids"),
        attention_mask=get("attention_mask"),
        response_mask=get("response_mask"),
        hidden_states=get("hidden_states"),
        target_logprobs=get("target_logprobs"),
        global_step=global_step,
    )
