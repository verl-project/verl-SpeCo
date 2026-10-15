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
# Copyright 2026 MIT HAN Lab
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

import logging
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataBuffer:
    """Buffer to store training data from multiple RL steps for draft model training.

    This buffer accumulates data (input_ids, responses, prompts, hidden_states) across
    RL training steps, allowing the draft model to train on a larger dataset that includes
    both current and previous step data.

    Args:
        max_size: Maximum number of samples to store in buffer
        store_hidden_states: Whether to store hidden_states (default: True)
    """

    def __init__(self, max_size: int = 10000, store_hidden_states: bool = True):
        self.max_size = max_size
        self.store_hidden_states = store_hidden_states
        self.buffer: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._global_sample_ids: set[object] = set()
        self._current_step: Optional[int] = 0

    def add_batch(self, batch: dict[str, Any]) -> bool:
        """Add a batch of data to the buffer.

        Args:
            batch: Dictionary containing:
                - input_ids: Tensor of shape [batch_size, seq_len]
                - responses: Tensor of shape [batch_size, response_len]
                - prompts: Tensor of shape [batch_size, prompt_len]
                - hiddens: Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        sample_id = batch.get("_speco_global_sample_id")
        if sample_id is not None and sample_id in self._global_sample_ids:
            return False
        if self.max_size <= 0:
            return False
        if len(self.buffer) >= self.max_size:
            evicted = self.buffer.popleft()
            evicted_id = evicted.get("_speco_global_sample_id")
            if evicted_id is not None:
                self._global_sample_ids.discard(evicted_id)
        batch["step"] = self._current_step
        self.buffer.append(batch)
        if sample_id is not None:
            self._global_sample_ids.add(sample_id)
        return True

    def update_rl_step(self, step: Optional[int] = None):
        """Increment the current RL step counter."""
        self._current_step = step

    def get_all_data(self) -> list[dict[str, Any]]:
        """Get all data from the buffer.

        Returns:
            List of dictionaries, each containing data for one sample
        """
        return list(self.buffer)

    def get_available_data(
        self,
        *,
        target_version: Optional[int] = None,
        reservation_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return unconsumed samples available to a plan.

        A plan may continue to see samples it already reserved, while samples
        reserved by another in-flight plan are hidden.
        """

        result: list[dict[str, Any]] = []
        for sample in self.buffer:
            reserved_by = sample.get("_drafter_reserved_by")
            if reserved_by is not None and reserved_by != reservation_id:
                continue
            if target_version is not None and int(
                sample.get("target_version", sample.get("step", -1))
            ) != int(target_version):
                continue
            result.append(sample)
        return result

    def reserve(
        self,
        reservation_id: str,
        *,
        target_version: int,
        max_samples: int,
    ) -> list[dict[str, Any]]:
        candidates = self.get_available_data(target_version=target_version)
        selected = candidates[: max(int(max_samples), 0)]
        for sample in selected:
            sample["_drafter_reserved_by"] = str(reservation_id)
        return selected

    def reserve_samples(
        self,
        reservation_id: str,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-reserve an existing replay session under a new plan id."""

        buffer_ids = {id(sample) for sample in self.buffer}
        selected = [
            sample
            for sample in samples
            if id(sample) in buffer_ids
            and sample.get("_drafter_reserved_by") in {None, str(reservation_id)}
        ]
        for sample in selected:
            sample["_drafter_reserved_by"] = str(reservation_id)
        return selected

    def release_reservation(self, reservation_id: str) -> int:
        released = 0
        for sample in self.buffer:
            if sample.get("_drafter_reserved_by") == str(reservation_id):
                sample.pop("_drafter_reserved_by", None)
                released += 1
        return released

    def consume(self, reservation_id: str, samples: list[dict[str, Any]]) -> int:
        consumed_ids = {
            id(sample)
            for sample in samples
            if sample.get("_drafter_reserved_by") == str(reservation_id)
        }
        if not consumed_ids:
            return 0
        before = len(self.buffer)
        self.buffer = deque(
            (sample for sample in self.buffer if id(sample) not in consumed_ids),
            maxlen=self.max_size,
        )
        self._global_sample_ids = {
            sample_id
            for sample in self.buffer
            if (sample_id := sample.get("_speco_global_sample_id")) is not None
        }
        return before - len(self.buffer)

    def get_data_from_last_n_steps(self, n: int) -> list[dict[str, Any]]:
        """Get data from the last n RL steps.

        Args:
            n: Number of recent steps to retrieve data from

        Returns:
            List of dictionaries containing data from last n steps
        """
        current_step = self._current_step or 0
        min_step = max(0, current_step - n)
        return [
            sample for sample in self.get_available_data() if sample["step"] >= min_step
        ]

    def get_data_count(self) -> int:
        """Get the current number of samples in the buffer."""
        return len(self.buffer)

    def get_data_count_from_last_n_steps(self, n: int) -> int:
        """Get number of samples from the last n steps."""
        return len(self.get_data_from_last_n_steps(n))

    def clear(self):
        """Clear all data from the buffer."""
        self.buffer.clear()
        self._global_sample_ids.clear()
        self._current_step = 0

    def get_current_step(self) -> int:
        """Get the current RL step number."""
        return self._current_step or 0

    def __len__(self) -> int:
        """Return the number of samples in buffer."""
        return len(self.buffer)
