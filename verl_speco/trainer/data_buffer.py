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

import torch

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
        self._current_step: Optional[int] = 0

    def add_batch(self, batch: dict[str, torch.Tensor]):
        """Add a batch of data to the buffer.

        Args:
            batch: Dictionary containing:
                - input_ids: Tensor of shape [batch_size, seq_len]
                - responses: Tensor of shape [batch_size, response_len]
                - prompts: Tensor of shape [batch_size, prompt_len]
                - hiddens: Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch["step"] = self._current_step
        self.buffer.append(batch)

    def update_rl_step(self, step: Optional[int] = None):
        """Increment the current RL step counter."""
        self._current_step = step

    def get_all_data(self) -> list[dict[str, torch.Tensor]]:
        """Get all data from the buffer.

        Returns:
            List of dictionaries, each containing data for one sample
        """
        return list(self.buffer)

    def get_data_from_last_n_steps(self, n: int) -> list[dict[str, torch.Tensor]]:
        """Get data from the last n RL steps.

        Args:
            n: Number of recent steps to retrieve data from

        Returns:
            List of dictionaries containing data from last n steps
        """
        current_step = self._current_step or 0
        min_step = max(0, current_step - n)
        return [sample for sample in self.buffer if sample["step"] >= min_step]

    def get_data_count(self) -> int:
        """Get the current number of samples in the buffer."""
        return len(self.buffer)

    def get_data_count_from_last_n_steps(self, n: int) -> int:
        """Get number of samples from the last n steps."""
        return len(self.get_data_from_last_n_steps(n))

    def clear(self):
        """Clear all data from the buffer."""
        self.buffer.clear()
        self._current_step = 0

    def get_current_step(self) -> int:
        """Get the current RL step number."""
        return self._current_step or 0

    def __len__(self) -> int:
        """Return the number of samples in buffer."""
        return len(self.buffer)
