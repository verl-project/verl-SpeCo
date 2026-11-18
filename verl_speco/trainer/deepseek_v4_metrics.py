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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

# Adapted from training_common/utils/debug.py.
import os
import random

import numpy as np
import torch

from verl_speco.backends.vllm_ascend_parallel import is_main_process


def set_seed(seed: int):
    import torch_npu

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_npu_peak_memory(device=None):
    import torch_npu

    device = torch_npu.npu.current_device() if device is None else device
    peak_bytes = torch_npu.npu.max_memory_allocated(device)
    return peak_bytes / 1024**3


def print_npu_memory(tag: str):
    import torch_npu

    if is_main_process():
        device = torch_npu.npu.current_device()
        allocated_gb = torch_npu.npu.memory_allocated(device) / (1024**3)
        reserved_gb = torch_npu.npu.memory_reserved(device) / (1024**3)
        print(
            f"[NPU Memory | {tag}] Allocated: {allocated_gb:.3f} GB | Reserved: {reserved_gb:.3f} GB"
        )


# Adapted from training_common/utils/metrics.py.


class MTPPrefixAccuracyComputer:
    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.cumulative_correct_mask = torch.ones(
            batch_size, seq_len, dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def step(
        self,
        student_preds: torch.Tensor,
        ground_truth: torch.Tensor,
        generation_mask: torch.Tensor,
    ):
        correct_at_k = (student_preds == ground_truth).float()
        self.cumulative_correct_mask = self.cumulative_correct_mask * correct_at_k
        correct_count = (self.cumulative_correct_mask * generation_mask).sum()
        token_count = generation_mask.sum()

        if token_count > 0:
            acc = (correct_count / token_count).item()
        else:
            acc = 0.0

        return acc, correct_count, token_count


# Adapted from training_common/utils/runtime_metrics.py.
from collections.abc import Sequence


class DeviceMTPMetricsAccumulator:
    """Accumulate detached token-weighted MTP metrics without host transfers."""

    def __init__(
        self,
        ttt_len: int,
        device: torch.device,
        denominator_epsilon: float = 1.0e-6,
    ) -> None:
        if int(ttt_len) <= 0:
            raise ValueError("ttt_len must be positive")
        self.ttt_len = int(ttt_len)
        self.denominator_epsilon = float(denominator_epsilon)
        self._totals = torch.zeros(
            (2, self.ttt_len),
            dtype=torch.float32,
            device=device,
        )

    @torch.no_grad()
    def update(
        self,
        step_losses: Sequence[torch.Tensor],
        token_counts: Sequence[torch.Tensor],
    ) -> None:
        if len(step_losses) != self.ttt_len or len(token_counts) != self.ttt_len:
            raise ValueError("MTP metric lengths must match ttt_len")
        losses = torch.stack([value.detach() for value in step_losses]).float()
        counts = torch.stack([value.detach() for value in token_counts]).float()
        self._totals[0].add_(losses * (counts + self.denominator_epsilon))
        self._totals[1].add_(counts)

    @torch.no_grad()
    def get_totals(
        self,
        reset: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_numerators = self._totals[0].clone()
        token_counts = self._totals[1].clone()
        if reset:
            self._totals.zero_()
        return loss_numerators, token_counts

    @torch.no_grad()
    def reset(self) -> None:
        self._totals.zero_()


def should_log_optimizer_step(step: int, interval: int) -> bool:
    if int(step) <= 0:
        raise ValueError("step must be positive")
    if int(interval) <= 0:
        raise ValueError("logging interval must be positive")
    return int(step) == 1 or int(step) % int(interval) == 0
