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

# Adapted from training_common/utils/parallel.py.
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def resolve_tensor_parallel_size(configured_size: int) -> int:
    """Resolve the optional training TP override without mutating the config."""
    override = os.environ.get("MTP_TENSOR_PARALLEL_SIZE")
    value = configured_size if override in (None, "") else override
    try:
        tensor_parallel_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MTP_TENSOR_PARALLEL_SIZE must be a positive integer, got {value!r}"
        ) from exc
    if tensor_parallel_size <= 0:
        raise ValueError(
            f"MTP_TENSOR_PARALLEL_SIZE must be a positive integer, got {value!r}"
        )
    return tensor_parallel_size


@dataclass(frozen=True)
class ParallelCoordinates:
    global_rank: int
    world_size: int
    tensor_parallel_size: int
    data_parallel_size: int
    tensor_parallel_rank: int
    data_parallel_rank: int


def parallel_coordinates(
    global_rank: int,
    world_size: int,
    tensor_parallel_size: int,
) -> ParallelCoordinates:
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if world_size <= 0 or world_size % tensor_parallel_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by "
            f"tensor_parallel_size={tensor_parallel_size}"
        )
    if not 0 <= global_rank < world_size:
        raise ValueError(
            f"global_rank={global_rank} is outside world_size={world_size}"
        )

    data_parallel_size = world_size // tensor_parallel_size
    return ParallelCoordinates(
        global_rank=global_rank,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        tensor_parallel_rank=global_rank % tensor_parallel_size,
        data_parallel_rank=global_rank // tensor_parallel_size,
    )


class DeepSpeedParallelTopology:
    """Two-dimensional TP x DP topology shared by vLLM and DeepSpeed."""

    def __init__(self, tensor_parallel_size: int):
        if not dist.is_initialized():
            if tensor_parallel_size != 1:
                raise RuntimeError("TP>1 requires torch.distributed initialization")
            self.coordinates = parallel_coordinates(0, 1, 1)
            self.tensor_parallel_group = None
            self.data_parallel_group = None
            return

        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        self.coordinates = parallel_coordinates(
            global_rank,
            world_size,
            tensor_parallel_size,
        )

        tp_group = None
        for dp_rank in range(self.data_parallel_size):
            ranks = list(
                range(
                    dp_rank * self.tensor_parallel_size,
                    (dp_rank + 1) * self.tensor_parallel_size,
                )
            )
            group = dist.new_group(ranks=ranks)
            if global_rank in ranks:
                tp_group = group

        dp_group = None
        for tp_rank in range(self.tensor_parallel_size):
            ranks = [
                dp_rank * self.tensor_parallel_size + tp_rank
                for dp_rank in range(self.data_parallel_size)
            ]
            group = dist.new_group(ranks=ranks)
            if global_rank in ranks:
                dp_group = group

        if tp_group is None or dp_group is None:
            raise RuntimeError("Failed to assign TP/DP process groups")
        self.tensor_parallel_group = tp_group
        self.data_parallel_group = dp_group

    @property
    def world_size(self) -> int:
        return self.coordinates.world_size

    @property
    def global_rank(self) -> int:
        return self.coordinates.global_rank

    @property
    def tensor_parallel_size(self) -> int:
        return self.coordinates.tensor_parallel_size

    @property
    def data_parallel_size(self) -> int:
        return self.coordinates.data_parallel_size

    @property
    def tensor_parallel_rank(self) -> int:
        return self.coordinates.tensor_parallel_rank

    @property
    def data_parallel_rank(self) -> int:
        return self.coordinates.data_parallel_rank

    # DeepSpeed's MPU interface.
    def get_model_parallel_group(self):
        return self.tensor_parallel_group

    def get_model_parallel_rank(self) -> int:
        return self.tensor_parallel_rank

    def get_model_parallel_world_size(self) -> int:
        return self.tensor_parallel_size

    def get_model_parallel_src_rank(self) -> int:
        return self.data_parallel_rank * self.tensor_parallel_size

    def get_data_parallel_group(self):
        return self.data_parallel_group

    def get_data_parallel_rank(self) -> int:
        return self.data_parallel_rank

    def get_data_parallel_world_size(self) -> int:
        return self.data_parallel_size

    def get_data_parallel_src_rank(self) -> int:
        return self.tensor_parallel_rank

    # Common aliases used by newer model-parallel integrations.
    get_tensor_model_parallel_group = get_model_parallel_group
    get_tensor_model_parallel_rank = get_model_parallel_rank
    get_tensor_model_parallel_world_size = get_model_parallel_world_size

    def metadata(self) -> dict[str, int]:
        return {
            "world_size": self.world_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "global_rank": self.global_rank,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "data_parallel_rank": self.data_parallel_rank,
        }


@torch.no_grad()
def audit_data_parallel_parameter_replicas(
    model,
    topology: DeepSpeedParallelTopology,
    *,
    sample_elements: int = 32,
) -> dict[str, int]:
    """Fail if corresponding model shards differ across the training DP group."""
    if sample_elements <= 0:
        raise ValueError("sample_elements must be positive")
    parameters = list(model.named_parameters())
    if topology.data_parallel_size == 1:
        return {"parameters": len(parameters), "sampled_elements": 0}

    checked = 0
    sampled = 0
    for name, parameter in parameters:
        flat = parameter.detach().reshape(-1)
        count = min(sample_elements, flat.numel())
        head_count = (count + 1) // 2
        tail_count = count - head_count
        pieces = [flat[:head_count]]
        if tail_count:
            pieces.append(flat[-tail_count:])
        local_sample = torch.cat(pieces).contiguous()
        gathered = [
            torch.empty_like(local_sample) for _ in range(topology.data_parallel_size)
        ]
        dist.all_gather(
            gathered,
            local_sample,
            group=topology.data_parallel_group,
        )
        reference = gathered[0]
        if any(not torch.equal(reference, replica) for replica in gathered[1:]):
            raise RuntimeError(
                "Parameter shards are not replicated across the DeepSpeed DP "
                f"group: parameter={name}, tp_rank={topology.tensor_parallel_rank}"
            )
        checked += 1
        sampled += count
    return {"parameters": checked, "sampled_elements": sampled}


# Adapted from training_common/utils/distributed.py.


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
