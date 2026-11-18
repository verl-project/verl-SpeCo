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

# Adapted from training_common/utils/samplers.py.
import torch
from torch.utils.data import Sampler


class DistributedLengthBucketBatchSampler(Sampler):
    """Build deterministic length-local batches, then shard batches across DP."""

    def __init__(
        self,
        lengths,
        batch_size,
        *,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        bucket_size_multiplier=50,
        drop_last=False,
        align_replicas_by_length=False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if bucket_size_multiplier <= 0:
            raise ValueError(
                f"bucket_size_multiplier must be positive, got {bucket_size_multiplier}"
            )

        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = self.batch_size * int(bucket_size_multiplier)
        self.drop_last = bool(drop_last)
        self.align_replicas_by_length = bool(align_replicas_by_length)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _global_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            indices = list(range(len(self.lengths)))

        batches = []
        for offset in range(0, len(indices), self.bucket_size):
            pool = indices[offset : offset + self.bucket_size]
            pool.sort(key=self.lengths.__getitem__, reverse=True)
            for batch_offset in range(0, len(pool), self.batch_size):
                batch = pool[batch_offset : batch_offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.align_replicas_by_length and self.num_replicas > 1:
            groups = [
                batches[offset : offset + self.num_replicas]
                for offset in range(0, len(batches), self.num_replicas)
            ]
            if groups and len(groups[-1]) < self.num_replicas:
                if self.drop_last:
                    groups.pop()
                else:
                    groups[-1].extend(
                        groups[-1][index % len(groups[-1])]
                        for index in range(self.num_replicas - len(groups[-1]))
                    )
            if self.shuffle and len(groups) > 1:
                order = torch.randperm(len(groups), generator=generator).tolist()
                groups = [groups[index] for index in order]
            batches = [batch for group in groups for batch in group]
        elif self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return batches

    def __iter__(self):
        batches = self._global_batches()
        if self.align_replicas_by_length:
            return iter(batches[self.rank :: self.num_replicas])
        if self.drop_last:
            usable = len(batches) - len(batches) % self.num_replicas
            batches = batches[:usable]
        elif batches and len(batches) % self.num_replicas:
            padding = self.num_replicas - len(batches) % self.num_replicas
            batches.extend(batches[index % len(batches)] for index in range(padding))
        return iter(batches[self.rank :: self.num_replicas])

    def __len__(self):
        if self.drop_last:
            global_batches = len(self.lengths) // self.batch_size
            return global_batches // self.num_replicas
        global_batches = (len(self.lengths) + self.batch_size - 1) // self.batch_size
        return (global_batches + self.num_replicas - 1) // self.num_replicas


class DistributedSourceLocalBatchSampler(Sampler):
    """Shuffle requests while keeping each request's segments sequential."""

    def __init__(
        self,
        source_ids,
        segment_starts,
        batch_size,
        *,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        if len(source_ids) != len(segment_starts):
            raise ValueError("source_ids and segment_starts must have equal length")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.source_ids = list(source_ids)
        self.segment_starts = [int(value) for value in segment_starts]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        source_to_indices = {}
        for index, source_id in enumerate(self.source_ids):
            source_to_indices.setdefault(source_id, []).append(index)
        for indices in source_to_indices.values():
            indices.sort(key=lambda index: (self.segment_starts[index], index))
        self._source_to_indices = source_to_indices
        self._sources = list(source_to_indices)
        self.source_count = len(self._sources)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _global_batches(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sources = list(self._sources)
        if self.shuffle and len(sources) > 1:
            order = torch.randperm(len(sources), generator=generator).tolist()
            sources = [sources[index] for index in order]

        ordered_indices = []
        for source_id in sources:
            ordered_indices.extend(self._source_to_indices[source_id])
        batches = [
            ordered_indices[offset : offset + self.batch_size]
            for offset in range(0, len(ordered_indices), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self):
        batches = self._global_batches()
        if self.drop_last:
            usable = len(batches) - len(batches) % self.num_replicas
            batches = batches[:usable]
        elif batches and len(batches) % self.num_replicas:
            padding = self.num_replicas - len(batches) % self.num_replicas
            batches.extend(batches[index % len(batches)] for index in range(padding))
        return iter(batches[self.rank :: self.num_replicas])

    def __len__(self):
        if self.drop_last:
            global_batches = len(self.source_ids) // self.batch_size
            return global_batches // self.num_replicas
        global_batches = (len(self.source_ids) + self.batch_size - 1) // self.batch_size
        return (global_batches + self.num_replicas - 1) // self.num_replicas


# Adapted from training_common/utils/data.py.
from dataclasses import dataclass


@dataclass(frozen=True)
class MTPSegment:
    sample_index: int
    start: int
    end: int
    loss_start: int
    loss_end: int
