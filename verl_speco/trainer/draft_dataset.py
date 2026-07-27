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
"""Dataset helpers for standalone draft feature stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from verl_speco.trainer.feature_store import DraftFeatureSample, DraftFeatureStore


@dataclass(frozen=True)
class DraftFeatureDataLoaderConfig:
    batch_size: int
    rank: int = 0
    world_size: int = 1
    shuffle: bool = True
    seed: int = 0
    repeat: bool = True


class DraftFeatureDataLoader:
    """Small iterable loader over a DraftFeatureStore.

    The first implementation deliberately keeps sharding simple:
    ``rank_keys = keys[rank::world_size]``. Distributed ranks are truncated to
    the same sample count so every rank executes the same number of FSDP
    collectives when the store size is not divisible by ``world_size``.
    """

    def __init__(self, store: DraftFeatureStore, config: DraftFeatureDataLoaderConfig):
        self.store = store
        self.config = config
        rank = int(config.rank)
        world_size = int(config.world_size)
        if world_size <= 0:
            raise ValueError(f"Invalid world_size: {world_size}")
        if not (0 <= rank < world_size):
            raise ValueError(
                f"Invalid rank/world_size configuration: rank={rank}, world_size={world_size}"
            )

    def __iter__(self) -> Iterator[list[DraftFeatureSample]]:
        epoch = 0
        while True:
            keys = list(
                self.store.iter_keys(
                    shuffle=bool(self.config.shuffle),
                    seed=int(self.config.seed) + epoch,
                )
            )
            if not keys:
                return
            rank = int(self.config.rank)
            world_size = int(self.config.world_size)
            rank_keys = keys[rank::world_size]
            if world_size > 1:
                rank_keys = rank_keys[: len(keys) // world_size]
            batch: list[DraftFeatureSample] = []
            for key in rank_keys:
                batch.append(self.store.read(key))
                if len(batch) >= int(self.config.batch_size):
                    yield batch
                    batch = []
            if batch:
                yield batch
            if not self.config.repeat:
                return
            epoch += 1
