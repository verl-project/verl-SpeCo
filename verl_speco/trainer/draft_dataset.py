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
    ``rank_keys = keys[rank::world_size]``. That matches the design doc's P2
    phase-1 recommendation and works for torchrun DP/FSDP ranks.
    """

    def __init__(self, store: DraftFeatureStore, config: DraftFeatureDataLoaderConfig):
        self.store = store
        self.config = config

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
            rank_keys = keys[int(self.config.rank) :: max(int(self.config.world_size), 1)]
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
