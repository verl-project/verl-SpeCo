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
"""Deterministic epoch/batch cursor for file-store resume."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.draft_dataset import (  # noqa: E402
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import DraftFeatureSample  # noqa: E402


class _Store:
    def __init__(self, num_samples: int = 10):
        self.keys = [f"k{i}" for i in range(num_samples)]

    def iter_keys(self, *, shuffle: bool = False, seed: int = 0):
        keys = list(self.keys)
        if shuffle:
            random.Random(int(seed)).shuffle(keys)
        return iter(keys)

    def read(self, key: str) -> DraftFeatureSample:
        index = int(key[1:])
        return DraftFeatureSample(
            input_ids=torch.arange(index, index + 4, dtype=torch.long),
            loss_mask=torch.ones(4, dtype=torch.float32),
            hidden_states=torch.zeros(4, 8, dtype=torch.float32),
        )


def _ids(batch) -> list[int]:
    return [int(sample.input_ids[0].item()) for sample in batch]


def _config(**kwargs):
    defaults = {"batch_size": 2, "shuffle": False, "repeat": False}
    defaults.update(kwargs)
    return DraftFeatureDataLoaderConfig(**defaults)


def test_loader_skips_consumed_batches_after_resume() -> None:
    loader = DraftFeatureDataLoader(_Store(), _config())
    iterator = iter(loader)
    assert _ids(next(iterator)) == [0, 1]
    assert _ids(next(iterator)) == [2, 3]
    assert loader.progress() == {"epoch": 0, "batch_index": 2}

    resumed = DraftFeatureDataLoader(
        _Store(), _config(start_epoch=0, skip_batches=2)
    )
    batches = [_ids(batch) for batch in resumed]
    assert batches == [[4, 5], [6, 7], [8, 9]]
    assert resumed.progress() == {"epoch": 0, "batch_index": 5}


def test_loader_resume_cursor_moves_to_next_epoch_when_fully_consumed() -> None:
    loader = DraftFeatureDataLoader(
        _Store(10), _config(batch_size=4, repeat=True)
    )
    iterator = iter(loader)
    assert _ids(next(iterator)) == [0, 1, 2, 3]
    assert _ids(next(iterator)) == [4, 5, 6, 7]
    assert _ids(next(iterator)) == [8, 9]
    assert loader.progress() == {"epoch": 0, "batch_index": 3}

    resumed = DraftFeatureDataLoader(
        _Store(10),
        _config(batch_size=4, repeat=True, start_epoch=0, skip_batches=3),
    )
    iterator = iter(resumed)
    assert _ids(next(iterator)) == [0, 1, 2, 3]
    assert resumed.progress() == {"epoch": 1, "batch_index": 1}


class _FlakyStore(_Store):
    def __init__(self, num_samples: int = 10, fail_key: str | None = None):
        super().__init__(num_samples)
        self.fail_key = fail_key

    def read(self, key: str):
        if key == self.fail_key:
            raise OSError("transient read failure")
        return super().read(key)


def test_loader_resume_rejects_read_errors_in_saved_epoch() -> None:
    loader = DraftFeatureDataLoader(_Store(), _config())
    iterator = iter(loader)
    assert _ids(next(iterator)) == [0, 1]
    assert _ids(next(iterator)) == [2, 3]
    assert loader.progress() == {"epoch": 0, "batch_index": 2}

    # The saved prefix cannot be rebuilt because k2 is unreadable now; skipping
    # two batches would land on a different boundary.
    resumed = DraftFeatureDataLoader(
        _FlakyStore(fail_key="k2"), _config(start_epoch=0, skip_batches=2)
    )
    with pytest.raises(ValueError, match="unreadable"):
        next(iter(resumed))


def test_loader_resume_rejects_cursor_beyond_epoch() -> None:
    resumed = DraftFeatureDataLoader(
        _Store(10), _config(start_epoch=0, skip_batches=99)
    )
    with pytest.raises(ValueError, match="exceeds"):
        next(iter(resumed))


def test_loader_reports_epoch_read_errors() -> None:
    loader = DraftFeatureDataLoader(_Store(), _config())
    assert loader.epoch_read_errors() == 0
    list(loader)
    assert loader.epoch_read_errors() == 0
