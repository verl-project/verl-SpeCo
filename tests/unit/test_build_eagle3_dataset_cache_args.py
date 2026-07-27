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
from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from verl_speco.data.preprocessing import build_eagle3_dataset


class _FakeDataset:
    """Minimal stand-in: build_eagle3_dataset only shuffles, maps, and formats."""

    column_names = ["conversations"]

    def __init__(self):
        self.map_kwargs = None

    def shuffle(self, seed=None):
        return self

    def map(self, fn, **kwargs):
        self.map_kwargs = kwargs
        return self

    def set_format(self, type=None):
        return self


def _build(dataset, **cache_kwargs):
    return build_eagle3_dataset(
        dataset=dataset,
        tokenizer=None,
        chat_template="qwen",
        num_proc=1,
        **cache_kwargs,
    )


def test_no_cache_args_disables_caching():
    dataset = _FakeDataset()
    _build(dataset)
    assert dataset.map_kwargs["load_from_cache_file"] is False
    assert dataset.map_kwargs["cache_file_name"] is None


def test_both_cache_args_enable_caching(tmp_path):
    dataset = _FakeDataset()
    cache_dir = str(tmp_path / "cache")
    _build(dataset, cache_dir=cache_dir, cache_key="unit")
    assert dataset.map_kwargs["load_from_cache_file"] is True
    assert dataset.map_kwargs["cache_file_name"] == os.path.join(cache_dir, "unit.pkl")
    assert os.path.isdir(cache_dir)


@pytest.mark.parametrize(
    "cache_kwargs", [{"cache_dir": "/tmp/x"}, {"cache_key": "unit"}]
)
def test_partial_cache_args_warn_and_fall_back(cache_kwargs):
    """A single cache argument used to leave load_from_cache_file/cache_file_name
    unbound and crash dataset.map with UnboundLocalError."""
    dataset = _FakeDataset()
    with pytest.warns(UserWarning, match="must be provided together"):
        _build(dataset, **cache_kwargs)
    assert dataset.map_kwargs["load_from_cache_file"] is False
    assert dataset.map_kwargs["cache_file_name"] is None
