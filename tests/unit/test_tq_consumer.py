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

from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.feature_store import (
    DraftFeatureSample,
    build_feature_store_from_config,
)
from verl_speco.trainer.tq_feature_store import ReadyEntry, TQFeatureStore
from verl_speco.trainer.tq_sample_source import (
    TQFeatureDataLoader,
    build_assignments,
)
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    SampleMetadata,
    encode_sample,
    make_ready_tag,
    make_sample_key,
)


def _config() -> dict:
    return {
        "enable": True,
        "ray": {"address": "ray-head:6379", "namespace": "speco-drafter"},
        "partition_id": "speco_drafter_features",
        "run_id": "run-a",
        "schema_version": PROTOCOL_SCHEMA_VERSION,
    }


def _metadata(sequence_no: int = 0) -> SampleMetadata:
    return SampleMetadata(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        run_id="run-a",
        sample_id=f"sample-{sequence_no}",
        sequence_no=sequence_no,
    )


def _sample() -> DraftFeatureSample:
    return DraftFeatureSample(
        algorithm="DSPARK",
        input_ids=torch.tensor([10, 11, 12]),
        loss_mask=torch.tensor([1.0, 1.0, 1.0]),
        position_ids=torch.tensor([0, 1, 2]),
        hidden_states=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        metadata={"target_model_revision": "producer-revision"},
    )


def _entry(sequence_no: int) -> ReadyEntry:
    meta = _metadata(sequence_no)
    return ReadyEntry(key=make_sample_key(meta), tag=make_ready_tag(meta))


def test_feature_store_factory_builds_tq_without_path() -> None:
    store = build_feature_store_from_config(
        {"type": "tq", "path": None},
        read_only=True,
        transfer_queue_cfg=_config(),
    )
    assert isinstance(store, TQFeatureStore)
    assert store.run_id == "run-a"


def test_tq_store_connect_filter_sort_and_minimal_decode(monkeypatch) -> None:
    import verl_speco.trainer.tq_feature_store as module

    calls: list[tuple] = []
    monkeypatch.setattr(module, "configure_transfer_queue", lambda cfg: True)
    monkeypatch.setattr(
        module,
        "connect_ray_cluster",
        lambda address, namespace: calls.append(("ray", address, namespace)),
    )
    monkeypatch.setattr(
        module,
        "connect_transfer_queue_client",
        lambda: calls.append(("tq",)),
    )
    entries = [_entry(2), _entry(1)]
    unrelated = ReadyEntry(
        key="other",
        tag={
            **entries[0].tag,
            "run_id": "another-run",
        },
    )
    monkeypatch.setattr(
        module,
        "list_samples",
        lambda: {
            entries[0].key: entries[0].tag,
            unrelated.key: unrelated.tag,
            entries[1].key: entries[1].tag,
            "control:v2:run-a:eos": {
                "record_type": "control",
                "status": "eos",
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "run_id": "run-a",
                "total_samples": 2,
            },
        },
    )
    fields_by_key = {
        entry.key: encode_sample(_sample(), _metadata(int(entry.tag["sequence_no"])))
        for entry in entries
    }
    monkeypatch.setattr(
        module,
        "get_samples",
        lambda keys: [(key, fields_by_key[key]) for key in keys],
    )

    store = TQFeatureStore.from_config(_config())
    store.connect()
    ready = store.list_ready()
    samples = store.get_many(ready)

    assert calls == [("ray", "ray-head:6379", "speco-drafter"), ("tq",)]
    assert [entry.tag["sequence_no"] for entry in ready] == [1, 2]
    assert len(samples) == 2
    # Model/revision/tokenizer/layers are intentionally not in the first
    # Consumer contract; tensor and run/protocol checks still execute.
    assert samples[0].metadata["target_model_revision"] == "producer-revision"
    eos = store.read_eos()
    assert eos is not None and eos.total_samples == 2


def test_build_assignments_is_disjoint_and_complete() -> None:
    entries = [_entry(index) for index in range(4)]
    assignments = build_assignments(entries, batch_size=2, world_size=2)
    assert [[entry.key for entry in rank] for rank in assignments] == [
        [entries[0].key, entries[1].key],
        [entries[2].key, entries[3].key],
    ]


class _FakeStore:
    def __init__(self, entries: list[ReadyEntry], *, eos: bool = True):
        self.entries = list(entries)
        self.eos = eos
        self.connected = False
        self.get_calls: list[list[str]] = []
        self.clear_calls: list[list[str]] = []

    def connect(self) -> None:
        self.connected = True

    def owner_ready(self) -> bool:
        return True

    def list_ready(self):
        return list(self.entries)

    def read_eos(self):
        return SimpleNamespace(total_samples=len(self.entries)) if self.eos else None

    def get_many(self, entries):
        self.get_calls.append([entry.key for entry in entries])
        return [_sample() for _ in entries]

    def clear_many(self, keys):
        normalized = list(keys)
        self.clear_calls.append(normalized)
        selected = set(normalized)
        self.entries = [entry for entry in self.entries if entry.key not in selected]


def test_world_size_one_streams_trains_then_clears_and_stops() -> None:
    entries = [_entry(0), _entry(1)]
    store = _FakeStore(entries)
    loader = TQFeatureDataLoader(
        store, batch_size=2, rank=0, world_size=1, poll_interval_seconds=0.01
    )
    iterator = iter(loader)
    batch = next(iterator)
    assert batch.local_keys == [entry.key for entry in entries]
    assert batch.global_keys == batch.local_keys
    assert batch.global_sequence_nos == [0, 1]
    assert store.get_calls == [batch.local_keys]

    loader.clear_completed_batch(batch.global_keys)
    with pytest.raises(StopIteration):
        next(iterator)
    assert store.clear_calls == [batch.local_keys]


def test_eos_drops_incomplete_global_tail() -> None:
    entry = _entry(0)
    store = _FakeStore([entry])
    loader = TQFeatureDataLoader(store, batch_size=2, rank=0, world_size=1)
    assert list(loader) == []
    assert store.clear_calls == [[entry.key]]


def test_rank0_discovery_failure_is_raised_without_clearing() -> None:
    store = _FakeStore([_entry(0)])

    def _fail_list():
        raise RuntimeError("list failed")

    store.list_ready = _fail_list
    loader = TQFeatureDataLoader(store, batch_size=1, rank=0, world_size=1)
    with pytest.raises(RuntimeError, match="list failed"):
        next(iter(loader))
    assert store.clear_calls == []


def test_nonzero_rank_uses_broadcast_assignment_without_listing(monkeypatch) -> None:
    import verl_speco.trainer.tq_sample_source as module

    entries = [_entry(0), _entry(1)]
    store = _FakeStore(entries, eos=False)

    def _broadcast(payload, src):
        assert src == 0
        payload[0] = {
            "kind": "batch",
            "global_keys": [entry.key for entry in entries],
            "assignments": [
                [{"key": entries[0].key, "tag": entries[0].tag}],
                [{"key": entries[1].key, "tag": entries[1].tag}],
            ],
        }

    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "broadcast_object_list", _broadcast)
    store.list_ready = lambda: pytest.fail("nonzero rank must not list TQ keys")

    loader = TQFeatureDataLoader(store, batch_size=1, rank=1, world_size=2)
    batch = next(iter(loader))
    assert batch.local_keys == [entries[1].key]
    assert batch.global_keys is None
    assert store.get_calls == [[entries[1].key]]
