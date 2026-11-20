# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Keep the reference pipeline usable when legacy TQ adapters are unavailable."""

import subprocess
import sys
from pathlib import Path
from queue import Queue

import pytest
import torch

from verl_speco.trainer.reference_feature_store import ReferenceFeatureStore
from verl_speco.trainer.reference_sample_source import ReferenceFeatureDataLoader
from verl_speco.trainer.standalone_data import ReadyEntry, StandaloneLocalBatch
from verl_speco.trainer.standalone_prefetch import StandalonePrefetchIterator


def test_reference_imports_and_factory_without_legacy_tq_modules():
    # A fresh interpreter catches eager/transitive imports that an already loaded
    # test process would hide. This models removing the legacy adapters later.
    script = """
import importlib.abc
import sys

blocked = {
    "verl_speco.trainer.tq_feature_store",
    "verl_speco.trainer.tq_sample_source",
    "verl_speco.trainer.tq_retry",
    "verl_speco.integration.transferqueue_bridge",
    "transfer_queue",
}
class NoLegacyTQ(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("Legacy TQ dependency: " + fullname)
sys.meta_path.insert(0, NoLegacyTQ())

from verl_speco.trainer.feature_store import build_feature_store_from_config
from verl_speco.trainer.reference_sample_source import ReferenceFeatureDataLoader
from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator
from verl_speco.trainer.standalone_prefetch import StandalonePrefetchIterator
from verl_speco.standalone_tq_producer import run_producer

store = build_feature_store_from_config(
    {"type": "ray_reference"}, read_only=True,
    transfer_queue_cfg={"run_id": "run"},
)
store.connect()
assert store.owner_ready()
assert type(store).__bases__ == (object,)
assert ReferenceFeatureDataLoader.__bases__ == (object,)
assert ReferenceBatchPrefetchIterator.__bases__ == (StandalonePrefetchIterator,)
store.close()
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _loader(commands, *, rank=0, world_size=1, prepare=None, cleanup=False):
    store = ReferenceFeatureStore({"run_id": "run"})
    return ReferenceFeatureDataLoader(
        store,
        batch_size=1,
        rank=rank,
        world_size=world_size,
        scheduled_commands=commands,
        prepare_batch=prepare
        or (lambda entries: {"input_ids": torch.tensor([[1, 2]])}),
        cleanup_after_load=cleanup,
    )


def _command(plan_id, key="sample"):
    return {
        "kind": "batch",
        "plan_id": plan_id,
        "global_keys": [key],
        "global_sequence_nos": [plan_id],
        "assignments": [[{"key": key, "tag": {"sequence_no": plan_id}}]],
    }


def test_reference_loader_ignores_duplicate_plans_without_repreparing():
    commands = Queue()
    for command in (_command(1), _command(1), _command(2), {"kind": "stop"}):
        commands.put(command)
    reads = []

    def prepare(entries):
        reads.append(entries[0].tag["sequence_no"])
        return {"input_ids": torch.tensor([[reads[-1]]])}

    batches = list(_loader(commands, prepare=prepare))
    assert reads == [1, 2]
    assert [batch.plan_id for batch in batches] == [1, 2]
    assert all(isinstance(batch, StandaloneLocalBatch) for batch in batches)
    assert all(
        batch.local_samples == [] and batch.tq_get_seconds == 0 for batch in batches
    )


def test_reference_assignments_are_metadata_only_and_rank_local():
    commands = Queue()
    command = _command(1)
    command["assignments"].append([{"key": "other-rank", "tag": {}}])
    commands.put(command)
    commands.put({"kind": "stop"})
    loader = _loader(
        commands, world_size=2, prepare=lambda _: pytest.fail("Unexpected read")
    )
    # Simulate the broadcast without requiring a distributed backend in this unit test.
    loader._broadcast_command = lambda command: command
    batch = next(loader.iter_assignments())
    assert batch.file_entries == [ReadyEntry("sample", {"sequence_no": 1})]
    assert batch.prepared_batch is None
    assert set(loader._cleanup_entries) == {"sample", "other-rank"}


@pytest.mark.parametrize("use_prefetch", [False, True])
def test_reference_read_failure_preserves_source_and_cleanup_metadata(use_prefetch):
    commands = Queue()
    commands.put(_command(1))
    calls = []

    def fail(entries):
        raise FileNotFoundError("missing hidden file")

    loader = _loader(commands, prepare=fail, cleanup=True)
    loader.store.hidden_states_store.release = calls.append
    source = StandalonePrefetchIterator(loader, 1) if use_prefetch else iter(loader)
    try:
        with pytest.raises((FileNotFoundError, RuntimeError)):
            next(source)
        assert calls == []
    finally:
        source.close()


def test_reference_loader_requires_scheduler_and_preparation():
    store = ReferenceFeatureStore({"run_id": "run"})
    with pytest.raises(ValueError, match="scheduler-assigned"):
        ReferenceFeatureDataLoader(store, batch_size=1, rank=0, world_size=1)


@pytest.mark.parametrize("clock_node", ["local", "remote", None])
def test_reference_delivery_timing_uses_only_the_same_ray_node(monkeypatch, clock_node):
    import verl_speco.trainer.reference_sample_source as source

    monkeypatch.setattr(source, "current_ray_node_id", lambda: "local")
    monkeypatch.setattr(source.time, "monotonic", lambda: 100.0)
    command = _command(1)
    command.update(
        clock_node_id=clock_node,
        dispatched_at=90.0,
        delivery_times={"metadata_event_age_time": 0.2},
    )
    commands = Queue()
    commands.put(command)
    commands.put({"kind": "stop"})
    batch = next(iter(_loader(commands)))
    assert batch.delivery_times["metadata_event_age_time"] == 0.2
    if clock_node == "local":
        assert batch.delivery_times["command_delivery_time"] == 10.0
    else:
        assert "command_delivery_time" not in batch.delivery_times
