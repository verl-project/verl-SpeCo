# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import sys
from types import SimpleNamespace

import pytest

from verl_speco.standalone_ray_runtime import (
    StandaloneProducerActor,
    StandaloneRayTrainer,
    _configure_driver_logging,
)
from verl_speco.standalone_tq_training_launcher import (
    PipelineCommands,
    _validate_runtime_backend_topology,
)


def _commands() -> PipelineCommands:
    return PipelineCommands(
        vllm=None,
        vllm_endpoints=("http://127.0.0.1:8000/v1",),
        owner=["python", "-m", "verl_speco.tq_owner"],
        producer=["producer"],
        consumer=["consumer"],
        producer_overrides=("producer=true",),
        consumer_overrides=("consumer=true",),
    )


def test_producer_shared_file_probe_is_owned_and_idempotent(tmp_path):
    from pathlib import Path
    config = {"speco": {"standalone_tq_producer": {"hidden_states_store": {"file_root": str(tmp_path)}}}}
    actor = StandaloneProducerActor(config)
    async def exercise():
        first = await actor.create_shared_file_probe()
        assert await actor.create_shared_file_probe() == first
        path = Path(first["path"])
        assert path.parent == tmp_path and path.read_text(encoding="ascii") == first["token"]
        with pytest.raises(ValueError, match="identity"):
            await actor.remove_shared_file_probe(first["path"], "0" * 32)
        path.write_text("wrong", encoding="ascii")
        with pytest.raises(ValueError, match="changed"):
            await actor.remove_shared_file_probe(**first)
        path.write_text(first["token"], encoding="ascii")
        await actor.remove_shared_file_probe(**first)
        assert not path.exists()
    asyncio.run(exercise())


def test_producer_rejects_reference_outside_shared_root_before_publication(tmp_path):
    from queue import Queue
    root = tmp_path / "shared"
    root.mkdir()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "hidden_states_store": {"file_root": str(root)}}}}, Queue(), reference_batch_size=1)
    async def exercise():
        with pytest.raises(ValueError, match="outside the configured shared file root"):
            await actor._on_references([{"key": "sample", "tag": {
                "backend": "file", "hidden_states_path": str(tmp_path / "outside.safetensors")}}])
        assert actor._events.empty() and not actor._active_references
    asyncio.run(exercise())


def test_reference_delivery_rejects_aliases_and_conflicting_retries_atomically():
    from copy import deepcopy
    from verl_speco.standalone_ray_runtime import _accept_reference_entries, _release_reference_entries
    from verl_speco.transport.drafter_sample_protocol import (
        PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key,
    )

    def entry(index, handle):
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(
            meta, backend="mooncake", handle=handle,
            request={"input_ids": [1, 2], "loss_mask": [0, 1], "position_ids": [0, 1],
                     "feature_positions": [0, 1], "draft_position_ids": [1, 2],
                     "vllm_prompt_token_ids": [1, 2], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"},
        )
        return {"key": make_sample_key(meta), "tag": tag}

    keys, refs = {}, {}
    first, second = entry(0, "a"), entry(1, "b")
    assert len(_accept_reference_entries([first], run_id="run", seen_keys=keys, seen_references=refs)) == 1
    assert not _accept_reference_entries([deepcopy(first)], run_id="run", seen_keys=keys, seen_references=refs)
    altered = deepcopy(first)
    altered["tag"]["request"]["source_metadata"]["changed"] = True
    with pytest.raises(ValueError, match="conflicting"):
        _accept_reference_entries([second, altered], run_id="run", seen_keys=keys, seen_references=refs)
    assert second["key"] not in keys and ("mooncake", "b") not in refs
    with pytest.raises(ValueError, match="multiple samples"):
        _accept_reference_entries([entry(2, "a")], run_id="run", seen_keys=keys, seen_references=refs)
    _release_reference_entries([SimpleNamespace(key=first["key"], tag=first["tag"])], refs)
    assert len(_accept_reference_entries([entry(2, "a")], run_id="run", seen_keys=keys, seen_references=refs)) == 1


def test_reference_events_retry_same_batch_id_and_use_full_batch():
    class Queue:
        def __init__(self):
            self.items = []

        def put(self, event):
            self.items.append(event)
            if len(self.items) == 1:
                raise ConnectionError("Delivery succeeded but acknowledgement was lost")

    queue = Queue()
    actor = StandaloneProducerActor({}, queue, reference_batch_size=64)
    assert actor._notification_batch_size == 64

    async def run():
        first = await actor._on_references([{"key": "first", "tag": {}}])
        second = await actor._on_references([{"key": "second", "tag": {}}])
        assert first["attempts"] == 2 and second["attempts"] == 1
        assert first["retry_wait_seconds"] > 0 and second["retry_wait_seconds"] == 0

    asyncio.run(run())
    assert [event["batch_id"] for event in queue.items] == [0, 0, 1]
    assert queue.items[0] == queue.items[1]


def test_concurrent_reference_publications_have_ordered_unique_batch_ids():
    from threading import Event

    entered, release = Event(), Event()
    class Queue:
        def __init__(self): self.items = []
        def put(self, event):
            self.items.append(event)
            if len(self.items) == 1:
                entered.set()
                assert release.wait(5)
    queue = Queue()
    actor = StandaloneProducerActor({}, queue, reference_batch_size=1)
    async def run():
        first = asyncio.create_task(actor._on_references([{"key": "a", "tag": {}}]))
        assert await asyncio.to_thread(entered.wait, 5)
        second = asyncio.create_task(actor._on_references([{"key": "b", "tag": {}}]))
        await asyncio.sleep(0)
        assert len(queue.items) == 1 and not second.done()
        release.set()
        await asyncio.gather(first, second)
    try:
        asyncio.run(run())
    finally:
        release.set()
    assert [item["batch_id"] for item in queue.items] == [0, 1]
    assert [item["entries"][0]["key"] for item in queue.items] == ["a", "b"]


def test_uncertain_reference_delivery_fences_later_publication(monkeypatch):
    class Queue:
        def __init__(self): self.items = []
        def put(self, event):
            self.items.append(event)
            raise ConnectionError("delivery succeeded but reply was lost")
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, queue, reference_batch_size=1)
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.retry_io", lambda func, **kwargs: func())
    async def run():
        with pytest.raises(ConnectionError, match="reply was lost"):
            await actor._on_references([{"key": "a", "tag": {}}])
        assert actor._active_references == {"a"} and actor._reference_aborted
        with pytest.raises(RuntimeError, match="publication stopped"):
            await actor._on_references([{"key": "b", "tag": {}}])
    asyncio.run(run())
    assert len(queue.items) == 1 and queue.items[0]["batch_id"] == 0


def test_cancelled_inflight_reference_delivery_fences_later_publication():
    from threading import Event

    entered, release, finished = Event(), Event(), Event()
    class Queue:
        def put(self, event):
            entered.set()
            assert release.wait(5)
            finished.set()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, Queue(), reference_batch_size=1)
    async def run():
        publication = asyncio.create_task(actor._on_references([{"key": "a", "tag": {}}]))
        assert await asyncio.to_thread(entered.wait, 5)
        publication.cancel()
        with pytest.raises(asyncio.CancelledError):
            await publication
        assert actor._reference_aborted and actor._active_references == {"a"}
        with pytest.raises(RuntimeError, match="publication stopped"):
            await actor._on_references([{"key": "b", "tag": {}}])
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    try:
        asyncio.run(run())
    finally:
        release.set()


def test_reference_publication_credits_bound_and_release_by_plan():
    from queue import Queue
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 2}}}, queue, reference_batch_size=2)
    entries = [{"key": str(i), "tag": {}} for i in range(4)]
    async def run():
        publication = asyncio.create_task(actor._on_references(entries))
        for _ in range(100):
            if queue.qsize() == 1:
                break
            await asyncio.sleep(0.01)
        first = queue.get_nowait()
        assert [item["key"] for item in first["entries"]] == ["0", "1"]
        assert actor._active_references == {"0", "1"} and not publication.done()
        with pytest.raises(ValueError, match="active references"):
            await actor.release_references(actor._incarnation, 0, "token", ["0", "missing"])
        assert actor._active_references == {"0", "1"}
        with pytest.raises(ValueError, match="registered dispatch"):
            await actor.release_references(actor._incarnation, 0, "token", ["0", "1"])
        with pytest.raises(ValueError, match="active references"):
            await actor.register_dispatch(actor._incarnation, 0, "token", ["0", "missing"])
        await actor.register_dispatch(actor._incarnation, 0, "token", ["0", "1"])
        await actor.register_dispatch(actor._incarnation, 0, "token", ["0", "1"])
        with pytest.raises(ValueError, match="Conflicting"):
            await actor.register_dispatch(actor._incarnation, 0, "wrong", ["0", "1"])
        with pytest.raises(ValueError, match="registered dispatch"):
            await actor.release_references(actor._incarnation, 0, "wrong", ["0", "1"])
        await actor.release_references(actor._incarnation, 0, "token", ["0", "1"])
        await actor.release_references(actor._incarnation, 0, "token", ["0", "1"])
        with pytest.raises(ValueError, match="Conflicting"):
            await actor.release_references(actor._incarnation, 0, "wrong", ["0", "1"])
        await asyncio.wait_for(publication, 2)
        second = queue.get_nowait()
        assert [item["key"] for item in second["entries"]] == ["2", "3"]
        assert second["batch_id"] == first["batch_id"] + 1
        assert actor._active_references == {"2", "3"}
        with pytest.raises(ValueError, match="stale producer incarnation"):
            await actor.release_references("0" * 32, 1, "next", ["2", "3"])
        with pytest.raises(ValueError, match="out-of-order"):
            await actor.release_references(actor._incarnation, 2, "future", ["2", "3"])
        actor._released_reference_plans.clear()  # Simulate eviction of old receipts.
        with pytest.raises(ValueError, match="Expired"):
            await actor.release_references(actor._incarnation, 0, "token", ["2", "3"])
        assert actor._active_references == {"2", "3"}
        await actor.register_dispatch(actor._incarnation, 1, "next", ["2", "3"])
        await actor.release_references(actor._incarnation, 1, "next", ["2", "3"])
    asyncio.run(run())


def test_parent_commit_publication_waits_for_all_rank_receipts():
    from queue import Queue

    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "publication_after_parent_commit": True, "max_published_references": 10}}},
        queue, reference_batch_size=2)
    assert actor._reference_limit == 2 and actor._credits_enabled

    async def run():
        publication = asyncio.create_task(actor._on_references(
            [{"key": str(i), "tag": {}} for i in range(4)]))
        for _ in range(100):
            if not queue.empty():
                break
            await asyncio.sleep(0.01)
        assert [item["key"] for item in queue.get_nowait()["entries"]] == ["0", "1"]
        await actor.register_dispatch(actor._incarnation, 0, "token", ["0", "1"], 2)
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 0)
        with pytest.raises(ValueError, match="every rank"):
            await actor.release_references(actor._incarnation, 0, "token", ["0", "1"])
        assert queue.empty() and not publication.done()
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 1)
        await actor.release_references(actor._incarnation, 0, "token", ["0", "1"])
        await asyncio.wait_for(publication, 2)
        assert [item["key"] for item in queue.get_nowait()["entries"]] == ["2", "3"]
    asyncio.run(run())


def test_reference_abort_stops_waiters_and_retains_active_references():
    from queue import Queue

    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1, "max_reference_execution_slots": 1}}},
        queue, reference_batch_size=1)

    async def run():
        await actor._acquire_reference_slot(SimpleNamespace(sequence_no=0))
        await actor._on_references([{"key": "first", "tag": {"sequence_no": 0}}])
        await actor.register_dispatch(actor._incarnation, 0, "token", ["first"])
        slot = asyncio.create_task(actor._acquire_reference_slot(SimpleNamespace(sequence_no=1)))
        publication = asyncio.create_task(actor._on_references([{"key": "second", "tag": {"sequence_no": 1}}]))
        await actor.pause()
        paused = asyncio.create_task(actor._before_request())
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="stale producer incarnation"):
            await actor.abort_references("0" * 32)
        assert not slot.done() and not publication.done() and not paused.done()
        await actor.abort_references(actor._incarnation)
        await actor.abort_references(actor._incarnation)  # Safe retry.
        for task in (slot, publication, paused):
            with pytest.raises(RuntimeError, match="stopped after pipeline failure"):
                await asyncio.wait_for(task, 2)
        with pytest.raises(RuntimeError, match="release stopped"):
            await actor.release_references(actor._incarnation, 0, "token", ["first"])
        with pytest.raises(RuntimeError, match="aborted Producer"):
            await actor.start()
        assert actor._active_references == {"first"}
        assert actor._registered_reference_plans[0] == ("token", ("first",))
        assert [entry["key"] for entry in queue.get_nowait()["entries"]] == ["first"]
        assert queue.empty()

        other = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_reference_execution_slots": 1}}}, Queue(), reference_batch_size=1)
        first, second = SimpleNamespace(sequence_no=0), SimpleNamespace(sequence_no=1)
        await other._acquire_reference_slot(first)
        await other.abort_references(other._incarnation)
        with pytest.raises(RuntimeError, match="admission stopped"):
            await other._transfer_unpublished_slot(first, second)
        await other._release_unpublished_slot(first)  # Local cleanup remains possible.
        assert not other._execution_pending

    asyncio.run(run())


@pytest.mark.parametrize("setting,batch,limit", [("true", 2, 0), (True, None, 0), (True, 2, 1)])
def test_parent_commit_publication_rejects_invalid_settings(setting, batch, limit):
    config = {"speco": {"standalone_tq_producer": {
        "publication_after_parent_commit": setting, "max_published_references": limit}}}
    with pytest.raises(ValueError):
        StandaloneProducerActor(config, reference_batch_size=batch)


def test_parent_commit_publication_rejects_transfer_queue_mode(monkeypatch):
    from copy import deepcopy

    config = deepcopy(_runtime_config())
    config.speco.standalone_tq_producer.publication_after_parent_commit = True
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=_FakeRay(_ProducerHandle()),
                                   worker_group_factory=lambda *a, **kw: _WorkerGroup())
    with pytest.raises(ValueError, match="requires ray_reference"):
        trainer.run()


def test_producer_discards_only_closed_unassigned_references():
    from queue import Queue
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 2, "max_reference_execution_slots": 2}}}, Queue(), reference_batch_size=2)
    async def run():
        await actor._acquire_reference_slot(SimpleNamespace(sequence_no=0))
        await actor._acquire_reference_slot(SimpleNamespace(sequence_no=1))
        await actor._on_references([{"key": "a", "tag": {"sequence_no": 0}},
                                    {"key": "b", "tag": {"sequence_no": 1}}])
        await actor.register_dispatch(actor._incarnation, 0, "receipt", ["a"], 1)
        with pytest.raises(RuntimeError, match="closed"):
            await actor.discard_unassigned_references(actor._incarnation, "discard", ["b"])
        await actor.stop()
        with pytest.raises(ValueError, match="unassigned"):
            await actor.discard_unassigned_references(actor._incarnation, "discard", ["a"])
        with pytest.raises(ValueError, match="stale"):
            await actor.discard_unassigned_references("0" * 32, "discard", ["b"])
        await actor.discard_unassigned_references(actor._incarnation, "discard", ["b"])
        await actor.discard_unassigned_references(actor._incarnation, "discard", ["b"])
        with pytest.raises(ValueError, match="Conflicting"):
            await actor.discard_unassigned_references(actor._incarnation, "discard", ["a"])
        assert actor._active_references == {"a"} and actor._active_reference_sequences == {"a": 0}
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
        await actor.release_references(actor._incarnation, 0, "receipt", ["a"])
        assert not actor._active_references and not actor._active_reference_sequences
    asyncio.run(run())


def test_real_ray_unassigned_discard_and_blocked_publication():
    import os
    import subprocess
    if os.environ.get("SPECO_TEST_REAL_RAY") != "1":
        pytest.skip("Set SPECO_TEST_REAL_RAY=1 to start a local Ray cluster")
    code = '''
import ray
import json
from ray.util.queue import Queue
from verl_speco.standalone_ray_runtime import StandaloneProducerActor
from verl_speco.transport.native_feature_transport import NativeSourceDirectory, NativeSourceDirectoryRouter, NativeProducerReferenceStore
from verl_speco.transport.drafter_sample_protocol import SampleMetadata, PROTOCOL_SCHEMA_VERSION, make_reference_tag, make_sample_key
@ray.remote(max_concurrency=4)
class Actor(StandaloneProducerActor):
    async def publish(self, key):
        await self._on_references([{"key": key, "tag": {}}])
        return self._incarnation
    async def publish_native(self, key, tag):
        await self._on_references([{"key": key, "tag": tag}])
        return self._incarnation
    async def active(self): return sorted(self._active_references)
@ray.remote(num_cpus=0)
class Source:
    def __init__(self, directory_name):
        import ast, os, threading, torch, uuid
        from contextlib import nullcontext
        from pathlib import Path
        from types import SimpleNamespace
        from verl_speco.transport import native_feature_transport as native
        class Engine:
            def initialize(self, *args): return 0
            def register_memory(self, *args): return 0
            def unregister_memory(self, *args): return 0
        original = native.NativeFeatureTransport
        native.NativeFeatureTransport = lambda **kwargs: original(**kwargs, engine_factory=Engine)
        for name, value in {"SPECO_NATIVE_DIRECTORY": directory_name, "SPECO_NATIVE_RAY_NAMESPACE": "native-test",
            "SPECO_NATIVE_RUN_ID": "run", "SPECO_NATIVE_GENERATION": "1", "SPECO_NATIVE_SOURCE_ENDPOINT": "source",
            "SPECO_NATIVE_METADATA_SERVER": "metadata", "SPECO_NATIVE_SOURCE_MAX_BYTES": "4194304",
            "SPECO_NATIVE_SOURCE_MAX_SLOTS": "1", "SPECO_NATIVE_CHUNK_BYTES": "2097152"}.items():
            os.environ[name] = value
        path = Path("verl_speco/mooncake_hidden_states_connector.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SpecoNativeHiddenStatesConnector")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
        upstream = SimpleNamespace(extract_from_kv_cache=lambda cache, slots, count: cache[:count], assert_finite=lambda *args: None)
        namespace = {"SpecoMooncakeHiddenStatesConnector": object, "torch": torch, "threading": threading,
                     "uuid": uuid, "os": os, "_upstream_hs": upstream,
                     "get_torch_device": lambda: SimpleNamespace(stream=lambda _: nullcontext())}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        self.connector = namespace["SpecoNativeHiddenStatesConnector"]()
        self.connector._kv_cache = torch.ones(2, 1, 2)
        self.connector._copy_stream = SimpleNamespace(wait_event=lambda _: None, synchronize=lambda: None)
        self.connector._block_size = 2
        self._publish("sample")
    def _publish(self, handle):
        import torch
        from types import SimpleNamespace
        self.connector._publish_sample(SimpleNamespace(block_ids=[0], token_ids=torch.tensor([1, 2]), mooncake_key=handle), None)
    def descriptor_value(self): return self.connector._native_source._slots["sample"]["descriptor"]
    def publish_extra(self): self._publish("filtered")
    def active(self): return len(self.connector._native_source._slots)
    def close(self):
        self.connector._native_service.close()
        self.connector._native_source.close()
        for name in ("request_queue", "response_queue"):
            self.connector._native_control_options[name].shutdown(force=True)
ray.init(num_cpus=2, include_dashboard=False, _node_ip_address="127.0.0.1", log_to_driver=False)
queue = Queue(maxsize=10)
try:
    actor = Actor.remote({"speco": {"standalone_tq_producer": {"max_published_references": 2}}}, queue, 1)
    incarnation = ray.get(actor.publish.remote("a"), timeout=20)
    ray.get(actor.publish.remote("b"), timeout=20)
    ray.get(actor.register_dispatch.remote(incarnation, 0, "receipt", ["a"], 1), timeout=10)
    blocked = actor.publish.remote("c")
    assert not ray.wait([blocked], timeout=0.5)[0]
    ray.get(actor.stop.remote(), timeout=10)
    for ref, message in ((blocked, "closure"),
                         (actor.discard_unassigned_references.remote("0" * 32, "token", ["b"]), "stale"),
                         (actor.discard_unassigned_references.remote(incarnation, "token", ["a"]), "unassigned")):
        try: ray.get(ref, timeout=10)
        except ray.exceptions.RayTaskError as error: assert message in str(error)
        else: raise AssertionError("invalid operation succeeded")
    for _ in range(2):
        ray.get(actor.discard_unassigned_references.remote(incarnation, "token", ["b"]), timeout=10)
    ray.get(actor.acknowledge_reference_rank.remote(incarnation, 0, "receipt", 0), timeout=10)
    ray.get(actor.release_references.remote(incarnation, 0, "receipt", ["a"]), timeout=10)
    assert ray.get(actor.active.remote(), timeout=10) == []
    directory_name = "native-test-" + __import__("uuid").uuid4().hex
    directory = ray.remote(max_concurrency=16)(NativeSourceDirectory).options(name=directory_name, namespace="native-test").remote(
        run_id="run", generation=1, max_sources=1, max_records=10,
        source_requirements={"max_bytes": 4194304, "max_slots": 1, "chunk_bytes": 2097152,
                             "metadata_server": "metadata", "reader_endpoints": ["reader"]})
    source = Source.remote(directory_name)
    descriptor = ray.get(source.descriptor_value.remote(), timeout=20)
    assert ray.get(directory.status.remote(1), timeout=10)["active_sample"][0]["handle"] == "sample"
    handle = NativeProducerReferenceStore(NativeSourceDirectoryRouter(directory, timeout=10)).resolve_reference("sample")
    assert json.loads(handle) == descriptor
    meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", "sample", 0)
    tag = make_reference_tag(meta, backend="native", handle=handle,
        request={"input_ids": [1], "loss_mask": [1], "position_ids": [0], "feature_positions": [0],
                 "draft_position_ids": [0], "vllm_prompt_token_ids": [1], "source_metadata": {}},
        feature_contract={"algorithm": "DSPARK"})
    key = make_sample_key(meta)
    remote_actor = Actor.remote({"speco": {"standalone_tq_producer": {"max_published_references": 1}}},
                                queue, 1, {"directory": directory, "timeout": 10.0})
    incarnation = ray.get(remote_actor.publish_native.remote(key, tag), timeout=20)
    ray.get(remote_actor.register_dispatch.remote(incarnation, 0, "native-receipt", [key], 1), timeout=20)
    ray.get(remote_actor.acknowledge_reference_rank.remote(incarnation, 0, "native-receipt", 0), timeout=20)
    assert ray.get(source.active.remote(), timeout=10) == 1
    ray.get(remote_actor.release_references.remote(incarnation, 0, "native-receipt", [key]), timeout=20)
    assert ray.get(source.active.remote(), timeout=10) == 0
    assert ray.get(directory.status.remote(), timeout=10)["active"] == 0
    try: ray.get(directory.resolve.remote("sample", 1), timeout=5)
    except ray.exceptions.RayTaskError as error: assert "retired" in str(error)
    else: raise AssertionError("retired handle remained available")
    ray.get(source.publish_extra.remote(), timeout=20)
    import asyncio
    from types import SimpleNamespace
    import verl_speco.producer.vllm_feature_client as client
    store = NativeProducerReferenceStore(NativeSourceDirectoryRouter(directory, timeout=10))
    endpoint = client.VllmEndpoint("http://vllm:8000/v1", 1)
    async def prefill(*args, **kwargs):
        return client.VllmResponse(None, endpoint.base_url, handle="filtered", transfer_params={"handle": "filtered"})
    client.request_prefill = prefill
    pool = client.VllmFeatureClientPool([endpoint], model="target", max_inflight_requests=1,
                                        request_timeout=10, hidden_states_store=store)
    pool._states = [client._EndpointState(endpoint, object(), asyncio.Semaphore(1))]
    response = asyncio.run(pool.prefill_reference(SimpleNamespace(prompt_token_ids=[1])))
    assert json.loads(response.reference)["lease_id"] == "filtered"
    client.delete_temporary_result(response)
    assert ray.get(source.active.remote(), timeout=10) == 0
    ray.get(remote_actor.stop.remote(), timeout=10)
    ray.get(remote_actor.seal_native_sources.remote(incarnation), timeout=20)
    try: ray.get(source.publish_extra.remote(), timeout=10)
    except ray.exceptions.RayTaskError as error: assert "sealed" in str(error)
    else: raise AssertionError("sealed source accepted a publication")
    ray.get(directory.close.remote(), timeout=10)
    ray.get(source.close.remote(), timeout=20)
    from verl_speco import standalone_ray_runtime as runtime
    from verl_speco.standalone_tq_training_launcher import build_pipeline_commands, resolve_pipeline_config
    launch_name = "native-launch-" + __import__("uuid").uuid4().hex
    launch_env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_VLLM_ENDPOINT": "http://10.0.0.1:8000/v1",
                  "SPECO_NATIVE_DIRECTORY": launch_name, "SPECO_NATIVE_RAY_NAMESPACE": "native-test",
                  "SPECO_NATIVE_RUN_ID": "run", "SPECO_NATIVE_GENERATION": "1",
                  "SPECO_NATIVE_METADATA_SERVER": "metadata", "SPECO_NATIVE_SOURCE_MAX_BYTES": "4194304",
                  "SPECO_NATIVE_SOURCE_MAX_SLOTS": "1", "SPECO_NATIVE_MAX_RECORDS": "10",
                  "SPECO_NATIVE_READER_ENDPOINTS": '["reader"]'}
    args = ["data.train_files=/data/train.jsonl", "actor_rollout_ref.model.path=/model",
            "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=1",
            "actor_rollout_ref.rollout.drafter.training.max_steps=1"]
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ=launch_env), args,
                                       ray_address="unused", env=launch_env)
    def run_actors(_commands, **kwargs):
        directory = kwargs["native_transport"]["directory"]
        source = Source.remote(launch_name)
        assert ray.get(source.active.remote(), timeout=20) == 1
        router = NativeSourceDirectoryRouter(directory, timeout=10)
        NativeProducerReferenceStore(router).release("sample")
        router.freeze_and_seal()
        ray.get(source.close.remote(), timeout=20)
        return 0
    runtime.run_ray_actors = run_actors
    assert runtime.run_ray_pipeline(commands, ray_module=ray, ray_address="unused", environ=launch_env,
                                    endpoint_ready=lambda _: True) == 0
    import time
    deadline = time.monotonic() + 10
    while True:
        try: ray.get_actor(launch_name, namespace="native-test")
        except ValueError: break
        if time.monotonic() >= deadline: raise AssertionError("drained native directory was not removed")
        time.sleep(0.1)
finally:
    queue.shutdown(force=True)
    ray.shutdown()
'''
    env = {**os.environ, "RAY_AUTH_MODE": "disabled"}
    env.pop("RAY_TMPDIR", None)
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_producer_close_wakes_blocked_admission_without_releasing_credits():
    from queue import Queue
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1, "max_reference_execution_slots": 1}}}, Queue(), reference_batch_size=1)
    async def run():
        await actor._acquire_reference_slot(SimpleNamespace(sequence_no=0))
        await actor._on_references([{"key": "a", "tag": {"sequence_no": 0}}])
        await actor.pause()
        pending = [asyncio.create_task(actor._before_request()),
                   asyncio.create_task(actor._acquire_reference_slot(SimpleNamespace(sequence_no=1))),
                   asyncio.create_task(actor._on_references([{"key": "b", "tag": {"sequence_no": 1}}]))]
        await asyncio.sleep(0)
        assert not any(task.done() for task in pending)
        await actor.stop()
        outcomes = await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2)
        assert all(isinstance(value, RuntimeError) and "closure" in str(value) for value in outcomes)
        assert actor._active_references == {"a"} and not actor._reference_aborted
    asyncio.run(run())


def test_unassigned_discard_returns_file_tensor_byte_credits(tmp_path):
    from queue import Queue
    import torch
    from safetensors.torch import save_file
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 80, "max_single_file_tensor_bytes": 40}}},
        Queue(), reference_batch_size=2)
    entries = []
    for index in range(2):
        path = tmp_path / f"{index}.safetensors"
        save_file({"hidden": torch.zeros(10)}, str(path))
        entries.append({"key": str(index), "tag": {"backend": "file", "hidden_states_path": str(path)}})
    async def run():
        await actor._on_references(entries)
        assert actor._published_file_bytes == 80
        await actor.stop()
        await actor.discard_unassigned_references(actor._incarnation, "first", ["0"])
        assert actor._published_file_bytes == 40 and actor._active_references == {"1"}
        await actor.discard_unassigned_references(actor._incarnation, "second", ["1"])
        assert actor._published_file_bytes == 0 and not actor._active_file_bytes
    asyncio.run(run())


def test_producer_requires_distinct_rank_acks_before_returning_credits():
    from queue import Queue
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 2}}}, Queue(), reference_batch_size=2)
    async def run():
        await actor._on_references([{"key": "a", "tag": {}}, {"key": "b", "tag": {}}])
        await actor.register_dispatch(actor._incarnation, 0, "token", ["a", "b"], 2)
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 0)
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 0)
        with pytest.raises(ValueError, match="every rank"):
            await actor.release_references(actor._incarnation, 0, "token", ["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            await actor.acknowledge_reference_rank(actor._incarnation, 0, "wrong", 1)
        with pytest.raises(ValueError, match="does not match"):
            await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 2)
        with pytest.raises(ValueError, match="stale producer incarnation"):
            await actor.acknowledge_reference_rank("0" * 32, 0, "token", 1)
        assert actor._active_references == {"a", "b"}
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 1)
        await actor.release_references(actor._incarnation, 0, "token", ["a", "b"])
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "token", 1)
        await actor.register_dispatch(actor._incarnation, 0, "token", ["a", "b"], 2)
        with pytest.raises(ValueError, match="Conflicting"):
            await actor.register_dispatch(actor._incarnation, 0, "token", ["a", "b"], 1)
        assert not actor._active_references
    asyncio.run(run())


def test_reference_dispatch_reserves_keys_and_requires_ordered_plans():
    from queue import Queue
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 4}}}, Queue(), reference_batch_size=2)

    async def run():
        await actor._on_references([{"key": str(i), "tag": {}} for i in range(4)])
        with pytest.raises(ValueError, match="out-of-order"):
            await actor.register_dispatch(actor._incarnation, 1, "later", ["2", "3"])
        await actor.register_dispatch(actor._incarnation, 0, "first", ["0", "1"])
        with pytest.raises(ValueError, match="unassigned"):
            await actor.register_dispatch(actor._incarnation, 1, "later", ["1", "2"])
        await actor.register_dispatch(actor._incarnation, 1, "later", ["2", "3"])
        with pytest.raises(ValueError, match="out-of-order"):
            await actor.release_references(actor._incarnation, 1, "later", ["2", "3"])
        assert actor._active_references == {"0", "1", "2", "3"}
        await actor.release_references(actor._incarnation, 0, "first", ["0", "1"])
        await actor.register_dispatch(actor._incarnation, 0, "first", ["0", "1"])
        await actor.release_references(actor._incarnation, 1, "later", ["2", "3"])
        assert not actor._registered_reference_plans and not actor._assigned_references

    asyncio.run(run())


def test_reference_publication_uses_remaining_credit_for_partial_batch():
    from queue import Queue
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 2}}}, queue, reference_batch_size=2)
    async def run():
        await actor._on_references([{"key": "a", "tag": {}}])
        assert queue.get_nowait()["entries"][0]["key"] == "a"
        pending = asyncio.create_task(actor._on_references([
            {"key": "b", "tag": {}}, {"key": "c", "tag": {}}]))
        for _ in range(100):
            if not queue.empty():
                break
            await asyncio.sleep(0.01)
        assert [entry["key"] for entry in queue.get_nowait()["entries"]] == ["b"]
        assert not pending.done()
        await actor.register_dispatch(actor._incarnation, 0, "first", ["a", "b"])
        await actor.release_references(actor._incarnation, 0, "first", ["a", "b"])
        await asyncio.wait_for(pending, 2)
        assert [entry["key"] for entry in queue.get_nowait()["entries"]] == ["c"]
    asyncio.run(run())


def test_file_reference_tensor_byte_budget_blocks_and_releases(tmp_path):
    from queue import Queue
    import torch
    from safetensors.torch import save_file
    from verl_speco.producer.hidden_states_store import file_reference_tensor_bytes
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16}}},
        queue, reference_batch_size=2)
    entries = []
    for index in range(4):
        path = tmp_path / f"{index}.safetensors"
        save_file({"hidden_states": torch.arange(4, dtype=torch.float32)}, str(path))
        assert path.stat().st_size > file_reference_tensor_bytes(str(path)) == 16
        entries.append({"key": str(index), "tag": {"backend": "file", "hidden_states_path": str(path)}})

    async def run():
        task = asyncio.create_task(actor._on_references(entries))
        for _ in range(100):
            if not queue.empty():
                break
            await asyncio.sleep(0.01)
        assert [entry["key"] for entry in queue.get_nowait()["entries"]] == ["0", "1"]
        assert actor._published_file_bytes == 32 and not task.done()
        await actor.register_dispatch(actor._incarnation, 0, "first", ["0", "1"])
        await actor.release_references(actor._incarnation, 0, "first", ["0", "1"])
        await actor.release_references(actor._incarnation, 0, "first", ["0", "1"])
        await asyncio.wait_for(task, 2)
        assert [entry["key"] for entry in queue.get_nowait()["entries"]] == ["2", "3"]
        assert actor._published_file_bytes == 32
        await actor.register_dispatch(actor._incarnation, 1, "second", ["2", "3"])
        await actor.release_references(actor._incarnation, 1, "second", ["2", "3"])
        assert actor._published_file_bytes == 0 and not actor._active_file_bytes
    asyncio.run(run())


def test_file_reference_tensor_byte_budget_fits_variable_size_partial_batches(tmp_path):
    from queue import Queue
    import torch
    from safetensors.torch import save_file
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16}}},
        queue, reference_batch_size=2)
    entries = []
    for index, size in enumerate((1, 3, 4, 1)):
        path = tmp_path / f"{index}.safetensors"
        save_file({"x": torch.arange(size, dtype=torch.float32)}, str(path))
        entries.append({"key": str(index), "tag": {"backend": "file", "hidden_states_path": str(path)}})

    async def run():
        task = asyncio.create_task(actor._on_references(entries))
        for _ in range(100):
            if not queue.empty():
                break
            await asyncio.sleep(0.01)
        assert [item["key"] for item in queue.get_nowait()["entries"]] == ["0", "1", "2"]
        assert actor._published_file_bytes == 32 and not task.done()
        await actor.register_dispatch(actor._incarnation, 0, "first", ["0", "1"])
        await actor.release_references(actor._incarnation, 0, "first", ["0", "1"])
        await asyncio.wait_for(task, 2)
        assert [item["key"] for item in queue.get_nowait()["entries"]] == ["3"]
        assert actor._published_file_bytes == 20
        await actor.register_dispatch(actor._incarnation, 1, "second", ["2", "3"])
        await actor.release_references(actor._incarnation, 1, "second", ["2", "3"])
        assert actor._published_file_bytes == 0
    asyncio.run(run())


def test_reference_capacity_rejects_non_reference_runtime():
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16,
        "max_reserved_file_tensor_bytes": 32}}}, reference_batch_size=2)
    with pytest.raises(ValueError, match="require ray_reference"):
        asyncio.run(actor._run_producer())


def test_file_reservation_returns_unused_bytes_after_publication(tmp_path):
    from queue import Queue
    import torch
    from safetensors.torch import save_file
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16,
        "max_reserved_file_tensor_bytes": 32}}}, Queue(), reference_batch_size=2)
    paths = []
    for index in range(2):
        path = tmp_path / f"small-{index}.safetensors"
        save_file({"x": torch.tensor([index], dtype=torch.float32)}, str(path))
        paths.append(path)
    async def run():
        await actor._acquire_reference_slot(SimpleNamespace(sequence_no=0))
        await actor._acquire_reference_slot(SimpleNamespace(sequence_no=1))
        waiting = asyncio.create_task(actor._acquire_reference_slot(SimpleNamespace(sequence_no=2)))
        await asyncio.sleep(0)
        assert not waiting.done()
        entries = [{"key": str(i), "tag": {"sequence_no": i, "backend": "file",
                   "hidden_states_path": str(path)}} for i, path in enumerate(paths)]
        await actor._on_references(entries)
        await asyncio.wait_for(waiting, 2)
        assert actor._published_file_bytes == 8 and actor._execution_pending == {2}
        await actor._release_unpublished_slot(SimpleNamespace(sequence_no=2))
        await actor.register_dispatch(actor._incarnation, 0, "first", ["0", "1"])
        await actor.release_references(actor._incarnation, 0, "first", ["0", "1"])
        assert not actor._execution_pending and actor._published_file_bytes == 0
    asyncio.run(run())


def test_reference_credit_metrics_follow_publication_and_release(tmp_path):
    from queue import Queue
    import torch
    from safetensors.torch import save_file

    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16,
        "max_reserved_file_tensor_bytes": 32}}}, Queue(), reference_batch_size=2)
    paths = [tmp_path / f"metric-{i}.safetensors" for i in range(2)]
    for path in paths:
        save_file({"x": torch.tensor([1], dtype=torch.float32)}, str(path))

    async def run():
        for i in range(2):
            await actor._acquire_reference_slot(SimpleNamespace(sequence_no=i))
        await actor._on_metrics(0, {})
        pending = actor._metrics_pending[-1]["metrics"]
        assert pending["producer/reference_execution_pending"] == 2
        assert pending["producer/reference_file_tensor_bytes"] == 0
        assert pending["producer/reference_reserved_file_tensor_bytes"] == 32
        entries = [{"key": str(i), "tag": {"sequence_no": i, "backend": "file",
                   "hidden_states_path": str(path)}} for i, path in enumerate(paths)]
        await actor._on_references(entries)
        await actor.register_dispatch(actor._incarnation, 0, "token", ["0", "1"])
        original = {"producer/samples_per_second": 3.0}
        await actor._on_metrics(2, original)
        before = actor._metrics_pending[-1]["metrics"]
        assert original == {"producer/samples_per_second": 3.0}
        assert before == {"producer/samples_per_second": 3.0,
                          "producer/reference_active": 2.0, "producer/reference_assigned": 2.0,
                          "producer/reference_plans": 1.0, "producer/reference_execution_pending": 0.0,
                          "producer/reference_execution_waiters": 0.0, "producer/reference_file_tensor_bytes": 8.0,
                          "producer/reference_reserved_file_tensor_bytes": 8.0}
        await actor.release_references(actor._incarnation, 0, "token", ["0", "1"])
        await actor._on_metrics(2, {})
        actor_snapshot = actor._metrics_pending[-1]["metrics"]
        assert not any(actor_snapshot.values())
        assert before["producer/reference_active"] == 2.0
        actor._notification_finishing = True
        await actor._notify_published_totals()
        events = []
        while not actor._events.empty():
            events.append(actor._events.get_nowait())
        metrics = [event["metrics"] for event in events if event["kind"] == "producer_metrics"]
        assert metrics == [pending, before, actor_snapshot]

    asyncio.run(run())


def test_file_reference_tensor_byte_budget_rejects_unsafe_input(tmp_path):
    from queue import Queue
    import torch
    from safetensors.torch import save_file
    with pytest.raises(ValueError, match="global batch"):
        StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_published_file_tensor_bytes": 31, "max_single_file_tensor_bytes": 16}}}, reference_batch_size=2)
    with pytest.raises(ValueError, match="requires enabled"):
        StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_reserved_file_tensor_bytes": 32}}}, reference_batch_size=2)
    with pytest.raises(ValueError, match="global batch"):
        StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16,
            "max_reserved_file_tensor_bytes": 31}}}, reference_batch_size=2)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_file_tensor_bytes": 32, "max_single_file_tensor_bytes": 16}}}, Queue(), reference_batch_size=2)
    path = tmp_path / "large.safetensors"
    save_file({"x": torch.arange(5, dtype=torch.float32)}, str(path))
    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"short")
    async def run():
        with pytest.raises(ValueError, match="max_single"):
            await actor._on_references([{"key": "large", "tag": {"backend": "file", "hidden_states_path": str(path)}}])
        with pytest.raises(ValueError, match="requires file"):
            await actor._on_references([{"key": "remote", "tag": {"backend": "mooncake", "handle": "x"}}])
        with pytest.raises(ValueError, match="duplicate file reference"):
            await actor._on_references([
                {"key": "same", "tag": {"backend": "file", "hidden_states_path": str(path)}},
                {"key": "same", "tag": {"backend": "file", "hidden_states_path": str(corrupt)}},
            ])
        with pytest.raises(ValueError, match="Truncated"):
            await actor._on_references([{"key": "bad", "tag": {"backend": "file", "hidden_states_path": str(corrupt)}}])
        assert actor._published_file_bytes == 0 and not actor._active_references
    asyncio.run(run())


def test_reference_incarnation_is_included_in_event_size_budget():
    from queue import Queue
    from verl_speco.standalone_ray_runtime import _bounded_metadata_size
    queue = Queue()
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 2, "reference_event_max_bytes": 1024}}},
        queue, reference_batch_size=1)
    async def run():
        await actor._on_references([{"key": str(i), "tag": {"text": "x" * 500}} for i in range(2)])
        events = [queue.get_nowait(), queue.get_nowait()]
        assert all(event["producer_incarnation"] == actor._incarnation for event in events)
        assert all(_bounded_metadata_size(event, 1024) <= 1024 for event in events)
    asyncio.run(run())


def test_reference_publication_failure_retains_credit(monkeypatch):
    from queue import Queue
    import verl_speco.standalone_ray_runtime as runtime
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, Queue(), reference_batch_size=1)
    def fail(operation, **kwargs):
        raise ConnectionError("ambiguous send")
    monkeypatch.setattr(runtime, "retry_io", fail)
    async def run():
        with pytest.raises(ConnectionError, match="ambiguous send"):
            await actor._on_references([{"key": "a", "tag": {}}])
        assert actor._active_references == {"a"}
        with pytest.raises(RuntimeError, match="aborted Producer actor"):
            await actor.start()
    asyncio.run(run())


def test_producer_restart_changes_incarnation_and_rejects_old_release():
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, reference_batch_size=1)
    async def run():
        async def empty_run():
            return None
        actor._run_producer = empty_run
        await actor.start()
        first = actor._incarnation
        await actor._task
        await actor.start()
        assert actor._incarnation != first
        await actor._task
        actor._active_references.add("same-key")
        with pytest.raises(ValueError, match="stale producer incarnation"):
            await actor.release_references(first, 0, "old-token", ["same-key"])
        assert actor._active_references == {"same-key"}
    asyncio.run(run())


@pytest.mark.parametrize("action", ["register", "release"])
def test_reference_credit_checks_incarnation_after_waiting_for_lock(action):
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, reference_batch_size=1)

    async def run():
        old_incarnation = actor._incarnation
        async with actor._reference_condition:
            method = actor.register_dispatch if action == "register" else actor.release_references
            stale = asyncio.create_task(method(old_incarnation, 0, "token", ["key"]))
            await asyncio.sleep(0)
            actor._incarnation = "f" * 32
            actor._active_references.add("key")
            if action == "release":
                actor._registered_reference_plans[0] = ("token", ("key",))
                actor._assigned_references.add("key")
                actor._last_registered_reference_plan = 0
        with pytest.raises(ValueError, match="stale producer incarnation"):
            await stale
        assert actor._active_references == {"key"}
        assert (0 in actor._registered_reference_plans) == (action == "release")

    asyncio.run(run())


def test_execution_slot_waiters_are_fifo_and_cancel_safe():
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_reference_execution_slots": 1}}}, reference_batch_size=1)
    async def run():
        request = lambda n: SimpleNamespace(sequence_no=n)
        await actor._acquire_reference_slot(request(0))
        first = asyncio.create_task(actor._acquire_reference_slot(request(1)))
        second = asyncio.create_task(actor._acquire_reference_slot(request(2)))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await actor._release_unpublished_slot(request(0))
        await asyncio.wait_for(second, 2)
        assert actor._execution_pending == {2} and not actor._execution_waiters
        await actor._release_unpublished_slot(request(2))
    asyncio.run(run())


def test_execution_slot_limit_must_cover_global_batch():
    with pytest.raises(ValueError, match="global training batch"):
        StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_reference_execution_slots": 1}}}, reference_batch_size=2)


def test_reference_publication_limit_must_cover_global_batch():
    with pytest.raises(ValueError, match="global training batch"):
        StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_published_references": 1}}}, reference_batch_size=2)


@pytest.mark.parametrize("credit_limit", [0, 64])
def test_file_actor_sends_completion_after_all_metadata_and_metrics(monkeypatch, credit_limit):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer

    config = _runtime_config()
    training = config.actor_rollout_ref.rollout.drafter.training
    monkeypatch.setattr(config.speco.standalone_tq_producer, "max_published_references", credit_limit, raising=False)
    monkeypatch.setattr(training, "feature_store", SimpleNamespace(type="ray_reference"), raising=False)

    async def run(config, **kwargs):
        assert kwargs["reference_batch_size"] == 64
        assert kwargs["on_published"] is None
        await kwargs["on_references"]([{"key": "first", "tag": {}}])
        await kwargs["on_references"]([{"key": "tail", "tag": {}}])
        await kwargs["on_metrics"](2, {"perf/test": 1.0})
        return producer.ProducerStats(published_count=2)

    monkeypatch.setattr(producer, "run_producer", run)
    events = Queue()
    actor = StandaloneProducerActor(config, events, reference_batch_size=64)
    result = asyncio.run(actor._run_producer())
    messages = [events.get_nowait() for _ in range(events.qsize())]
    assert result.published_count == 2
    assert [event["kind"] for event in messages] == [
        "samples_ready", "samples_ready", "producer_metrics", "producer_done",
    ]
    if credit_limit:
        assert messages[-1]["producer_incarnation"] == messages[0]["producer_incarnation"] == actor._incarnation
    else:
        assert "producer_incarnation" not in messages[-1]


def test_producer_actor_injects_directory_backed_reference_store(monkeypatch):
    from queue import Queue
    import verl_speco.standalone_tq_producer as producer
    from verl_speco.transport.native_feature_transport import NativeProducerReferenceStore
    config = _runtime_config()
    monkeypatch.setattr(config.actor_rollout_ref.rollout.drafter.training, "feature_store",
                        SimpleNamespace(type="ray_reference"), raising=False)
    directory = object()
    async def run(_config, **kwargs):
        store = kwargs["hidden_states_store"]
        assert isinstance(store, NativeProducerReferenceStore) and store.router.directory is directory
        return producer.ProducerStats(published_count=0)
    monkeypatch.setattr(producer, "run_producer", run)
    actor = StandaloneProducerActor(config, Queue(), reference_batch_size=1,
                                    native_transport={"directory": directory, "timeout": 1})
    assert asyncio.run(actor._run_producer()).published_count == 0


def test_actor_persistent_lease_tracks_admission_ack_release_and_normal_drain(tmp_path):
    from queue import Queue
    import sqlite3
    path = tmp_path / "leases.sqlite"
    config = {"speco": {"standalone_tq_producer": {
        "max_published_references": 1, "reference_lease_journal_path": str(path)}},
        "actor_rollout_ref": {"rollout": {"drafter": {"training": {"transfer_queue": {"run_id": "run"}}}}}}
    actor = StandaloneProducerActor(config, Queue(), reference_batch_size=1)
    async def run():
        await actor._on_references([{"key": "sample", "tag": {}}])
        assert actor._lease_journal.db.execute("SELECT COUNT(*) FROM payloads").fetchone()[0] == 1
        await actor.register_dispatch(actor._incarnation, 0, "receipt", ["sample"], 2)
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
        with pytest.raises(ValueError, match="every rank"):
            await actor.release_references(actor._incarnation, 0, "receipt", ["sample"])
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 1)
        assert actor._lease_journal.db.execute("SELECT phase FROM leases WHERE plan=0").fetchone() == ("RELEASE_INTENT",)
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
        await actor.release_references(actor._incarnation, 0, "receipt", ["sample"])
        await actor._close_reference_admission()
        assert actor._lease_journal is None
    asyncio.run(run())
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM owner").fetchone()[0] == 0


def test_persistent_lease_journal_survives_process_exit_without_clean_handoff(tmp_path):
    import subprocess
    import sqlite3
    from verl_speco.standalone_ray_runtime import _ReferenceLeaseJournal

    path = tmp_path / "crashed.sqlite"
    code = """
import os, sys
from verl_speco.standalone_ray_runtime import _ReferenceLeaseJournal
journal = _ReferenceLeaseJournal(sys.argv[1], 'run', 'crashed-incarnation')
journal.admit([{'key': 'sample', 'tag': {'backend': 'file'}}])
identity = {'receipt': 'receipt', 'keys': ['sample'], 'owner_count': 2, 'native_sources': {}}
journal.transition(0, 'DISPATCH_INTENT', identity)
journal.transition(0, 'LEASED', identity)
journal.acknowledge(0, 0)
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 17, result.stderr
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT phase FROM leases WHERE plan=0").fetchone() == ("LEASED",)
        assert db.execute("SELECT rank FROM acknowledgements WHERE plan=0").fetchall() == [(0,)]
        assert db.execute("SELECT COUNT(*) FROM payloads").fetchone()[0] == 1
    with pytest.raises(RuntimeError, match="already has an owner"):
        _ReferenceLeaseJournal(path, "run", "successor")


def test_persistent_lease_journal_requires_all_acknowledgements_and_clean_handoff(tmp_path):
    from verl_speco.standalone_ray_runtime import _ReferenceLeaseJournal
    path = tmp_path / "leases.sqlite"
    identity = {"receipt": "receipt", "keys": ["sample"], "owner_count": 2, "native_sources": {}}
    journal = _ReferenceLeaseJournal(path, "run", "first")
    journal.admit([{"key": "sample", "tag": {"backend": "file"}}])
    journal.transition(0, "DISPATCH_INTENT", identity)
    journal.transition(0, "LEASED", identity)
    journal.acknowledge(0, 0)
    journal.acknowledge(0, 0)
    with pytest.raises(ValueError, match="every owner"):
        journal.transition(0, "RELEASE_INTENT", identity)
    with pytest.raises(RuntimeError, match="already has an owner"):
        _ReferenceLeaseJournal(path, "run", "second")
    journal.acknowledge(0, 1)
    journal.transition(0, "MATERIALIZED", identity)
    journal.transition(0, "RELEASE_INTENT", identity)
    journal.transition(0, "RELEASED", identity)
    with pytest.raises(RuntimeError, match="unresolved"):
        journal.close(drained=True)
    journal.retire(["sample"])
    journal.close(drained=True)
    successor = _ReferenceLeaseJournal(path, "run", "second")
    successor.close()
    with pytest.raises(RuntimeError, match="already has an owner"):
        _ReferenceLeaseJournal(path, "run", "third")


def test_native_abort_freezes_sources_without_returning_active_credits(monkeypatch):
    from verl_speco.transport.native_feature_transport import NativeSourceDirectoryRouter
    router = NativeSourceDirectoryRouter(object(), timeout=1)
    calls = []
    monkeypatch.setattr(router, "stop_admission", lambda: calls.append("frozen"))
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, reference_batch_size=1, native_transport=router)
    async def run():
        actor._active_references.add("retained")
        actor._run_gate.clear()
        with pytest.raises(ValueError, match="stale"):
            await actor.abort_references("old")
        assert calls == []
        await actor.abort_references(actor._incarnation)
        assert actor._reference_aborted and actor._run_gate.is_set()
        assert actor._active_references == {"retained"}
        with pytest.raises(RuntimeError, match="stopped"):
            await actor._before_request()
    asyncio.run(run())
    assert calls == ["frozen"]


def test_native_source_seal_requires_drained_producer(monkeypatch):
    from verl_speco.transport.native_feature_transport import NativeSourceDirectoryRouter
    calls = []
    router = NativeSourceDirectoryRouter(object(), timeout=1)
    monkeypatch.setattr(router, "freeze_and_seal", lambda: calls.append("sealed"))
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, reference_batch_size=1, native_transport=router)
    async def run():
        with pytest.raises(RuntimeError, match="stopped, drained"):
            await actor.seal_native_sources(actor._incarnation)
        await actor._close_reference_admission()
        actor._active_references.add("sample")
        with pytest.raises(RuntimeError, match="stopped, drained"):
            await actor.seal_native_sources(actor._incarnation)
        actor._active_references.clear()
        with pytest.raises(ValueError, match="stale"):
            await actor.seal_native_sources("old")
        await actor.seal_native_sources(actor._incarnation)
    asyncio.run(run())
    assert calls == ["sealed"]


def test_native_source_seal_failure_aborts_reference_admission(monkeypatch):
    from verl_speco.transport.native_feature_transport import NativeSourceDirectoryRouter
    router = NativeSourceDirectoryRouter(object(), timeout=1)
    def fail():
        raise TimeoutError("seal response lost")
    monkeypatch.setattr(router, "freeze_and_seal", fail)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, reference_batch_size=1, native_transport=router)
    async def run():
        await actor._close_reference_admission()
        with pytest.raises(TimeoutError, match="response lost"):
            await actor.seal_native_sources(actor._incarnation)
        assert actor._reference_aborted
        with pytest.raises(RuntimeError, match="stopped, drained"):
            await actor.seal_native_sources(actor._incarnation)
    asyncio.run(run())


def test_ray_reference_pipeline_never_starts_tq_and_preserves_untrained_files(tmp_path, monkeypatch):
    from verl_speco import standalone_ray_runtime as runtime

    hidden_dir = tmp_path / "hidden"
    hidden_dir.mkdir()
    untrained = hidden_dir / "untrained.safetensors"
    untrained.write_bytes(b"untrained")
    commands = _commands()
    commands = PipelineCommands(
        **{**commands.__dict__, "consumer_overrides": ("training.feature_store.type=ray_reference",)}
    )
    original_mkdtemp = runtime.tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") == "speco-vllm-hidden-states-":
            return str(hidden_dir)
        return original_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(runtime.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(runtime, "validate_tq_backend", lambda *args: pytest.fail("TQ validation"))
    starts = []
    monkeypatch.setattr(runtime, "run_ray_actors", lambda *args, **kwargs: starts.append(kwargs["pipeline_started_at"]) or 0)
    result = runtime.run_ray_pipeline(
        commands, ray_module=object(), ray_address="unused", environ={},
        endpoint_ready=lambda _: True,
        popen=lambda *args, **kwargs: pytest.fail("Started a TQ owner"),
    )
    assert result == 0 and untrained.exists() and len(starts) == 1 and starts[0] > 0


@pytest.mark.parametrize("failure", [False, True])
def test_native_ray_pipeline_owns_directory_and_preserves_unfenced_source(monkeypatch, failure, caplog):
    from verl_speco import standalone_ray_runtime as runtime
    from verl_speco.standalone_tq_training_launcher import build_pipeline_commands, resolve_pipeline_config
    args = ["data.train_files=/data/train.jsonl", "actor_rollout_ref.model.path=/model",
            "actor_rollout_ref.rollout.drafter.model_path=/draft"]
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_NATIVE_SOURCE_ENDPOINT": "source:1234",
           "SPECO_NATIVE_READER_ENDPOINTS": '["reader:1234"]'}
    config = resolve_pipeline_config(args, environ=env)
    commands = build_pipeline_commands(config, args, ray_address="ray:6379", env=env)
    calls = []
    class Method:
        def remote(self): return {"run_id": config.run_id, "generation": 0}
    actor = SimpleNamespace(scope=Method(), status=SimpleNamespace(remote=lambda: {"active": 1}))
    class ActorClass:
        def options(self, **kwargs):
            calls.append(("options", kwargs))
            return self
        def remote(self, **kwargs):
            calls.append(("directory", kwargs))
            return actor
    class Ray:
        def remote(self, **kwargs): return lambda _: ActorClass()
        def get(self, value, timeout): return value
        def kill(self, value, **kwargs): calls.append(("kill", value))
    process = SimpleNamespace(pid=123)
    def popen(command, *, env):
        calls.append(("popen", env))
        return process
    def run_actors(_commands, **kwargs):
        assert kwargs["native_transport"]["directory"] is actor
        if failure:
            raise RuntimeError("injected failure")
        return 0
    monkeypatch.setattr(runtime, "run_ray_actors", run_actors)
    monkeypatch.setattr(runtime, "_wait_for_vllm_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_stop_process", lambda value: calls.append(("stop", value)) if value else None)
    if failure:
        with pytest.raises(RuntimeError, match="injected failure"):
            runtime.run_ray_pipeline(commands, ray_module=Ray(), ray_address="ray:6379", environ=env,
                                     popen=popen, endpoint_ready=lambda _: False)
    else:
        assert runtime.run_ray_pipeline(commands, ray_module=Ray(), ray_address="ray:6379", environ=env,
                                        popen=popen, endpoint_ready=lambda _: False) == 0
    child = next(value for kind, value in calls if kind == "popen")
    assert child["SPECO_NATIVE_RUN_ID"] == config.run_id
    assert child["SPECO_NATIVE_DIRECTORY"] == commands.native_settings["directory"]
    assert (("stop", process) in calls) is not failure
    assert (("kill", actor) in calls) is not failure
    if failure:
        assert "Preserved native directory status" in caplog.text


def test_native_ray_pipeline_rejects_unidentified_existing_local_service():
    from verl_speco import standalone_ray_runtime as runtime
    from verl_speco.standalone_tq_training_launcher import build_pipeline_commands, resolve_pipeline_config
    args = ["data.train_files=/data/train.jsonl", "actor_rollout_ref.model.path=/model"]
    env = {"SPECO_VLLM_HIDDEN_STATES_STORE": "native", "SPECO_NATIVE_SOURCE_ENDPOINT": "source:1234",
           "SPECO_NATIVE_READER_ENDPOINTS": '["reader:1234"]'}
    commands = build_pipeline_commands(resolve_pipeline_config(args, environ=env), args,
                                       ray_address="ray:6379", env=env)
    with pytest.raises(RuntimeError, match="existing native vLLM requires explicit"):
        runtime.run_ray_pipeline(commands, ray_module=object(), ray_address="ray:6379",
                                 environ=env, endpoint_ready=lambda _: True)


def test_producer_restart_rechecks_credits_after_fingerprinting(tmp_path, monkeypatch):
    from copy import deepcopy
    from threading import Event
    import verl_speco.standalone_ray_runtime as runtime
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"row\n")
    config = deepcopy(_runtime_config())
    config.speco.standalone_tq_producer.input_path = str(source)
    config.speco.standalone_tq_producer.max_published_references = 1
    config.actor_rollout_ref.rollout.drafter.training.feature_store = SimpleNamespace(type="ray_reference")
    started, release = Event(), Event()
    original = runtime.build_input_fingerprint
    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, "build_input_fingerprint", delayed)
    actor = StandaloneProducerActor(config, reference_batch_size=1)

    async def run():
        incarnation = actor._incarnation
        start_task = asyncio.create_task(actor.start())
        assert await asyncio.to_thread(started.wait, 5)
        async with actor._reference_condition:
            actor._active_references.add("key")
        release.set()
        with pytest.raises(RuntimeError, match="outstanding reference credits"):
            await start_task
        assert actor._incarnation == incarnation and actor._active_references == {"key"}
        assert actor._task is None

    asyncio.run(run())


def test_producer_start_and_stop_serialize_input_fingerprinting(tmp_path, monkeypatch):
    from copy import deepcopy
    from threading import Event
    import verl_speco.standalone_ray_runtime as runtime
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"row\n")
    config = deepcopy(_runtime_config())
    config.speco.standalone_tq_producer.input_path = str(source)
    config.actor_rollout_ref.rollout.drafter.training.feature_store = SimpleNamespace(type="ray_reference")
    started, release = Event(), Event()
    original = runtime.build_input_fingerprint
    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, "build_input_fingerprint", delayed)
    actor = StandaloneProducerActor(config)
    async def producer():
        await asyncio.Event().wait()
    actor._run_producer = producer
    async def run():
        start_task = asyncio.create_task(actor.start())
        assert await asyncio.to_thread(started.wait, 5)
        stop_task = asyncio.create_task(actor.stop())
        await asyncio.sleep(0)
        assert not stop_task.done()
        release.set()
        started_result = await start_task
        assert started_result["input_identity"]["kind"] == "file"
        assert await stop_task == {"stopped": True}
    try:
        asyncio.run(run())
    finally:
        release.set()


def test_producer_actor_reuses_existing_producer_and_supports_stop(monkeypatch) -> None:
    started = asyncio.Event()

    async def fake_run_producer(config, **kwargs):
        assert config == {"producer": True}
        assert set(kwargs) == {
            "before_request",
            "on_published",
            "on_metrics",
            "get_runtime_state",
        }
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setitem(
        sys.modules,
        "verl_speco.standalone_tq_producer",
        SimpleNamespace(run_producer=fake_run_producer),
    )
    actor = StandaloneProducerActor({"producer": True})

    async def exercise():
        assert await actor.start() == {"started": True}
        await started.wait()
        assert await actor.start() == {"started": False}
        assert await actor.stop() == {"stopped": True}

    asyncio.run(exercise())


def test_producer_actor_coalesces_publish_notifications_as_cumulative_total(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class _Stats:
        published_count: int

    class _Events:
        def __init__(self):
            self.items = []

        def put(self, value):
            self.items.append(value)

    async def fake_run_producer(config, **kwargs):
        for sequence_no in range(3):
            await kwargs["on_published"](sequence_no)
        return _Stats(published_count=3)

    monkeypatch.setitem(
        sys.modules,
        "verl_speco.standalone_tq_producer",
        SimpleNamespace(run_producer=fake_run_producer),
    )
    events = _Events()
    actor = StandaloneProducerActor({"producer": True}, events)

    async def exercise():
        assert await actor.start() == {"started": True}
        assert await actor.wait() == {"published_count": 3}

    asyncio.run(exercise())

    assert events.items == [
        {"kind": "samples_published", "published_total": 3}
    ]


def test_producer_actor_delivers_partial_metrics_when_stopped(monkeypatch):
    started = asyncio.Event()
    events = []

    async def fake_run_producer(config, **kwargs):
        try:
            await kwargs["on_published"](0)
            started.set()
            await asyncio.Event().wait()
        finally:
            await kwargs["on_metrics"](1, {"producer/sample_total_time": 2.0})

    monkeypatch.setitem(sys.modules, "verl_speco.standalone_tq_producer",
                        SimpleNamespace(run_producer=fake_run_producer))
    actor = StandaloneProducerActor({"producer": True}, SimpleNamespace(put=events.append))

    async def exercise():
        await actor.start()
        await started.wait()
        await actor.stop()

    asyncio.run(exercise())
    assert {"kind": "producer_metrics", "published_total": 1,
            "metrics": {"producer/sample_total_time": 2.0}} in events


def test_ray_backend_rejects_multi_node_local_runtime() -> None:
    with pytest.raises(ValueError, match="nnodes=1"):
        _validate_runtime_backend_topology(
            "ray", ["speco.draft_training.nnodes=2"], environ={},
        )

    _validate_runtime_backend_topology(
        "subprocess", ["speco.draft_training.nnodes=2"]
    )


def test_multinode_ray_requires_shared_handles_and_remote_endpoints():
    args = ["speco.draft_training.nnodes=1"]
    env = {"SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379"}
    with pytest.raises(ValueError, match="mooncake"):
        _validate_runtime_backend_topology("ray", args, environ=env)
    env["SPECO_VLLM_HIDDEN_STATES_STORE"] = "mooncake"
    with pytest.raises(ValueError, match="remotely reachable"):
        _validate_runtime_backend_topology("ray", args, environ=env)
    env["SPECO_VLLM_ENDPOINTS"] = "[http://10.0.0.2:8000/v1,http://10.0.0.3:8000/v1]"
    with pytest.raises(ValueError, match="reachable Mooncake master"):
        _validate_runtime_backend_topology("ray", args, environ=env)
    env["SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER"] = "10.0.0.4:50051"
    _validate_runtime_backend_topology("ray", args, environ=env)
    env["SPECO_VLLM_HIDDEN_STATES_MOONCAKE_LOCAL_HOSTNAME"] = "10.0.0.9"
    with pytest.raises(ValueError, match="each worker"):
        _validate_runtime_backend_topology("ray", args, environ=env)


def test_multinode_ray_allows_explicit_shared_file_root(tmp_path):
    from omegaconf import OmegaConf
    from verl_speco.standalone_tq_training_launcher import _hidden_states_store_overrides, _vllm_kv_transfer_config
    env = {"SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379", "SPECO_VLLM_ENDPOINTS": "[http://10.0.0.2:8000/v1]",
           "SPECO_VLLM_SHARED_HIDDEN_STATES_ROOT": str(tmp_path)}
    args = ["speco.draft_training.nnodes=2", "speco.draft_training.trainer_node_ips=[10.0.0.3,10.0.0.4]"]
    _validate_runtime_backend_topology("ray", args, environ=env)
    assert _vllm_kv_transfer_config(env)["kv_connector_extra_config"]["shared_storage_path"] == str(tmp_path)
    overrides = _hidden_states_store_overrides(env)
    assert OmegaConf.from_dotlist(overrides).speco.standalone_tq_producer.hidden_states_store.file_root == str(tmp_path)
    env["SPECO_VLLM_SHARED_HIDDEN_STATES_ROOT"] = "relative/path"
    with pytest.raises(ValueError, match="absolute"):
        _validate_runtime_backend_topology("ray", ["speco.draft_training.nnodes=1"], environ=env)
    env["SPECO_VLLM_SHARED_HIDDEN_STATES_ROOT"] = str(tmp_path)
    env["SPECO_VLLM_HIDDEN_STATES_STORE"] = "mooncake"
    with pytest.raises(ValueError, match="file store"):
        _hidden_states_store_overrides(env)


def test_multinode_ray_requires_explicit_trainer_nodes():
    with pytest.raises(ValueError, match="trainer_node_ips"):
        _validate_runtime_backend_topology(
            "ray", ["speco.draft_training.nnodes=2"],
            environ={"SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379"},
        )


def test_multinode_dry_run_validates_explicit_node_list():
    env = {"SPECO_EXTERNAL_RAY_ADDRESS": "10.0.0.1:6379",
           "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
           "SPECO_VLLM_ENDPOINT": "http://10.0.0.3:8000/v1",
           "SPECO_VLLM_HIDDEN_STATES_MOONCAKE_MASTER": "10.0.0.4:50051"}
    args = ["speco.draft_training.nnodes=2"]
    for value in (None, "[10.0.0.1]", "[10.0.0.1,10.0.0.1]", "[bad,10.0.0.2]", "[broken"):
        current = args + ([f"speco.draft_training.trainer_node_ips={value}"] if value else [])
        with pytest.raises(ValueError, match="trainer_node_ips"):
            _validate_runtime_backend_topology("ray", current, environ=env)
    _validate_runtime_backend_topology(
        "ray", args + ["speco.draft_training.trainer_node_ips=[10.0.0.1,10.0.0.2]"],
        environ=env,
    )


def test_live_trainer_topology_checks_rank_node_and_device_identity():
    from verl_speco.trainer.draft_training_loop import _validate_live_trainer_topology
    nodes = ["10.0.0.2", "10.0.0.1"]
    records = [{"rank": i, "ip": f"10.0.0.{1 + i // 2}",
                "node_id": f"node-{i // 2}", "accelerators": [str(i % 2)]} for i in range(4)]
    _validate_live_trainer_topology(records, nodes, 2)
    with pytest.raises(ValueError, match="same accelerator"):
        _validate_live_trainer_topology([*records[:1], {**records[1], "accelerators": ["0"]}, *records[2:]], nodes, 2)
    with pytest.raises(ValueError, match="differs from the plan"):
        _validate_live_trainer_topology([{**records[0], "ip": "10.0.0.2"}, *records[1:]], nodes, 2)
    with pytest.raises(ValueError, match="Cannot verify"):
        _validate_live_trainer_topology([{**records[0], "error": "Ray unavailable"}, *records[1:]], nodes, 2)
    with pytest.raises(ValueError, match="world size"):
        _validate_live_trainer_topology(records[:-1], nodes, 2)
    with pytest.raises(ValueError, match="multiple Ray nodes"):
        _validate_live_trainer_topology([records[0], {**records[1], "node_id": "other"}, *records[2:]], nodes, 2)


def test_pinned_trainer_groups_reserve_distinct_nodes_and_cleanup_on_timeout():
    from verl_speco.standalone_ray_runtime import _create_pinned_trainer_groups

    nodes = [{"Alive": True, "NodeManagerAddress": ip,
              "Resources": {"CPU": 8, "NPU": 8, f"node:{ip}": 1}}
             for ip in ("10.0.0.1", "10.0.0.2")]
    calls, removed = [], []
    class Group:
        def ready(self):
            return self
    def create(bundles, **kwargs):
        group = Group()
        calls.append((bundles, kwargs, group))
        return group
    ray = SimpleNamespace(nodes=lambda: nodes, get=lambda refs, timeout: None)
    kwargs = dict(ray_module=ray, node_ips=["10.0.0.2", "10.0.0.1"],
                  processes_per_node=4, device_name="npu", name="run", timeout=3,
                  placement_api=(create, removed.append))
    groups = _create_pinned_trainer_groups(**kwargs)
    assert len(groups) == 2 and not removed
    assert calls[0][0][0] == {"CPU": 1, "NPU": 1, "node:10.0.0.1": 0.001}
    assert calls[1][0][0]["node:10.0.0.2"] == 0.001
    assert all(call[1]["strategy"] == "STRICT_PACK" for call in calls)
    def timeout(refs, timeout):
        raise TimeoutError("busy")
    ray.get = timeout
    with pytest.raises(TimeoutError, match="busy"):
        _create_pinned_trainer_groups(**kwargs)
    assert removed == [calls[2][2], calls[3][2]]
    kwargs["node_ips"] = ["10.0.0.1", "10.0.0.1"]
    with pytest.raises(ValueError, match="distinct"):
        _create_pinned_trainer_groups(**kwargs)
    kwargs["node_ips"] = ["10.0.0.1", "10.0.0.9"]
    with pytest.raises(ValueError, match="capacity"):
        _create_pinned_trainer_groups(**kwargs)


def test_unpinned_group_timeout_removes_owned_placement():
    from verl_speco.standalone_ray_runtime import _create_unpinned_trainer_group
    created, removed = [], []
    class Group:
        def ready(self): return "ready"
    def create(bundles, **kwargs):
        group = Group()
        created.append((bundles, kwargs, group))
        return group
    class Ray:
        def get(self, value, timeout):
            assert value == "ready" and timeout == 2
            raise TimeoutError("injected placement timeout")
    with pytest.raises(TimeoutError, match="placement timeout"):
        _create_unpinned_trainer_group(Ray(), processes_per_node=2, device_name="npu",
                                       name="task", timeout=2, placement_api=(create, removed.append))
    assert created[0][0] == [{"CPU": 1, "NPU": 1}, {"CPU": 1, "NPU": 1}]
    assert created[0][1]["strategy"] == "STRICT_PACK"
    assert removed == [created[0][2]]


def test_single_node_worker_group_failure_removes_placement(monkeypatch):
    import importlib
    import verl.utils.device
    from omegaconf import OmegaConf
    from verl_speco import standalone_ray_runtime as runtime
    config = OmegaConf.create({"speco": {"draft_training": {"nnodes": 1, "nproc_per_node": 2}}})
    group, removed = object(), []
    class Pool:
        def __init__(self, **kwargs): self.pgs = None
    def failing_group(**kwargs):
        assert kwargs["resource_pool"].pgs == [group]
        raise RuntimeError("injected worker creation failure")
    monkeypatch.setattr(runtime, "_ray_controller_types", lambda: (lambda **kw: object(), Pool, failing_group))
    monkeypatch.setattr(runtime, "_create_unpinned_trainer_group", lambda *a, **kw: [group])
    monkeypatch.setattr(verl.utils.device, "get_device_name", lambda: "npu")
    monkeypatch.setattr(importlib.import_module("ray.util.placement_group"), "remove_placement_group", removed.append)
    with pytest.raises(RuntimeError, match="worker creation failure"):
        runtime.build_consumer_worker_group(config, ray_module=SimpleNamespace(remote=lambda cls: cls))
    assert removed == [group]


def test_worker_group_uses_the_pinned_placement_groups(monkeypatch):
    from omegaconf import OmegaConf
    from verl_speco import standalone_ray_runtime as runtime
    import verl.utils.device

    config = OmegaConf.create({"speco": {"draft_training": {
        "nnodes": 2, "nproc_per_node": 4,
        "trainer_node_ips": ["10.0.0.1", "10.0.0.2"],
        "placement_timeout_seconds": 20,
    }}})
    observed, groups = {}, [object(), object()]
    class Pool:
        def __init__(self, **kwargs):
            observed["pool"] = kwargs
            self.pgs = None
    def make_worker(**kwargs):
        observed["worker"] = kwargs
        return object()
    def make_group(**kwargs):
        observed["group"] = kwargs
        return "worker-group"
    def reserve(ray, **kwargs):
        observed["reservation"] = kwargs
        return groups
    monkeypatch.setattr(runtime, "_ray_controller_types", lambda: (make_worker, Pool, make_group))
    monkeypatch.setattr(runtime, "_create_pinned_trainer_groups", reserve)
    monkeypatch.setattr(verl.utils.device, "get_device_name", lambda: "npu")
    ray = SimpleNamespace(remote=lambda cls: cls)
    assert runtime.build_consumer_worker_group(config, ray_module=ray) == "worker-group"
    assert observed["pool"]["process_on_nodes"] == [4, 4]
    assert observed["reservation"]["node_ips"] == ["10.0.0.1", "10.0.0.2"]
    assert observed["group"]["resource_pool"].pgs is groups
    assert observed["group"]["device_name"] == "npu"


@pytest.mark.parametrize("mode", ["rank_ack", "shared_root"])
def test_producer_handle_is_passed_for_ack_or_shared_witness(monkeypatch, tmp_path, mode):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"per_rank_reference_ack": mode == "rank_ack"}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    config.speco.standalone_tq_producer.max_published_references = 4
    if mode == "shared_root":
        config.speco.standalone_tq_producer.hidden_states_store = {"file_root": str(tmp_path)}
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    source, observed = _ProducerHandle(), {}
    class Remote:
        def options(self, **kwargs): return self
        def remote(self, *args): return source
    class Ray:
        def remote(self, **kwargs): return lambda cls: Remote()
        def kill(self, actor, **kwargs): pass
    class Queue:
        def shutdown(self, **kwargs): pass
    def factory(cfg, **kwargs):
        observed.update(kwargs)
        return _WorkerGroup()
    trainer = StandaloneRayTrainer(_commands(), ray_module=Ray(), worker_group_factory=factory,
                                   queue_factory=lambda **kwargs: Queue())
    trainer.init_workers()
    try:
        assert observed["producer_handle"] is source
    finally:
        trainer.close()


def test_rank_ack_requires_a_producer_credit_limit(monkeypatch):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"per_rank_reference_ack": True}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    config.actor_rollout_ref.rollout.drafter.training.feature_store = SimpleNamespace(type="ray_reference")
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=SimpleNamespace())
    with pytest.raises(ValueError, match="requires Producer reference credits"):
        trainer.run()


@pytest.mark.parametrize("source,target,directory,ack", [
    ("native", "file", True, True), ("file", "native", True, True),
    ("native", "native", False, True), ("native", "native", True, False),
])
def test_native_scheduler_rejects_unsafe_configuration(monkeypatch, source, target, directory, ack):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"per_rank_reference_ack": ack}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    config.speco.standalone_tq_producer.max_published_references = 4
    config.speco.standalone_tq_producer.hidden_states_store = {"backend": source}
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.target_feature_replay = {"hidden_states_store": {"backend": target}}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=SimpleNamespace(),
                                   native_transport={"directory": object()} if directory else None)
    with pytest.raises(ValueError, match="Native transport requires"):
        trainer.run()


def test_shared_file_root_must_match_between_roles_before_worker_creation(monkeypatch, tmp_path):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    config.speco.standalone_tq_producer.hidden_states_store = {"file_root": str(tmp_path / "producer")}
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.target_feature_replay = {"hidden_states_store": {"file_root": str(tmp_path / "consumer")}}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=SimpleNamespace())
    with pytest.raises(ValueError, match="same shared file root"):
        trainer.run()


def test_worker_init_failure_cleans_created_producer_and_queues(monkeypatch):
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: _runtime_config())
    killed, closed = [], []
    class Queue:
        def shutdown(self, **kwargs): closed.append(self)
    class Ray(_FakeRay):
        def kill(self, actor, **kwargs): killed.append(actor)
    ray = Ray(_ProducerHandle())
    queues = [Queue(), Queue()]
    def fail_group(*args, **kwargs):
        raise RuntimeError("injected group failure")
    source = iter(queues)
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray, worker_group_factory=fail_group,
                                   queue_factory=lambda **kwargs: next(source))
    with pytest.raises(RuntimeError, match="group failure"):
        trainer.init_workers()
    assert killed == [ray.producer] and closed == queues
    assert trainer.producer is None and trainer.commands_queue is None and trainer.events_queue is None


def test_close_releases_only_owned_actors_and_queues_and_is_idempotent():
    trainer = StandaloneRayTrainer.__new__(StandaloneRayTrainer)
    killed, closed = [], []
    trainer.ray = SimpleNamespace(kill=lambda actor, **kw: killed.append(actor))
    trainer.producer = "producer"
    trainer.consumer_wg = SimpleNamespace(workers=["rank-0", "rank-1"])
    trainer.commands_queue = SimpleNamespace(shutdown=lambda **kw: closed.append("commands"))
    trainer.events_queue = SimpleNamespace(shutdown=lambda **kw: closed.append("events"))
    trainer.close()
    trainer.close()
    assert killed == ["producer", "rank-0", "rank-1"]
    assert closed == ["commands", "events"]


@dataclass(frozen=True)
class _Ref:
    name: str


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback

    def remote(self):
        return self.callback()


class _ProducerHandle:
    def __init__(self):
        self.stopped = False
        self.start = _RemoteMethod(lambda: _Ref("start"))
        self.wait = _RemoteMethod(lambda: _Ref("producer"))
        self.stop = _RemoteMethod(self._stop)

    def _stop(self):
        self.stopped = True
        return _Ref("stop")


class _RemoteProducerClass:
    def __init__(self, handle):
        self.handle = handle

    def options(self, **kwargs):
        assert kwargs["name"].startswith("speco_standalone_producer_")
        return self

    def remote(self, config, events, *args):
        assert config is _runtime_config()
        return self.handle


class _WorkerGroup:
    def run_standalone_training(self):
        return [_Ref("consumer-0"), _Ref("consumer-1")]


class _FakeRay:
    def __init__(self, producer):
        self.producer = producer

    def remote(self, **kwargs):
        assert kwargs == {"max_concurrency": 4}
        return lambda cls: _RemoteProducerClass(self.producer)

    def get(self, ref, timeout=None):
        return {"ref": ref.name}

    def wait(self, refs, num_returns, timeout=0):
        assert num_returns == 1
        stop = next((ref for ref in refs if ref.name == "stop"), None)
        if stop is not None:
            return [stop], [ref for ref in refs if ref != stop]
        producer = next((ref for ref in refs if ref.name == "producer"), None)
        completed = producer or next(
            ref for ref in refs if ref.name.startswith("consumer")
        )
        return [completed], [ref for ref in refs if ref != completed]


class _Queue:
    def put(self, value):
        pass

    def get(self, *, block, timeout=None):
        if not block or timeout == 0.0:
            from queue import Empty
            raise Empty
        raise AssertionError("event queue should not be read in this test")


class _Store:
    def connect(self):
        pass

    def list_ready(self):
        return []

    def close(self):
        pass


class _ReadyStore(_Store):
    def __init__(self):
        self.ready = [
            SimpleNamespace(key=f"sample-{index}", tag={"sequence_no": index})
            for index in range(4)
        ]

    def list_ready(self):
        return list(self.ready)


class _RecordingQueue(_Queue):
    def __init__(self, *, events=None, on_event=None):
        self.items = [] if events is None else list(events)
        self.on_event = on_event

    def put(self, value):
        self.items.append(value)

    def get(self, *, block, timeout=None):
        if not self.items:
            from queue import Empty

            raise Empty
        value = self.items.pop(0)
        if self.on_event is not None:
            self.on_event(value)
        return value


class _FlowRay(_FakeRay):
    def __init__(self, producer):
        super().__init__(producer)
        self.wait_calls = 0

    def wait(self, refs, num_returns, timeout=0):
        self.wait_calls += 1
        if refs and refs[0].name == "stop":
            return refs[:1], refs[1:]
        if self.wait_calls == 1:
            return [], list(refs)
        producer = next((ref for ref in refs if ref.name == "producer"), refs[0])
        return [producer], [ref for ref in refs if ref != producer]


def test_driver_stops_on_background_finalization_failure(monkeypatch):
    queues = iter((_RecordingQueue(), _RecordingQueue(events=[
        {"kind": "finalization_failed", "plan_id": 0},
    ])))
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=_FlowRay(_ProducerHandle()),
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda config: _ReadyStore(),
    )
    with pytest.raises(RuntimeError, match="failed to finalize trained plan=0"):
        trainer.run()


def test_file_driver_kills_group_when_one_rank_reports_training_failure(monkeypatch):
    from copy import deepcopy

    config = deepcopy(_runtime_config())
    training = config.actor_rollout_ref.rollout.drafter.training
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.get = lambda key, default=None: getattr(training, key, default)
    training.transfer_queue = {"consumer_prefetch_batches": 2}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    queues = iter((_RecordingQueue(), _RecordingQueue(events=[
        {"kind": "consumer_failed", "rank": 1, "error": "injected optimizer failure"},
    ])))
    ray = _FlowRay(_ProducerHandle())
    killed = []
    ray.kill = lambda worker, **kw: killed.append((worker, kw))
    group = _WorkerGroup()
    group.workers = ["rank0", "rank1"]
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=ray, worker_group_factory=lambda *a, **k: group,
        queue_factory=lambda **kwargs: next(queues), feature_store_factory=lambda _: _ReadyStore(),
    )
    trainer.producer = ray.producer
    trainer.consumer_wg = group
    trainer.commands_queue, trainer.events_queue = next(queues), next(queues)
    original_wait = ray.wait
    ray.wait = lambda refs, **kw: ((refs, []) if refs[0].name == "stop" else original_wait(refs, **kw))
    with pytest.raises(RuntimeError, match="rank=1 failed: injected optimizer failure"):
        trainer.run()
    assert killed == [("rank0", {"no_restart": True}), ("rank1", {"no_restart": True})]


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("stuck", ["consumer", "producer"])
def test_shutdown_timeout_is_bounded_for_each_store(monkeypatch, reference, stuck):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"consumer_shutdown_timeout_seconds": 0.01,
                                             "producer_stop_timeout_seconds": 0.01}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference" if reference else "tq")
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    training.max_steps = 1
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    cleaned, killed = [], []
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.cleanup_standalone_samples",
                        lambda *args, **kwargs: cleaned.append(True) or [])
    events = _RecordingQueue(events=[{"kind": "consumer_ready", "initial_optimizer_step": 1}])
    class Ray(_FakeRay):
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop":
                return ([], refs) if stuck == "producer" else (refs[:1], refs[1:])
            if all(ref.name.startswith("consumer") for ref in refs):
                return ([], refs) if stuck == "consumer" else (refs[:1], refs[1:])
            return [], refs
        def kill(self, worker, **kwargs):
            killed.append(worker)
    ray = Ray(_ProducerHandle())
    group = _WorkerGroup()
    group.workers = ["rank0", "rank1"]
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray, feature_store_factory=lambda _: _ReadyStore())
    trainer.producer, trainer.consumer_wg = ray.producer, group
    trainer.commands_queue, trainer.events_queue = _RecordingQueue(), events
    if stuck == "consumer":
        with pytest.raises(TimeoutError, match="Consumer shutdown timed out"):
            trainer.run()
        assert killed == ["rank0", "rank1"]
    else:
        with pytest.raises(RuntimeError, match="Failed to stop standalone Producer safely"):
            trainer.run()
    assert not cleaned


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "5"])
def test_commit_gate_timeout_is_validated_before_workers(value):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"commit_gate_timeout_seconds": value}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    trainer = StandaloneRayTrainer.__new__(StandaloneRayTrainer)
    trainer.consumer_config = config
    with pytest.raises(ValueError, match="commit_gate_timeout_seconds"):
        trainer.run()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "5"])
def test_producer_start_timeout_is_validated_before_workers(value):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"producer_start_timeout_seconds": value}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    trainer = StandaloneRayTrainer.__new__(StandaloneRayTrainer)
    trainer.consumer_config = config
    with pytest.raises(ValueError, match="producer_start_timeout_seconds"):
        trainer.run()


def test_producer_start_timeout_stops_actor_without_starting_consumers(monkeypatch):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"producer_start_timeout_seconds": 0.01}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    class Ray(_FakeRay):
        def get(self, ref, timeout=None):
            if ref.name == "start":
                assert timeout == 0.01
                raise TimeoutError("injected slow start")
            return super().get(ref, timeout=timeout)
    class Group:
        workers = []
        def run_standalone_training(self):
            pytest.fail("Consumers must not start before Producer acknowledgement")
    ray = Ray(_ProducerHandle())
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray, feature_store_factory=lambda _: _Store())
    trainer.producer, trainer.consumer_wg = ray.producer, Group()
    trainer.commands_queue, trainer.events_queue = _RecordingQueue(), _RecordingQueue()
    with pytest.raises(TimeoutError, match="Producer start timed out"):
        trainer.run()
    assert ray.producer.stopped


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "5"])
def test_consumer_shutdown_timeout_is_validated_before_workers(value):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"consumer_shutdown_timeout_seconds": value}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    trainer = StandaloneRayTrainer.__new__(StandaloneRayTrainer)
    trainer.consumer_config = config
    with pytest.raises(ValueError, match="consumer_shutdown_timeout_seconds"):
        trainer.run()


@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("credits", [False, True])
def test_failure_preserves_unfenced_references(monkeypatch, nodes, credits):
    from copy import deepcopy

    config = deepcopy(_runtime_config())
    config.speco.draft_training.nnodes = nodes
    if credits:
        config.speco.standalone_tq_producer.max_published_references = 4 * nodes
    training = config.actor_rollout_ref.rollout.drafter.training
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.get = lambda key, default=None: getattr(training, key, default)
    training.transfer_queue = {"consumer_prefetch_batches": 2}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    cleaned = []
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.cleanup_standalone_samples",
                        lambda store, entries, **kwargs: cleaned.extend(entries))
    events = _RecordingQueue(events=[
        {"kind": "consumer_failed", "rank": 1, "error": "injected failure"},
        {"kind": "samples_ready", "entries": [{"key": "late", "tag": {}}]},
    ])
    source, calls = _ProducerHandle(), []
    source.abort_references = SimpleNamespace(remote=lambda incarnation: (calls.append(("abort", incarnation)), _Ref("abort"))[1])
    source.stop = _RemoteMethod(lambda: (calls.append(("stop", None)), source._stop())[1])
    ray = _FlowRay(source)
    if credits:
        original_get = ray.get
        ray.get = lambda ref, timeout=None: ({"started": True, "producer_incarnation": "f" * 32}
                                           if ref.name == "start" else original_get(ref, timeout))
    original_wait = ray.wait
    ray.wait = lambda refs, **kw: ((refs, []) if refs[0].name == "stop" else original_wait(refs, **kw))
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=ray, feature_store_factory=lambda _: _ReadyStore(),
    )
    trainer.producer, trainer.consumer_wg = ray.producer, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = _RecordingQueue(), events
    with pytest.raises(RuntimeError, match="injected failure"):
        trainer.run()
    assert not cleaned
    assert calls == [("abort", "f" * 32 if credits else None), ("stop", None)]


@pytest.mark.parametrize("start_failure", ["invalid_receipt", "lost_reply"])
def test_invalid_producer_start_receipt_stops_started_actor(monkeypatch, start_failure):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.standalone_tq_producer.max_reference_execution_slots = 4
    training = config.actor_rollout_ref.rollout.drafter.training
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.get = lambda key, default=None: getattr(training, key, default)
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    ray = _FlowRay(_ProducerHandle())
    original_get = ray.get
    if start_failure == "lost_reply":
        def get(ref, timeout=None):
            if ref.name == "start":
                raise TimeoutError("startup reply lost")
            return original_get(ref)
        ray.get = get
    original_wait = ray.wait
    ray.wait = lambda refs, **kw: ((refs, []) if refs[0].name == "stop" else original_wait(refs, **kw))
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray, feature_store_factory=lambda _: _ReadyStore())
    trainer.producer, trainer.consumer_wg = ray.producer, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = _RecordingQueue(), _RecordingQueue()
    expected = "did not establish a valid incarnation" if start_failure == "invalid_receipt" else "Producer start timed out"
    with pytest.raises(RuntimeError if start_failure == "invalid_receipt" else TimeoutError, match=expected):
        trainer.run()
    assert ray.producer.stopped


def test_scheduler_executes_one_complete_standalone_batch(monkeypatch) -> None:
    store = _ReadyStore()
    commands = _RecordingQueue()
    events = _RecordingQueue(
        events=[
            {
                "kind": "training_completed",
                "keys": [f"sample-{index}" for index in range(4)],
                "successful": True,
            }
        ],
        on_event=lambda event: store.ready.clear(),
    )
    queues = iter((commands, events))
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=_FlowRay(_ProducerHandle()),
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=lambda: next(queues),
        feature_store_factory=lambda config: store,
    )

    assert trainer.run() == 0
    assert commands.items[0]["kind"] == "batch"
    assert commands.items[0]["global_keys"] == [
        f"sample-{index}" for index in range(4)
    ]
    assert commands.items[-1] == {"kind": "stop"}


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_scheduler_prefills_capacity_and_uses_cached_keys(monkeypatch, depth):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    config.actor_rollout_ref.rollout.drafter.training.transfer_queue = {"consumer_prefetch_batches": depth}
    # deepcopy preserves the fixture's get closure, so bind it to this object.
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    store = _ReadyStore()
    store.ready = [SimpleNamespace(key=f"sample-{i}", tag={"sequence_no": i}) for i in range(12)]
    list_calls = []
    original_list = store.list_ready
    store.list_ready = lambda: (list_calls.append(1), original_list())[1]
    commands = _RecordingQueue()

    def on_event(event):
        if event["kind"] == "training_completed":
            if depth == 0:
                # No later plan is submitted until this batch completes.
                assert sum(c["kind"] == "batch" for c in commands.items) == event["plan_id"] + 1
            if event["plan_id"] == 0:
                assert sum(c["kind"] == "batch" for c in commands.items) == depth + 1
                assert len(list_calls) == 1  # all initial plans use one snapshot
            keys = set(event["keys"])
            store.ready = [e for e in store.ready if e.key not in keys]

    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        *[{"kind": "training_completed", "plan_id": i,
           "keys": [f"sample-{j}" for j in range(i * 4, (i + 1) * 4)], "successful": True}
          for i in range(3)],
    ], on_event=on_event)

    class EventRay(_FakeRay):
        def wait(self, refs, **kwargs):
            if events.items:
                return [], list(refs)
            return super().wait(refs, **kwargs)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=EventRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(), queue_factory=lambda: next(queues),
        feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    batches = [c for c in commands.items if c["kind"] == "batch"]
    assert [c["plan_id"] for c in batches] == [0, 1, 2]
    assert len({key for c in batches for key in c["global_keys"]}) == 12


@pytest.mark.parametrize("initial_step", [8, 9])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_scheduler_respects_restored_remaining_budget(monkeypatch, initial_step, depth):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.transfer_queue = {"consumer_prefetch_batches": depth}
    training.max_steps = 9
    training.get = lambda key, default=None: getattr(training, key, default)
    store = _ReadyStore()
    commands = _RecordingQueue()
    event_items = [{"kind": "consumer_ready", "initial_optimizer_step": initial_step}]
    if initial_step < 9:
        event_items.append({"kind": "training_completed", "plan_id": 0,
                            "keys": [f"sample-{i}" for i in range(4)], "successful": True})
    events = _RecordingQueue(events=event_items)

    class WaitingRay(_FakeRay):
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop" or all(ref.name.startswith("consumer") for ref in refs):
                return refs[:1], refs[1:]
            return [], list(refs)

    queues = iter((commands, events))
    producer = _ProducerHandle()
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=WaitingRay(producer), worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    assert sum(c["kind"] == "batch" for c in commands.items) == 9 - initial_step
    assert commands.items[-1] == {"kind": "stop"}
    assert producer.stopped


def test_fast_producer_watermarks_and_duplicate_events_do_not_block_training(monkeypatch):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    training.get = lambda key, default=None: getattr(training, key, default)
    store = _ReadyStore()
    store.ready.clear()
    calls = []
    original_list = store.list_ready
    store.list_ready = lambda: (calls.append(1), original_list())[1]
    commands = _RecordingQueue()
    published = 0
    list_count_before_full_publication = None

    def on_event(event):
        nonlocal published, list_count_before_full_publication
        if event["kind"] == "samples_published":
            total = event["published_total"]
            if total == 24:
                if list_count_before_full_publication is None:
                    list_count_before_full_publication = len(calls)
                else:
                    assert len(calls) == list_count_before_full_publication
            store.ready.extend(SimpleNamespace(key=f"sample-{i}", tag={"sequence_no": i})
                               for i in range(published, total))
            published = total
        elif event["kind"] == "training_completed":
            keys = set(event["keys"])
            store.ready = [e for e in store.ready if e.key not in keys]

    completions = [{"kind": "training_completed", "plan_id": i,
                    "keys": [f"sample-{j}" for j in range(i * 4, (i + 1) * 4)], "successful": True}
                   for i in range(6)]
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "samples_published", "published_total": 12},
        {"kind": "samples_published", "published_total": 24},
        {"kind": "samples_published", "published_total": 24},
        completions[0], completions[0], *completions[1:],
    ], on_event=on_event)
    producer = _ProducerHandle()
    actions = []
    producer.pause = _RemoteMethod(lambda: (actions.append("pause"), _Ref("pause"))[1])
    producer.resume = _RemoteMethod(lambda: (actions.append("resume"), _Ref("resume"))[1])

    class EventRay(_FakeRay):
        def wait(self, refs, **kwargs):
            if events.items:
                return [], list(refs)
            return super().wait(refs, **kwargs)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=EventRay(producer), worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: store,
    )
    assert trainer.run() == 0
    assert actions == ["pause", "resume"]
    assert [c["plan_id"] for c in commands.items if c["kind"] == "batch"] == list(range(6))


def test_driver_logs_producer_snapshot_and_drains_final_metrics(monkeypatch):
    store = _ReadyStore()
    list_calls = []
    original_list = store.list_ready
    store.list_ready = lambda: (list_calls.append(1), original_list())[1]
    commands = _RecordingQueue()
    def on_event(event):
        if event["kind"] == "producer_metrics":
            # One initialization list before this low-frequency metric snapshot.
            assert len(list_calls) == 1
        elif event["kind"] == "training_completed":
            assert len(list_calls) == 2
            store.ready.clear()

    events = _RecordingQueue(events=[
        {"kind": "producer_metrics", "published_total": 100,
         "metrics": {"producer/samples_per_second": 2.0}},
        {"kind": "training_completed", "keys": [f"sample-{i}" for i in range(4)],
         "successful": True},
        {"kind": "training_metrics", "step": 17, "metrics": {"dspark/loss": 0.5}},
        {"kind": "training_metrics", "step": 17, "metrics": {"tokens/run_source": 120.0,
                                                                  "perf/run_checkpoint_blocking_time": 0.0}},
    ], on_event=on_event)

    class MetricsRay(_FlowRay):
        def wait(self, refs, num_returns, timeout=0):
            self.wait_calls += 1
            if self.wait_calls <= 2:
                return [], list(refs)
            producer = next((ref for ref in refs if ref.name == "producer"), refs[0])
            return [producer], [ref for ref in refs if ref != producer]

    queues = iter((commands, events))
    calls = []
    module = "verl_speco.standalone_ray_runtime."
    monkeypatch.setattr(module + "compose_runtime_config", lambda _: _runtime_config())
    monkeypatch.setattr(module + "_build_standalone_tracking", lambda *a, **k: ["tracker"])
    monkeypatch.setattr(module + "_log_producer_tracking_metrics",
                        lambda trackers, metrics, **k: calls.append(("producer", dict(metrics), k)))
    monkeypatch.setattr(module + "_log_standalone_tracking_metrics",
                        lambda trackers, metrics, **k: calls.append(("train", dict(metrics), k)))
    monkeypatch.setattr(module + "_finish_standalone_tracking",
                        lambda trackers, **k: calls.append(("finish", trackers)))
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=MetricsRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: store, pipeline_started_at=0.0,
    )
    assert trainer.run() == 0
    assert calls[:3] == [
        ("producer", {"producer/samples_per_second": 2.0, "tq/ready_samples": 4.0},
         {"published_total": 100}),
        ("train", {"dspark/loss": 0.5}, {"step": 17}),
        ("train", {"tokens/run_source": 120.0, "perf/run_checkpoint_blocking_time": 0.0}, {"step": 17}),
    ]
    assert calls[3][0] == "train" and calls[3][2] == {"step": 17}
    measured = calls[3][1]
    assert measured["perf/pipeline_active_seconds"] > 0
    assert measured["tokens/pipeline_source_per_second"] == pytest.approx(120 / measured["perf/pipeline_active_seconds"])
    assert calls[4] == ("finish", ["tracker"])


def test_pipeline_throughput_uses_driver_duration_and_validated_checkpoint_wait():
    from verl_speco.standalone_ray_runtime import _pipeline_run_metrics
    result = _pipeline_run_metrics({"tokens/run_source": 120, "perf/run_checkpoint_blocking_time": 2}, 10)
    assert result == {"perf/pipeline_active_seconds": 10, "tokens/pipeline_source_per_second": 12,
                      "tokens/pipeline_source_per_second_excluding_checkpoint_wait": 15}
    assert _pipeline_run_metrics({"tokens/run_source": float("nan")}, 10) == {}
    assert _pipeline_run_metrics({"tokens/run_source": 120}, 0) == {}
    assert "tokens/pipeline_source_per_second_excluding_checkpoint_wait" not in _pipeline_run_metrics(
        {"tokens/run_source": 120, "perf/run_checkpoint_blocking_time": 12}, 10)


_CONFIG = None


def _runtime_config():
    global _CONFIG
    if _CONFIG is None:
        training = SimpleNamespace(batch_size_per_gpu=4, transfer_queue={})
        training.get = lambda key, default=None: getattr(training, key, default)
        draft_training = SimpleNamespace(
            nproc_per_node=1,
            nnodes=1,
            scheduler={},
        )
        draft_training.get = lambda key, default=None: getattr(
            draft_training, key, default
        )
        _CONFIG = SimpleNamespace(
            speco=SimpleNamespace(
                draft_training=draft_training,
                standalone_tq_producer=SimpleNamespace(max_pending_samples=20),
            ),
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    drafter=SimpleNamespace(training=training)
                )
            ),
        )
    return _CONFIG


def test_queue_config_defaults_low_watermark_to_half_producer_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(_commands(), ray_module=SimpleNamespace())

    config = trainer._queue_config()

    assert config.global_batch_size == 4
    assert config.low_watermark_samples == 10
    assert config.high_watermark_samples == 20


def test_driver_logging_overrides_preinstalled_warning_handler(monkeypatch) -> None:
    monkeypatch.delenv("SPECO_STANDALONE_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    original_handlers = list(root.handlers)
    original_level = root.level
    module_logger = logging.getLogger("verl_speco.standalone_ray_runtime")
    original_module_level = module_logger.level
    try:
        root.handlers[:] = [handler]
        root.setLevel(logging.WARNING)

        _configure_driver_logging()

        assert root.level == logging.INFO
        assert handler.level == logging.INFO
        assert (
            logging.getLogger("verl_speco.standalone_ray_runtime").level
            == logging.INFO
        )
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        module_logger.setLevel(original_module_level)


def test_driver_logging_supports_scheduler_debug(monkeypatch) -> None:
    monkeypatch.setenv("SPECO_STANDALONE_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    handler = logging.StreamHandler()
    original_handlers = list(root.handlers)
    original_level = root.level
    module_logger = logging.getLogger("verl_speco.standalone_ray_runtime")
    original_module_level = module_logger.level
    try:
        root.handlers[:] = [handler]

        _configure_driver_logging()

        assert root.level == logging.DEBUG
        assert handler.level == logging.DEBUG
        assert module_logger.level == logging.DEBUG
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        module_logger.setLevel(original_module_level)


def test_trainer_finishes_after_producer_and_consumers_complete(monkeypatch) -> None:
    producer = _ProducerHandle()
    ray = _FakeRay(producer)
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=ray,
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    assert trainer.run() == 0
    assert producer.stopped is False


class _FailingRay(_FakeRay):
    def get(self, ref, timeout=None):
        if ref.name == "consumer-0":
            raise RuntimeError("consumer failed")
        return super().get(ref)


class _EarlyConsumerRay(_FakeRay):
    def wait(self, refs, num_returns, timeout=0):
        assert num_returns == 1
        consumer = next((ref for ref in refs if ref.name.startswith("consumer")), refs[0])
        return [consumer], [ref for ref in refs if ref != consumer]

    def get(self, ref, timeout=None):
        if ref.name.startswith("consumer"):
            return {"optimizer_steps_total": 0}
        return super().get(ref)


def test_trainer_rejects_unexpected_early_consumer_exit(monkeypatch) -> None:
    producer = _ProducerHandle()
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=_EarlyConsumerRay(producer),
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    with pytest.raises(RuntimeError, match="before receiving"):
        trainer.run()

    assert producer.stopped is True


def test_step_budget_survives_consumer_exit_before_completion_event(monkeypatch):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.step = 1
    training.get = lambda key, default=None: getattr(training, key, default)
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "training_completed", "plan_id": 0,
         "keys": [f"sample-{i}" for i in range(4)], "successful": True},
    ])
    commands = _RecordingQueue()

    class FinishedConsumerRay(_FakeRay):
        calls = 0

        def wait(self, refs, **kwargs):
            if refs[0].name == "stop" or all(ref.name.startswith("consumer") for ref in refs):
                return refs[:1], refs[1:]
            self.calls += 1
            if self.calls == 2:
                consumer = next(r for r in refs if r.name.startswith("consumer"))
                return [consumer], [r for r in refs if r != consumer]
            return [], list(refs)

        def get(self, ref, timeout=None):
            if ref.name.startswith("consumer"):
                return {"optimizer_steps_total": 1}
            return super().get(ref)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=FinishedConsumerRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: _ReadyStore(),
    )
    assert trainer.run() == 0
    assert [c["kind"] for c in commands.items] == ["batch", "stop"]


def test_trainer_propagates_consumer_error_and_stops_producer(monkeypatch) -> None:
    producer = _ProducerHandle()
    ray = _FailingRay(producer)
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=ray,
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        trainer.run()


@pytest.mark.parametrize("producer_node", ["driver-node", "other-node", None])
@pytest.mark.parametrize("end", ["eos", "max_steps"])
@pytest.mark.parametrize("shutdown_conflict", ["none", "metadata", "incarnation", "stop_failure", "stop_timeout", "discard_failure"])
@pytest.mark.parametrize("release_fails", [False, True])
def test_mooncake_driver_cleans_tail_and_late_shutdown_references(monkeypatch, end, producer_node, shutdown_conflict, release_fails):
    if release_fails and shutdown_conflict != "none":
        pytest.skip("Cleanup failure and identity conflict are tested separately")
    if shutdown_conflict not in {"none", "discard_failure"} and end == "eos":
        pytest.skip("Only max_steps has a late shutdown event")
    if shutdown_conflict == "discard_failure" and end != "eos":
        pytest.skip("Discard failure is covered by the end-of-stream tail")
    from copy import deepcopy
    from verl_speco.producer import hidden_states_store
    from verl_speco.trainer.tq_feature_store import ReferenceFeatureStore
    from verl_speco.transport.drafter_sample_protocol import (
        PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key,
    )

    config = deepcopy(_runtime_config())
    use_credits = end == "eos" and shutdown_conflict in {"none", "discard_failure"}
    if shutdown_conflict == "incarnation" or use_credits:
        config.speco.standalone_tq_producer.max_published_references = 2
    if shutdown_conflict == "stop_timeout":
        config.speco.draft_training.scheduler = {"producer_stop_timeout_seconds": 0.01}
        config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 2 if end == "eos" else 1
    training.max_steps = 0 if end == "eos" else 1
    training.transfer_queue = {"run_id": "run", "ray": {"address": "unused"}, "consumer_prefetch_batches": 1}
    training.target_feature_replay = {"hidden_states_store": {"backend": "mooncake", "mooncake": {"master_server_address": "host:50051"}}}
    deleted = []
    def release(reference):
        deleted.append(reference)
        if release_fails:
            raise OSError("injected release failure")
    backend = SimpleNamespace(backend="mooncake", release=release, close=lambda: None)
    monkeypatch.setattr(hidden_states_store, "build_hidden_states_store", lambda cfg: backend)

    def entry(index):
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(
            meta, backend="mooncake", handle=f"hs-{index}",
            request={"input_ids": [1, 2], "loss_mask": [0, 1], "position_ids": [0, 1],
                     "feature_positions": [0, 1], "draft_position_ids": [1, 2],
                     "vllm_prompt_token_ids": [1, 2], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"},
        )
        return {"key": make_sample_key(meta), "tag": tag}

    items = [{"kind": "consumer_ready", "initial_optimizer_step": 0},
             {"kind": "samples_ready", "batch_id": 0, "entries": [entry(0)]}]
    if end == "eos":
        items.append({"kind": "producer_done"})
        if use_credits:
            items[1]["producer_incarnation"] = items[2]["producer_incarnation"] = "f" * 32
        expected = ["hs-0"]
    else:
        # Budget allows only plan 0; the other references must be discarded.
        items[1]["entries"].append(entry(1))
        items.extend([
            {"kind": "training_materialized", "plan_id": 0},
            {"kind": "training_applied", "plan_id": 0},
            {"kind": "training_completed", "plan_id": 0, "keys": [entry(0)["key"]], "successful": True},
            {"kind": "samples_ready", "batch_id": 1, "entries": [entry(2)]},
        ])
        expected = ["hs-1", "hs-2"]
        if shutdown_conflict == "metadata":
            forged = entry(1)
            forged["tag"]["handle"] = "hs-foreign"
            items[-1]["entries"] = [forged]
        elif shutdown_conflict == "incarnation":
            items[1]["producer_incarnation"] = "f" * 32
            items[-1]["producer_incarnation"] = "e" * 32
    commands = _RecordingQueue()
    def receive(event):
        if event["kind"] in {"training_materialized", "training_applied", "training_completed"}:
            command = next(item for item in commands.items if item.get("plan_id") == event["plan_id"])
            event["receipt_token"] = command["receipt_token"]
    events = _RecordingQueue(events=items, on_event=receive)

    class Ray(_FakeRay):
        def get(self, ref, timeout=None):
            if ref.name == "start" and (shutdown_conflict == "incarnation" or use_credits):
                return {"started": True, "producer_incarnation": "f" * 32}
            if ref.name == "stop" and shutdown_conflict == "stop_failure":
                raise RuntimeError("injected stop failure")
            return super().get(ref)
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop":
                return ([], refs) if shutdown_conflict == "stop_timeout" else (refs[:1], refs[1:])
            if events.items:
                return [], refs
            consumers = [ref for ref in refs if ref.name.startswith("consumer")]
            return consumers[:1], [ref for ref in refs if ref not in consumers[:1]]

    import time
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.current_ray_node_id", lambda: "driver-node")
    items[1]["clock_node_id"] = producer_node
    items[1]["sent_at"] = time.monotonic() - 1 if producer_node == "driver-node" else 10000.0
    configs = []
    def factory(cfg):
        configs.append(cfg)
        return ReferenceFeatureStore.from_config(cfg)

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=Ray(_ProducerHandle()), worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda **kw: next(queues), feature_store_factory=factory,
    )
    trainer.producer = trainer.ray.producer
    trainer.producer.register_dispatch = SimpleNamespace(remote=lambda *args: _Ref("register"))
    trainer.producer.release_references = SimpleNamespace(remote=lambda *args: _Ref("release"))
    discarded = []
    def discard(incarnation, token, keys):
        assert incarnation == "f" * 32 and token and not deleted
        if shutdown_conflict == "discard_failure":
            raise RuntimeError("injected discard failure")
        discarded.extend(keys)
        return _Ref("discard")
    trainer.producer.discard_unassigned_references = SimpleNamespace(remote=discard)
    trainer.consumer_wg = _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    if release_fails:
        with pytest.raises(RuntimeError, match="Failed to release .* references"):
            trainer.run()
        assert deleted
        return
    if shutdown_conflict != "none":
        error = {"metadata": "conflicting reference metadata", "incarnation": "stale producer incarnation",
                 "stop_failure": "Failed to stop standalone Producer safely",
                 "stop_timeout": "Failed to stop standalone Producer safely",
                 "discard_failure": "injected discard failure"}[shutdown_conflict]
        with pytest.raises(RuntimeError if shutdown_conflict in {"stop_failure", "stop_timeout", "discard_failure"} else ValueError, match=error):
            trainer.run()
        assert deleted == []
        return
    assert trainer.run() == 0
    if use_credits:
        assert discarded == [entry(0)["key"]]
    assert configs[0]["hidden_states_store"]["mooncake"]["master_server_address"] == "host:50051"
    assert sorted(deleted) == expected
    batches = [command for command in commands.items if command["kind"] == "batch"]
    if end == "max_steps":
        assert len(batches) == 1
        command = batches[0]
        assert command["clock_node_id"] == "driver-node"
        assert command["delivery_times"]["scheduler_ready_wait_time"] >= 0
        if producer_node == "driver-node":
            assert 1 <= command["delivery_times"]["metadata_event_age_time"] < 10
        else:
            assert "metadata_event_age_time" not in command["delivery_times"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "5"])
def test_producer_stop_timeout_is_validated_before_workers(value):
    from copy import deepcopy
    config = deepcopy(_runtime_config())
    config.speco.draft_training.scheduler = {"producer_stop_timeout_seconds": value}
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    trainer = StandaloneRayTrainer.__new__(StandaloneRayTrainer)
    trainer.consumer_config = config
    with pytest.raises(ValueError, match="producer_stop_timeout_seconds"):
        trainer.run()


def test_rank_token_balance_validates_configuration_before_workers(monkeypatch):
    from copy import deepcopy
    from verl_speco.standalone_ray_runtime import _balance_assignments_by_tokens
    config = deepcopy(_runtime_config())
    # The lightweight test namespace binds get methods to its original instance.
    config.speco.draft_training.get = lambda key, default=None: getattr(config.speco.draft_training, key, default)
    config.speco.draft_training.scheduler = {"balance_by_tokens": True}
    with pytest.raises(ValueError, match="requires ray_reference"):
        _balance_assignments_by_tokens(config)
    config.actor_rollout_ref.rollout.drafter.training.feature_store = SimpleNamespace(type="ray_reference")
    assert _balance_assignments_by_tokens(config)
    config.speco.draft_training.scheduler = {"balance_by_tokens": "true"}
    with pytest.raises(ValueError, match="must be a boolean"):
        _balance_assignments_by_tokens(config)


def test_disk_reference_identity_ledger_rejects_conflicts_and_enforces_capacity():
    from copy import deepcopy
    from verl_speco.standalone_ray_runtime import _ReferenceIdentityLedger, _accept_reference_entries, _release_reference_entries
    from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key
    def entry(index):
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(
            meta, backend="mooncake", handle=f"h{index}",
            request={"input_ids": [1], "loss_mask": [1], "position_ids": [0],
                     "feature_positions": [0], "draft_position_ids": [0],
                     "vllm_prompt_token_ids": [1], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"},
        )
        return {"key": make_sample_key(meta), "tag": tag}
    ledger = _ReferenceIdentityLedger(max_entries=2)
    from pathlib import Path
    directory = Path(ledger.directory.name)
    references = {}
    try:
        assert (directory / "identities.sqlite").is_file()
        assert ledger.db.execute("PRAGMA cache_size").fetchone()[0] == -2048
        first = entry(0)
        accepted = _accept_reference_entries([first], run_id="run", seen_keys=ledger, seen_references=references)
        _release_reference_entries(accepted, references)
        assert not _accept_reference_entries([deepcopy(first)], run_id="run", seen_keys=ledger, seen_references=references)
        altered = deepcopy(first)
        altered["tag"]["request"]["source_metadata"]["version"] = 2
        with pytest.raises(ValueError, match="conflicting"):
            _accept_reference_entries([altered], run_id="run", seen_keys=ledger, seen_references=references)
        with pytest.raises(ValueError, match="entry limit"):
            _accept_reference_entries([entry(1), entry(2)], run_id="run", seen_keys=ledger, seen_references=references)
        assert ledger.count == 1 and not references
        assert _accept_reference_entries([entry(1)], run_id="run", seen_keys=ledger, seen_references=references)
        assert ledger.count == 2
        with pytest.raises(ValueError, match="must not be empty"):
            ledger.record_batch(2, [])
        assert ledger.record_batch(0, [first])
        assert not ledger.record_batch(0, [deepcopy(first)])
        with pytest.raises(ValueError, match="conflicting contents"):
            ledger.record_batch(0, [altered])
        assert ledger.record_batch(1, [entry(1)])
    finally:
        ledger.close()
    assert not directory.exists()


@pytest.mark.parametrize("conflict", [False, True])
def test_driver_validates_retried_metadata_batch_identity(monkeypatch, conflict):
    from copy import deepcopy
    from pathlib import Path
    import verl_speco.standalone_ray_runtime as runtime
    from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key
    config = deepcopy(_runtime_config())
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.transfer_queue = {"consumer_prefetch_batches": 2}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    directories = []
    original_ledger = runtime._ReferenceIdentityLedger
    class Ledger(original_ledger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            directories.append(Path(self.directory.name))
    monkeypatch.setattr(runtime, "_ReferenceIdentityLedger", Ledger)
    meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", "sample", 0)
    tag = make_reference_tag(meta, backend="mooncake", handle="h0",
        request={"input_ids": [1], "loss_mask": [1], "position_ids": [0],
                 "feature_positions": [0], "draft_position_ids": [0],
                 "vllm_prompt_token_ids": [1], "source_metadata": {}},
        feature_contract={"algorithm": "DSPARK"})
    first = {"kind": "samples_ready", "batch_id": 0, "entries": [{"key": make_sample_key(meta), "tag": tag}]}
    repeated = deepcopy(first)
    if conflict:
        repeated["entries"][0]["tag"]["request"]["source_metadata"]["version"] = 2
    events = _RecordingQueue(events=[first, repeated, {"kind": "consumer_failed", "rank": 0, "error": "end test"}])
    class Ray(_FakeRay):
        def wait(self, refs, **kwargs):
            return (refs[:1], refs[1:]) if refs[0].name == "stop" else ([], refs)
    class Store(_Store):
        run_id = "run"
    ray = Ray(_ProducerHandle())
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray,
        worker_group_factory=lambda *a, **kw: _WorkerGroup(),
        queue_factory=_RecordingQueue, feature_store_factory=lambda _: Store())
    trainer.producer, trainer.consumer_wg = ray.producer, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = _RecordingQueue(), events
    with pytest.raises((ValueError, RuntimeError), match="conflicting contents" if conflict else "end test"):
        trainer.run()
    assert len(directories) == 1 and not directories[0].exists()


@pytest.mark.parametrize("tail_count,algorithm,ack_mode", [
    (1, "DSPARK", "valid"), (1, "DFLASH", "valid"), (1, "EAGLE3", "valid"),
    (1, "UNKNOWN", "valid"), (3, "DSPARK", "valid"),
    (1, "DSPARK", "missing"), (1, "DSPARK", "stale"),
    (1, "DSPARK", "missing_applied"), (1, "DSPARK", "stale_applied"),
])
def test_driver_partial_tail_trains_each_sample_or_preserves_too_small_tail(monkeypatch, tail_count, algorithm, ack_mode):
    from copy import deepcopy
    from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key
    config = deepcopy(_runtime_config())
    runtime = config.speco.draft_training
    runtime.nnodes = 2
    runtime.scheduler = {"tail_policy": "partial"}
    runtime.get = lambda key, default=None: getattr(runtime, key, default)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    config.actor_rollout_ref.rollout.drafter.speculative_algorithm = algorithm
    training.batch_size_per_gpu = 2
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    def entry(index):
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(meta, backend="mooncake", handle=f"h{index}",
            request={"input_ids": [1], "loss_mask": [1], "position_ids": [0],
                     "feature_positions": [0], "draft_position_ids": [0],
                     "vllm_prompt_token_ids": [1], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"})
        return {"key": make_sample_key(meta), "tag": tag}
    entries = [entry(i) for i in range(tail_count)]
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        *[{"kind": "samples_ready", "batch_id": index, "entries": [entry]}
          for index, entry in enumerate(entries)],
        {"kind": "producer_done"},
    ])
    class Commands(_RecordingQueue):
        def put(self, command):
            super().put(command)
            if command["kind"] == "batch":
                if ack_mode != "missing":
                    events.put({"kind": "training_materialized", "plan_id": command["plan_id"],
                                "receipt_token": "stale" if ack_mode == "stale" else command["receipt_token"]})
                if ack_mode in {"valid", "stale_applied"}:
                    events.put({"kind": "training_applied", "plan_id": command["plan_id"],
                                "receipt_token": "stale" if ack_mode == "stale_applied" else command["receipt_token"]})
                events.put({"kind": "training_completed", "plan_id": command["plan_id"],
                            "receipt_token": command["receipt_token"],
                            "keys": command["global_keys"], "successful": True})
    class Ray(_FakeRay):
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop":
                return refs[:1], refs[1:]
            if events.items:
                return [], refs
            return refs[:1], refs[1:]
    class Store(_Store):
        run_id = "run"
    commands = Commands()
    ray = Ray(_ProducerHandle())
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray,
        worker_group_factory=lambda *a, **kw: _WorkerGroup(),
        queue_factory=_RecordingQueue, feature_store_factory=lambda _: Store())
    trainer.producer, trainer.consumer_wg = ray.producer, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    if tail_count < 2 and algorithm == "UNKNOWN":
        with pytest.raises(RuntimeError, match="empty-rank training is unsupported"):
            trainer.run()
        assert not [cmd for cmd in commands.items if cmd["kind"] == "batch"]
    elif ack_mode != "valid":
        expected = "optimizer acknowledgement" if ack_mode.endswith("applied") else "materialization acknowledgement"
        with pytest.raises(RuntimeError, match=expected):
            trainer.run()
    else:
        assert trainer.run() == 0
        batches = [cmd for cmd in commands.items if cmd["kind"] == "batch"]
        assert len(batches) == 1
        assert [len(group) for group in batches[0]["assignments"]] == ([1, 0] if tail_count == 1 else [2, 1])
        assert batches[0]["global_keys"] == [entry["key"] for entry in entries]
        if tail_count == 1:
            assert batches[0]["padding_ranks"] == [1]
            assert batches[0]["padding_reference"] == entries[0]


@pytest.mark.parametrize("mode,early_release", [(mode, early) for mode in ("valid", "missing", "stale", "out_of_range", "optimizer_failure")
                                               for early in (False, True)] + [("unsafe_copy", True), ("cleanup_failure", True)])
@pytest.mark.parametrize("parent_commit", [False, True])
def test_driver_verifies_direct_rank_receipts_before_credit_release(monkeypatch, mode, parent_commit, early_release):
    from copy import deepcopy
    from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key
    config = deepcopy(_runtime_config())
    runtime = config.speco.draft_training
    runtime.nproc_per_node = 2
    runtime.scheduler = {"per_rank_reference_ack": True, "release_payload_on_materialization": early_release}
    observed = []
    if mode == "optimizer_failure":
        def reject_commit(*args, **kwargs):
            pytest.fail("Failed optimizer must not commit scheduler progress")
        monkeypatch.setattr("verl_speco.standalone_ray_runtime.DrafterScheduler.complete_standalone_training", reject_commit)
    if early_release:
        monkeypatch.setattr("verl_speco.standalone_ray_runtime.cleanup_standalone_samples",
                            lambda *a, **k: ["unreleased"] if mode == "cleanup_failure" and k.get("reason") == "materialized" else [])
    runtime.get = lambda key, default=None: getattr(runtime, key, default)
    config.speco.standalone_tq_producer.max_published_references = 0 if parent_commit else 2
    config.speco.standalone_tq_producer.publication_after_parent_commit = parent_commit
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 1
    training.max_steps = 1
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    def entry(index):
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(meta, backend="mooncake", handle=f"h{index}",
            request={"input_ids": [1], "loss_mask": [1], "position_ids": [0],
                     "feature_positions": [0], "draft_position_ids": [0],
                     "vllm_prompt_token_ids": [1], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"})
        return {"key": make_sample_key(meta), "tag": tag}
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "samples_ready", "batch_id": 0, "entries": [entry(0), entry(1)],
         "producer_incarnation": "f" * 32},
        {"kind": "producer_done", "producer_incarnation": "f" * 32},
    ], on_event=lambda event: observed.append(event["kind"]))
    class Commands(_RecordingQueue):
        def put(self, command):
            super().put(command)
            if command["kind"] != "batch":
                return
            pid, token = command["plan_id"], command["receipt_token"]
            source.acknowledge_reference_rank.remote(command["producer_incarnation"], pid, token, 0)
            event = {"kind": "reference_rank_materialized", "plan_id": pid, "receipt_token": token, "rank": 0}
            if early_release and mode != "unsafe_copy":
                event["source_payload_safe"] = True
            events.put(event)
            events.put(dict(event))  # An ambiguous queue retry is idempotent.
            if mode != "missing":
                if mode in {"valid", "unsafe_copy", "cleanup_failure", "optimizer_failure"}:
                    source.acknowledge_reference_rank.remote(command["producer_incarnation"], pid, token, 1)
                events.put({"kind": "reference_rank_materialized", "plan_id": pid,
                            "receipt_token": "stale" if mode == "stale" else token,
                            "rank": 2 if mode == "out_of_range" else 1,
                            **({"source_payload_safe": True} if early_release else {})})
            events.put({"kind": "training_materialized", "plan_id": pid, "receipt_token": token,
                        **({"source_payload_safe": True} if early_release else {})})
            if mode == "optimizer_failure":
                events.put({"kind": "consumer_failed", "rank": 0, "error": "injected optimizer failure"})
                return
            events.put({"kind": "training_applied", "plan_id": pid, "receipt_token": token})
            events.put({"kind": "training_completed", "plan_id": pid, "receipt_token": token,
                        "keys": command["global_keys"], "successful": True})
    class Ray(_FakeRay):
        def get(self, ref, timeout=None):
            return {"started": True, "producer_incarnation": "f" * 32} if ref.name == "start" else super().get(ref, timeout)
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop": return refs[:1], refs[1:]
            if events.items: return [], refs
            return refs[:1], refs[1:]
    class Store(_Store): run_id = "run"
    source, received, released = _ProducerHandle(), [], []
    def register(inc, plan, token, keys, count):
        assert inc == "f" * 32 and count == 2 and len(keys) == 2
        return _Ref("registered")
    def acknowledge(inc, plan, token, rank):
        received.append(rank)
        return _Ref("ack")
    def release(inc, plan, token, keys):
        assert set(received) == {0, 1}
        assert observed[-1] == ("training_materialized" if early_release else "training_completed")
        released.append(keys)
        return _Ref("released")
    source.register_dispatch = SimpleNamespace(remote=register)
    source.acknowledge_reference_rank = SimpleNamespace(remote=acknowledge)
    source.release_references = SimpleNamespace(remote=release)
    ray = Ray(source)
    trainer = StandaloneRayTrainer(_commands(), ray_module=ray, feature_store_factory=lambda _: Store())
    trainer.producer, trainer.consumer_wg = source, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = Commands(), events
    if mode == "valid":
        assert trainer.run() == 0
        assert received == [0, 1] and len(released) == 1
    else:
        with pytest.raises(RuntimeError, match={"missing": "every rank", "stale": "does not match",
                                                "out_of_range": "Invalid rank", "unsafe_copy": "independent-copy",
                                                "cleanup_failure": "cleanup failed", "optimizer_failure": "injected optimizer failure"}[mode]):
            trainer.run()
        assert len(released) == int(mode == "optimizer_failure" and early_release)


@pytest.mark.parametrize("successful", [False, None, "true"])
def test_driver_rejects_untrusted_training_completion(monkeypatch, successful):
    import copy

    config = copy.deepcopy(_runtime_config())
    monkeypatch.setattr(sys.modules[__name__], "_CONFIG", config)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    commands = _RecordingQueue()
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "training_completed", "plan_id": 0,
         "keys": [f"sample-{i}" for i in range(4)], "successful": successful},
    ])

    class EventRay(_FakeRay):
        def wait(self, refs, **kwargs):
            return (refs[:1], refs[1:]) if refs[0].name == "stop" else ([], list(refs))

    queues = iter((commands, events))
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(
        _commands(), ray_module=EventRay(_ProducerHandle()),
        worker_group_factory=lambda *a, **k: _WorkerGroup(),
        queue_factory=lambda: next(queues), feature_store_factory=lambda _: _ReadyStore(),
    )
    with pytest.raises(RuntimeError, match="Invalid or unsuccessful training completion"):
        trainer.run()


def test_reference_metadata_is_split_by_bytes_and_retried_in_order(monkeypatch):
    import json
    from verl_speco.standalone_ray_runtime import _bounded_metadata_size

    class Queue:
        def __init__(self):
            self.items = []
            self.failed = False
        def put(self, event):
            self.items.append(event)
            if event["batch_id"] == 1 and not self.failed:
                self.failed = True
                raise ConnectionError("ack lost")
    queue = Queue()
    config = {"speco": {"standalone_tq_producer": {"reference_event_max_bytes": 1024}}}
    actor = StandaloneProducerActor(config, queue, reference_batch_size=20)
    entries = [{"key": str(i), "tag": {"source": "中" * 80}} for i in range(5)]
    metrics = asyncio.run(actor._on_references(entries))
    assert metrics["attempts"] > 1
    events = queue.items
    assert [event["batch_id"] for event in events] == [0, 1, 1, 2, 3, 4]
    assert events[1] == events[2]
    delivered = [event for i, event in enumerate(events) if i != 2]
    assert [entry for event in delivered for entry in event["entries"]] == entries
    assert all(_bounded_metadata_size(event, 1024) == len(json.dumps(
        event, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
               for event in delivered)


def test_oversized_reference_entry_fails_before_any_event():
    class Queue:
        def put(self, event):
            pytest.fail("no fragment should be sent when any entry is too large")
    config = {"speco": {"standalone_tq_producer": {"reference_event_max_bytes": 1024}}}
    actor = StandaloneProducerActor(config, Queue())
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(actor._on_references([{"key": "small", "tag": {}},
                                          {"key": "large", "tag": {"x": "x" * 2000}}]))
    assert actor._reference_batch_id == 0


def test_driver_accepts_next_materialization_before_previous_completion(monkeypatch):
    from copy import deepcopy
    from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key

    config = deepcopy(_runtime_config())
    runtime = config.speco.draft_training
    runtime.get = lambda key, default=None: getattr(runtime, key, default)
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 1
    training.max_steps = 2
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    def entry(index):
        metadata = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index)
        tag = make_reference_tag(metadata, backend="mooncake", handle=f"h{index}",
            request={"input_ids": [1], "loss_mask": [1], "position_ids": [0],
                     "feature_positions": [0], "draft_position_ids": [0],
                     "vllm_prompt_token_ids": [1], "source_metadata": {}},
            feature_contract={"algorithm": "DSPARK"})
        return {"key": make_sample_key(metadata), "tag": tag}
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "samples_ready", "batch_id": 0, "entries": [entry(0), entry(1)]},
        {"kind": "producer_done"},
    ])
    class Commands(_RecordingQueue):
        def put(self, item):
            super().put(item)
            batches = [value for value in self.items if value["kind"] == "batch"]
            if item["kind"] == "batch" and len(batches) == 2:
                for command in batches:
                    events.put({"kind": "training_materialized", "plan_id": command["plan_id"],
                                "receipt_token": command["receipt_token"]})
                for command in batches:
                    events.put({"kind": "training_applied", "plan_id": command["plan_id"],
                                "receipt_token": command["receipt_token"]})
                    events.put({"kind": "training_completed", "plan_id": command["plan_id"],
                                "receipt_token": command["receipt_token"],
                                "keys": command["global_keys"], "successful": True})
    class Ray(_FakeRay):
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop":
                return refs[:1], refs[1:]
            if events.items:
                return [], refs
            return refs[:1], refs[1:]
    class Store(_Store):
        run_id = "run"
    commands = Commands()
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=Ray(_ProducerHandle()),
        worker_group_factory=lambda *a, **kw: _WorkerGroup(),
        queue_factory=_RecordingQueue, feature_store_factory=lambda _: Store())
    trainer.producer, trainer.consumer_wg = trainer.ray.producer, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    assert trainer.run() == 0
    assert [item["plan_id"] for item in commands.items if item["kind"] == "batch"] == [0, 1]


def _live_topology_gloo_worker(rank, port, bad, result_queue):
    import datetime
    import ray
    import torch.distributed as dist
    import verl_speco.trainer.draft_training_loop as loop
    from types import SimpleNamespace
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=25))
    def node_id():
        if bad == "lookup" and rank == 1:
            raise RuntimeError("injected Ray context failure")
        if bad == "owner_same":
            return "node-shared"
        return f"node-{rank}"
    ray.get_runtime_context = lambda: SimpleNamespace(
        get_node_id=node_id, get_accelerator_ids=lambda: {"GPU": ["0"]})
    ray.util.get_node_ip_address = lambda: f"10.0.0.{1 if bad == 'placement' else rank + 1}"
    loop.get_device_name = lambda: "cuda"
    changed = bad == "config" and rank == 1
    single_node = changed or bad in {"owner_same", "owner_split"}
    config = SimpleNamespace(speco=SimpleNamespace(draft_training=SimpleNamespace(
        nnodes=1 if single_node else 2, nproc_per_node=2 if single_node else 1,
        trainer_node_ips=None if single_node else ["10.0.0.1", "10.0.0.2"])),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(
            training={"transfer_queue": {"consumer_device_batch_bytes": 1 if bad == "budget" and rank == 1 else 0,
                                         "consumer_node_reference_owner": bad in {"owner_same", "owner_split"}}}))))
    try:
        try:
            loop._verify_live_trainer_topology(config, rank=rank, world_size=2, scheduled=True)
            result_queue.put((rank, "ok"))
        except ValueError as exc:
            result_queue.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("bad", ["none", "placement", "config", "lookup", "budget", "owner_same", "owner_split"])
def test_live_trainer_topology_gloo_agrees_before_model_loading(bad):
    import multiprocessing
    import socket
    import torch.distributed as dist
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_live_topology_gloo_worker, args=(rank, port, bad, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=20)
        assert all(process.exitcode == 0 for process in processes)
        assert {item[0] for item in results} == {0, 1}
        expected = {"placement": "differs from the plan", "config": "disagree on the configured topology",
                    "lookup": "Cannot verify live trainer placement", "budget": "disagree on the configured topology",
                    "owner_split": "must share one verified Ray node"}.get(bad, "ok")
        assert all(expected in item[1] for item in results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
