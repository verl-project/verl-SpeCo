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

torch = pytest.importorskip("torch")


class _NativeTestEngine:
    def initialize(self, *args): return 0
    def register_memory(self, *args): return 0
    def unregister_memory(self, *args): return 0


@pytest.mark.parametrize("fault", [None, "transfer", "stale", "payload"])
def test_native_transport_leases_and_copy_lifecycle(fault, monkeypatch):
    import ctypes
    from copy import deepcopy
    from verl_speco.transport import native_feature_transport as native
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport

    monkeypatch.setattr(native, "_REGISTERED_OWNERS", set())
    registrations, reads = {}, []
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, address, size):
            registrations[address] = size
            return 0
        def unregister_memory(self, address):
            del registrations[address]
            return 0
        def transfer_sync_read(self, segment, local, remote, size):
            assert all(any(base <= address and address + size <= base + capacity
                           for base, capacity in registrations.items()) for address in (local, remote))
            reads.append(size)
            if fault == "transfer":
                return 1
            if fault == "stale" and len(reads) == 2:
                ctypes.memset(remote, 0, 64)
            ctypes.memmove(local, remote, size)
            if fault == "payload" and len(reads) == 1:
                ctypes.memset(local + 256, 255, 1)
            return 0
    def make():
        return NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                      metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                      max_slots=2, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    source, receiver = make(), make()
    values = {"hidden": torch.arange(12).reshape(3, 4).bfloat16(), "scalar": torch.tensor(3.0)}
    descriptor = source.publish(values, lease_id="lease", readers=[0, 1])
    with pytest.raises(ValueError, match="already active"):
        source.publish(values, lease_id="lease", readers=[0])
    with pytest.raises(ValueError, match="slot or byte budget"):
        source.publish(values, lease_id="second", readers=[0])
    with pytest.raises(RuntimeError, match="unacknowledged"):
        source.release(descriptor)
    with pytest.raises(RuntimeError, match="active source leases"):
        source.close()
    with pytest.raises(ValueError, match="reader differs"):
        source.acknowledge(descriptor, reader=2)
    changed = deepcopy(descriptor)
    changed["generation"] += 1
    with pytest.raises(ValueError, match="descriptor or scope"):
        receiver.read(changed)
    assert reads == []
    if fault:
        with pytest.raises(RuntimeError, match="failed|lease changed|checksum differs"):
            receiver.read(descriptor)
        assert receiver.metrics() == {}
    else:
        result = receiver.read(descriptor, output_device="cpu")
        assert all(torch.equal(result[name], value) for name, value in values.items())
        metrics = receiver.metrics()
        assert metrics["native/reads_total"] == 1
        assert metrics["native/transfer_bytes_total"] == descriptor["nbytes"] + 64
        stages = ("register", "transfer", "checksum", "materialize", "lease_verify", "cleanup")
        assert all(metrics[f"native/{stage}_seconds_total"] >= 0 for stage in stages)
        assert sum(metrics[f"native/{stage}_seconds_total"] for stage in stages) == pytest.approx(metrics["native/read_seconds_total"])
    assert len(registrations) == (2 if fault == "transfer" else 1)
    if fault:
        with pytest.raises(RuntimeError, match="unacknowledged"):
            source.release(descriptor)
        with pytest.raises(RuntimeError, match="active source leases"):
            source.close()
        if fault == "transfer":
            with pytest.raises(RuntimeError, match="process fencing"):
                receiver.close()
        else:
            receiver.close()
        return
    source.acknowledge(descriptor, reader=0)
    source.acknowledge(descriptor, reader=0)  # Idempotent until release.
    with pytest.raises(RuntimeError, match="unacknowledged"):
        source.release(descriptor)
    source.acknowledge(descriptor, reader=1)
    source.release(descriptor)
    source.acknowledge(descriptor, reader=0)
    source.release(descriptor)
    with pytest.raises(ValueError, match="acknowledgement identity"):
        source.acknowledge(descriptor, reader=2)
    with pytest.raises(ValueError, match="already completed"):
        source.publish(values, lease_id="lease", readers=[0])
    assert not registrations
    assert all(torch.equal(result[name], value) for name, value in values.items())
    source.close()
    receiver.close()
    assert not native._REGISTERED_OWNERS


def test_native_producer_store_resolves_handles_and_discards_unpublished():
    import json
    from verl_speco.transport.native_feature_transport import NativeProducerReferenceStore
    descriptor = {"version": 2, "lease_id": "123", "segment": "source", "sha256": "a" * 64}
    calls = []
    class Router:
        def resolve(self, handle):
            calls.append(("resolve", handle))
            return descriptor
        def discard_unbound(self, value): calls.append(("discard", value))
    store = NativeProducerReferenceStore(Router())
    canonical = store.resolve_reference("123")
    assert json.loads(canonical) == descriptor
    store.release(canonical)
    store.release("123")  # A JSON scalar is an opaque handle, not a descriptor.
    assert calls == [("resolve", "123"), ("discard", descriptor),
                     ("resolve", "123"), ("discard", descriptor)]


def test_native_source_freeze_and_seal_blocks_late_publications():
    import asyncio
    from queue import Queue
    from verl_speco.transport.native_feature_transport import (
        NativeFeatureTransport, NativeSourceControlClient, NativeSourceControlService, NativeSourceDirectory,
    )


    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source", metadata_server="metadata",
                                    device="cpu", max_bytes=4 * 1024**2, max_slots=1,
                                    chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    options = {"request_queue": Queue(), "response_queue": Queue(), "channel_id": "a" * 32, "timeout": 1.0}
    directory = NativeSourceDirectory(run_id="run", generation=1, max_sources=1, max_records=2)
    service = NativeSourceControlService(source, **options)
    client = NativeSourceControlClient(**options)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="first")

    async def freeze():
        await directory.register_source("source", "b" * 32, options)
        await directory.publish("first", descriptor)
        status = await directory.status(1)
        assert status["active"] == status["records"] == status["sources"] == 1
        assert status["active_sample"][0]["handle"] == "first"
        assert await directory.freeze(False) == ["source"]
        assert (await directory.status())["active"] == 1
        assert (await directory.status())["frozen"] is True
        with pytest.raises(ValueError, match="status limit"):
            await directory.status(65)
        waiter = asyncio.create_task(directory.resolve("missing", 1.0))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="active references"):
            await directory.freeze()
        with pytest.raises(ValueError, match="closed or retired"):
            await waiter
        with pytest.raises(RuntimeError, match="frozen"):
            await directory.register_source("source", "b" * 32, options)
        await directory.retire("first", descriptor["sha256"])
        assert (await directory.status())["active"] == 0
        class NoScan(dict):
            def values(self): pytest.fail("freeze scanned retired handles")
        directory._records = NoScan(directory._records)
        assert await directory.freeze() == ["source"]
        with pytest.raises(ValueError, match="not registered"):
            await directory.publish("first", descriptor)
    asyncio.run(freeze())
    client.stop_publication()
    source.validate_published(descriptor)
    assert source._registered
    with pytest.raises(RuntimeError, match="sealed"):
        source.publish({"ids": torch.arange(2)}, lease_id="before-retirement")
    source.discard_unbound(descriptor)
    client.seal_if_empty()
    with pytest.raises(RuntimeError, match="sealed"):
        source.publish({"ids": torch.arange(2)}, lease_id="late")
    service.close()
    source.close()


def test_native_admission_stop_attempts_all_sources_after_control_failure(monkeypatch):
    from threading import Barrier
    from types import SimpleNamespace
    from verl_speco.transport import native_feature_transport as native
    from verl_speco.transport.native_feature_transport import NativeSourceDirectoryRouter

    router = NativeSourceDirectoryRouter(object(), timeout=1)
    calls = []
    resolving = Barrier(2)
    def directory_call(method, *arguments):
        if method == "freeze":
            assert arguments == (False,)
            return ["first", "second"]
        assert method == "source_route"
        resolving.wait(timeout=1)
        return {"segment": arguments[0]}
    def client(descriptor):
        def stop():
            calls.append(descriptor["segment"])
            if descriptor["segment"] == "first":
                raise TimeoutError("unconfirmed")
        return SimpleNamespace(stop_publication=stop)
    monkeypatch.setattr(router, "_directory_call", directory_call)
    monkeypatch.setattr(native, "NativeSourceControlClient", lambda **options: client(options))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unconfirmed for 1 sources"):
            router.stop_admission()
    assert sorted(calls) == ["first", "second"]


def test_native_directory_checks_each_source_configuration():
    import asyncio
    from queue import Queue
    from verl_speco.transport.native_feature_transport import NativeSourceDirectory
    expected = {"max_bytes": 100, "max_slots": 4, "chunk_bytes": 8,
                "metadata_server": "metadata", "reader_endpoints": ["reader"]}
    directory = NativeSourceDirectory(run_id="run", generation=1, max_sources=1, max_records=1,
                                      source_requirements=expected)
    options = {"request_queue": Queue(), "response_queue": Queue(), "channel_id": "a" * 32, "timeout": 1.0}
    settings = {key: value for key, value in expected.items() if key != "reader_endpoints"}
    async def run():
        for invalid in ({**settings, "max_bytes": 99}, {**settings, "max_slots": 3},
                        {**settings, "chunk_bytes": 16}, {**settings, "metadata_server": "other"}):
            with pytest.raises(ValueError, match="directory requirements"):
                await directory.register_source("source", "b" * 32, options, invalid)
        with pytest.raises(ValueError, match="directory requirements"):
            await directory.register_source("reader", "b" * 32, options, settings)
        await directory.register_source("source", "b" * 32, options, settings)
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["publish", "retire"])
def test_native_directory_mutation_failure_retains_active_fence(phase):
    import asyncio
    from queue import Queue
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport, NativeSourceDirectory
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source", metadata_server="metadata",
                                    device="cpu", max_bytes=4 * 1024**2, max_slots=1,
                                    chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="sample")
    directory = NativeSourceDirectory(run_id="run", generation=1, max_sources=1, max_records=2)
    options = {"request_queue": Queue(), "response_queue": Queue(), "channel_id": "a" * 32, "timeout": 1.0}
    class FailUpdate(dict):
        def __setitem__(self, key, value): raise MemoryError("injected index failure")
    async def run():
        await directory.register_source("source", "b" * 32, options)
        if phase == "retire":
            await directory.publish("sample", descriptor)
        directory._records = FailUpdate(directory._records)
        with pytest.raises(MemoryError, match="index failure"):
            if phase == "publish":
                await directory.publish("sample", descriptor)
            else:
                await directory.retire("sample", descriptor["sha256"])
        with pytest.raises(RuntimeError, match="active references"):
            await directory.freeze()
    try:
        asyncio.run(run())
    finally:
        source.discard_unbound(descriptor)
        source.close()


def test_native_source_seal_with_live_memory_fails_closed():
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source", metadata_server="metadata",
                                    device="cpu", max_bytes=4 * 1024**2, max_slots=1,
                                    chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="first")
    with pytest.raises(RuntimeError, match="registered memory"):
        source.seal_if_empty()
    with pytest.raises(RuntimeError, match="sealed"):
        source.publish({"ids": torch.arange(2)}, lease_id="late")
    source.discard_unbound(descriptor)
    source.seal_if_empty()
    source.close()


def test_native_source_directory_binds_and_retires_handles():
    import asyncio
    from queue import Queue
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport, NativeSourceDirectory
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source", metadata_server="metadata",
                                    device="cpu", max_bytes=8 * 1024**2, max_slots=2,
                                    chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    first = source.publish({"ids": torch.arange(2)}, lease_id="first")
    second = source.publish({"ids": torch.arange(2)}, lease_id="second")
    directory = NativeSourceDirectory(run_id="run", generation=1, max_sources=1, max_records=1)
    options = {"request_queue": Queue(), "response_queue": Queue(), "channel_id": "a" * 32, "timeout": 1.0}
    async def run():
        with pytest.raises(ValueError, match="not registered"):
            await directory.publish("first", first)
        await directory.register_source("source", "b" * 32, options)
        await directory.register_source("source", "b" * 32, options)
        with pytest.raises(ValueError, match="another incarnation"):
            await directory.register_source("source", "c" * 32, options)
        assert (await directory.source_route("source"))["channel_id"] == "a" * 32
        waiter = asyncio.create_task(directory.resolve("first", 1.0))
        await asyncio.sleep(0)
        await directory.publish("first", first)
        await directory.publish("first", first)
        assert await waiter == first
        assert directory._descriptor_bytes > 0
        limited = NativeSourceDirectory(run_id="run", generation=1, max_sources=1, max_records=2,
                                        max_descriptor_bytes=1)
        await limited.register_source("source", "b" * 32, options)
        with pytest.raises(ValueError, match="descriptor byte limit"):
            await limited.publish("first", first)
        await limited.close()
        with pytest.raises(ValueError, match="record limit"):
            await directory.publish("second", second)
        with pytest.raises(ValueError, match="retirement identity"):
            await directory.retire("first", "0" * 64)
        await directory.retire("first", first["sha256"])
        await directory.retire("first", first["sha256"])
        assert directory._descriptor_bytes == 0
        with pytest.raises(ValueError, match="retired"):
            await directory.resolve("first", 1.0)
        with pytest.raises(ValueError, match="reused or retired"):
            await directory.publish("first", first)
        waiting = asyncio.create_task(directory.resolve("missing", 1.0))
        await asyncio.sleep(0)
        await directory.close()
        with pytest.raises(ValueError, match="closed"):
            await waiting
    asyncio.run(run())
    for descriptor in (first, second):
        source.discard_unbound(descriptor)
    source.close()


def test_native_source_control_routes_and_timeout():
    from queue import Queue
    from verl_speco.transport.native_feature_transport import (
        NativeFeatureTransport, NativeSourceControlClient, NativeSourceControlRouter, NativeSourceControlService,
    )
    routes, sources, services, descriptors = {}, [], [], []
    for index in range(2):
        endpoint = f"source-{index}"
        source = NativeFeatureTransport(run_id="run", generation=1, endpoint=endpoint, metadata_server="metadata",
                                        device="cpu", max_bytes=4 * 1024**2, max_slots=1,
                                        chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
        options = {"request_queue": Queue(maxsize=1), "response_queue": Queue(maxsize=1),
                   "channel_id": f"{index:032x}", "timeout": 2.0}
        routes[endpoint] = options
        sources.append(source)
        descriptors.append(source.publish({"ids": torch.arange(2)}, lease_id="sample"))
        services.append(NativeSourceControlService(source, **options))
    router = NativeSourceControlRouter(routes)
    with pytest.raises(ValueError, match="no configured"):
        router.validate_published({"segment": "other"})
    for descriptor in descriptors:
        router.validate_published(descriptor)
        router.bind_readers(descriptor, readers=[0], plan_id=0, receipt_token="receipt")
        router.acknowledge(descriptor, reader=0, plan_id=0, receipt_token="receipt")
        router.release(descriptor, plan_id=0, receipt_token="receipt")
    for service, source in zip(services, sources, strict=True):
        service.close()
        assert not source._slots
        with pytest.raises(RuntimeError, match="sealed"):
            source.publish({"ids": torch.arange(2)}, lease_id="after-control-close")
        source.close()
    requests = Queue(maxsize=1)
    stalled = NativeSourceControlClient(requests, Queue(maxsize=1), channel_id="a" * 32, timeout=0.01)
    with pytest.raises(TimeoutError, match="deadline"):
        stalled.validate_published({"segment": "source"})
    with pytest.raises(RuntimeError, match="quarantined"):
        stalled.validate_published({"segment": "source"})
    assert requests.get_nowait()["method"] == "validate_published"
    replies = Queue(maxsize=1)
    replies.put({"channel_id": "a" * 32, "request_id": "0" * 32, "ok": True})
    mismatch = NativeSourceControlClient(Queue(maxsize=1), replies, channel_id="a" * 32, timeout=1.0)
    with pytest.raises(ValueError, match="acknowledgement identity"):
        mismatch.validate_published({"segment": "source"})
    with pytest.raises(RuntimeError, match="quarantined"):
        mismatch.validate_published({"segment": "source"})


def test_native_unbound_discard_never_releases_dispatched_source():
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                    metadata_server="metadata", device="cpu", max_bytes=8 * 1024**2,
                                    max_slots=2, chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    tail = source.publish({"ids": torch.arange(2)}, lease_id="tail")
    assigned = source.publish({"ids": torch.arange(2)}, lease_id="assigned")
    source.bind_readers(assigned, readers=[0], plan_id=0, receipt_token="receipt")
    with pytest.raises(ValueError, match="dispatched"):
        source.discard_unbound(assigned)
    source.discard_unbound(tail)
    source.discard_unbound(tail)
    assert set(source._slots) == {"assigned"}
    source.acknowledge(assigned, reader=0, plan_id=0, receipt_token="receipt")
    source.release(assigned, plan_id=0, receipt_token="receipt")
    source.close()


@pytest.mark.parametrize("slot_limit", [17, 18])
def test_native_source_supports_more_than_sixteen_bounded_slots(slot_limit):
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport

    slot_bytes = 4 * 1024**2 - 1
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                    metadata_server="metadata", device="cpu", max_bytes=17 * slot_bytes,
                                    max_slots=slot_limit, chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptors = [source.publish({"ids": torch.tensor([i])}, lease_id=str(i), readers=[0]) for i in range(17)]
    assert source._registered_bytes == 17 * slot_bytes
    with pytest.raises(ValueError, match="slot limit|slot or byte budget"):
        source.publish({"ids": torch.tensor([18])}, lease_id="overflow", readers=[0])
    source.acknowledge(descriptors[0], reader=0)
    source.release(descriptors[0])
    assert source._registered_bytes == 16 * slot_bytes
    replacement = source.publish({"ids": torch.tensor([18])}, lease_id="replacement", readers=[0])
    for descriptor in [*descriptors[1:], replacement]:
        source.acknowledge(descriptor, reader=0)
        source.release(descriptor)
    assert source._registered_bytes == 0
    source.close()


def test_native_transport_quarantines_uncertain_lease_invalidation(monkeypatch):
    from verl_speco.transport import native_feature_transport as native

    monkeypatch.setattr(native, "_REGISTERED_OWNERS", set())
    unregistered = []
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): unregistered.append(args); return 0

    source = native.NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                           metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                           max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0])
    source.acknowledge(descriptor, reader=0)
    def fail_zero(_):
        raise RuntimeError("device submission failed")
    monkeypatch.setattr(torch.Tensor, "zero_", fail_zero)
    with pytest.raises(RuntimeError, match="device submission failed"):
        source.release(descriptor)
    assert "lease" in source._slots and source in native._REGISTERED_OWNERS and not unregistered
    with pytest.raises(RuntimeError, match="quarantined"):
        source.release(descriptor)
    with pytest.raises(RuntimeError, match="process fencing"):
        source.close()


@pytest.mark.parametrize("failure", ["register", "unregister"])
def test_native_transport_quarantines_ambiguous_registration(failure, monkeypatch):
    import gc
    import weakref
    from verl_speco.transport import native_feature_transport as native
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    monkeypatch.setattr(native, "_REGISTERED_OWNERS", set())
    registrations = {}
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, address, size):
            registrations[address] = size
            return int(failure == "register")
        def unregister_memory(self, address):
            if failure == "unregister":
                return 1
            del registrations[address]
            return 0
    transport = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                        metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                        max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    if failure == "register":
        with pytest.raises(RuntimeError, match="register_memory"):
            transport.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0])
    else:
        descriptor = transport.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0])
        transport.acknowledge(descriptor, reader=0)
        with pytest.raises(RuntimeError, match="unregister_memory"):
            transport.release(descriptor)
    assert len(registrations) == len(transport._registered) == 1
    assert transport._registered_bytes == 4 * 1024**2 - 1
    with pytest.raises(RuntimeError, match="process fencing"):
        transport.close()
    owner = weakref.ref(transport)
    del transport
    gc.collect()
    assert owner() is not None and len(native._REGISTERED_OWNERS) == 1


def test_native_transport_rejects_invalid_layout_before_read():
    from copy import deepcopy
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport, _fingerprint
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, address, size): return 0
        def unregister_memory(self, address): return 0
        def transfer_sync_read(self, *args): pytest.fail("invalid descriptor triggered native read")
    transport = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                        metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                        max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    original = transport.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0])
    with pytest.raises(ValueError, match="slot or byte budget"):
        transport.read(original)
    for mutate in (lambda d: d["fields"][0].update(offset=0),
                   lambda d: d["fields"][0].update(shape=[10**9]),
                   lambda d: d["fields"].append(deepcopy(d["fields"][0])),
                   lambda d: d["fields"][0].update(dtype=[]),
                   lambda d: d.update(address=2**64)):
        descriptor = deepcopy(original)
        mutate(descriptor)
        descriptor["sha256"] = _fingerprint({key: value for key, value in descriptor.items() if key != "sha256"})
        with pytest.raises(ValueError, match="descriptor|layout"):
            transport.read(descriptor)
    transport.acknowledge(original, reader=0)
    transport.release(original)
    transport.close()


def test_native_registered_owner_survives_until_successful_release(monkeypatch):
    import gc
    import weakref
    from verl_speco.transport import native_feature_transport as native
    monkeypatch.setattr(native, "_REGISTERED_OWNERS", set())
    transport = native.NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                               metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                               max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptor = transport.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0])
    reference = weakref.ref(transport)
    del transport
    gc.collect()
    assert reference() is not None
    owner = reference()
    owner.acknowledge(descriptor, reader=0)
    owner.release(descriptor)
    owner.close()
    del owner
    gc.collect()
    assert reference() is None and not native._REGISTERED_OWNERS


def test_native_transport_rejects_inherited_process_handles(monkeypatch):
    from verl_speco.transport import native_feature_transport as native
    pid = [1]
    monkeypatch.setattr(native, "os", SimpleNamespace(getpid=lambda: pid[0]))
    transport = native.NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                               metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                               max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptor = transport.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0])
    pid[0] = 2
    with pytest.raises(RuntimeError, match="different process"):
        transport.acknowledge(descriptor, reader=0)
    with pytest.raises(RuntimeError, match="different process"):
        transport.close()
    pid[0] = 1
    transport.acknowledge(descriptor, reader=0)
    transport.release(descriptor)
    transport.close()


def test_native_lease_binds_scheduler_dispatch_before_acknowledgement():
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                    metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                    max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="lease")
    with pytest.raises(RuntimeError, match="not been bound"):
        source.release(descriptor)
    with pytest.raises(RuntimeError, match="not been bound"):
        source.acknowledge(descriptor, reader=0)
    binding = {"plan_id": 7, "receipt_token": "dispatch"}
    source.bind_readers(descriptor, readers=[1, 0], **binding)
    source.bind_readers(descriptor, readers=[0, 1], **binding)
    with pytest.raises(ValueError, match="another dispatch"):
        source.bind_readers(descriptor, readers=[0], **binding)
    with pytest.raises(ValueError, match="acknowledgement identity"):
        source.acknowledge(descriptor, reader=0, plan_id=7, receipt_token="old")
    source.acknowledge(descriptor, reader=0, **binding)
    with pytest.raises(RuntimeError, match="unacknowledged"):
        source.release(descriptor, **binding)
    source.acknowledge(descriptor, reader=1, **binding)
    source.release(descriptor, **binding)
    source.bind_readers(descriptor, readers=[0, 1], **binding)
    source.acknowledge(descriptor, reader=1, **binding)
    source.release(descriptor, **binding)
    with pytest.raises(ValueError, match="identity"):
        source.release(descriptor, plan_id=7, receipt_token="old")
    source.close()


def test_native_preselected_readers_bind_dispatch_without_rewriting_acknowledgements():
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                    metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                    max_slots=2, chunk_bytes=2 * 1024**2, engine_factory=_NativeTestEngine)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="lease", readers=[0, 1])
    binding = {"plan_id": 7, "receipt_token": "dispatch"}
    with pytest.raises(ValueError, match="another dispatch"):
        source.bind_readers(descriptor, readers=[0], **binding)
    source.bind_readers(descriptor, readers=[1, 0], **binding)
    source.bind_readers(descriptor, readers=[0, 1], **binding)
    with pytest.raises(ValueError, match="identity"):
        source.acknowledge(descriptor, reader=0)
    source.acknowledge(descriptor, reader=0, **binding)
    source.acknowledge(descriptor, reader=1, **binding)
    source.release(descriptor, **binding)

    legacy = source.publish({"ids": torch.arange(2)}, lease_id="legacy", readers=[0, 1])
    source.acknowledge(legacy, reader=0)
    with pytest.raises(ValueError, match="another dispatch"):
        source.bind_readers(legacy, readers=[0, 1], **binding)
    source.acknowledge(legacy, reader=1)
    source.release(legacy)
    source.close()


def test_native_payload_checksum_handles_multiple_bounded_chunks(monkeypatch):
    import hashlib
    from verl_speco.transport import native_feature_transport as native
    monkeypatch.setattr(native, "_CHECKSUM_CHUNK", 7)
    payload = torch.arange(31, dtype=torch.uint8)
    assert native._content_hash(payload) == hashlib.sha256(bytes(range(31))).hexdigest()


def test_native_slow_reader_keeps_source_lease_until_completion():
    import ctypes
    import threading
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    registered, entered, resume, results = {}, threading.Event(), threading.Event(), []
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, address, size): registered[address] = size; return 0
        def unregister_memory(self, address): del registered[address]; return 0
        def transfer_sync_read(self, segment, local, remote, size):
            if not entered.is_set():
                entered.set()
                if not resume.wait(timeout=10):
                    raise TimeoutError("test reader stalled")
            ctypes.memmove(local, remote, size)
            return 0
    def make():
        return NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                      metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                      max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    source, receiver = make(), make()
    descriptor = source.publish({"ids": torch.arange(8)}, lease_id="lease")
    binding = {"plan_id": 4, "receipt_token": "receipt"}
    source.bind_readers(descriptor, readers=[1], **binding)
    def read():
        try:
            results.append(receiver.read(descriptor))
        except Exception as exc:
            results.append(exc)
    worker = threading.Thread(target=read)
    worker.start()
    try:
        assert entered.wait(timeout=10)
        with pytest.raises(RuntimeError, match="unacknowledged"):
            source.release(descriptor, **binding)
        assert len(registered) == 2
    finally:
        resume.set()
        worker.join(timeout=10)
    assert not worker.is_alive() and len(results) == 1 and isinstance(results[0], dict)
    assert torch.equal(results[0]["ids"], torch.arange(8))
    source.acknowledge(descriptor, reader=1, **binding)
    source.release(descriptor, **binding)
    source.close()
    receiver.close()
    assert not registered


def test_native_read_snapshots_descriptor_before_native_transfer():
    import ctypes
    from copy import deepcopy
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    mutable, reads = {}, []
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, address, size): return 0
        def unregister_memory(self, address): return 0
        def transfer_sync_read(self, segment, local, remote, size):
            if not reads:
                mutable["address"] = 1
            reads.append(size)
            ctypes.memmove(local, remote, size)
            return 0
    def make():
        return NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                      metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                      max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    source, receiver = make(), make()
    mutable.update(source.publish({"ids": torch.arange(4)}, lease_id="lease", readers=[0]))
    original = deepcopy(mutable)
    assert torch.equal(receiver.read(mutable)["ids"], torch.arange(4))
    assert mutable["address"] == 1 and len(reads) == 2
    source.acknowledge(original, reader=0)
    source.release(original)
    source.close()
    receiver.close()


def test_native_hidden_store_timeout_never_releases_source(monkeypatch):
    from verl_speco.producer import hidden_states_store as module
    now, calls = [0.0], []
    class Reader:
        def read(self, descriptor, *, output_device):
            calls.append(("read", descriptor, output_device))
            now[0] = 2.0
            return {"hidden_states": torch.ones(2)}
        def close(self): calls.append(("close",))
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    store = module.NativeHiddenStatesStore(Reader(), lambda descriptor: calls.append(("release", descriptor)))
    with pytest.raises(TimeoutError, match="deadline"):
        store.load("{}", timeout=1)
    assert calls == [("read", {}, "cpu")]
    store.close()
    assert calls[-1] == ("close",)


def test_native_hidden_store_rejects_unawaited_release():
    from verl_speco.producer.hidden_states_store import NativeHiddenStatesStore
    calls = []
    async def release(descriptor):
        calls.append(descriptor)
    store = NativeHiddenStatesStore(SimpleNamespace(read=lambda *args, **kwargs: {}), release)
    with pytest.raises(RuntimeError, match="synchronously"):
        store.release("{}")
    assert not calls

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
        "expected_feature": {
            "algorithm": "DSPARK",
            "target_model_path": "/models/target",
            "target_revision": "producer-revision",
            "tokenizer_fingerprint": "tokenizer-sha256-a",
            "target_layer_ids": [2, 8],
            "hidden_states_layout": "dflash_aux",
            "hidden_dtype": "float32",
        },
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
        metadata={
            "target_model_path": "/models/target",
            "target_revision": "producer-revision",
            "tokenizer_fingerprint": "tokenizer-sha256-a",
            "target_layer_ids": [2, 8],
            "hidden_states_layout": "dflash_aux",
        },
    )


def _entry(sequence_no: int) -> ReadyEntry:
    meta = _metadata(sequence_no)
    return ReadyEntry(key=make_sample_key(meta), tag=make_ready_tag(meta))


@pytest.mark.parametrize("store_type", ["tq", "ray_reference"])
def test_feature_store_factory_builds_tq_without_path(store_type) -> None:
    store = build_feature_store_from_config(
        {"type": store_type, "path": None},
        read_only=True,
        transfer_queue_cfg=_config(),
    )
    assert isinstance(store, TQFeatureStore)
    assert store.run_id == "run-a"
    if store_type == "ray_reference":
        from verl_speco.trainer.tq_feature_store import ReferenceFeatureStore
        assert isinstance(store, ReferenceFeatureStore)


def test_tq_store_defaults_to_current_protocol_version() -> None:
    config = _config()
    del config["schema_version"]

    store = TQFeatureStore.from_config(config)

    assert store.schema_version == PROTOCOL_SCHEMA_VERSION
    assert store.expected_config.schema_version == PROTOCOL_SCHEMA_VERSION


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
    assert samples[0].metadata["target_revision"] == "producer-revision"
    eos = store.read_eos()
    assert eos is not None and eos.total_samples == 2


def test_token_balanced_assignments_preserve_batch_and_reduce_skew() -> None:
    entries = [_entry(index) for index in range(4)]
    for entry, length in zip(entries, (100, 90, 2, 1), strict=True):
        entry.tag["request"] = {"feature_positions": list(range(length))}
    balanced = build_assignments(entries, batch_size=2, world_size=2, balance_by_tokens=True)
    assert [[e.key for e in rank] for rank in balanced] == [
        [entries[0].key, entries[3].key], [entries[1].key, entries[2].key]
    ]
    assert balanced == build_assignments(entries, batch_size=2, world_size=2, balance_by_tokens=True)
    assert sorted(e.key for rank in balanced for e in rank) == sorted(e.key for e in entries)


def test_token_balanced_assignments_reject_missing_metadata_and_non_boolean() -> None:
    entries = [_entry(index) for index in range(2)]
    with pytest.raises(ValueError, match="feature position metadata"):
        build_assignments(entries, batch_size=1, world_size=2, balance_by_tokens=True)
    with pytest.raises(ValueError, match="boolean"):
        build_assignments(entries, batch_size=1, world_size=2, balance_by_tokens="false")


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


def test_scheduled_loader_uses_driver_selected_keys_without_listing() -> None:
    entries = [_entry(0), _entry(1)]
    store = _FakeStore(entries, eos=False)

    class Commands:
        def __init__(self):
            self.values = [
                {
                    "kind": "batch",
                    "global_keys": [entry.key for entry in entries],
                    "global_sequence_nos": [0, 1],
                    "assignments": [
                        [{"key": entry.key, "tag": entry.tag} for entry in entries]
                    ],
                },
                {"kind": "stop"},
            ]

        def get(self, *, block):
            assert block is True
            return self.values.pop(0)

    store.list_ready = lambda: pytest.fail("scheduled Consumer must not list TQ")
    loader = TQFeatureDataLoader(
        store,
        batch_size=2,
        rank=0,
        world_size=1,
        scheduled_commands=Commands(),
    )

    batches = list(loader)

    assert len(batches) == 1
    assert batches[0].global_keys == [entry.key for entry in entries]
    assert store.get_calls == [[entry.key for entry in entries]]


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


@pytest.mark.parametrize("sender_node,receiver_node,started,expected", [
    ("driver-node", "consumer-node", 10000.0, None),
    ("same-node", "same-node", 99.0, 1.0),
    (None, "consumer-node", 10000.0, None),
    (None, None, 10000.0, None),
])
def test_command_delivery_requires_shared_ray_clock(
    monkeypatch, sender_node, receiver_node, started, expected,
):
    from queue import Queue
    import time
    import verl_speco.trainer.tq_sample_source as module

    entry = _entry(0)
    store = _FakeStore([entry], eos=False)
    commands = Queue()
    commands.put({
        "kind": "batch", "assignments": [[{"key": entry.key, "tag": entry.tag}]],
        "dispatched_at": started, "clock_node_id": sender_node,
        "delivery_times": {"scheduler_ready_wait_time": 0.2},
    })
    monkeypatch.setattr(module, "current_ray_node_id", lambda: receiver_node)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: 100.0, perf_counter=time.perf_counter, sleep=time.sleep,
    ))
    loader = TQFeatureDataLoader(
        store, batch_size=1, rank=0, world_size=1, scheduled_commands=commands,
    )
    batch = next(iter(loader))
    assert batch.local_keys == [entry.key]
    assert batch.delivery_times["scheduler_ready_wait_time"] == 0.2
    if expected is None:
        assert "command_delivery_time" not in batch.delivery_times
    else:
        assert batch.delivery_times["command_delivery_time"] == expected


@pytest.mark.parametrize("balanced", [False, True])
def test_partial_assignments_cover_each_entry_once_without_empty_ranks(balanced):
    entries = [_entry(index) for index in range(5)]
    for entry, length in zip(entries, (100, 1, 90, 2, 3), strict=True):
        entry.tag["request"] = {"feature_positions": list(range(length))}
    groups = build_assignments(entries, batch_size=3, world_size=2,
                               allow_uneven=True, balance_by_tokens=balanced)
    assert [len(group) for group in groups] == [3, 2]
    assert sorted(e.key for group in groups for e in group) == sorted(e.key for e in entries)
    assert groups == build_assignments(entries, batch_size=3, world_size=2,
                                       allow_uneven=True, balance_by_tokens=balanced)
    with pytest.raises(ValueError, match="Expected exactly"):
        build_assignments(entries[:1], batch_size=3, world_size=2, allow_uneven=True)
    with pytest.raises(ValueError, match="Expected exactly"):
        build_assignments(entries, batch_size=3, world_size=2)


@pytest.mark.parametrize("balanced", [False, True])
def test_partial_assignments_allow_explicit_empty_ranks_without_repeating_samples(balanced):
    entries = [_entry(0)]
    entries[0].tag["request"] = {"feature_positions": [0, 1]}
    groups = build_assignments(entries, batch_size=2, world_size=3,
                               balance_by_tokens=balanced, allow_uneven=True, allow_empty=True)
    assert groups == [[entries[0]], [], []]
    with pytest.raises(ValueError, match="Empty ranks require uneven"):
        build_assignments(entries, batch_size=2, world_size=3, allow_empty=True)
