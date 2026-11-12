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

import pytest
import torch

from verl_speco.producer.hidden_states_store import (
    FILE_BACKEND,
    MOONCAKE_BACKEND,
    FileHiddenStatesStore,
    HiddenStatesStoreConfig,
    MooncakeHiddenStatesStore,
    MooncakeStoreSettings,
    build_hidden_states_store,
)


class _FakeMooncake:
    """Mirrors the hs_connectors MooncakeHiddenStatesStore contract.

    ``get_sample`` takes the full transfer manifest (the packed tensor layout
    it decodes with), ``delete_sample`` takes the bare handle string.
    """

    def __init__(self, payload):
        self.payload = payload
        self.loaded: list[dict] = []
        self.deleted: list[str] = []

    def get_sample(self, transfer_manifest, *, timeout=120.0):
        self.loaded.append(transfer_manifest)
        return self.payload

    def delete_sample(self, key):
        self.deleted.append(key)


def test_build_store_defaults_to_file() -> None:
    assert isinstance(build_hidden_states_store(None), FileHiddenStatesStore)


def test_build_store_selects_mooncake() -> None:
    store = build_hidden_states_store({"backend": "mooncake"})
    assert isinstance(store, MooncakeHiddenStatesStore)
    settings = MooncakeStoreSettings(local_hostname="127.0.0.1")
    assert HiddenStatesStoreConfig("mooncake", settings).mooncake is settings


def test_managed_native_reader_keeps_source_until_scheduler_release():
    import ctypes
    import json
    from verl_speco.producer.hidden_states_store import build_managed_native_reader
    from verl_speco.trainer.feature_store import build_feature_store_from_config
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport

    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0
        def transfer_sync_read(self, segment, local, remote, size):
            ctypes.memmove(local, remote, size)
            return 0

    settings = {"endpoints": ["localhost:1001", "localhost:1002"], "generation": 1,
                "metadata_server": "metadata", "max_bytes": 4 * 1024**2, "chunk_bytes": 2 * 1024**2}
    for bad in ({**settings, "endpoints": ["same", "same"]}, {**settings, "endpoints": ["only-one"]},
                {**settings, "unexpected": 1}):
        with pytest.raises(ValueError, match="Native reader"):
            build_managed_native_reader(bad, run_id="run", rank=0, world_size=2, device="cpu", engine_factory=Engine)
    reader = build_managed_native_reader(settings, run_id="run", rank=1, world_size=2, device="cpu", engine_factory=Engine)
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source", metadata_server="metadata",
                                    device="cpu", max_bytes=4 * 1024**2, max_slots=1,
                                    chunk_bytes=2 * 1024**2, engine_factory=Engine)
    descriptor = source.publish({"ids": torch.arange(4)}, lease_id="sample", readers=[1])
    handle = json.dumps(descriptor)
    assert torch.equal(reader.load(handle)[0]["ids"], torch.arange(4))
    reader.release(handle)
    assert "sample" in source._slots
    with pytest.raises(ValueError, match="requires ray_reference"):
        build_feature_store_from_config({"type": "torch_shard"}, hidden_states_store=reader)
    store = build_feature_store_from_config({"type": "ray_reference"}, read_only=True,
                                            transfer_queue_cfg={"run_id": "run", "ray": {"address": "unused"}},
                                            hidden_states_store=reader)
    assert store.hidden_states_store is reader
    store.close_local()
    source.acknowledge(descriptor, reader=1)
    source.release(descriptor)
    source.close()


def test_shared_file_root_limits_reads_and_cleanup(tmp_path):
    from safetensors.torch import save_file
    root = tmp_path / "shared"
    root.mkdir()
    inside, outside = root / "inside.safetensors", tmp_path / "outside.safetensors"
    for path in (inside, outside):
        save_file({"token_ids": torch.arange(2), "hidden_states": torch.ones(2, 1, 2)}, str(path))
    store = build_hidden_states_store({"backend": "file", "file_root": str(root)})
    assert store.load(str(inside))[0]["token_ids"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="outside the configured shared file root"):
        store.load(str(outside))
    with pytest.raises(ValueError, match="outside the configured shared file root"):
        store.release(str(outside))
    assert outside.exists()
    store.release(str(inside))
    assert not inside.exists()


def test_shared_file_root_requires_absolute_existing_file_backend(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        HiddenStatesStoreConfig.from_mapping({"file_root": "relative/path"})
    with pytest.raises(ValueError, match="file backend"):
        HiddenStatesStoreConfig.from_mapping({"backend": "mooncake", "file_root": str(tmp_path)})
    with pytest.raises(ValueError, match="existing absolute"):
        build_hidden_states_store({"file_root": str(tmp_path / "missing")})


def test_shared_file_store_rejects_replaced_root(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    store = FileHiddenStatesStore(str(root))
    root.rename(tmp_path / "previous")
    root.mkdir()
    with pytest.raises(ValueError, match="shared file root changed"):
        store.validate_reference(str(root / "sample.safetensors"))


def test_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported hidden-state store backend"):
        HiddenStatesStoreConfig(backend="redis")


def test_settings_default_hostname_is_non_empty() -> None:
    settings = MooncakeStoreSettings(local_hostname="")
    assert settings.local_hostname


def test_mooncake_store_load_and_release_round_trip() -> None:
    settings = MooncakeStoreSettings(master_server_address="10.0.0.1:50051")
    store = MooncakeHiddenStatesStore(settings)
    fake = _FakeMooncake(
        {
            "hidden_states": torch.zeros((2, 3), dtype=torch.bfloat16),
            "token_ids": torch.tensor([1, 2]),
        }
    )
    store._store = fake
    manifest = {
        "handle": "key-1",
        "version": 1,
        "tensors": {},
        "metadata": {"total_aligned_bytes": 20},
    }

    payload, byte_size = store.load(manifest)

    assert set(payload) == {"hidden_states", "token_ids"}
    assert byte_size == 2 * 3 * 2 + 2 * 8
    # The manifest reaches the store unchanged; release uses the bare handle.
    assert fake.loaded == [manifest]
    store.release("key-1")
    assert fake.deleted == ["key-1"]


def test_mooncake_store_load_rejects_bare_handle() -> None:
    # A bare handle cannot be decoded (no tensor layout), so refuse it loudly
    # instead of failing deep inside the store with an AttributeError.
    store = MooncakeHiddenStatesStore(MooncakeStoreSettings())
    store._store = _FakeMooncake({})
    with pytest.raises(TypeError, match="full transfer manifest"):
        store.load("key-1")


def test_mooncake_store_release_propagates_deletion_failure() -> None:
    store = MooncakeHiddenStatesStore(MooncakeStoreSettings())

    class _Broken:
        def delete_sample(self, key):
            raise RuntimeError("gone")

    store._store = _Broken()
    # Deletion errors must surface so callers with a retry policy can react.
    with pytest.raises(RuntimeError, match="gone"):
        store.release("missing")


def _native_connector_for_test(upstream, device):
    import ast
    import os
    import threading
    import uuid
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "verl_speco/mooncake_hidden_states_connector.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SpecoNativeHiddenStatesConnector")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
                        type_ignores=[])
    namespace = {"SpecoMooncakeHiddenStatesConnector": object, "torch": torch, "threading": threading,
                 "uuid": uuid, "os": os, "_upstream_hs": upstream, "get_torch_device": lambda: device}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["SpecoNativeHiddenStatesConnector"]


@pytest.mark.parametrize("publish_fails", [False, True])
def test_native_connector_captures_device_payload_and_preserves_ambiguous_source(monkeypatch, publish_fails):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    events, published = [], []
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0
    class Stream:
        def wait_event(self, value): events.append(("wait", value))
        def synchronize(self): events.append(("synchronize", None))
    def extract(cache, slots, count):
        assert slots.tolist() == [8, 9, 10, 11, 16, 17, 18, 19] and count == 3
        return cache[:3]
    upstream = SimpleNamespace(extract_from_kv_cache=extract, assert_finite=lambda *args: None)
    cls = _native_connector_for_test(upstream, SimpleNamespace(stream=lambda _: nullcontext()))
    connector = cls()
    connector._kv_cache = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
    connector._copy_stream, connector._block_size = Stream(), 4
    connector._native_source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source",
        metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2, max_slots=1,
        chunk_bytes=2 * 1024**2, engine_factory=Engine)
    def publish(handle, descriptor):
        published.append((handle, descriptor))
        if publish_fails:
            raise TimeoutError("ambiguous publish")
    connector._native_directory = SimpleNamespace(publish=SimpleNamespace(remote=publish))
    connector._native_control_timeout = 1.0
    connector._native_ready = True
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda result, timeout: result))
    pending = SimpleNamespace(block_ids=[2, 4], token_ids=torch.tensor([10, 11, 12]), mooncake_key="sample")
    if publish_fails:
        with pytest.raises(TimeoutError, match="ambiguous"):
            connector._publish_sample(pending, "ready")
    else:
        connector._publish_sample(pending, "ready")
    assert events == [("wait", "ready"), ("synchronize", None)]
    assert published[0][0] == "sample" and "sample" in connector._native_source._slots
    connector._native_source.discard_unbound(published[0][1])
    connector._native_source.close()


@pytest.mark.parametrize("registration_fails", [False, True])
def test_native_connector_registers_source_and_retains_ambiguous_setup(monkeypatch, registration_fails):
    import sys
    import threading
    import types
    from queue import Queue
    from types import SimpleNamespace
    from verl_speco.transport import native_feature_transport as native
    registrations, errors = [], []
    entered, proceed, second_done = threading.Event(), threading.Event(), threading.Event()
    def register(endpoint, source_id, options, source_settings):
        registrations.append((endpoint, source_id, options, source_settings))
        entered.set()
        assert proceed.wait(3)
        if registration_fails:
            raise TimeoutError("registration reply lost")
    directory = SimpleNamespace(scope=SimpleNamespace(remote=lambda: {"run_id": "run", "generation": 1}),
                                register_source=SimpleNamespace(remote=register))
    ray = types.ModuleType("ray")
    ray.__path__ = []
    ray.is_initialized = lambda: True
    ray.get_actor = lambda name, namespace: directory
    ray.get = lambda value, timeout: value
    ray_util, ray_queue = types.ModuleType("ray.util"), types.ModuleType("ray.util.queue")
    ray_util.__path__, ray_queue.Queue = [], Queue
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.queue", ray_queue)
    monkeypatch.setattr(native, "NativeFeatureTransport", lambda **kwargs: SimpleNamespace(settings=kwargs))
    monkeypatch.setattr(native, "NativeSourceControlService", lambda source, **kwargs: SimpleNamespace(source=source))
    for name, value in {"SPECO_NATIVE_DIRECTORY": "dir", "SPECO_NATIVE_RUN_ID": "run",
                        "SPECO_NATIVE_GENERATION": "1", "SPECO_NATIVE_SOURCE_ENDPOINT": "source",
                        "SPECO_NATIVE_METADATA_SERVER": "metadata", "SPECO_NATIVE_SOURCE_MAX_BYTES": "4194304",
                        "SPECO_NATIVE_SOURCE_MAX_SLOTS": "1", "SPECO_NATIVE_CHUNK_BYTES": "2097152"}.items():
        monkeypatch.setenv(name, value)
    connector = _native_connector_for_test(None, None)()
    connector._kv_cache = torch.zeros(1)
    def initialize(done=None):
        try:
            connector._ensure_store()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()
    first = threading.Thread(target=initialize)
    second = threading.Thread(target=initialize, args=(second_done,))
    first.start()
    assert entered.wait(3)
    second.start()
    assert not second_done.wait(0.05)
    proceed.set()
    first.join(timeout=3)
    second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    if registration_fails:
        assert {type(error) for error in errors} == {TimeoutError, RuntimeError}
        assert connector._native_init_failed and not connector._native_ready
    else:
        assert not errors and connector._native_ready
        connector._ensure_store()
    assert len(registrations) == 1 and connector._native_source is connector._native_service.source
    assert registrations[0][0] == "source" and len(registrations[0][1]) == 32
    assert connector._native_source.settings["device"] == torch.device("cpu")


def test_connector_register_kv_caches_uses_device_stream(monkeypatch) -> None:
    pytest.importorskip(
        "hs_connectors", reason="optional vLLM hidden-state connector is not installed"
    )
    import verl_speco.mooncake_hidden_states_connector as connector_module
    from verl_speco.mooncake_hidden_states_connector import (
        SpecoMooncakeHiddenStatesConnector,
    )

    stream_sentinel = object()

    class _Layer:
        pass

    layer = _Layer()

    class _Device:
        def Stream(self):
            return stream_sentinel

    connector = object.__new__(SpecoMooncakeHiddenStatesConnector)
    connector._vllm_config = object()

    monkeypatch.setattr(
        connector_module._upstream_hs,
        "get_tensor_model_parallel_rank",
        lambda: 3,
    )
    monkeypatch.setattr(
        connector_module._upstream_hs,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {"cache": layer},
    )
    monkeypatch.setattr(connector_module, "get_torch_device", lambda: _Device())

    kv_caches = {"cache": torch.zeros(1)}
    connector.register_kv_caches(kv_caches)

    assert connector._kv_cache is kv_caches["cache"]
    assert connector._copy_stream is stream_sentinel
    assert connector._is_tp_rank_zero is False


def test_file_store_load_and_release(tmp_path) -> None:
    from safetensors.torch import save_file

    path = tmp_path / "sample.safetensors"
    tensors = {
        "hidden_states": torch.zeros((4, 2), dtype=torch.bfloat16),
        "token_ids": torch.arange(4),
    }
    save_file(tensors, str(path))

    store = FileHiddenStatesStore()
    payload, byte_size = store.load(str(path))
    assert set(payload) == {"hidden_states", "token_ids"}
    assert byte_size == path.stat().st_size

    store.release(str(path))
    assert not path.exists()
    assert not (tmp_path / "sample.safetensors.lock").exists()


def test_file_backend_constant_matches_default() -> None:
    assert FILE_BACKEND == "file"
    assert MOONCAKE_BACKEND == "mooncake"


def test_speco_connector_restores_rank0_finished_count() -> None:
    pytest.importorskip(
        "hs_connectors", reason="optional vLLM hidden-state connector is not installed"
    )
    from hs_connectors.mooncake_hidden_states_connector import (
        MooncakeHiddenStatesConnector,
    )

    from verl_speco.mooncake_hidden_states_connector import (
        SpecoMooncakeHiddenStatesConnector,
    )

    assert issubclass(SpecoMooncakeHiddenStatesConnector, MooncakeHiddenStatesConnector)
    assert SpecoMooncakeHiddenStatesConnector.get_finished_count(None) == 1


def test_release_initializes_store_for_unread_reference(monkeypatch):
    store = MooncakeHiddenStatesStore(MooncakeStoreSettings())
    fake = _FakeMooncake({})
    monkeypatch.setattr(store, "_ensure_store", lambda: fake)
    store.release("never-read")
    assert fake.deleted == ["never-read"]


def test_connector_publish_passes_transfer_manifest(monkeypatch) -> None:
    """The publish path must hand the store the transfer manifest, not the key.

    ``put_sample`` reads the packed tensor layout from the manifest; passing
    the bare ``mooncake_key`` string (as an earlier revision did) fails against
    the real hs_connectors store.
    """
    pytest.importorskip(
        "hs_connectors", reason="optional vLLM hidden-state connector is not installed"
    )
    import contextlib
    from types import SimpleNamespace

    import verl_speco.mooncake_hidden_states_connector as connector_module
    from verl_speco.mooncake_hidden_states_connector import (
        SpecoMooncakeHiddenStatesConnector,
    )

    class _Stream:
        def wait_event(self, event):
            pass

        def synchronize(self):
            pass

    class _Device:
        def Stream(self):
            return _Stream()

        def stream(self, stream):
            return contextlib.nullcontext()

    connector = object.__new__(SpecoMooncakeHiddenStatesConnector)
    connector._kv_cache = torch.zeros(8, 4)
    connector._copy_stream = _Stream()
    connector._block_size = 4
    connector._store = _RecordingPublishStore()
    manifest = {
        "handle": "key-1",
        "version": 1,
        "tensors": {},
        "metadata": {"total_aligned_bytes": 12},
    }
    pending = SimpleNamespace(
        block_ids=[0],
        token_ids=torch.tensor([1, 2]),
        req_id="req-1",
        mooncake_key="key-1",
        transfer_manifest=manifest,
    )

    hidden = torch.zeros(2, 3, dtype=torch.float16)
    monkeypatch.setattr(
        connector_module._upstream_hs, "extract_from_kv_cache", lambda *a, **k: hidden
    )
    monkeypatch.setattr(connector_module._upstream_hs, "assert_finite", lambda *a: None)
    monkeypatch.setattr(connector_module, "get_torch_device", lambda: _Device())

    connector._publish_sample(pending, object())

    assert connector._store.puts[0][0] is manifest
    assert connector._store.errors == []


class _RecordingPublishStore:
    def __init__(self):
        self.puts: list[tuple[dict, dict]] = []
        self.errors: list[tuple[dict, str]] = []

    def put_sample(self, transfer_manifest, tensors):
        self.puts.append((transfer_manifest, tensors))

    def put_error(self, transfer_manifest, error):
        self.errors.append((transfer_manifest, error))
