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
