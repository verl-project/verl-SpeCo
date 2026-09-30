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
    def __init__(self, payload):
        self.payload = payload
        self.deleted: list[str] = []

    def get_sample(self, key):
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

    payload, byte_size = store.load("key-1")

    assert set(payload) == {"hidden_states", "token_ids"}
    assert byte_size == 2 * 3 * 2 + 2 * 8
    store.release("key-1")
    assert fake.deleted == ["key-1"]


def test_mooncake_store_release_is_best_effort() -> None:
    store = MooncakeHiddenStatesStore(MooncakeStoreSettings())

    class _Broken:
        def delete_sample(self, key):
            raise RuntimeError("gone")

    store._store = _Broken()
    store.release("missing")  # must not raise


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
    from hs_connectors.mooncake_hidden_states_connector import (
        MooncakeHiddenStatesConnector,
    )

    from verl_speco.mooncake_hidden_states_connector import (
        SpecoMooncakeHiddenStatesConnector,
    )

    assert issubclass(
        SpecoMooncakeHiddenStatesConnector, MooncakeHiddenStatesConnector
    )
    assert SpecoMooncakeHiddenStatesConnector.get_finished_count(None) == 1
