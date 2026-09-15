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
"""The vLLM draft publish stages its CUDA-IPC buckets in non-expandable segments."""

from __future__ import annotations

import sys
import types

import pytest

from verl_speco.integration.vllm_runtime import _ipc_safe_allocator


@pytest.fixture
def fake_verl_device(monkeypatch):
    calls: list[bool] = []
    # The CPU CI runs without verl installed; stub the parents so the import
    # machinery reaches the fake leaf module.
    for parent in ("verl", "verl.utils"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    module = types.ModuleType("verl.utils.device")
    module.set_expandable_segments = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.utils.device", module)
    return calls


def test_ipc_send_disables_expandable_segments_and_leaves_them_off(
    fake_verl_device, monkeypatch
) -> None:
    """Torch has no getter for the prior state, so nothing is restored unless
    the process explicitly asked for expandable segments via the env; verl's
    own sync re-enables them per step where it wants them."""
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with _ipc_safe_allocator(True):
        assert fake_verl_device == [False]
    assert fake_verl_device == [False]


def test_ipc_send_restores_env_requested_expandable_segments(
    fake_verl_device, monkeypatch
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with _ipc_safe_allocator(True):
        assert fake_verl_device == [False]
    assert fake_verl_device == [False, True]


def test_ipc_send_keeps_expandable_segments_off_on_failure(
    fake_verl_device, monkeypatch
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    with pytest.raises(RuntimeError, match="send failed"):
        with _ipc_safe_allocator(True):
            raise RuntimeError("send failed")
    assert fake_verl_device == [False]


def test_shm_transport_leaves_the_allocator_alone(fake_verl_device) -> None:
    with _ipc_safe_allocator(False):
        pass
    assert fake_verl_device == []


def test_verl_without_the_helper_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "verl.utils.device", None)
    with _ipc_safe_allocator(True):
        pass
