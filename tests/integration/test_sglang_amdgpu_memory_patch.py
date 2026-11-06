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
"""The ROCm memory-capacity fallback honors the shared SGLang patch switch."""

from __future__ import annotations

import sys
import types

import pytest

from verl_speco.integration import sglang_runtime


def _original_probe() -> float:
    return 288.0 * 1024


@pytest.fixture
def fake_sglang_common(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # SGLang is not installed in CPU CI; only the probed module is needed.
    common = types.ModuleType("sglang.srt.utils.common")
    common.get_amdgpu_memory_capacity = _original_probe
    utils = types.ModuleType("sglang.srt.utils")
    utils.common = common
    srt = types.ModuleType("sglang.srt")
    srt.utils = utils
    sglang = types.ModuleType("sglang")
    sglang.srt = srt
    for name, module in {
        "sglang": sglang,
        "sglang.srt": srt,
        "sglang.srt.utils": utils,
        "sglang.srt.utils.common": common,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sglang_runtime, "_AMDGPU_MEM_CAPACITY_PATCHED", False)
    return common


@pytest.mark.parametrize("value", ["1", "true"])
def test_disable_switch_skips_memory_capacity_patch(
    monkeypatch: pytest.MonkeyPatch, fake_sglang_common: types.ModuleType, value: str
) -> None:
    monkeypatch.setenv("VERL_DISABLE_SGLANG_PATCH", value)

    sglang_runtime._install_amdgpu_memory_capacity_patch()

    assert fake_sglang_common.get_amdgpu_memory_capacity is _original_probe
    assert sglang_runtime._AMDGPU_MEM_CAPACITY_PATCHED is False


@pytest.mark.parametrize("value", [None, "0"])
def test_memory_capacity_patch_installed_by_default(
    monkeypatch: pytest.MonkeyPatch,
    fake_sglang_common: types.ModuleType,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("VERL_DISABLE_SGLANG_PATCH", raising=False)
    else:
        monkeypatch.setenv("VERL_DISABLE_SGLANG_PATCH", value)

    sglang_runtime._install_amdgpu_memory_capacity_patch()

    patched = fake_sglang_common.get_amdgpu_memory_capacity
    assert patched is not _original_probe
    assert patched() == _original_probe()
