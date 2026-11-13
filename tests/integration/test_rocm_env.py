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
"""ROCm driver env setup: GPU visibility masks and the legacy IPC mode default."""

from __future__ import annotations

import os
import sys

import pytest

from verl_speco.integration import rocm_env
from verl_speco.integration.rocm_env import (
    align_visible_devices_with_ray,
    default_legacy_ipc_mode,
    neutralize_hip_visible_devices,
)


def _set_rocm_build(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    # ROCm is detected from torch.version.hip, or HIP_PLATFORM when torch is missing
    # (the contract-test CI job has no torch).
    try:
        import torch
    except ImportError:
        if enabled:
            monkeypatch.setenv("HIP_PLATFORM", "amd")
        else:
            monkeypatch.delenv("HIP_PLATFORM", raising=False)
        return
    monkeypatch.setattr(torch.version, "hip", "7.2.0" if enabled else None, raising=False)


@pytest.fixture
def rocm(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    _set_rocm_build(monkeypatch, True)
    # vllm is not installed in CPU CI; force the same "import failed" path everywhere.
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return monkeypatch


def _masks() -> tuple[str | None, str | None]:
    return os.environ.get("HIP_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES")


def test_hip_only_mask_moves_to_cuda(rocm: pytest.MonkeyPatch) -> None:
    rocm.setenv("HIP_VISIBLE_DEVICES", "2,3")

    assert neutralize_hip_visible_devices(context="in test") is True
    assert _masks() == (None, "2,3")


def test_cuda_only_mask_is_left_alone(rocm: pytest.MonkeyPatch) -> None:
    rocm.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert neutralize_hip_visible_devices() is False
    assert _masks() == (None, "0,1")


def test_identical_masks_drop_hip(rocm: pytest.MonkeyPatch) -> None:
    rocm.setenv("HIP_VISIBLE_DEVICES", "4")
    rocm.setenv("CUDA_VISIBLE_DEVICES", "4")

    assert neutralize_hip_visible_devices() is True
    assert _masks() == (None, "4")


def test_conflicting_masks_raise_and_keep_env(rocm: pytest.MonkeyPatch) -> None:
    rocm.setenv("HIP_VISIBLE_DEVICES", "0")
    rocm.setenv("CUDA_VISIBLE_DEVICES", "1")

    with pytest.raises(ValueError, match="Conflicting GPU visibility masks"):
        neutralize_hip_visible_devices()
    assert _masks() == ("0", "1")


def test_empty_hip_mask_counts_as_unset(rocm: pytest.MonkeyPatch) -> None:
    rocm.setenv("HIP_VISIBLE_DEVICES", "")
    rocm.setenv("CUDA_VISIBLE_DEVICES", "1")

    assert neutralize_hip_visible_devices() is True
    assert _masks() == (None, "1")


def test_hip_mask_kept_when_vllm_is_unavailable(rocm: pytest.MonkeyPatch) -> None:
    # The fixture already makes ``import vllm.platforms.rocm`` fail; the HIP-only
    # restriction must survive as CUDA_VISIBLE_DEVICES instead of being dropped.
    with pytest.raises(ImportError):
        import vllm.platforms.rocm  # noqa: F401
    rocm.setenv("HIP_VISIBLE_DEVICES", "5")

    neutralize_hip_visible_devices()
    assert _masks() == (None, "5")


def test_non_rocm_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_rocm_build(monkeypatch, False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    assert neutralize_hip_visible_devices() is False
    assert _masks() == ("0", "1")


@pytest.mark.parametrize(
    ("ray_keyword", "expected"),
    [("HIP_VISIBLE_DEVICES", (True, ("2,3", None))), ("CUDA_VISIBLE_DEVICES", (False, (None, "2,3")))],
)
def test_cuda_mask_follows_ray_amd_keyword(
    rocm: pytest.MonkeyPatch, ray_keyword: str, expected: tuple
) -> None:
    # Ray >= 2.60 keys AMD GPUs on HIP only; Ray <= 2.56 used CUDA when HIP was unset.
    rocm.setattr(rocm_env, "_ray_amd_visible_devices_env_var", lambda: ray_keyword)
    rocm.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    moved, masks = expected
    assert align_visible_devices_with_ray() is moved
    assert _masks() == masks


def test_align_without_cuda_mask_is_noop(rocm: pytest.MonkeyPatch) -> None:
    rocm.setattr(rocm_env, "_ray_amd_visible_devices_env_var", lambda: "HIP_VISIBLE_DEVICES")

    assert align_visible_devices_with_ray() is False
    assert _masks() == (None, None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, (True, "1")), ("", (True, "1")), ("1", (False, "1")), ("0", (False, "0"))],
)
def test_legacy_ipc_mode_defaults_on_and_keeps_explicit_value(
    rocm: pytest.MonkeyPatch, value: str | None, expected: tuple
) -> None:
    if value is None:
        rocm.delenv("HSA_ENABLE_IPC_MODE_LEGACY", raising=False)
    else:
        rocm.setenv("HSA_ENABLE_IPC_MODE_LEGACY", value)

    applied, final = expected
    assert default_legacy_ipc_mode() is applied
    assert os.environ.get("HSA_ENABLE_IPC_MODE_LEGACY") == final


def test_legacy_ipc_mode_non_rocm_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_rocm_build(monkeypatch, False)
    monkeypatch.delenv("HSA_ENABLE_IPC_MODE_LEGACY", raising=False)

    assert default_legacy_ipc_mode() is False
    assert "HSA_ENABLE_IPC_MODE_LEGACY" not in os.environ
