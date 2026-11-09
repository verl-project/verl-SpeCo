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
"""ROCm DFlash verification samples from the target instead of greedy argmax."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

# sglang_patch imports torch at module level; the CPU CI image has none.
torch = pytest.importorskip("torch")

_STUBBED = (
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.engine",
    "sglang.srt.utils",
    "sglang.srt.speculative",
    "sglang.srt.speculative.dflash_utils",
)


@pytest.fixture
def sglang_patch(monkeypatch):
    """Import sglang_patch against stub sglang modules (it imports sglang eagerly)."""
    stubs = {name: types.ModuleType(name) for name in _STUBBED}
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(stubs[parent], child, module)
    stubs["sglang.srt.entrypoints.engine"].run_scheduler_process = lambda *a, **k: None
    stubs["sglang.srt.utils"].MultiprocessingSerializer = types.SimpleNamespace()
    # The HIP build of SGLang: no sgl_kernel sampling ops, verification falls back to greedy.
    dflash_utils = stubs["sglang.srt.speculative.dflash_utils"]
    dflash_utils._DFLASH_SAMPLING_VERIFY_AVAILABLE = False
    dflash_utils.top_k_renorm_prob = None
    dflash_utils.top_p_renorm_prob = None
    dflash_utils.tree_speculative_sampling_target_only = None
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.sglang_patch", raising=False
    )
    module = importlib.import_module("verl_speco.integration.sglang_patch")
    monkeypatch.setattr(module, "_SGLANG_ROCM_DFLASH_SAMPLING_PATCHED", False)
    yield module
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.sglang_patch", raising=False
    )


def _dflash_utils():
    return sys.modules["sglang.srt.speculative.dflash_utils"]


def test_rocm_installs_torch_target_sampler(monkeypatch, sglang_patch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising=False)

    sglang_patch.patch_sglang_rocm_dflash_target_sampling()

    dflash_utils = _dflash_utils()
    assert dflash_utils._DFLASH_SAMPLING_VERIFY_AVAILABLE is True
    assert (
        dflash_utils.tree_speculative_sampling_target_only
        is sglang_patch._tree_speculative_sampling_target_only_torch
    )
    assert dflash_utils.top_k_renorm_prob is sglang_patch._top_k_renorm_prob_torch
    assert dflash_utils.top_p_renorm_prob is sglang_patch._top_p_renorm_prob_torch


def test_non_rocm_leaves_sglang_untouched(monkeypatch, sglang_patch) -> None:
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    sglang_patch.patch_sglang_rocm_dflash_target_sampling()

    assert _dflash_utils()._DFLASH_SAMPLING_VERIFY_AVAILABLE is False
    assert _dflash_utils().tree_speculative_sampling_target_only is None


def test_native_kernel_is_kept(monkeypatch, sglang_patch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising=False)
    native = object()
    dflash_utils = _dflash_utils()
    dflash_utils._DFLASH_SAMPLING_VERIFY_AVAILABLE = True
    dflash_utils.tree_speculative_sampling_target_only = native

    sglang_patch.patch_sglang_rocm_dflash_target_sampling()

    assert dflash_utils.tree_speculative_sampling_target_only is native


def test_chain_verify_emits_target_distribution(sglang_patch) -> None:
    """The first two emitted tokens follow the target distribution, not its argmax."""
    torch.manual_seed(0)
    bs, n, vocab = 200_000, 4, 6
    # Same buffers as sglang's dflash_utils._get_or_create_chain_verify_buffers.
    retrieve_index = torch.arange(bs * n, dtype=torch.int64).view(bs, n)
    row_next = torch.arange(1, n + 1, dtype=torch.int64)
    row_next[-1] = -1
    retrieve_next_token = row_next.unsqueeze(0).expand(bs, -1).clone()
    retrieve_next_sibling = torch.full((bs, n), -1, dtype=torch.int64)
    predicts = torch.empty((bs * n,), dtype=torch.int32)
    accept_index = torch.empty((bs, n), dtype=torch.int32)
    accept_token_num = torch.empty((bs,), dtype=torch.int32)

    target = torch.softmax(torch.randn(n, vocab), dim=-1)
    target_probs = target.unsqueeze(0).expand(bs, -1, -1).contiguous()
    # A deterministic draft that proposes a non-argmax token first.
    draft = [0, int(target[0].argsort()[-2]), int(target[1].argmax()), 1]
    candidates = torch.tensor(draft, dtype=torch.int64).expand(bs, -1).contiguous()

    sglang_patch._tree_speculative_sampling_target_only_torch(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=torch.rand(bs, n),
        uniform_samples_for_final_sampling=torch.rand(bs),
        target_probs=target_probs,
        draft_probs=torch.zeros_like(target_probs),
    )

    # Emitted tokens: accepted drafts, then the bonus token (as dflash_info.verify).
    accepted = accept_token_num.long()
    rows = torch.arange(bs)
    bonus = predicts[accept_index[rows, accepted].long()].long()
    first = torch.where(accepted > 0, candidates[:, 1], bonus)
    first_freq = torch.bincount(first, minlength=vocab).double() / bs
    assert (first_freq - target[0].double()).abs().sum() < 0.02

    took_draft = accepted > 0
    second = torch.where(accepted > 1, candidates[:, 2], bonus)[took_draft]
    second_freq = torch.bincount(second, minlength=vocab).double() / len(second)
    assert (second_freq - target[1].double()).abs().sum() < 0.03
