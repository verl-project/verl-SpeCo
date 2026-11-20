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
import importlib.util
import logging
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


def test_rocm_launches_linear_triton_kernel_without_ascend_options(
    monkeypatch, sglang_patch
) -> None:
    launches = []

    class _Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, kwargs))

    monkeypatch.setattr(sglang_patch, "_linear_triton_on_rocm", lambda probs: True)
    monkeypatch.setattr(
        sglang_patch, "_tree_speculative_sampling_target_only_linear_triton_kernel", _Kernel()
    )
    monkeypatch.setattr(
        sglang_patch,
        "_triton_ascend_available",
        lambda: pytest.fail("ROCm must not probe triton-ascend"),
    )
    bs, n, vocab = 3, 4, 8

    assert sglang_patch._try_tree_speculative_sampling_target_only_linear_triton(
        predicts=torch.empty(bs * n, dtype=torch.int32),
        accept_index=torch.empty((bs, n), dtype=torch.int32),
        accept_token_num=torch.empty(bs, dtype=torch.int32),
        candidates=torch.zeros((bs, n), dtype=torch.int64),
        retrive_index=torch.arange(bs * n).view(bs, n),
        retrive_next_token=torch.zeros((bs, n), dtype=torch.int64),
        uniform_samples=torch.rand(bs, n),
        uniform_samples_for_final_sampling=torch.rand(bs),
        target_probs=torch.full((bs, n, vocab), 1.0 / vocab),
        threshold_single=1.0,
        threshold_acc=1.0,
    )
    [(grid, kwargs)] = launches
    assert grid == (bs,)
    assert "multibuffer" not in kwargs


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


def _chain_buffers(bs: int, n: int, device="cpu") -> dict:
    # Same buffers as sglang's dflash_utils._get_or_create_chain_verify_buffers.
    row_next = torch.arange(1, n + 1, dtype=torch.int64, device=device)
    row_next[-1] = -1
    return {
        "retrive_index": torch.arange(bs * n, dtype=torch.int64, device=device).view(bs, n),
        "retrive_next_token": row_next.unsqueeze(0).expand(bs, -1).contiguous(),
        "retrive_next_sibling": torch.full((bs, n), -1, dtype=torch.int64, device=device),
    }


def _verify(sglang_patch, path: str, candidates, target_probs, uniforms, final_uniforms):
    """Run one verify path; returns (accept_token_num, emitted tokens with -1 padding)."""
    bs, n = candidates.shape
    device = candidates.device
    out = {
        "predicts": torch.full((bs * n,), -1, dtype=torch.int32, device=device),
        "accept_index": torch.full((bs, n), -1, dtype=torch.int32, device=device),
        "accept_token_num": torch.zeros((bs,), dtype=torch.int32, device=device),
    }
    buffers = _chain_buffers(bs, n, device)
    common = dict(
        **out,
        candidates=candidates,
        retrive_index=buffers["retrive_index"],
        retrive_next_token=buffers["retrive_next_token"],
        uniform_samples=uniforms,
        uniform_samples_for_final_sampling=final_uniforms,
        target_probs=target_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
    )
    if path == "triton":
        assert sglang_patch._try_tree_speculative_sampling_target_only_linear_triton(**common)
    else:
        sglang_patch._tree_speculative_sampling_target_only_linear_torch(
            **common,
            retrive_next_sibling=buffers["retrive_next_sibling"],
            draft_probs=torch.zeros_like(target_probs),
        )
    # Emitted tokens: accepted drafts, then the bonus token (as dflash_info.verify).
    accepted = out["accept_token_num"].long()
    rows = torch.arange(bs, device=device)
    bonus = out["predicts"][out["accept_index"][rows, accepted].long()].long()
    pos = torch.arange(n, device=device).unsqueeze(0)
    shifted = torch.cat([candidates[:, 1:], candidates[:, :1]], dim=1)
    emitted = torch.where(pos < accepted.unsqueeze(1), shifted, -1)
    emitted = torch.where(pos == accepted.unsqueeze(1), bonus.unsqueeze(1), emitted)
    return accepted, emitted


def test_rocm_kernel_launch_failure_warns_and_uses_torch(
    monkeypatch, sglang_patch, caplog
) -> None:
    class _BrokenKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                raise RuntimeError("hipErrorNoBinaryForGpu")

            return _launch

    monkeypatch.setattr(sglang_patch, "_linear_triton_on_rocm", lambda probs: True)
    monkeypatch.setattr(
        sglang_patch, "_tree_speculative_sampling_target_only_linear_triton_kernel", _BrokenKernel()
    )
    torch.manual_seed(0)
    bs, n, vocab = 64, 4, 8
    target_probs = torch.softmax(torch.randn(bs, n, vocab), dim=-1)
    candidates = torch.randint(0, vocab, (bs, n))
    uniforms, final_uniforms = torch.rand(bs, n), torch.rand(bs)
    buffers = _chain_buffers(bs, n)

    def _run(fn, **extra):
        out = {
            "predicts": torch.full((bs * n,), -1, dtype=torch.int32),
            "accept_index": torch.full((bs, n), -1, dtype=torch.int32),
            "accept_token_num": torch.zeros((bs,), dtype=torch.int32),
        }
        fn(
            **out,
            **buffers,
            candidates=candidates,
            uniform_samples=uniforms,
            uniform_samples_for_final_sampling=final_uniforms,
            target_probs=target_probs,
            draft_probs=torch.zeros_like(target_probs),
            **extra,
        )
        return out

    with caplog.at_level(logging.WARNING, logger=sglang_patch.logger.name):
        got = _run(sglang_patch._tree_speculative_sampling_target_only_torch)
        _run(sglang_patch._tree_speculative_sampling_target_only_torch)
    expected = _run(sglang_patch._tree_speculative_sampling_target_only_linear_torch)

    warnings = [r for r in caplog.records if "falling back" in r.getMessage()]
    assert len(warnings) == 1
    assert "hipErrorNoBinaryForGpu" in warnings[0].getMessage()
    for name in expected:
        assert torch.equal(got[name], expected[name]), name


def test_rocm_missing_kernel_warns(monkeypatch, sglang_patch, caplog) -> None:
    monkeypatch.setattr(sglang_patch, "_linear_triton_on_rocm", lambda probs: True)
    monkeypatch.setattr(
        sglang_patch, "_tree_speculative_sampling_target_only_linear_triton_kernel", None
    )
    bs, n = 2, 4

    with caplog.at_level(logging.WARNING, logger=sglang_patch.logger.name):
        assert not sglang_patch._try_tree_speculative_sampling_target_only_linear_triton(
            predicts=torch.empty(bs * n, dtype=torch.int32),
            accept_index=torch.empty((bs, n), dtype=torch.int32),
            accept_token_num=torch.empty(bs, dtype=torch.int32),
            candidates=torch.zeros((bs, n), dtype=torch.int64),
            uniform_samples=torch.rand(bs, n),
            uniform_samples_for_final_sampling=torch.rand(bs),
            target_probs=torch.full((bs, n, 8), 1.0 / 8),
            threshold_single=1.0,
            threshold_acc=1.0,
            **{k: v for k, v in _chain_buffers(bs, n).items() if k != "retrive_next_sibling"},
        )
    assert any("kernel_not_compiled" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("env", "enabled", "deprecated_warning"),
    [
        ({}, True, False),
        ({"VERL_SGLANG_LINEAR_TRITON_VERIFY": "0"}, False, False),
        ({"VERL_SGLANG_NPU_EAGLE_LINEAR_TRITON": "0"}, False, True),
        (
            {"VERL_SGLANG_LINEAR_TRITON_VERIFY": "1", "VERL_SGLANG_NPU_EAGLE_LINEAR_TRITON": "0"},
            True,
            False,
        ),
    ],
)
def test_linear_triton_switch_and_deprecated_alias(
    monkeypatch, sglang_patch, caplog, env, enabled, deprecated_warning
) -> None:
    for name in ("VERL_SGLANG_LINEAR_TRITON_VERIFY", "VERL_SGLANG_NPU_EAGLE_LINEAR_TRITON"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with caplog.at_level(logging.WARNING, logger=sglang_patch.logger.name):
        assert sglang_patch._sglang_npu_eagle_linear_triton_enabled() is enabled
    assert any("deprecated" in r.getMessage() for r in caplog.records) is deprecated_warning


_ROCM_GPU = (
    bool(getattr(torch.version, "hip", None))
    and torch.cuda.is_available()
    and importlib.util.find_spec("triton") is not None
)
rocm_gpu = pytest.mark.skipif(not _ROCM_GPU, reason="needs a ROCm GPU with Triton")


def _tv(tokens: torch.Tensor, probs: torch.Tensor) -> float:
    freq = torch.bincount(tokens, minlength=probs.numel()).double() / tokens.numel()
    return float((freq - probs.double()).abs().sum() / 2)


def _two_sample_tv(a: torch.Tensor, b: torch.Tensor, vocab: int) -> float:
    fa = torch.bincount(a, minlength=vocab).double() / a.numel()
    fb = torch.bincount(b, minlength=vocab).double() / b.numel()
    return float((fa - fb).abs().sum() / 2)


def _run_both(sglang_patch, candidates, target_probs, rounds: int):
    """Triton vs torch: identical uniforms must give identical tokens; returns sample batches."""
    bs, n = candidates.shape
    samples = {"triton": [], "torch": [], "torch_indep": []}
    for _ in range(rounds):
        uniforms = torch.rand(bs, n, device="cuda")
        final_uniforms = torch.rand(bs, device="cuda")
        tri = _verify(sglang_patch, "triton", candidates, target_probs, uniforms, final_uniforms)
        ref = _verify(sglang_patch, "torch", candidates, target_probs, uniforms, final_uniforms)
        # Same uniforms, same tokens, up to float rounding where a uniform lands on a CDF
        # boundary (the kernel sums SUB_BLOCK partial sums, torch uses one cumsum).
        mismatch = float((tri[1] != ref[1]).any(dim=1).double().mean())
        assert mismatch < 1e-3, f"triton vs torch token mismatch rate {mismatch:.2e}"
        indep = _verify(
            sglang_patch,
            "torch",
            candidates,
            target_probs,
            torch.rand(bs, n, device="cuda"),
            torch.rand(bs, device="cuda"),
        )
        samples["triton"].append(tri)
        samples["torch"].append(ref)
        samples["torch_indep"].append(indep)
    return {
        k: (torch.cat([a for a, _ in v]), torch.cat([e for _, e in v])) for k, v in samples.items()
    }


@rocm_gpu
def test_rocm_triton_verify_all_accepted(sglang_patch) -> None:
    torch.manual_seed(0)
    bs, n, vocab = 8192, 4, 32
    draft = [0, 5, 17, 9]
    target = torch.zeros(n, vocab)
    for pos in range(n - 1):
        target[pos, draft[pos + 1]] = 1.0  # the target agrees with every draft token
    target[n - 1] = torch.softmax(torch.randn(vocab), dim=-1)
    target_probs = target.cuda().unsqueeze(0).expand(bs, -1, -1).contiguous()
    candidates = torch.tensor(draft, device="cuda").expand(bs, -1).contiguous()

    samples = _run_both(sglang_patch, candidates, target_probs, rounds=4)

    accepted, emitted = samples["triton"]
    assert bool((accepted == n - 1).all())
    assert bool((emitted[:, : n - 1] == torch.tensor(draft[1:], device="cuda")).all())
    bonus = emitted[:, n - 1]
    assert _tv(bonus.cpu(), target[n - 1]) < 0.02
    assert _two_sample_tv(bonus, samples["torch_indep"][1][:, n - 1], vocab) < 0.03


@rocm_gpu
def test_rocm_triton_verify_first_rejected(sglang_patch) -> None:
    torch.manual_seed(1)
    bs, n, vocab = 8192, 4, 32
    target = torch.softmax(torch.randn(n, vocab), dim=-1)
    draft = [0, 3, 7, 11]
    target[0, draft[1]] = 0.0  # the first draft token is impossible under the target
    target[0] /= target[0].sum()
    target_probs = target.cuda().unsqueeze(0).expand(bs, -1, -1).contiguous()
    candidates = torch.tensor(draft, device="cuda").expand(bs, -1).contiguous()

    samples = _run_both(sglang_patch, candidates, target_probs, rounds=4)

    accepted, emitted = samples["triton"]
    assert bool((accepted == 0).all())
    bonus = emitted[:, 0]
    assert bool((bonus != draft[1]).all())
    assert _tv(bonus.cpu(), target[0]) < 0.02
    assert _two_sample_tv(bonus, samples["torch_indep"][1][:, 0], vocab) < 0.03


@rocm_gpu
def test_rocm_triton_verify_sparse_top_k(sglang_patch) -> None:
    """Top-k-filtered target rows over a real vocab: almost every probability is exactly 0."""
    torch.manual_seed(2)
    bs, n, vocab, k = 4096, 4, 151936, 8
    target = torch.zeros(n, vocab)
    support = torch.stack([torch.randperm(vocab)[:k] for _ in range(n)])
    target.scatter_(1, support, torch.softmax(torch.randn(n, k), dim=-1))
    # Draft: the most likely token at position 0 (often accepted), outside the support at
    # position 1 (always rejected there), in the support at position 2 (not reached).
    outside = int(next(t for t in range(vocab) if t not in set(support[1].tolist())))
    draft = [0, int(target[0].argmax()), outside, int(support[2, 0])]
    target_probs = target.cuda().unsqueeze(0).expand(bs, -1, -1).contiguous()
    candidates = torch.tensor(draft, device="cuda").expand(bs, -1).contiguous()

    samples = _run_both(sglang_patch, candidates, target_probs, rounds=4)
    del target_probs

    accepted, emitted = samples["triton"]
    assert bool((accepted <= 1).all())
    emitted = emitted.cpu()
    for pos in range(2):
        tokens = emitted[:, pos][emitted[:, pos] >= 0]
        assert torch.isin(tokens, support[pos]).all(), f"token outside top-k at {pos}"
    first = emitted[:, 0]
    assert _tv(first, target[0]) < 0.03
    assert _two_sample_tv(first, samples["torch_indep"][1][:, 0].cpu(), vocab) < 0.04
    second = emitted[:, 1][emitted[:, 1] >= 0]
    assert second.numel() > 2000
    assert _tv(second, target[1]) < 0.05
