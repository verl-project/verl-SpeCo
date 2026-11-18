# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-in tiny live-cache test, including Ascend FRACTAL_NZ storage.

No checkpoint, target model, distributed workers or RL task is needed.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from verl_speco.integration.deepseek_v4_online import (
    EmbeddedDraftLoader,
    apply_shared_token_weights,
    runtime_weight_name,
)
from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECO_RUN_NPU_TESTS") != "1",
    reason="Set SPECO_RUN_NPU_TESTS=1 to explicitly enable NPU execution",
)


@pytest.fixture
def native_internal_format(storage_format):
    if storage_format != 29:
        yield
        return
    import torch_npu

    # _npuConfig implements only __setattr__, not a readable Python attribute.
    # Read the native option; monkeypatch cannot capture this write-only setting.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    if isinstance(previous, bytes):
        previous = previous.decode()
    try:
        # Match native NPUModelRunner initialization for genuine NZ coverage.
        torch.npu.config.allow_internal_format = True
        yield
    finally:
        # An unset/empty native option has the same disabled default.
        torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": previous or "disable"})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("storage_format", [2, 29])
@pytest.mark.usefixtures("native_internal_format")
def test_live_cache_commit_and_restore_match_cold_router(dtype, storage_format):
    import torch_npu

    torch.npu.set_device(0)
    model = nn.Module()
    model.gate = nn.Linear(32, 16, bias=False, dtype=dtype, device="npu:0")
    model.gate.precast_fp32_weight = True
    with torch.no_grad():
        model.gate.weight.fill_(1)
    model.gate.weight_fp32 = torch_npu.npu_format_cast(
        model.gate.weight.detach().float().clone(), storage_format
    )
    cache = model.gate.weight_fp32
    assert torch_npu.get_npu_format(cache) == storage_format, (
        "Cache setup did not create the requested format; no update has run"
    )
    pointers = model.gate.weight.data_ptr(), cache.data_ptr()
    x = torch.arange(64, dtype=torch.float32, device="npu:0").reshape(2, 32) / 64
    for fill in (2, 3):
        loader = EmbeddedDraftLoader(model)
        loader.receive(
            [
                (
                    runtime_weight_name(0, 1, "gate.weight"),
                    torch.full((16, 32), fill, dtype=dtype),
                )
            ]
        )
        assert loader.finish() == 1
        torch.npu.synchronize()
        assert model.gate.weight_fp32 is cache
        assert (model.gate.weight.data_ptr(), cache.data_ptr()) == pointers
        assert torch_npu.get_npu_format(cache) == storage_format
        cold = torch.full((16, 32), fill, dtype=torch.float32, device="npu:0")
        torch.testing.assert_close(
            F.linear(x, cache), F.linear(x, cold), rtol=0, atol=0
        )
    # The same helper is used after checkpoint restore and optimizer updates.
    with torch.no_grad():
        model.gate.weight.fill_(1)
    refresh_fp32_weight_caches(model)
    torch.npu.synchronize()
    assert (model.gate.weight.data_ptr(), cache.data_ptr()) == pointers
    assert torch_npu.get_npu_format(cache) == storage_format
    torch.testing.assert_close(
        F.linear(x, cache), F.linear(x, torch.ones_like(cold)), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("storage_format", [2, 29])
@pytest.mark.usefixtures("native_internal_format")
def test_actor_shared_token_sync_preserves_native_format_and_cache(
    dtype, storage_format
):
    import torch_npu

    torch.npu.set_device(0)
    wrapper = nn.Module()
    wrapper.runtime_contract = SimpleNamespace(vocab_size=5, hidden_size=32)
    wrapper.model = nn.Module()
    wrapper.model.model = nn.Module()
    wrapper.model.model.embed_tokens = nn.Embedding(4, 32, device="npu:0", dtype=dtype)
    wrapper.model.lm_head = nn.Linear(32, 4, bias=False, device="npu:0", dtype=dtype)
    modules = [wrapper.model.model.embed_tokens, wrapper.model.lm_head]
    wrapper.requires_grad_(False)
    for module in modules:
        module.shard_indices = SimpleNamespace(
            org_vocab_start_index=3, org_vocab_end_index=5
        )
        module.weight.data = torch_npu.npu_format_cast(
            module.weight.data, storage_format
        )
        assert int(torch_npu.get_npu_format(module.weight)) == storage_format
    head = modules[1]
    head.precast_fp32_weight = True
    head.weight_fp32 = torch_npu.npu_format_cast(
        head.weight.float().clone(), storage_format
    )
    pointers = [m.weight.data_ptr() for m in modules] + [head.weight_fp32.data_ptr()]
    for version in [1, 2]:
        state = {
            "embed_tokens.weight": torch.full((5, 32), float(version)),
            "lm_head.weight": torch.full((5, 32), float(version * 3)),
        }
        apply_shared_token_weights(wrapper, state, tp_rank=1, tp_size=2)
        torch.npu.synchronize()
        for module, name in zip(modules, state, strict=True):
            assert torch.equal(module.weight[:2].cpu(), state[name][3:5].to(dtype))
            assert module.weight[2:].cpu().eq(0).all()
            assert int(torch_npu.get_npu_format(module.weight)) == storage_format
        assert pointers == [m.weight.data_ptr() for m in modules] + [
            head.weight_fp32.data_ptr()
        ]
        assert int(torch_npu.get_npu_format(head.weight_fp32)) == storage_format
        torch.testing.assert_close(
            head.weight_fp32.cpu(), head.weight.cpu().float(), rtol=0, atol=0
        )
