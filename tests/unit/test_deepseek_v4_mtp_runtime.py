# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Guard MTP gradient reconnection without replacing native forward values."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl_speco.models.deepseek_v4 import modeling_dspark, modeling_mtp


def test_mtp_installs_shared_dsa_and_rope_adapters(monkeypatch):
    calls = []
    monkeypatch.setattr(
        modeling_mtp,
        "DeepseekV4ServingTrainingWrapper",
        lambda cfg: torch.nn.Linear(2, 2),
    )
    monkeypatch.setattr(
        modeling_mtp,
        "_install_dspark_training_safe_rope",
        lambda: calls.append("rope") or True,
    )
    monkeypatch.setattr(
        modeling_mtp,
        "_install_dspark_dsa_query_training_replay",
        lambda: calls.append("dsa") or True,
    )
    model = modeling_mtp.DeepseekV4MTP(SimpleNamespace())
    assert calls == ["rope", "dsa"]
    assert model.using_training_dsa_query_replay
    monkeypatch.setattr(
        modeling_mtp, "_install_dspark_dsa_query_training_replay", lambda: False
    )
    with pytest.raises(RuntimeError, match="MTP query-gradient"):
        modeling_mtp.DeepseekV4MTP(SimpleNamespace())


def test_shared_prolog_retains_native_values_but_reconnects_query_gradient(monkeypatch):
    native_q, native_qr = torch.randn(2, 3), torch.randn(2, 3)

    class DSA:
        def _mla_prolog_single_stream(self, hidden, cos, sin):
            # Mimics an inference operator that returns detached query tensors.
            return native_q, native_qr, None

    npu = ModuleType("torch_npu")
    npu.npu_rotary_mul = lambda *args: None
    dsa = ModuleType("vllm_ascend.attention.dsa_v1")
    dsa.AscendDSAImpl = DSA
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.dsa_v1", dsa)
    weight = torch.randn(3, 3, requires_grad=True)
    hidden = torch.randn(2, 3, requires_grad=True)
    monkeypatch.setattr(
        modeling_dspark,
        "_replay_dspark_dsa_query",
        lambda self, x, *args: (x @ weight, x * 2),
    )
    assert modeling_dspark._install_dspark_dsa_query_training_replay()
    q, qr, scale = DSA()._mla_prolog_single_stream(hidden, None, None)
    assert torch.equal(q, native_q) and torch.equal(qr, native_qr)
    assert scale is None
    (q.sum() + qr.sum()).backward()
    expected_weight = hidden.detach().sum(0).unsqueeze(1).expand(3, 3)
    torch.testing.assert_close(weight.grad, expected_weight)
    torch.testing.assert_close(hidden.grad, weight.detach().sum(1).expand(2, 3) + 2)
    with torch.no_grad():
        q, qr, _ = DSA()._mla_prolog_single_stream(hidden, None, None)
    assert q is native_q and qr is native_qr
