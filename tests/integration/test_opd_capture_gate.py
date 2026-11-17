# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Request-level capture gating, including the installed MRV2 NPU transport."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration import vllm_runtime, specopd_vllm
from verl_speco.integration.verify_trace import CAPTURE_KEY
from verl_speco.transport.specopd_protocol import PLAN_KEY


@pytest.mark.parametrize("method", ["draft_opd", "spec_opd"])
@pytest.mark.parametrize("skip", [False, True])
def test_server_sends_capture_decision_in_native_sampling_params(monkeypatch, method, skip):
    monkeypatch.setattr(vllm_runtime, "_load_env_drafter_config", lambda: {
        "training": {"drafter_opd": {"enable": True, "method": method, "capture_verify_trace": True}},
    })
    class StopGeneration(Exception):
        pass
    class Server(vllm_runtime._SpecoVLLMHttpServerMixin):
        config = SimpleNamespace(response_length=32)
        model_config = SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=99))
        global_steps = 5
        _speco_serving_drafter_revision = 0
    async def generate(**kwargs):
        extra = kwargs["sampling_params"].get("extra_args", {})
        if method == "draft_opd":
            assert extra[CAPTURE_KEY] is not skip
            assert PLAN_KEY not in extra
        else:
            assert (PLAN_KEY in extra) is not skip
            assert CAPTURE_KEY not in extra
        assert "_verl_skip_drafter_collection" not in kwargs["sampling_params"]
        raise StopGeneration
    with pytest.raises(StopGeneration):
        asyncio.run(vllm_runtime._speco_generate_with_request_stats_unblocked(
            Server(), generate, (), {"prompt_ids": [1, 2], "request_id": "gate-test",
            "sampling_params": {"_verl_skip_drafter_collection": skip, "_specopd_drafter_revision": 0}},
        ))


@pytest.mark.parametrize("collect", [False, True])
def test_installed_mrv2_npu_copy_is_gated(monkeypatch, collect):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
    class Handler(DraftTokensHandler):
        pass
    monkeypatch.setattr(vllm_runtime, "_load_env_drafter_config", lambda: {
        "training": {"drafter_opd": {"enable": True, "method": "spec_opd"}},
    })
    specopd_vllm.patch_draft_token_handler(Handler)
    handler = object.__new__(Handler)
    handler._specopd_collecting_requests = {"r"} if collect else set()
    batch = SimpleNamespace(req_ids=["r"], has_structured_output_reqs=False)
    tokens = torch.tensor([[11, 12, 13]], device="npu:0")
    handler.set_draft_tokens(batch, tokens)
    result = handler.get_draft_tokens()
    assert result.draft_token_ids == ([[11, 12, 13]] if collect else [[-1, -1, -1]])
    assert (handler._specopd_real_draft_ids is not None) is collect
    assert not batch.has_structured_output_reqs
