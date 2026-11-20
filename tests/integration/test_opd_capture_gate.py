# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Request-level capture gating, including the installed MRV2 NPU transport."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration import vllm_runtime
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


@pytest.mark.parametrize("quota,local_cap,expected", [(10, 16, 10), (2, 1, 1), (0, 0, 0)])
def test_specopd_global_quota_gates_before_generation(monkeypatch, quota, local_cap, expected):
    monkeypatch.setattr(vllm_runtime, "_load_env_drafter_config", lambda: {
        "training": {"specopd": {"enable": True, "max_collect_samples_per_step": 80},
                     "max_collect_samples_per_step_per_replica": local_cap},
    })

    class StopGeneration(Exception):
        pass

    class Server(vllm_runtime._SpecoVLLMHttpServerMixin):
        config = SimpleNamespace(response_length=32)
        model_config = SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=99))
        global_steps = 5
        _speco_serving_drafter_revision = 0

    server = Server()
    enabled = []

    async def generate(**kwargs):
        plan = kwargs["sampling_params"]["extra_args"][PLAN_KEY]
        enabled.append(plan["max_branches"] > 0)
        raise StopGeneration

    async def request(index):
        await vllm_runtime._speco_generate_with_request_stats_unblocked(
            server, generate, (), {"prompt_ids": [1, 2], "request_id": str(index),
                                   "sampling_params": {"_specopd_drafter_revision": 0}},
        )

    async def run():
        with pytest.raises(RuntimeError, match="acknowledged replica quota"):
            await request(0)
        for epoch in (5, 10):
            server.global_steps = epoch
            enabled.clear()
            for index in range(18):
                if index in (0, 1):
                    reply = await server.speco_set_serving_drafter_revision(0, collection_quota=quota)
                    assert reply["collection_quota"] == quota
                with pytest.raises(StopGeneration):
                    await request(index)
            assert enabled == [True] * expected + [False] * (18 - expected)

    asyncio.run(run())
