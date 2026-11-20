# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Request-level capture gating, including the installed MRV2 NPU transport."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration import vllm_runtime
from verl_speco.transport.specopd_protocol import PLAN_KEY


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
