# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from verl_speco.integration import agent_loop_runtime


def test_agent_loop_patch_supports_release_v080_llm_server_client(monkeypatch) -> None:
    class AgentLoopWorker:
        async def generate_sequences(self, batch):
            return batch

        async def _run_agent_loop(self, sampling_params, trajectory):
            return sampling_params, trajectory

        def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
            return inputs, input_non_tensor_batch, validate

    class AgentLoopManager:
        def __init__(self):
            self.config = None

        async def generate_sequences(self, prompts):
            return prompts

    class LLMServerClient:
        async def generate(self, *, sampling_params):
            return sampling_params

    release_v080_module = SimpleNamespace(
        AgentLoopWorker=AgentLoopWorker,
        AgentLoopManager=AgentLoopManager,
        LLMServerClient=LLMServerClient,
    )
    monkeypatch.setattr(agent_loop_runtime, "_PATCHED", False)
    monkeypatch.setattr(agent_loop_runtime, "_load_agent_loop_module", lambda: release_v080_module)

    assert agent_loop_runtime.install_agent_loop_runtime_patch() is True
    assert LLMServerClient._speco_patched_generate is True
    assert AgentLoopWorker._speco_patched_generate_sequences is True
    assert AgentLoopManager._speco_patched_generate_sequences is True

    global_steps_token = agent_loop_runtime._CURRENT_GLOBAL_STEPS.set(17)
    try:
        sampling_params = asyncio.run(LLMServerClient().generate(sampling_params={"temperature": 1.0}))
    finally:
        agent_loop_runtime._CURRENT_GLOBAL_STEPS.reset(global_steps_token)
    assert sampling_params["_verl_global_steps"] == 17
