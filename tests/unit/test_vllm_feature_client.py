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

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import verl_speco.producer.vllm_feature_client as client_module
from verl_speco.producer.vllm_feature_client import (
    RawVllmFeature,
    VllmEndpoint,
    VllmFeatureClientPool,
    VllmResponse,
    request_generate,
)


def test_request_generate_only_requests_generated_token_ids() -> None:
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        prompt_token_ids=[1, 2],
                        token_ids=[3, 4],
                    )
                ],
                kv_transfer_params={"hidden_states_path": "/tmp/result.safetensors"},
            )

    client = SimpleNamespace(completions=Completions())

    response = asyncio.run(
        request_generate(
            VllmEndpoint("http://vllm:8000/v1", 1),
            client,
            [1, 2],
            model="target",
            max_tokens=128,
            timeout=30,
        )
    )

    assert response.generated_token_ids == (3, 4)
    assert calls[0]["max_tokens"] == 128
    assert calls[0]["extra_body"] == {"return_token_ids": True}


def test_pool_retry_fails_over_to_another_endpoint(monkeypatch) -> None:
    calls: list[str] = []
    first = VllmEndpoint("http://vllm-a:8000/v1", 1)
    second = VllmEndpoint("http://vllm-b:8000/v1", 1)

    async def fake_prefill(endpoint, client, prompt_token_ids, **kwargs):
        del client, prompt_token_ids, kwargs
        calls.append(endpoint.base_url)
        if endpoint == first:
            raise ConnectionError("endpoint unavailable")
        return VllmResponse("/tmp/result.safetensors", endpoint.base_url)

    async def no_backoff(_seconds):
        return None

    monkeypatch.setattr(client_module, "request_prefill", fake_prefill)
    monkeypatch.setattr(client_module.asyncio, "sleep", no_backoff)
    monkeypatch.setattr(
        client_module,
        "load_hidden_state_result",
        lambda response: RawVllmFeature(
            payload={},
            temporary_path=response.hidden_states_path,
            endpoint_url=response.endpoint_url,
            byte_size=0,
        ),
    )
    pool = VllmFeatureClientPool(
        [first, second],
        model="target",
        max_inflight_requests=1,
        request_timeout=10,
    )
    pool._states = [
        client_module._EndpointState(endpoint, object(), asyncio.Semaphore(1))
        for endpoint in (first, second)
    ]

    raw = asyncio.run(
        pool.prefill(SimpleNamespace(prompt_token_ids=[1, 2], sample_id="sample-a"))
    )

    assert calls == [first.base_url, second.base_url]
    assert raw.endpoint_url == second.base_url
    assert pool._states[0].inflight == pool._states[1].inflight == 0
    assert pool._states[0].requests == 0
    assert pool._states[1].requests == 1
