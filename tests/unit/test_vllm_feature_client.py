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

import torch
import pytest

import verl_speco.producer.vllm_feature_client as client_module
from verl_speco.producer.hidden_states_store import HiddenStatesStore
from verl_speco.producer.vllm_feature_client import (
    RawVllmFeature,
    VllmEndpoint,
    VllmFeatureClientPool,
    VllmResponse,
    delete_temporary_result,
    load_hidden_state_result,
    request_generate,
)


class _RecordingStore(HiddenStatesStore):
    backend = "mooncake"

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.released: list[str] = []

    def load(self, reference: str):
        self.loaded.append(reference)
        return {"hidden_states": torch.zeros(1, dtype=torch.bfloat16)}, 2

    def release(self, reference: str) -> None:
        self.released.append(reference)


def test_prefill_reference_does_not_load_or_release_file(monkeypatch):
    endpoint = VllmEndpoint("http://vllm:8000/v1", 1)
    store = _RecordingStore()
    store.backend = "file"

    async def fake_prefill(*args, **kwargs):
        return VllmResponse("/shared/a.safetensors", endpoint.base_url)

    monkeypatch.setattr(client_module, "request_prefill", fake_prefill)
    pool = VllmFeatureClientPool(
        [endpoint], model="target", max_inflight_requests=1,
        request_timeout=10, hidden_states_store=store,
    )
    pool._states = [client_module._EndpointState(endpoint, object(), asyncio.Semaphore(1))]
    response = asyncio.run(pool.prefill_reference(SimpleNamespace(prompt_token_ids=[1, 2])))
    assert isinstance(response, VllmResponse)
    assert response.hidden_states_path == "/shared/a.safetensors"
    assert store.loaded == store.released == []
    assert pool._states[0].inflight == 0


def test_prefill_reference_transfers_handle_without_reading(monkeypatch):
    endpoint = VllmEndpoint("http://vllm:8000/v1", 1)
    store = _RecordingStore()
    params = {"handle": "hs-123", "version": 1, "metadata": {"rows": 2}}

    async def prefill(*args, **kwargs):
        return VllmResponse(None, endpoint.base_url, handle="hs-123", transfer_params=params)

    monkeypatch.setattr(client_module, "request_prefill", prefill)
    pool = VllmFeatureClientPool(
        [endpoint], model="target", max_inflight_requests=1,
        request_timeout=10, hidden_states_store=store,
    )
    pool._states = [client_module._EndpointState(endpoint, object(), asyncio.Semaphore(1))]
    response = asyncio.run(pool.prefill_reference(SimpleNamespace(prompt_token_ids=[1, 2])))
    assert response.transfer_params == params
    assert store.loaded == store.released == []
    delete_temporary_result(response)
    assert store.released == ["hs-123"]


def test_request_generate_returns_store_handle() -> None:
    import httpx

    async def run():
        def respond(request):
            return httpx.Response(200, json={
                "choices": [{"prompt_token_ids": [1, 2], "token_ids": [3, 4]}],
                "kv_transfer_params": {"handle": "hs-123"},
            })
        async with httpx.AsyncClient(
            base_url="http://vllm:8000/v1/", transport=httpx.MockTransport(respond)
        ) as client:
            return await request_generate(
                VllmEndpoint("http://vllm:8000/v1", 1), client, [1, 2],
                model="target", max_tokens=8, timeout=30,
            )

    response = asyncio.run(run())
    assert response.handle == "hs-123"
    assert response.hidden_states_path is None
    assert response.reference == "hs-123"
    assert response.is_handle is True


def test_load_and_delete_route_handles_through_the_store() -> None:
    store = _RecordingStore()
    response = VllmResponse(None, "http://vllm:8000/v1", handle="hs-9")

    raw = load_hidden_state_result(response, store)

    assert raw.temporary_path == "hs-9"
    assert raw.is_handle is True
    assert raw.byte_size == 2
    assert store.loaded == ["hs-9"]

    delete_temporary_result(raw)
    assert store.released == ["hs-9"]


@pytest.mark.parametrize("generate", [False, True])
@pytest.mark.parametrize("suffix", ["", "/"])
def test_direct_http_request_body_and_client_lifecycle(monkeypatch, generate, suffix):
    import json
    import httpx

    calls = []
    original_client = httpx.AsyncClient

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={
            "choices": [{"prompt_token_ids": [1, 2], "token_ids": [3, 4]}],
            "kv_transfer_params": {"hidden_states_path": "/tmp/result.safetensors"},
        })

    def create_client(**kwargs):
        assert kwargs["limits"].max_connections == 32
        assert kwargs["limits"].max_keepalive_connections == 32
        return original_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, "AsyncClient", create_client)

    async def run():
        pool = VllmFeatureClientPool(
            [VllmEndpoint("http://vllm:8000/v1" + suffix, 32)], model="target",
            max_inflight_requests=32, request_timeout=30,
        )
        await pool.start()
        client = pool._states[0].client
        request = SimpleNamespace(prompt_token_ids=[1, 2], max_tokens=128)
        try:
            for _ in range(2):
                response = await (pool.generate_reference(request) if generate
                                  else pool.prefill_reference(request))
                assert response.hidden_states_path == "/tmp/result.safetensors"
                if generate:
                    assert response.generated_token_ids == (3, 4)
                assert pool._states[0].client is client
        finally:
            await pool.close()
        assert client.is_closed

    asyncio.run(run())
    assert len(calls) == 2
    for request in calls:
        assert request.method == "POST"
        assert str(request.url) == "http://vllm:8000/v1/completions"
        assert request.headers["Authorization"] == "Bearer EMPTY"
        assert json.loads(request.content) == {
            "model": "target", "prompt": [1, 2],
            "max_tokens": 128 if generate else 1, "return_token_ids": True,
        }


def test_direct_http_errors_keep_status_and_response_body():
    import httpx

    async def run():
        async with httpx.AsyncClient(
            base_url="http://vllm:8000/v1/",
            transport=httpx.MockTransport(lambda request: httpx.Response(
                503, text="server busy", headers={"x-request-id": "req-7"}
            )),
        ) as client:
            with pytest.raises(httpx.HTTPStatusError) as error:
                await client_module.request_prefill(
                    VllmEndpoint("http://vllm:8000/v1", 1), client, [1, 2],
                    model="target", timeout=30,
                )
            assert client_module._http_error_detail(error.value) == (
                503, "req-7", "server busy"
            )

    asyncio.run(run())


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
        lambda response, store=None: RawVllmFeature(
            payload={},
            temporary_path=response.reference,
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


def test_pool_releases_reference_when_load_fails_before_retry(monkeypatch) -> None:
    endpoint = VllmEndpoint("http://vllm:8000/v1", 1)
    handles = iter(["hs-1", "hs-2"])
    load_calls = {"n": 0}

    async def fake_prefill(endpoint, client, prompt_token_ids, **kwargs):
        del client, prompt_token_ids, kwargs
        return VllmResponse(None, endpoint.base_url, handle=next(handles))

    def fake_load(response, store=None):
        del store
        load_calls["n"] += 1
        if load_calls["n"] == 1:
            raise RuntimeError("transient load failure")
        return RawVllmFeature(
            payload={},
            temporary_path=response.reference,
            endpoint_url=response.endpoint_url,
            byte_size=0,
        )

    async def no_backoff(_seconds):
        return None

    monkeypatch.setattr(client_module, "request_prefill", fake_prefill)
    monkeypatch.setattr(client_module, "load_hidden_state_result", fake_load)
    monkeypatch.setattr(client_module.asyncio, "sleep", no_backoff)

    store = _RecordingStore()
    pool = VllmFeatureClientPool(
        [endpoint],
        model="target",
        max_inflight_requests=1,
        request_timeout=10,
        hidden_states_store=store,
    )
    pool._states = [
        client_module._EndpointState(endpoint, object(), asyncio.Semaphore(1))
    ]

    raw = asyncio.run(
        pool.prefill(SimpleNamespace(prompt_token_ids=[1, 2], sample_id="sample-a"))
    )

    assert raw.temporary_path == "hs-2"
    assert store.released == ["hs-1"]
    assert load_calls["n"] == 2


def test_success_logging_is_rate_limited_per_endpoint_counter() -> None:
    pool = VllmFeatureClientPool(
        [VllmEndpoint("http://vllm:8000/v1", 1)],
        model="target",
        max_inflight_requests=1,
        request_timeout=10,
        success_log_interval=100,
    )

    assert [count for count in range(1, 202) if pool._should_log_success(count)] == [
        1,
        2,
        3,
        100,
        200,
    ]

    disabled = VllmFeatureClientPool(
        [VllmEndpoint("http://vllm:8000/v1", 1)],
        model="target",
        max_inflight_requests=1,
        request_timeout=10,
        success_log_interval=0,
    )
    assert disabled._should_log_success(1) is False


@pytest.mark.parametrize("cancel", [False, True])
def test_http_diagnostics_exclude_endpoint_wait_and_release_on_cancel(monkeypatch, cancel):
    clock = [0.0]
    monkeypatch.setattr(client_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    endpoint = VllmEndpoint("http://vllm:8000/v1", 1)
    store = _RecordingStore()
    store.backend = "file"
    pool = VllmFeatureClientPool([endpoint], model="target", max_inflight_requests=1,
                                 request_timeout=10, hidden_states_store=store)

    async def exercise():
        entered, finish = asyncio.Event(), asyncio.Event()
        async def prefill(*args, **kwargs):
            entered.set()
            await finish.wait()
            return VllmResponse("/tmp/hidden", endpoint.base_url)
        monkeypatch.setattr(client_module, "request_prefill", prefill)
        semaphore = asyncio.Semaphore(0)
        pool._states = [client_module._EndpointState(endpoint, object(), semaphore)]
        task = asyncio.create_task(pool.prefill_reference(SimpleNamespace(prompt_token_ids=[1])))
        await asyncio.sleep(0)
        clock[0] = 3.0
        assert pool.diagnostics.http_inflight == 0
        semaphore.release()
        await asyncio.wait_for(entered.wait(), 2)
        assert pool.diagnostics.http_inflight == 1
        clock[0] = 8.0
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            finish.set()
            response = await task
            assert response.timings["endpoint_wait"] == 3.0
            assert response.timings["http"] == 5.0
            assert response.timings["global_wait"] >= 0
        assert pool.diagnostics.http_inflight == 0
        assert pool.diagnostics.snapshot()["http_calls"] == 1

    asyncio.run(exercise())
