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
"""Bounded asynchronous vLLM hidden-state requests for the Producer."""

from __future__ import annotations

import asyncio
import errno
import importlib
import inspect
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_SECONDS = 2


@dataclass(frozen=True)
class VllmEndpoint:
    base_url: str
    max_concurrency: int

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("VllmEndpoint.base_url must not be empty")
        if self.max_concurrency <= 0:
            raise ValueError("VllmEndpoint.max_concurrency must be positive")


@dataclass(frozen=True)
class VllmResponse:
    hidden_states_path: str
    endpoint_url: str
    generated_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RawVllmFeature:
    payload: dict[str, Any]
    temporary_path: str
    endpoint_url: str
    byte_size: int
    generated_token_ids: tuple[int, ...] = ()


@dataclass
class _EndpointState:
    endpoint: VllmEndpoint
    client: Any
    semaphore: asyncio.Semaphore
    inflight: int = 0
    requests: int = 0


async def request_prefill(
    endpoint: VllmEndpoint,
    client: Any,
    prompt_token_ids: list[int],
    *,
    model: str,
    timeout: float,
) -> VllmResponse:
    response = await client.completions.create(
        model=model,
        prompt=prompt_token_ids,
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=timeout,
    )
    choices = getattr(response, "choices", None) or []
    if choices:
        actual = getattr(choices[0], "prompt_token_ids", None)
        if actual is not None and list(actual) != prompt_token_ids:
            raise ValueError("vLLM prompt_token_ids mismatch")
    params = getattr(response, "kv_transfer_params", None)
    if not isinstance(params, Mapping):
        raise ValueError("vLLM response missing kv_transfer_params")
    path = params.get("hidden_states_path")
    if not path:
        raise ValueError("vLLM response missing hidden_states_path")
    return VllmResponse(os.fspath(path), endpoint.base_url)


async def request_generate(
    endpoint: VllmEndpoint,
    client: Any,
    prompt_token_ids: list[int],
    *,
    model: str,
    max_tokens: int,
    timeout: float,
) -> VllmResponse:
    """Generate response tokens; hidden states are collected by a later prefill."""

    response = await client.completions.create(
        model=model,
        prompt=prompt_token_ids,
        max_tokens=max_tokens,
        extra_body={"return_token_ids": True},
        timeout=timeout,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("vLLM generation response has no choices")
    actual_prompt = getattr(choices[0], "prompt_token_ids", None)
    if actual_prompt is not None and list(actual_prompt) != prompt_token_ids:
        raise ValueError("vLLM generation prompt_token_ids mismatch")
    generated = getattr(choices[0], "token_ids", None)
    if not isinstance(generated, (list, tuple)) or not generated:
        raise ValueError(
            "vLLM generation response missing token_ids; enable return_token_ids support"
        )
    params = getattr(response, "kv_transfer_params", None)
    if not isinstance(params, Mapping):
        raise ValueError("vLLM generation response missing kv_transfer_params")
    path = params.get("hidden_states_path")
    if not path:
        raise ValueError("vLLM generation response missing hidden_states_path")
    return VllmResponse(
        os.fspath(path),
        endpoint.base_url,
        tuple(int(token_id) for token_id in generated),
    )


def load_hidden_state_result(response: VllmResponse) -> RawVllmFeature:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("vLLM Producer requires safetensors") from exc
    path = Path(response.hidden_states_path)
    _wait_for_lock(Path(f"{path}.lock"))
    if not path.is_file():
        raise FileNotFoundError(f"vLLM hidden-states file not found: {path}")
    return RawVllmFeature(
        payload=dict(load_file(str(path), device="cpu")),
        temporary_path=str(path),
        endpoint_url=response.endpoint_url,
        byte_size=int(path.stat().st_size),
        generated_token_ids=response.generated_token_ids,
    )


def delete_temporary_result(raw: RawVllmFeature) -> None:
    path = Path(raw.temporary_path)
    path.unlink(missing_ok=True)
    Path(f"{path}.lock").unlink(missing_ok=True)


def choose_endpoint(states: Sequence[_EndpointState]) -> _EndpointState:
    if not states:
        raise RuntimeError("No vLLM endpoints are configured")
    return min(states, key=lambda state: (state.inflight, state.requests))


class VllmFeatureClientPool:
    def __init__(
        self,
        endpoints: Sequence[VllmEndpoint],
        *,
        model: str,
        max_inflight_requests: int,
        request_timeout: float,
        success_log_interval: int = 100,
    ) -> None:
        if not endpoints:
            raise ValueError("At least one vLLM endpoint is required")
        if max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive")
        if not model:
            raise ValueError("vllm_model must not be empty")
        self.endpoints = list(endpoints)
        self.model = model
        self.request_timeout = float(request_timeout)
        self.success_log_interval = int(success_log_interval)
        if self.success_log_interval < 0:
            raise ValueError("success_log_interval must be non-negative")
        self._global_semaphore = asyncio.Semaphore(max_inflight_requests)
        self._states: list[_EndpointState] = []

    async def start(self) -> None:
        if self._states:
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("vLLM Producer requires the openai package") from exc
        # Suppress the SDK transport's one-line INFO message for every 2xx.
        # This pool emits endpoint-aware, rate-limited success logs below.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpx2").setLevel(logging.WARNING)
        self._states = [
            _EndpointState(
                endpoint=endpoint,
                client=AsyncOpenAI(
                    base_url=endpoint.base_url,
                    api_key="EMPTY",
                    max_retries=0,
                ),
                semaphore=asyncio.Semaphore(endpoint.max_concurrency),
            )
            for endpoint in self.endpoints
        ]

    async def prefill(self, request: Any) -> RawVllmFeature:
        return await self._request(request, generate=False)

    async def generate(self, request: Any) -> RawVllmFeature:
        return await self._request(request, generate=True)

    async def _request(self, request: Any, *, generate: bool) -> RawVllmFeature:
        if not self._states:
            raise RuntimeError("VllmFeatureClientPool.start() must be called first")
        async with self._global_semaphore:
            return await self._request_with_retries(request, generate=generate)

    async def _request_with_retries(
        self,
        request: Any,
        *,
        generate: bool,
    ) -> RawVllmFeature:
        total_attempts = _DEFAULT_MAX_RETRIES + 1
        failed_in_round: set[str] = set()
        for attempt in range(1, total_attempts + 1):
            candidates = [
                state
                for state in self._states
                if state.endpoint.base_url not in failed_in_round
            ]
            if not candidates:
                failed_in_round.clear()
                candidates = self._states
            state = choose_endpoint(candidates)
            state.inflight += 1
            request_started = time.monotonic()
            try:
                async with state.semaphore:
                    if generate:
                        response = await request_generate(
                            state.endpoint,
                            state.client,
                            list(request.prompt_token_ids),
                            model=self.model,
                            max_tokens=int(request.max_tokens),
                            timeout=self.request_timeout,
                        )
                    else:
                        response = await request_prefill(
                            state.endpoint,
                            state.client,
                            list(request.prompt_token_ids),
                            model=self.model,
                            timeout=self.request_timeout,
                        )
                    raw = await asyncio.to_thread(load_hidden_state_result, response)
                state.requests += 1
                if self._should_log_success(state.requests):
                    logger.info(
                        "vLLM endpoint request succeeded endpoint=%s "
                        "request_type=%s successful_requests=%s inflight=%s "
                        "elapsed=%.3fs",
                        state.endpoint.base_url,
                        "generate" if generate else "prefill",
                        state.requests,
                        state.inflight,
                        time.monotonic() - request_started,
                    )
                return raw
            except ValueError:
                # Response validation failures are deterministic protocol/data
                # errors, equivalent to speculators' InvalidResponseError.
                raise
            except Exception as exc:
                failed_in_round.add(state.endpoint.base_url)
                if attempt >= total_attempts:
                    status_code, request_id, response_body = _http_error_detail(exc)
                    logger.error(
                        "vLLM request failed after %s attempts last_endpoint=%s "
                        "request_type=%s sample_id=%s status_code=%s request_id=%s "
                        "response_body=%s elapsed=%.3fs error=%s",
                        total_attempts,
                        state.endpoint.base_url,
                        "generate" if generate else "prefill",
                        getattr(request, "sample_id", None),
                        status_code,
                        request_id,
                        response_body,
                        time.monotonic() - request_started,
                        exc,
                    )
                    raise
                backoff = _RETRY_BACKOFF_BASE_SECONDS**attempt
                logger.warning(
                    "vLLM request aborted attempt=%s/%s endpoint=%s "
                    "request_type=%s sample_id=%s status_code=%s request_id=%s "
                    "response_body=%s elapsed=%.3fs error=%s; failing over in %ss",
                    attempt,
                    total_attempts,
                    state.endpoint.base_url,
                    "generate" if generate else "prefill",
                    getattr(request, "sample_id", None),
                    *_http_error_detail(exc),
                    time.monotonic() - request_started,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
            finally:
                state.inflight = max(state.inflight - 1, 0)
        raise RuntimeError(
            "unreachable: vLLM request retry loop exhausted without returning"
        )

    def _should_log_success(self, count: int) -> bool:
        interval = self.success_log_interval
        return interval > 0 and (count <= 3 or count % interval == 0)

    async def close(self) -> None:
        states, self._states = self._states, []
        for state in states:
            close = getattr(state.client, "close", None)
            if not callable(close):
                continue
            result = close()
            if inspect.isawaitable(result):
                await result


def _http_error_detail(exc: Exception) -> tuple[Any, Any, str]:
    """Extract bounded HTTP diagnostics without depending on one SDK version."""

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    headers = getattr(response, "headers", {}) or {}
    request_id = headers.get("x-request-id") if hasattr(headers, "get") else None
    body = getattr(response, "text", "") if response is not None else ""
    if not isinstance(body, str):
        body = repr(body)
    return status_code, request_id, body[:2048]


def _wait_for_lock(lock_path: Path, timeout: float = 30.0) -> None:
    if not lock_path.exists():
        return
    try:
        fcntl: Any = importlib.import_module("fcntl")
    except ImportError:
        # vLLM's file connector is Linux-only. Keep the old existence-based
        # fallback for dependency-light tests on other platforms.
        deadline = time.monotonic() + timeout
        while lock_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for vLLM hidden-state lock: {lock_path}"
                )
            time.sleep(0.01)
        return

    deadline = time.monotonic() + timeout
    with lock_path.open("rb") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for vLLM hidden-state lock: {lock_path}"
                    ) from exc
                time.sleep(0.01)


__all__ = [
    "RawVllmFeature",
    "VllmEndpoint",
    "VllmFeatureClientPool",
    "VllmResponse",
    "choose_endpoint",
    "delete_temporary_result",
    "load_hidden_state_result",
    "request_prefill",
    "request_generate",
]
