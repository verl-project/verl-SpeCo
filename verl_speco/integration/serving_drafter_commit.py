# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Serving version boundary shared by independent probes and online publishers."""

import asyncio
import inspect
from contextlib import asynccontextmanager

from verl_speco.integration.rollout_admission import RolloutAdmission


class AscendRolloutAdmission(RolloutAdmission):
    """Extra commit diagnostics for backend=vllm_ascend only."""

    def __init__(self):
        super().__init__()
        self._waiting = 0

    @property
    def is_open(self):
        return self._open.is_set()

    @property
    def active_requests(self):
        return self._active

    @property
    def waiting_requests(self):
        return self._waiting

    @asynccontextmanager
    async def request(self):
        if not self._open.is_set():
            self._waiting += 1
            try:
                while not self._open.is_set():
                    await self._open.wait()
            finally:
                self._waiting -= 1
        async with super().request():
            yield


def serving_versions(server):
    gate = server._speco_rollout_admission()
    return {
        "target_version": getattr(server, "global_steps", None),
        "drafter_revision": int(getattr(server, "_speco_serving_drafter_revision", 0)),
        "admission_open": gate.is_open,
        "active_requests": gate.active_requests,
        "waiting_requests": gate.waiting_requests,
    }


def commit_drafter_revision(server, expected_revision, revision, target_version):
    state = serving_versions(server)
    if state["admission_open"] or state["active_requests"]:
        raise RuntimeError(
            "Serving must be closed and idle before committing a draft revision"
        )
    if state["target_version"] != target_version:
        raise RuntimeError("Target version changed at the draft commit boundary")
    if (
        state["drafter_revision"] != expected_revision
        or revision != expected_revision + 1
    ):
        raise RuntimeError("Stale or non-monotonic serving draft revision")
    server._speco_serving_drafter_revision = revision
    return serving_versions(server)


async def _call(server, method, *args):
    function = getattr(server, method)
    remote = getattr(function, "remote", None)
    result = remote(*args) if callable(remote) else function(*args)
    return await result if inspect.isawaitable(result) else result


async def _settled(servers, method, *args):
    results = await asyncio.gather(
        *[_call(server, method, *args) for server in servers], return_exceptions=True
    )
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(
            f"Serving phase {method} failed after all replies settled: {failures}"
        )
    return results


async def publish_serving_revision(
    servers, *, target_version, revision, publish, while_paused=None
):
    """Never reopen admission on failure; the caller owns weight rollback/recovery."""
    if not servers:
        raise ValueError("No serving servers")
    states = await _settled(servers, "speco_get_serving_versions")
    if any(state["target_version"] != target_version for state in states):
        raise RuntimeError("Serving target versions differ")
    old = revision - 1
    if any(state["drafter_revision"] != old for state in states):
        raise RuntimeError("Serving draft revisions differ or update is stale")
    await _settled(servers, "speco_close_rollout_admission")
    paused = await _settled(servers, "abort_all_requests")
    if any(not isinstance(reply, dict) or reply.get("error") for reply in paused):
        raise RuntimeError(f"Serving pause failed: {paused}")
    await _settled(servers, "speco_wait_rollout_idle")
    if while_paused is not None:
        await while_paused()
    result = await publish()
    committed = await _settled(
        servers, "speco_commit_serving_drafter_revision", old, revision, target_version
    )
    if any(
        state["target_version"] != target_version
        or state["drafter_revision"] != revision
        for state in committed
    ):
        raise RuntimeError("Serving revision acknowledgement differs")
    await _settled(servers, "resume_generation")
    await _settled(servers, "speco_resume_rollout_admission")
    return {"result": result, "before": states, "committed": committed}


async def collective_rpc_all_dp(engine, method, *, dp_size, timeout=None, kwargs=None):
    """vLLM 0.26 DP utility broadcast otherwise discards every result except DP0."""
    if dp_size == 1:
        return await engine.collective_rpc(method, timeout=timeout, kwargs=kwargs)
    core = engine.engine_core
    engines = getattr(core, "core_engines", None)
    call = getattr(core, "_call_utility_async", None)
    if not callable(call) or engines is None or len(engines) != dp_size:
        raise RuntimeError(
            "This async vLLM client cannot return every DP engine result"
        )
    replies = await asyncio.gather(
        *[
            call("collective_rpc", method, timeout, (), kwargs, engine=identity)
            for identity in engines
        ],
        return_exceptions=True,
    )
    if any(isinstance(reply, BaseException) for reply in replies):
        raise RuntimeError(f"DP collective failed after all replies settled: {replies}")
    if any(not isinstance(reply, (list, tuple)) for reply in replies):
        raise RuntimeError("DP collective returned an unsupported reply layout")
    return [value for reply in replies for value in reply]
