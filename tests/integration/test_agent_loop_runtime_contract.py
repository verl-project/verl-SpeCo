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

import pytest

from verl_speco.integration import agent_loop_runtime
from verl_speco.integration.partial_rollout_stats import (
    REVISION_MAX,
    REVISION_MIN,
    merge_serving_segments,
    patch_partial_rollout_client,
    record_serving_segment,
)
from verl_speco.integration.rollout_admission import RolloutAdmission


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
    monkeypatch.setattr(
        agent_loop_runtime, "_load_agent_loop_module", lambda: release_v080_module
    )

    assert agent_loop_runtime.install_agent_loop_runtime_patch() is True
    assert LLMServerClient._speco_patched_generate is True
    assert AgentLoopWorker._speco_patched_generate_sequences is True
    assert AgentLoopManager._speco_patched_generate_sequences is True

    global_steps_token = agent_loop_runtime._CURRENT_GLOBAL_STEPS.set(17)
    try:
        sampling_params = asyncio.run(
            LLMServerClient().generate(sampling_params={"temperature": 1.0})
        )
    finally:
        agent_loop_runtime._CURRENT_GLOBAL_STEPS.reset(global_steps_token)
    assert sampling_params["_verl_global_steps"] == 17

    global_steps_token = agent_loop_runtime._CURRENT_GLOBAL_STEPS.set(18)
    skip_collection_token = agent_loop_runtime._CURRENT_SKIP_DRAFTER_COLLECTION.set(
        True
    )
    try:
        sampling_params = asyncio.run(
            LLMServerClient().generate(sampling_params={"temperature": 1.0})
        )
    finally:
        agent_loop_runtime._CURRENT_SKIP_DRAFTER_COLLECTION.reset(skip_collection_token)
        agent_loop_runtime._CURRENT_GLOBAL_STEPS.reset(global_steps_token)
    assert sampling_params["_verl_global_steps"] == 18
    assert sampling_params["_verl_skip_drafter_collection"] is True
    assert "_verl_skip_rollout_idle_event" not in sampling_params


def test_agent_loop_drain_waits_for_an_inflight_async_generation(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class AgentLoopWorker:
        async def generate_sequences(self, batch):
            started.set()
            await release.wait()
            return batch

        async def _run_agent_loop(self, sampling_params, trajectory):
            return sampling_params, trajectory

        def _postprocess(self, inputs, input_non_tensor_batch=None, validate=False):
            return inputs

    class AgentLoopManager:
        def __init__(self):
            self.config = None

        async def generate_sequences(self, prompts):
            return prompts

    class LLMServerClient:
        async def generate(self, *, sampling_params):
            return sampling_params

    monkeypatch.setattr(agent_loop_runtime, "_PATCHED", False)
    monkeypatch.setattr(
        agent_loop_runtime,
        "_load_agent_loop_module",
        lambda: SimpleNamespace(
            AgentLoopWorker=AgentLoopWorker,
            AgentLoopManager=AgentLoopManager,
            LLMServerClient=LLMServerClient,
        ),
    )
    assert agent_loop_runtime.install_agent_loop_runtime_patch() is True

    async def exercise() -> None:
        worker = AgentLoopWorker()
        generation = asyncio.create_task(worker.generate_sequences({}))
        await started.wait()
        drain = asyncio.create_task(worker.speco_drain())
        await asyncio.sleep(0)
        assert not drain.done()
        release.set()
        await generation
        await drain

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_tq_background_tasks_settle_after_submission_rpc_returned(cancel):
    async def exercise():
        release = asyncio.Event()
        worker = SimpleNamespace(background_tasks=set())
        task = asyncio.create_task(release.wait())
        worker.background_tasks.add(task)
        task.add_done_callback(worker.background_tasks.discard)
        await asyncio.sleep(0)
        if cancel:
            await asyncio.wait_for(
                agent_loop_runtime._speco_worker_cancel_pending(worker), 1
            )
            assert task.cancelled()
        else:
            drain = asyncio.create_task(agent_loop_runtime._speco_worker_drain(worker))
            await asyncio.sleep(0)
            assert not drain.done()
            release.set()
            await asyncio.wait_for(drain, 1)
        assert not worker.background_tasks

    asyncio.run(exercise())


def test_native_tq_worker_bridge_keeps_manager_and_exposes_serialized_rpcs(monkeypatch):
    import sys

    class Worker:
        pass

    class Manager:
        pass

    Manager.__module__ = "verl.trainer.ppo.v1.agent_loop_tq"
    native = SimpleNamespace(
        AgentLoopManagerTQ=Manager,
        AgentLoopWorkerTQ=SimpleNamespace(__ray_actor_class__=Worker),
    )
    monkeypatch.setitem(
        sys.modules, "verl.trainer.ppo.v1", SimpleNamespace(agent_loop_tq=native)
    )
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(remote=lambda cls: SimpleNamespace(__ray_actor_class__=cls)),
    )
    assert agent_loop_runtime.install_v1_agent_loop_worker_bridge(Manager)
    bridged = native.AgentLoopWorkerTQ.__ray_actor_class__
    assert issubclass(bridged, Worker)
    assert bridged.speco_drain is agent_loop_runtime._speco_worker_drain
    assert (
        bridged.speco_cancel_pending is agent_loop_runtime._speco_worker_cancel_pending
    )
    assert native.AgentLoopManagerTQ is Manager
    assert agent_loop_runtime.install_v1_agent_loop_worker_bridge(Manager)
    assert native.AgentLoopWorkerTQ.__ray_actor_class__ is bridged
    assert not agent_loop_runtime.install_v1_agent_loop_worker_bridge(
        type("Custom", (), {})
    )


def _serving_segment(rounds, accepted, revision):
    return {
        "_speco_vllm_request_id": "request",
        "_speco_vllm_request_verify_rounds": rounds,
        "_speco_vllm_request_accepted_tokens": accepted,
        "_speco_vllm_request_draft_tokens": rounds * 7,
        "_speco_vllm_request_elapsed_sec": 0.25,
        REVISION_MIN: revision,
        REVISION_MAX: revision,
    }


def test_partial_acceptance_is_weighted_by_verify_rounds_not_chunk_means():
    import json

    fields = {"min_global_steps": 1, "max_global_steps": 2}
    merge_serving_segments(
        fields, [_serving_segment(1, 6, 0), _serving_segment(3, 3, 1)]
    )
    assert fields["_verl_request_mean_accept_len"] == 3.25
    assert fields[REVISION_MIN] == 0
    assert fields[REVISION_MAX] == 1
    assert fields["_speco_vllm_request_elapsed_sec"] == 0.5
    assert fields["max_global_steps"] == 2
    projections = json.loads(fields["_speco_drafter_revision_stats"])
    assert [
        (row["revision"], row["verify_rounds"], row["accepted_tokens"])
        for row in projections
    ] == [(0, 1, 6), (1, 3, 3)]


def test_missing_segment_provenance_or_counters_does_not_fabricate_evidence():
    fields = {}
    merge_serving_segments(fields, [_serving_segment(2, 4, 1), {}])
    assert fields[REVISION_MIN] is None
    assert fields[REVISION_MAX] is None
    assert fields["_verl_request_mean_accept_len"] is None
    assert fields["_speco_drafter_revision_stats"] is None


@pytest.mark.parametrize("fail", [False, True])
def test_partial_client_context_is_per_request_and_restored_after_failure(fail):
    class Client:
        async def generate(self, revision):
            record_serving_segment(
                SimpleNamespace(
                    token_ids=[1], extra_fields=_serving_segment(2, revision, revision)
                )
            )
            await asyncio.sleep(0)
            if fail and revision == 2:
                raise RuntimeError("aborted client")
            return SimpleNamespace(token_ids=[1], extra_fields={})

    patch_partial_rollout_client(Client)
    patch_partial_rollout_client(Client)

    async def scenario():
        outputs = await asyncio.gather(
            Client().generate(1), Client().generate(2), return_exceptions=True
        )
        assert outputs[0].extra_fields[REVISION_MAX] == 1
        if fail:
            assert isinstance(outputs[1], RuntimeError)
        else:
            assert outputs[1].extra_fields[REVISION_MAX] == 2
        record_serving_segment(
            SimpleNamespace(token_ids=[99], extra_fields=_serving_segment(1, 1, 99))
        )
        assert (await Client().generate(1)).extra_fields[REVISION_MIN] == 1

    asyncio.run(scenario())


def test_close_blocks_new_requests_but_waits_only_for_admitted_rpc():
    async def scenario():
        gate = RolloutAdmission()
        started, finish, retry_started = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def admitted():
            async with gate.request():
                started.set()
                await finish.wait()

        async def retry():
            async with gate.request():
                retry_started.set()

        first = asyncio.create_task(admitted())
        await started.wait()
        gate.close()
        queued = asyncio.create_task(retry())
        idle = asyncio.create_task(gate.wait_idle())
        await asyncio.sleep(0)
        assert not idle.done()
        assert not retry_started.is_set()
        finish.set()
        await first
        await asyncio.wait_for(idle, 1)
        assert not queued.done()  # A paused agent retry must not hold the barrier.
        gate.resume()
        await asyncio.wait_for(queued, 1)
        assert retry_started.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_request_releases_idle_barrier(cancel):
    async def scenario():
        gate = RolloutAdmission()
        started = asyncio.Event()

        async def request():
            async with gate.request():
                started.set()
                if cancel:
                    await asyncio.Event().wait()
                raise RuntimeError("serving failed")

        task = asyncio.create_task(request())
        await started.wait()
        gate.close()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await task
        await asyncio.wait_for(gate.wait_idle(), 1)
        gate.resume()
        async with gate.request():
            pass

    asyncio.run(scenario())
