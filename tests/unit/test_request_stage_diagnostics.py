# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Concurrent/background request diagnostics do not change operation semantics."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl_speco.integration import request_stage_diagnostics as diag
from verl_speco.trainer.v1.specopd_adapter import prepare_draft_rollout


def setup(monkeypatch, failing=False):
    rows = []
    monkeypatch.setenv(diag.FLAG, "1")
    monkeypatch.setattr(
        diag, "emit", lambda kind, **kw: rows.append(dict(kind=kind, **kw))
    )
    tq = ModuleType("transfer_queue")

    async def put(*args, **kwargs):
        await asyncio.sleep(0)
        return "put"

    tq.async_kv_put = tq.async_kv_batch_put = put
    monkeypatch.setitem(sys.modules, "transfer_queue", tq)

    class Client:
        async def generate(self):
            await asyncio.sleep(0)
            return "tokens"

    class Worker:
        async def _compute_score(self, output):
            await asyncio.sleep(0)
            if failing:
                raise ValueError("reward error")

        async def _compute_teacher_logprobs(self, output):
            await asyncio.sleep(0)

        async def _agent_loop_postprocess(self, output):
            await self._compute_score(output)
            await self._compute_teacher_logprobs(output)
            assert await tq.async_kv_batch_put() == "put"

        async def _run_agent_loop(self, params, trajectory, **kwargs):
            value = await Client().generate()
            await self._agent_loop_postprocess(value)
            return value

        async def _run_prompt(self, uid, trajectory):
            await tq.async_kv_put()
            result = await asyncio.gather(
                *(
                    self._run_agent_loop({}, trajectory, uid=uid, session_id=i)
                    for i in range(5)
                )
            )
            await tq.async_kv_put()
            return result

    diag.install_request_stage_diagnostics(Worker, Client)
    diag.install_request_stage_diagnostics(Worker, Client)
    return Worker, rows


def test_concurrent_background_prompts_are_isolated_and_bounded(monkeypatch):
    worker, rows = setup(monkeypatch)

    async def run():
        # Native V1 submits fire-and-forget tasks; the prompt owns the collector.
        tasks = [
            asyncio.create_task(worker()._run_prompt(str(step), {"step": step}))
            for step in [5, 6]
        ]
        assert rows == []
        results = await asyncio.gather(*tasks)
        assert results == [["tokens"] * 5] * 2

    asyncio.run(run())
    assert len(rows) == 2
    for row in rows:
        assert row["kind"] == "prompt_summary"
        assert row["count"] == 5
        assert len(row["slowest"]) == 3
        assert len({r["source_step"] for r in row["slowest"]}) == 1
        assert not row["failed"]
        assert row["status_notify_s"] >= 0
        for request in row["slowest"]:
            assert all(
                name in request
                for name in [
                    "llm_rpc_s",
                    "reward_s",
                    "teacher_s",
                    "tq_put_s",
                    "postprocess_other_s",
                ]
            )
    assert {r["slowest"][0]["source_step"] for r in rows} == {"5", "6"}


def test_disabled_flag_keeps_original_method(monkeypatch):
    monkeypatch.delenv(diag.FLAG, raising=False)

    class Worker:
        async def _run_prompt(self):
            pass

    original = Worker._run_prompt
    diag.install_request_stage_diagnostics(Worker)
    assert Worker._run_prompt is original


def test_failures_propagate_and_context_is_reset(monkeypatch):
    worker, rows = setup(monkeypatch, failing=True)
    with pytest.raises(ValueError, match="reward error"):
        asyncio.run(worker()._run_agent_loop({}, {"step": 7}, uid="bad"))
    assert rows[0]["failed"]
    assert diag._request.get() is None
    assert diag._prompt.get() is None


def test_replay_wait_preserves_result_and_arguments(monkeypatch):
    monkeypatch.setenv(diag.FLAG, "1")
    rows = []
    monkeypatch.setattr(diag, "emit", lambda kind, **kw: rows.append((kind, kw)))
    replay = SimpleNamespace(sample=lambda *, global_steps: global_steps + 1)
    diag.install_replay_sample_timing(replay)
    diag.install_replay_sample_timing(replay)
    assert replay.sample(global_steps=23) == 24
    assert rows[0][0] == "replay_sample"
    assert rows[0][1]["source_step"] == "23"


@pytest.mark.parametrize(
    "due,blocked,previous,expected",
    [
        (False, False, False, True),
        (True, False, False, False),
        (True, True, False, True),
        (True, False, True, True),
    ],
)
def test_v1_draft_collection_flag_uses_tensor_dict_channel(
    due, blocked, previous, expected
):
    from tensordict import TensorDict, NonTensorData

    batch = TensorDict(
        {"_specopd_skip_collection": NonTensorData(previous)}, batch_size=[]
    )
    trainer = SimpleNamespace(
        global_steps=5,
        config={
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "training": {
                            "drafter_opd": {"enable": True, "method": "draft_opd"}
                        }
                    }
                }
            }
        },
        _speco_opd_trace_collection_due=lambda **kw: due,
        _speco_drafter_schedule_config=lambda: "config",
        _speco_get_drafter_scheduler=lambda: SimpleNamespace(
            training_quota_blocks_collection=lambda step, cfg: blocked
        ),
    )
    assert prepare_draft_rollout(trainer, batch) is batch
    assert batch.get("_specopd_skip_collection").data is expected


@pytest.mark.parametrize("positional", [False, True])
def test_partial_client_carries_skip_and_revision_without_base_client(positional):
    from verl_speco.integration import agent_loop_runtime as runtime
    from verl_speco.integration.partial_rollout_stats import (
        patch_partial_rollout_client,
    )

    class Client:
        async def generate(self, prompt_ids, sampling_params):
            assert sampling_params["_verl_skip_drafter_collection"] is True
            assert sampling_params["_specopd_drafter_revision"] == 13
            return SimpleNamespace(extra_fields={})

    patch_partial_rollout_client(Client)
    skip = runtime._CURRENT_SKIP_DRAFTER_COLLECTION.set(True)
    revision = runtime._CURRENT_SPECOPD_REVISION.set(13)
    try:
        if positional:
            output = asyncio.run(Client().generate([1], {}))
        else:
            output = asyncio.run(Client().generate(prompt_ids=[1], sampling_params={}))
        assert isinstance(output.extra_fields, dict)
    finally:
        runtime._CURRENT_SKIP_DRAFTER_COLLECTION.reset(skip)
        runtime._CURRENT_SPECOPD_REVISION.reset(revision)


def test_capture_settings_cache_tracks_config_changes_without_mutable_results(monkeypatch):
    import json
    from verl_speco.integration import vllm_runtime as runtime
    original = json.loads
    calls = []
    def counted(raw):
        calls.append(raw)
        return original(raw)
    runtime._verify_trace_runtime_settings.cache_clear()
    monkeypatch.setattr(runtime.json, "loads", counted)
    off = '{"training": {"drafter_opd": {"capture_verify_trace": false}}}'
    on = '{"speculative_algorithm": "DSPARK", "training": {"drafter_opd": {"capture_verify_trace": true}}}'
    for _ in range(10):
        assert runtime._verify_trace_runtime_settings(off) == (False, "eagle3")
    assert runtime._verify_trace_runtime_settings(on) == (True, "dspark")
    assert len(calls) == 2
    runtime._verify_trace_runtime_settings.cache_clear()


def test_enabled_worker_survives_by_value_serialization(monkeypatch):
    """Ray exports dynamic worker classes, including every wrapper closure."""
    import cloudpickle

    worker, _ = setup(monkeypatch)
    # Force the stricter by-value case rather than relying on module lookup.
    cloudpickle.register_pickle_by_value(diag)
    try:
        restored = cloudpickle.loads(cloudpickle.dumps(worker))
        tq = sys.modules["transfer_queue"]
        cloudpickle.loads(cloudpickle.dumps(tq.async_kv_put))
        cloudpickle.loads(cloudpickle.dumps(tq.async_kv_batch_put))
    finally:
        cloudpickle.unregister_pickle_by_value(diag)
    assert asyncio.run(restored()._run_prompt("uid", {"step": 5})) == ["tokens"] * 5
    assert diag._request.get() is None
    assert diag._prompt.get() is None
    assert diag._scope.get() is None


def test_enabled_ray_actor_class_is_serializable(monkeypatch):
    ray = pytest.importorskip("ray")
    worker, _ = setup(monkeypatch)
    actor = ray.remote(worker)
    exported = actor.__ray_metadata__.modified_class
    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(exported))
    assert asyncio.run(restored()._run_prompt("uid", {"step": 5})) == ["tokens"] * 5
