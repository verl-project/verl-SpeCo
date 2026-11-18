# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration.serving_drafter_commit import (
    collective_rpc_all_dp,
    commit_drafter_revision,
    publish_serving_revision,
    serving_versions,
)
from verl_speco.integration.vllm_runtime import _SpecoVLLMHttpServerMixin


@pytest.fixture(autouse=True)
def ascend_backend(monkeypatch):
    monkeypatch.setenv(
        "VERL_SPECO_DRAFTER_CONFIG", '{"training":{"backend":"vllm_ascend"}}'
    )
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)


class Server(_SpecoVLLMHttpServerMixin):
    def __init__(self):
        self.global_steps = 7
        self.calls = []

    async def abort_all_requests(self):
        self.calls.append("pause")
        return {"aborted_count": 0}

    async def resume_generation(self):
        self.calls.append("resume")


def test_commit_requires_closed_idle_matching_versions():
    async def run():
        server = Server()
        with pytest.raises(RuntimeError, match="closed and idle"):
            commit_drafter_revision(server, 0, 1, 7)
        await server.speco_close_rollout_admission()
        with pytest.raises(RuntimeError, match="Target version"):
            commit_drafter_revision(server, 0, 1, 8)
        assert commit_drafter_revision(server, 0, 1, 7)["drafter_revision"] == 1
        with pytest.raises(RuntimeError, match="Stale"):
            commit_drafter_revision(server, 0, 1, 7)
        assert serving_versions(server)["target_version"] == 7

    asyncio.run(run())


def test_request_waits_for_commit_and_sees_new_draft_but_same_target():
    async def run():
        server = Server()
        waiting = []

        async def request():
            async with server._speco_rollout_admission().request():
                return serving_versions(server)

        async def paused():
            waiting.append(asyncio.create_task(request()))
            await asyncio.sleep(0)
            assert not waiting[0].done()

        async def publish():
            assert not serving_versions(server)["admission_open"]
            return {"workers": 16}

        result = await publish_serving_revision(
            [server], target_version=7, revision=1, publish=publish, while_paused=paused
        )
        state = await waiting[0]
        assert state["target_version"] == 7 and state["drafter_revision"] == 1
        assert result["result"]["workers"] == 16
        assert server.calls == ["pause", "resume"]

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["pause", "publish", "commit"])
def test_failure_keeps_generation_closed(monkeypatch, phase):
    async def run():
        server = Server()
        if phase == "pause":

            async def fail_pause():
                return {"error": "engine pause failed"}

            monkeypatch.setattr(server, "abort_all_requests", fail_pause)
        if phase == "commit":

            async def fail_commit(*args):
                raise RuntimeError("commit failed")

            monkeypatch.setattr(
                server, "speco_commit_serving_drafter_revision", fail_commit
            )

        async def publish():
            if phase == "publish":
                raise RuntimeError("publish failed")

        with pytest.raises(RuntimeError):
            await publish_serving_revision(
                [server], target_version=7, revision=1, publish=publish
            )
        assert not serving_versions(server)["admission_open"]
        assert "resume" not in server.calls

    asyncio.run(run())


def test_all_server_replies_settle_before_error_returns():
    async def run():
        first, second = Server(), Server()
        settled = []

        async def fail_commit(*args):
            raise RuntimeError("first failed")

        async def slow_commit(*args):
            await asyncio.sleep(0.02)
            settled.append(True)
            return commit_drafter_revision(second, *args)

        first.speco_commit_serving_drafter_revision = fail_commit
        second.speco_commit_serving_drafter_revision = slow_commit

        async def publish():
            return None

        with pytest.raises(RuntimeError):
            await publish_serving_revision(
                [first, second], target_version=7, revision=1, publish=publish
            )
        assert settled == [True]
        assert not serving_versions(first)["admission_open"]
        assert not serving_versions(second)["admission_open"]

    asyncio.run(run())


def test_async_dp_rpc_keeps_every_dp_reply_instead_of_only_first():
    async def run():
        calls = []

        async def utility(method, *args, engine):
            calls.append((method, engine))
            return [{"dp_rank": engine, "tp_rank": t} for t in range(2)]

        engine = SimpleNamespace(
            engine_core=SimpleNamespace(
                core_engines=[0, 1], _call_utility_async=utility
            )
        )
        result = await collective_rpc_all_dp(engine, "probe", dp_size=2)
        assert len(result) == 4 and {row["dp_rank"] for row in result} == {0, 1}
        assert calls == [("collective_rpc", 0), ("collective_rpc", 1)]
        with pytest.raises(RuntimeError, match="every DP engine"):
            await collective_rpc_all_dp(
                SimpleNamespace(engine_core=object()), "probe", dp_size=2
            )

    asyncio.run(run())


def test_cancelled_waiter_releases_waiting_counter():
    async def run():
        server = Server()
        await server.speco_close_rollout_admission()

        async def request():
            async with server._speco_rollout_admission().request():
                raise AssertionError("Closed gate admitted a request")

        task = asyncio.create_task(request())
        await asyncio.sleep(0)
        assert serving_versions(server)["waiting_requests"] == 1
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert serving_versions(server)["waiting_requests"] == 0
        assert serving_versions(server)["active_requests"] == 0

    asyncio.run(run())


def test_native_server_mask_uses_model_vocab_and_delegates_tokenization():
    from verl_speco.integration.deepseek_v4_server_probe import ModelVocabularyTokenizer

    class Tokenizer:
        def __len__(self):
            return 129283

        def encode(self, text):
            return [1, 2]

    tokenizer = ModelVocabularyTokenizer(Tokenizer(), 129280)
    assert len(tokenizer) == 129280
    assert tokenizer.encode("test") == [1, 2]
    with pytest.raises(ValueError, match="cover"):
        ModelVocabularyTokenizer(Tokenizer(), 129284)
