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
"""Lifecycle/transport contracts without a live Ray/vLLM cluster."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.integration import specopd_vllm, vllm_runtime
from verl_speco.integration.ci_result import CIResult
from verl_speco.integration.specopd_driver import SpecOPDDriver
from verl_speco.integration.specopd_window import make_window
from verl_speco.specopd_config import SpecOPDConfig
from verl_speco.integration.specopd_protocol import PLAN_KEY, decode_bundle


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_vllm_scheduler_to_final_output_channel(monkeypatch, async_scheduling, collect):
    monkeypatch.setenv(vllm_runtime.SPECO_DRAFTER_CONFIG_ENV, "{}")
    plan = make_window(SpecOPDConfig(proposal_width=3), "external", 0, 0, 3, 20)
    if not collect:
        plan["max_branches"] = 0
    request = SimpleNamespace(
        all_token_ids=[1, 2, 3],
        spec_token_ids=[11, 12, 13],
        num_output_placeholders=16 if async_scheduling else 0,
        is_finished=lambda: False,
        sampling_params=SimpleNamespace(extra_args={"specopd_plan": plan}),
    )

    class Scheduler:
        def __init__(self):
            self.requests = {"internal": request}
            self.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
            self.use_v2_model_runner = async_scheduling

        def schedule(self):
            request.spec_token_ids = []
            return SimpleNamespace(
                num_scheduled_tokens={"internal": 4},
                scheduled_spec_decode_tokens={"internal": [-1] * 3 if async_scheduling else [11, 12, 13]}
            )

        def update_from_output(self, schedule, result):
            self.requests.clear()
            return {
                0: SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            request_id="internal", finished=True,
                            trace_headers={vllm_runtime.SPECO_VLLM_REQUEST_STATS_TRACE_HEADER:
                                '{"verify_rounds":2,"draft_tokens":6,"accepted_tokens":3}'},
                        )
                    ]
                )
            }

    class RequestState:
        def make_request_output(self):
            return SimpleNamespace(request_id="external")

    class OutputProcessor:
        def __init__(self):
            self.request_states = {"internal": RequestState()}

        def process_outputs(self, outputs):
            return [
                self.request_states[o.request_id].make_request_output() for o in outputs
            ]

    for name, values in (
        ("vllm.v1.core.sched.scheduler", {"Scheduler": Scheduler}),
        (
            "vllm.v1.engine.output_processor",
            {"OutputProcessor": OutputProcessor, "RequestState": RequestState},
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    assert specopd_vllm.install_specopd_vllm_patch()
    scheduler = Scheduler()
    schedule = scheduler.schedule()
    result = SimpleNamespace(req_id_to_index={"internal": 0}, sampled_token_ids=[[20]])
    if async_scheduling and collect:
        from verl_speco.integration.specopd_async import VERIFIED_KEY
        setattr(result, VERIFIED_KEY, {"internal": dict(prefix_len=3, proposal_ids=[11, 12, 13], scheduled_drafts=3)})
    outputs = scheduler.update_from_output(schedule, result)
    [final] = OutputProcessor().process_outputs(outputs[0].outputs)
    fields = vllm_runtime._vllm_request_accept_stats_to_scalar_extra_fields(final)
    assert fields["_speco_vllm_request_verify_rounds"] == 2
    assert fields["_speco_vllm_request_accepted_tokens"] == 3
    summaries, _ = vllm_runtime._pop_vllm_request_stats_for_rollout_output("external")
    wire = summaries["external"]["_specopd_proposals"]
    assert fields["_specopd_proposals"] == wire
    bundle = decode_bundle(wire)
    assert bundle["collection_enabled"] is collect
    assert len(bundle["records"]) == int(collect)
    if collect:
        assert bundle["records"][0]["proposal_ids"] == [11, 12, 13]
    assert not vllm_runtime._pop_vllm_request_stats_for_rollout_output("external")[0]
    assert not getattr(scheduler, "_specopd_schedules", {})


@pytest.mark.parametrize("enabled,async_scheduling,v2", [
    (True, True, False), (True, True, True), (True, False, False), (False, True, False),
])
def test_startup_checks_effective_runner_before_scheduling(monkeypatch, enabled, async_scheduling, v2):
    monkeypatch.setenv(vllm_runtime.SPECO_DRAFTER_CONFIG_ENV,
                       '{"training":{"specopd":{"enable":' + str(enabled).lower() + '}}}')
    calls = []

    class Scheduler:
        def __init__(self):
            self.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
            self.use_v2_model_runner = v2
            self.requests = {}

        def schedule(self):
            calls.append("schedule")

        def update_from_output(self, *args):
            return {}

    module = ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(vllm_runtime, "patch_vllm_request_acceptance_stats", lambda: True)
    # An importable/installed MRV2 hook does not select the engine's runner.
    monkeypatch.setattr(specopd_vllm, "install_specopd_draft_token_transport_patch", lambda: True)
    assert specopd_vllm.install_specopd_draft_token_transport_patch()
    assert specopd_vllm.install_specopd_vllm_patch()
    assert specopd_vllm.install_specopd_vllm_patch()  # idempotent
    if enabled and async_scheduling and not v2:
        with pytest.raises(RuntimeError, match="active MRV2"):
            Scheduler()
        assert not calls
    else:
        Scheduler().schedule()
        assert calls == ["schedule"]


def configuration():
    training = {"dspark_block_size": 3, "specopd": {"enable": True, "proposal_width": 3}}
    drafter = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "DSPARK",
        "training": training,
    }
    actor = {"model": {"use_remove_padding": False}, "rollout": {"name": "vllm", "drafter": drafter}}
    return {"trainer": {"use_v1": True}, "actor_rollout_ref": actor}


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


def test_five_step_twenty_update_schedule_with_one_publication(monkeypatch):
    from verl_speco.integration import specopd_training_transport

    monkeypatch.setattr(specopd_training_transport, "dispatch_prepared_group",
        lambda trainer, samples, step, revision, indices, metrics, spec:
        trainer._require_speco_worker_group().train_specopd_group(step, revision, indices))
    events = []
    cfg = configuration()
    rollout = cfg["actor_rollout_ref"]["rollout"]
    rollout["drafter"]["training"].update(step=20, collect_interval_steps=5, training_interval_steps=5)
    ci = CIResult(training=True)
    trainer = SimpleNamespace(config=cfg, global_steps=1, actor_revision=0, _speco_ci_result=ci)
    worker = SimpleNamespace(train_specopd_group=lambda step, revision, indices: events.append(("train", step, 20)) or [dict(trained=True, revision=revision+1, optimizer_updates=20)])
    trainer._ray_get_if_needed = lambda x: x
    trainer._require_speco_worker_group = lambda: worker
    trainer._speco_set_drafter_global_step = lambda: None
    trainer._speco_sync_target_lm_head_weight = lambda: {}
    trainer._speco_publish_drafter_weights = lambda *a, **kw: events.append(("publish", trainer.global_steps)) or {"drafter/published": 1}
    ack = SimpleNamespace(speco_set_serving_drafter_revision=SimpleNamespace(remote=lambda rev: dict(actor_revision=trainer.actor_revision, drafter_revision=rev)))
    trainer.llm_server_manager = SimpleNamespace(server_handles=[ack])
    driver = SpecOPDDriver(trainer)
    driver.bootstrapped = True
    for step in range(1, 11):
        trainer.global_steps = step
        due = driver.begin_step(step)
        assert due == (step in (5, 10))
        if due:
            driver.actor_revision = trainer.actor_revision
            driver.pending = ({"_speco_position_count": 2, "_specopd_proposal_anchors": 1, "_speco_sample_count": 1},)
            driver.state = "scored"
        driver.before_actor_update()
        trainer.actor_revision = step
        driver.finish_step()
        assert driver.metrics["specopd/ack_actor_revision"] == step
        assert driver.metrics["specopd/ack_drafter_revision"] == step // 5
        assert driver.metrics["specopd/optimizer_updates"] == (20 if due else 0)
        assert driver.metrics["specopd/published"] == int(due)
    assert events == [("train", 5, 20), ("publish", 5), ("train", 10, 20), ("publish", 10)]
    assert ci.data["optimizer_steps"] == 40
    assert ci.data["publications"] == 2

    assert driver.pending is None and driver.state == "ready"
    with pytest.raises(RuntimeError, match="already been consumed"):
        driver.before_actor_update()
