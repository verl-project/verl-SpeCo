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

from transformers import Qwen3Config, Qwen3ForCausalLM

from verl_speco.integration import specopd_driver, specopd_vllm, vllm_runtime
from verl_speco.integration.specopd_driver import SpecOPDDriver
from verl_speco.integration.specopd_window import make_window
from verl_speco.joint_scoring import score_joint_batch
from verl_speco.specopd_config import SpecOPDConfig
from verl_speco.integration.specopd_protocol import (
    PLAN_KEY,
    decode_bundle,
    encode_bundle,
    token_digest,
)


def test_vllm_scheduler_to_final_output_channel(monkeypatch):
    plan = make_window(SpecOPDConfig(proposal_width=3), "external", 0, 0, 3, 20)
    request = SimpleNamespace(
        all_token_ids=[1, 2, 3],
        spec_token_ids=[11, 12, 13],
        num_output_placeholders=0,
        sampling_params=SimpleNamespace(extra_args={"specopd_plan": plan}),
    )

    class Scheduler:
        def __init__(self):
            self.requests = {"internal": request}

        def schedule(self):
            request.spec_token_ids = []
            return SimpleNamespace(
                scheduled_spec_decode_tokens={"internal": [11, 12, 13]}
            )

        def update_from_output(self, schedule, result):
            self.requests.clear()
            return {
                0: SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            request_id="internal", finished=True, trace_headers=None
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
    outputs = scheduler.update_from_output(
        schedule,
        SimpleNamespace(req_id_to_index={"internal": 0}, sampled_token_ids=[[20]]),
    )
    OutputProcessor().process_outputs(outputs[0].outputs)
    wire = specopd_vllm.pop_completed("external")
    assert decode_bundle(wire)["records"][0]["proposal_ids"] == [11, 12, 13]
    assert specopd_vllm.pop_completed("external") is None
    assert not scheduler._specopd_schedules


def configuration():
    return {
        "trainer": {"use_v1": False},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp2"},
            "model": {"use_remove_padding": False},
            "rollout": {
                "name": "vllm",
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "speculative_algorithm": "DSPARK",
                    "vllm": {"draft_sample_method": "probabilistic"},
                    "training": {
                        "dspark_block_size": 3,
                        "step": 1,
                        "collect_interval_steps": 1,
                        "training_interval_steps": 1,
                        "specopd": {"enable": True, "proposal_width": 3},
                    },
                },
            },
        },
    }


@pytest.mark.parametrize("separate_llm_manager", [False, True])
@pytest.mark.parametrize("score_target_logz", [False, True])
def test_fit_hooks_replace_normal_forward_and_train_before_actor_update(
    monkeypatch, separate_llm_manager, score_target_logz,
):
    from verl_speco.integration import specopd_training_transport

    monkeypatch.setattr(specopd_training_transport, "dispatch_prepared_group",
        lambda trainer, samples, step, revision, indices, metrics, spec:
        trainer._require_speco_worker_group().train_specopd_group(samples, step, revision, indices))
    events, calls = [], []
    acknowledgements = []
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
    )
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg)
    model.register_forward_hook(lambda *args: calls.append("actor-forward"))
    wire = encode_bundle(
        {
            "schema": 1,
            "request_id": "r",
            "actor_revision": 0,
            "drafter_revision": 0,
            "proposal_temperature": 1.0,
            "records": [
                {
                    "branch_id": "r:0",
                    "prefix_len": 3,
                    "proposal_ids": [11, 12, 13],
                    "prefix_digest": token_digest([1, 2, 3]),
                    "round_id": 0,
                    "accepted_count": 0,
                }
            ],
        },
        10000,
    )
    batch = SimpleNamespace(
        meta_info={},
        non_tensor_batch={"_specopd_proposals": [wire]},
        batch={
            "prompts": torch.tensor([[1, 2, 3]]),
            "responses": torch.tensor([[4, 5, 6]]),
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
            "attention_mask": torch.ones(1, 6),
        },
    )

    class ActorGroup:
        def compute_joint_old_log_prob(self, requests, **options):
            events.append("joint")
            assert options["return_target_logz"] == score_target_logz
            options.pop("microbatch_sizes", None)
            return [score_joint_batch(model, requests, **options)]

    class DrafterGroup:
        def prepare_specopd_bootstrap(self, step):
            events.append("bootstrap")
            return [{"revision": 0}]

        def train_specopd_group(self, samples, step, revision, indices):
            events.append("draft-update")
            assert len(samples[0][0].branches[0].token_ids) == 3
            assert (samples[0][1].branches[0].target_logz is not None) == score_target_logz
            assert indices == (1, 2)
            return [{"trained": True, "revision": revision + 1}]

    class Trainer:
        def __init__(self):
            self.config = configuration()
            training = self.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]
            training["specopd"]["score_target_logz"] = score_target_logz
            from verl_speco.integration.ci_result import CIResult

            self._speco_ci_result = CIResult(training=True)
            self.global_steps = 1
            self.actor_rollout_wg = ActorGroup()
            self.drafter_wg = DrafterGroup()
            self.serving_actor_revision = 0
            def publish_actor():
                events.append("actor-publish")
                self.serving_actor_revision = self.global_steps
            def acknowledge(revision):
                acknowledgements.append((self.serving_actor_revision, revision))
                return {"actor_revision": self.serving_actor_revision, "drafter_revision": revision}
            self.checkpoint_manager = SimpleNamespace(
                update_weights=publish_actor
            )
            self.generator = SimpleNamespace(generate_sequences=lambda data: data, server_handles=[
                SimpleNamespace(speco_set_serving_drafter_revision=SimpleNamespace(remote=acknowledge)),
            ])
            if separate_llm_manager:
                self.llm_server_manager = SimpleNamespace(server_handles=self.generator.server_handles)
                del self.generator.server_handles

        def _compute_old_log_prob(self, data):
            pytest.fail("legacy old-logprob must not run after the joint forward")

        def _update_actor(self, data):
            events.append("actor-update")
            return {}

        def _speco_rollout_generation_target(self):
            return self.generator

        def _speco_store_rollout_metrics(self, result):
            assert not result.meta_info.get("validate", False)

        def _speco_oldlogprob_aux_layer_ids(self):
            return [0, 1]

        def _speco_oldlogprob_aux_layer_id_space(self):
            return "decoder"

        def _ray_get_if_needed(self, value):
            return value

        def _require_speco_worker_group(self):
            return self.drafter_wg

        def _speco_set_drafter_global_step(self):
            events.append("set-step")

        def _speco_sync_target_lm_head_weight(self):
            events.append("head-sync")
            return {}

        def _speco_publish_drafter_weights(self, *args, **kwargs):
            events.append("draft-publish")
            return {"drafter/published": 1}

        def _speco_update_output_metrics(self, output, metrics):
            output.update(metrics)

    monkeypatch.setattr(
        specopd_driver, "make_oldlogprob", lambda batch, results: results
    )
    trainer = Trainer()
    original = trainer._compute_old_log_prob
    with specopd_driver.specopd_fit_hooks(trainer):
        validation = SimpleNamespace(meta_info={"validate": True})
        trainer.generator.generate_sequences(validation)
        trainer.generator.generate_sequences(batch)
        scored, mfu = trainer._compute_old_log_prob(batch)
        assert len(calls) == 1 and scored[0].response_log_probs.shape == (3,)
        actor_output = trainer._update_actor(batch)
        trainer.checkpoint_manager.update_weights()
        assert actor_output["specopd/published"] == 1
        assert actor_output["specopd/published_drafter_revision"] == 1
        assert actor_output["specopd/actor_revision"] == 0
        # The next serving target revision advances after actor publication,
        # while drafter publication has its own independent acknowledged ID.
        trainer.global_steps = 2
        next_bundle = decode_bundle(wire)
        next_bundle.update(actor_revision=1, drafter_revision=1)
        batch.non_tensor_batch["_specopd_proposals"] = [encode_bundle(next_bundle, 10000)]
        trainer.generator.generate_sequences(batch)
        scored, _ = trainer._compute_old_log_prob(batch)
        assert len(calls) == 2
        actor_output = trainer._update_actor(batch)
        trainer.checkpoint_manager.update_weights()
        assert actor_output["specopd/published_drafter_revision"] == 2
        assert actor_output["specopd/actor_revision"] == 1
        with pytest.raises(RuntimeError, match="exactly one rollout"):
            trainer._compute_old_log_prob(batch)
    assert trainer._compute_old_log_prob == original
    assert trainer._speco_ci_result.data["optimizer_steps"] == 2
    assert trainer._speco_ci_result.data["publications"] == 2
    assert events == [
        "set-step",
        "head-sync",
        "bootstrap",
        "draft-publish",
        "joint",
        "set-step",
        "head-sync",
        "draft-update",
        "actor-update",
        "actor-publish",
        "draft-publish",
        "joint",
        "set-step",
        "head-sync",
        "draft-update",
        "actor-update",
        "actor-publish",
        "draft-publish",
    ]
    assert batch.meta_info == {}
    assert acknowledgements == [(0, 0), (1, 1), (2, 2)]


def test_proposal_agent_bridge_isolated_across_concurrent_requests():
    import asyncio

    from verl_speco.integration.agent_loop_runtime import (
        _run_agent_loop_with_student_timing,
    )
    from verl_speco.integration.specopd_agent import (
        PROPOSAL_TURNS,
        collect_proposal_output,
        merge_proposal_output,
    )

    async def trajectory(identity):
        collect_proposal_output(SimpleNamespace(extra_fields={"_specopd_proposals": identity}))
        await asyncio.sleep(0)
        output = SimpleNamespace(extra_fields={})
        merge_proposal_output(output)
        return output

    async def run():
        return await asyncio.gather(*(_run_agent_loop_with_student_timing(trajectory, name) for name in ("a", "b")))

    outputs = asyncio.run(run())
    assert [o.extra_fields["_specopd_proposals"] for o in outputs] == ["a", "b"]
    assert PROPOSAL_TURNS.get() is None


def test_proposal_bridge_rejects_multiple_generation_requests():
    from verl_speco.integration.specopd_agent import (
        PROPOSAL_TURNS,
        merge_proposal_output,
    )

    token = PROPOSAL_TURNS.set(["a", "b"])
    try:
        with pytest.raises(ValueError, match="exactly one"):
            merge_proposal_output(SimpleNamespace(extra_fields={}))
    finally:
        PROPOSAL_TURNS.reset(token)


def test_failed_joint_forward_never_calls_legacy_fallback(monkeypatch):
    # An exception is intentionally propagated: the trainer must not run a
    # second normal forward and pretend the auxiliary path succeeded.
    trainer = SimpleNamespace(config=configuration())
    driver = specopd_driver.SpecOPDDriver(trainer)
    driver.cycle.begin(1)
    with pytest.raises(ValueError, match="missing"):
        driver.old_log_prob(
            SimpleNamespace(
                batch={"input_ids": torch.tensor([[1]])}, non_tensor_batch={}
            )
        )


@pytest.mark.parametrize("replies", [
    [{"actor_revision": 0, "drafter_revision": 1}],
    [{"actor_revision": None, "drafter_revision": 0}],
    [{"actor_revision": 0, "drafter_revision": 0}, {"actor_revision": 1, "drafter_revision": 0}],
])
def test_serving_ack_rejects_wrong_draft_or_inconsistent_actor_revisions(replies):
    servers = [SimpleNamespace(speco_set_serving_drafter_revision=SimpleNamespace(
        remote=lambda revision, reply=reply: reply,
    )) for reply in replies]
    trainer = SimpleNamespace(
        config=configuration(),
        _speco_rollout_generation_target=lambda: SimpleNamespace(server_handles=servers),
        _ray_get_if_needed=lambda values: values,
    )
    driver = specopd_driver.SpecOPDDriver(trainer)
    with pytest.raises(RuntimeError, match="acknowledge"):
        driver.acknowledge_serving_revision(0)


@pytest.mark.parametrize("limit,quotas", [(80, [10] * 8), (3, [1] * 3 + [0] * 5)])
def test_serving_ack_distributes_global_collection_limit(limit, quotas):
    cfg = configuration()
    cfg["actor_rollout_ref"]["rollout"]["drafter"]["training"]["specopd"]["max_collect_samples_per_step"] = limit
    received = []

    def acknowledge(revision, *, collection_quota):
        received.append(collection_quota)
        return {"actor_revision": 4, "drafter_revision": revision, "collection_quota": collection_quota}

    server = SimpleNamespace(speco_set_serving_drafter_revision=SimpleNamespace(remote=acknowledge))
    trainer = SimpleNamespace(config=cfg, llm_server_manager=SimpleNamespace(server_handles=[server] * 8),
                              _ray_get_if_needed=lambda values: values)
    specopd_driver.SpecOPDDriver(trainer).acknowledge_serving_revision(0)
    assert received == quotas


def test_mrv2_request_tracking_is_runner_local_and_cleans_evictions():
    class Runner:
        def __init__(self):
            self.draft_tokens_handler = SimpleNamespace()
        def add_requests(self, output):
            return "added"
        def _remove_request(self, request_id):
            return "removed"
    assert specopd_vllm.patch_draft_token_request_tracking(Runner)
    left, right = Runner(), Runner()
    def request(rid, branches):
        return SimpleNamespace(req_id=rid, sampling_params=SimpleNamespace(
            extra_args={"specopd_plan": {"max_branches": branches}}))
    assert left.add_requests(SimpleNamespace(scheduled_new_reqs=[request("yes", 8), request("no", 0)])) == "added"
    right.add_requests(SimpleNamespace(scheduled_new_reqs=[]))
    assert set(left.draft_tokens_handler._specopd_plans) == {"yes"}
    assert right.draft_tokens_handler._specopd_plans == {}
    assert left._remove_request("yes") == "removed"
    assert left.draft_tokens_handler._specopd_plans == {}


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
        trainer._require_speco_worker_group().train_specopd_group(samples, step, revision, indices))
    events = []
    training = dict(step=20, collect_interval_steps=5, training_interval_steps=5,
                    dspark_block_size=7, specopd=dict(enable=True))
    cfg = dict(trainer=dict(use_v1=True, v1=dict(trainer_mode="sync")), actor_rollout_ref=dict(
        actor=dict(strategy="fsdp2", ulysses_sequence_parallel_size=4),
        model=dict(use_remove_padding=True), rollout=dict(name="vllm", temperature=1.0,
            drafter=dict(enable=True, enable_drafter_training=True, speculative_algorithm="DSPARK",
                         vllm=dict(draft_sample_method="greedy"), training=training))))
    trainer = SimpleNamespace(config=cfg, global_steps=1, actor_revision=0)
    worker = SimpleNamespace(train_specopd_group=lambda samples, step, revision, indices: events.append(("train", step, 20)) or [dict(trained=True, revision=revision+1, optimizer_updates=20)])
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
            driver.cycle.actor_revision = trainer.actor_revision
            request = SimpleNamespace(branches=[SimpleNamespace(token_ids=(1, 2))])
            driver.cycle.scored(((request, None),), trainer.actor_revision, driver.cycle.drafter_revision)
        driver.before_actor_update()
        trainer.actor_revision = step
        driver.finish_step()
        assert driver.metrics["specopd/ack_actor_revision"] == step
        assert driver.metrics["specopd/ack_drafter_revision"] == step // 5
        assert driver.metrics["specopd/optimizer_updates"] == (20 if due else 0)
        assert driver.metrics["specopd/published"] == int(due)
    assert events == [("train", 5, 20), ("publish", 5), ("train", 10, 20), ("publish", 10)]
