# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""CPU contracts for the V1 Draft-OPD adapter."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin
from verl_speco.integration.agent_loop_runtime import (
    _build_speco_agent_loop_worker_class,
)
from verl_speco.integration.verify_trace import TRACE_KEY, TURN_TRACES, token_digest
from verl_speco.backends.drafter_opd import preprocess_trace
from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer
from tests.integration.test_drafter_runtime_control_contract import (
    _oldlogprob_collect_plan_trainer,
)


def run_inline(coroutine):
    # These fake parents never suspend. Drive the coroutine without opening a
    # Windows event-loop socket (unavailable in restricted CPU CI sandboxes).
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    finally:
        coroutine.close()
    raise AssertionError("CPU postprocess stub unexpectedly suspended")


def config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "speculative_algorithm": "DSPARK",
                        "training": {
                            "collect_hidden_states_from_old_logprob": True,
                            "publish_async": False,
                            "drafter_opd": {
                                "enable": True,
                                "capture_verify_trace": True,
                            },
                        },
                    },
                }
            },
            "trainer": {
                "v1": {"trainer_mode": "sync", "sync": {"parameter_sync_step": 1}}
            },
            "algorithm": {},
        }
    )


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("actor_rollout_ref.rollout.drafter.enable", False, "drafter.enable=true"),
        (
            "actor_rollout_ref.rollout.drafter.enable_drafter_training",
            False,
            "enable_drafter_training=true",
        ),
        (
            "actor_rollout_ref.rollout.drafter.training.drafter_opd.capture_verify_trace",
            False,
            "capture_verify_trace=true",
        ),
        (
            "actor_rollout_ref.rollout.drafter.speculative_algorithm",
            "EAGLE3",
            "speculative_algorithm",
        ),
        (
            "actor_rollout_ref.rollout.drafter.training.dspark_loss_mode",
            "restricted_ce",
            "loss_mode=full_vocab",
        ),
        (
            "actor_rollout_ref.rollout.drafter.training.packing.enable",
            True,
            "packing.enable=false",
        ),
        ("trainer.v1.trainer_mode", "unsupported", "trainer_mode=sync"),
        (
            "actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy",
            "rollout_idle_worker",
            "execution.strategy=sync",
        ),
        ("trainer.v1.sync.parameter_sync_step", 2, "parameter_sync_step=1"),
        (
            "actor_rollout_ref.rollout.drafter.training.drafter_opd.max_policy_staleness",
            -1,
            "max_policy_staleness",
        ),
        (
            "actor_rollout_ref.rollout.drafter.training.drafter_opd.max_drafter_staleness",
            1.5,
            "max_drafter_staleness",
        ),
        (
            "actor_rollout_ref.rollout.drafter.training.publish_async",
            True,
            "publish_async=false",
        ),
        (
            "actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.enabled",
            True,
            "freeze.enabled=false",
        ),
    ],
)
def test_configuration_fails_before_worker_creation(path, value, match):
    cfg = config()
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match=match):
        SpecoV1Mixin._speco_validate_v1_training_config(cfg)


@pytest.mark.parametrize(
    "enabled,last_n,allowed", [(False, 5, True), (True, 1, True), (True, 2, False)]
)
def test_buffer_gate(enabled, last_n, allowed):
    cfg = config()
    cfg.actor_rollout_ref.rollout.drafter.training.update(
        use_data_buffer=enabled, sample_last_n_steps=last_n
    )
    if allowed:
        SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    else:
        with pytest.raises(ValueError, match="sample_last_n_steps=1"):
            SpecoV1Mixin._speco_validate_v1_training_config(cfg)


@pytest.mark.parametrize(
    "mode, local_steps", [("colocate_async", 1), ("separate_async", 2)]
)
def test_async_config_permits_replay_and_preserves_proximal_actor(mode, local_steps):
    cfg = config()
    cfg.trainer.v1.trainer_mode = mode
    cfg.trainer.v1[mode] = {"parameter_sync_step": local_steps}
    training = cfg.actor_rollout_ref.rollout.drafter.training
    training.update(use_data_buffer=True, sample_last_n_steps=5)
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    if mode == "colocate_async":
        cfg.trainer.v1[mode].parameter_sync_step = 2
        with pytest.raises(ValueError, match="colocate_async.parameter_sync_step=1"):
            SpecoV1Mixin._speco_validate_v1_training_config(cfg)


def test_strict_drafter_lag_requires_publication_on_each_training_boundary():
    cfg = config()
    cfg.trainer.v1.trainer_mode = "colocate_async"
    training = cfg.actor_rollout_ref.rollout.drafter.training
    training.update(training_interval_steps=2, publish_interval_steps=4)
    with pytest.raises(ValueError, match="publication on every drafter training step"):
        SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    training.publish_interval_steps = 2
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
def test_supported_algorithms_and_disabled_opd(algorithm):
    cfg = config()
    cfg.actor_rollout_ref.rollout.drafter.speculative_algorithm = algorithm
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    cfg.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable = False
    cfg.trainer.v1.trainer_mode = "separate_async"
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)


def trace(layout):
    return {
        "layout": layout,
        "num_spec_tokens": 3,
        "prompt_length": 2,
        "response_length": 4,
        "drafter_revision_min": 0,
        "drafter_revision_max": 0,
        "prefix_digest": token_digest([11, 12, 21, 22, 23, 24]),
        "steps": [
            {
                "anchor": 1,
                "commit": 1,
                "drafts": 0,
                "accepted": 0,
                "raw_generated": 1,
                "truncated": False,
            },
            {
                "anchor": 2,
                "commit": 3,
                "drafts": 3,
                "accepted": 2,
                "raw_generated": 3,
                "truncated": False,
            },
        ],
    }


def tq_batch(monkeypatch, evidence, *, mode="sync", step=1, revision=0, policy_tag=0):
    # Deliberately long synthetic padding: it must not rely on the short-row rule.
    data = {
        "prompts": torch.tensor([[11, 12], [31, 32]]),
        "responses": torch.tensor([[21, 22, 23, 24], [41, 42, 43, 44]]),
        "input_ids": torch.tensor([[11, 12, 21, 22, 23, 24], [31, 32, 41, 42, 43, 44]]),
        "response_mask": torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]]),
    }
    fields = [
        {TRACE_KEY: evidence, "_speco_vllm_request_id": "real"},
        {TRACE_KEY: evidence},
    ]

    def get(**kwargs):
        return (
            {"extra_fields": fields}
            if kwargs["select_fields"] == ["extra_fields"]
            else data
        )

    monkeypatch.setitem(
        sys.modules, "transfer_queue", SimpleNamespace(kv_batch_get=get)
    )
    trainer = SpecoV1Mixin()
    trainer.config = config()
    trainer.config.trainer.v1.trainer_mode = mode
    trainer.global_steps = step
    trainer._speco_v1_opd_drafter_revision = revision
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    batch = SimpleNamespace(
        keys=["real", "padding"],
        partition_id="train",
        tags=[{"min_global_steps": policy_tag}, {"is_padding": True}],
    )
    return trainer._speco_v1_batch_data(batch)[1]


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
@pytest.mark.parametrize("multiturn", [False, True])
def test_tq_and_legacy_collection_and_anchor_plans_match(
    monkeypatch, layout, multiturn
):
    evidence = trace(layout)
    if multiturn:
        second = {
            "layout": layout,
            "num_spec_tokens": 3,
            "prompt_length": 5,
            "response_length": 1,
            "drafter_revision_min": 0,
            "drafter_revision_max": 0,
            "steps": [
                {
                    "anchor": 4,
                    "commit": 1,
                    "drafts": 3,
                    "accepted": 0,
                    "raw_generated": 1,
                    "truncated": False,
                }
            ],
        }
        first = deepcopy(evidence)
        first["response_length"] = 3
        first["steps"][1].update(commit=2, truncated=True)
        evidence = {"turns": [first, second]}
    v1 = tq_batch(monkeypatch, evidence)
    assert v1.non_tensor_batch["_speco_v1_is_padding"].tolist() == [False, True]
    assert v1.non_tensor_batch[TRACE_KEY].tolist() == [
        {**evidence, "target_revision_min": 0},
        None,
    ]
    assert v1.non_tensor_batch["_speco_opd_policy_lag"].tolist() == [0, 0]
    assert v1.non_tensor_batch["_speco_opd_drafter_lag"].tolist() == [0, 0]
    assert v1.batch["attention_mask"][0].tolist() == [1] * 6
    assert v1.batch["response_mask"][0].tolist() == [1, 0, 0, 1]
    legacy = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[21, 22, 23, 24]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.tensor([[1, 0, 0, 1]]),
        },
        non_tensors={TRACE_KEY: np.asarray([evidence], dtype=object)},
    )
    samples = []
    for batch in (legacy, v1):
        trainer = _oldlogprob_collect_plan_trainer()
        trainer._speco_drafter_training_config()["drafter_opd"] = {"enable": True}
        plan = trainer._speco_build_oldlogprob_collect_plan(batch)
        assert plan["collect_mask"].tolist() == (
            [True] if batch is legacy else [True, False]
        )
        assert trainer._speco_last_oldlogprob_padding_skipped == (
            0 if batch is legacy else 1
        )
        captured = {}
        trainer._speco_execute_collection = lambda plan, payload: captured.setdefault(
            "payload", payload
        )
        from verl_speco.integration.oldlogprob_runtime import (
            OLD_LOGPROB_HIDDEN_STATES_KEY,
        )

        trainer._speco_collect_oldlogprob_features(
            batch, plan, {OLD_LOGPROB_HIDDEN_STATES_KEY: torch.ones(len(batch), 3, 2)}
        )
        samples.append(captured["payload"].buckets[0][0])
    for key in ("input_ids", "response_mask", "hidden_positions"):
        assert torch.equal(samples[0][key], samples[1][key])
    assert {**samples[0][TRACE_KEY], "target_revision_min": 0} == samples[1][TRACE_KEY]
    plans = [
        preprocess_trace(s, start=0, end=6, layout=layout, capacity=3, device="cpu")
        for s in samples
    ]
    assert torch.equal(*plans)
    assert (plans[0][:, 0] >= 0).any()


def test_all_padding_returns_no_collection_plan(monkeypatch):
    batch = tq_batch(monkeypatch, trace("dspark"))
    batch.non_tensor_batch["_speco_v1_is_padding"][:] = True
    trainer = _oldlogprob_collect_plan_trainer()
    assert trainer._speco_build_oldlogprob_collect_plan(batch) is None
    assert trainer._speco_last_oldlogprob_padding_skipped == 2
    assert trainer._speco_last_oldlogprob_short_response_skipped == 0


def test_missing_real_trace_still_fails_with_request_id(monkeypatch):
    batch = tq_batch(monkeypatch, None)
    trainer = _oldlogprob_collect_plan_trainer()
    trainer._speco_drafter_training_config()["drafter_opd"] = {"enable": True}
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    from verl_speco.integration.oldlogprob_runtime import OLD_LOGPROB_HIDDEN_STATES_KEY

    with pytest.raises(ValueError, match="step=1, batch_idx=0, request_id=real"):
        trainer._speco_collect_oldlogprob_features(
            batch, plan, {OLD_LOGPROB_HIDDEN_STATES_KEY: torch.ones(2, 3, 2)}
        )


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
@pytest.mark.parametrize("policy_limit,drafter_limit", [(0, 0), (2, 0), (0, 1), (2, 1)])
def test_async_staleness_excludes_each_axis_before_collection(
    monkeypatch, mode, policy_limit, drafter_limit
):
    turn = trace("dspark")
    turn["drafter_revision_max"] = 1
    evidence = {"turns": [turn], "target_revision_min": 1}
    batch = tq_batch(monkeypatch, evidence, mode=mode, step=4, revision=1, policy_tag=3)
    assert batch.non_tensor_batch["_speco_opd_policy_lag"].tolist() == [2, 0]
    assert batch.non_tensor_batch["_speco_opd_drafter_lag"].tolist() == [1, 0]
    assert batch.non_tensor_batch["_speco_opd_mixed_revision_turns"].tolist() == [1, 0]
    trainer = _oldlogprob_collect_plan_trainer()
    trainer._speco_drafter_training_config()["drafter_opd"] = {
        "enable": True,
        "max_policy_staleness": policy_limit,
        "max_drafter_staleness": drafter_limit,
    }
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert trainer._speco_last_opd_stale_samples_skipped == {
        "stale_samples_skipped_policy": int(policy_limit < 2),
        "stale_samples_skipped_drafter": int(drafter_limit < 1),
    }
    if policy_limit < 2 or drafter_limit < 1:
        assert plan is None
    else:
        assert plan["collect_mask"].tolist() == [True, False]


def test_buffer_rechecks_revisions_when_training_advances():
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = config().actor_rollout_ref
    trainer.current_rl_step = 3
    trainer.opd_drafter_revision = 2
    opd = trainer.config.rollout.drafter.training.drafter_opd
    old = {TRACE_KEY: {**trace("dspark"), "target_revision_min": 0}}
    fresh = {
        TRACE_KEY: {
            **trace("dspark"),
            "target_revision_min": 2,
            "drafter_revision_min": 2,
            "drafter_revision_max": 2,
        }
    }
    assert trainer._filter_opd_training_data([old, fresh, {}]) == [fresh]
    opd.max_policy_staleness = 2
    assert trainer._filter_opd_training_data([old, fresh]) == [fresh]
    opd.max_drafter_staleness = 2
    assert trainer._filter_opd_training_data([old, fresh]) == [old, fresh]
    trainer.current_rl_step += 1
    assert trainer._filter_opd_training_data([old, fresh]) == [fresh]

    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.use_data_buffer = True
    trainer.collected_data = []
    trainer.batch_size = 1
    trainer.buffer_version = 1
    buffered = [{**old, "step": 1}, {**fresh, "step": 3}]
    trainer.data_buffer = SimpleNamespace(
        get_all_data=lambda: buffered,
        get_data_from_last_n_steps=lambda _: buffered,
    )
    status = trainer.get_training_data_status(sample_last_n_steps=3)
    assert status["buffer_samples"] == 2
    assert status["trainable_samples"] == 1
    assert status["data_version"] == 3


def test_separate_async_exports_head_from_same_snapshot_as_hidden_states(monkeypatch):
    seen = []

    class Worker:
        policy = 1

        def save_model_to_cpu(self, slot):
            seen.append(("save", slot))

        def restore_model_from_cpu(self, slot):
            self.policy = slot

        def clear_cpu_model(self, slot):
            seen.append(("clear", slot))

        def compute_log_prob(self, control):
            assert self.policy == 0

    trainer = SpecoV1Mixin()
    trainer.config = config()
    trainer.config.trainer.v1.trainer_mode = "separate_async"
    trainer.local_trigger_step = 1
    trainer.actor_rollout_wg = Worker()
    trainer._speco_v1_cache_old_policy_head = lambda: None

    def start(self, training_plan):
        assert self.actor_rollout_wg.policy == 0
        return "head"

    monkeypatch.setattr(
        SpecoRayPPOTrainer, "_speco_start_target_lm_head_weight_sync", start
    )
    trainer._speco_compute_collection_log_prob(None)
    assert trainer.actor_rollout_wg.policy == 1
    assert trainer._speco_start_target_lm_head_weight_sync(None) == "head"
    assert trainer.actor_rollout_wg.policy == 1
    assert seen == [("save", 1), ("clear", 1)] * 2


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("published", [False, True])
def test_serving_revision_advances_only_after_publication(mode, published):
    events = []

    class Upstream:
        def on_step_end(self):
            events.append("sync_actor")

    class Trainer(SpecoV1Mixin, Upstream):
        def _speco_rollout_idle_worker_enabled(self):
            return False

        def _speco_rollout_admission_rpc(self, method, *args):
            events.append((method, *args))

        def _speco_publish_drafter_weights(self, *args, **kwargs):
            events.append("publish")
            return {"drafter/published": int(published)}

        def _speco_wait_pending_drafter_publish(self):
            events.append("ack")
            return 0

    trainer = Trainer()
    trainer.config = config()
    trainer.config.trainer.v1.trainer_mode = mode
    trainer.checkpoint_manager = trainer.standalone_checkpoint_manager = (
        SimpleNamespace(
            abort_replicas=lambda: events.append("abort"),
            sleep_replicas=lambda: events.append("sleep"),
        )
    )
    trainer._speco_colocate_probe_boundary_enabled = lambda: False
    trainer._speco_v1_pending_training = True
    trainer._speco_v1_opd_drafter_revision = 2
    if mode == "colocate_async":
        trainer.on_sample_end()
    trainer.on_step_end()
    boundary = (
        [("speco_close_rollout_admission",), "abort", ("speco_wait_rollout_idle", True)]
        if mode != "sync"
        else []
    )
    if mode == "colocate_async":
        boundary.append("sleep")
    assert events == [
        *boundary,
        "sync_actor",
        "publish",
        "ack",
        *([("speco_set_serving_drafter_revision", 2)] if published else []),
        *([("speco_resume_rollout_admission",)] if mode != "sync" else []),
    ]


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_engine_abort_delivers_verified_prefix_and_resumes_at_prefill(
    monkeypatch, layout
):
    from verl_speco.integration import vllm_runtime as runtime
    from verl_speco.integration.partial_rollout_stats import (
        patch_partial_rollout_client,
    )
    from verl_speco.integration.verify_trace import (
        collect_turn_trace,
        finish_serving_trace,
        merge_turn_traces,
    )

    class Scheduler:
        num_spec_tokens = 3
        num_sampled_tokens_per_step = 1

        def update_from_output(self, scheduled, model_output):
            ids = model_output.sampled_token_ids[0]
            self.requests["internal"]._output_token_ids.extend(ids)
            return {
                0: SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            request_id="internal",
                            new_token_ids=ids,
                            finish_reason=model_output.reason,
                        )
                    ]
                )
            }

    class RequestState:
        def make_request_output(self, finish_reason):
            if finish_reason:
                return SimpleNamespace(request_id="external", finished=True)
            return None

    class Processor:
        def process_outputs(self, outputs):
            for item in outputs:
                self.request_states[item.request_id].make_request_output(
                    item.finish_reason
                )

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.core.sched.scheduler",
        SimpleNamespace(Scheduler=Scheduler),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.output_processor",
        SimpleNamespace(OutputProcessor=Processor, RequestState=RequestState),
    )
    monkeypatch.setattr(
        runtime,
        "_load_env_drafter_config",
        lambda: {
            "speculative_algorithm": layout.upper(),
            "training": {"drafter_opd": {"capture_verify_trace": True}},
        },
    )
    assert runtime.patch_vllm_request_acceptance_stats()

    def segment(prompt, steps, *, abort, version):
        scheduler = Scheduler()
        scheduler.requests = {
            "internal": SimpleNamespace(
                prompt_token_ids=prompt, _output_token_ids=[], is_finished=lambda: False
            )
        }
        processor = Processor()
        processor.request_states = {"internal": RequestState()}
        for ids, drafts, reason in steps:
            outputs = scheduler.update_from_output(
                SimpleNamespace(
                    num_scheduled_tokens={"internal": 1},
                    num_invalid_spec_tokens={},
                    scheduled_spec_decode_tokens={"internal": [0] * drafts},
                ),
                SimpleNamespace(
                    req_id_to_index={"internal": 0},
                    sampled_token_ids=[ids],
                    reason=reason,
                ),
            )
            processor.process_outputs(outputs[0].outputs)
        if abort:
            # Core aborts emit a terminal output with no scheduler trace header.
            processor.process_outputs(
                [SimpleNamespace(request_id="internal", finish_reason="abort")]
            )
        fields = runtime._vllm_staged_request_accept_stats_to_extra_fields(
            (), {"request_id": "external"}
        )
        output = SimpleNamespace(
            extra_fields=fields,
            stop_reason="aborted" if abort else "completed",
            token_ids=scheduler.requests["internal"]._output_token_ids,
        )
        finish_serving_trace(output, prompt, version, version, target_version=version)
        collect_turn_trace(output)
        return output

    class FullyAsyncClient:
        async def generate(self, prompt):
            first = segment(
                prompt, [([21], 0, None), ([22, 23], 3, None)], abort=True, version=2
            )
            prefix = [*prompt, *first.token_ids]
            empty = segment(prefix, [], abort=True, version=3)
            last = segment(
                prefix,
                [([24], 0, None), ([25, 26], 3, "length")],
                abort=False,
                version=4,
            )
            return SimpleNamespace(
                prompt_ids=prompt,
                response_ids=first.token_ids + empty.token_ids + last.token_ids,
                extra_fields={},
            )

    patch_partial_rollout_client(FullyAsyncClient)
    token = TURN_TRACES.set([])
    try:
        output = run_inline(FullyAsyncClient().generate([11, 12]))
        merge_turn_traces(output)
    finally:
        TURN_TRACES.reset(token)
    evidence = output.extra_fields[TRACE_KEY]
    first, last = evidence["turns"]
    assert evidence["target_revision_min"] == 2
    assert first["response_length"] == 3
    assert last["prompt_length"] == 5
    assert last["steps"][0]["anchor"] == 4
    assert last["steps"][0]["drafts"] == 0
    original = {**first, "response_length": 6, "steps": first["steps"] + last["steps"]}
    kwargs = dict(start=0, end=8, layout=layout, capacity=3, device="cpu")
    resumed = preprocess_trace({TRACE_KEY: evidence}, **kwargs)
    assert torch.equal(resumed, preprocess_trace({TRACE_KEY: original}, **kwargs))
    assert resumed[:, 0].ge(0).nonzero().flatten().tolist() == [2, 5]


def test_nongenerating_abort_is_optional_but_nonempty_mismatch_raises():
    from verl_speco.integration.verify_trace import finish_serving_trace

    output = SimpleNamespace(extra_fields={}, token_ids=[], stop_reason="aborted")
    finish_serving_trace(output, [11, 12], 0, 0)
    output.token_ids.append(21)
    with pytest.raises(ValueError, match="trace missing"):
        finish_serving_trace(output, [11, 12], 0, 0)
    output.extra_fields[TRACE_KEY] = trace("dspark")
    with pytest.raises(ValueError, match="response length differ"):
        finish_serving_trace(output, [11, 12], 0, 0)


def test_rollout_boundary_drains_requests_submitted_after_first_abort(monkeypatch):
    import asyncio

    from verl_speco.integration.vllm_runtime import _SpecoVLLMHttpServerMixin

    events = []

    async def fake_wait_for(awaitable, timeout):
        if not events:
            awaitable.close()
            raise asyncio.TimeoutError
        await awaitable

    class Server(_SpecoVLLMHttpServerMixin):
        async def abort_all_requests(self, *, reset_prefix_cache):
            assert not reset_prefix_cache
            events.append("abort")
            return {}

    server = Server()
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    run_inline(server.speco_wait_rollout_idle(True))
    assert events == ["abort"]


@pytest.mark.parametrize("as_list", [False, True])
def test_agent_worker_single_output_keeps_digest_validation(as_list):
    class Parent:
        async def _agent_loop_postprocess(self, output, validate, **kwargs):
            return output

    worker_cls = _build_speco_agent_loop_worker_class(Parent)
    worker = worker_cls.__new__(worker_cls)
    worker.config = config()
    output = SimpleNamespace(
        prompt_ids=[11, 12], response_ids=[21, 22, 23, 24], extra_fields={}
    )
    token = TURN_TRACES.set([trace("dspark")])
    try:
        value = [output] if as_list else output
        assert run_inline(worker._agent_loop_postprocess(value, False)) is value
        assert output.extra_fields[TRACE_KEY]["turns"] == [trace("dspark")]
        output.response_ids[-1] = 99
        with pytest.raises(ValueError, match="tokens differ"):
            run_inline(worker._agent_loop_postprocess(value, False))
    finally:
        TURN_TRACES.reset(token)


def test_multi_output_rejected_even_when_trace_missing():
    class Parent:
        async def _agent_loop_postprocess(self, output, validate, **kwargs):
            return output

    worker_cls = _build_speco_agent_loop_worker_class(Parent)
    worker = worker_cls.__new__(worker_cls)
    worker.config = config()
    with pytest.raises(ValueError, match="multi-output.*P3"):
        run_inline(worker._agent_loop_postprocess([object(), object()], False))


@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_update_actor_forwards_opd_metrics(layout):
    class Upstream:
        def _update_actor(self, batch, metrics):
            metrics["actor/updated"] = 1
            return "updated"

    class Harness(SpecoV1Mixin, Upstream):
        def _speco_online_enabled_from_config(self, cfg):
            return True

        def _speco_rollout_idle_worker_enabled(self):
            return False

        def _speco_v1_collect_acceptance_feedback(self, batch):
            pass

        def _speco_on_before_actor_update(self):
            return SimpleNamespace(
                training_plan=SimpleNamespace(launch=True), metrics={}
            )

        def _speco_train_drafter(self, plan):
            return True, {
                f"{layout}/opd/trace_blocks": 3,
                "drafter/train_optimizer_step_max": 2,
            }

    trainer = Harness()
    trainer.config = config()
    trainer._speco_freeze_policy = None
    metrics = {}
    assert trainer._update_actor(None, metrics) == "updated"
    assert metrics[f"{layout}/opd/trace_blocks"] == 3
    assert metrics["drafter/train_optimizer_step_max"] == 2
    assert trainer._speco_v1_pending_training


def test_padding_excluded_from_non_opd_training_too(monkeypatch):
    batch = tq_batch(monkeypatch, None)
    trainer = _oldlogprob_collect_plan_trainer()
    assert isinstance(trainer, SpecoRayPPOTrainer)
    plan = trainer._speco_build_oldlogprob_collect_plan(batch)
    assert plan is None  # real row has only two assistant tokens (< hidden_rows)
    assert trainer._speco_last_oldlogprob_padding_skipped == 1
