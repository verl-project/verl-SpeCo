# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Phase diagnostics must not add synchronization or change engine semantics."""

from contextlib import contextmanager
import json
from types import SimpleNamespace
from types import MethodType

import pytest

from verl_speco.integration.phase_timing import (
    install_engine_phase_timing,
    phase_span,
    phase_timing_enabled,
    configure_phase_logger,
    log_driver_phase_setup,
    ref_dispatch_trace,
    install_ref_worker_phase_timing,
)
from verl_speco.trainer.scheduler.collection_executor import (
    CallbackDrafterCollectionExecutor,
)


@pytest.fixture(autouse=True)
def capture_dedicated_phase_logger(caplog):
    from verl_speco.integration.phase_timing import logger

    handlers, level, propagate, disabled = (
        list(logger.handlers),
        logger.level,
        logger.propagate,
        logger.disabled,
    )
    logger.addHandler(caplog.handler)
    yield
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
    logger.setLevel(level)
    logger.propagate, logger.disabled = propagate, disabled


def test_driver_trace_survives_disabled_and_filtered_logger(
    monkeypatch, caplog, capsys
):
    import logging
    from verl_speco.integration.phase_timing import logger

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    logger.disabled = True
    logger.setLevel(logging.ERROR)
    trainer_logger = logging.getLogger("test.disabled.trainer")
    trainer_logger.disabled = True
    log_driver_phase_setup(
        enabled=True,
        trainer_logger=trainer_logger,
        source_file="deployed/trainer.py",
        ref_available=True,
    )
    with phase_span("ref_driver", source_step=12):
        pass
    output = capsys.readouterr().err
    assert "[SpecoPhaseSetup]" in output and "trainer_logger_disabled=True" in output
    assert '"phase": "ref_driver"' in output
    configure_phase_logger()
    assert (
        sum(bool(getattr(h, "_speco_phase_handler", False)) for h in logger.handlers)
        == 1
    )
    assert trainer_logger.disabled  # Do not globally unmute training logs.


def test_disabled_span_does_not_read_clocks(monkeypatch, caplog):
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)
    monkeypatch.setattr(
        "verl_speco.integration.phase_timing.time.perf_counter",
        lambda: pytest.fail("clock"),
    )
    metrics = {}
    with phase_span("disabled", metrics=metrics):
        pass
    assert not metrics
    assert not caplog.records


@pytest.mark.parametrize("bubble,enabled", [(True, True), (True, False), (False, True)])
def test_root_diagnostic_env_preserves_custom_ray_settings(
    monkeypatch, bubble, enabled
):
    from omegaconf import OmegaConf
    from verl_speco.main import configure_phase_diagnostic_runtime_env

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true" if enabled else "false")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", "true")
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "training": {
                            "scheduler": {
                                "execution": {
                                    "strategy": "rollout_idle_worker"
                                    if bubble
                                    else "sync"
                                }
                            }
                        }
                    }
                }
            },
            "ray_kwargs": {
                "ray_init": {
                    "runtime_env": {
                        "env_vars": {"CUSTOM_SETTING": "keep"},
                        "working_dir": "/workspace",
                    }
                }
            },
        }
    )
    before = OmegaConf.to_container(config)
    configure_phase_diagnostic_runtime_env(config)
    env = config.ray_kwargs.ray_init.runtime_env
    assert env.env_vars.CUSTOM_SETTING == "keep"
    assert env.working_dir == "/workspace"
    if bubble and enabled:
        assert env.env_vars.VERL_SPECO_PHASE_TIMING == "true"
        assert env.env_vars.VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL == "true"
    else:
        assert OmegaConf.to_container(config) == before


def test_span_records_exception_and_propagates_same_object(monkeypatch, caplog):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    error = ValueError("original")
    metrics = {}
    with pytest.raises(ValueError) as raised:
        with phase_span("failed", metrics=metrics, source_step=8):
            raise error
    assert raised.value is error
    row = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert row["failed"] and row["source_step"] == 8
    assert row["end_unix_s"] >= row["start_unix_s"]
    assert metrics["timing_s/phase_failed"] >= 0


def test_disabled_engine_keeps_original_methods(monkeypatch):
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)
    engine = SimpleNamespace(to=lambda: "original")
    original = engine.to
    install_engine_phase_timing(engine, role="actor", rank=0)
    assert engine.to is original
    assert not hasattr(engine, "_speco_phase_timed")


@pytest.mark.parametrize(
    "representation", ["dataproto", "mapping", "wrapped", "tensordict"]
)
def test_ref_trace_correlates_worker_entry_and_engine_without_changing_result(
    monkeypatch, caplog, representation
):
    from verl_speco.integration.phase_timing import _REF_TRACE_KEY

    if representation == "tensordict":
        td = pytest.importorskip("tensordict")
        from tensordict.tensorclass import NonTensorData
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    engine = SimpleNamespace(infer_batch=lambda data: data)
    install_engine_phase_timing(engine, role="ref", rank=2)

    def compute(data):
        return engine.infer_batch(data)

    compute.dispatch_marker = "preserve"
    worker = SimpleNamespace(rank=2, compute_ref_log_prob=compute)
    install_ref_worker_phase_timing(worker)
    wrapped = worker.compute_ref_log_prob
    install_ref_worker_phase_timing(worker)
    assert worker.compute_ref_log_prob is wrapped
    assert wrapped.dispatch_marker == "preserve"
    batch = SimpleNamespace(meta_info={"existing": 42})
    with ref_dispatch_trace(batch, source_step=8) as trace:
        if representation == "dataproto":
            worker_input = batch
        elif representation == "mapping":
            worker_input = dict(batch.meta_info)
        elif representation == "wrapped":
            worker_input = {_REF_TRACE_KEY: SimpleNamespace(data=dict(trace))}
        else:
            # Same metadata representation produced by DataProto.to_tensordict.
            worker_input = td.TensorDict({}, batch_size=[])
            for key, value in batch.meta_info.items():
                worker_input[key] = NonTensorData(value)
        assert worker.compute_ref_log_prob(worker_input) is worker_input
    assert batch.meta_info == {"existing": 42}
    rows = [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records]
    entry = next(row for row in rows if row["phase"] == "ref_worker")
    infer = next(row for row in rows if row["phase"] == "engine_infer_batch")
    assert entry["trace_id"] == infer["trace_id"] == trace["trace_id"]
    assert entry["source_step"] == 8 and entry["rank"] == 2
    assert entry["dispatch_to_worker_observed_s"] >= 0
    assert infer["worker_entry_to_call_s"] >= 0
    caplog.clear()
    engine.infer_batch(batch)
    assert "trace_id" not in json.loads(
        caplog.records[-1].getMessage().split(" ", 1)[1]
    )


@pytest.mark.parametrize(
    "value", [None, 42, "invalid", SimpleNamespace(data="invalid")]
)
def test_invalid_tensordict_trace_does_not_change_ref_result(
    monkeypatch, caplog, value
):
    from verl_speco.integration.phase_timing import _REF_TRACE_KEY

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    batch = {_REF_TRACE_KEY: value}
    worker = SimpleNamespace(rank=0, compute_ref_log_prob=lambda data: data)
    install_ref_worker_phase_timing(worker)
    assert worker.compute_ref_log_prob(batch) is batch
    row = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert row["phase"] == "ref_worker"
    assert "trace_id" not in row and "dispatch_to_worker_observed_s" not in row


def test_ref_trace_restores_metadata_and_context_on_exception(monkeypatch, caplog):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    from verl_speco.integration.phase_timing import _REF_TRACE_KEY, _ref_trace

    sentinel = object()
    batch = SimpleNamespace(meta_info={_REF_TRACE_KEY: sentinel})
    error = ValueError("original")

    def compute(data):
        raise error

    worker = SimpleNamespace(rank=0, compute_ref_log_prob=compute)
    install_ref_worker_phase_timing(worker)
    with pytest.raises(ValueError) as raised:
        with ref_dispatch_trace(batch, source_step=4):
            worker.compute_ref_log_prob(data=batch)
    assert raised.value is error
    assert batch.meta_info[_REF_TRACE_KEY] is sentinel
    assert _ref_trace.get() is None
    row = json.loads(caplog.records[-1].getMessage().split(" ", 1)[1])
    assert row["failed"] and row["source_step"] == 4


def test_disabled_ref_trace_leaves_metadata_and_rpc_untouched(monkeypatch, caplog):
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)
    monkeypatch.setattr(
        "verl_speco.integration.phase_timing.time.perf_counter",
        lambda: pytest.fail("diagnostic clock when disabled"),
    )
    batch = SimpleNamespace(meta_info={"existing": 1})

    def original(data):
        return data

    worker = SimpleNamespace(compute_ref_log_prob=original)
    install_ref_worker_phase_timing(worker)
    with ref_dispatch_trace(batch, source_step=4) as trace:
        assert trace == {}
        assert worker.compute_ref_log_prob(batch) is batch
    assert worker.compute_ref_log_prob is original
    assert batch.meta_info == {"existing": 1}
    assert not caplog.records


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_head_export_subphases_preserve_collective_and_owner_copy(
    monkeypatch, caplog, rank, enabled
):
    from verl_speco.integration import rollout_publish as publish

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true" if enabled else "false")
    monkeypatch.setattr(publish, "_worker_bubble_enabled", lambda worker: True)
    monkeypatch.setattr(publish, "_is_veomni_actor_worker", lambda worker: False)
    calls = []

    class Head:
        shape = (6, 3)

        def to_local(self):
            return SimpleNamespace(device=SimpleNamespace(type="cuda"))

        def detach(self):
            return self

        def full_tensor(self):
            calls.append("gather")
            return self

        def to(self, **kwargs):
            calls.append("cpu_copy")
            return self

        def contiguous(self):
            calls.append("contiguous")
            return self

    head = Head()
    monkeypatch.setattr(
        publish, "_torch_module", lambda: SimpleNamespace(bfloat16="bf16")
    )
    monkeypatch.setattr(
        publish,
        "_head_only_export_candidate",
        lambda worker, rows: ("lm_head.weight", head, "head_only_ready"),
    )
    payload = publish.export_actor_lm_head_weight(
        SimpleNamespace(_is_actor=True, actor=object(), rank=rank),
        require_head_only=True,
    )
    assert calls == (["gather", "cpu_copy", "contiguous"] if rank == 0 else ["gather"])
    assert (payload is None) == (rank != 0)
    if payload is not None:
        assert payload["weight"] is head and payload["export_strategy"] == "direct_full"
    rows = [
        json.loads(r.getMessage().split(" ", 1)[1])
        for r in caplog.records
        if r.getMessage().startswith("[SpecoPhase]")
    ]
    assert [r["phase"] for r in rows] == (
        (
            [
                "lm_head_gather",
                "lm_head_cpu_copy",
                "lm_head_cpu_contiguous",
                "lm_head_payload_build",
            ]
            if rank == 0
            else ["lm_head_gather"]
        )
        if enabled
        else []
    )


def test_engine_preserves_order_results_mode_and_migration_flags(monkeypatch, caplog):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "1")
    calls = []

    @contextmanager
    def mode(**kwargs):
        calls.append(("enter", kwargs))
        yield "context-result"
        calls.append(("exit", kwargs))

    engine = SimpleNamespace(
        train_mode=mode,
        to=lambda *args, **kwargs: calls.append(("to", args, kwargs)),
        train_batch=lambda data, **kwargs: (data, kwargs),
    )
    install_engine_phase_timing(engine, role="actor", rank=2)
    original = engine.train_batch
    install_engine_phase_timing(engine, role="actor", rank=2)
    assert engine.train_batch is original
    with engine.train_mode(disable_auto_offload=False) as result:
        assert result == "context-result"
        engine.to("cpu", False, True, grad=False)
        assert engine.train_batch("data", loss_function="loss") == (
            "data",
            {"loss_function": "loss"},
        )
    assert [item[0] for item in calls] == ["enter", "to", "exit"]
    rows = [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records]
    migration = next(row for row in rows if row["phase"] == "engine_to")
    assert migration["device"] == "cpu"
    assert not migration["model"] and migration["optimizer"] and not migration["grad"]
    assert {r["phase"] for r in rows} == {
        "train_mode_enter",
        "train_mode_exit",
        "engine_to",
        "engine_train_batch",
    }


def test_mode_suppression_is_preserved(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")

    @contextmanager
    def mode():
        try:
            yield
        except ValueError:
            pass

    engine = SimpleNamespace(eval_mode=mode)
    install_engine_phase_timing(engine, role="ref", rank=0)
    with engine.eval_mode():
        raise ValueError("suppressed by original context")


@pytest.mark.parametrize("bubble", [False, True])
def test_collection_calls_unchanged_and_sync_emits_no_trace(
    monkeypatch, caplog, bubble
):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    calls = []

    def call(value):
        calls.append(value)
        return []

    executor = CallbackDrafterCollectionExecutor(
        set_step=call,
        stage_submit=call,
        commit_submit=call,
        abort_submit=call,
        rollback_submit=call,
        finalize_submit=call,
        resolve=lambda value: value,
        phase_timing=bubble,
    )
    payload = SimpleNamespace(collection_id="c4", buckets=[[{"sample": 1}]])
    executor.set_global_step(4)
    assert executor.stage(payload) == []
    assert executor.commit(payload) == []
    assert executor.finalize(payload) == []
    assert len(calls) == 4
    assert len(caplog.records) == (4 if bubble else 0)


def test_explicit_off_overrides_enabled_environment(monkeypatch, caplog):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    assert phase_timing_enabled()
    with phase_span("sync", enabled=False):
        pass
    assert not caplog.records


@pytest.mark.parametrize("bubble", [False, True])
def test_driver_observes_ref_without_polling_and_restores_hooks(monkeypatch, bubble):
    import time

    from test_bubble_review_regressions import _methods

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_online_fit_hooks"},
        time=time,
        MethodType=MethodType,
    )
    driver = cls()
    driver.global_steps = 8
    target = SimpleNamespace(generate_sequences=lambda: None)
    driver._speco_rollout_generation_target = lambda: target
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    driver.checkpoint_manager = SimpleNamespace(update_weights=lambda: None)
    old = driver._compute_old_log_prob = lambda batch: (batch, 0.5)
    ref_inputs = []

    def ref(batch):
        ref_inputs.append(dict(batch.meta_info))
        return batch

    driver._compute_ref_log_prob = ref
    driver._update_actor = lambda batch: batch
    driver._speco_oldlogprob_collection_requested = lambda: True
    driver._speco_oldlogprob_collection_enabled = lambda: True
    driver._speco_oldlogprob_entropy_hook_enabled = lambda: False
    driver._speco_plan_drafter_collection = lambda source: SimpleNamespace(
        collect=False,
        collect_interval_matched=False,
    )
    driver._speco_log_drafter_collection_plan = lambda plan: None
    driver._speco_wait_pending_drafter_publish = lambda: None
    driver._pending_target_lm_head_sync = {"target_version": 8, "stage": "fetch"}
    driver._speco_poll_target_lm_head_prefetch = lambda: pytest.fail("new polling")
    recorded = []
    driver._speco_record_rollout_idle_metrics = recorded.append
    # The old-logprob non-collection branch invokes this global in the hook.
    from verl_speco.trainer.scheduler import DrafterCollectionSource

    cls._speco_online_fit_hooks.__globals__["DrafterCollectionSource"] = (
        DrafterCollectionSource
    )
    token = SimpleNamespace(meta_info={"existing": 42})
    with pytest.raises(ValueError, match="exit"):
        with contextmanager(driver._speco_online_fit_hooks)():
            assert driver._compute_old_log_prob(token) == (token, 0.5)
            assert driver._compute_ref_log_prob(token) is token
            if not bubble:
                assert driver._compute_ref_log_prob is ref
            raise ValueError("exit")
    assert driver._compute_ref_log_prob is ref
    assert driver._compute_old_log_prob is old
    assert token.meta_info == {"existing": 42}
    assert ("_speco_ref_phase_trace" in ref_inputs[0]) is bubble
    if bubble:
        assert recorded[0]["bubble/old_log_prob_source_step"] == 8
        assert "timing_s/oldlogprob_compute" in recorded[0]
        assert recorded[1]["bubble/ref_head_pending_at_entry"] == 1
        assert recorded[1]["bubble/ref_head_target_version"] == 8
        assert "timing_s/phase_ref_driver" in recorded[1]
    else:
        assert not recorded


@pytest.mark.parametrize("bubble", [False, True])
def test_worker_init_instruments_only_bubble(monkeypatch, bubble):
    from verl_speco.integration import rollout_publish as publish

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    for name in (
        "install_rollout_runtime_for_worker",
        "install_oldlogprob_hidden_runtime_for_worker",
        "validate_oldlogprob_hidden_runtime_for_worker",
    ):
        monkeypatch.setattr(publish, name, lambda worker: None)

    class Base:
        def init_model(self):
            return "original-result"

    class Worker(publish.DraftWeightPublishMixin, Base):
        pass

    worker = Worker()
    worker.config = {
        "rollout": {
            "drafter": {
                "training": {
                    "scheduler": {
                        "execution": {
                            "strategy": "rollout_idle_worker" if bubble else "sync"
                        }
                    }
                }
            }
        }
    }
    engine = SimpleNamespace(train_batch=lambda: "unchanged")
    worker.actor = SimpleNamespace(engine=engine)
    original = engine.train_batch
    assert worker.init_model() == "original-result"
    assert (engine.train_batch is original) is (not bubble)
    assert engine.train_batch() == "unchanged"
