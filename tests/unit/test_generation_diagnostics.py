# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded host-only generation counters; concurrent sums are not wall time."""

import json
from types import SimpleNamespace

import pytest

from verl_speco.integration import generation_diagnostics as diagnostics


@pytest.fixture(autouse=True)
def isolate_diagnostic_logger():
    logger = diagnostics.logger
    saved = list(logger.handlers), logger.level, logger.disabled, logger.propagate
    yield
    for handler in list(logger.handlers):
        if handler not in saved[0]:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(saved[1])
    logger.disabled, logger.propagate = saved[2:]


def test_background_counters_are_nested_and_context_local(monkeypatch):
    values = iter([0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(diagnostics.time, "perf_counter", lambda: next(values))
    counts = {}
    with diagnostics.background_diagnostics(counts):
        with diagnostics.background_span("service"):
            with diagnostics.background_span("event_drain"):
                pass
    assert counts == {
        "event_drain_calls": 1,
        "event_drain_sum_s": 1.0,
        "event_drain_max_s": 1.0,
        "service_calls": 1,
        "service_sum_s": 4.0,
        "service_max_s": 4.0,
    }
    # Outside the event thread's context there must be no clocks or counters.
    with diagnostics.background_span("data_probe"):
        pass
    assert "data_probe_calls" not in counts


def test_engine_timing_reads_existing_metrics_without_filling_missing_values():
    metrics = SimpleNamespace(
        arrival_time=10.0,
        first_scheduled_time=12.0,
        first_token_time=13.0,
        last_token_time=20.0,
        finished_time=21.0,
        time_in_queue=1.5,
    )
    assert diagnostics.request_engine_timing(metrics) == {
        "queue_s": 1.5,
        "time_to_first_token_s": 3.0,
        "decode_span_s": 7.0,
        "request_span_s": 11.0,
    }
    assert diagnostics.request_engine_timing(None) == {}
    assert (
        diagnostics.request_engine_timing(
            SimpleNamespace(
                arrival_time=10.0,
                first_scheduled_time=9.0,
                first_token_time=float("nan"),
                last_token_time="unknown",
                time_in_queue=-1.0,
            )
        )
        == {}
    )


def test_recent_vllm_request_state_stats_never_mixes_clock_domains():
    metrics = SimpleNamespace(
        arrival_time=1792495000.0,
        queued_ts=100.0,
        scheduled_ts=102.0,
        first_token_ts=103.0,
        last_token_ts=120.0,
        first_token_latency=3.5,
    )
    assert diagnostics.request_engine_timing(metrics) == {
        "queue_s": 2.0,
        "prefill_span_s": 1.0,
        "decode_span_s": 17.0,
        "time_to_first_token_s": 3.5,
    }
    assert (
        diagnostics.request_engine_timing(
            SimpleNamespace(
                arrival_time=1792495000.0,
                queued_ts=0.0,
                scheduled_ts=0.0,
                first_token_ts=0.0,
                last_token_ts=0.0,
                first_token_latency=0.0,
            )
        )
        == {}
    )


@pytest.mark.parametrize(
    "output,state",
    [
        (SimpleNamespace(), "final_output_not_seen"),
        (SimpleNamespace(finished=False), "final_output_not_seen"),
        (SimpleNamespace(finished=True), "metrics_missing"),
        (
            SimpleNamespace(finished=True, metrics=SimpleNamespace()),
            "timestamps_missing_or_invalid",
        ),
        (
            SimpleNamespace(finished=True, metrics=SimpleNamespace(time_in_queue=1.0)),
            "available",
        ),
    ],
)
def test_engine_observation_explains_missing_timings(output, state):
    observation = {"outputs": 0, "final_outputs": 0}
    diagnostics.observe_engine_output(observation, output)
    assert observation["state"] == state
    assert observation["outputs"] == 1
    assert "timing" not in observation or state == "available"


def test_runtime_identity_reports_loaded_code_once(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    records = []
    monkeypatch.setattr(
        diagnostics,
        "emit_summary",
        lambda kind, **fields: records.append((kind, fields)),
    )
    owner = SimpleNamespace(engine=SimpleNamespace(log_stats=False))
    diagnostics.emit_runtime_identity(owner, "test", diagnostics.request_engine_timing)
    diagnostics.emit_runtime_identity(owner, "test", diagnostics.request_engine_timing)
    assert len(records) == 1
    assert records[0][0] == "runtime_identity"
    loaded = records[0][1]["loaded"][0]
    assert loaded["file"] == diagnostics.request_engine_timing.__code__.co_filename
    assert len(loaded["loaded_code_sha256"]) == 16


def test_bad_optional_engine_metrics_cannot_replace_request_result():
    class Metrics:
        @property
        def arrival_time(self):
            raise RuntimeError("unsupported metric accessor")

    observation = {"outputs": 0, "final_outputs": 0}
    diagnostics.observe_engine_output(
        observation, SimpleNamespace(finished=True, metrics=Metrics())
    )
    assert observation["state"] == "diagnostic_error"


def test_sidecar_io_diagnostics_are_cumulative_and_rate_limited(monkeypatch):
    clock, records = [0.0], []
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        diagnostics,
        "emit_summary",
        lambda kind, **fields: records.append((kind, fields)),
    )
    owner = SimpleNamespace()
    for _ in range(100):
        diagnostics.record_sidecar_io(owner, "/stats", 0.01, succeeded=True)
    assert len(records) == 1
    clock[0] = 31.0
    diagnostics.record_sidecar_io(owner, "/stats", 0.2, succeeded=False)
    assert len(records) == 2
    assert records[-1][1]["calls"] == 101
    assert records[-1][1]["sum_s"] == pytest.approx(1.2)
    assert records[-1][1]["max_s"] == 0.2
    assert records[-1][1]["failures"] == 1


def test_episode_counts_missing_engine_timings_without_claiming_zero(monkeypatch):
    records = []
    monkeypatch.setattr(
        diagnostics, "emit_summary", lambda kind, **fields: records.append(fields)
    )
    owner = SimpleNamespace()
    diagnostics.record_request(
        owner,
        {"total_s": 3.0, "engine_timing_status": {"state": "metrics_missing"}},
        failed=False,
    )
    diagnostics.flush_requests(owner)
    assert records[0]["engine_timing_status_counts"] == {"metrics_missing": 1}
    assert "engine_timing" not in records[0]["slowest"]


def _empty_idle_ablation_owner():
    return SimpleNamespace(
        global_steps=1,
        _drafter_scheduler=SimpleNamespace(_training_quota_debt_steps=0),
        _drafter_runtime_state=SimpleNamespace(status=SimpleNamespace(name="IDLE")),
    )


def _empty_idle_ablation_config():
    return {
        "collect_hidden_states_from_old_logprob": True,
        "collect_hidden_states_from_sgl": False,
        "collect_interval_steps": 4,
        "training_interval_steps": 4,
        "scheduler": {"training_quota": {"enable": True}},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("_training_quota_debt_steps", 20),
        ("_training_quota_data_version", 8),
        ("_training_quota_collection_step", 8),
        ("_idle_worker_writer_state_version", 4),
        ("_idle_optimizer_initialized_groups", {("0", "1")}),
        ("_latest_published_drafter_version", 4),
    ],
)
def test_empty_poll_ablation_rejects_any_quota_or_initialized_state(
    monkeypatch, field, value
):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", "true")
    owner = _empty_idle_ablation_owner()
    setattr(owner._drafter_scheduler, field, value)
    assert not diagnostics.empty_idle_poll_ablation_allowed(
        owner, _empty_idle_ablation_config
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("_pending_target_lm_head_sync", {"stage": "fetch"}),
        ("_pending_drafter_publish_refs", [object()]),
        ("_speco_rollout_training_lease", object()),
        ("_speco_idle_completed_publications", [object()]),
        ("_speco_last_collected_samples", 32),
        ("_speco_last_published_drafter_step", 4),
        ("global_steps", 5),
    ],
)
def test_empty_poll_ablation_rejects_pending_work_or_later_cycle(
    monkeypatch, field, value
):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", "true")
    owner = _empty_idle_ablation_owner()
    setattr(owner, field, value)
    assert not diagnostics.empty_idle_poll_ablation_allowed(
        owner, _empty_idle_ablation_config
    )


@pytest.mark.parametrize(
    "guard",
    [
        "off",
        "phase_off",
        "sglang",
        "no_oldlogprob",
        "quota_off",
        "running",
        "active",
        "future",
    ],
)
def test_empty_idle_poll_ablation_is_fail_closed(monkeypatch, guard):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", "true")
    owner, config = _empty_idle_ablation_owner(), _empty_idle_ablation_config()
    if guard == "off":
        monkeypatch.delenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL")
    elif guard == "phase_off":
        monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "false")
    elif guard == "sglang":
        config["collect_hidden_states_from_sgl"] = True
    elif guard == "no_oldlogprob":
        config["collect_hidden_states_from_old_logprob"] = False
    elif guard == "quota_off":
        config["scheduler"]["training_quota"]["enable"] = False
    elif guard == "running":
        owner._drafter_runtime_state.status.name = "RUNNING"
    else:
        setattr(
            owner._drafter_runtime_state,
            "active_plan" if guard == "active" else "training_ref",
            object(),
        )
    assert not diagnostics.empty_idle_poll_ablation_allowed(owner, lambda: config)


def test_empty_idle_poll_ablation_allows_only_explicit_initial_control(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", "true")
    owner = _empty_idle_ablation_owner()
    assert diagnostics.empty_idle_poll_ablation_allowed(
        owner, _empty_idle_ablation_config
    )


def test_empty_poll_ablation_preserves_post_rollout_drain(monkeypatch):
    from contextlib import nullcontext
    from test_bubble_review_regressions import _methods

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_PAUSE_EMPTY_IDLE_POLL", "true")
    monkeypatch.setattr(diagnostics, "emit_runtime_identity", lambda *args: None)
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_start_rollout_idle_event_loop",
            "_speco_service_rollout_idle_events",
        },
    )
    owner = cls()
    owner.__dict__.update(_empty_idle_ablation_owner().__dict__)
    owner._speco_rollout_idle_worker_enabled = lambda: True
    owner._speco_drafter_schedule_config = lambda: SimpleNamespace(
        idle_worker_historical_window_readmission=True
    )
    owner._speco_rollout_idle_event_bus_name = lambda: "diagnostic-bus"
    owner._speco_rollout_idle_poll_interval_sec = lambda: 0.05
    owner._speco_drafter_training_config = _empty_idle_ablation_config
    assert owner._speco_start_rollout_idle_event_loop() == (None, None)
    assert owner._speco_idle_poll_diagnostics == {"diagnostic_empty_poll_paused": 1}
    calls = []
    owner._speco_bubble_training_lock = nullcontext
    owner._speco_drain_rollout_idle_events = lambda: calls.append("drain") or {
        "events": 2
    }
    owner._speco_release_completed_rollout_training_lease = (
        lambda: calls.append("completion") or {}
    )
    owner._speco_record_rollout_idle_metrics = lambda metrics: None
    result = owner._speco_service_rollout_idle_events(allow_launch=False)
    assert calls == ["drain", "completion"]
    assert result["events"] == 2
    assert result["bubble/post_generation_launch_suppressed"] == 1


def test_summary_retains_only_slowest_request_and_flushes_once(monkeypatch):
    records = []
    monkeypatch.setattr(
        diagnostics,
        "emit_summary",
        lambda kind, **fields: records.append((kind, fields)),
    )
    owner = SimpleNamespace(replica_rank=2)
    for index in range(100):
        diagnostics.record_request(
            owner,
            {
                "source_step_hint": 4,
                "request_id": str(index),
                "total_s": float(index),
                "generate_s": float(index),
                "admission_s": 0.1,
            },
            failed=index == 0,
        )
    assert not records
    diagnostics.flush_requests(owner)
    diagnostics.flush_requests(owner)
    assert len(records) == 1
    kind, summary = records[0]
    assert kind == "replica_episode"
    assert summary["requests"] == 100 and summary["failed"] == 1
    assert summary["slowest"]["request_id"] == "99"
    assert summary["generate_max_s"] == 99.0
    assert summary["source_step_min"] == summary["source_step_max"] == 4
    assert owner._speco_generation_request_summary is None
    assert len(json.dumps(summary)) < 2000


def test_metadata_never_materializes_prompt_tensors():
    class TensorLike:
        def __len__(self):
            raise AssertionError("must not inspect a device tensor")

    metadata = diagnostics.request_metadata(
        (TensorLike(), {"_verl_global_steps": 3}, "id"), {}
    )
    assert metadata == {
        "source_step_hint": 3,
        "request_id": "id",
        "prompt_tokens": None,
    }


def test_background_probe_failure_is_counted_and_reraised():
    counts = {}
    with pytest.raises(ValueError, match="original"):
        with (
            diagnostics.background_diagnostics(counts),
            diagnostics.background_span("data_probe"),
        ):
            raise ValueError("original")
    assert counts["data_probe_calls"] == 1
    assert counts["data_probe_sum_s"] >= 0


def test_summary_prints_with_disabled_training_loggers(monkeypatch, capsys):
    from verl_speco.integration.phase_timing import logger

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    saved = list(logger.handlers), logger.level, logger.disabled, logger.propagate
    try:
        logger.disabled = True
        diagnostics.emit_summary("driver_rollout", source_step=3, background={})
        output = capsys.readouterr().err
        assert "[SpecoGeneration]" in output and '"source_step": 3' in output
    finally:
        for handler in list(logger.handlers):
            if handler not in saved[0]:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(saved[1])
        logger.disabled, logger.propagate = saved[2:]


@pytest.mark.parametrize("enabled", [False, True])
def test_event_loop_collects_only_when_diagnostics_enabled(monkeypatch, enabled):
    import ast
    import logging
    from pathlib import Path

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", str(enabled).lower())
    tree = ast.parse(
        (
            Path(__file__).parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
        ).read_text(encoding="utf-8")
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpecoRayPPOTrainer"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_speco_start_rollout_idle_event_loop"
    )

    class Event:
        polls = 0

        def wait(self, interval):
            self.polls += 1
            return self.polls > 2

    class Thread:
        def __init__(self, *, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    calls = []

    def service():
        calls.append(1)
        with diagnostics.background_span("data_probe"):
            pass

    namespace = {
        "threading": SimpleNamespace(Event=Event, Thread=Thread),
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<driver>", "exec"),
        namespace,
    )
    owner = SimpleNamespace(
        _speco_rollout_idle_worker_enabled=lambda: True,
        _speco_drafter_schedule_config=lambda: SimpleNamespace(
            idle_worker_historical_window_readmission=True
        ),
        _speco_rollout_idle_event_bus_name=lambda: "bus",
        _speco_rollout_idle_poll_interval_sec=lambda: 0.05,
        _speco_service_rollout_idle_events=service,
    )
    namespace[method.name](owner)
    assert len(calls) == 2  # Diagnostic mode must not add polls or probes.
    if enabled:
        assert owner._speco_idle_poll_diagnostics["service_calls"] == 2
        assert owner._speco_idle_poll_diagnostics["data_probe_calls"] == 2
    else:
        assert owner._speco_idle_poll_diagnostics is None


def test_idle_snapshot_reads_all_replicas_without_electing_writer():
    from collections import deque

    groups = {index: (str(index * 2), str(index * 2 + 1)) for index in range(6)}
    scheduler = SimpleNamespace(
        _replica_idle_worker_groups=groups,
        _replica_last_observed_idle_window_sec={index: index + 0.5 for index in groups},
        _replica_idle_window_samples_sec={index: deque(range(8)) for index in groups},
        _idle_worker_writer_group=groups[2],
        _idle_workers={},
        _training_quota_debt_steps=7,
    )
    owner = SimpleNamespace(
        _drafter_scheduler=scheduler,
        _speco_get_drafter_scheduler=lambda: pytest.fail("must not initialize/select"),
        _ray_get_if_needed=lambda value: pytest.fail("must not resolve refs"),
    )
    snapshot = diagnostics.idle_snapshot(owner)
    assert len(snapshot["replicas"]) == 6
    assert [
        replica["replica_rank"] for replica in snapshot["replicas"] if replica["writer"]
    ] == [2]
    assert snapshot["replicas"][5]["last_observed_tail_s"] == 5.5
    assert snapshot["replicas"][5]["current_generation_group_tail_s"] is None
    assert snapshot["replicas"][0]["recent_tail_s"] == [4, 5, 6, 7]
    assert snapshot["control"]["quota_debt_steps"] == 7
    assert len(scheduler._replica_idle_window_samples_sec[0]) == 8
    assert scheduler._idle_worker_writer_group == groups[2]


def test_admission_and_gate_counts_are_not_optimizer_work():
    counts = {}
    plan = SimpleNamespace(
        reason="window_too_small",
        launch=False,
        max_batches=0,
        target_worker_ids=("2", "3"),
        idle_usable_window_sec=0.1,
    )
    with diagnostics.background_diagnostics(counts):
        for _ in range(3):
            diagnostics.record_admission(plan)
        diagnostics.record_gate_metrics(
            {
                "bubble/training_blocked_by_publish_backpressure": 1,
                "bubble/idle_optimizer_steps": 5,
                "scheduler/pending_training_count": 0,
            }
        )
    assert counts["admission_reasons"] == {"window_too_small": 3}
    assert counts["last_admission"]["idle_usable_window_sec"] == 0.1
    assert counts["gate_counts"] == {
        "bubble/training_blocked_by_publish_backpressure": 1
    }
    assert not plan.launch and plan.max_batches == 0


def test_allocator_snapshot_never_synchronizes_or_scans_tensors():
    device = SimpleNamespace(
        memory_allocated=lambda: 100,
        memory_reserved=lambda: 300,
        synchronize=lambda: pytest.fail("unexpected device sync"),
        empty_cache=lambda: pytest.fail("unexpected cleanup"),
    )
    assert diagnostics.allocator_snapshot(device) == {
        "memory_allocated_bytes": 100,
        "memory_reserved_bytes": 300,
    }
    assert diagnostics.allocator_snapshot(None) == {}


@pytest.mark.parametrize("bubble", [False, True])
def test_pipeline_boundary_preserves_exception_and_order(monkeypatch, bubble):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    records, calls = [], []
    monkeypatch.setattr(
        diagnostics,
        "emit_summary",
        lambda kind, **fields: records.append((kind, fields)),
    )
    owner = SimpleNamespace(
        global_steps=4, _speco_rollout_idle_worker_enabled=lambda: bubble
    )
    with pytest.raises(ValueError, match="original failure"):
        with diagnostics.pipeline_boundary(owner, "checkpoint") as timings:
            with diagnostics.boundary_stage(timings, "drafter_save_wait"):
                calls.append("save")
                raise ValueError("original failure")
    assert calls == ["save"]
    if bubble:
        assert len(records) == 1 and records[0][1]["failed"]
        assert "drafter_save_wait_s" in records[0][1]["stages"]
    else:
        assert timings is None and not records


def test_unserializable_diagnostics_do_not_fail_training(monkeypatch, capsys):
    from verl_speco.integration.phase_timing import logger

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    saved = list(logger.handlers), logger.level, logger.disabled, logger.propagate
    try:
        diagnostics.emit_summary("test", invalid=object())
        assert "summary_serialization_failed" in capsys.readouterr().err
    finally:
        for handler in list(logger.handlers):
            if handler not in saved[0]:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(saved[1])
        logger.disabled, logger.propagate = saved[2:]


@pytest.mark.parametrize("failure_stage", ["configure", "write", "serialization_warning"])
@pytest.mark.parametrize("training_fails", [False, True])
def test_diagnostic_output_failure_preserves_pipeline_result(
    monkeypatch, failure_stage, training_fails
):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")

    def fail(*args, **kwargs):
        raise OSError("diagnostic output unavailable")

    monkeypatch.setattr(diagnostics, "configure_phase_logger", lambda: None)
    if failure_stage == "configure":
        monkeypatch.setattr(diagnostics, "configure_phase_logger", fail)
    else:
        monkeypatch.setattr(diagnostics.logger, "warning", fail)
    owner = SimpleNamespace(
        global_steps=4, _speco_rollout_idle_worker_enabled=lambda: True
    )
    original_error = RuntimeError("original checkpoint failure")

    def run():
        with diagnostics.pipeline_boundary(owner, "checkpoint") as timings:
            if failure_stage == "serialization_warning":
                timings["unsupported"] = object()
            if training_fails:
                raise original_error
            return "checkpoint saved"

    if training_fails:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is original_error
    else:
        assert run() == "checkpoint saved"
