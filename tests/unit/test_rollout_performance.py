# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded diagnostics, no implicit CUDA initialization and exact concurrency accounting."""

from types import SimpleNamespace

import pytest

from verl_speco.integration import rollout_performance as perf
from verl_speco.integration import generation_diagnostics as diagnostics


def test_disabled_diagnostics_do_not_inspect_cuda_or_owner(monkeypatch):
    monkeypatch.delenv("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF", raising=False)
    owner = SimpleNamespace()
    assert perf.runtime_snapshot(owner, "test", "before") == {}
    with perf.generation_execution(owner):
        pass
    assert vars(owner) == {}


def test_snapshot_never_initializes_cuda_or_reads_tensors(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF", "true")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: pytest.fail("must not initialize CUDA")
    )
    monkeypatch.setattr(diagnostics, "emit_summary", lambda *a, **k: None)
    owner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks=100, block_size=16),
            scheduler_config=SimpleNamespace(max_num_seqs=256),
        ),
        model_runner=SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks=99)),
    )
    result = perf.runtime_snapshot(owner, "vllm_worker", "first")
    assert not result["cuda_initialized"]
    assert result["cache_config"]["num_gpu_blocks"] == 100
    assert result["kv_cache_config"]["num_blocks"] == 99
    assert result["scheduler_config"]["max_num_seqs"] == 256


def test_nested_requests_account_concurrency_without_summing_wall_times(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF", "true")
    clock = [0.0]
    monkeypatch.setattr(perf.time, "perf_counter", lambda: clock[0])
    logs = []
    monkeypatch.setattr(
        diagnostics, "emit_summary", lambda kind, **fields: logs.append(fields)
    )
    owner = SimpleNamespace(replica_rank=0)
    with perf.generation_execution(owner):
        clock[0] = 1.0
        with perf.generation_execution(owner):
            clock[0] = 3.0
        clock[0] = 4.0
    result = logs[0]
    assert result["frontend_peak_active"] == 2
    assert result["frontend_mean_active"] == 1.5
    assert result["execution_wall_s"] == 4.0
    assert result["execution_sum_s"] == 6.0
    assert result["submit_span_s"] == 1.0
    assert result["engine_scheduler_counts"] == "unavailable_in_frontend"
    assert owner._speco_perf_execution is None


def test_failed_requests_preserve_exception_and_logging_is_bounded(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF", "true")
    logs = []
    monkeypatch.setattr(
        diagnostics, "emit_summary", lambda kind, **fields: logs.append(fields)
    )
    owner = SimpleNamespace()
    for _ in range(5):
        with pytest.raises(ValueError, match="original"):
            with perf.generation_execution(owner):
                raise ValueError("original")
    assert len(logs) == 3
    assert all(row["failed"] == 1 for row in logs)


def test_snapshot_error_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF", "true")
    monkeypatch.setattr(
        perf,
        "runtime_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    logs = []
    monkeypatch.setattr(
        diagnostics, "emit_summary", lambda kind, **fields: logs.append(kind)
    )
    assert perf.safe_runtime_snapshot(None, "test", "before") == {
        "unavailable": ["RuntimeError"]
    }
    assert logs == ["rollout_perf_snapshot_failed"]


def test_request_summary_uses_the_runtime_verify_rounds_field(monkeypatch):
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_ROLLOUT_PERF", "true")
    owner = SimpleNamespace()
    diagnostics.record_request(
        owner, {"accepted_tokens": 7, "verify_rounds": 3, "total_s": 1.0}, failed=False
    )
    summary = owner._speco_generation_request_summary
    assert summary["request_accepted_tokens_sum"] == 7
    assert summary["request_verify_rounds_sum"] == 3


def test_host_diagnostic_helpers_do_not_import_torch_or_sync_devices(monkeypatch):
    import builtins
    import sys

    from verl_speco.trainer.performance_diagnostics import (
        compilation_metadata,
        training_stage_clock,
    )

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "torch", "host diagnostics must not initialize torch"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    trainer = SimpleNamespace(
        device_module=SimpleNamespace(synchronize=lambda: pytest.fail("device fence"))
    )
    assert isinstance(training_stage_clock(trainer), float)
    assert compilation_metadata()["torch_version"] == "unavailable"
