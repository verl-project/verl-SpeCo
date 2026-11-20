# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Diagnostic output survives disabled loggers without altering actor dispatch."""

import pytest

from verl_speco.integration import vllm_rollout_capture as capture
from verl_speco.trainer.scheduler import collection_executor as executor


def test_capture_trace_bypasses_logger(monkeypatch, capsys):
    monkeypatch.setenv("SPECO_CAPTURE_DIAGNOSTICS", "1")
    monkeypatch.setattr(capture.logger, "disabled", True)
    capture.capture_trace("[vLLM capture timing] read_s=%.6f", 0.25)
    assert "read_s=0.250000" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["vllm", "oldlogprob"])
def test_dispatch_trace_preserves_callback_order(monkeypatch, capsys, source):
    monkeypatch.setenv("SPECO_CAPTURE_DIAGNOSTICS", "1")
    calls = []

    def submit(value):
        calls.append(("submit", value))
        return "ref"

    def resolve(value):
        calls.append(("resolve", value))
        return "result"

    obj = executor.CallbackDrafterCollectionExecutor(
        submit,
        submit,
        submit,
        submit,
        submit,
        submit,
        resolve,
    )
    assert (
        obj._call("stage", submit, "payload", collection_id=f"collection-2-{source}")
        == "result"
    )
    assert calls == [("submit", "payload"), ("resolve", "ref")]
    output = capsys.readouterr().out
    assert ("[vLLM collection dispatch]" in output) == (source == "vllm")


def test_disabled_dispatch_diagnostics_keep_fast_path(monkeypatch, capsys):
    monkeypatch.setenv("SPECO_CAPTURE_DIAGNOSTICS", "0")
    monkeypatch.setattr(executor, "phase_timing_enabled", lambda: False)

    def unexpected_span(*args, **kwargs):
        raise AssertionError("Disabled diagnostics must not enter timing span")

    monkeypatch.setattr(executor, "phase_span", unexpected_span)
    identity = lambda x: x
    obj = executor.CallbackDrafterCollectionExecutor(*([identity] * 7))
    assert (
        obj._call("stage", identity, "payload", collection_id="collection-2-vllm")
        == "payload"
    )
    assert capsys.readouterr().out == ""
