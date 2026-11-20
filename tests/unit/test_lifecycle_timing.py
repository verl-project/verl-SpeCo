# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Lifecycle diagnostics preserve return values, failures and coroutine behavior."""

import asyncio
import inspect
import json
import logging

import pytest

from verl_speco.integration import lifecycle_timing as tracing


def events(caplog):
    return [
        json.loads(r.message.split("[SpecoLifecycle] ", 1)[1])
        for r in caplog.records
        if "[SpecoLifecycle] " in r.message
    ]


@pytest.fixture
def diagnostic(monkeypatch, caplog):
    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setattr(tracing, "configure_phase_logger", lambda: None)
    monkeypatch.setattr(tracing, "logger", logging.getLogger("lifecycle_test"))
    caplog.set_level(logging.WARNING, logger="lifecycle_test")


def test_disabled_does_not_wrap_or_create_state(monkeypatch):
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)

    class Server:
        def sleep(self, level=2):
            return level

    original = Server.sleep
    tracing.install_server_lifecycle_timing(Server)
    assert Server.sleep is original
    server = Server()
    with tracing.lifecycle_span(server, "sleep"):
        assert server.sleep() == 2
    assert not server.__dict__


def test_pause_resume_tracks_interval_and_wraps_only_once(
    diagnostic, caplog, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(tracing.time, "perf_counter", lambda: clock[0])

    class Server:
        replica_rank = 1

        async def abort_all_requests(self, reset_prefix_cache=True):
            clock[0] += 2
            return reset_prefix_cache

        async def resume_generation(self):
            clock[0] += 1
            return "resumed"

    signature = inspect.signature(Server.abort_all_requests)
    tracing.install_server_lifecycle_timing(Server)
    wrapped = Server.abort_all_requests
    tracing.install_server_lifecycle_timing(Server)
    assert Server.abort_all_requests is wrapped
    assert inspect.signature(wrapped) == signature
    assert inspect.iscoroutinefunction(wrapped)
    server = Server()
    assert asyncio.run(server.abort_all_requests(False)) is False
    clock[0] += 10
    assert asyncio.run(server.abort_all_requests()) is True
    assert asyncio.run(server.resume_generation()) == "resumed"
    records = events(caplog)
    assert [r["event"] for r in records] == ["begin", "end"] * 3
    assert records[-1]["pause_to_resume_s"] == 13
    assert records[3]["first_call"] is False


def test_failed_resume_keeps_pause_and_original_error(diagnostic, caplog):
    error = RuntimeError("original")

    class Server:
        def pause_generation(self):
            return 1

        def resume_generation(self):
            raise error

    tracing.install_server_lifecycle_timing(Server)
    server = Server()
    assert server.pause_generation() == 1
    with pytest.raises(RuntimeError) as caught:
        server.resume_generation()
    assert caught.value is error
    assert "paused_at" in server._speco_lifecycle_trace
    assert events(caplog)[-1]["failed"] is True


def test_sleep_wake_and_inherited_methods(diagnostic, caplog):
    class Base:
        def sleep(self, level=2):
            return level

        async def wake_up(self):
            return "awake"

    class Child(Base):
        pass

    original = Base.sleep
    tracing.install_server_lifecycle_timing(Child)
    server = Child()
    assert server.sleep(1) == 1
    assert asyncio.run(server.wake_up()) == "awake"
    assert Base.sleep is original
    assert events(caplog)[-1]["sleep_to_wake_s"] >= 0


def test_runtime_disable_retains_original_calls_without_logs(
    diagnostic, caplog, monkeypatch
):
    class Server:
        def clear_kv_cache(self):
            return "same"

    tracing.install_server_lifecycle_timing(Server)
    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING")
    server = Server()
    assert server.clear_kv_cache() == "same"
    assert events(caplog) == []
    assert not server.__dict__


def test_initial_load_stays_single_flight_and_retryable(diagnostic):
    from verl_speco.integration.vllm_runtime import _SpecoVLLMHttpServerMixin

    class Server:
        _speco_initial_draft_weights_required = True
        _speco_initial_draft_weights_ready = False
        _speco_initial_draft_weights_lock = None
        calls = 0
        fail = True

        async def collective_rpc(self, name):
            assert name == "speco_ensure_draft_initialized"
            self.calls += 1
            await asyncio.sleep(0)
            if self.fail:
                raise RuntimeError("load failed")

    async def run():
        server = Server()
        load = _SpecoVLLMHttpServerMixin._speco_ensure_initial_draft_weights
        with pytest.raises(RuntimeError, match="load failed"):
            await load(server)
        assert server._speco_initial_draft_weights_ready is False
        server.fail = False
        await asyncio.gather(load(server), load(server), load(server))
        assert server.calls == 2
        assert server._speco_initial_draft_weights_ready is True
        await load(server)
        assert server.calls == 2

    asyncio.run(run())


def test_server_lifecycle_survives_fresh_process_import(monkeypatch, tmp_path):
    import importlib
    import os
    from pathlib import Path
    import subprocess
    import sys

    cloudpickle = pytest.importorskip("cloudpickle")
    from verl_speco.integration.vllm_runtime import _build_speco_vllm_http_server_class

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "1")
    module_name = "lifecycle_fresh_base"
    (tmp_path / (module_name + ".py")).write_text(
        "class Server:\n"
        "    _speco_vllm_http_methods_patched = True\n"
        "    async def sleep(self): return 'slept'\n"
        "    async def wait_for_requests_to_drain(self): return 'drained'\n"
        "    def trace(self): return self.__dict__.get('_speco_lifecycle_trace', {})\n"
        "vLLMHttpServer = Server\n", encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(module_name)
    try:
        server_cls = _build_speco_vllm_http_server_class(module)
        blob = tmp_path / "server.pkl"
        blob.write_bytes(cloudpickle.dumps(server_cls))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")))
        code = "import asyncio,cloudpickle,sys; C=cloudpickle.load(open(sys.argv[1],'rb')); s=C(); assert asyncio.run(s.sleep())=='slept'; assert asyncio.run(s.wait_for_requests_to_drain())=='drained'; assert s.trace().get('calls')=={'sleep':1,'wait_for_requests_to_drain':1},s.trace(); print('FRESH_PROCESS_TRACE_PASS')"
        completed = subprocess.run([sys.executable, "-c", code, str(blob)], env=env, capture_output=True, text=True, timeout=45)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "FRESH_PROCESS_TRACE_PASS" in completed.stdout
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("fail", [False, True])
def test_trainer_sample_end_scope_preserves_sleep_result(monkeypatch, caplog, fail):
    import ast
    from pathlib import Path
    from verl_speco.integration import phase_timing

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "1")
    monkeypatch.setattr(phase_timing, "configure_phase_logger", lambda: None)
    monkeypatch.setattr(phase_timing, "logger", logging.getLogger("sample_end_test"))
    caplog.set_level(logging.WARNING, logger="sample_end_test")
    error = RuntimeError("sleep failed")

    class Parent:
        calls = 0
        def on_sample_end(self):
            self.calls += 1
            if fail:
                raise error
            return 42

    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/v1/speco_mixin.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpecoV1Mixin")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "on_sample_end")
    derived = ast.ClassDef(name="Subject", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
    namespace = {"Parent": Parent, "phase_span": phase_timing.phase_span}
    exec(compile(module, str(path), "exec"), namespace)
    owner = namespace["Subject"]()
    owner._speco_rollout_idle_worker_enabled = lambda: False
    owner._speco_colocate_probe_boundary_enabled = lambda: False
    if fail:
        with pytest.raises(RuntimeError) as caught:
            owner.on_sample_end()
        assert caught.value is error
    else:
        assert owner.on_sample_end() == 42
    assert owner.calls == 1
    records = [json.loads(r.message.split("[SpecoPhase] ", 1)[1]) for r in caplog.records if "[SpecoPhase] " in r.message]
    assert len(records) == 1 and records[0]["phase"] == "v1_sample_end"
    assert records[0]["failed"] == fail
