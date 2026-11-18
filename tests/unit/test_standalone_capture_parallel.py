# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU spawn/protocol checks; no model, vLLM runtime or NPU allocation."""

import json
import multiprocessing as mp
import os
import queue
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.integration import standalone_capture_parallel as parallel


class Engine:
    def __init__(self, rank):
        self.rank = rank
        self.remaining = 0
        self.rpc_count = 0

    def collective_rpc(self, *args, **kwargs):
        self.rpc_count += 1
        return [{"owner_dp": self.rank}]

    def has_unfinished_requests(self):
        return bool(self.remaining)

    def add_request(self, request_id, prompt, params):
        self.prompt = prompt["prompt_token_ids"]
        self.remaining = self.rank + 1

    def step(self):
        if not self.remaining:
            return []
        self.remaining -= 1
        return [
            SimpleNamespace(
                finished=not self.remaining,
                prompt_token_ids=self.prompt,
                outputs=[SimpleNamespace(token_ids=[42])],
            )
        ]

    def abort_request(self, ids):
        self.remaining = 0


def fake_frontend(rank, size, port, options, commands, results):
    os.environ.update(parallel._dp_environment(rank, size, port))
    request = -1
    trace = []
    engine = Engine(rank)
    try:
        if rank == 1 and options.get("fail_start"):
            raise RuntimeError("synthetic startup failure")
        if not options.get("never_ready"):
            results.put(("ready", rank, request, None))
        while True:
            request, method, args, kwargs = commands.get()
            trace.append(method)
            if method == "shutdown":
                break
            if rank == 1 and method == "step" and options.get("fail_step"):
                raise RuntimeError("synthetic step failure")
            value = parallel._execute(engine, rank, method, args, kwargs)
            results.put(("result", rank, request, value))
    except RuntimeError:
        results.put(("error", rank, request, traceback.format_exc()))
    finally:
        Path(options["model"], f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "methods": trace,
                    "rpc_count": engine.rpc_count,
                    "dp_rank": os.environ["VLLM_DP_RANK"],
                    "dp_size": os.environ["VLLM_DP_SIZE"],
                    "visible": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                }
            )
        )


def options(tmp_path, **kwargs):
    return {
        "model": str(tmp_path),
        "tensor_parallel_size": 8,
        "max_model_len": 4096,
        "enable_expert_parallel": True,
        **kwargs,
    }


@pytest.fixture
def spawn(monkeypatch):
    monkeypatch.setattr(parallel, "_dp_frontend", fake_frontend)
    monkeypatch.setattr(parallel, "_free_port", lambda: 12345)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ",".join(map(str, range(16))))


def test_two_frontends_progress_together_and_only_dp0_returns_features(tmp_path, spawn):
    llm = parallel.ParallelCaptureLLM(
        data_parallel_size=2, options=options(tmp_path), timeout=10
    )
    try:
        assert llm.collective_rpc(
            "speco_standalone_capture", kwargs={"operation": "begin"}
        ) == [{"owner_dp": 0}]
        llm.add_request("r", {"prompt_token_ids": [1, 2, 3]}, {})
        outputs = []
        while llm.has_unfinished_requests():
            outputs.extend(output for output in llm.step() if output.finished)
        assert len(outputs) == 1
        assert outputs[0].prompt_token_ids == [1, 2, 3]
        assert outputs[0].outputs[0].token_ids == [42]
    finally:
        llm.shutdown_capture_frontends()
    llm.shutdown_capture_frontends()
    for rank in range(2):
        trace = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert trace["methods"].count("step") == 2
        assert trace["rpc_count"] == int(rank == 0)
        assert trace["dp_rank"] == str(rank) and trace["dp_size"] == "2"
        assert len(trace["visible"].split(",")) == 16
    assert all(not process.is_alive() for process in llm._processes)


def test_frontend_runtime_failure_propagates_and_owned_processes_close(tmp_path, spawn):
    llm = parallel.ParallelCaptureLLM(
        data_parallel_size=2, options=options(tmp_path, fail_step=True), timeout=10
    )
    try:
        llm.add_request("r", {"prompt_token_ids": [1, 2, 3]}, {})
        with pytest.raises(RuntimeError, match="synthetic step failure"):
            llm.step()
    finally:
        llm.shutdown_capture_frontends()
    assert all(not process.is_alive() for process in llm._processes)


def test_startup_failure_cleans_up_both_frontends(tmp_path, spawn):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(RuntimeError, match="synthetic startup failure"):
        parallel.ParallelCaptureLLM(
            data_parallel_size=2, options=options(tmp_path, fail_start=True), timeout=10
        )
    assert {p.pid for p in mp.active_children()} <= before


def test_rejects_dp_without_ep_or_insufficient_devices(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="expert parallelism"):
        parallel.ParallelCaptureLLM(
            data_parallel_size=2,
            options=options(tmp_path, enable_expert_parallel=False),
        )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(ValueError, match="16 visible"):
        parallel.ParallelCaptureLLM(data_parallel_size=2, options=options(tmp_path))


def test_real_frontend_bootstrap_uses_env_dp_and_single_process_llm_argument(
    monkeypatch,
):
    from verl_speco import collect_deepseek_v4_features as capture

    for name, value in parallel._dp_environment(1, 2, 12345).items():
        monkeypatch.setenv(name, value)
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        config = SimpleNamespace(
            data_parallel_size=2, data_parallel_rank=1, enable_expert_parallel=True
        )
        return SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=config)
            )
        )

    monkeypatch.setattr(capture, "build_capture_llm", build)
    commands, results = queue.Queue(), queue.Queue()
    commands.put((-1, "shutdown", (), {}))
    parallel._dp_frontend(
        1, 2, 12345, {"enable_expert_parallel": True}, commands, results
    )
    assert calls == [{"enable_expert_parallel": True, "data_parallel_size": 1}]
    assert results.get_nowait() == ("ready", 1, -1, None)
    assert os.environ["VLLM_DP_SIZE"] == "2"


def test_builder_selects_parallel_frontends_without_constructing_parent_llm(
    monkeypatch,
):
    from verl_speco import collect_deepseek_v4_features as capture

    versions = {"vllm": "0.26.0+empty", "vllm-ascend": "0.26.0rc1"}
    monkeypatch.setattr(capture.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(parallel, "ParallelCaptureLLM", lambda **kwargs: kwargs)
    result = capture.build_capture_llm(
        "/model",
        SimpleNamespace(block_size=5),
        tensor_parallel_size=8,
        data_parallel_size=2,
        enable_expert_parallel=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
    )
    assert result["data_parallel_size"] == 2
    assert result["options"]["enable_expert_parallel"] is True
    assert result["options"]["tensor_parallel_size"] == 8


def test_draft_update_rpc_reaches_every_dp_frontend(tmp_path, spawn):
    llm = parallel.ParallelCaptureLLM(
        data_parallel_size=2, options=options(tmp_path), timeout=10
    )
    try:
        result = llm.draft_update_rpc(
            "speco_standalone_draft_update", kwargs={"operation": "prepare"}
        )
        assert result == [{"owner_dp": 0}, {"owner_dp": 1}]
    finally:
        llm.shutdown_capture_frontends()
