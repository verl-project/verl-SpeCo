# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Single-node external DP frontends for offline target feature acquisition.

Uses the VLLM_DP_* convention of AIR's offline capture launcher. Every DP
frontend executes the request so EP collectives keep progressing; only DP0
captures and returns features. The existing collector and store remain owners
of feature validation and persistence.
"""

from __future__ import annotations

import atexit
import gc
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
from types import SimpleNamespace


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _dp_environment(rank, size, port):
    return {
        "VLLM_DP_RANK": str(rank),
        "VLLM_DP_RANK_LOCAL": str(rank),
        "VLLM_DP_SIZE": str(size),
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": str(port),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def _execute(engine, rank, method, args, kwargs):
    if method == "draft_update_rpc":
        return engine.collective_rpc(*args, **kwargs)
    if method == "collective_rpc":
        if rank != 0:
            return None
        return engine.collective_rpc(*args, **kwargs)
    result = getattr(engine, method)(*args, **kwargs)
    if method == "step":
        # Do not send vLLM's internal output objects between frontends.
        return (
            [
                SimpleNamespace(
                    finished=output.finished,
                    prompt_token_ids=output.prompt_token_ids,
                    outputs=[
                        SimpleNamespace(token_ids=item.token_ids)
                        for item in output.outputs
                    ],
                )
                for output in result
            ]
            if rank == 0
            else None
        )
    return result if method == "has_unfinished_requests" else None


def _dp_frontend(rank, size, port, options, commands, results):
    os.environ.update(_dp_environment(rank, size, port))
    llm = None
    request = -1
    try:
        from verl_speco.collect_deepseek_v4_features import build_capture_llm

        # LLM rejects an explicit DP>1 argument. ParallelConfig falls back to
        # VLLM_DP_* when this argument is 1, matching the original AIR launcher.
        llm = build_capture_llm(**options, data_parallel_size=1)
        parallel = llm.llm_engine.vllm_config.parallel_config
        if (
            int(parallel.data_parallel_size) != size
            or int(parallel.data_parallel_rank) != rank
        ):
            raise RuntimeError("vLLM did not apply the external DP topology")
        if not parallel.enable_expert_parallel:
            raise RuntimeError("External DP capture requires expert parallelism")
        results.put(("ready", rank, request, None))
        while True:
            request, method, args, kwargs = commands.get()
            if method == "shutdown":
                break
            result = _execute(llm.llm_engine, rank, method, args, kwargs)
            results.put(("result", rank, request, result))
    except BaseException:
        results.put(("error", rank, request, traceback.format_exc()))
        raise
    finally:
        if llm is not None:
            del llm
            gc.collect()


class ParallelCaptureLLM:
    """Proxy the existing single-request capture protocol across DP frontends."""

    def __init__(self, *, data_parallel_size, options, timeout=1800):
        self.size = int(data_parallel_size)
        required = self.size * int(options["tensor_parallel_size"])
        if self.size < 2 or not options.get("enable_expert_parallel"):
            raise ValueError("Parallel capture requires DP>1 and expert parallelism")
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if (
            visible
            and len([item for item in visible.split(",") if item.strip()]) < required
        ):
            raise ValueError(f"TP*DP requires {required} visible logical NPUs")
        if not visible:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
                map(str, range(required))
            )
        self.model_path = options["model"]
        self.model_config = SimpleNamespace(max_model_len=options["max_model_len"])
        self.llm_engine = self
        self.timeout = timeout
        self._request = 0
        self._closed = False
        self._failure = None
        self._processes = []
        self._commands = []
        context = mp.get_context("spawn")
        self._results = context.Queue()
        port = _free_port()
        try:
            for rank in range(self.size):
                commands = context.Queue()
                process = context.Process(
                    target=_dp_frontend,
                    args=(rank, self.size, port, options, commands, self._results),
                    name=f"speco-capture-dp-{rank}",
                )
                process.start()
                self._processes.append(process)
                self._commands.append(commands)
            self._receive("ready", -1)
        except BaseException:
            self.shutdown_capture_frontends()
            raise
        atexit.register(self.shutdown_capture_frontends)

    def _receive(self, kind, request):
        pending = set(range(self.size))
        values = [None] * self.size
        deadline = time.monotonic() + self.timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"DP capture timed out waiting for ranks {sorted(pending)}"
                )
            try:
                received, rank, job, value = self._results.get(
                    timeout=min(1, remaining)
                )
            except queue.Empty:
                dead = [
                    rank
                    for rank in pending
                    if self._processes[rank].exitcode is not None
                ]
                if dead:
                    raise RuntimeError(
                        f"DP capture frontend exited before responding: {dead}"
                    ) from None
                continue
            if received == "error":
                raise RuntimeError(f"DP capture rank {rank} failed:\n{value}")
            if received != kind or job != request or rank not in pending:
                raise RuntimeError(
                    "Unexpected DP capture response; session cannot continue"
                )
            pending.remove(rank)
            values[rank] = value
        return values

    def _exchange(self, method, *args, **kwargs):
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("DP capture frontends are closed")
        self._request += 1
        for commands in self._commands:
            commands.put((self._request, method, args, kwargs))
        try:
            return self._receive("result", self._request)
        except Exception as error:
            self._failure = error
            raise

    def has_unfinished_requests(self):
        return any(self._exchange("has_unfinished_requests"))

    def add_request(self, *args, **kwargs):
        self._exchange("add_request", *args, **kwargs)

    def step(self):
        return self._exchange("step")[0]

    def abort_request(self, *args, **kwargs):
        self._exchange("abort_request", *args, **kwargs)

    def collective_rpc(self, *args, **kwargs):
        return self._exchange("collective_rpc", *args, **kwargs)[0]

    def draft_update_rpc(self, *args, **kwargs):
        return [
            result
            for lane in self._exchange("draft_update_rpc", *args, **kwargs)
            for result in lane
        ]

    def get_tokenizer(self):
        from vllm.tokenizers import get_tokenizer

        return get_tokenizer(self.model_path, tokenizer_mode="deepseek_v4")

    def shutdown_capture_frontends(self):
        if self._closed:
            return
        self._closed = True
        for commands in self._commands:
            commands.put((-1, "shutdown", (), {}))
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for commands in self._commands:
            commands.close()
            commands.join_thread()
        self._results.close()
        self._results.join_thread()
        atexit.unregister(self.shutdown_capture_frontends)
