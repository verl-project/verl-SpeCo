# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise acceptance transport without importing the PPO dependency stack."""

import ast
import asyncio
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verl_speco.integration import vllm_runtime as runtime


def _trainer_helpers():
    torch = pytest.importorskip("torch")
    path = (
        Path(__file__).resolve().parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    )
    source = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        "_speco_float_values",
        "_speco_optional_float",
        "_speco_vllm_spec_decode_stats_from_batch",
        "_speco_vllm_spec_decode_metrics_from_stats",
        "_speco_merge_vllm_spec_decode_stats",
    }
    constants = {
        "_SPECO_VLLM_SPEC_DECODE_DRAFTS_KEY",
        "_SPECO_VLLM_SPEC_DECODE_ACCEPTED_TOKENS_KEY",
        "_SPECO_VLLM_REQUEST_VERIFY_ROUNDS_KEY",
        "_SPECO_VLLM_REQUEST_ACCEPTED_TOKENS_KEY",
        "_SPECO_VLLM_REQUEST_ID_KEY",
        "SPECO_VLLM_SPEC_DECODE_MEAN_ACCEPTANCE_METRIC",
    }
    nodes = [
        node
        for node in source.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
    ]
    original = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "SpecoRayPPOTrainer"
    )
    nodes.append(
        ast.ClassDef(
            name="Subject",
            bases=[],
            keywords=[],
            decorator_list=[],
            body=[
                node
                for node in original.body
                if isinstance(node, ast.FunctionDef)
                and node.name
                in {
                    "_speco_store_rollout_metrics",
                    "_speco_current_step_rollout_metrics",
                }
            ],
        )
    )
    namespace = {
        "torch": torch,
        "math": math,
        "Any": Any,
        "logger": logging.getLogger(__name__),
        "SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX": runtime.SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX,
        "_get_nested": runtime._get_nested,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("mode", ["sync", "rollout_idle_worker"])
def test_generate_drains_acceptance_once_and_restores_trainer_metric(
    monkeypatch, caplog, mode
):
    helpers = _trainer_helpers()
    driver = helpers["Subject"]()
    driver.global_steps = 1
    pops = []

    class Upstream:
        async def generate(self, request_id):
            self._speco_vllm_spec_decode_pending_stats = (
                {"drafts": 4, "accepted_tokens": 12}
                if request_id == "a"
                else {"drafts": 2, "accepted_tokens": 0}
            )
            return SimpleNamespace(extra_fields={})

    class Server(runtime._SpecoVLLMHttpServerMixin, Upstream):
        def _speco_pop_vllm_spec_decode_stats(self):
            pops.append(True)
            return super()._speco_pop_vllm_spec_decode_stats()

    async def acquire(*args, **kwargs):
        return False, 0.0

    monkeypatch.setattr(
        runtime,
        "_load_env_drafter_config",
        lambda: {
            "enable": True,
            "training": {"scheduler": {"execution": {"strategy": mode}}},
        },
    )
    monkeypatch.setattr(runtime, "acquire_rollout_resource_lease", acquire)
    monkeypatch.setattr(runtime, "_emit_rollout_idle_worker_event", lambda **kw: True)
    server = Server()

    async def scenario():
        for request_id, expected in (("a", 4), ("b", 2)):
            output = await server.generate(request_id=request_id)
            assert output.extra_fields["_speco_vllm_spec_decode_drafts"] == expected
            driver._speco_store_rollout_metrics(
                SimpleNamespace(
                    non_tensor_batch={
                        key: [value] for key, value in output.extra_fields.items()
                    }
                )
            )

    asyncio.run(scenario())
    assert len(pops) == 2  # One drain per output, not inner + outer drains.
    metric = "drafter/spec_decode/mean_acceptance_length"
    assert driver._speco_current_step_rollout_metrics()[metric] == 3.0
    assert (
        "available=True source=aggregate verify_rounds=4.0 accepted_tokens=12.0"
        in caplog.text
    )
    driver.global_steps = 2
    driver._speco_store_rollout_metrics(SimpleNamespace(non_tensor_batch={}))
    assert metric not in driver._speco_current_step_rollout_metrics()
    assert "available=False source=missing" in caplog.text


@pytest.mark.parametrize("kind", ["scheduler", "iteration", "both", "dict"])
def test_acceptance_callback_compatibility_does_not_double_count(kind):
    spec = SimpleNamespace(num_drafts=4, num_draft_tokens=20, num_accepted_tokens=12)
    scheduler = (
        SimpleNamespace(spec_decoding_stats=spec)
        if kind in {"scheduler", "both"}
        else None
    )
    iteration = (
        SimpleNamespace(spec_decoding_stats=spec)
        if kind in {"iteration", "both"}
        else None
    )
    if kind == "dict":
        iteration = {
            "spec_decoding_stats": {
                "num_drafts": 4,
                "num_draft_tokens": 20,
                "num_accepted_tokens": 12,
            }
        }
    stats = runtime._new_vllm_spec_decode_stats()
    source = runtime._record_vllm_spec_decode_scheduler_stats(
        stats, scheduler, iteration
    )
    assert source == ("scheduler" if scheduler is not None else "iteration")
    assert stats == {"drafts": 4, "draft_tokens": 20, "accepted_tokens": 12}


def test_stat_logger_consumes_iteration_only_keyword_callback(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.v1.metrics.loggers", SimpleNamespace(StatLoggerBase=object)
    )
    server = SimpleNamespace()
    logger = runtime._build_speco_vllm_stat_logger(server)(None)
    logger.record(
        iteration_stats=SimpleNamespace(
            spec_decoding_stats=SimpleNamespace(num_drafts=2, num_accepted_tokens=0)
        )
    )
    assert server._speco_vllm_spec_decode_pending_stats == {
        "drafts": 2,
        "draft_tokens": 0,
        "accepted_tokens": 0,
    }
    assert server._speco_acceptance_stats_source == "iteration"


def _request_batch(**overrides):
    fields = {
        "_speco_vllm_spec_decode_drafts": [0, 0],
        "_speco_vllm_spec_decode_accepted_tokens": [0, 0],
        "_speco_vllm_request_id": ["a", "b"],
        "_speco_vllm_request_verify_rounds": [4, 1],
        "_speco_vllm_request_accepted_tokens": [12, 0],
    }
    fields.update(overrides)
    return SimpleNamespace(non_tensor_batch=fields)


def test_complete_request_fallback_is_weighted_and_never_mixed_with_aggregate():
    helpers = _trainer_helpers()
    stats_from_batch = helpers["_speco_vllm_spec_decode_stats_from_batch"]
    metrics = helpers["_speco_vllm_spec_decode_metrics_from_stats"]
    diagnostics = {}
    stats = stats_from_batch(_request_batch(), diagnostics=diagnostics)
    assert diagnostics["source"] == "request"
    assert metrics(stats)["drafter/spec_decode/mean_acceptance_length"] == 3.4
    stats = stats_from_batch(
        _request_batch(
            _speco_vllm_spec_decode_drafts=[4, 0],
            _speco_vllm_spec_decode_accepted_tokens=[12, 0],
        ),
        diagnostics=diagnostics,
    )
    assert diagnostics["source"] == "aggregate"
    assert metrics(stats)["drafter/spec_decode/mean_acceptance_length"] == 4.0


@pytest.mark.parametrize(
    "field,values",
    [
        ("_speco_vllm_request_id", ["a", "a"]),
        ("_speco_vllm_request_id", ["a", None]),
        ("_speco_vllm_request_verify_rounds", [4]),
        ("_speco_vllm_request_verify_rounds", [4, None]),
        ("_speco_vllm_request_verify_rounds", [4, -1]),
        ("_speco_vllm_request_verify_rounds", [4, 0.5]),
        ("_speco_vllm_request_verify_rounds", [[4, 5], 1]),
        ("_speco_vllm_request_accepted_tokens", [12, float("nan")]),
        ("_speco_vllm_request_accepted_tokens", [12, None]),
        ("_speco_vllm_request_accepted_tokens", [12]),
    ],
)
def test_incomplete_request_fallback_omits_metric(field, values):
    helpers = _trainer_helpers()
    stats = helpers["_speco_vllm_spec_decode_stats_from_batch"](
        _request_batch(**{field: values})
    )
    assert stats == {}
    assert helpers["_speco_vllm_spec_decode_metrics_from_stats"](stats) == {}


def test_nested_counts_and_all_rejected_tokens_remain_valid():
    helpers = _trainer_helpers()
    stats = helpers["_speco_vllm_spec_decode_stats_from_batch"](
        _request_batch(
            _speco_vllm_request_verify_rounds=[[4], [1]],
            _speco_vllm_request_accepted_tokens=[[0], [0]],
            _speco_vllm_request_id=[["a"], ["b"]],
        )
    )
    assert (
        helpers["_speco_vllm_spec_decode_metrics_from_stats"](stats)[
            "drafter/spec_decode/mean_acceptance_length"
        ]
        == 1.0
    )
