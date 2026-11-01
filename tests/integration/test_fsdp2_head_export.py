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
"""Real CPU DTensor reference tests for the GPU FSDP2 head-only export."""

import ast
import logging
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from verl_speco.integration import rollout_publish
from verl_speco.trainer.scheduler import DrafterExecutionStrategy


def _worker(module, rank=0, strategy="fsdp2"):
    return SimpleNamespace(
        _is_actor=True,
        rank=rank,
        config={
            "actor": {"strategy": strategy},
            "rollout": {
                "drafter": {
                    "speculative_algorithm": "DSPARK",
                    "training": {"gpu_transfer_optimizations": True},
                }
            },
        },
        actor=SimpleNamespace(engine=SimpleNamespace(module=module)),
    )


def test_plain_parameter_uses_original_engine_path():
    module = torch.nn.Module()
    module.lm_head = torch.nn.Linear(4, 8, bias=False)
    worker = _worker(module)
    calls = []

    def engine_export(**kwargs):
        calls.append(kwargs)
        return iter(module.named_parameters()), None

    worker.actor.engine.get_per_tensor_param = engine_export
    payload = rollout_publish.export_actor_lm_head_weight(worker)
    assert payload["export_strategy"] == "engine_full_param"
    diagnostics = payload["export_diagnostics"]
    assert diagnostics["fallback_reason"] == "head_not_dtensor"
    assert diagnostics["parameter_type"].endswith(".Parameter")
    assert diagnostics["source_file"].endswith("rollout_publish.py")
    assert diagnostics["elapsed_s"] >= diagnostics["host_copy_s"] >= 0
    metrics = rollout_publish.target_lm_head_export_metrics(payload)
    assert metrics["drafter/target_lm_head_direct_full_export"] == 0
    assert metrics["drafter/target_lm_head_engine_full_export"] == 1
    assert metrics["drafter/target_lm_head_export_fallback_reason"] == 4
    torch.testing.assert_close(
        payload["weight"], module.lm_head.weight.to(torch.bfloat16)
    )
    assert len(calls) == 1


@pytest.mark.parametrize("prefix", [(), ("actor_rollout_ref",)])
@pytest.mark.parametrize(
    "option,value", [("lora_rank", 8), ("lora_adapter_path", "adapter")]
)
def test_adapter_export_keeps_engine_merge_path(prefix, option, value):
    worker = _worker(torch.nn.Module())
    if prefix:
        worker.config = {"actor_rollout_ref": worker.config}
    config = worker.config["actor_rollout_ref"] if prefix else worker.config
    config["model"] = {option: value}
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    calls = []

    def merged_engine_export(**kwargs):
        calls.append(kwargs)
        return iter([("lm_head.weight", weight)]), None

    worker.actor.engine.get_per_tensor_param = merged_engine_export
    payload = rollout_publish.export_actor_lm_head_weight(worker)
    assert len(calls) == 1
    assert payload["export_strategy"] == "engine_full_param"
    assert payload["export_diagnostics"]["fallback_reason"] == "lora_export"
    torch.testing.assert_close(
        payload["weight"], weight.to(torch.bfloat16), rtol=0, atol=0
    )


def test_older_worker_metadata_is_reported_as_unknown():
    metrics = rollout_publish.target_lm_head_export_metrics(
        {"weight": torch.ones(2, 2)}
    )
    assert metrics["drafter/target_lm_head_export_diagnostics_present"] == 0
    assert metrics["drafter/target_lm_head_export_fallback_reason"] == -1
    assert metrics["drafter/target_lm_head_direct_full_export"] == 0
    assert not any(key.startswith("timing_s/") for key in metrics)


def test_row_selection_keeps_sparse_path(monkeypatch):
    module = torch.nn.Module()
    module.lm_head = torch.nn.Linear(4, 8, bias=False)

    def forbidden_full_export(*args, **kwargs):
        raise AssertionError("row selection must not invoke full-head export")

    monkeypatch.setattr(
        rollout_publish, "_export_fsdp2_lm_head_full", forbidden_full_export
    )
    payload = rollout_publish.export_actor_lm_head_weight(
        _worker(module), row_indices=[1, 5]
    )
    assert payload["export_strategy"] == "direct_sparse"
    torch.testing.assert_close(
        payload["weight"],
        module.lm_head.weight.detach()[[1, 5]].to(torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_driver_returns_worker_diagnostics_in_metrics_and_logs(caplog):
    # Execute the actual driver method without importing Ray/verl's GPU stack.
    path = (
        Path(rollout_publish.__file__).parents[1] / "trainer" / "speco_ray_trainer.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_speco_start_target_lm_head_weight_sync"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    dispatch_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_speco_dispatch_target_lm_head_payload"
    )
    module.body.append(dispatch_method)
    namespace = {
        "time": time,
        "logger": logging.getLogger("head-export-test"),
        "_get_nested": rollout_publish._get_nested,
        "target_lm_head_export_metrics": rollout_publish.target_lm_head_export_metrics,
        "DrafterExecutionStrategy": DrafterExecutionStrategy,
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - execute repository method under test
    diagnostics = {
        "fallback_reason": "head_not_dtensor",
        "source_file": "/worker/rollout_publish.py",
    }
    payload = {
        "export_strategy": "engine_full_param",
        "export_diagnostics": diagnostics,
    }
    worker = SimpleNamespace(
        config={},
        global_steps=8,
        device_name="cpu",
        _speco_rollout_idle_worker_enabled=lambda: False,
        _speco_drafter_schedule_config=lambda: SimpleNamespace(
            execution_strategy=DrafterExecutionStrategy.SYNC
        ),
        _speco_drafter_training_config=dict,
        _speco_get_drafter_target_lm_head_row_selection=lambda: None,
        _speco_actor_rollout_method=lambda name: lambda *args, **kwargs: [payload],
        _ray_get_if_needed=lambda value: value,
        _first_non_null=lambda values: values[0],
        _speco_build_drafter_target_lm_head_sync_args=lambda value: (value, 8, None),
        speco_sync_target_lm_head_weight=lambda *args, **kwargs: object(),
    )
    worker._speco_dispatch_target_lm_head_payload = lambda value, **kwargs: namespace[
        dispatch_method.name
    ](worker, value, **kwargs)
    with caplog.at_level(logging.WARNING):
        metrics, pending = namespace[method.name](worker)
    assert pending is not None
    assert metrics["drafter/target_lm_head_export_fallback_reason"] == 4
    assert metrics["drafter/target_lm_head_engine_full_export"] == 1
    assert "/worker/rollout_publish.py" in caplog.text
    assert "head_not_dtensor" in caplog.text


@pytest.mark.parametrize(
    "strategy,reason,code",
    [
        ("fsdp", "actor_not_fsdp2", 1),
        ("fsdp2", "head_not_found", 3),
    ],
)
def test_diagnostic_reasons_before_collectives(strategy, reason, code):
    diagnostics = {}
    supported, payload = rollout_publish._export_fsdp2_lm_head_full(
        _worker(torch.nn.Module(), strategy=strategy), diagnostics
    )
    assert not supported and payload is None
    assert diagnostics["fallback_reason"] == reason
    metrics = rollout_publish.target_lm_head_export_metrics(
        {"export_diagnostics": diagnostics}
    )
    assert metrics["drafter/target_lm_head_export_fallback_reason"] == code


def _distributed_export_check(rank, rendezvous, backend="gloo"):
    device_type = "cuda" if backend == "nccl" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = dist.device_mesh.init_device_mesh(device_type, (2,))
        # Exercise row/column sharding, replicated heads, tied aliases, and the
        # embedding fallback. All tests use the same collectives on both ranks.
        for placement, layout, algorithm in [
            (Shard(0), "head", "DSPARK"),
            (Shard(1), "head", "DSPARK"),
            (Replicate(), "head", "DSPARK"),
            (Shard(0), "tied", "DSPARK"),
            (Shard(0), "embedding", "DSPARK"),
            (Shard(0), "head", None),
            (Shard(0), "head", "DFLASH"),
            (Shard(0), "head", "EAGLE3"),
        ]:
            full_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 37
            module = torch.nn.Module()
            if layout in {"tied", "embedding"}:
                module.model = torch.nn.Module()
                module.model.embed_tokens = torch.nn.Embedding(8, 4)
            if layout != "embedding":
                module.lm_head = torch.nn.Linear(4, 8, bias=False)
            parameter = torch.nn.Parameter(
                distribute_tensor(full_weight, mesh, [placement])
            )
            if layout in {"tied", "embedding"}:
                module.model.embed_tokens.weight = parameter
            if layout != "embedding":
                module.lm_head.weight = parameter
            worker = _worker(module, rank)
            if algorithm is None:
                del worker.config["rollout"]["drafter"]["speculative_algorithm"]
            else:
                worker.config["rollout"]["drafter"]["speculative_algorithm"] = algorithm

            def forbidden_engine_export(**kwargs):
                raise AssertionError(
                    "head-only export must not enumerate engine weights"
                )

            worker.actor.engine.get_per_tensor_param = forbidden_engine_export
            before = parameter.to_local().detach().clone()
            payload = rollout_publish.export_actor_lm_head_weight(worker)
            assert parameter.grad is None
            torch.testing.assert_close(parameter.to_local(), before, rtol=0, atol=0)
            assert parameter.requires_grad
            if rank != 0:
                assert payload is None
            else:
                assert payload["export_strategy"] == "direct_full"
                diagnostics = payload["export_diagnostics"]
                assert diagnostics["fallback_reason"] == "none"
                assert diagnostics["shape"] == [8, 4]
                assert diagnostics["worker_rank"] == 0
                assert diagnostics["cast_before_copy"] == int(device_type == "cuda")
                assert (
                    diagnostics["source_bytes"]
                    == full_weight.numel() * full_weight.element_size()
                )
                assert diagnostics["payload_bytes"] == full_weight.numel() * 2
                assert diagnostics["placements"] == [str(placement)]
                metrics = rollout_publish.target_lm_head_export_metrics(payload)
                assert metrics["drafter/target_lm_head_direct_full_export"] == 1
                assert metrics["drafter/target_lm_head_export_fallback_reason"] == 0
                assert metrics["timing_s/drafter_head_export_gather_submit"] >= 0
                assert metrics["timing_s/drafter_head_export_host_copy"] >= 0
                expected_name = (
                    "model.embed_tokens.weight"
                    if layout == "embedding"
                    else "lm_head.weight"
                )
                assert payload["name"] == expected_name
                assert payload["row_indices"] is None
                assert payload["source_vocab_size"] == 8
                assert payload["selected_rows"] is None

                # Run the original engine export with the same logical weight.
                legacy = _worker(None, rank)
                legacy.actor.engine.get_per_tensor_param = (
                    lambda name=expected_name, weight=full_weight, **kwargs: (
                        iter([(name, weight)]),
                        None,
                    )
                )
                reference = rollout_publish.export_actor_lm_head_weight(legacy)
                torch.testing.assert_close(
                    payload["weight"], reference["weight"], rtol=0, atol=0
                )
                assert payload["weight"].dtype == torch.bfloat16
                assert payload["weight"].is_contiguous()
                # A small trainable drafter projection receives exactly the same
                # loss, gradient and SGD update through either frozen target head.
                updates = []
                for head in (reference["weight"], payload["weight"]):
                    projection = torch.nn.Parameter(torch.eye(4))
                    optimizer = torch.optim.SGD([projection], lr=0.01)
                    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11
                    logits = torch.nn.functional.linear(
                        hidden @ projection, head.float()
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits, torch.tensor([1, 3, 5])
                    )
                    loss.backward()
                    grad = projection.grad.clone()
                    optimizer.step()
                    updates.append(
                        (
                            logits.detach(),
                            loss.detach(),
                            grad,
                            projection.detach().clone(),
                        )
                    )
                for old, new in zip(*updates, strict=True):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
            # Export is a detached snapshot of this version, not live storage.
            with torch.no_grad():
                parameter.to_local().add_(10)
            if rank == 0:
                torch.testing.assert_close(
                    payload["weight"], reference["weight"], rtol=0, atol=0
                )

        # A collective failure must propagate, never enter the engine fallback.
        original = DTensor.full_tensor

        def fail_collective(self, *args, **kwargs):
            raise RuntimeError("injected collective failure")

        DTensor.full_tensor = fail_collective
        try:
            with pytest.raises(RuntimeError, match="injected collective failure"):
                rollout_publish.export_actor_lm_head_weight(worker)
        finally:
            DTensor.full_tensor = original
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_two_rank_export_matches_legacy_weights_loss_gradients_and_update(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_export_check,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA GPUs and NCCL",
)
def test_two_gpu_export_matches_legacy_weights_loss_gradients_and_update(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_export_check,
        args=((tmp_path / "cuda-rendezvous").as_uri(), "nccl"),
        nprocs=2,
        join=True,
    )
