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
"""Exact values, lifetimes and optimizer references across transfer fast paths."""

import ast
import copy
import logging
import pickle
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from verl_speco.integration import rollout_publish
from verl_speco.integration.tensor_wire import decode_cpu_tensors, encode_cpu_tensors
from verl_speco.trainer.gpu_transfer import FrozenHeadHostCache, ModelHostBufferPool
from verl_speco.trainer.gpu_transfer import load_model_from_host


def _bits(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_wire_roundtrip_preserves_bits_and_readonly_ray_layout(dtype, shape):
    import numpy as np

    tensor = torch.zeros(shape, dtype=dtype)
    if tensor.numel():
        tensor.reshape(-1)[0] = 1
    if tensor.is_floating_point() and tensor.numel() > 2:
        tensor.reshape(-1)[1:3] = torch.tensor(
            [float("inf"), float("nan")], dtype=dtype
        )
    source = {"weight": tensor.clone()}
    wire = encode_cpu_tensors(source)
    assert wire is not None
    # Ray NumPy arrays are read-only views after object-store materialization.
    wire = pickle.loads(pickle.dumps(wire))
    wire["_speco_cpu_tensor_wire_v1"]["weight"]["data"].setflags(write=False)
    result = decode_cpu_tensors(wire)
    assert result["weight"].dtype == dtype and result["weight"].shape == shape
    assert torch.equal(_bits(result["weight"]), _bits(source["weight"]))
    assert (
        np.shares_memory(
            result["weight"].numpy()
            if dtype != torch.bfloat16
            else _bits(result["weight"]).numpy(),
            wire["_speco_cpu_tensor_wire_v1"]["weight"]["data"],
        )
        or not tensor.numel()
    )
    # Consumer operations read the source and copy into their own model.
    destination = torch.empty_like(tensor)
    destination.copy_(result["weight"])
    assert torch.equal(_bits(destination), _bits(source["weight"]))


@pytest.mark.parametrize("corruption", ["dtype", "shape", "bytes", "envelope"])
def test_wire_rejects_corruption_instead_of_publishing_partial_weights(corruption):
    wire = encode_cpu_tensors({"weight": torch.ones(2, 3)})
    item = wire["_speco_cpu_tensor_wire_v1"]["weight"]
    if corruption == "dtype":
        item["dtype"] = "not_a_dtype"
    elif corruption == "shape":
        item["shape"] = (-1, 3)
    elif corruption == "bytes":
        item["data"] = item["data"][:-1]
    else:
        wire["extra"] = 1
    with pytest.raises(ValueError):
        decode_cpu_tensors(wire)


def test_wire_legacy_and_unsupported_layouts_are_not_mutated():
    source = {"weight": torch.arange(6).reshape(2, 3).t()}
    assert encode_cpu_tensors(source) is None
    assert decode_cpu_tensors(source) is source


def test_empty_wire_is_rejected():
    with pytest.raises(ValueError):
        decode_cpu_tensors({"_speco_cpu_tensor_wire_v1": {}})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "config_layout", ["scoped", "root", "serialized", "serialized_new"]
)
def test_actor_head_wire_preserves_metadata_and_exact_values(
    monkeypatch, enabled, config_layout
):
    source = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    payload = {
        "weight": source,
        "row_indices": None,
        "source_vocab_size": 4,
        "name": "lm_head.weight",
        "export_strategy": "direct_full",
        "export_diagnostics": {},
    }
    monkeypatch.setattr(
        rollout_publish, "export_actor_lm_head_weight", lambda *a, **k: payload
    )
    monkeypatch.setattr(
        rollout_publish,
        "_torch_module",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    training = {"gpu_transfer_optimizations": enabled}
    scoped = {"rollout": {"drafter": {"training": training}}}
    if config_layout.startswith("serialized"):
        import json

        worker = type(
            "Worker",
            (),
            {
                (
                    "_speco_drafter_config_env"
                    if config_layout == "serialized_new"
                    else "_speco_sglang_drafter_config_env"
                ): json.dumps({"training": training})
            },
        )()
        # Actual verl workers may retain runtime drafter fields but omit training.
        worker.config = {"rollout": {"drafter": {"enable": True}}}
    else:
        worker = SimpleNamespace(
            config=scoped
            if config_layout == "scoped"
            else {"actor_rollout_ref": scoped}
        )
    result = rollout_publish.DraftWeightPublishMixin.get_actor_lm_head_weight(worker)
    assert result["name"] == payload["name"] and result["source_vocab_size"] == 4
    actual = (
        decode_cpu_tensors(result["head_wire"])["weight"]
        if enabled
        else result["weight"]
    )
    assert torch.equal(_bits(actual), _bits(source))
    metrics = rollout_publish.target_lm_head_export_metrics(result)
    assert metrics["drafter/target_lm_head_numpy_transport"] == int(enabled)
    assert metrics["drafter/target_lm_head_direct_full_export"] == 1


def test_host_cache_checkpoint_state_after_restore(tmp_path):
    trainer = _head_trainer()
    model = trainer.backend.target_lm_head
    parameter = model.fc.weight
    expected = parameter.detach().clone()
    cache = FrozenHeadHostCache()
    assert cache.capture(parameter, 4)
    parameter.data = parameter.detach().clone()
    cache.arm(parameter)
    assert cache.restore(parameter, 4)[0]
    checkpoint = tmp_path / "head.pt"
    torch.save(model.state_dict(), checkpoint)
    with torch.no_grad():
        parameter.zero_()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert torch.equal(parameter, expected)
    assert parameter.dtype == expected.dtype and not parameter.requires_grad


def test_bubble_head_export_keeps_its_own_reference_transport(monkeypatch):
    payload = {
        "weight": torch.ones(4, 3, dtype=torch.bfloat16),
        "row_indices": None,
        "export_strategy": "direct_full",
    }
    monkeypatch.setattr(
        rollout_publish, "export_actor_lm_head_weight", lambda *a, **k: payload
    )
    worker = SimpleNamespace(
        config={
            "rollout": {
                "drafter": {
                    "training": {
                        "gpu_transfer_optimizations": True,
                        "scheduler": {"execution": {"strategy": "rollout_idle_worker"}},
                    }
                }
            }
        }
    )
    result = rollout_publish.DraftWeightPublishMixin.get_actor_lm_head_weight(worker)
    assert result is payload and "head_wire" not in result


def test_worker_head_wire_forwards_version_and_apply_contract():
    path = Path(__file__).parents[2] / "verl_speco/workers/speco_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpecoWorker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "sync_target_lm_head_weight"
    )
    method.decorator_list = []
    namespace = {"time": time, "logger": logging.getLogger(__name__)}
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - production method
    calls = []

    def sync(weight, **kwargs):
        calls.append((weight, kwargs))
        return {"accepted": True, "applied": False}

    worker = SimpleNamespace(
        enable_drafter=True,
        in_drafter_train_group=True,
        trainer=SimpleNamespace(sync_target_lm_head_weight=sync),
        rank=0,
        is_drafter_group_leader=False,
    )
    weight = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    payload = {
        "head_wire": encode_cpu_tensors({"weight": weight}),
        "weight": None,
        "source_vocab_size": 4,
        "defer_device_apply": True,
    }
    assert namespace["sync_target_lm_head_weight"](worker, payload, 42)["accepted"]
    assert torch.equal(_bits(calls[0][0]), _bits(weight))
    assert calls[0][1] == {
        "global_step": 42,
        "row_indices": None,
        "source_vocab_size": 4,
        "defer_device_apply": True,
    }
    # Target filtering must happen before decoding any large or invalid payload.
    rejected = namespace["sync_target_lm_head_weight"](
        worker,
        {
            **payload,
            "target_worker_ids": ("other",),
            "require_target_head_ack": True,
            "head_wire": "must not be decoded",
        },
        42,
    )
    assert rejected["reason"] == "not_target_worker" and rejected["worker_id"] == "0"
    assert len(calls) == 1


def test_publish_materialization_decodes_both_legacy_and_numpy_refs(monkeypatch):
    class Ref:
        def __init__(self, value):
            self.value = value

    monkeypatch.setattr(
        rollout_publish,
        "_ray_module",
        lambda: SimpleNamespace(ObjectRef=Ref, get=lambda ref: ref.value),
    )
    source = {"layer.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    for encoded in (False, True):
        stored = pickle.loads(
            pickle.dumps(encode_cpu_tensors(source) if encoded else source)
        )
        result, used = rollout_publish.materialize_draft_weights_payload(
            {"weights_ref": Ref(stored)}
        )
        assert used and set(result) == set(source)
        assert torch.equal(_bits(result["layer.weight"]), _bits(source["layer.weight"]))


def _head_trainer():
    path = Path(__file__).parents[2] / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DrafterBaseTrainer"
    )
    names = {
        "_target_lm_head_module",
        "_move_target_lm_head",
        "_apply_pending_target_lm_head_weight",
        "_record_gpu_transfer",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    cls.bases = []
    namespace = {
        "torch": torch,
        "time": time,
        "json": __import__("json"),
        "logging": logging,
        "device_name": "cuda",
        "logger": logging.getLogger(__name__),
        "last_hidden_logprob_check_enabled": lambda: False,
    }
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - production methods
    trainer = namespace["DrafterBaseTrainer"]()
    head = torch.nn.Module()
    head.fc = torch.nn.Linear(3, 4, bias=False).requires_grad_(False)
    trainer.backend = SimpleNamespace(target_lm_head=head)
    trainer.rank = 0
    trainer._gpu_head_host_reuse = True
    trainer._target_lm_head_weight_step = 4
    trainer._pending_target_lm_head_weight = None
    trainer._pending_target_lm_head_row_indices = None
    trainer._pending_target_lm_head_source_vocab_size = None
    trainer._pending_target_lm_head_chunked_apply = False
    return trainer


def test_head_restore_reuses_host_storage_with_exact_values_and_parameter_identity():
    trainer = _head_trainer()
    p = trainer.backend.target_lm_head.fc.weight
    original = p.detach().clone()
    cache = trainer._gpu_head_host_cache = FrozenHeadHostCache()
    assert cache.capture(p, 4)
    host_pointer = cache.host.data_ptr()
    # Simulate Module.to() changing storage; optional CUDA test below uses CUDA.
    p.data = p.detach().clone()
    cache.arm(p)
    assert trainer._move_target_lm_head("cpu")
    assert p is trainer.backend.target_lm_head.fc.weight
    assert p.data_ptr() == host_pointer and torch.equal(p, original)
    assert trainer._gpu_transfer_metrics["drafter/gpu_head_restore_cache_hit"] == 1


@pytest.mark.parametrize("change", ["version", "mutation", "replacement", "trainable"])
def test_head_cache_invalidates_when_supervision_is_no_longer_identical(change):
    trainer = _head_trainer()
    p = trainer.backend.target_lm_head.fc.weight
    cache = trainer._gpu_head_host_cache = FrozenHeadHostCache()
    cache.capture(p, 4)
    p.data = p.detach().clone()
    cache.arm(p)
    if change == "version":
        trainer._target_lm_head_weight_step = 5
    elif change == "mutation":
        p.add_(1)
    elif change == "replacement":
        trainer.backend.target_lm_head.fc.weight = torch.nn.Parameter(
            p.clone(), requires_grad=False
        )
    else:
        p.requires_grad_(True)
    expected = trainer.backend.target_lm_head.fc.weight.detach().clone()
    trainer._move_target_lm_head("cpu")
    assert torch.equal(trainer.backend.target_lm_head.fc.weight, expected)
    assert trainer._gpu_transfer_metrics["drafter/gpu_head_restore_cache_hit"] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_direct_host_head_apply_matches_original_conversion_bitwise(dtype):
    trainer = _head_trainer()
    trainer.backend.target_lm_head.to(dtype=dtype)
    source = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3) / 7
    expected = source.to(dtype=dtype).clone()
    trainer._pending_target_lm_head_weight = source
    assert trainer._apply_pending_target_lm_head_weight()
    assert torch.equal(_bits(trainer.backend.target_lm_head.fc.weight), _bits(expected))
    assert trainer._pending_target_lm_head_weight is None


def test_reusable_pool_preserves_aliases_optimizer_and_next_update():
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(3, 3, bias=False)
    model.alias = model.layer
    model.register_buffer("counter", torch.tensor([4], dtype=torch.int64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(9).reshape(3, 3).float() / 10
    model.layer(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    ids = [id(p) for p in model.parameters()]
    pool = ModelHostBufferPool(1024 * 1024)
    for iteration in range(2):
        if iteration:
            # Simulate activation copying CPU storage to fresh device storage.
            model._apply(lambda tensor: tensor.clone())
        used, stats = pool.offload(model)
        assert used and stats["pool_bytes"] <= 1024 * 1024
        assert stats["reused_bytes"] == (stats["copied_bytes"] if iteration else 0)
        assert [id(p) for p in model.parameters()] == ids
        assert model.layer.weight is model.alias.weight
        for subject, opt in ((model, optimizer), (reference, reference_optimizer)):
            opt.zero_grad(set_to_none=True)
            subject.layer(inputs).square().sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.testing.assert_close(
            model.state_dict(), reference.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=0
        )


def test_pool_bound_and_preflight_fallback_do_not_change_model():
    model = torch.nn.Linear(3, 3)
    pool = ModelHostBufferPool(0)
    used, stats = pool.offload(model)
    assert used and stats["pool_bytes"] == 0 and stats["uncached_bytes"] > 0
    model(torch.ones(1, 3)).sum().backward()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    used, stats = pool.offload(model)
    assert not used and stats["fallback_reason"] == "unsupported_layout_or_live_grad"
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]) and p.grad is not None


def _distributed_pool_check(rank, rendezvous, backend):
    device = "cuda" if backend == "nccl" else "cpu"
    if device == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = torch.distributed.device_mesh.init_device_mesh(device, (2,))
        model = torch.nn.Module()
        full = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(device)
        model.register_parameter(
            "weight", torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
        )
        original_id = id(model.weight)
        pool = ModelHostBufferPool(
            1024 * 1024, pinned=device == "cuda", asynchronous=device == "cuda"
        )
        for iteration in range(2):
            if iteration:
                used_load, load_stats = load_model_from_host(
                    model,
                    device,
                    device_module=torch.cuda if device == "cuda" else None,
                )
                assert used_load and load_stats["copied_bytes"] > 0
                with torch.no_grad():
                    model.weight.add_(1)
                full += 1
            used, stats = pool.offload(
                model, device_module=torch.cuda if device == "cuda" else None
            )
            assert used and isinstance(model.weight, DTensor)
            assert id(model.weight) == original_id
            assert model.weight.to_local().device.type == "cpu"
            # Gather on the mesh device, exactly as the legacy export path does.
            restored = model.weight.to(device).full_tensor()
            assert torch.equal(restored, full)
            assert (
                stats["reused_bytes"] > 0 if iteration else stats["allocated_bytes"] > 0
            )
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_dtensor_pool_preserves_shards_and_reuses_buffers(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_pool_check,
        args=((tmp_path / "pool-rendezvous").as_uri(), "gloo"),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_rank_cuda_dtensor_pool_pinned_async(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_pool_check,
        args=((tmp_path / "pool-cuda-rendezvous").as_uri(), "nccl"),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_head_cache_roundtrip():
    trainer = _head_trainer()
    p = trainer.backend.target_lm_head.fc.weight
    expected = p.detach().clone()
    assert trainer._move_target_lm_head(torch.device("cuda", 0))
    assert trainer._move_target_lm_head("cpu")
    assert torch.equal(p, expected)
    assert trainer._gpu_transfer_metrics["drafter/gpu_head_restore_cache_hit"] == 1


def _cuda_fsdp2_optimizer_check(rank, rendezvous):
    from torch.distributed.fsdp import fully_shard

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(123)
        mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (2,))
        model = torch.nn.Linear(8, 8, bias=False).cuda()
        reference = copy.deepcopy(model)
        fully_shard(model, mesh=mesh)
        fully_shard(reference, mesh=mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        parameter_ids = [id(p) for p in model.parameters()]
        pool = ModelHostBufferPool(1024 * 1024, pinned=True, asynchronous=True)
        for iteration in range(3):
            inputs = torch.ones(2, 8, device="cuda") * (iteration + 1)
            for subject, opt in ((model, optimizer), (reference, reference_optimizer)):
                subject(inputs).square().sum().backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            used, stats = pool.offload(model, device_module=torch.cuda)
            assert used and stats["pool_bytes"] <= 1024 * 1024
            assert [id(p) for p in model.parameters()] == parameter_ids
            # Compare checkpoint-visible local shards against the legacy path.
            reference.cpu()
            for name, tensor in model.state_dict().items():
                expected = reference.state_dict()[name]
                torch.testing.assert_close(
                    tensor.to_local(), expected.to_local(), rtol=0, atol=0
                )
            used_load, load_stats = load_model_from_host(
                model, torch.device("cuda", rank), device_module=torch.cuda
            )
            assert used_load and load_stats["copied_bytes"] > 0
            reference.cuda()
            for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                for key, value in optimizer.state[p].items():
                    expected = reference_optimizer.state[q][key]
                    if isinstance(value, DTensor):
                        value, expected = value.to_local(), expected.to_local()
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two CUDA devices and FSDP2"
)
def test_cuda_fsdp2_next_updates_and_optimizer_equal_legacy_offload(tmp_path):
    torch.multiprocessing.spawn(
        _cuda_fsdp2_optimizer_check,
        args=((tmp_path / "fsdp2-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_master_switch_and_legacy_overrides():
    from verl_speco.trainer.gpu_transfer import gpu_transfer_enabled

    assert not gpu_transfer_enabled(None)
    assert gpu_transfer_enabled({})
    assert gpu_transfer_enabled({"gpu_transfer_optimizations": True})
    assert not gpu_transfer_enabled(
        {"gpu_transfer_optimizations": False, "gpu_head_numpy_transport": True},
        "gpu_head_numpy_transport",
    )
    assert not gpu_transfer_enabled(
        {"gpu_head_numpy_transport": False}, "gpu_head_numpy_transport"
    )


def test_batched_load_keeps_values_rng_aliases_and_next_optimizer_updates(tmp_path):
    model = torch.nn.Module()
    model.layer = torch.nn.Linear(3, 4)
    model.alias = model.layer
    model.register_buffer("counter", torch.arange(4))
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    ids = [id(p) for p in model.parameters()]
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7
    for _ in range(3):
        rng_before = torch.get_rng_state().clone()
        used, stats = load_model_from_host(model, "cpu")
        assert used and stats["tensor_count"] == 3
        assert stats["allocated_bytes"] == stats["copied_bytes"] > 0
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert [id(p) for p in model.parameters()] == ids
        assert model.alias.weight is model.layer.weight
        for subject, opt in ((model, optimizer), (reference, reference_optimizer)):
            subject.layer(inputs).square().sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.testing.assert_close(
            model.state_dict(), reference.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=0
        )
    checkpoint = tmp_path / "loaded.pt"
    torch.save(model.state_dict(), checkpoint)
    reference.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(
        model.state_dict(), reference.state_dict(), rtol=0, atol=0
    )


def test_batched_load_preflight_fallback_keeps_live_gradients():
    model = torch.nn.Linear(3, 4)
    model(torch.ones(1, 3)).sum().backward()
    before = copy.deepcopy(model.state_dict())
    gradients = [p.grad for p in model.parameters()]
    used, stats = load_model_from_host(model, "cpu")
    assert not used and stats["copied_bytes"] == 0
    assert [
        p.grad is g for p, g in zip(model.parameters(), gradients, strict=True)
    ] == [True, True]
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["allocation", "copy"])
def test_batched_load_failure_does_not_publish_partial_parameters(monkeypatch, failure):
    model = torch.nn.Linear(3, 4)
    before = copy.deepcopy(model.state_dict())
    pointers = [p.data_ptr() for p in model.parameters()]
    if failure == "allocation":
        original = torch.empty_like
        calls = []

        def allocate(tensor, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise torch.OutOfMemoryError("injected allocation failure")
            return original(tensor, **kwargs)

        monkeypatch.setattr(torch, "empty_like", allocate)
        used, stats = load_model_from_host(model, "cpu")
        assert not used and stats["fallback_reason"] == "staging_allocation_oom"
        assert stats["copied_bytes"] == 0
    else:
        original = torch.Tensor.copy_
        calls = []

        def copy_tensor(destination, source, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("injected copy failure")
            return original(destination, source, **kwargs)

        monkeypatch.setattr(torch.Tensor, "copy_", copy_tensor)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            load_model_from_host(model, "cpu")
    assert [p.data_ptr() for p in model.parameters()] == pointers
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_actor_config_resolution_prefers_explicit_disable_and_handles_bad_fallback():
    import json

    worker = type(
        "Worker",
        (),
        {
            "_speco_sglang_drafter_config_env": json.dumps(
                {"training": {"gpu_transfer_optimizations": True}}
            )
        },
    )()
    worker.config = {
        "rollout": {"drafter": {"training": {"gpu_transfer_optimizations": False}}}
    }
    training, source = rollout_publish._gpu_transfer_training_config(worker)
    assert not training["gpu_transfer_optimizations"]
    assert source == "rollout.drafter.training"
    # A partial local training config must not hide an explicit global disable.
    type(worker)._speco_sglang_drafter_config_env = json.dumps(
        {"training": {"gpu_transfer_optimizations": False}}
    )
    worker.config = {"rollout": {"drafter": {"training": {}}}}
    training, source = rollout_publish._gpu_transfer_training_config(worker)
    assert not training["gpu_transfer_optimizations"]
    assert source == "serialized_drafter_config"
    worker.config = {}
    type(worker)._speco_sglang_drafter_config_env = "invalid json"
    assert rollout_publish._gpu_transfer_training_config(worker) == (
        None,
        "invalid_serialized_config",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pageable_batched_offload_finishes_before_return_and_reuses_storage(dtype):
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 16)).to(
        device="cuda", dtype=dtype
    )
    identities = {name: id(p) for name, p in model.named_parameters()}
    pool = ModelHostBufferPool(1024 * 1024)
    host_pointers = None
    for iteration in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(iteration + 1)
        expected = {
            name: p.detach().cpu().clone() for name, p in model.named_parameters()
        }
        used, stats = pool.offload(model, device_module=torch.cuda)
        assert used and stats["batched_copy_tensors"] == len(expected)
        assert stats["pinned_bytes"] == 0
        assert {name: id(p) for name, p in model.named_parameters()} == identities
        for name, p in model.named_parameters():
            assert torch.equal(_bits(p), _bits(expected[name]))
        pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
        if host_pointers is not None:
            assert pointers == host_pointers
            assert stats["allocated_bytes"] == 0
        host_pointers = pointers
        used, _ = load_model_from_host(
            model, torch.device("cuda"), device_module=torch.cuda
        )
        assert used


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_pool_is_bounded_and_failure_is_not_retried(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)).cuda()
    pool = ModelHostBufferPool(1 << 20, pinned=True, pin_budget_bytes=1024)
    used, stats = pool.offload(model, device_module=torch.cuda)
    assert used and 0 < stats["pool_pinned_bytes"] <= 1024
    assert stats["pool_pinned_bytes"] == stats["pinned_bytes"]
    assert load_model_from_host(model, torch.device("cuda"), device_module=torch.cuda)[
        0
    ]
    used, stats = pool.offload(model, device_module=torch.cuda)
    assert used and stats["allocated_bytes"] == 0
    assert stats["pool_pinned_bytes"] <= 1024

    model = torch.nn.Linear(16, 16).cuda()
    pool = ModelHostBufferPool(1 << 20, pinned=True)
    original = torch.empty_like
    attempts = []

    def fail_pin(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            attempts.append(1)
            raise RuntimeError("injected pinned allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", fail_pin)
    for _ in range(2):
        used, stats = pool.offload(model, device_module=torch.cuda)
        assert used and stats["pool_pinned_bytes"] == 0
        assert stats["pin_disabled_after_failure"] == 1
        assert load_model_from_host(
            model, torch.device("cuda"), device_module=torch.cuda
        )[0]
    assert len(attempts) == 1


@pytest.mark.parametrize("failed_copy", [1, 2])
def test_model_offload_waits_on_copy_failure_without_rebinding(
    monkeypatch, failed_copy
):
    from types import SimpleNamespace

    model = torch.nn.Linear(3, 4)
    before = copy.deepcopy(model.state_dict())
    pointers = [p.data_ptr() for p in model.parameters()]
    source_pointers = set(pointers)
    original_device = torch.Tensor.device
    original_copy = torch.Tensor.copy_
    events = []

    def reported_device(tensor):
        if tensor.data_ptr() in source_pointers:
            return torch.device("cuda:0")
        return original_device.__get__(tensor)

    def copy_tensor(destination, source, **kwargs):
        assert kwargs["non_blocking"]
        events.append("copy")
        if events.count("copy") == failed_copy:
            raise RuntimeError("injected model offload copy failure")
        return original_copy(destination, source, **kwargs)

    def current_stream(device):
        assert device == torch.device("cuda:0")
        return SimpleNamespace(synchronize=lambda: events.append("wait"))

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "device", property(reported_device))
        patch.setattr(torch.Tensor, "copy_", copy_tensor)
        with pytest.raises(RuntimeError, match="injected model offload copy failure"):
            ModelHostBufferPool(1024).offload(
                model, device_module=SimpleNamespace(current_stream=current_stream)
            )
    assert events == ["copy"] * failed_copy + ["wait"]
    assert [p.data_ptr() for p in model.parameters()] == pointers
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
