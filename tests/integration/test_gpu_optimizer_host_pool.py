# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exact optimizer values, pool budgets and the completion-before-rebind fence."""

import ast
import copy
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from verl_speco.trainer.gpu_transfer import OptimizerHostBufferPool


@pytest.fixture
def transfer(monkeypatch):
    accelerator = set()
    pinned = set()
    device_property = torch.Tensor.device
    empty_like = torch.empty_like
    is_pinned = torch.Tensor.is_pinned
    events = []

    def pointer(tensor):
        return tensor.untyped_storage().data_ptr()

    def reported_device(tensor):
        return (
            torch.device("cuda:0")
            if pointer(tensor) in accelerator
            else device_property.__get__(tensor)
        )

    def allocate(tensor, **kwargs):
        pin = kwargs.pop("pin_memory", False)
        result = empty_like(tensor, **kwargs)
        if pin:
            pinned.add(pointer(result))
        return result

    monkeypatch.setattr(torch.Tensor, "device", property(reported_device))
    monkeypatch.setattr(
        torch.Tensor,
        "is_pinned",
        lambda self: pointer(self) in pinned or is_pinned(self),
    )
    monkeypatch.setattr(torch, "empty_like", allocate)
    module = SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(
            synchronize=lambda: events.append("wait")
        )
    )
    return SimpleNamespace(
        accelerator=accelerator,
        pinned=pinned,
        events=events,
        module=module,
        pointer=pointer,
        allocate=allocate,
    )


def optimizer(transfer, size=8):
    value = torch.arange(size, dtype=torch.float32)
    transfer.accelerator.add(transfer.pointer(value))
    return SimpleNamespace(state={"p": {"exp_avg": value, "step": 2}}), value


def test_first_copy_is_pinned_exact_and_second_copy_reuses_storage(transfer):
    opt, source = optimizer(transfer)
    pool = OptimizerHostBufferPool()
    used, first = pool.offload(opt, max_bytes=1024, device_module=transfer.module)
    host = opt.state["p"]["exp_avg"]
    assert used and torch.equal(host, source)
    assert host.is_pinned() and first["allocated_bytes"] == 32
    assert transfer.events == ["wait"]
    replacement = source + 10
    transfer.accelerator.add(transfer.pointer(replacement))
    opt.state["p"]["exp_avg"] = replacement
    used, second = pool.offload(opt, max_bytes=1024, device_module=transfer.module)
    assert used and opt.state["p"]["exp_avg"] is host
    assert second["allocated_bytes"] == 0 and second["reused_bytes"] == 32
    assert torch.equal(host, replacement)
    assert opt.state["p"]["step"] == 2


def test_budget_declines_before_allocation_or_state_changes(transfer):
    opt, source = optimizer(transfer)
    used, stats = OptimizerHostBufferPool().offload(
        opt, max_bytes=1, device_module=transfer.module
    )
    assert not used and stats["fallback_reason"] == "shared_pool_budget"
    assert opt.state["p"]["exp_avg"] is source and not transfer.events


def test_pin_failure_falls_back_once_and_reuses_pageable_buffers(transfer, monkeypatch):
    opt, source = optimizer(transfer)
    attempts = []

    def allocate(tensor, **kwargs):
        if kwargs.get("pin_memory"):
            attempts.append(1)
            raise RuntimeError("pin allocation unavailable")
        return transfer.allocate(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", allocate)
    pool = OptimizerHostBufferPool()
    for _ in range(2):
        opt.state["p"]["exp_avg"] = source
        assert pool.offload(opt, max_bytes=1024, device_module=transfer.module)[0]
    assert attempts == [1]
    assert not opt.state["p"]["exp_avg"].is_pinned()


def test_copy_failure_joins_work_and_does_not_rebind(transfer, monkeypatch):
    opt, source = optimizer(transfer)

    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        OptimizerHostBufferPool().offload(
            opt, max_bytes=1024, device_module=transfer.module
        )
    assert opt.state["p"]["exp_avg"] is source
    assert transfer.events == ["wait"]


def test_unsupported_layout_does_not_modify_state(transfer):
    opt, _ = optimizer(transfer)
    source = torch.ones(2, 3).t()
    transfer.accelerator.add(transfer.pointer(source))
    opt.state["p"]["exp_avg"] = source
    assert not OptimizerHostBufferPool().offload(
        opt, max_bytes=1024, device_module=transfer.module
    )[0]
    assert opt.state["p"]["exp_avg"] is source


def load_base_method():
    tree = ast.parse(
        Path("verl_speco/trainer/base_trainer.py").read_text(encoding="utf-8")
    )
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_offload_optimizer_state_to_cpu"
    )
    namespace = {
        "torch": torch,
        "logging": logging,
        "device_name": "cuda",
        "time": time,
        "logger": logging.getLogger(__name__),
        "offload_fsdp_optimizer": lambda opt: None,
    }
    exec(  # noqa: S102 - execute the isolated production method
        compile(ast.Module(body=[method], type_ignores=[]), "production", "exec"),
        namespace,
    )
    return namespace[method.name]


def test_first_success_marker_is_set_only_after_wait_and_budget_is_shared(transfer):
    opt, source = optimizer(transfer)
    trainer = SimpleNamespace(
        optimizer=opt,
        rank=0,
        current_rl_step=2,
        _gpu_optimizer_host_reuse=True,
        _gpu_offload_pool_max_bytes=64,
        _gpu_model_host_pool=SimpleNamespace(entries={"model": (None, None, 32)}),
        _drafter_strategy=lambda: "fsdp2",
        device_module=transfer.module,
        _record_gpu_transfer=lambda *args: None,
    )

    def wait():
        assert opt.state["p"]["exp_avg"] is source
        assert not getattr(opt, "_speco_first_blocking_optimizer_offload_done", False)
        transfer.events.append("wait")

    transfer.module.current_stream = lambda device: SimpleNamespace(synchronize=wait)
    load_base_method()(trainer)
    assert opt._speco_first_blocking_optimizer_offload_done
    assert transfer.events == ["wait"]
    assert torch.equal(opt.state["p"]["exp_avg"], source)


def test_next_adamw_update_matches_without_transfer(monkeypatch):
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    original = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    parameter.grad = torch.ones_like(parameter)
    original.step()
    candidate_parameter = torch.nn.Parameter(parameter.detach().clone())
    candidate = torch.optim.AdamW([candidate_parameter], lr=1e-3, foreach=False)
    candidate.load_state_dict(copy.deepcopy(original.state_dict()))
    # Only metadata is spoofed; underlying copies and AdamW math run on CPU.
    with monkeypatch.context() as patch:
        device_property = torch.Tensor.device
        pointers = {
            v.untyped_storage().data_ptr()
            for state in candidate.state.values()
            for key, v in state.items()
            if torch.is_tensor(v) and key != "step"
        }
        patch.setattr(
            torch.Tensor,
            "device",
            property(
                lambda t: (
                    torch.device("cuda:0")
                    if t.untyped_storage().data_ptr() in pointers
                    else device_property.__get__(t)
                )
            ),
        )
        allocate = torch.empty_like
        patch.setattr(
            torch,
            "empty_like",
            lambda tensor, **kw: allocate(
                tensor, **{k: v for k, v in kw.items() if k != "pin_memory"}
            ),
        )
        pool = OptimizerHostBufferPool()
        assert pool.offload(
            candidate,
            max_bytes=1024,
            device_module=SimpleNamespace(
                current_stream=lambda device: SimpleNamespace(synchronize=lambda: None)
            ),
        )[0]
    parameter.grad = torch.full_like(parameter, 0.25)
    candidate_parameter.grad = torch.full_like(parameter, 0.25)
    original.step()
    candidate.step()
    assert torch.equal(parameter, candidate_parameter)
    for key, value in original.state[parameter].items():
        assert torch.equal(value, candidate.state[candidate_parameter][key])
    assert set(candidate.state_dict()) == {"state", "param_groups"}


def test_changed_shape_evicts_old_storage_and_respects_budget(transfer):
    opt, _ = optimizer(transfer)
    pool = OptimizerHostBufferPool()
    assert pool.offload(opt, max_bytes=32, device_module=transfer.module)[0]
    source = torch.arange(4, dtype=torch.float32)
    transfer.accelerator.add(transfer.pointer(source))
    opt.state["p"]["exp_avg"] = source
    used, stats = pool.offload(opt, max_bytes=16, device_module=transfer.module)
    assert used and stats["allocated_bytes"] == 16
    assert stats["reused_bytes"] == 0 and stats["pool_bytes"] == 16
    assert len(pool.entries) == 1
    assert torch.equal(opt.state["p"]["exp_avg"], source)


def test_wait_failure_never_rebinds_or_consumes_first_marker(transfer):
    opt, source = optimizer(transfer)

    def fail():
        raise RuntimeError("stream wait failed")

    transfer.module.current_stream = lambda device: SimpleNamespace(synchronize=fail)
    trainer = SimpleNamespace(
        optimizer=opt,
        _gpu_optimizer_host_reuse=True,
        _gpu_offload_pool_max_bytes=1024,
        _drafter_strategy=lambda: "fsdp2",
        device_module=transfer.module,
        _record_gpu_transfer=lambda *args: None,
    )
    with pytest.raises(RuntimeError, match="stream wait failed"):
        load_base_method()(trainer)
    assert opt.state["p"]["exp_avg"] is source
    assert not getattr(opt, "_speco_first_blocking_optimizer_offload_done", False)
    assert opt._speco_host_transition_failed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cuda_optimizer_roundtrip_checkpoint_and_next_update(tmp_path):
    parameter = torch.nn.Parameter(torch.arange(8, device="cuda", dtype=torch.float32))
    baseline_parameter = torch.nn.Parameter(parameter.detach().clone())
    opt = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    baseline = torch.optim.AdamW([baseline_parameter], lr=1e-3, foreach=False)
    pool = OptimizerHostBufferPool()
    first_storage = {}
    for index in range(3):
        parameter.grad = torch.full_like(parameter, index + 0.25)
        baseline_parameter.grad = parameter.grad.clone()
        opt.step()
        baseline.step()
        used, stats = pool.offload(opt, max_bytes=1024, device_module=torch.cuda)
        assert used and stats["pinned_bytes"] == 64
        assert stats["allocated_bytes"] == (64 if index == 0 else 0)
        for key in ("exp_avg", "exp_avg_sq"):
            host = opt.state[parameter][key]
            assert torch.equal(host, baseline.state[baseline_parameter][key].cpu())
            if index == 0:
                first_storage[key] = host.data_ptr()
            assert host.data_ptr() == first_storage[key]
        checkpoint = tmp_path / "optimizer.pt"
        torch.save(opt.state_dict(), checkpoint)
        restored = torch.load(checkpoint, weights_only=True)
        assert set(restored) == {"state", "param_groups"}
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(restored["state"][0][key], opt.state[parameter][key])
            opt.state[parameter][key] = opt.state[parameter][key].to(
                "cuda", non_blocking=True
            )
        assert torch.equal(parameter, baseline_parameter)


def _cuda_dtensor_optimizer_check(rank, rendezvous):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=rendezvous,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        full = torch.arange(24, dtype=torch.float32, device="cuda").reshape(6, 4)
        value = distribute_tensor(full, mesh, [Shard(0)])
        opt = SimpleNamespace(state={"p": {"exp_avg": value}})
        pool = OptimizerHostBufferPool()
        for iteration in range(2):
            if iteration:
                opt.state["p"]["exp_avg"] = opt.state["p"]["exp_avg"].to("cuda") + 1
                full += 1
            used, stats = pool.offload(opt, max_bytes=1024, device_module=torch.cuda)
            host = opt.state["p"]["exp_avg"]
            assert used and isinstance(host, DTensor)
            assert host.placements == value.placements
            assert host.device_mesh == value.device_mesh
            assert host.to_local().device.type == "cpu"
            assert torch.equal(host.to("cuda").full_tensor(), full)
            assert (
                stats["reused_bytes"] > 0 if iteration else stats["allocated_bytes"] > 0
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_rank_cuda_optimizer_dtensor_roundtrip(tmp_path):
    torch.multiprocessing.spawn(
        _cuda_dtensor_optimizer_check,
        args=((tmp_path / "optimizer-rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "device,bubble,strategy,enabled",
    [
        ("cuda", True, "fsdp2", True),
        ("npu", False, "fsdp2", True),
        ("cuda", False, "ddp", True),
        ("cuda", False, "fsdp2", False),
    ],
)
def test_pool_guards_keep_bubble_npu_other_strategies_and_rollback_original(
    device,
    bubble,
    strategy,
    enabled,
):
    method = load_base_method()
    method.__globals__["device_name"] = device
    opt = SimpleNamespace(state={"p": {"exp_avg": torch.ones(8)}})
    trainer = SimpleNamespace(
        optimizer=opt,
        _bubble_time_enabled=bubble,
        _gpu_optimizer_host_reuse=enabled,
        _drafter_strategy=lambda: strategy,
        _use_blocking_npu_optimizer_offload=lambda: False,
        _tensor_local_shard=lambda value: value,
    )
    method(trainer)
    assert not hasattr(opt, "_speco_host_buffer_pool")
