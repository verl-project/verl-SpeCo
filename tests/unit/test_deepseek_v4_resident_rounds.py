# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real CPU Adam state, injected parking, and the two-round admission protocol."""

import asyncio
import json
from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from verl_speco.probe_deepseek_v4_online_rounds import plan, validate_worker_rounds
from verl_speco.trainer import ascend_state_parking as parking_module
from verl_speco.trainer.ascend_state_parking import (
    AscendStateParking,
    state_probe,
    trainer_tensors,
)
from verl_speco.trainer.deepseek_v4_resident_probe import trainer_config
from verl_speco.trainer.vllm_ascend_online import VllmAscendOnlineMixin


class FakeAllocator:
    """Simulate unmapping by overwriting live tensors, then restoring their bytes."""

    def __init__(self, engine):
        self.engine = engine
        self.pointer_to_data = {
            1: SimpleNamespace(handle=(0, 1, 1, 0), tag=AscendStateParking.tag)
        }
        self.allocator_and_pools = {}
        self.backup = []
        self.corrupt = False

    def get_current_usage(self):
        return 0

    @contextmanager
    def use_memory_pool(self, tag):
        self.allocator_and_pools[tag] = (object(), None)
        yield

    def use_allocation_tag(self, tag):
        return nullcontext()

    def sleep(self, *, offload_tags):
        assert offload_tags == (AscendStateParking.tag,)
        self.backup = [(p, p.detach().clone()) for _, p in trainer_tensors(self.engine)]
        with torch.no_grad():
            for parameter, _ in self.backup:
                parameter.zero_()

    def wake_up(self, *, tags):
        assert tags == [AscendStateParking.tag]
        with torch.no_grad():
            for parameter, value in self.backup:
                parameter.copy_(value)
            if self.corrupt:
                self.engine.model_engine.module.weight.add_(1)


def engine():
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    return SimpleNamespace(
        model_engine=SimpleNamespace(
            module=model, optimizer=optimizer, lr_scheduler=scheduler, global_steps=0
        ),
        teacher=torch.nn.Linear(2, 2, bias=False),
    )


def optimizer_step(engine):
    runtime = engine.model_engine
    runtime.optimizer.zero_grad()
    runtime.module(torch.ones(1, 2)).square().sum().backward()
    runtime.optimizer.step()
    runtime.lr_scheduler.step()
    runtime.global_steps += 1


def test_two_real_adam_steps_keep_moments_objects_addresses_and_scheduler_across_parking():
    runtime = engine()
    allocator = FakeAllocator(runtime)
    parking = AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    with parking.allocation_scope():
        optimizer_step(runtime)
    before = state_probe(runtime)
    parking.park(runtime)
    assert not torch.equal(runtime.model_engine.module.weight, allocator.backup[0][1])
    parking.wake(runtime)
    assert state_probe(runtime) == before
    with parking.allocation_scope():
        optimizer_step(runtime)
    after = state_probe(runtime)
    assert after["optimizer_steps"] == [2.0]
    assert after["digest"] != before["digest"]
    assert after["addresses"] == before["addresses"]
    for key in ("engine_id", "optimizer_id", "inner_optimizer_id", "scheduler_id"):
        assert before[key] == after[key]
    assert after["scheduler_state"]["last_epoch"] == 2
    parking.park(runtime)
    parking.wake(runtime)
    assert state_probe(runtime) == after


def test_corrupt_wake_fails_closed_and_cannot_train():
    runtime = engine()
    allocator = FakeAllocator(runtime)
    parking = AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    with parking.allocation_scope():
        optimizer_step(runtime)
    parking.park(runtime)
    allocator.corrupt = True
    with pytest.raises(RuntimeError, match="changed across parking"):
        parking.wake(runtime)
    assert parking.state == "failed"
    with pytest.raises(RuntimeError, match="parked/failed"), parking.allocation_scope():
        pass


def test_unmanaged_npu_storage_is_rejected_before_sleep(monkeypatch):
    runtime = engine()
    allocator = FakeAllocator(runtime)
    parking = AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    tensor = SimpleNamespace(
        is_meta=False,
        shape=(4,),
        dtype=torch.float32,
        device=SimpleNamespace(type="npu"),
        numel=lambda: 4,
        untyped_storage=lambda: SimpleNamespace(
            data_ptr=lambda: 100, nbytes=lambda: 16
        ),
    )
    monkeypatch.setattr(
        parking_module, "trainer_tensors", lambda _: [("escaped", tensor)]
    )
    with pytest.raises(RuntimeError, match="escaped"):
        parking.check_coverage(runtime)


def test_park_requires_wake_and_exclusive_allocator():
    runtime = engine()
    allocator = FakeAllocator(runtime)
    parking = AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    with pytest.raises(RuntimeError, match="not parked"):
        parking.wake(runtime)
    with parking.allocation_scope():
        optimizer_step(runtime)
    parking.park(runtime)
    with pytest.raises(RuntimeError, match="not active"):
        parking.park(runtime)
    allocator.get_current_usage = lambda: 1
    with pytest.raises(RuntimeError, match="exclusive"):
        AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    parking.wake(runtime)


def worker_report(revision):
    return {
        "status": "passed",
        "workers": [
            {
                "rank": rank,
                "pid": 100 + rank,
                "round": revision,
                "global_step": revision,
                "state": "parked",
                "optimizer_steps": [float(revision)],
                "wake_verified": revision == 2,
                "identities": {"engine_id": rank + 1, "optimizer_id": rank + 100},
            }
            for rank in range(2)
        ],
    }


@pytest.mark.parametrize("change", ["none", "pid", "optimizer", "step", "missing"])
def test_round_report_rejects_restart_reset_or_missing_rank(change):
    first, second = worker_report(1), worker_report(2)
    if change == "pid":
        second["workers"][0]["pid"] += 1
    if change == "optimizer":
        second["workers"][0]["identities"]["optimizer_id"] += 1
    if change == "step":
        second["workers"][0]["optimizer_steps"] = [1.0]
    if change == "missing":
        second["workers"].pop()
    if change == "none":
        validate_worker_rounds([first, second], 2)
    else:
        with pytest.raises(RuntimeError):
            validate_worker_rounds([first, second], 2)


def test_plan_keeps_one_trainer_and_replays_revision_chain_without_large_checkpoints(
    tmp_path,
):
    args = SimpleNamespace(
        output=str(tmp_path),
        model="/model",
        trajectories="/traj",
        tensor_parallel_size=8,
        data_parallel_size=2,
        max_model_len=4096,
        response_tokens=64,
    )
    invocation = plan(args)
    assert "--nproc_per_node=16" in invocation["trainer_command"]
    assert (
        invocation["trainer_environment"]["PYTORCH_NPU_ALLOC_CONF"]
        == "expandable_segments:False"
    )
    assert len(invocation["serving_stages"]) == 3
    assert invocation["serving_stages"][2]["command"].count("--snapshot") == 2
    assert "checkpoint" not in " ".join(invocation["trainer_command"])


@pytest.fixture
def config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 2,
                "hc_mult": 2,
                "vocab_size": 64,
                "num_hidden_layers": 3,
                "dspark_target_layer_ids": [0, 1],
                "dspark_block_size": 2,
            }
        )
    )
    return trainer_config(str(tmp_path), str(tmp_path / "work"), 8, 2)


def test_probe_explicit_groups_do_not_lift_production_dp_guard(config):
    class Base:
        def __init__(self, config, **kwargs):
            self.config = config
            self.training_group_world_size = 8
            self.dp_group_world_size = 2

    class Trainer(VllmAscendOnlineMixin, Base):
        pass

    with pytest.raises(ValueError, match="engine DP=1"):
        Trainer(config)
    session = Trainer(config, _allow_probe_engine_dp=True)
    assert session.cfg.model.vllm_tensor_parallel_size == 8
    assert session.cfg.model.vllm_data_parallel_size == 2


def test_cleanup_resets_collection_cursor_without_discarding_engine():
    session = VllmAscendOnlineMixin.__new__(VllmAscendOnlineMixin)
    session.collected_data = deque([object()])
    session.data_buffer = []
    session._mark_buffer_changed = lambda: None
    session.model = SimpleNamespace(zero_grad=lambda: None)
    session.ascend_engine = object()
    session._online_cursor = 1
    session._online_cursor_step = 0
    previous = session.ascend_engine
    asyncio.run(session.cleanup_training())
    assert session.ascend_engine is previous
    assert session._online_cursor == 0 and session._online_cursor_step is None
    assert not session.collected_data


def test_second_server_commit_uses_revision_two_and_skips_probe_collection(monkeypatch):
    monkeypatch.setenv(
        "VERL_SPECO_DRAFTER_CONFIG", '{"training":{"backend":"vllm_ascend"}}'
    )
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)
    from verl_speco import probe_deepseek_v4_server_update as update
    from verl_speco.integration import deepseek_v4_ipc_probe as ipc
    from verl_speco.integration.vllm_runtime import _SpecoVLLMHttpServerMixin

    class Server(_SpecoVLLMHttpServerMixin):
        def __init__(self):
            self.global_steps = 0
            self._speco_serving_drafter_revision = 1

        async def set_global_steps(self, step):
            self.global_steps = step

        async def abort_all_requests(self):
            return {"aborted_count": 0}

        async def resume_generation(self):
            return None

        async def generate(self, ids, params, request):
            assert params["_verl_skip_drafter_collection"]
            async with self._speco_rollout_admission().request():
                revision = self._speco_serving_drafter_revision
                return SimpleNamespace(
                    token_ids=[3, 4],
                    extra_fields={
                        "global_steps": 0,
                        "_speco_drafter_revision_min": revision,
                        "_speco_drafter_revision_max": revision,
                    },
                )

        async def probe_rpc(self, method, params):
            status = {
                "prepare": "prepared",
                "arm": "armed",
                "receive": "ipc_committed",
                "finalize": "finalized",
            }[params["operation"]]
            return [
                {
                    "tp_rank": 0,
                    "dp_rank": 0,
                    "status": status,
                    "channel": "ipc:///test",
                    "bucket_mb": 1,
                }
            ]

    class Actor:
        def __init__(self, server):
            self.server = server

        def __getattr__(self, name):
            return SimpleNamespace(remote=getattr(self.server, name))

    monkeypatch.setattr(
        ipc, "send_checkpoint_channels", lambda *args: [{"status": "sent"}]
    )
    monkeypatch.setattr(
        update.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1024**3, f_frsize=1)
    )
    report = asyncio.run(
        update.run_probe(
            Actor(Server()),
            SimpleNamespace(
                checkpoint="/unused",
                tensor_parallel_size=1,
                data_parallel_size=1,
                ipc_timeout=1,
                revision=2,
                skip_probe_collection=True,
            ),
            [1, 2],
        )
    )
    assert report["status"] == "passed" and report["drafter_revision"] == 2


@pytest.mark.parametrize("version", [1, 2])
def test_container_memory_budget_reclaims_clean_cache_but_not_shared_or_dirty(
    tmp_path, version
):
    from verl_speco.trainer.ascend_state_parking import available_host_memory

    proc = tmp_path / "meminfo"
    proc.write_text("MemAvailable: 1000 kB\n")
    group = tmp_path / "cgroup"
    group.mkdir()
    if version == 2:
        (group / "memory.max").write_text("1000")
        (group / "memory.current").write_text("900")
        (group / "memory.stat").write_text(
            "file 600\nshmem 100\nfile_dirty 50\nfile_writeback 25\n"
        )
    else:
        memory = group / "memory"
        memory.mkdir()
        (memory / "memory.limit_in_bytes").write_text("1000")
        (memory / "memory.usage_in_bytes").write_text("900")
        (memory / "memory.stat").write_text(
            "total_cache 600\ntotal_shmem 100\ntotal_dirty 50\ntotal_writeback 25\n"
        )
    assert available_host_memory(group, proc) == 525


def test_probe_config_constructs_the_actual_base_trainer_on_cpu(config):
    import ast
    import logging
    from pathlib import Path

    from verl_speco.trainer.data_buffer import DataBuffer

    # Execute the real constructor, stubbing device/group queries only. Importing
    # the entire base module locally would require the native Ascend/verl stack.
    path = Path("verl_speco/trainer/base_trainer.py")
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DrafterBaseTrainer"
    )
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            ast.ClassDef(
                name="ActualBase", bases=[], keywords=[], body=[init], decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    tp, dp = object(), object()
    namespace = {
        "logger": logging.getLogger(__name__),
        "torch": torch,
        "dist": SimpleNamespace(
            is_initialized=lambda: False,
            get_world_size=lambda group: 8 if group is tp else 2,
            get_rank=lambda group: 0,
        ),
        "get_device_id": lambda: 0,
        "get_torch_device": lambda: torch.cpu,
        "device_name": "cpu",
        "deque": deque,
        "DataBuffer": DataBuffer,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 -- trusted repository constructor
    base = namespace["ActualBase"]
    base._create_copy_stream = lambda self: None
    base._is_block_drafter_backend = lambda self: True

    class Trainer(VllmAscendOnlineMixin, base):
        pass

    session = Trainer(
        config,
        world_size=16,
        rollout_dp_rank=0,
        training_device_mesh=None,
        training_process_group=tp,
        data_parallel_process_group=dp,
        _allow_probe_engine_dp=True,
    )
    assert session.step == 1 and session.batch_size == 1
    assert session.ascend_engine is None
    assert session.training_group_world_size == 8 and session.dp_group_world_size == 2
    assert session.config.rollout.drafter.training.collect_interval_steps == 5


@pytest.mark.parametrize("fail_body", [False, True])
def test_cache_release_occurs_only_after_pool_exits_and_always_restores_api(fail_body):
    from verl_speco.trainer.ascend_state_parking import defer_pool_cache_release

    state = {"pool_active": False, "calls": 0}

    def empty_cache():
        assert not state["pool_active"], "captures_underway.empty()"
        state["calls"] += 1

    backend = SimpleNamespace(
        empty_cache=empty_cache, memory=SimpleNamespace(empty_cache=empty_cache)
    )

    @contextmanager
    def native_pool():
        state["pool_active"] = True
        try:
            yield
        finally:
            state["pool_active"] = False

    def body():
        with defer_pool_cache_release(backend) as requests, native_pool():
            backend.empty_cache()
            backend.memory.empty_cache()
            assert state["calls"] == 0
            assert len(requests) == 2
            if fail_body:
                raise ValueError("original failure")

    if fail_body:
        with pytest.raises(ValueError, match="original failure"):
            body()
        assert state["calls"] == 0
    else:
        body()
        assert state["calls"] == 1
    assert (
        backend.empty_cache is empty_cache and backend.memory.empty_cache is empty_cache
    )
    backend.empty_cache()


@pytest.mark.parametrize("size", [66191360, 2**24 + 1, 2**30 + 1, 2**40 + 7])
def test_large_state_sampling_uses_exact_bounded_slices_without_device_indexing(size):
    from verl_speco.trainer.ascend_state_parking import sample_state_tensor

    class LargeTensor:
        def __init__(self):
            self.slices = []

        def detach(self):
            return self

        def is_contiguous(self):
            return True

        def reshape(self, *args):
            return self

        def numel(self):
            return size

        def __getitem__(self, index):
            raise AssertionError("Advanced indexing must not run on NPU")

        def narrow(self, dim, start, count):
            assert (
                dim == 0
                and isinstance(start, int)
                and 0 <= start < size
                and start + count <= size
            )
            self.slices.append((start, count))
            return torch.arange(start, start + count, dtype=torch.int64)

    large = LargeTensor()
    sample = sample_state_tensor(large)
    assert sample.numel() == 512
    assert sample[0].item() == 0 and sample[-1].item() == size - 1
    assert len(large.slices) == 16


def test_state_sampling_handles_small_and_noncontiguous_tensors():
    from verl_speco.trainer.ascend_state_parking import sample_state_tensor

    value = torch.arange(12).reshape(3, 4).t()
    assert not value.is_contiguous()
    assert torch.equal(sample_state_tensor(value), value[:1, :].reshape(-1))
    assert torch.equal(sample_state_tensor(torch.tensor(7)), torch.tensor([7]))


def test_old_float_index_rounding_reproduces_the_reported_large_tensor_bound():
    size = 66191360
    unsafe = torch.linspace(0, size - 1, 512, dtype=torch.float32).long()
    assert (
        unsafe[-1].item() == size
    )  # float32 rounds the endpoint beyond the valid range.


def test_zero_structural_meta_is_checked_as_schema_across_park_and_wake():
    runtime = engine()
    optimizer_step(runtime)
    real_optimizer = runtime.model_engine.optimizer
    runtime.model_engine.optimizer = SimpleNamespace(
        optimizer=real_optimizer,
        round_robin_bit16_meta=[[torch.empty(66191360, device="meta")]],
    )
    before = state_probe(runtime)
    assert len(before["meta_schema"]) == 1
    allocator = FakeAllocator(runtime)
    parking = AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    with parking.allocation_scope():
        pass
    parking.park(runtime)
    parking.wake(runtime)
    assert state_probe(runtime) == before
    parking.park(runtime)
    runtime.model_engine.optimizer.round_robin_bit16_meta[0][0] = torch.empty(
        66191361, device="meta"
    )
    with pytest.raises(RuntimeError, match="changed across parking"):
        parking.wake(runtime)


@pytest.mark.parametrize("location", ["model", "teacher", "adam"])
def test_real_weights_and_adam_moments_must_not_be_meta(location):
    runtime = engine()
    optimizer_step(runtime)
    if location == "model":
        runtime.model_engine.module.weight = torch.nn.Parameter(
            torch.empty(2, 2, device="meta")
        )
    elif location == "teacher":
        runtime.teacher.register_buffer("placeholder", torch.empty(2, device="meta"))
    else:
        state = next(iter(runtime.model_engine.optimizer.state.values()))
        state["exp_avg"] = torch.empty(2, 2, device="meta")
    with pytest.raises(RuntimeError, match="Unexpected meta tensor"):
        state_probe(runtime)


@pytest.mark.parametrize("trainer_base,serving_base", [(61000, 62000), (30000, 31000)])
def test_hccl_roles_use_nonoverlapping_host_and_npu_ranges(trainer_base, serving_base):
    from verl_speco.probe_deepseek_v4_online_rounds import hccl_environments

    environments = hccl_environments(trainer_base, serving_base)
    used = set()
    for environment in environments:
        for value in environment.values():
            low, high = map(int, value.split("-"))
            ports = set(range(low, high + 1))
            assert len(ports) == 100 and not used.intersection(ports)
            used.update(ports)


@pytest.mark.parametrize(
    "trainer_base,serving_base",
    [(61000, 61000), (61000, 61100), (1000, 62000), (65500, 62000)],
)
def test_invalid_or_overlapping_hccl_ranges_fail_before_launch(
    trainer_base, serving_base
):
    from verl_speco.probe_deepseek_v4_online_rounds import hccl_environments

    with pytest.raises(ValueError):
        hccl_environments(trainer_base, serving_base)


def test_serving_subprocess_overrides_inherited_default_hccl_ports(
    tmp_path, monkeypatch
):
    import sys

    from verl_speco.probe_deepseek_v4_training_cycle import run_stage

    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "16666")
    monkeypatch.setenv("HCCL_HOST_SOCKET_PORT_RANGE", "60000-60015")
    args = SimpleNamespace(
        output=str(tmp_path),
        model="/model",
        trajectories="/traj",
        tensor_parallel_size=8,
        data_parallel_size=2,
        max_model_len=4096,
        response_tokens=64,
    )
    invocation = plan(args)
    stage = dict(invocation["serving_stages"][0])
    stage["command"] = [
        sys.executable,
        "-c",
        'import os,json; print(json.dumps({k:os.environ[k] for k in ("HCCL_NPU_SOCKET_PORT_RANGE","HCCL_HOST_SOCKET_PORT_RANGE")}))',
    ]
    run_stage(stage, invocation, tmp_path)
    result = json.loads((tmp_path / "serving_0.log").read_text())
    assert result == {
        "HCCL_NPU_SOCKET_PORT_RANGE": "62000-62099",
        "HCCL_HOST_SOCKET_PORT_RANGE": "62100-62199",
    }
    assert (
        invocation["trainer_environment"]["HCCL_NPU_SOCKET_PORT_RANGE"] == "61000-61099"
    )
    assert (
        invocation["trainer_environment"]["HCCL_HOST_SOCKET_PORT_RANGE"]
        == "61100-61199"
    )


def test_worker_rpc_returns_effective_communication_environment(monkeypatch):
    import sys
    from types import ModuleType

    from verl_speco.integration.deepseek_v4_ipc_probe import (
        StandaloneIPCDraftWorkerExtension,
    )

    distributed = ModuleType("vllm.distributed")
    distributed.get_dp_group = lambda: SimpleNamespace(rank_in_group=1)
    distributed.get_tensor_model_parallel_rank = lambda: 3
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "62000-62099")
    monkeypatch.setenv("HCCL_HOST_SOCKET_PORT_RANGE", "62100-62199")
    report = StandaloneIPCDraftWorkerExtension.speco_standalone_hccl_environment(None)
    assert report["tp_rank"] == 3 and report["dp_rank"] == 1
    assert report["environment"]["HCCL_NPU_SOCKET_PORT_RANGE"] == "62000-62099"


def test_camem_borrowed_callbacks_survive_pool_exit_gc_and_second_round():
    """C init_module borrows bound methods; the instance alone does not own them."""
    import gc
    import weakref

    class BorrowingAllocator(FakeAllocator):
        def python_malloc_callback(self, handle):
            return ("allocated", handle)

        def python_free_callback(self, ptr):
            return (0, 1, ptr, 0)

        @contextmanager
        def use_memory_pool(self, tag):
            # Weak references model the C extension's borrowed pointers without
            # invoking an unsafe dangling PyObject pointer in a CPU unit test.
            malloc, free = self.python_malloc_callback, self.python_free_callback
            self.borrowed = (weakref.ref(malloc), weakref.ref(free))
            with super().use_memory_pool(tag):
                yield

    runtime = engine()
    allocator = BorrowingAllocator(runtime)
    unpinned = weakref.ref(allocator.python_malloc_callback)
    gc.collect()
    assert unpinned() is None
    with allocator.use_memory_pool("unpinned_reproduction"):
        assert all(callable(reference()) for reference in allocator.borrowed)
    gc.collect()
    assert all(reference() is None for reference in allocator.borrowed)

    def reuse_pool(pool):
        gc.collect()
        malloc, free = (reference() for reference in allocator.borrowed)
        assert callable(malloc) and callable(free)
        assert malloc((0, 1, 2, 3)) == ("allocated", (0, 1, 2, 3))
        assert free(2) == (0, 1, 2, 0)
        return nullcontext()

    parking = AscendStateParking(allocator, reuse_pool, lambda: None)
    with parking.allocation_scope():
        optimizer_step(runtime)
    gc.collect()
    assert (
        tuple(reference() for reference in allocator.borrowed) == parking.callback_refs
    )
    assert allocator.python_malloc_callback is parking.callback_refs[0]
    assert allocator.python_free_callback is parking.callback_refs[1]
    parking.park(runtime)
    parking.wake(runtime)
    with parking.allocation_scope():
        optimizer_step(runtime)
    assert state_probe(runtime)["optimizer_steps"] == [2.0]
    # Allocator callbacks can also be invoked when the parking helper is gone.
    borrowed = allocator.borrowed
    del parking
    gc.collect()
    assert all(callable(reference()) for reference in borrowed)


def test_camem_rejects_non_callable_callbacks_before_pool_registration():
    from verl_speco.trainer.ascend_state_parking import retain_camem_callbacks

    allocator = SimpleNamespace(
        python_malloc_callback=(1, 2), python_free_callback=lambda ptr: ptr
    )
    with pytest.raises(RuntimeError, match="callable malloc/free"):
        retain_camem_callbacks(allocator)


def cpu_state_ranges(runtime, excluded=()):
    """Exercise the same storage-ownership audit with actual CPU storages."""
    tensors = list(trainer_tensors(runtime))
    blocked = {
        value.untyped_storage().data_ptr()
        for name, value in tensors
        if not value.is_meta and any(part in name for part in excluded)
    }
    ranges = set()
    for _name, value in tensors:
        if value.is_meta:
            continue
        storage = value.untyped_storage()
        if storage.data_ptr() not in blocked:
            ranges.add((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
    return sorted(ranges)


def test_pool_policy_retains_only_bounded_derived_statistics_and_checks_values():
    from verl_speco.trainer.ascend_state_parking import allocation_inventory

    runtime = engine()
    optimizer_step(runtime)
    runtime.model_engine.optimizer._global_grad_norm = torch.tensor(1.25)
    ranges = cpu_state_ranges(runtime, ("_global_grad_norm",))
    inventory = allocation_inventory(runtime, ranges, device_type="cpu")
    assert inventory["unsupported"] == []
    assert inventory["resident_statistics_bytes"] == 4
    assert len(inventory["resident_statistics"]) == 1
    before = state_probe(runtime)
    runtime.model_engine.optimizer._global_grad_norm.add_(1)
    assert state_probe(runtime)["digest"] != before["digest"]
    assert state_probe(runtime)["addresses"] == before["addresses"]


@pytest.mark.parametrize("location", ["model", "teacher", "adam", "unknown", "step"])
def test_pool_policy_rejects_unmanaged_training_state_even_if_scalar(location):
    from verl_speco.trainer.ascend_state_parking import allocation_inventory

    runtime = engine()
    optimizer_step(runtime)
    optimizer = runtime.model_engine.optimizer
    if location == "model":
        excluded = ("model/weight",)
    elif location == "teacher":
        excluded = ("teacher/weight",)
    elif location == "adam":
        excluded = ("exp_avg",)
    elif location == "step":
        excluded = (":step",)
    else:
        optimizer.unreviewed_statistic = torch.tensor(1.0)
        excluded = ("unreviewed_statistic",)
    inventory = allocation_inventory(
        runtime, cpu_state_ranges(runtime, excluded), device_type="cpu"
    )
    assert inventory["unsupported"]
    assert not inventory["resident_statistics"]


@pytest.mark.parametrize("kind", ["large_storage", "vector", "nan", "autograd"])
def test_derived_statistic_cannot_hide_other_state_or_invalid_data(kind):
    from verl_speco.trainer.ascend_state_parking import allocation_inventory

    runtime = engine()
    optimizer_step(runtime)
    value = {
        "large_storage": torch.ones(1024)[:1],
        "vector": torch.ones(2),
        "nan": torch.tensor(float("nan")),
        "autograd": torch.tensor(1.0, requires_grad=True),
    }[kind]
    runtime.model_engine.optimizer._global_grad_norm = value
    inventory = allocation_inventory(
        runtime, cpu_state_ranges(runtime, ("_global_grad_norm",)), device_type="cpu"
    )
    assert inventory["unsupported"]
    assert not inventory["resident_statistics"]


def test_small_allocator_probe_reuses_packaged_zero_policy_without_target_model():
    from pathlib import Path

    from omegaconf import OmegaConf

    from verl_speco.probe_ascend_state_parking import probe_config

    base = OmegaConf.to_container(
        OmegaConf.load(
            Path(parking_module.__file__).parents[1]
            / "config/deepseek_v4_dspark_backend.yaml"
        ).deepspeed,
        resolve=True,
    )
    small = probe_config()
    assert small["bf16"] == base["bf16"]
    assert small["gradient_clipping"] == base["gradient_clipping"]
    assert small["train_micro_batch_size_per_gpu"] == 1
    assert small["gradient_accumulation_steps"] == 1
    assert "optimizer" not in small  # The probe supplies a real torch AdamW.
    for key, value in base["zero_optimization"].items():
        if key not in {"allgather_bucket_size", "reduce_bucket_size"}:
            assert small["zero_optimization"][key] == value
    assert (
        base["zero_optimization"]["reduce_bucket_size"]
        != small["zero_optimization"]["reduce_bucket_size"]
    )


def test_coverage_failure_records_every_unmanaged_storage_not_just_first():
    from verl_speco.trainer.ascend_state_parking import allocation_inventory

    runtime = engine()
    optimizer_step(runtime)
    inventory = allocation_inventory(runtime, [], device_type="cpu")
    names = [item["name"] for item in inventory["unsupported"]]
    assert any(name.startswith("model/") for name in names)
    assert any(name.startswith("teacher/") for name in names)
    assert any("exp_avg" in name for name in names)
    assert all(item["storage_bytes"] > 0 for item in inventory["unsupported"])


def test_retained_statistic_address_is_still_checked_across_park_and_wake():
    runtime = engine()
    optimizer_step(runtime)
    runtime.model_engine.optimizer._global_grad_norm = torch.tensor(1.25)
    allocator = FakeAllocator(runtime)
    parking = AscendStateParking(allocator, lambda _: nullcontext(), lambda: None)
    with parking.allocation_scope():
        pass
    parking.park(runtime)
    runtime.model_engine.optimizer._global_grad_norm = torch.tensor(1.25)
    with pytest.raises(RuntimeError, match="changed across parking"):
        parking.wake(runtime)


def test_production_lifecycle_validation_is_explicit_and_keeps_original_probe_default(
    tmp_path,
):
    args = SimpleNamespace(
        output=str(tmp_path / "probe"),
        model="/model",
        trajectories="/trajectory",
        tensor_parallel_size=8,
        data_parallel_size=2,
        max_model_len=4096,
        response_tokens=64,
    )
    assert "--production-lifecycle" not in plan(args)["trainer_command"]
    args.production_lifecycle = True
    assert "--production-lifecycle" in plan(args)["trainer_command"]
    cfg = trainer_config("/model", "/work", 8, 2, state_parking="camem")
    assert cfg.rollout.drafter.training.vllm_ascend.state_parking == "camem"


def test_production_lifecycle_report_requires_worker_evidence():
    first, second = worker_report(1), worker_report(2)
    with pytest.raises(RuntimeError, match="production lifecycle hooks"):
        validate_worker_rounds([first, second], 2, production_lifecycle=True)
    for report in (first, second):
        for row in report["workers"]:
            row["state_lifecycle"] = "production_hooks"
    validate_worker_rounds([first, second], 2, production_lifecycle=True)


def test_rollout_release_waits_for_every_rank_without_waking_early():
    from verl_speco.trainer.deepseek_v4_resident_probe import wait_for_rollout_release

    now = [0.0]
    samples = []

    def gather(row):
        samples.append(dict(row))
        # This rank is ready immediately, but its peer releases later.
        return [row, {**row, "free": 0 if len(samples) == 1 else 100}]

    result = wait_for_rollout_release(
        100,
        free_bytes=lambda: 100,
        gather=gather,
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    assert result["observations"] == 2
    assert result["wait_seconds"] == 0.5
    assert all(row["free"] >= row["needed"] for row in result["ranks"])


def test_rollout_release_timeout_is_collective_and_keeps_capacity_requirement():
    from verl_speco.trainer.deepseek_v4_resident_probe import wait_for_rollout_release

    with pytest.raises(TimeoutError, match="device release timed out"):
        wait_for_rollout_release(
            100,
            free_bytes=lambda: 100,
            gather=lambda row: [row, {**row, "free": 99, "expired": True}],
        )


def test_rollout_release_observation_error_is_settled_before_exit():
    from verl_speco.trainer.deepseek_v4_resident_probe import wait_for_rollout_release

    samples = []

    def unavailable():
        raise RuntimeError("device query failed")

    def gather(row):
        samples.append(row)
        return [row]

    with pytest.raises(RuntimeError, match="device query failed"):
        wait_for_rollout_release(100, free_bytes=unavailable, gather=gather)
    assert len(samples) == 1 and samples[0]["free"] is None


def test_shutdown_wake_waits_locally_before_touching_unmapped_storage():
    from verl_speco.trainer.deepseek_v4_resident_probe import wake_parked_for_shutdown

    events = []
    capacity = iter([99, 100])
    parked = SimpleNamespace(
        state="parked",
        coverage_inventory={"managed_bytes": 100},
        wake=lambda engine: events.append(engine),
    )
    result = wake_parked_for_shutdown(
        parked,
        "engine",
        free_bytes=lambda: next(capacity),
        sleep=lambda _: None,
    )
    assert result["observations"] == 2
    assert len(result["ranks"]) == 1
    assert events == ["engine"]


def test_shutdown_wake_never_maps_storage_when_capacity_times_out():
    from verl_speco.trainer.deepseek_v4_resident_probe import wake_parked_for_shutdown

    events = []
    parked = SimpleNamespace(
        state="parked",
        coverage_inventory={"managed_bytes": 100},
        wake=lambda _: events.append("wake"),
    )
    with pytest.raises(TimeoutError):
        wake_parked_for_shutdown(parked, None, free_bytes=lambda: 99, timeout=0)
    assert not events and parked.state == "parked"


def test_shutdown_active_state_needs_no_device_query_or_collective():
    from verl_speco.trainer.deepseek_v4_resident_probe import wake_parked_for_shutdown

    def unexpected():
        raise AssertionError("active storage must not be remapped")

    assert (
        wake_parked_for_shutdown(
            SimpleNamespace(state="active"), None, free_bytes=unexpected
        )
        is None
    )


@pytest.mark.parametrize(
    "command",
    [
        {},
        {"actor_features": "features.pt"},
        {"actor_target_output": "target.pt"},
        {
            "actor_features": "features.pt",
            "actor_target_output": "target.pt",
            "store": "store",
        },
    ],
)
def test_resident_command_rejects_missing_or_ambiguous_source(command):
    from verl_speco.trainer.deepseek_v4_resident_probe import collect_probe_window

    with pytest.raises(ValueError):
        collect_probe_window(None, command, dp_rank=0, expected_revision=0)


@pytest.mark.parametrize(
    "invalid_field", [None, "target_version", "global_step", "scope"]
)
def test_resident_actor_command_loads_paired_exports_before_collection(
    tmp_path, invalid_field
):
    from verl_speco.trainer.deepseek_v4_resident_probe import collect_probe_window

    scope = "full_target_real_verl_fsdp2_oldlogprob_synthetic_short_inputs"
    contract = {"layout": "test_actor_contract"}
    ids = torch.tensor([[1, 2, 3]])
    hidden = torch.randn(1, 3, 4)
    features = {
        "contract": contract,
        "scope": scope,
        "target_version": 0,
        "samples": [{"batch": {"input_ids": ids}, "hidden_states": hidden}],
    }
    target = {
        "contract": contract,
        "scope": scope,
        "global_step": 0,
        "ds4_target_output": {"norm_weight": torch.ones(4)},
    }
    if invalid_field == "target_version":
        features[invalid_field] = 1
    elif invalid_field == "global_step":
        target[invalid_field] = 1
    elif invalid_field == "scope":
        features[invalid_field] = "unverified_scope"
    feature_path, target_path = tmp_path / "features.pt", tmp_path / "target.pt"
    torch.save(features, feature_path)
    torch.save(target, target_path)
    events = []
    session = SimpleNamespace(
        contract=SimpleNamespace(identity=lambda: contract),
        current_rl_step=0,
        dp_group_world_size=1,
        sync_target_output=lambda payload, version: events.append(
            ("target", payload, version)
        ),
        collect_online_data=lambda batch, hs: events.append(("features", batch, hs)),
    )
    command = {"actor_features": feature_path, "actor_target_output": target_path}
    if invalid_field:
        with pytest.raises(ValueError, match="matching full-target"):
            collect_probe_window(session, command, dp_rank=0, expected_revision=0)
        assert events == []
        return
    result = collect_probe_window(session, command, dp_rank=0, expected_revision=0)
    assert result["source"] == "actor_oldlogprob"
    assert [event[0] for event in events] == ["target", "features"]
    assert events[0][2] == 0
    torch.testing.assert_close(events[1][1]["input_ids"], ids)
    torch.testing.assert_close(events[1][2], hidden)
