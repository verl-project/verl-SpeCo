# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-in resident-probe state parking using Ascend's address-preserving allocator."""

import bisect
import hashlib
import json
import os
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def defer_pool_cache_release(backend):
    """DeepSpeed's cache release must run after use_mem_pool ends allocation capture."""
    requests = []
    targets = []
    seen = set()
    for owner in (backend, getattr(backend, "memory", None)):
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        original = getattr(owner, "empty_cache", None)
        if callable(original):
            targets.append((owner, original))

    def deferred():
        requests.append(traceback.format_stack(limit=8) if not requests else None)

    successful = False
    try:
        for owner, _ in targets:
            owner.empty_cache = deferred
        yield requests
        successful = True
    finally:
        for owner, original in targets:
            owner.empty_cache = original
        if successful and requests:
            # The caller keeps this context outside the allocator's context.
            targets[0][1]()


def available_host_memory(cgroup_root="/sys/fs/cgroup", meminfo="/proc/meminfo"):
    """Support v1/v2 limits; clean file cache can be reclaimed for CPU parking."""
    fields = {}
    for line in Path(meminfo).read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0]) * 1024
    available = fields["MemAvailable"]
    group = Path(cgroup_root)
    if (group / "memory.max").exists():
        maximum = (group / "memory.max").read_text().strip()
        if maximum == "max":
            return available
        usage = int((group / "memory.current").read_text())
        stats = dict(
            line.split() for line in (group / "memory.stat").read_text().splitlines()
        )
        clean_file = int(stats.get("file", 0)) - sum(
            int(stats.get(key, 0)) for key in ("shmem", "file_dirty", "file_writeback")
        )
    elif (group / "memory/memory.limit_in_bytes").exists():
        group = group / "memory"
        maximum = (group / "memory.limit_in_bytes").read_text().strip()
        usage = int((group / "memory.usage_in_bytes").read_text())
        stats = dict(
            line.split() for line in (group / "memory.stat").read_text().splitlines()
        )

        def stat(name):
            return int(stats.get("total_" + name, stats.get(name, 0)))

        clean_file = stat("cache") - stat("shmem") - stat("dirty") - stat("writeback")
    else:
        return available
    return max(0, min(available, int(maximum) - usage + max(0, clean_file)))


def tensor_leaves(value, prefix, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            yield from tensor_leaves(
                item,
                f"{prefix}/{index}:{key if isinstance(key, str) else type(key).__name__}",
                seen,
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{prefix}/{index}", seen)


def trainer_tensors(engine):
    for prefix, model in (
        ("model", engine.model_engine.module),
        ("teacher", engine.teacher),
    ):
        yield from (
            (f"{prefix}/{name}", value) for name, value in model.named_parameters()
        )
        yield from (
            (f"{prefix}/buffer/{name}", value) for name, value in model.named_buffers()
        )
        # Ascend stores this derived router tensor as a plain attribute, not a
        # registered buffer. It must be checked across park/wake just like weights.
        for name, module in model.named_modules():
            cache = getattr(module, "weight_fp32", None)
            if torch.is_tensor(cache):
                yield f"{prefix}/cache/{name}.weight_fp32", cache
    optimizer = engine.model_engine.optimizer
    seen = set()
    depth = 0
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        yield from tensor_leaves(vars(optimizer), f"optimizer/{depth}")
        optimizer = getattr(optimizer, "optimizer", None)
        depth += 1


def sample_state_tensor(value):
    """Copy bounded slices; never use float-derived indices or NPU advanced gather."""
    if getattr(value, "is_meta", False):
        raise ValueError("Meta tensor has no resident data to sample")
    flat = (
        value.detach().reshape(-1)
        if value.is_contiguous()
        else value.detach()[
            tuple([slice(0, 1)] * (value.ndim - 1) + [slice(0, 512)])
        ].reshape(-1)
    )
    size = flat.numel()
    if size <= 512:
        return flat.cpu().contiguous()
    # Python integer arithmetic remains exact beyond float32's 24-bit precision.
    # Narrow views also avoid device index-check kernels converting indices to float.
    return torch.cat(
        [
            flat.narrow(0, (size - 32) * block // 15, 32).cpu().contiguous()
            for block in range(16)
        ]
    )


def state_probe(engine):
    """Bounded probes of values and identities; not a full optimizer byte checksum."""
    digest = hashlib.sha256()
    addresses = {}
    meta_schema = {}
    for name, value in trainer_tensors(engine):
        if value.is_meta:
            parts = name.split("/")
            # DeepSpeed ZeRO-1/2 keeps shape-only padding/layout descriptors in
            # this reviewed attribute. Model weights and Adam moments must be real.
            structural = (
                len(parts) >= 3
                and parts[0] == "optimizer"
                and parts[1].isdigit()
                and parts[2].partition(":")[2] == "round_robin_bit16_meta"
            )
            if not structural:
                raise RuntimeError(f"Unexpected meta tensor in resident data: {name}")
            meta_schema[name] = (
                tuple(value.shape),
                str(value.dtype),
                value.requires_grad,
            )
            continue
        if value.numel() == 0:
            continue
        addresses[name] = (value.data_ptr(), tuple(value.shape), str(value.dtype))
        try:
            sample = sample_state_tensor(value)
        except Exception as error:
            raise RuntimeError(
                f"Cannot sample resident tensor {name}, device={value.device}, "
                f"shape={tuple(value.shape)}, dtype={value.dtype}"
            ) from error
        if sample.is_floating_point() and not bool(torch.isfinite(sample).all()):
            raise RuntimeError(f"Non-finite resident state probe: {name}")
        digest.update(name.encode())
        digest.update(sample.reshape(-1).view(torch.uint8).numpy().tobytes())
    scheduler = engine.model_engine.lr_scheduler
    digest.update(json.dumps(meta_schema, sort_keys=True).encode())
    scheduler_state = scheduler.state_dict() if scheduler is not None else {}
    digest.update(json.dumps(scheduler_state, sort_keys=True, default=str).encode())
    optimizer = engine.model_engine.optimizer
    inner = getattr(optimizer, "optimizer", optimizer)
    # Preserve scalar optimizer step counters as well as moment tensor probes.
    scalar_steps = [
        float(item["step"])
        for item in getattr(inner, "state", {}).values()
        if isinstance(item, dict) and "step" in item
    ]
    return {
        "digest": digest.hexdigest(),
        "addresses": addresses,
        "meta_schema": meta_schema,
        "engine_id": id(engine.model_engine),
        "optimizer_id": id(optimizer),
        "inner_optimizer_id": id(inner),
        "scheduler_id": id(scheduler),
        "optimizer_steps": scalar_steps,
        "global_step": int(engine.model_engine.global_steps),
        "scheduler_state": scheduler_state,
    }


def retain_camem_callbacks(allocator):
    """Keep the exact bound methods borrowed by CaMem's C extension alive.

    Keeping only the allocator instance alive is insufficient: each class
    method lookup creates a temporary bound method. CaMem's init_module stores
    borrowed PyObject pointers and never increments their reference counts.
    Pin the methods on the singleton before its first pool registers them, so
    subsequent allocations and frees still see the same live callables.
    """
    names = ("python_malloc_callback", "python_free_callback")
    callbacks = tuple(getattr(allocator, name, None) for name in names)
    if not all(callable(callback) for callback in callbacks):
        raise RuntimeError("CaMem allocator must expose callable malloc/free callbacks")
    for name, callback in zip(names, callbacks, strict=True):
        setattr(allocator, name, callback)
    return callbacks


def allocation_inventory(engine, ranges, *, device_type="npu"):
    """Audit every live tensor against its retention policy at a quiescent barrier.

    Weights, teacher buffers and Adam/master state must be managed. DeepSpeed's
    root optimizer _global_grad_norm is a derived scalar, recomputed every step;
    it may remain on a communication stream outside the pool. Retain its exact
    value/address (state_probe still checks both), with bounded physical storage.
    No other escaped tensor, including a scalar Adam state, is exempted.
    """
    ranges = sorted(ranges)
    starts = [start for start, _ in ranges]
    managed_count = 0
    resident_statistics = {}
    unsupported = []
    resident_storages = {}
    for name, value in trainer_tensors(engine):
        if value.is_meta or value.device.type != device_type or not value.numel():
            continue
        storage = value.untyped_storage()
        ptr, size = storage.data_ptr(), storage.nbytes()
        index = bisect.bisect_right(starts, ptr) - 1
        if index >= 0 and ptr + size <= ranges[index][1]:
            managed_count += 1
            continue
        details = {
            "address": ptr,
            "storage_bytes": size,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        parts = name.split("/")
        reviewed_statistic = (
            len(parts) == 3
            and parts[:2] == ["optimizer", "0"]
            and parts[2].partition(":")[2] == "_global_grad_norm"
            and value.numel() == 1
            and value.is_floating_point()
            and not value.requires_grad
            and size <= 64
            and bool(torch.isfinite(value.detach().cpu()).all())
        )
        if reviewed_statistic:
            resident_statistics[name] = details
            resident_storages[ptr] = size
        else:
            unsupported.append({"name": name, **details})
    return {
        "managed_bytes": sum(end - start for start, end in ranges),
        "managed_tensor_count": managed_count,
        "resident_statistics": resident_statistics,
        "resident_statistics_bytes": sum(resident_storages.values()),
        "unsupported": unsupported,
    }


class AscendStateParking:
    """Never move Parameter objects with .to(); retain pool storage addresses."""

    tag = "speco_resident_training"

    def __init__(
        self, allocator=None, pool_context=None, synchronize=None, cache_backend=None
    ):
        if allocator is None:
            if "expandable_segments:True" in os.environ.get(
                "PYTORCH_NPU_ALLOC_CONF", ""
            ):
                raise ValueError(
                    "Resident parking requires expandable_segments:False before importing torch"
                )
            from vllm_ascend.device_allocator.camem import (
                CaMemAllocator,
                camem_available,
            )

            if not camem_available:
                raise RuntimeError("Ascend CaMem allocator extension is unavailable")
            allocator = CaMemAllocator.get_instance()
        if allocator.get_current_usage():
            raise RuntimeError(
                "Resident trainer requires an exclusive allocator instance"
            )
        self.allocator = allocator
        # The singleton retains these beyond the lifetime of any pool context
        # or parking helper. Injected CPU allocators need not expose callbacks.
        self.callback_refs = (
            retain_camem_callbacks(allocator)
            if any(
                hasattr(allocator, name)
                for name in ("python_malloc_callback", "python_free_callback")
            )
            else ()
        )
        self.pool_context = pool_context or torch.npu.memory.use_mem_pool
        self.synchronize = synchronize or torch.npu.synchronize
        self.cache_backend = cache_backend or getattr(torch, "npu", None)
        self.deferred_cache_release_calls = 0
        self.cache_release_origin = None
        self.pool = None
        self.state = "new"
        self.probe = None
        self.coverage_inventory = None

    @contextmanager
    def allocation_scope(self):
        if self.state not in {"new", "active"}:
            raise RuntimeError("Cannot allocate in a parked/failed training pool")
        deferred = []
        try:
            with defer_pool_cache_release(self.cache_backend) as deferred:
                if self.pool is None:
                    with self.allocator.use_memory_pool(self.tag):
                        self.pool = self.allocator.allocator_and_pools[self.tag][0]
                        self.state = "active"
                        yield
                else:
                    with (
                        self.allocator.use_allocation_tag(self.tag),
                        self.pool_context(self.pool),
                    ):
                        yield
        except BaseException:
            self.state = "failed"
            raise
        finally:
            self.deferred_cache_release_calls += len(deferred)
            if deferred and self.cache_release_origin is None:
                self.cache_release_origin = deferred[0]

    def check_coverage(self, engine):
        ranges = sorted(
            (ptr, ptr + data.handle[1])
            for ptr, data in self.allocator.pointer_to_data.items()
            if data.tag == self.tag
        )
        if not ranges or len(ranges) != len(self.allocator.pointer_to_data):
            raise RuntimeError("Unexpected/missing allocations in resident state pool")
        self.coverage_inventory = allocation_inventory(engine, ranges)
        if self.coverage_inventory["unsupported"]:
            raise RuntimeError(
                "Live NPU state escaped the managed memory pool: "
                + json.dumps(self.coverage_inventory["unsupported"], sort_keys=True)
            )
        if self.coverage_inventory["resident_statistics_bytes"] > 4096:
            raise RuntimeError("Resident optimizer statistics exceed the 4 KiB budget")
        return self.coverage_inventory["managed_bytes"]

    def park(self, engine):
        if self.state != "active":
            raise RuntimeError("Resident state is not active")
        try:
            self.synchronize()
            managed_bytes = self.check_coverage(engine)
            self.probe = state_probe(engine)
            self.allocator.sleep(offload_tags=(self.tag,))
            self.state = "parked"
            return managed_bytes
        except BaseException:
            self.state = "failed"
            raise

    def wake(self, engine):
        if self.state != "parked":
            raise RuntimeError("Resident state is not parked")
        try:
            self.allocator.wake_up(tags=[self.tag])
            self.synchronize()
            if state_probe(engine) != self.probe:
                raise RuntimeError(
                    "Resident state values, addresses, optimizer or scheduler changed across parking"
                )
            self.state = "active"
            return self.probe
        except BaseException:
            self.state = "failed"
            raise
