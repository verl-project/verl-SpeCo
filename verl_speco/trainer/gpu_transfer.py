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
"""Exact-value host caches; no changes to model precision or optimizer math."""

import logging
import time

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def gpu_transfer_enabled(training_config, legacy_key=None):
    """One public switch; preserve explicit legacy overrides for older configs."""
    if training_config is None:
        return False
    enabled = bool(training_config.get("gpu_transfer_optimizations", True))
    return enabled and bool(
        training_config.get(legacy_key, True) if legacy_key else True
    )


class FrozenHeadHostCache:
    """Keep the existing CPU storage alive while a frozen head is on device."""

    def __init__(self):
        self.host = None
        self.parameter = None
        self.version = None
        self.device_signature = None

    def capture(self, parameter, version):
        self.clear()
        if parameter.device.type != "cpu" or parameter.requires_grad:
            return False
        self.host = parameter.detach()
        self.parameter = parameter
        self.version = version
        return True

    def arm(self, parameter):
        if self.host is not None and parameter is self.parameter:
            self.device_signature = self._signature(parameter)

    @staticmethod
    def _signature(parameter):
        return (
            tuple(parameter.shape),
            parameter.dtype,
            tuple(parameter.stride()),
            parameter.device,
            parameter.data_ptr(),
            parameter._version,
        )

    @torch.no_grad()
    def restore(self, parameter, version):
        reason = None
        if self.host is None or parameter is not self.parameter:
            reason = "missing_or_replaced_parameter"
        elif parameter.requires_grad:
            reason = "head_became_trainable"
        elif version != self.version:
            reason = "version_changed"
        elif self.device_signature != self._signature(parameter):
            reason = "head_mutated_or_layout_changed"
        if reason:
            self.clear()
            return False, reason, 0
        size = self.host.numel() * self.host.element_size()
        # Preserve the Parameter object, as Module.to() normally does. Only
        # restore after the synchronous training/publish calls have returned.
        parameter.data = self.host
        self.clear()
        return True, "none", size

    def clear(self):
        self.host = self.parameter = self.device_signature = None
        self.version = None


def _signature(tensor):
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return (
        type(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tuple(local.shape),
        tuple(local.stride()),
        (id(tensor.device_mesh), tuple(tensor.placements))
        if isinstance(tensor, DTensor)
        else None,
    )


@torch.no_grad()
def load_model_from_host(model, device, *, device_module=None):
    """Stage same-dtype shards, join H2D copies once, then rebuild FSDP views.

    Retain every CPU source until copying completes. GPU buffers are temporary:
    retaining them between training rounds would defeat model offload. Decline
    unsupported models before pooled copies; never retry after copy/apply errors.
    """
    started = time.perf_counter()
    device = torch.device(device)
    stats = {"allocated_bytes": 0, "copied_bytes": 0, "pinned_source_bytes": 0}
    if device.type not in {"cpu", "cuda"} or (
        device.type == "cuda" and device_module is None
    ):
        return False, {
            **stats,
            "fallback_reason": "unsupported_device_or_missing_stream_api",
        }
    try:
        from torch.distributed.fsdp import FSDPModule

        fsdp_type = FSDPModule
        allowed = {torch.nn.Module._apply, FSDPModule._apply}
    except ImportError:
        fsdp_type = ()
        allowed = {torch.nn.Module._apply}
    if any(type(module)._apply not in allowed for module in model.modules()):
        return False, {**stats, "fallback_reason": "custom_module_apply"}
    reshard_started = time.perf_counter()
    for module in model.modules():
        if isinstance(module, fsdp_type):
            module.reshard()
    stats["reshard_s"] = time.perf_counter() - reshard_started
    parameters = dict(model.named_parameters())
    tensors = {
        **parameters,
        **{"@buffer:" + name: value for name, value in model.named_buffers()},
    }
    if not tensors or any(
        type(tensor) not in (torch.Tensor, torch.nn.Parameter, DTensor)
        or tensor.device.type != "cpu"
        or tensor.layout != torch.strided
        or tensor.grad is not None
        or not (
            tensor.to_local() if isinstance(tensor, DTensor) else tensor
        ).is_contiguous()
        or (
            isinstance(tensor, DTensor)
            and tensor.device_mesh.device_type != device.type
        )
        for tensor in tensors.values()
    ):
        return False, {
            **stats,
            "fallback_reason": "unsupported_layout_device_or_live_grad",
        }
    parameter_ids = {name: id(parameter) for name, parameter in parameters.items()}
    allocate_started = time.perf_counter()
    staged = {}
    try:
        for tensor in tensors.values():
            staged[id(tensor)] = torch.empty_like(tensor, device=device)
            local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            size = local.numel() * local.element_size()
            stats["allocated_bytes"] += size
            if local.is_pinned():
                stats["pinned_source_bytes"] += size
    except torch.OutOfMemoryError:
        # No copies or parameter rebinding have happened. Free staging and let
        # the original incremental loader try its smaller temporary footprint.
        staged.clear()
        return False, {**stats, "fallback_reason": "staging_allocation_oom"}
    stats["allocate_s"] = time.perf_counter() - allocate_started
    copy_started = time.perf_counter()
    try:
        for tensor in tensors.values():
            staged[id(tensor)].copy_(
                tensor.detach(), non_blocking=device.type == "cuda"
            )
            local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            stats["copied_bytes"] += local.numel() * local.element_size()
    finally:
        stats["copy_enqueue_s"] = time.perf_counter() - copy_started
        # Also join successfully submitted copies if a later copy failed, before
        # their source/staging references can be released by exception unwinding.
        wait_started = time.perf_counter()
        if device.type == "cuda":
            device_module.current_stream(device).synchronize()
        stats["copy_wait_s"] = time.perf_counter() - wait_started
    apply_started = time.perf_counter()
    model._apply(lambda tensor: staged[id(tensor)])
    staged.clear()
    # FSDP view rebuilding can enqueue device padding copies. Complete those as
    # well before exposing the model to optimizer load or a new training stream.
    if device.type == "cuda":
        device_module.current_stream(device).synchronize()
    if {
        name: id(parameter) for name, parameter in model.named_parameters()
    } != parameter_ids:
        raise RuntimeError(
            "Batched model load changed Parameter identities; refusing stale optimizer references"
        )
    stats.update(
        apply_s=time.perf_counter() - apply_started,
        tensor_count=len(tensors),
        elapsed_s=time.perf_counter() - started,
        fallback_reason="none",
    )
    return True, stats


class ModelHostBufferPool:
    """Reuse CPU shard buffers through the same Module._apply used by .cpu().

    Copy all tensors BEFORE rebinding any parameter. DTensor.copy_ keeps its
    mesh/placement metadata; FSDP2._apply resets sharded views exactly as .cpu().
    Async copies are joined before exposing CPU tensors to those view resets.
    Unsupported layouts return False before pooled copies begin. Copy or apply
    failures are fatal; never fall back after a partially completed transition.
    """

    def __init__(
        self, max_bytes, *, pinned=False, asynchronous=False, pin_budget_bytes=256 << 20
    ):
        self.max_bytes = max(int(max_bytes), 0)
        self.pinned = bool(pinned)
        self.asynchronous = bool(asynchronous and pinned)
        self.pin_budget_bytes = min(max(int(pin_budget_bytes), 0), self.max_bytes)
        self.pin_failed = False
        self.entries = {}

    @torch.no_grad()
    def offload(self, model, *, device_module=None):
        started = time.perf_counter()
        stats = {
            "reused_bytes": 0,
            "allocated_bytes": 0,
            "copied_bytes": 0,
            "pinned_bytes": 0,
            "pin_fallbacks": 0,
            "uncached_bytes": 0,
        }
        try:
            from torch.distributed.fsdp import FSDPModule

            fsdp_type = FSDPModule
            allowed = {torch.nn.Module._apply, FSDPModule._apply}
        except ImportError:
            fsdp_type = ()
            allowed = {torch.nn.Module._apply}
        if any(type(module)._apply not in allowed for module in model.modules()):
            return False, {**stats, "fallback_reason": "custom_module_apply"}
        # FSDPModule._apply() reshards before visiting Parameters. Match that
        # ordering BEFORE capturing tensor identities, including after export
        # paths which may have temporarily registered unsharded Parameters.
        reshard_started = time.perf_counter()
        for module in model.modules():
            if isinstance(module, fsdp_type):
                module.reshard()
        stats["reshard_s"] = time.perf_counter() - reshard_started
        tensors = dict(model.named_parameters())
        parameter_ids = {name: id(parameter) for name, parameter in tensors.items()}
        tensors.update(
            {"@buffer:" + name: value for name, value in model.named_buffers()}
        )
        if not tensors or any(
            type(t) not in (torch.Tensor, torch.nn.Parameter, DTensor)
            or t.layout != torch.strided
            or t.grad is not None
            or not (t.to_local() if isinstance(t, DTensor) else t).is_contiguous()
            for t in tensors.values()
        ):
            return False, {
                **stats,
                "fallback_reason": "unsupported_layout_or_live_grad",
            }
        # Purge removed/replaced entries, keeping the cache bounded by names and
        # local storage bytes. No full_tensor(), gather, or dtype conversion.
        allocate_started = time.perf_counter()
        self.entries = {
            name: entry
            for name, entry in self.entries.items()
            if name in tensors and entry[0] == _signature(tensors[name])
        }
        retained = sum(entry[2] for entry in self.entries.values())
        pinned_retained = sum(
            entry[2]
            for entry in self.entries.values()
            if (
                entry[1].to_local() if isinstance(entry[1], DTensor) else entry[1]
            ).is_pinned()
        )
        staged = {}
        for name, tensor in tensors.items():
            local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            size = local.numel() * local.element_size()
            entry = self.entries.get(name)
            if entry is None:
                pin = (
                    self.pinned
                    and not self.pin_failed
                    and local.device.type == "cuda"
                    and retained + size <= self.max_bytes
                    and pinned_retained + size <= self.pin_budget_bytes
                )
                try:
                    host = torch.empty_like(tensor, device="cpu", pin_memory=pin)
                except RuntimeError:
                    if not pin:
                        raise
                    self.pin_failed = True
                    stats["pin_fallbacks"] += 1
                    host = torch.empty_like(tensor, device="cpu")
                stats["allocated_bytes"] += size
                if retained + size <= self.max_bytes:
                    self.entries[name] = (_signature(tensor), host, size)
                    retained += size
                    if pin and not self.pin_failed:
                        pinned_retained += size
                else:
                    stats["uncached_bytes"] += size
            else:
                host = entry[1]
                stats["reused_bytes"] += size
            staged[id(tensor)] = host
        stats["allocate_s"] = time.perf_counter() - allocate_started
        enqueue_started = time.perf_counter()
        async_devices = set()
        stats["batched_copy_tensors"] = 0
        try:
            for tensor in tensors.values():
                host = staged[id(tensor)]
                local = host.to_local() if isinstance(host, DTensor) else host
                source = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                pin = local.is_pinned()
                stats["copied_bytes"] += local.numel() * local.element_size()
                if pin:
                    stats["pinned_bytes"] += local.numel() * local.element_size()
                # Pageable CUDA copies also support non-blocking submission. Keep
                # every source and destination alive and join the stream once before
                # exposing host storage to FSDP, rather than synchronizing each copy.
                # No additional staging buffers or pinned-memory budget is required.
                asynchronous = (
                    source.device.type == "cuda" and device_module is not None
                )
                if asynchronous:
                    # A failing copy may already have submitted device work.
                    async_devices.add(source.device)
                host.copy_(tensor.detach(), non_blocking=asynchronous)
                if asynchronous:
                    stats["batched_copy_tensors"] += 1
        finally:
            stats["copy_enqueue_s"] = time.perf_counter() - enqueue_started
            wait_started = time.perf_counter()
            # Join submitted copies even when a later copy raises, while all
            # source and destination references are still alive. Failed copies
            # must propagate without rebinding any model parameter.
            for device in async_devices:
                device_module.current_stream(device).synchronize()
            stats["copy_wait_s"] = time.perf_counter() - wait_started
        apply_started = time.perf_counter()
        model._apply(lambda tensor: staged[id(tensor)])
        if {
            name: id(parameter) for name, parameter in model.named_parameters()
        } != parameter_ids:
            raise RuntimeError(
                "Pooled offload changed Parameter identities; refusing stale optimizer references"
            )
        stats.update(
            apply_s=time.perf_counter() - apply_started,
            pool_bytes=retained,
            pool_pinned_bytes=pinned_retained,
            pin_budget_bytes=self.pin_budget_bytes,
            pin_disabled_after_failure=int(self.pin_failed),
            elapsed_s=time.perf_counter() - started,
            fallback_reason="none",
        )
        return True, stats


class OptimizerHostBufferPool:
    """Reuse exact CPU optimizer tensors; rebind only after all copies complete.

    The caller shares its existing CPU-pool budget with model shards. No CUDA
    allocation is retained, and optimizer.state_dict() does not include this
    cache. Unsupported layouts/budgets decline before any state is modified.
    """

    def __init__(self):
        self.entries = {}
        self.pin_failed = False

    @torch.no_grad()
    def offload(self, optimizer, *, max_bytes, device_module):
        started = time.perf_counter()
        stats = {
            "allocated_bytes": 0,
            "reused_bytes": 0,
            "copied_bytes": 0,
            "pinned_bytes": 0,
            "pin_fallbacks": 0,
        }
        tensors = []
        for parameter, state in optimizer.state.items():
            if not isinstance(state, dict):
                continue
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                local = value.to_local() if isinstance(value, DTensor) else value
                if local.device.type == "cpu":
                    continue
                if (
                    type(value) not in (torch.Tensor, DTensor)
                    or local.device.type != "cuda"
                    or value.requires_grad
                    or value.layout != torch.strided
                    or not local.is_contiguous()
                ):
                    self.entries.clear()
                    return False, {
                        **stats,
                        "fallback_reason": "unsupported_state_tensor",
                    }
                tensors.append(
                    (
                        (id(parameter), key),
                        state,
                        key,
                        value,
                        local.numel() * local.element_size(),
                    )
                )
        if not tensors:
            self.entries.clear()
            return False, {**stats, "fallback_reason": "no_accelerator_state"}
        total_bytes = sum(entry[4] for entry in tensors)
        if total_bytes > max(int(max_bytes), 0):
            self.entries.clear()
            return False, {
                **stats,
                "fallback_reason": "shared_pool_budget",
                "required_bytes": total_bytes,
            }
        signatures = {entry[0]: _signature(entry[3]) for entry in tensors}
        self.entries = {
            key: cached
            for key, cached in self.entries.items()
            if key in signatures and cached[0] == signatures[key]
        }
        allocate_started = time.perf_counter()
        staged = {}
        try:
            for cache_key, state, key, value, size in tensors:
                signature = signatures[cache_key]
                cached = self.entries.get(cache_key)
                if cached is not None and cached[0] == signature:
                    host = cached[1]
                    stats["reused_bytes"] += size
                else:
                    try:
                        host = torch.empty_like(
                            value, device="cpu", pin_memory=not self.pin_failed
                        )
                    except RuntimeError:
                        if self.pin_failed:
                            raise
                        self.pin_failed = True
                        stats["pin_fallbacks"] += 1
                        host = torch.empty_like(value, device="cpu")
                    stats["allocated_bytes"] += size
                staged[cache_key] = (signature, host, size)
        except RuntimeError:
            # Nothing has been copied/rebound. Drop scratch and let the original
            # incremental offloader handle unsupported allocation behavior.
            staged.clear()
            self.entries.clear()
            return False, {**stats, "fallback_reason": "host_allocation_failed"}
        stats["allocate_s"] = time.perf_counter() - allocate_started
        copied_devices = set()
        enqueue_started = time.perf_counter()
        try:
            for cache_key, state, key, value, size in tensors:
                host = staged[cache_key][1]
                local = value.to_local() if isinstance(value, DTensor) else value
                host_local = host.to_local() if isinstance(host, DTensor) else host
                # Record before enqueue so partially submitted copies are joined
                # even if a later copy raises.
                copied_devices.add(local.device)
                host.copy_(value.detach(), non_blocking=True)
                stats["copied_bytes"] += size
                if host_local.is_pinned():
                    stats["pinned_bytes"] += size
        finally:
            stats["copy_enqueue_s"] = time.perf_counter() - enqueue_started
            wait_started = time.perf_counter()
            for device in copied_devices:
                device_module.current_stream(device).synchronize()
            stats["copy_wait_s"] = time.perf_counter() - wait_started
        for cache_key, state, key, value, size in tensors:
            state[key] = staged[cache_key][1]
        self.entries = staged
        stats.update(
            pool_bytes=total_bytes,
            pool_budget_bytes=max(int(max_bytes), 0),
            pin_disabled_after_failure=int(self.pin_failed),
            elapsed_s=time.perf_counter() - started,
            fallback_reason="none",
        )
        return True, stats
