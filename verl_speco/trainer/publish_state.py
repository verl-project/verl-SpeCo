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
"""Selective FSDP2 hot-publish export; checkpoint export stays independent."""

import fnmatch
import logging
import time
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def export_fsdp2_publish_state(
    model: torch.nn.Module,
    *,
    is_frozen: Callable[[str], bool],
    trains_lm_head: bool,
    patterns: str | Sequence[str] | None,
    dtype: torch.dtype | None,
    cast_before_copy: bool = False,
    diagnostics: dict | None = None,
) -> dict[str, torch.Tensor] | None:
    """Return leader-owned CPU snapshots, or None for unsupported layouts.

    Use DCP's canonical names, as full checkpoint export does, but keep tensors
    sharded until the publish selection is known. Every rank traverses the same
    sorted selection. Do not catch failures after the first collective: falling
    back there could leave ranks executing incompatible collective sequences.
    """
    started = time.perf_counter()
    state = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=False, cpu_offload=False)
    )
    if not state or not any(isinstance(value, DTensor) for value in state.values()):
        return None
    # Restrict the fast path to the normal, world-wide FSDP2 training mesh.
    # Subgroup exporters may use a different publication leader.
    world_ranks = list(range(dist.get_world_size()))
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor):
            return None
        if (
            isinstance(tensor, DTensor)
            and sorted(tensor.device_mesh.mesh.reshape(-1).tolist()) != world_ranks
        ):
            return None
    if isinstance(patterns, str):
        patterns = [patterns]
    selected = {}
    skipped_bytes = 0
    for name, tensor in state.items():
        skip = (
            not torch.is_floating_point(tensor)
            or is_frozen(name)
            or ("lm_head.weight" in name and not trains_lm_head)
            or (
                patterns
                and not any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
            )
        )
        if skip:
            skipped_bytes += tensor.numel() * tensor.element_size()
        else:
            selected[name] = tensor
    leader = dist.get_rank() == 0
    result = {}
    copied_bytes = 0
    source_bytes = 0
    cast_s = 0.0
    host_copy_s = 0.0
    gather_s = 0.0
    async_devices = set()
    for name in sorted(selected):
        gather_started = time.perf_counter()
        tensor = selected[name].detach()
        if isinstance(tensor, DTensor):
            mesh_device = torch.device(tensor.device_mesh.device_type)
            if tensor.to_local().device.type != mesh_device.type:
                tensor = tensor.to(device=mesh_device)
            tensor = tensor.full_tensor()
        gather_s += time.perf_counter() - gather_started
        if leader:
            source_bytes += tensor.numel() * tensor.element_size()
            cast_started = time.perf_counter()
            if cast_before_copy and tensor.device.type == "cuda" and dtype is not None:
                tensor = tensor.to(dtype=dtype)
            cast_s += time.perf_counter() - cast_started
            # copy=True also isolates replicated CPU buffers from later updates.
            copy_started = time.perf_counter()
            target_dtype = dtype or tensor.dtype
            batched = (
                cast_before_copy
                and tensor.device.type == "cuda"
                and tensor.dtype == target_dtype
                and tensor.is_contiguous()
            )
            if batched:
                async_devices.add(tensor.device)
            value = tensor.to(
                device="cpu", dtype=target_dtype, copy=True, non_blocking=batched
            ).contiguous()
            result[name] = value
            copied_bytes += value.numel() * value.element_size()
            host_copy_s += time.perf_counter() - copy_started
    # The CPU snapshot must be complete before serialization or later training.
    wait_started = time.perf_counter()
    for device in async_devices:
        torch.cuda.current_stream(device).synchronize()
    wait_s = time.perf_counter() - wait_started
    stats = {
        "entries": len(result),
        "source_bytes": source_bytes,
        "payload_bytes": copied_bytes,
        "skipped_bytes": skipped_bytes,
        "gather_submit_s": gather_s,
        "cast_submit_s": cast_s,
        "host_copy_submit_s": host_copy_s,
        "copy_wait_s": wait_s,
        "batched_copy": int(bool(async_devices)),
        "snapshot_independent": 1,
        "elapsed_s": time.perf_counter() - started,
    }
    if diagnostics is not None:
        diagnostics.update(stats)
    logger.warning(
        "[drafter publish] selective_fsdp2 rank=%s entries=%s copied_bytes=%s skipped_bytes=%s elapsed_s=%.4f",
        dist.get_rank(),
        len(result),
        copied_bytes,
        skipped_bytes,
        time.perf_counter() - started,
    )
    if leader:
        logger.warning(
            "[drafter publish transfer] cast_before_copy=%s source_bytes=%s payload_bytes=%s "
            "cast_submit_s=%.4f host_copy_s=%.4f snapshot_independent=1 details=%s",
            cast_before_copy,
            source_bytes,
            copied_bytes,
            cast_s,
            host_copy_s,
            stats,
        )
    return result
