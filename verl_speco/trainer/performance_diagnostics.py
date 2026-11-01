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
"""Metadata-only diagnostics: never read accelerator values or synchronize."""

import os
import sys

import torch


def optimizer_state_metadata(optimizer):
    stats = {
        "optimizer_id": id(optimizer),
        "state_entries": len(optimizer.state),
        "tensor_count": 0,
        "cpu_bytes": 0,
        "accelerator_bytes": 0,
        "pinned_cpu_bytes": 0,
        "unknown_tensors": 0,
        "dtype_bytes": {},
        "first_blocking_done": int(
            bool(
                getattr(
                    optimizer, "_speco_first_blocking_optimizer_offload_done", False
                )
            )
        ),
    }
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            stats["tensor_count"] += 1
            try:
                local = (
                    value.to_local()
                    if callable(getattr(value, "to_local", None))
                    else value
                )
                size = local.numel() * local.element_size()
                dtype = str(local.dtype)
                stats["dtype_bytes"][dtype] = stats["dtype_bytes"].get(dtype, 0) + size
                if local.device.type == "cpu":
                    stats["cpu_bytes"] += size
                    if local.is_pinned():
                        stats["pinned_cpu_bytes"] += size
                else:
                    stats["accelerator_bytes"] += size
            except (AttributeError, TypeError, RuntimeError):
                stats["unknown_tensors"] += 1
    return stats


def compilation_metadata():
    # Do not import Dynamo/Inductor just to measure their initialization.
    utils = sys.modules.get("torch._dynamo.utils")
    counters = getattr(utils, "counters", {})
    values = {}
    for group in ("frames", "stats", "inductor", "aot_autograd"):
        for name, value in counters.get(group, {}).items():
            if isinstance(value, (int, float)):
                values[f"{group}/{name}"] = value
    return {
        "counters": values,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "TORCHINDUCTOR_CACHE_DIR": os.getenv(
            "TORCHINDUCTOR_CACHE_DIR", "torch_default"
        ),
        "TRITON_CACHE_DIR": os.getenv("TRITON_CACHE_DIR", "triton_default"),
    }


def allocator_metadata(device_module):
    """Read allocator counters only; never force device completion."""
    try:
        stats = device_module.memory_stats()
    except (AttributeError, RuntimeError):
        return {}
    if not isinstance(stats, dict) or not stats:
        return {}
    return {
        "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
        "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        "allocation_retries": int(stats.get("num_alloc_retries", 0)),
        "ooms": int(stats.get("num_ooms", 0)),
    }
