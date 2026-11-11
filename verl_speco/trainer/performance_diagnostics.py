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
"""Metadata diagnostics plus explicitly enabled cold-step completion timing."""

import os
import sys
import time


def optimizer_state_metadata(optimizer):
    import torch

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
    torch = sys.modules.get("torch")
    utils = sys.modules.get("torch._dynamo.utils")
    counters = getattr(utils, "counters", {})
    values = {}
    for group in ("frames", "stats", "inductor", "aot_autograd"):
        for name, value in counters.get(group, {}).items():
            if isinstance(value, (int, float)):
                values[f"{group}/{name}"] = value
    return {
        "counters": values,
        "torch_version": str(getattr(torch, "__version__", "unavailable")),
        "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
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


def accelerator_runtime_metadata():
    """Read CUDA metadata only after initialization; never initialize or fence it."""
    import torch

    result = {"cuda_initialized": torch.cuda.is_initialized()}
    if result["cuda_initialized"]:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        props = torch.cuda.get_device_properties(device)
        result["cuda"] = {
            "device": device,
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "unavailable")),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
    return result


def training_stage_clock(trainer):
    """Opt-in CUDA completion fences for cold-step diagnosis, not benchmarking."""
    if getattr(trainer, "_diagnostic_gpu_stage_sync", False):
        trainer.device_module.synchronize()
    return time.perf_counter()


def gpu_first_step_diagnostics_enabled(worker, execution_strategy, coordinated_idle):
    from verl_speco.utils.env import env_flag

    return (
        execution_strategy == "rollout_idle_worker"
        and not coordinated_idle
        and getattr(worker, "device_name", "") == "cuda"
        and getattr(worker.trainer, "optimizer_steps_total", 0) < 2
        and env_flag("VERL_SPECO_DIAGNOSTIC_GPU_FIRST_STEPS")
    )
