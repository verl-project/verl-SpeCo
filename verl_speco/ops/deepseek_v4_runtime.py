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
"""DeepSeek V4 Ascend implementation maintained inside SpeCo.

Source provenance is recorded in docs/deepseek_v4_backend_provenance.md.
"""

from __future__ import annotations

import importlib
import inspect
import math

# Adapted from training_common/modeling/ds4_runtime.py.
import os
import types
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging.version import InvalidVersion, Version
from torch import nn

from verl_speco.ops.ascend_moe_combine import (
    allocate_route_buffer,
    combine_routes,
    expert_matmul,
    native_route_combine_enabled,
)
from verl_speco.ops.deepseek_v4_dsa import explicit_shared_kv_attention_backward
from verl_speco.ops.deepseek_v4_hc import (
    hc_post_explicit_backward_chunk,
    hc_pre_explicit_backward_chunk,
)
from verl_speco.ops.deepseek_v4_moe import (
    plan_budgeted_expert_groups as _plan_budgeted_expert_groups,
)
from verl_speco.ops.deepseek_v4_router import (
    disable_runner_internal_router,
    patch_model_internal_router_training_forward,
)

_VLLM_ASCEND_TRAINING_DIAGNOSTICS: dict[str, int] = {}
_VLLM_COLLECTIVE_AUTOGRAD_REGISTERED = False
_NPU_FORMAT_CAST_AUTOGRAD_REGISTERED = False


_DETERMINISTIC_REDUCE_BUFFERS: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
_DSA_BACKWARD_HOST_METADATA: dict[
    tuple[Any, ...],
    tuple[list[int], list[int], list[int] | None],
] = {}
_DSA_BACKWARD_HOST_METADATA_LIMIT = 256
_FLOAT32_EXACT_INTEGER_LIMIT = 1 << 24
_VLLM_MOE_TRAINING_DP_REPLICATION_ENABLED = False
_DSA_TRAINING_SCATTER_WRAPPER = None
_DSA_TRAINING_SPARSE_ATTENTION_WRAPPER = None
_BF16_QNORM_SOURCES: dict[int, tuple[Any, ...]] = {}
_VLLM_ASCEND_BACKWARD_PROFILE_ENABLED = False
_VLLM_ASCEND_BACKWARD_PROFILE_RECORDS: list[
    tuple[str, float, Any | None, Any | None]
] = []
# This development build reports a pre-0.23 package version while exposing
# the post-0.23 AscendMoERunner source API. Keep the exception exact; all
# other older versions continue through the legacy AscendFusedMoE path.
_VLLM_ASCEND_RUNNER_MOE_VERSION_WHITELIST = frozenset({"0.19.1rc2.dev1302+gdd13a7f29"})
_BudgetedExpertPlan = tuple[
    list[int],
    list[tuple[int, int, int, int]],
    int,
    int,
]


def _increment_training_diagnostic(name: str) -> None:
    _VLLM_ASCEND_TRAINING_DIAGNOSTICS[name] = (
        _VLLM_ASCEND_TRAINING_DIAGNOSTICS.get(name, 0) + 1
    )


def _add_training_diagnostic(name: str, value: int) -> None:
    _VLLM_ASCEND_TRAINING_DIAGNOSTICS[name] = _VLLM_ASCEND_TRAINING_DIAGNOSTICS.get(
        name, 0
    ) + int(value)


def _record_max_training_diagnostic(name: str, value: int) -> None:
    _VLLM_ASCEND_TRAINING_DIAGNOSTICS[name] = max(
        _VLLM_ASCEND_TRAINING_DIAGNOSTICS.get(name, 0),
        int(value),
    )


def get_vllm_ascend_training_diagnostics() -> dict[str, int]:
    return dict(_VLLM_ASCEND_TRAINING_DIAGNOSTICS)


def set_vllm_ascend_backward_profiling(enabled: bool) -> None:
    global _VLLM_ASCEND_BACKWARD_PROFILE_ENABLED
    _VLLM_ASCEND_BACKWARD_PROFILE_ENABLED = bool(enabled)
    _VLLM_ASCEND_BACKWARD_PROFILE_RECORDS.clear()


@contextmanager
def _profile_vllm_ascend_backward_phase(name: str):
    if not _VLLM_ASCEND_BACKWARD_PROFILE_ENABLED:
        yield
        return

    start_event = None
    end_event = None
    npu = getattr(torch, "npu", None)
    event_type = getattr(npu, "Event", None)
    if event_type is not None:
        try:
            start_event = event_type(enable_timing=True)
            end_event = event_type(enable_timing=True)
            start_event.record()
        except (RuntimeError, TypeError):
            start_event = None
            end_event = None
    host_started = perf_counter()
    try:
        yield
    finally:
        host_seconds = perf_counter() - host_started
        if end_event is not None:
            try:
                end_event.record()
            except (RuntimeError, TypeError):
                start_event = None
                end_event = None
        _VLLM_ASCEND_BACKWARD_PROFILE_RECORDS.append(
            (name, host_seconds, start_event, end_event)
        )


def _profile_vllm_ascend_backward(name: str):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            phase_name = name(*args, **kwargs) if callable(name) else name
            with _profile_vllm_ascend_backward_phase(phase_name):
                return function(*args, **kwargs)

        return wrapped

    return decorator


def _linear_backward_profile_name(prefix: str, *, quantized: bool) -> str:
    if ".markov_head." in prefix:
        return "backward_markov_head"
    if ".self_attn.wq" in prefix:
        return "backward_query_projection"
    if ".self_attn." in prefix:
        return "backward_attention_projection"
    return "backward_quant_linear" if quantized else "backward_linear"


def consume_vllm_ascend_backward_profile() -> dict[
    str,
    dict[str, float | int],
]:
    global _VLLM_ASCEND_BACKWARD_PROFILE_ENABLED
    summary: dict[str, dict[str, float | int]] = {}
    for (
        name,
        host_seconds,
        start_event,
        end_event,
    ) in _VLLM_ASCEND_BACKWARD_PROFILE_RECORDS:
        values = summary.setdefault(
            name,
            {"calls": 0, "host_seconds": 0.0, "device_seconds": 0.0},
        )
        values["calls"] = int(values["calls"]) + 1
        values["host_seconds"] = float(values["host_seconds"]) + host_seconds
        if start_event is not None and end_event is not None:
            try:
                elapsed_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
            except (RuntimeError, TypeError):
                elapsed_seconds = 0.0
            values["device_seconds"] = float(values["device_seconds"]) + elapsed_seconds
    _VLLM_ASCEND_BACKWARD_PROFILE_RECORDS.clear()
    _VLLM_ASCEND_BACKWARD_PROFILE_ENABLED = False
    return summary


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _env_nonnegative_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _dsa_host_tensor_key(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        str(tensor.dtype),
    )


def _dsa_backward_host_metadata_key(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    compressed_start_loc: torch.Tensor | None,
) -> tuple[Any, ...]:
    return (
        _dsa_host_tensor_key(query_start_loc),
        _dsa_host_tensor_key(seq_lens),
        _dsa_host_tensor_key(compressed_start_loc),
    )


def _register_dsa_backward_host_metadata(
    built_metadata: Any,
    common_metadata: Any,
    compress_ratio: int,
) -> bool:
    _increment_training_diagnostic("dsa_backward_host_metadata_registration_attempts")

    def metadata_value(container, name):
        if isinstance(container, Mapping):
            return container.get(name)
        return getattr(container, name, None)

    metadata_containers = [
        ("common", common_metadata),
        ("direct", built_metadata),
    ]
    seen_container_ids = {id(common_metadata), id(built_metadata)}
    for attribute, layout in (
        ("prefill", "prefill"),
        ("prefill_metadata", "prefill_metadata"),
        ("req_metadata", "req_metadata"),
    ):
        candidate = metadata_value(built_metadata, attribute)
        if candidate is not None and id(candidate) not in seen_container_ids:
            metadata_containers.append((layout, candidate))
            seen_container_ids.add(id(candidate))

    def metadata_tensors(container):
        query_start_loc = metadata_value(container, "query_start_loc")
        if query_start_loc is None:
            query_start_loc = metadata_value(container, "cu_seqlens_q")
        seq_lens = metadata_value(container, "seq_lens")
        if seq_lens is None:
            seq_lens = metadata_value(container, "seqused_kv")
        return query_start_loc, seq_lens

    def compressed_tensor(container):
        compressed_start_loc = None
        if compress_ratio == 4:
            compressed_start_loc = metadata_value(container, "cu_c4_cmp_seqlen_list")
        elif compress_ratio > 4:
            compressed_start_loc = metadata_value(container, "cu_c128_cmp_seqlen_list")
        if compressed_start_loc is None and compress_ratio > 1:
            compressed_start_loc = metadata_value(container, "cu_cmp_seqlen_list")
        if compressed_start_loc is None and compress_ratio > 1:
            compressed_start_loc = metadata_value(container, "cu_seqlens_cmp_kv")
        return compressed_start_loc

    compatible_containers = []
    compressed_candidates = []
    seen_compressed_keys = set()
    for metadata_layout, metadata_container in metadata_containers:
        query_start_loc, seq_lens = metadata_tensors(metadata_container)
        if isinstance(query_start_loc, torch.Tensor) and isinstance(
            seq_lens,
            torch.Tensor,
        ):
            compatible_containers.append(
                (
                    metadata_layout,
                    metadata_container,
                    query_start_loc,
                    seq_lens,
                )
            )
        candidate = compressed_tensor(metadata_container)
        candidate_key = _dsa_host_tensor_key(candidate)
        if candidate_key is not None and candidate_key not in seen_compressed_keys:
            compressed_candidates.append(candidate)
            seen_compressed_keys.add(candidate_key)

    if not compatible_containers:
        _increment_training_diagnostic(
            "dsa_backward_host_metadata_registration_missing_device_tensors"
        )
        return False

    query_start_loc_cpu = metadata_value(common_metadata, "query_start_loc_cpu")
    seq_lens_cpu = metadata_value(common_metadata, "_seq_lens_cpu")
    if seq_lens_cpu is None:
        seq_lens_cpu = metadata_value(common_metadata, "seq_lens_cpu")
    if not isinstance(query_start_loc_cpu, torch.Tensor) or not isinstance(
        seq_lens_cpu,
        torch.Tensor,
    ):
        _increment_training_diagnostic(
            "dsa_backward_host_metadata_registration_missing_cpu_tensors"
        )
        return False

    num_decodes = int(metadata_value(built_metadata, "num_decodes") or 0)
    num_common_requests = int(seq_lens_cpu.shape[0])
    registered = False
    for (
        metadata_layout,
        metadata_container,
        query_start_loc,
        seq_lens,
    ) in compatible_containers:
        num_metadata_requests = int(seq_lens.shape[0])
        cpu_request_offset = 0
        if (
            metadata_layout in {"prefill", "prefill_metadata"}
            and num_metadata_requests != num_common_requests
        ):
            cpu_request_offset = num_decodes
        query_cpu = query_start_loc_cpu[
            cpu_request_offset : cpu_request_offset + num_metadata_requests + 1
        ]
        seq_cpu = seq_lens_cpu[
            cpu_request_offset : cpu_request_offset + num_metadata_requests
        ]
        if (
            int(query_cpu.numel()) != num_metadata_requests + 1
            or int(seq_cpu.numel()) != num_metadata_requests
        ):
            _increment_training_diagnostic(
                "dsa_backward_host_metadata_registration_shape_mismatches"
            )
            continue
        query_base = int(query_cpu[0].item())
        query_offsets = [int(value) - query_base for value in query_cpu.tolist()]
        kv_lengths = [int(value) for value in seq_cpu.tolist()]

        local_compressed = compressed_tensor(metadata_container)
        aliases = (
            [local_compressed]
            if isinstance(local_compressed, torch.Tensor)
            else compressed_candidates or [None]
        )
        for compressed_start_loc in aliases:
            compressed_offsets = None
            if isinstance(compressed_start_loc, torch.Tensor):
                effective_ratio = max(1, int(compress_ratio))
                compressed_offsets = [0]
                for length in kv_lengths:
                    compressed_offsets.append(
                        compressed_offsets[-1] + length // effective_ratio
                    )

            key = _dsa_backward_host_metadata_key(
                query_start_loc,
                seq_lens,
                compressed_start_loc,
            )
            if (
                key not in _DSA_BACKWARD_HOST_METADATA
                and len(_DSA_BACKWARD_HOST_METADATA)
                >= _DSA_BACKWARD_HOST_METADATA_LIMIT
            ):
                _DSA_BACKWARD_HOST_METADATA.pop(next(iter(_DSA_BACKWARD_HOST_METADATA)))
                _increment_training_diagnostic("dsa_backward_host_metadata_evictions")
            _DSA_BACKWARD_HOST_METADATA[key] = (
                query_offsets,
                kv_lengths,
                compressed_offsets,
            )
            registered = True
        _increment_training_diagnostic(
            f"dsa_backward_host_metadata_{metadata_layout}_registrations"
        )
    if not registered:
        return False
    _increment_training_diagnostic("dsa_backward_host_metadata_registrations")
    return True


def _lookup_dsa_backward_host_metadata(
    op_kwargs: dict[str, Any],
) -> tuple[list[int], list[int], list[int] | None] | None:
    query_start_loc = op_kwargs.get("cu_seqlens_q")
    seq_lens = op_kwargs.get("seqused_kv")
    compressed_start_loc = op_kwargs.get("cu_seqlens_cmp_kv")
    if not isinstance(query_start_loc, torch.Tensor) or not isinstance(
        seq_lens,
        torch.Tensor,
    ):
        return None
    return _DSA_BACKWARD_HOST_METADATA.get(
        _dsa_backward_host_metadata_key(
            query_start_loc,
            seq_lens,
            compressed_start_loc,
        )
    )


def _configure_vllm_moe_training_dp_replication(enabled: bool) -> bool:
    """Keep vLLM MoE weights TP-sharded but replicated across training DP."""
    global _VLLM_MOE_TRAINING_DP_REPLICATION_ENABLED
    _VLLM_MOE_TRAINING_DP_REPLICATION_ENABLED = bool(enabled)
    if not enabled:
        return False

    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
    )

    current_make = FusedMoEParallelConfig.make
    if getattr(current_make, "_speco_deepseek_v4_training_dp_replication_patch", False):
        return True

    original_make = current_make
    signature = inspect.signature(original_make)

    def _training_make(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        dp_size = int(bound.arguments.get("dp_size_", 1))
        parallel_config = bound.arguments.get("vllm_parallel_config")
        if _VLLM_MOE_TRAINING_DP_REPLICATION_ENABLED and dp_size > 1:
            if bool(getattr(parallel_config, "enable_expert_parallel", False)):
                raise RuntimeError(
                    "DS4 training DP replication is incompatible with vLLM expert "
                    "parallelism; set vllm_enable_expert_parallel=false"
                )
            # Upstream flattens TP across DP when EP is disabled. Training needs
            # corresponding TP shards to be true DeepSpeed DP replicas instead.
            bound.arguments["dp_size_"] = 1
        return original_make(*bound.args, **bound.kwargs)

    _training_make._speco_deepseek_v4_training_dp_replication_patch = True
    _training_make._speco_deepseek_v4_original = original_make
    FusedMoEParallelConfig.make = staticmethod(_training_make)
    return True


def _build_dsa_common_structure_tensors(
    *,
    num_reqs: int,
    query_len: int,
    block_size: int,
    num_blocks_per_req: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    query_lens_cpu = torch.full(
        (num_reqs,),
        query_len,
        dtype=torch.int32,
    )
    query_start_loc_cpu = (
        torch.arange(
            num_reqs + 1,
            dtype=torch.int32,
        )
        * query_len
    )
    block_table = torch.arange(
        num_reqs * num_blocks_per_req,
        dtype=torch.int32,
        device=device,
    ).reshape(num_reqs, num_blocks_per_req)
    request_slot_bases = (
        torch.arange(num_reqs, dtype=torch.int64, device=device)
        * num_blocks_per_req
        * block_size
    )
    local_slots = torch.arange(
        query_len,
        dtype=torch.int64,
        device=device,
    )
    slot_mapping = (request_slot_bases.unsqueeze(1) + local_slots.unsqueeze(0)).reshape(
        -1
    )
    return {
        "query_lens_cpu": query_lens_cpu,
        "query_start_loc_cpu": query_start_loc_cpu,
        "query_start_loc": query_start_loc_cpu.to(device),
        "seq_lens_cpu": query_lens_cpu.clone(),
        "seq_lens": query_lens_cpu.to(device),
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "is_prefilling": torch.ones(
            (num_reqs,),
            dtype=torch.bool,
            device=device,
        ),
        "num_computed_tokens_cpu": torch.zeros(
            (num_reqs,),
            dtype=torch.int32,
        ),
    }


def _tensor_parallel_gradient_sum(gradient: torch.Tensor) -> torch.Tensor:
    """Sum a copy-to-TP boundary gradient over the active TP group."""
    try:
        from vllm.distributed import get_tp_group

        tp_group = get_tp_group()
    except (ImportError, AssertionError):
        return gradient
    if int(tp_group.world_size) > 1:
        return tp_group.all_reduce(gradient.contiguous())
    return gradient


class _TensorParallelCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    @_profile_vllm_ascend_backward("backward_tp_collective")
    def backward(ctx, grad_output):
        return _tensor_parallel_gradient_sum(grad_output)


def _register_vllm_collective_autograd() -> None:
    global _VLLM_COLLECTIVE_AUTOGRAD_REGISTERED
    if _VLLM_COLLECTIVE_AUTOGRAD_REGISTERED:
        return

    def setup_collective_context(ctx, inputs, output):
        _, dim, world_size, group_name = inputs
        ctx.dim = dim
        ctx.world_size = world_size
        ctx.group_name = group_name

    def reduce_scatter_backward(ctx, grad_output):
        grad_input = torch.ops.vllm.all_gather(
            grad_output,
            ctx.dim,
            ctx.world_size,
            group_name=ctx.group_name,
        )
        return grad_input, None, None, None

    def all_gather_backward(ctx, grad_output):
        grad_input = torch.ops.vllm.reduce_scatter(
            grad_output,
            ctx.dim,
            ctx.world_size,
            group_name=ctx.group_name,
        )
        return grad_input, None, None, None

    def setup_all_reduce_context(ctx, inputs, output):
        pass

    def all_reduce_backward(ctx, grad_output):
        # This is reduce-from-tensor-parallel, not a sum of independent rank
        # losses. Every TP rank holds the same logical loss, so its derivative
        # through the forward SUM is identity. Copy-to-TP boundaries perform
        # the matching gradient SUM explicitly.
        return grad_output, None

    registrations = (
        ("vllm::reduce_scatter", reduce_scatter_backward),
        ("vllm::all_gather", all_gather_backward),
        ("vllm::all_reduce", all_reduce_backward),
    )
    for op_name, backward in registrations:
        setup_context = (
            setup_all_reduce_context
            if op_name == "vllm::all_reduce"
            else setup_collective_context
        )
        try:
            torch.library.register_autograd(
                op_name,
                backward,
                setup_context=setup_context,
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise RuntimeError(
                    f"Failed to register the {op_name} autograd formula"
                ) from exc
    _VLLM_COLLECTIVE_AUTOGRAD_REGISTERED = True


def _register_npu_format_cast_autograd() -> None:
    global _NPU_FORMAT_CAST_AUTOGRAD_REGISTERED
    if _NPU_FORMAT_CAST_AUTOGRAD_REGISTERED:
        return

    def backward(ctx, grad_output):
        return grad_output, None

    try:
        torch.library.register_autograd("npu::npu_format_cast", backward)
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise RuntimeError(
                "Failed to register the npu_format_cast autograd formula"
            ) from exc
    _NPU_FORMAT_CAST_AUTOGRAD_REGISTERED = True


# ==========================================
# DeepSpeed MoE Classes for DeepSeek-V4
# ==========================================


def _build_vllm_ascend_config(config):
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.usage.usage_lib import UsageContext
    except Exception as exc:
        raise RuntimeError(
            "ds4_layout='vllm_ascend' requires vLLM and vLLM-Ascend to be importable."
        ) from exc

    def _to_plain_container(value):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            return value
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    model_path = getattr(config, "vllm_model_path", None) or getattr(
        config, "_name_or_path", None
    )
    if not model_path:
        raise ValueError("ds4_layout='vllm_ascend' requires config.vllm_model_path")

    tensor_parallel_size = int(getattr(config, "vllm_tensor_parallel_size", 1))
    num_speculative_tokens = int(getattr(config, "num_nextn_predict_layers", 3))
    speculative_config = getattr(config, "vllm_speculative_config", None)
    if speculative_config is None:
        speculative_config = {
            "num_speculative_tokens": num_speculative_tokens,
            "method": "mtp",
        }
    speculative_config = _to_plain_container(speculative_config)

    additional_config = getattr(config, "vllm_additional_config", None)
    if additional_config is None:
        additional_config = {
            "enable_cpu_binding": True,
            "enable_flashcomm1": False,
            "multistream_overlap_shared_expert": False,
        }
    additional_config = _to_plain_container(additional_config)

    compilation_config = getattr(config, "vllm_compilation_config", None)
    if compilation_config is None:
        compilation_config = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [
                1,
                4,
                8,
                12,
                16,
                20,
                24,
                28,
                32,
                36,
                40,
                44,
                48,
                52,
                56,
                60,
                64,
            ],
        }
    compilation_config = _to_plain_container(compilation_config)
    launch_blocking = os.environ.get(
        "ASCEND_LAUNCH_BLOCKING",
        "0",
    ).strip().lower() not in {"", "0", "false", "off", "no"}
    enforce_eager = launch_blocking or bool(
        getattr(config, "vllm_enforce_eager", False)
    )
    if enforce_eager:
        # vLLM rejects ACL graph construction under synchronous NPU debugging.
        # Custom training paths can also explicitly require eager execution to
        # preserve their autograd-visible operator graph.
        compilation_config = {
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [],
        }

    model_loader_extra_config = getattr(config, "vllm_model_loader_extra_config", None)
    if model_loader_extra_config is None:
        model_loader_extra_config = {"enable_multithread_load": True, "num_threads": 16}
    model_loader_extra_config = _to_plain_container(model_loader_extra_config)

    engine_args = EngineArgs(
        model=model_path,
        tokenizer=model_path,
        served_model_name=getattr(config, "vllm_served_model_name", None),
        tokenizer_mode=getattr(config, "vllm_tokenizer_mode", "deepseek_v4"),
        trust_remote_code=True,
        dtype=getattr(config, "vllm_dtype", "auto"),
        kv_cache_dtype=getattr(config, "vllm_cache_dtype", "auto"),
        max_model_len=int(
            os.environ.get(
                "MTP_DS4_VLLM_MAX_MODEL_LEN",
                getattr(config, "vllm_max_model_len", 135168),
            )
        ),
        max_num_batched_tokens=int(
            getattr(config, "vllm_max_num_batched_tokens", 4096)
        ),
        max_num_seqs=int(getattr(config, "vllm_max_num_seqs", 32)),
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=int(getattr(config, "vllm_data_parallel_size", 1)),
        enable_expert_parallel=bool(
            getattr(config, "vllm_enable_expert_parallel", False)
        ),
        quantization=getattr(config, "vllm_quantization", "ascend"),
        block_size=int(getattr(config, "vllm_block_size", 128)),
        enable_chunked_prefill=bool(
            getattr(config, "vllm_enable_chunked_prefill", True)
        ),
        enable_prefix_caching=bool(getattr(config, "vllm_enable_prefix_caching", True)),
        async_scheduling=bool(getattr(config, "vllm_async_scheduling", True)),
        gpu_memory_utilization=float(
            os.environ.get(
                "MTP_DS4_VLLM_GPU_MEMORY_UTILIZATION",
                getattr(config, "vllm_gpu_memory_utilization", 0.70),
            )
        ),
        safetensors_load_strategy=getattr(
            config, "vllm_safetensors_load_strategy", "prefetch"
        ),
        additional_config=additional_config,
        compilation_config=compilation_config,
        model_loader_extra_config=model_loader_extra_config,
        speculative_config=speculative_config,
        enforce_eager=enforce_eager,
    )
    return engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)


def _set_vllm_ascend_env_defaults(config) -> None:
    if not bool(getattr(config, "vllm_set_ascend_env_defaults", True)):
        return
    defaults = {
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "8",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "ACL_OP_INIT_MODE": "1",
        "USE_MULTI_GROUPS_KV_CACHE": "1",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "512",
        "USE_MULTI_BLOCK_POOL": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    # The training adapters implement the standard same-shape TP all-reduce
    # path. FlashComm1 changes this boundary to token reduce-scatter.
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"


def _patch_vllm_ascend_rope_precompute_device() -> None:
    try:
        from vllm.platforms import current_platform
        from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return

    if getattr(ComplexExpRotaryEmbedding, "_speco_deepseek_v4_device_patch", False):
        return

    original = ComplexExpRotaryEmbedding.precompute_freqs_cis

    def patched_precompute_freqs_cis(*args, **kwargs):
        inv_freq = original(*args, **kwargs)
        target_device = current_platform.device_type
        if target_device and inv_freq.device.type != target_device:
            inv_freq = inv_freq.to(target_device)
        return inv_freq

    ComplexExpRotaryEmbedding.precompute_freqs_cis = staticmethod(
        patched_precompute_freqs_cis
    )
    ComplexExpRotaryEmbedding._speco_deepseek_v4_device_patch = True


def _installed_vllm_ascend_version() -> Version:
    override = os.environ.get("MTP_VLLM_ASCEND_VERSION")
    candidates = (("MTP_VLLM_ASCEND_VERSION", override),) if override else ()
    discovered = list(candidates)
    if not override:
        for distribution_name in ("vllm-ascend", "vllm_ascend"):
            try:
                discovered.append(
                    (distribution_name, package_version(distribution_name))
                )
                break
            except PackageNotFoundError:
                continue

    if not discovered:
        raise RuntimeError(
            "Cannot determine the installed vLLM-Ascend version. Install it as "
            "a package (editable installs are supported), or set "
            "MTP_VLLM_ASCEND_VERSION explicitly."
        )

    source, raw_version = discovered[0]
    try:
        return Version(str(raw_version))
    except InvalidVersion as error:
        raise RuntimeError(
            f"Invalid vLLM-Ascend version from {source}: {raw_version!r}"
        ) from error


def _uses_runner_based_vllm_ascend_moe(version: Version) -> bool:
    return (
        version >= Version("0.23.0rc0")
        or str(version).lower() in _VLLM_ASCEND_RUNNER_MOE_VERSION_WHITELIST
    )


def _vllm_ascend_moe_runtime_type() -> tuple[type[nn.Module], bool]:
    """Select the MoE container from the actually imported source."""
    import warnings

    from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module

    installed_version = _installed_vllm_ascend_version()
    expected_runner = _uses_runner_based_vllm_ascend_moe(installed_version)

    runner_type = getattr(fused_moe_module, "AscendMoERunner", None)
    legacy_type = getattr(fused_moe_module, "AscendFusedMoE", None)

    has_runner = isinstance(runner_type, type)
    has_legacy = isinstance(legacy_type, type)

    # Metadata and source files are agreed.
    if expected_runner and has_runner:
        return runner_type, True

    if not expected_runner and has_legacy:
        return legacy_type, False

    # Nightly image: metadata and actual API do not match.
    # We choose the actually available implementation.
    if has_runner:
        warnings.warn(
            f"vLLM-Ascend metadata version {installed_version} suggests the "
            "legacy AscendFusedMoE API, but the imported source only provides "
            "AscendMoERunner. Selecting the runner-based compatibility branch. "
            f"Source: {getattr(fused_moe_module, '__file__', '<unknown>')}",
            RuntimeWarning,
            stacklevel=2,
        )
        return runner_type, True

    if has_legacy:
        warnings.warn(
            f"vLLM-Ascend metadata version {installed_version} suggests the "
            "AscendMoERunner API, but the imported source only provides "
            "AscendFusedMoE. Selecting the legacy compatibility branch. "
            f"Source: {getattr(fused_moe_module, '__file__', '<unknown>')}",
            RuntimeWarning,
            stacklevel=2,
        )
        return legacy_type, False

    raise RuntimeError(
        "The imported vLLM-Ascend source contains neither "
        "AscendMoERunner nor AscendFusedMoE. "
        f"Version: {installed_version}. "
        f"Source: {getattr(fused_moe_module, '__file__', '<unknown>')}"
    )


def _patch_vllm_ascend_model_internal_router_training_forward(
    moe_types: tuple[type[nn.Module], ...] | None = None,
    module_names: tuple[str, ...] = (
        "vllm_ascend.models.deepseek_v4.model",
        "vllm_ascend.models.deepseek_v4",
    ),
) -> bool:
    """Build internal-router logits before entering inference wrappers."""

    from verl_speco.backends.vllm_ascend_compat import split_moe_api

    if moe_types is None and split_moe_api():
        from verl_speco.backends.vllm_ascend_compat import model_symbol
        from verl_speco.ops.ascend_moe_training import patch_split_moe_model

        return patch_split_moe_model(model_symbol("model", "DeepseekV4MoE"))
    if moe_types is None:
        discovered: list[type[nn.Module]] = []
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception:  # noqa: BLE001,S112 -- optional accelerator capability fallback
                continue
            candidate = getattr(module, "DeepseekV4MoE", None)
            if isinstance(candidate, type) and candidate not in discovered:
                discovered.append(candidate)
        moe_types = tuple(discovered)

    return patch_model_internal_router_training_forward(
        moe_types,
        on_patched_call=lambda: _increment_training_diagnostic(
            "moe_model_external_router_training_calls"
        ),
    )


def _vllm_ascend_moe_weights_module(module: nn.Module) -> nn.Module:
    routed_experts = getattr(module, "routed_experts", None)
    return routed_experts if isinstance(routed_experts, nn.Module) else module


def _prepare_vllm_ascend_imports() -> None:
    try:
        import sys

        import vllm_ascend.patch.platform
        import vllm_ascend.patch.worker  # noqa: F401
        from vllm_ascend.utils import adapt_patch
    except Exception:  # noqa: TRY203 -- preserve import failure traceback propagation
        raise

    from verl_speco.backends.vllm_ascend_compat import loaded_model_modules

    loaded_mtp_modules = loaded_model_modules(sys.modules)
    if loaded_mtp_modules:
        raise RuntimeError(
            "vLLM-Ascend DeepSeek V4 modules were imported before the Ascend FusedMoE "
            f"binding was prepared: {loaded_mtp_modules}. Import the MTP wrapper before "
            "importing vllm_ascend.models.deepseek_v4/deepseek_v4_mtp directly."
        )

    adapt_patch(is_global_patch=True)
    adapt_patch(is_global_patch=False)
    moe_type, runner_based = _vllm_ascend_moe_runtime_type()
    if not runner_based:
        # Legacy vLLM exposed FusedMoE as a class. New releases install an
        # AscendMoERunner through adapt_patch and must retain the factory.
        import vllm.model_executor.layers.fused_moe as fused_moe_pkg

        fused_moe_pkg.FusedMoE = moe_type
    _register_vllm_collective_autograd()
    _register_npu_format_cast_autograd()


def _ensure_vllm_ascend_parallel_initialized(vllm_config, config) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    parallel_config = vllm_config.parallel_config
    base_vllm_world_size = int(parallel_config.world_size)
    across_dp_world_size = int(
        getattr(parallel_config, "world_size_across_dp", base_vllm_world_size)
    )
    vllm_world_size = across_dp_world_size
    if torch.distributed.is_initialized():
        existing_world_size = torch.distributed.get_world_size()
        # External-launcher configurations may already include DP in
        # parallel_config.world_size; ordinary configurations expose it only in
        # world_size_across_dp.
        if existing_world_size in (base_vllm_world_size, across_dp_world_size):
            vllm_world_size = existing_world_size
    rank = 0
    local_rank = 0
    if torch.distributed.is_initialized():
        distributed_world_size = torch.distributed.get_world_size()
        if distributed_world_size != vllm_world_size:
            raise RuntimeError(
                "Existing DeepSpeed/HCCL world size does not match the vLLM "
                f"TP*DP topology: torch={distributed_world_size}, "
                f"vllm={vllm_world_size}, "
                f"tp={parallel_config.tensor_parallel_size}, "
                f"dp={parallel_config.data_parallel_size}"
            )
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    else:
        env_world_size = int(os.environ.get("WORLD_SIZE", vllm_world_size))
        if env_world_size != vllm_world_size:
            raise RuntimeError(
                "Launcher world size does not match the vLLM TP*DP topology: "
                f"env={env_world_size}, vllm={vllm_world_size}"
            )
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    data_parallel_size = int(parallel_config.data_parallel_size)
    if data_parallel_size > 1:
        model_parallel_world_size = int(
            parallel_config.tensor_parallel_size
            * parallel_config.pipeline_parallel_size
        )
        dp_rank = rank // model_parallel_world_size
        dp_local_rank = local_rank // model_parallel_world_size
        if not 0 <= dp_rank < data_parallel_size:
            raise RuntimeError(
                "DeepSpeed rank cannot be mapped onto the vLLM DP topology: "
                f"global_rank={rank}, model_parallel_world_size={model_parallel_world_size}, "
                f"dp_rank={dp_rank}, dp_size={data_parallel_size}"
            )
        parallel_config.data_parallel_rank = dp_rank
        parallel_config.data_parallel_rank_local = dp_local_rank
        parallel_config.data_parallel_index = dp_rank
    backend = "hccl" if hasattr(torch, "npu") and torch.npu.is_available() else "gloo"
    init_method = getattr(config, "vllm_distributed_init_method", None)
    if init_method in ("", "auto", "none", "None"):
        init_method = None
    init_method = init_method or os.environ.get("VLLM_DS4_MTP_DIST_INIT")
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        init_method = f"tcp://{master_addr}:{master_port}"
    with torch.no_grad(), set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=vllm_world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
            backend=backend,
        )
        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            getattr(parallel_config, "prefill_context_parallel_size", 1),
            getattr(parallel_config, "decode_context_parallel_size", 1),
        )
        try:
            from vllm_ascend.distributed.parallel_state import (
                init_ascend_model_parallel,
            )
        except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
            return
        init_ascend_model_parallel(parallel_config)


class _NoOpWeightPrefetchMethod:
    MLP_GATE_UP = "gate_up"
    MLP_DOWN = "down"

    def __getattr__(self, name):
        if name.startswith("maybe_prefetch_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def _ensure_vllm_ascend_noop_weight_prefetch() -> None:
    try:
        import vllm_ascend.utils as ascend_utils
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return
    get_weight_prefetch_method = getattr(
        ascend_utils,
        "get_weight_prefetch_method",
        None,
    )
    # Recent vLLM-Ascend releases removed this global prefetch registry. The
    # training path does not enable weight prefetching, so no compatibility
    # object is needed when the legacy getter is absent.
    if get_weight_prefetch_method is None:
        return
    if get_weight_prefetch_method() is None:
        ascend_utils._WEIGHT_PREFETCH_METHOD = _NoOpWeightPrefetchMethod()


def _patch_vllm_ascend_moe_unpermute_backward() -> bool:
    try:
        from vllm_ascend.ops.fused_moe.token_dispatcher import (
            TokenDispatcherWithAllGather,
        )
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(
        TokenDispatcherWithAllGather, "_speco_deepseek_v4_train_unpermute_patch", False
    ):
        return True

    original_token_combine = TokenDispatcherWithAllGather.token_combine

    def _autograd_token_combine(self, hidden_states, combine_metadata, bias=None):
        if (
            bias is not None
            or not torch.is_grad_enabled()
            or not isinstance(hidden_states, torch.Tensor)
        ):
            return original_token_combine(
                self, hidden_states, combine_metadata, bias=bias
            )

        topk_weights = combine_metadata.topk_weights
        expanded_row_idx = combine_metadata.expanded_row_idx
        restore_shape = tuple(combine_metadata.restore_shape)
        if (
            not isinstance(topk_weights, torch.Tensor)
            or not isinstance(expanded_row_idx, torch.Tensor)
            or hidden_states.dim() != 2
            or len(restore_shape) < 2
        ):
            return original_token_combine(
                self, hidden_states, combine_metadata, bias=bias
            )

        flat_weights = topk_weights.reshape(-1).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        expanded_to_sorted = expanded_row_idx.reshape(-1).to(
            device=hidden_states.device, dtype=torch.long
        )
        if flat_weights.numel() != expanded_to_sorted.numel():
            return original_token_combine(
                self, hidden_states, combine_metadata, bias=bias
            )

        flat_num_tokens = int(math.prod(restore_shape[:-1]))
        hidden_dim = int(restore_shape[-1])
        if flat_num_tokens <= 0 or hidden_dim != hidden_states.shape[-1]:
            return original_token_combine(
                self, hidden_states, combine_metadata, bias=bias
            )
        if flat_weights.numel() % flat_num_tokens != 0:
            return original_token_combine(
                self, hidden_states, combine_metadata, bias=bias
            )

        top_k = flat_weights.numel() // flat_num_tokens
        valid = (expanded_to_sorted >= 0) & (
            expanded_to_sorted < hidden_states.shape[0]
        )
        output = hidden_states.new_zeros((flat_num_tokens, hidden_dim))
        if bool(valid.any()):
            token_ids = (
                torch.arange(
                    flat_weights.numel(), device=hidden_states.device, dtype=torch.long
                )
                // top_k
            )
            sorted_pos = expanded_to_sorted[valid]
            weighted_states = hidden_states.index_select(0, sorted_pos) * flat_weights[
                valid
            ].unsqueeze(-1)
            output.index_add_(0, token_ids[valid], weighted_states)
        return output.view(restore_shape)

    TokenDispatcherWithAllGather.token_combine = _autograd_token_combine
    TokenDispatcherWithAllGather._speco_deepseek_v4_train_unpermute_patch = True
    return True


def _patch_vllm_ascend_moe_forward_training_fallback() -> bool:
    from verl_speco.backends.vllm_ascend_compat import split_moe_api

    if split_moe_api():
        return _patch_vllm_ascend_model_internal_router_training_forward()
    try:
        moe_type, runner_based = _vllm_ascend_moe_runtime_type()
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(moe_type, "_speco_deepseek_v4_train_forward_patch", False):
        return True

    original_forward = moe_type.forward
    original_shared_forward = getattr(moe_type, "_forward_shared_experts", None)

    if runner_based:

        def _autograd_forward(self, hidden_states, router_logits, input_ids=None):
            if not torch.is_grad_enabled():
                return original_forward(self, hidden_states, router_logits, input_ids)

            # New vLLM wraps _forward_impl in an inference-oriented custom op.
            # Keep the complete upstream forward orchestration, but call the
            # runner implementation directly so autograd can see the graph.
            original_entry = self._forward_entry

            def direct_entry(
                entry_hidden_states,
                entry_router_logits,
                shared_experts_input,
                entry_input_ids,
                _layer_name,
                _hidden_dim_unpadded,
            ):
                if not self.is_internal_router:
                    _increment_training_diagnostic(
                        "moe_runner_external_router_training_calls"
                    )
                    return self._forward_impl(
                        entry_hidden_states,
                        entry_router_logits,
                        shared_experts_input,
                        entry_input_ids,
                    )

                # AscendMoERunner's serving path recomputes internal-router
                # logits from gate.weight_fp32.  That inference-only copy is
                # detached from the trainable gate.weight, so the router never
                # receives a gradient.  Compute the same FP32 projection from
                # the real parameter and temporarily select the external-logit
                # branch of _forward_impl.
                gate = self.gate
                if gate is None or not isinstance(
                    getattr(gate, "weight", None),
                    torch.Tensor,
                ):
                    raise RuntimeError(
                        "Internal MoE router is enabled but its trainable "
                        "gate.weight is unavailable"
                    )
                differentiable_router_logits = F.linear(
                    entry_hidden_states.float(),
                    gate.weight.float(),
                )
                with disable_runner_internal_router(self):
                    output = self._forward_impl(
                        entry_hidden_states,
                        differentiable_router_logits,
                        shared_experts_input,
                        entry_input_ids,
                    )
                _increment_training_diagnostic(
                    "moe_internal_router_trainable_weight_calls"
                )
                return output

            self._forward_entry = direct_entry
            try:
                return original_forward(self, hidden_states, router_logits, input_ids)
            finally:
                self._forward_entry = original_entry

        if original_shared_forward is not None:

            def _autograd_shared_forward(self, hidden_states, fused_moe_evts):
                if not torch.is_grad_enabled() or self._shared_experts is None:
                    return original_shared_forward(self, hidden_states, fused_moe_evts)
                shared_gate_up = self._shared_experts_part1(hidden_states)
                return self._shared_experts_part2(hidden_states, shared_gate_up)

            moe_type._forward_shared_experts = _autograd_shared_forward

        moe_type.forward = _autograd_forward
        moe_type._speco_deepseek_v4_train_forward_patch = True
        return True

    original_shared_forward_impl = moe_type.shared_forward_impl

    def _autograd_shared_forward(self, hidden_states, fused_moe_evts):
        if not torch.is_grad_enabled() or self._shared_experts is None:
            return original_shared_forward(self, hidden_states, fused_moe_evts)
        shared_gate_up = self._shared_experts_part1(hidden_states)
        return self._shared_experts_part2(hidden_states, shared_gate_up)

    def _autograd_forward(self, hidden_states, router_logits):
        if not torch.is_grad_enabled():
            return original_forward(self, hidden_states, router_logits)

        self.ensure_moe_quant_config_init()
        shared_input = hidden_states if self._shared_experts is not None else None
        return self.runner.forward_impl(
            self,
            hidden_states,
            router_logits,
            shared_input,
        )

    def _autograd_shared_forward_impl(self, hidden_states, router_logits):
        if not torch.is_grad_enabled():
            return original_shared_forward_impl(self, hidden_states, router_logits)
        if self.is_internal_router:
            gate = self.gate
            if gate is None:
                raise RuntimeError("Internal MoE router is enabled but gate is missing")
            router_logits = F.linear(hidden_states.float(), gate.weight.float())
        fused_moe_results = self.forward_impl(
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out
        if self._shared_experts is None:
            return routed_out
        # The grad-enabled shared-expert fallback does not consume inference
        # stream events, so do not route training through the detached event path.
        shared_out = self._forward_shared_experts(hidden_states, None)
        return shared_out, routed_out

    moe_type.forward = _autograd_forward
    moe_type._forward_shared_experts = _autograd_shared_forward
    moe_type.shared_forward_impl = _autograd_shared_forward_impl
    moe_type._speco_deepseek_v4_train_forward_patch = True
    return True


def _patch_vllm_ascend_routed_moe_training_backward() -> bool:
    try:
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts
        from vllm_ascend.quantization.methods.w8a8_dynamic import (
            AscendW8A8DynamicFusedMoEMethod,
        )
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(
        AscendW8A8DynamicFusedMoEMethod,
        "_speco_deepseek_v4_train_backward_patch",
        False,
    ):
        return True

    original_apply = AscendW8A8DynamicFusedMoEMethod.apply

    class _RoutedMoEInputGrad(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            native_output,
            x,
            router_logits,
            topk_ids,
            topk_weights,
            layer,
            scoring_func,
            renormalize,
            routed_scaling_factor,
            activation,
        ):
            ctx.layer = layer
            ctx.scoring_func = scoring_func
            ctx.renormalize = bool(renormalize)
            ctx.routed_scaling_factor = float(routed_scaling_factor)
            ctx.activation = getattr(activation, "value", activation)
            ctx.input_dtype = x.dtype
            ctx.save_for_backward(x, router_logits, topk_ids, topk_weights)
            return native_output.clone()

        @staticmethod
        @_profile_vllm_ascend_backward("backward_routed_moe")
        def backward(ctx, grad_output):
            x, router_logits, topk_ids, topk_weights = ctx.saved_tensors
            layer = ctx.layer
            if ctx.activation not in (None, "silu"):
                raise RuntimeError(
                    "The routed MoE training fallback currently supports only SiLU, "
                    f"got {ctx.activation!r}"
                )

            w13 = getattr(layer, "w13_weight", None)
            w2 = getattr(layer, "w2_weight", None)
            w13_scale = getattr(layer, "w13_weight_scale_fp32", None)
            w2_scale = getattr(layer, "w2_weight_scale", None)
            if any(value is None for value in (w13, w2, w13_scale, w2_scale)):
                raise RuntimeError(
                    "W8A8 routed MoE weights or per-channel scales are missing"
                )

            flat_x = x.reshape(-1, x.shape[-1])
            flat_grad = grad_output.reshape(-1, grad_output.shape[-1]).float()
            ids = topk_ids.reshape(flat_x.shape[0], -1).long()
            weights = topk_weights.reshape_as(ids).float()
            grad_x = torch.zeros_like(flat_x, dtype=torch.float32)
            route_sensitivities = torch.zeros_like(weights, dtype=torch.float32)
            swiglu_limit = float(getattr(layer, "swiglu_limit", 0) or 0)

            flat_ids = ids.reshape(-1)
            token_ids = (
                torch.arange(flat_ids.numel(), device=flat_ids.device, dtype=torch.long)
                // ids.shape[1]
            )
            slot_ids = (
                torch.arange(flat_ids.numel(), device=flat_ids.device, dtype=torch.long)
                % ids.shape[1]
            )

            for expert_id_tensor in torch.unique(flat_ids):
                expert_id = int(expert_id_tensor.item())
                selected = flat_ids == expert_id
                selected_tokens = token_ids[selected]
                selected_slots = slot_ids[selected]
                if selected_tokens.numel() == 0:
                    continue

                expert_x = flat_x.index_select(0, selected_tokens).float()
                expert_grad = flat_grad.index_select(0, selected_tokens)
                expert_weight = weights[selected_tokens, selected_slots].unsqueeze(-1)

                gate_up_weight = w13[expert_id].detach().float()
                gate_up_scale = w13_scale[expert_id].detach().float().reshape(-1)
                if gate_up_weight.shape[0] != flat_x.shape[-1]:
                    gate_up_weight = gate_up_weight.transpose(0, 1)
                gate_up_weight = gate_up_weight * gate_up_scale.unsqueeze(0)

                down_weight = w2[expert_id].detach().float()
                down_scale = w2_scale[expert_id].detach().float().reshape(-1)
                if down_weight.shape[-1] != flat_grad.shape[-1]:
                    down_weight = down_weight.transpose(0, 1)
                down_weight = down_weight * down_scale.unsqueeze(0)

                gate_up = expert_x.matmul(gate_up_weight)
                gate, up = gate_up.chunk(2, dim=-1)
                if swiglu_limit > 0:
                    gate_unclamped = gate
                    up_unclamped = up
                    gate = gate.clamp(max=swiglu_limit)
                    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
                sigmoid_gate = torch.sigmoid(gate)
                silu_gate = gate * sigmoid_gate
                activated = silu_gate * up
                expert_output = activated.matmul(down_weight)

                route_sensitivities[selected_tokens, selected_slots] = (
                    expert_grad * expert_output
                ).sum(dim=-1)

                grad_activated = (expert_grad * expert_weight).matmul(
                    down_weight.transpose(0, 1)
                )
                grad_gate = (
                    grad_activated
                    * up
                    * sigmoid_gate
                    * (1.0 + gate * (1.0 - sigmoid_gate))
                )
                grad_up = grad_activated * silu_gate
                if swiglu_limit > 0:
                    grad_gate = grad_gate * (gate_unclamped <= swiglu_limit)
                    grad_up = grad_up * (
                        (up_unclamped >= -swiglu_limit) & (up_unclamped <= swiglu_limit)
                    )
                grad_gate_up = torch.cat((grad_gate, grad_up), dim=-1)
                expert_grad_x = grad_gate_up.matmul(gate_up_weight.transpose(0, 1))
                grad_x.index_add_(0, selected_tokens, expert_grad_x)

            logits = router_logits.reshape(-1, router_logits.shape[-1]).float()
            if ctx.scoring_func == "softmax":
                all_scores = logits.softmax(dim=-1)
                score_derivative = None
            elif ctx.scoring_func == "sigmoid":
                all_scores = logits.sigmoid()
                score_derivative = all_scores * (1.0 - all_scores)
            elif ctx.scoring_func == "sqrtsoftplus":
                softplus = F.softplus(logits)
                all_scores = softplus.sqrt()
                score_derivative = torch.sigmoid(logits) / (
                    2.0 * all_scores.clamp_min(1e-20)
                )
            else:
                raise RuntimeError(
                    f"Unsupported routed MoE scoring function {ctx.scoring_func!r}"
                )

            selected_scores = all_scores.gather(1, ids)
            if ctx.renormalize:
                score_sum = selected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
                effective_scale = weights.sum(dim=-1, keepdim=True)
                weighted_sensitivity = (route_sensitivities * selected_scores).sum(
                    dim=-1, keepdim=True
                )
                grad_selected_scores = (
                    effective_scale
                    * (route_sensitivities * score_sum - weighted_sensitivity)
                    / score_sum.square()
                )
            else:
                configured_scale = ctx.routed_scaling_factor
                nominal = selected_scores * configured_scale
                effective_scale = (
                    torch.where(
                        nominal.abs() > 1e-20,
                        weights / nominal,
                        torch.ones_like(weights),
                    )
                    * configured_scale
                )
                grad_selected_scores = route_sensitivities * effective_scale

            grad_router = torch.zeros_like(logits)
            grad_router.scatter_add_(1, ids, grad_selected_scores)
            if ctx.scoring_func == "softmax":
                grad_router = all_scores * (
                    grad_router - (grad_router * all_scores).sum(dim=-1, keepdim=True)
                )
            else:
                grad_router = grad_router * score_derivative

            return (
                None,
                grad_x.reshape_as(x).to(ctx.input_dtype),
                grad_router.reshape_as(router_logits).to(router_logits.dtype),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def _autograd_apply(self, layer, x, router_logits, top_k, renormalize, **kwargs):
        if not torch.is_grad_enabled() or not x.is_floating_point():
            return original_apply(
                self,
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                **kwargs,
            )
        if (
            self.dynamic_eplb
            or kwargs.get("expert_map") is not None
            or kwargs.get("log2phy") is not None
        ):
            raise RuntimeError(
                "The routed MoE training fallback supports the TP-sharded, "
                "non-EP layout only; dynamic EPLB and expert remapping are "
                "not supported"
            )
        if getattr(layer, "zero_expert_num", 0):
            raise RuntimeError(
                "The routed MoE training fallback does not support zero experts"
            )
        if getattr(layer, "mix_placement", False):
            raise RuntimeError(
                "The routed MoE training fallback does not support mix placement"
            )

        with torch.no_grad():
            native_result = original_apply(
                self,
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                **kwargs,
            )
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=kwargs.get("use_grouped_topk", False),
                renormalize=renormalize,
                topk_group=kwargs.get("topk_group"),
                num_expert_group=kwargs.get("num_expert_group"),
                custom_routing_function=kwargs.get("custom_routing_function"),
                scoring_func=kwargs.get("scoring_func", "softmax"),
                routed_scaling_factor=kwargs.get("routed_scaling_factor", 1.0),
                e_score_correction_bias=kwargs.get("e_score_correction_bias"),
                num_experts=router_logits.shape[-1],
                tid2eid=kwargs.get("tid2eid"),
            )

        native_result.routed_out = _RoutedMoEInputGrad.apply(
            native_result.routed_out,
            x,
            router_logits,
            topk_ids,
            topk_weights.to(x.dtype),
            layer,
            kwargs.get("scoring_func", "softmax"),
            renormalize,
            kwargs.get("routed_scaling_factor", 1.0),
            kwargs.get("activation", "silu"),
        )
        return native_result

    AscendW8A8DynamicFusedMoEMethod.apply = _autograd_apply
    AscendW8A8DynamicFusedMoEMethod._speco_deepseek_v4_train_backward_patch = True
    return True


class _ConsolidatedRoutedMoE(torch.autograd.Function):
    """Run expert shards in a loop but expose one gradient per full weight."""

    @staticmethod
    def forward(ctx, x, expert_ids, route_weights, w13, w2, swiglu_limit):
        ctx.native_gmm_replay = native_route_combine_enabled()
        native_values = (
            allocate_route_buffer(x, route_weights)
            if native_route_combine_enabled()
            else None
        )
        output = torch.zeros_like(x) if native_values is None else None
        for expert_id in expert_ids.unique().tolist():
            selected = (expert_ids == expert_id).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_ids = selected[:, 0]
            slot_ids = selected[:, 1]
            expert_x = x.index_select(0, token_ids)
            gate_up = expert_matmul(expert_x, w13, expert_id)
            activated = _routed_moe_swiglu(gate_up, swiglu_limit)
            del gate_up
            expert_output = expert_matmul(activated, w2, expert_id)
            if native_values is not None:
                routes = token_ids * route_weights.shape[1] + slot_ids
                native_values.index_copy_(0, routes, expert_output)
            else:
                expert_output = expert_output * route_weights[token_ids, slot_ids, None]
                output.index_add_(0, token_ids, expert_output)

        if native_values is not None:
            output = combine_routes(
                native_values,
                torch.arange(
                    route_weights.numel(), device=expert_ids.device, dtype=torch.long
                ),
                route_weights,
            )

        ctx.save_for_backward(x, expert_ids, route_weights, w13, w2)
        ctx.swiglu_limit = swiglu_limit
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, expert_ids, route_weights, w13, w2 = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        needs = ctx.needs_input_grad
        grad_x = torch.zeros_like(x) if needs[0] else None
        grad_route_weights = torch.empty_like(route_weights) if needs[2] else None
        grad_w13 = torch.zeros_like(w13) if needs[3] else None
        grad_w2 = torch.zeros_like(w2) if needs[4] else None

        for expert_id in expert_ids.unique().tolist():
            selected = (expert_ids == expert_id).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_ids = selected[:, 0]
            slot_ids = selected[:, 1]
            expert_x = x.index_select(0, token_ids)
            gate_up = expert_matmul(
                expert_x, w13, expert_id, native=ctx.native_gmm_replay
            )
            gate, up = gate_up.chunk(2, dim=-1)
            if swiglu_limit > 0:
                unclamped_gate = gate
                unclamped_up = up
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

            sigmoid_gate = torch.sigmoid(gate)
            silu_gate = gate * sigmoid_gate
            activated = silu_gate * up
            expert_output = expert_matmul(
                activated, w2, expert_id, native=ctx.native_gmm_replay
            )
            selected_grad_output = grad_output.index_select(0, token_ids)
            selected_weights = route_weights[token_ids, slot_ids]
            grad_expert_output = selected_grad_output * selected_weights[:, None]

            if grad_route_weights is not None:
                grad_route_weights[token_ids, slot_ids] = (
                    selected_grad_output * expert_output
                ).sum(dim=-1)
            if grad_w2 is not None:
                grad_w2[expert_id].add_(
                    activated.transpose(0, 1).matmul(grad_expert_output)
                )

            grad_activated = grad_expert_output.matmul(w2[expert_id].transpose(0, 1))
            grad_gate = (
                grad_activated * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
            )
            grad_up = grad_activated * silu_gate
            if swiglu_limit > 0:
                grad_gate = grad_gate * (unclamped_gate <= swiglu_limit)
                grad_up = grad_up * (
                    (unclamped_up >= -swiglu_limit) & (unclamped_up <= swiglu_limit)
                )
            grad_gate_up = torch.cat((grad_gate, grad_up), dim=-1)

            if grad_w13 is not None:
                grad_w13[expert_id].add_(expert_x.transpose(0, 1).matmul(grad_gate_up))
            if grad_x is not None:
                grad_expert_x = grad_gate_up.matmul(w13[expert_id].transpose(0, 1))
                grad_x.index_add_(0, token_ids, grad_expert_x)

        if grad_x is not None:
            grad_x = _tensor_parallel_gradient_sum(grad_x)
        if grad_route_weights is not None:
            grad_route_weights = _tensor_parallel_gradient_sum(grad_route_weights)
        _increment_training_diagnostic("tp_routed_moe_grad_sum_calls")
        _increment_training_diagnostic("bf16_routed_moe_consolidated_backward_calls")
        return grad_x, None, grad_route_weights, grad_w13, grad_w2, None


def _prepare_grouped_moe_routes(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the expert-major route plan once for forward and backward."""
    flat_expert_ids = expert_ids.reshape(-1).long()
    flat_route_ids = torch.arange(
        flat_expert_ids.numel(),
        dtype=torch.long,
        device=expert_ids.device,
    )
    sort_keys = flat_expert_ids * flat_expert_ids.numel() + flat_route_ids
    permutation = _argsort_grouped_moe_integer_keys(
        sort_keys,
        max_key_exclusive=num_experts * flat_expert_ids.numel(),
    )
    sorted_route_ids = flat_route_ids.index_select(0, permutation)
    slots_per_token = int(expert_ids.shape[-1])
    sorted_token_ids = torch.div(
        sorted_route_ids,
        slots_per_token,
        rounding_mode="floor",
    )
    group_counts = torch.zeros(
        num_experts,
        dtype=torch.int64,
        device=expert_ids.device,
    )
    group_counts.scatter_add_(
        0,
        flat_expert_ids,
        torch.ones_like(flat_expert_ids, dtype=torch.int64),
    )
    # Route outputs are accumulated in expert-id order to preserve the audited
    # deterministic reduction.  Cache this permutation instead of sorting the
    # same [num_tokens, top_k] tensor again in backward.
    expert_order = _argsort_grouped_moe_integer_keys(
        expert_ids.long(),
        dim=-1,
        max_key_exclusive=num_experts,
    )
    return sorted_route_ids, sorted_token_ids, group_counts, expert_order


def _argsort_grouped_moe_integer_keys(
    values: torch.Tensor,
    *,
    dim: int = -1,
    max_key_exclusive: int,
) -> torch.Tensor:
    """Use AiCore FP32 argsort only while every integer key stays exact."""
    use_fp32 = (
        values.device.type == "npu"
        and _env_flag("MTP_DS4_FP32_ROUTE_ARGSORT", True)
        and _grouped_moe_keys_fit_float32(max_key_exclusive)
    )
    if use_fp32:
        _increment_training_diagnostic("npu_fp32_route_argsort_calls")
        return torch.argsort(values.float(), dim=dim)
    if values.device.type == "npu":
        _increment_training_diagnostic("npu_integer_route_argsort_fallback_calls")
    return torch.argsort(values, dim=dim)


def _grouped_moe_keys_fit_float32(max_key_exclusive: int) -> bool:
    return 0 <= max_key_exclusive <= _FLOAT32_EXACT_INTEGER_LIMIT


def _npu_grouped_moe_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_counts: torch.Tensor,
) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError("DS4 grouped routed-MoE replay requires an NPU tensor")
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("DS4 grouped routed-MoE replay requires torch_npu") from exc
    if not hasattr(torch_npu, "npu_grouped_matmul"):
        raise RuntimeError(
            "The installed torch_npu does not provide npu_grouped_matmul"
        )
    return torch_npu.npu_grouped_matmul(
        x=[x.contiguous()],
        weight=[weight],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_counts,
    )[0]


def _npu_grouped_moe_weight_grad(
    activation: torch.Tensor,
    output_grad: torch.Tensor,
    group_counts: torch.Tensor,
) -> torch.Tensor:
    """Compute per-expert activation.T @ output_grad without host routing."""
    if activation.device.type != "npu":
        raise RuntimeError("DS4 grouped routed-MoE backward requires NPU tensors")
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "DS4 grouped routed-MoE backward requires torch_npu"
        ) from exc
    if not hasattr(torch_npu, "npu_grouped_matmul"):
        raise RuntimeError(
            "The installed torch_npu does not provide npu_grouped_matmul"
        )
    return torch_npu.npu_grouped_matmul(
        x=[activation.transpose(0, 1)],
        weight=[output_grad],
        split_item=2,
        group_list_type=1,
        group_type=2,
        group_list=group_counts,
    )[0]


def _routed_moe_swiglu(
    gate_up: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Use serving SwiGLU with non-mutating clamps under ordinary autograd."""
    fused_requested = _env_flag("MTP_DS4_FUSED_ROUTED_MOE_SWIGLU", True)
    _record_max_training_diagnostic(
        "bf16_routed_moe_swiglu_limit",
        int(swiglu_limit),
    )
    # Shared experts follow AscendSiluAndMulWithClamp: clamp, concatenate,
    # then npu_swiglu. Retain the custom-forward memory optimization without grad.
    needs_autograd = torch.is_grad_enabled() and gate_up.requires_grad
    if gate_up.device.type == "npu" and fused_requested:
        try:
            import torch_npu
        except ImportError:
            torch_npu = None
        if torch_npu is not None and hasattr(torch_npu, "npu_swiglu"):
            if swiglu_limit > 0:
                gate, up = gate_up.chunk(2, dim=-1)
                if needs_autograd:
                    gate = gate.clamp(max=swiglu_limit)
                    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
                    gate_up = torch.cat([gate, up], dim=-1)
                else:
                    gate.clamp_(max=swiglu_limit)
                    up.clamp_(min=-swiglu_limit, max=swiglu_limit)
            _increment_training_diagnostic("bf16_routed_moe_fused_swiglu_calls")
            return torch_npu.npu_swiglu(gate_up)
        _increment_training_diagnostic("bf16_routed_moe_fused_swiglu_fallback_calls")

    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def _combine_grouped_routes_fixed_order(
    route_values: torch.Tensor,
    sorted_route_ids: torch.Tensor,
    expert_order: torch.Tensor,
) -> torch.Tensor:
    """Stream token routes into a fixed expert-order reduction."""
    num_tokens = int(expert_order.shape[0])
    routes_per_token = int(expert_order.shape[-1])
    num_routes = int(sorted_route_ids.numel())

    # Map each original token-slot route to its expert-major row. Selecting one
    # route position at a time avoids materializing two [tokens, topk, hidden]
    # tensors while preserving the exact expert-order accumulation sequence.
    inverse_route_rows = torch.empty_like(sorted_route_ids)
    sorted_rows = torch.arange(
        num_routes,
        dtype=sorted_route_ids.dtype,
        device=sorted_route_ids.device,
    )
    inverse_route_rows.index_copy_(0, sorted_route_ids, sorted_rows)
    token_route_offsets = torch.arange(
        num_tokens,
        dtype=sorted_route_ids.dtype,
        device=sorted_route_ids.device,
    ).mul_(routes_per_token)

    output = route_values.new_zeros((num_tokens, route_values.shape[-1]))
    for route_index in range(routes_per_token):
        flat_route_ids = token_route_offsets + expert_order[:, route_index].to(
            dtype=sorted_route_ids.dtype
        )
        route_rows = inverse_route_rows.index_select(0, flat_route_ids)
        output.add_(route_values.index_select(0, route_rows))
    _increment_training_diagnostic("bf16_routed_moe_streaming_route_combine_calls")
    return output


def _combine_grouped_routes_token_unpermute(
    route_values: torch.Tensor,
    sorted_route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    force: bool = False,
) -> torch.Tensor | None:
    """Use Ascend's fused route weighting and token combine when available."""
    if route_values.device.type != "npu" or not (
        force or _env_flag("MTP_DS4_FUSED_ROUTED_MOE_TOKEN_UNPERMUTE")
    ):
        return None
    try:
        import torch_npu
    except ImportError:
        torch_npu = None
    if torch_npu is None or not hasattr(torch_npu, "npu_moe_token_unpermute"):
        _increment_training_diagnostic("bf16_routed_moe_token_unpermute_fallback_calls")
        return None

    expanded_row_idx = torch.empty(
        (sorted_route_ids.numel(),),
        dtype=torch.int32,
        device=sorted_route_ids.device,
    )
    expanded_row_idx.index_copy_(
        0,
        sorted_route_ids,
        torch.arange(
            sorted_route_ids.numel(),
            dtype=torch.int32,
            device=sorted_route_ids.device,
        ),
    )
    output = torch_npu.npu_moe_token_unpermute(
        permuted_tokens=route_values,
        sorted_indices=expanded_row_idx,
        probs=route_weights,
    )
    expected_shape = (int(route_weights.shape[0]), int(route_values.shape[-1]))
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "npu_moe_token_unpermute returned an unexpected routed-MoE "
            f"shape: expected={expected_shape}, actual={tuple(output.shape)}"
        )
    _increment_training_diagnostic("bf16_routed_moe_token_unpermute_calls")
    return output


def _budgeted_expert_groups(
    group_counts_host: list[int],
) -> tuple[list[tuple[int, int, int, int]], int, int]:
    route_budget = os.environ.get(
        "MTP_DS4_GROUPED_ROUTED_MOE_ROUTE_BUDGET",
        "auto",
    )
    max_experts_per_group = int(
        os.environ.get(
            "MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP",
            "1",
        )
    )
    groups, effective_budget = _plan_budgeted_expert_groups(
        group_counts_host,
        route_budget,
        max_experts_per_group,
    )
    return groups, effective_budget, max_experts_per_group


def _build_budgeted_expert_plan(
    group_counts: torch.Tensor,
) -> _BudgetedExpertPlan:
    group_counts_host = [int(count) for count in group_counts.detach().cpu().tolist()]
    groups, effective_budget, max_experts_per_group = _budgeted_expert_groups(
        group_counts_host
    )
    _increment_training_diagnostic("bf16_routed_moe_budgeted_plan_builds")
    return (
        group_counts_host,
        groups,
        effective_budget,
        max_experts_per_group,
    )


def _accumulate_expert_group_in_order(
    output: torch.Tensor,
    route_values: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    group_counts: list[int],
) -> None:
    """Accumulate one grouped-GMM result in deterministic expert order."""
    offset = 0
    for count in group_counts:
        count = int(count)
        if count > 0:
            end = offset + count
            output.index_add_(
                0,
                sorted_token_ids[offset:end],
                route_values[offset:end],
            )
        offset += count


def _grouped_routed_moe_budgeted_forward(
    x: torch.Tensor,
    route_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    group_counts: torch.Tensor,
    swiglu_limit: float,
    plan: _BudgetedExpertPlan,
) -> torch.Tensor:
    """Run several small experts per GMM under a fixed route-row budget."""
    group_counts_host, groups, effective_budget, _ = plan
    native_values = (
        allocate_route_buffer(x, route_weights)
        if native_route_combine_enabled()
        else None
    )
    output = torch.zeros_like(x) if native_values is None else None
    for expert_start, expert_end, route_start, route_end in groups:
        token_ids = sorted_token_ids[route_start:route_end]
        slot_ids = sorted_slot_ids[route_start:route_end]
        chunk_counts_host = group_counts_host[expert_start:expert_end]
        chunk_counts = group_counts[expert_start:expert_end].contiguous()
        expert_x = x.index_select(0, token_ids)
        gate_up = _npu_grouped_moe_matmul(
            expert_x,
            w13[expert_start:expert_end],
            chunk_counts,
        )
        activated = _routed_moe_swiglu(gate_up, swiglu_limit)
        del gate_up
        expert_output = _npu_grouped_moe_matmul(
            activated,
            w2[expert_start:expert_end],
            chunk_counts,
        )
        if native_values is not None:
            native_values[route_start:route_end].copy_(expert_output)
        else:
            expert_output.mul_(route_weights[token_ids, slot_ids, None])
            _accumulate_expert_group_in_order(
                output,
                expert_output,
                token_ids,
                chunk_counts_host,
            )
        _record_max_training_diagnostic(
            "bf16_routed_moe_budgeted_max_group_routes",
            route_end - route_start,
        )
        _record_max_training_diagnostic(
            "bf16_routed_moe_budgeted_max_group_experts",
            expert_end - expert_start,
        )

    _increment_training_diagnostic("bf16_routed_moe_budgeted_forward_calls")
    for _ in groups:
        _increment_training_diagnostic("bf16_routed_moe_budgeted_forward_groups")
    _record_max_training_diagnostic(
        "bf16_routed_moe_budgeted_effective_route_budget",
        effective_budget,
    )
    _increment_training_diagnostic("bf16_routed_moe_low_memory_forward_calls")
    if native_values is not None:
        output = combine_routes(
            native_values,
            sorted_token_ids * route_weights.shape[1] + sorted_slot_ids,
            route_weights,
        )
    return output


def _grouped_routed_moe_budgeted_backward(
    x: torch.Tensor,
    route_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    group_counts: torch.Tensor,
    grad_output: torch.Tensor,
    needs: tuple[bool, ...],
    swiglu_limit: float,
    plan: _BudgetedExpertPlan,
):
    """Replay grouped experts while bounding every activation by route budget."""
    grad_x = torch.zeros_like(x) if needs[0] else None
    grad_route_weights = torch.zeros_like(route_weights) if needs[2] else None
    grad_w13 = torch.zeros_like(w13) if needs[3] else None
    grad_w2 = torch.zeros_like(w2) if needs[4] else None
    group_counts_host, groups, effective_budget, _ = plan
    _increment_training_diagnostic("bf16_routed_moe_budgeted_plan_reuses")

    for expert_start, expert_end, route_start, route_end in groups:
        token_ids = sorted_token_ids[route_start:route_end]
        slot_ids = sorted_slot_ids[route_start:route_end]
        chunk_counts_host = group_counts_host[expert_start:expert_end]
        chunk_counts = group_counts[expert_start:expert_end].contiguous()
        w13_group = w13[expert_start:expert_end]
        w2_group = w2[expert_start:expert_end]
        expert_x = x.index_select(0, token_ids)
        gate_up = _npu_grouped_moe_matmul(
            expert_x,
            w13_group,
            chunk_counts,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        if swiglu_limit > 0:
            unclamped_gate = gate
            unclamped_up = up
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        activated = silu_gate * up
        expert_output = _npu_grouped_moe_matmul(
            activated,
            w2_group,
            chunk_counts,
        )
        selected_grad_output = grad_output.index_select(0, token_ids)
        selected_weights = route_weights[token_ids, slot_ids]
        grad_expert_output = selected_grad_output * selected_weights[:, None]

        if grad_route_weights is not None:
            grad_route_weights[token_ids, slot_ids] = (
                selected_grad_output * expert_output
            ).sum(dim=-1)
        del expert_output

        if grad_w2 is not None:
            chunk_grad_w2 = _npu_grouped_moe_weight_grad(
                activated,
                grad_expert_output,
                chunk_counts,
            )
            expected_shape = w2[expert_start:expert_end].shape
            if chunk_grad_w2.shape != expected_shape:
                raise RuntimeError(
                    "Budgeted routed-MoE w2 gradient shape mismatch: "
                    f"expected={tuple(expected_shape)}, "
                    f"actual={tuple(chunk_grad_w2.shape)}"
                )
            grad_w2[expert_start:expert_end].copy_(chunk_grad_w2)

        grad_activated = _npu_grouped_moe_matmul(
            grad_expert_output,
            w2_group.transpose(1, 2),
            chunk_counts,
        )
        grad_gate = (
            grad_activated * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        )
        grad_up = grad_activated * silu_gate
        if swiglu_limit > 0:
            grad_gate = grad_gate * (unclamped_gate <= swiglu_limit)
            grad_up = grad_up * (
                (unclamped_up >= -swiglu_limit) & (unclamped_up <= swiglu_limit)
            )
        grad_gate_up = torch.cat((grad_gate, grad_up), dim=-1)
        if grad_w13 is not None:
            chunk_grad_w13 = _npu_grouped_moe_weight_grad(
                expert_x,
                grad_gate_up,
                chunk_counts,
            )
            expected_shape = w13[expert_start:expert_end].shape
            if chunk_grad_w13.shape != expected_shape:
                raise RuntimeError(
                    "Budgeted routed-MoE w13 gradient shape mismatch: "
                    f"expected={tuple(expected_shape)}, "
                    f"actual={tuple(chunk_grad_w13.shape)}"
                )
            grad_w13[expert_start:expert_end].copy_(chunk_grad_w13)
        if grad_x is not None:
            grad_expert_x = _npu_grouped_moe_matmul(
                grad_gate_up,
                w13_group.transpose(1, 2),
                chunk_counts,
            )
            _accumulate_expert_group_in_order(
                grad_x,
                grad_expert_x,
                token_ids,
                chunk_counts_host,
            )

        _record_max_training_diagnostic(
            "bf16_routed_moe_budgeted_max_group_routes",
            route_end - route_start,
        )
        _record_max_training_diagnostic(
            "bf16_routed_moe_budgeted_max_group_experts",
            expert_end - expert_start,
        )

    _increment_training_diagnostic("bf16_routed_moe_budgeted_backward_calls")
    for _ in groups:
        _increment_training_diagnostic("bf16_routed_moe_budgeted_backward_groups")
    _record_max_training_diagnostic(
        "bf16_routed_moe_budgeted_effective_route_budget",
        effective_budget,
    )
    _increment_training_diagnostic("bf16_routed_moe_low_memory_backward_calls")
    return grad_x, grad_route_weights, grad_w13, grad_w2


def _grouped_routed_moe_low_memory_backward(
    x: torch.Tensor,
    route_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    group_counts: torch.Tensor,
    grad_output: torch.Tensor,
    needs: tuple[bool, ...],
    swiglu_limit: float,
    *,
    native_gmm_replay: bool = False,
):
    """Replay and differentiate one expert at a time to bound route memory."""
    grad_x = torch.zeros_like(x) if needs[0] else None
    grad_route_weights = torch.zeros_like(route_weights) if needs[2] else None
    grad_w13 = torch.zeros_like(w13) if needs[3] else None
    grad_w2 = torch.zeros_like(w2) if needs[4] else None

    group_counts_host = group_counts.detach().cpu().tolist()
    offset = 0
    for expert_id, count in enumerate(group_counts_host):
        count = int(count)
        if count == 0:
            continue
        end = offset + count
        token_ids = sorted_token_ids[offset:end]
        slot_ids = sorted_slot_ids[offset:end]
        expert_x = x.index_select(0, token_ids)
        gate_up = expert_matmul(
            expert_x,
            w13,
            expert_id,
            group_counts[expert_id : expert_id + 1],
            native=native_gmm_replay,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        if swiglu_limit > 0:
            unclamped_gate = gate
            unclamped_up = up
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        activated = silu_gate * up
        expert_output = expert_matmul(
            activated,
            w2,
            expert_id,
            group_counts[expert_id : expert_id + 1],
            native=native_gmm_replay,
        )
        selected_grad_output = grad_output.index_select(0, token_ids)
        selected_weights = route_weights[token_ids, slot_ids]
        grad_expert_output = selected_grad_output * selected_weights[:, None]

        if grad_route_weights is not None:
            grad_route_weights[token_ids, slot_ids] = (
                selected_grad_output * expert_output
            ).sum(dim=-1)
        if grad_w2 is not None:
            grad_w2[expert_id].add_(
                activated.transpose(0, 1).matmul(grad_expert_output)
            )

        grad_activated = grad_expert_output.matmul(w2[expert_id].transpose(0, 1))
        grad_gate = (
            grad_activated * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        )
        grad_up = grad_activated * silu_gate
        if swiglu_limit > 0:
            grad_gate = grad_gate * (unclamped_gate <= swiglu_limit)
            grad_up = grad_up * (
                (unclamped_up >= -swiglu_limit) & (unclamped_up <= swiglu_limit)
            )
        grad_gate_up = torch.cat((grad_gate, grad_up), dim=-1)

        if grad_w13 is not None:
            grad_w13[expert_id].add_(expert_x.transpose(0, 1).matmul(grad_gate_up))
        if grad_x is not None:
            grad_expert_x = grad_gate_up.matmul(w13[expert_id].transpose(0, 1))
            grad_x.index_add_(0, token_ids, grad_expert_x)
        offset = end

    _increment_training_diagnostic("bf16_routed_moe_low_memory_backward_calls")
    return grad_x, grad_route_weights, grad_w13, grad_w2


def _grouped_routed_moe_low_memory_forward(
    x: torch.Tensor,
    route_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_slot_ids: torch.Tensor,
    group_counts: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Execute one expert at a time while reusing expert-major route metadata."""
    native_values = (
        allocate_route_buffer(x, route_weights)
        if native_route_combine_enabled()
        else None
    )
    output = torch.zeros_like(x) if native_values is None else None
    group_counts_host = group_counts.detach().cpu().tolist()
    offset = 0
    for expert_id, count in enumerate(group_counts_host):
        count = int(count)
        if count == 0:
            continue
        end = offset + count
        token_ids = sorted_token_ids[offset:end]
        slot_ids = sorted_slot_ids[offset:end]
        expert_x = x.index_select(0, token_ids)
        gate_up = expert_matmul(
            expert_x, w13, expert_id, group_counts[expert_id : expert_id + 1]
        )
        activated = _routed_moe_swiglu(gate_up, swiglu_limit)
        del gate_up
        expert_output = expert_matmul(
            activated, w2, expert_id, group_counts[expert_id : expert_id + 1]
        )
        if native_values is not None:
            native_values[offset:end].copy_(expert_output)
        else:
            expert_output.mul_(route_weights[token_ids, slot_ids, None])
            output.index_add_(0, token_ids, expert_output)
        offset = end

    _increment_training_diagnostic("bf16_routed_moe_low_memory_forward_calls")
    if native_values is not None:
        output = combine_routes(
            native_values,
            sorted_token_ids * route_weights.shape[1] + sorted_slot_ids,
            route_weights,
        )
    return output


class _GroupedConsolidatedRoutedMoE(torch.autograd.Function):
    """Use GMM for expert replay while retaining explicit audited gradients."""

    @staticmethod
    def forward(ctx, x, expert_ids, route_weights, w13, w2, swiglu_limit):
        ctx.native_gmm_replay = native_route_combine_enabled()
        (
            sorted_route_ids,
            sorted_token_ids,
            group_counts,
            expert_order,
        ) = _prepare_grouped_moe_routes(expert_ids, int(w13.shape[0]))
        routes_per_token = int(expert_ids.shape[-1])
        sorted_slot_ids = torch.remainder(sorted_route_ids, routes_per_token)
        ctx.grouped_moe_mode = "full_grouped"
        ctx.budgeted_expert_plan = None
        if _env_flag("MTP_DS4_GROUPED_ROUTED_MOE_EXPERT_STREAMING"):
            max_experts_per_group = int(
                os.environ.get(
                    "MTP_DS4_GROUPED_ROUTED_MOE_MAX_EXPERTS_PER_GROUP",
                    "1",
                )
            )
            if max_experts_per_group > 1:
                ctx.grouped_moe_mode = "budgeted"
                ctx.budgeted_expert_plan = _build_budgeted_expert_plan(group_counts)
                output = _grouped_routed_moe_budgeted_forward(
                    x,
                    route_weights,
                    w13,
                    w2,
                    sorted_token_ids,
                    sorted_slot_ids,
                    group_counts,
                    swiglu_limit,
                    ctx.budgeted_expert_plan,
                )
            else:
                ctx.grouped_moe_mode = "one_expert"
                output = _grouped_routed_moe_low_memory_forward(
                    x,
                    route_weights,
                    w13,
                    w2,
                    sorted_token_ids,
                    sorted_slot_ids,
                    group_counts,
                    swiglu_limit,
                )
        else:
            expert_x = x.index_select(0, sorted_token_ids)
            gate_up = _npu_grouped_moe_matmul(expert_x, w13, group_counts)
            activated = _routed_moe_swiglu(gate_up, swiglu_limit)
            expert_output = _npu_grouped_moe_matmul(
                activated,
                w2,
                group_counts,
            )
            if native_route_combine_enabled():
                output = combine_routes(expert_output, sorted_route_ids, route_weights)
            else:
                output = _combine_grouped_routes_token_unpermute(
                    expert_output,
                    sorted_route_ids,
                    route_weights,
                )
            if output is None:
                sorted_weights = route_weights.reshape(-1).index_select(
                    0,
                    sorted_route_ids,
                )
                expert_output.mul_(sorted_weights[:, None])
                _increment_training_diagnostic(
                    "bf16_routed_moe_inplace_route_weight_calls"
                )
                output = _combine_grouped_routes_fixed_order(
                    expert_output,
                    sorted_route_ids,
                    expert_order,
                )

        ctx.save_for_backward(
            x,
            route_weights,
            w13,
            w2,
            sorted_route_ids,
            sorted_token_ids,
            group_counts,
            expert_order,
        )
        ctx.swiglu_limit = swiglu_limit
        _increment_training_diagnostic("bf16_routed_moe_grouped_forward_calls")
        return output

    @staticmethod
    @_profile_vllm_ascend_backward("backward_routed_moe")
    def backward(ctx, grad_output):
        (
            x,
            route_weights,
            w13,
            w2,
            sorted_route_ids,
            sorted_token_ids,
            group_counts,
            expert_order,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        needs = ctx.needs_input_grad
        routes_per_token = int(route_weights.shape[-1])
        sorted_slot_ids = torch.remainder(sorted_route_ids, routes_per_token)
        grouped_moe_mode = ctx.grouped_moe_mode
        if grouped_moe_mode != "full_grouped":
            if grouped_moe_mode == "budgeted":
                if ctx.budgeted_expert_plan is None:
                    raise RuntimeError(
                        "Budgeted routed-MoE backward is missing its forward group plan"
                    )
                gradients = _grouped_routed_moe_budgeted_backward(
                    x,
                    route_weights,
                    w13,
                    w2,
                    sorted_token_ids,
                    sorted_slot_ids,
                    group_counts,
                    grad_output,
                    needs,
                    swiglu_limit,
                    ctx.budgeted_expert_plan,
                )
            elif grouped_moe_mode == "one_expert":
                gradients = _grouped_routed_moe_low_memory_backward(
                    x,
                    route_weights,
                    w13,
                    w2,
                    sorted_token_ids,
                    sorted_slot_ids,
                    group_counts,
                    grad_output,
                    needs,
                    swiglu_limit,
                    native_gmm_replay=ctx.native_gmm_replay,
                )
            else:
                raise RuntimeError(
                    f"Unknown grouped routed-MoE mode: {grouped_moe_mode!r}"
                )
            grad_x, grad_route_weights, grad_w13, grad_w2 = gradients
            if grad_x is not None:
                grad_x = _tensor_parallel_gradient_sum(grad_x)
            if grad_route_weights is not None:
                grad_route_weights = _tensor_parallel_gradient_sum(grad_route_weights)
            _increment_training_diagnostic("tp_routed_moe_grad_sum_calls")
            _increment_training_diagnostic("bf16_routed_moe_grouped_backward_calls")
            return grad_x, None, grad_route_weights, grad_w13, grad_w2, None

        full_grouped_backward = _env_flag("MTP_DS4_GROUPED_ROUTED_MOE_FULL_BACKWARD")

        # The grouped kernels return complete gradient tensors.  Allocating and
        # zero-filling full expert-weight buffers before replacing them wastes a
        # large memory-bandwidth pass on every MoE backward.
        grad_x = (
            None
            if full_grouped_backward
            else (torch.zeros_like(x) if needs[0] else None)
        )
        grad_route_weights = (
            torch.empty(
                route_weights.shape,
                dtype=route_weights.dtype,
                device=route_weights.device,
            )
            if needs[2]
            else None
        )
        grad_w13 = (
            None
            if full_grouped_backward
            else (torch.zeros_like(w13) if needs[3] else None)
        )
        grad_w2 = (
            None
            if full_grouped_backward
            else (torch.zeros_like(w2) if needs[4] else None)
        )

        expert_x = x.index_select(0, sorted_token_ids)
        gate_up = _npu_grouped_moe_matmul(expert_x, w13, group_counts)
        gate, up = gate_up.chunk(2, dim=-1)
        if swiglu_limit > 0:
            unclamped_gate = gate
            unclamped_up = up
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

        sigmoid_gate = torch.sigmoid(gate)
        silu_gate = gate * sigmoid_gate
        activated = silu_gate * up
        selected_grad_output = grad_output.index_select(0, sorted_token_ids)
        selected_weights = route_weights.reshape(-1).index_select(
            0,
            sorted_route_ids,
        )
        grad_expert_output = selected_grad_output * selected_weights[:, None]

        if grad_route_weights is not None:
            # Replaying the expert output is needed only for the router
            # gradient.  Frozen-router runs now skip this grouped matmul.
            expert_output = _npu_grouped_moe_matmul(
                activated,
                w2,
                group_counts,
            )
            route_sensitivities = (selected_grad_output * expert_output).sum(dim=-1)
            grad_route_weights.view(-1).index_copy_(
                0,
                sorted_route_ids,
                route_sensitivities,
            )
            del expert_output, route_sensitivities

        if full_grouped_backward:
            grad_activated = _npu_grouped_moe_matmul(
                grad_expert_output,
                w2.transpose(1, 2),
                group_counts,
            )
            grad_gate = (
                grad_activated * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
            )
            grad_up = grad_activated * silu_gate
            if swiglu_limit > 0:
                grad_gate = grad_gate * (unclamped_gate <= swiglu_limit)
                grad_up = grad_up * (
                    (unclamped_up >= -swiglu_limit) & (unclamped_up <= swiglu_limit)
                )
            grad_gate_up = torch.cat((grad_gate, grad_up), dim=-1)

            if needs[4]:
                grad_w2 = _npu_grouped_moe_weight_grad(
                    activated,
                    grad_expert_output,
                    group_counts,
                )
                if grad_w2.shape != w2.shape:
                    raise RuntimeError(
                        "Grouped routed-MoE w2 gradient shape mismatch: "
                        f"expected={tuple(w2.shape)}, actual={tuple(grad_w2.shape)}"
                    )
            if needs[3]:
                grad_w13 = _npu_grouped_moe_weight_grad(
                    expert_x,
                    grad_gate_up,
                    group_counts,
                )
                if grad_w13.shape != w13.shape:
                    raise RuntimeError(
                        "Grouped routed-MoE w13 gradient shape mismatch: "
                        f"expected={tuple(w13.shape)}, actual={tuple(grad_w13.shape)}"
                    )
            if needs[0]:
                grad_expert_x = _npu_grouped_moe_matmul(
                    grad_gate_up,
                    w13.transpose(1, 2),
                    group_counts,
                )
                grad_x = _combine_grouped_routes_fixed_order(
                    grad_expert_x,
                    sorted_route_ids,
                    expert_order,
                )
        else:
            group_counts_host = group_counts.detach().cpu().tolist()
            offset = 0
            for expert_id, count in enumerate(group_counts_host):
                count = int(count)
                if count == 0:
                    continue
                end = offset + count
                selected_x = expert_x[offset:end]
                selected_activated = activated[offset:end]
                selected_grad_expert_output = grad_expert_output[offset:end]
                selected_grad_activated = selected_grad_expert_output.matmul(
                    w2[expert_id].transpose(0, 1)
                )
                selected_gate = gate[offset:end]
                selected_up = up[offset:end]
                selected_sigmoid_gate = sigmoid_gate[offset:end]
                selected_silu_gate = silu_gate[offset:end]
                selected_grad_gate = (
                    selected_grad_activated
                    * selected_up
                    * selected_sigmoid_gate
                    * (1.0 + selected_gate * (1.0 - selected_sigmoid_gate))
                )
                selected_grad_up = selected_grad_activated * selected_silu_gate
                if swiglu_limit > 0:
                    selected_grad_gate = selected_grad_gate * (
                        unclamped_gate[offset:end] <= swiglu_limit
                    )
                    selected_unclamped_up = unclamped_up[offset:end]
                    selected_grad_up = selected_grad_up * (
                        (selected_unclamped_up >= -swiglu_limit)
                        & (selected_unclamped_up <= swiglu_limit)
                    )
                selected_grad_gate_up = torch.cat(
                    (selected_grad_gate, selected_grad_up),
                    dim=-1,
                )

                if grad_w2 is not None:
                    grad_w2[expert_id].add_(
                        selected_activated.transpose(0, 1).matmul(
                            selected_grad_expert_output
                        )
                    )
                if grad_w13 is not None:
                    grad_w13[expert_id].add_(
                        selected_x.transpose(0, 1).matmul(selected_grad_gate_up)
                    )
                if grad_x is not None:
                    token_ids = sorted_token_ids[offset:end]
                    grad_expert_x = selected_grad_gate_up.matmul(
                        w13[expert_id].transpose(0, 1)
                    )
                    grad_x.index_add_(0, token_ids, grad_expert_x)
                offset = end

        if grad_x is not None:
            grad_x = _tensor_parallel_gradient_sum(grad_x)
        if grad_route_weights is not None:
            grad_route_weights = _tensor_parallel_gradient_sum(grad_route_weights)
        _increment_training_diagnostic("tp_routed_moe_grad_sum_calls")
        _increment_training_diagnostic("bf16_routed_moe_grouped_backward_calls")
        if full_grouped_backward:
            _increment_training_diagnostic(
                "bf16_routed_moe_full_grouped_backward_calls"
            )
        return grad_x, None, grad_route_weights, grad_w13, grad_w2, None


def _patch_vllm_ascend_unquantized_moe_training_forward() -> bool:
    from verl_speco.backends.vllm_ascend_compat import split_moe_api

    if split_moe_api():
        from verl_speco.backends.vllm_ascend_compat import (
            patch_module_apply_visitor,
            unquantized_moe_type,
        )

        patch_module_apply_visitor(unquantized_moe_type())
        return _patch_vllm_ascend_model_internal_router_training_forward()
    try:
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts
        from vllm_ascend.ops.fused_moe.fused_moe import (
            AscendUnquantizedFusedMoEMethod,
        )
        from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(
        AscendUnquantizedFusedMoEMethod,
        "_speco_deepseek_v4_training_forward_patch",
        False,
    ):
        return True

    original_apply = AscendUnquantizedFusedMoEMethod.apply

    def _training_apply(
        self,
        layer,
        x=None,
        use_grouped_topk=None,
        top_k=None,
        router_logits=None,
        renormalize=None,
        **kwargs,
    ):
        # UnquantizedFusedMoEMethod is also an nn.Module, but vLLM uses the
        # name "apply" for its expert forward. Preserve nn.Module.apply(fn)
        # for DeepSpeed and other module-tree visitors.
        if x is None and callable(layer) and not kwargs:
            return nn.Module.apply(self, layer)
        if x is None or top_k is None or router_logits is None or renormalize is None:
            raise TypeError("Incomplete arguments for unquantized routed MoE apply")
        if not torch.is_grad_enabled() or not x.is_floating_point():
            return original_apply(
                self,
                layer,
                x,
                use_grouped_topk,
                top_k,
                router_logits,
                renormalize,
                **kwargs,
            )
        if self.dynamic_eplb or kwargs.get("expert_map") is not None:
            raise RuntimeError(
                "BF16 routed MoE training supports the TP-sharded, non-EP "
                "layout only; dynamic EPLB and expert remapping are not supported"
            )
        if kwargs.get("log2phy") is not None or getattr(layer, "zero_expert_num", 0):
            raise RuntimeError(
                "BF16 routed MoE training does not support expert remapping"
            )
        activation = kwargs.get("activation", "silu")
        activation = getattr(activation, "value", activation)
        if isinstance(activation, str):
            activation = activation.lower()
        if activation != "silu":
            raise RuntimeError(f"Unsupported routed MoE activation {activation!r}")

        custom_routing_function = kwargs.get("custom_routing_function")
        if custom_routing_function is not None:
            raise RuntimeError(
                "BF16 routed MoE training requires the standard vLLM expert selector"
            )
        scoring_func = kwargs.get("scoring_func", "softmax")
        routed_scaling_factor = float(kwargs.get("routed_scaling_factor", 1.0))
        if routed_scaling_factor <= 0.0:
            raise RuntimeError(
                "BF16 routed MoE training requires a positive "
                f"routed_scaling_factor, got {routed_scaling_factor}"
            )
        with torch.no_grad():
            native_topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=kwargs.get("topk_group"),
                num_expert_group=kwargs.get("num_expert_group"),
                custom_routing_function=None,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=kwargs.get("e_score_correction_bias"),
                num_experts=kwargs.get("num_experts", layer.w13_weight.shape[0]),
                tid2eid=self.tid2eid,
            )

        # The native Ascend path returns an already-combined tensor, so its
        # selector applies routed_scaling_factor internally. The training path
        # returns (shared, routed), and DeepseekV4MoE.forward applies that factor
        # while combining the tuple. Remove it here to avoid scaling routed
        # experts twice, while preserving the native selector's forward values.
        native_topk_weights = native_topk_weights / routed_scaling_factor

        # Expert IDs are discrete. Rebuild the selected unscaled routing weights
        # from router_logits so backward avoids the inference-only gating op.
        if scoring_func == "softmax":
            all_scores = router_logits.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            all_scores = router_logits.sigmoid()
        elif scoring_func == "sqrtsoftplus":
            all_scores = F.softplus(router_logits).sqrt()
        else:
            raise RuntimeError(
                f"Unsupported routed MoE scoring function {scoring_func!r}"
            )
        replay_topk_weights = all_scores.gather(1, topk_ids.long())
        if renormalize:
            replay_topk_weights = replay_topk_weights / replay_topk_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
        replay_topk_weights = replay_topk_weights.to(native_topk_weights.dtype)
        topk_weights = native_topk_weights.detach() + (
            replay_topk_weights - replay_topk_weights.detach()
        )
        flat_x = x.reshape(-1, x.shape[-1])
        ids = topk_ids.reshape(flat_x.shape[0], -1)
        weights = topk_weights.reshape(flat_x.shape[0], -1).to(x.dtype)
        swiglu_limit = float(getattr(layer, "swiglu_limit", 0.0) or 0.0)
        routed_moe_function = (
            _GroupedConsolidatedRoutedMoE
            if _env_flag("MTP_DS4_GROUPED_ROUTED_MOE_REPLAY")
            else _ConsolidatedRoutedMoE
        )
        output = routed_moe_function.apply(
            flat_x,
            ids,
            weights,
            layer.w13_weight,
            layer.w2_weight,
            swiglu_limit,
        )

        return FusedExpertsResult(
            routed_out=output.reshape_as(x),
            swiglu_limit=int(swiglu_limit),
        )

    AscendUnquantizedFusedMoEMethod.apply = _training_apply
    AscendUnquantizedFusedMoEMethod._speco_deepseek_v4_training_forward_patch = True
    return True


def _patch_vllm_ascend_linear_training_backward() -> bool:
    try:
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(AscendLinearMethod, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    original_apply = AscendLinearMethod.apply

    class _QuantLinearInputGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, layer, bias, method):
            ctx.layer = layer
            ctx.backward_profile_name = _linear_backward_profile_name(
                str(getattr(layer, "prefix", "")),
                quantized=True,
            )
            ctx.input_dtype = x.dtype
            ctx.input_size = x.shape[-1]
            with torch.no_grad():
                return original_apply(method, layer, x, bias)

        @staticmethod
        @_profile_vllm_ascend_backward(
            lambda ctx, grad_output: ctx.backward_profile_name
        )
        def backward(ctx, grad_output):
            layer = ctx.layer
            weight = getattr(layer, "weight", None)
            scale = getattr(layer, "weight_scale_fp32", None)
            if scale is None:
                scale = getattr(layer, "weight_scale", None)
            if weight is None or scale is None:
                raise RuntimeError(
                    f"Cannot build W8A8 input gradient for {getattr(layer, 'prefix', type(layer).__name__)}: "
                    "weight or weight_scale is missing"
                )

            scale = scale.detach().float().reshape(-1)
            input_size = int(ctx.input_size)
            if weight.shape[0] == input_size:
                input_major = True
                output_size = int(weight.shape[1])
            elif weight.shape[1] == input_size:
                input_major = False
                output_size = int(weight.shape[0])
            else:
                raise RuntimeError(
                    f"Unexpected quantized weight shape {tuple(weight.shape)} for input size {input_size}"
                )
            if scale.numel() != output_size:
                raise RuntimeError(
                    f"Weight scale size {scale.numel()} does not match output size {output_size}"
                )

            grad_flat = grad_output.reshape(-1, output_size).float()
            grad_input = torch.zeros(
                grad_flat.shape[0],
                input_size,
                dtype=torch.float32,
                device=grad_output.device,
            )
            chunk_size = 4096
            for start in range(0, output_size, chunk_size):
                end = min(start + chunk_size, output_size)
                grad_chunk = grad_flat[:, start:end] * scale[start:end]
                if input_major:
                    weight_chunk = weight[:, start:end].detach().float()
                    grad_input.add_(grad_chunk.matmul(weight_chunk.transpose(0, 1)))
                else:
                    weight_chunk = weight[start:end, :].detach().float()
                    grad_input.add_(grad_chunk.matmul(weight_chunk))
            return (
                grad_input.reshape(*grad_output.shape[:-1], input_size).to(
                    ctx.input_dtype
                ),
                None,
                None,
                None,
            )

    def _autograd_apply(self, layer, x, bias=None):
        if not torch.is_grad_enabled() or not x.is_floating_point():
            return original_apply(self, layer, x, bias)
        return _QuantLinearInputGrad.apply(x, layer, bias, self)

    AscendLinearMethod.apply = _autograd_apply
    AscendLinearMethod._speco_deepseek_v4_train_backward_patch = True
    return True


def _patch_vllm_ascend_unquantized_linear_training_backward() -> bool:
    try:
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(
        AscendUnquantizedLinearMethod,
        "_speco_deepseek_v4_train_backward_patch",
        False,
    ):
        return True

    original_apply = AscendUnquantizedLinearMethod.apply

    class _FusedQNormUnquantizedLinearGrad(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            native_output,
            normalized_output,
            norm_input,
            norm_weight,
            norm_bias,
            linear_weight,
            linear_bias,
            norm_module,
            epsilon,
            sum_norm_grad_output_across_tp,
        ):
            ctx.norm_module = norm_module
            ctx.epsilon = float(epsilon)
            ctx.sum_norm_grad_output_across_tp = bool(sum_norm_grad_output_across_tp)
            ctx.input_dtype = norm_input.dtype
            ctx.norm_weight_dtype = norm_weight.dtype
            ctx.linear_weight_dtype = linear_weight.dtype
            ctx.has_norm_bias = bool(norm_bias.numel())
            ctx.norm_bias_dtype = norm_bias.dtype if ctx.has_norm_bias else None
            ctx.has_linear_bias = bool(linear_bias.numel())
            ctx.linear_bias_dtype = linear_bias.dtype if ctx.has_linear_bias else None
            ctx.save_for_backward(
                normalized_output,
                norm_input,
                norm_weight,
                norm_bias,
                linear_weight,
                linear_bias,
            )
            return native_output.clone()

        @staticmethod
        @_profile_vllm_ascend_backward("backward_qnorm_projection")
        def backward(ctx, grad_output):
            _increment_training_diagnostic("bf16_qnorm_linear_fused_backward_calls")
            (
                normalized_output,
                norm_input,
                norm_weight,
                norm_bias,
                linear_weight,
                _linear_bias,
            ) = ctx.saved_tensors
            ctx.norm_module._speco_deepseek_v4_replay_grad_calls = (
                getattr(
                    ctx.norm_module,
                    "_speco_deepseek_v4_replay_grad_calls",
                    0,
                )
                + 1
            )

            output_size = int(linear_weight.shape[0])
            input_size = int(linear_weight.shape[1])
            grad_flat = grad_output.reshape(-1, output_size)
            normalized_flat = normalized_output.reshape(-1, input_size)
            compute_dtype = (
                torch.float32
                if torch.float32 in (ctx.input_dtype, ctx.linear_weight_dtype)
                else ctx.input_dtype
            )
            grad_compute = grad_flat.to(compute_dtype)

            grad_normed = grad_compute.matmul(
                linear_weight.to(compute_dtype)
            ).reshape_as(norm_input)
            grad_normed = _tensor_parallel_gradient_sum(grad_normed)
            _increment_training_diagnostic("tp_column_linear_grad_sum_calls")

            norm_grad_output = (
                _tensor_parallel_gradient_sum(grad_normed)
                if ctx.sum_norm_grad_output_across_tp
                else grad_normed
            )
            if ctx.sum_norm_grad_output_across_tp:
                _increment_training_diagnostic("tp_replicated_norm_grad_sum_calls")
            grad_input_float, grad_norm_weight_float = _explicit_rms_norm_vjp(
                norm_input,
                norm_weight,
                norm_grad_output,
                ctx.epsilon,
            )

            grad_norm_bias = None
            if ctx.has_norm_bias and ctx.needs_input_grad[4]:
                grad_norm_bias = grad_normed.reshape(
                    -1,
                    grad_normed.shape[-1],
                ).sum(dim=0)
                grad_norm_bias = grad_norm_bias.reshape_as(norm_bias).to(
                    ctx.norm_bias_dtype
                )

            grad_linear_weight = None
            if ctx.needs_input_grad[5]:
                grad_linear_weight = grad_compute.transpose(0, 1).matmul(
                    normalized_flat.to(compute_dtype)
                )
                grad_linear_weight = grad_linear_weight.to(ctx.linear_weight_dtype)

            grad_linear_bias = None
            if ctx.has_linear_bias and ctx.needs_input_grad[6]:
                grad_linear_bias = grad_flat.sum(dim=0).to(ctx.linear_bias_dtype)

            return (
                None,
                None,
                (
                    grad_input_float.to(ctx.input_dtype)
                    if ctx.needs_input_grad[2]
                    else None
                ),
                (
                    grad_norm_weight_float.to(ctx.norm_weight_dtype)
                    if ctx.needs_input_grad[3]
                    else None
                ),
                grad_norm_bias,
                grad_linear_weight,
                grad_linear_bias,
                None,
                None,
                None,
            )

    class _UnquantizedLinearGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, layer, method):
            ctx.has_bias = bias is not None
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.input_dtype = x.dtype
            ctx.weight_dtype = weight.dtype
            prefix = str(getattr(layer, "prefix", ""))
            ctx.backward_profile_name = _linear_backward_profile_name(
                prefix,
                quantized=False,
            )
            ctx.sum_grad_input_across_tp = prefix.endswith(
                (".self_attn.wq_b", ".mlp.shared_experts.gate_up_proj")
            )
            ctx.save_for_backward(x, weight)
            with torch.no_grad():
                return original_apply(method, layer, x, bias)

        @staticmethod
        @_profile_vllm_ascend_backward(
            lambda ctx, grad_output: ctx.backward_profile_name
        )
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            output_size = int(weight.shape[0])
            input_size = int(weight.shape[1])
            grad_flat = grad_output.reshape(-1, output_size)
            x_flat = x.reshape(-1, input_size)
            compute_dtype = (
                torch.float32
                if torch.float32 in (ctx.input_dtype, ctx.weight_dtype)
                else ctx.input_dtype
            )

            grad_input = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_flat.to(compute_dtype).matmul(
                    weight.to(compute_dtype)
                )
                grad_input = grad_input.reshape_as(x).to(ctx.input_dtype)
                if ctx.sum_grad_input_across_tp:
                    grad_input = _tensor_parallel_gradient_sum(grad_input)
                    _increment_training_diagnostic("tp_column_linear_grad_sum_calls")

            grad_weight = None
            if ctx.needs_input_grad[1]:
                grad_weight = (
                    grad_flat.transpose(0, 1)
                    .to(compute_dtype)
                    .matmul(x_flat.to(compute_dtype))
                )
                grad_weight = grad_weight.to(ctx.weight_dtype)

            grad_bias = None
            if ctx.has_bias and ctx.needs_input_grad[2]:
                grad_bias = grad_flat.sum(dim=0).to(ctx.bias_dtype)

            return grad_input, grad_weight, grad_bias, None, None

    def _autograd_apply(self, layer, x, bias=None):
        weight = layer.weight
        if not torch.is_grad_enabled() or not x.is_floating_point():
            return original_apply(self, layer, x, bias)
        prefix = str(getattr(layer, "prefix", ""))
        qnorm_source = None
        if prefix.endswith(".self_attn.wq_b") and _env_flag(
            "MTP_DS4_FUSED_QNORM_UNQUANTIZED_BACKWARD",
            True,
        ):
            qnorm_source = _take_bf16_qnorm_source(x)
        if qnorm_source is not None:
            _increment_training_diagnostic("bf16_qnorm_linear_source_hits")
            (
                norm_input,
                norm_weight,
                norm_bias,
                norm_module,
                epsilon,
                sum_norm_grad_output_across_tp,
            ) = qnorm_source
            with torch.no_grad():
                native_output = original_apply(self, layer, x, bias)
            linear_bias = bias if bias is not None else x.new_empty((0,))
            return _FusedQNormUnquantizedLinearGrad.apply(
                native_output,
                x,
                norm_input,
                norm_weight,
                norm_bias,
                weight,
                linear_bias,
                norm_module,
                epsilon,
                sum_norm_grad_output_across_tp,
            )
        if prefix.endswith(".self_attn.wq_b"):
            _increment_training_diagnostic("bf16_qnorm_linear_source_misses")
        return _UnquantizedLinearGrad.apply(x, weight, bias, layer, self)

    AscendUnquantizedLinearMethod.apply = _autograd_apply
    AscendUnquantizedLinearMethod._speco_deepseek_v4_train_backward_patch = True
    return True


class _DeterministicTPRowReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_output, comm_group):
        world_size = int(comm_group.world_size)
        if world_size == 1:
            return local_output.clone()
        buffered = _env_flag("MTP_DS4_BUFFERED_DETERMINISTIC_REDUCE")
        process_group = getattr(comm_group, "device_group", None)
        if buffered and dist.is_initialized() and process_group is not None:
            local_flat = local_output.contiguous().reshape(-1)
            local_numel = int(local_flat.numel())
            key = (
                str(local_output.device),
                local_output.dtype,
                world_size,
                id(process_group),
            )
            buffers = _DETERMINISTIC_REDUCE_BUFFERS.get(key)
            required_gathered = world_size * local_numel
            if (
                buffers is None
                or buffers["gathered"].numel() < required_gathered
                or buffers["accumulator"].numel() < local_numel
            ):
                buffers = {
                    "gathered": torch.empty(
                        required_gathered,
                        dtype=local_output.dtype,
                        device=local_output.device,
                    ),
                    "accumulator": torch.empty(
                        local_numel,
                        dtype=torch.float32,
                        device=local_output.device,
                    ),
                }
                _DETERMINISTIC_REDUCE_BUFFERS[key] = buffers
                _increment_training_diagnostic(
                    "tp_buffered_deterministic_reduce_allocations"
                )

            gathered = buffers["gathered"][:required_gathered]
            accumulator = buffers["accumulator"][:local_numel]
            dist.all_gather_into_tensor(
                gathered,
                local_flat,
                group=process_group,
            )
            rank_outputs = gathered.reshape(world_size, local_numel)
            accumulator.copy_(rank_outputs[0])
            for rank_index in range(1, world_size):
                # The FP32 accumulator performs the same BF16->FP32 promotion
                # without allocating a full temporary tensor for every rank.
                accumulator.add_(rank_outputs[rank_index])
            _increment_training_diagnostic("tp_buffered_deterministic_reduce_calls")
            reduced = accumulator.to(local_output.dtype)
            if reduced.data_ptr() == accumulator.data_ptr():
                reduced = reduced.clone()
            return reduced.reshape_as(local_output)

        gathered = comm_group.all_gather(local_output, dim=0)
        rank_outputs = gathered.reshape(world_size, *local_output.shape)
        reduced = rank_outputs[0].float()
        for rank_index in range(1, world_size):
            # Preserve fixed rank order while avoiding both the explicit cast
            # buffer and a new FP32 accumulator allocation on every add.
            reduced.add_(rank_outputs[rank_index])
        if buffered:
            _increment_training_diagnostic(
                "tp_buffered_deterministic_reduce_fallback_calls"
            )
        return reduced.to(local_output.dtype)

    @staticmethod
    @_profile_vllm_ascend_backward("backward_tp_collective")
    def backward(ctx, grad_output):
        return grad_output, None


class _TPColumnShardedProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, comm_group, tp_rank, tp_size):
        ctx.comm_group = comm_group
        ctx.tp_rank = int(tp_rank)
        ctx.tp_size = int(tp_size)
        ctx.input_shape = tuple(x.shape)
        ctx.save_for_backward(x, weight)
        local_output = F.linear(x, weight)
        if ctx.tp_size == 1:
            return local_output
        return comm_group.all_gather(local_output.contiguous(), dim=-1)

    @staticmethod
    @_profile_vllm_ascend_backward("backward_tp_projection")
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        local_output_size = int(weight.shape[0])
        start = ctx.tp_rank * local_output_size
        grad_local_output = grad_output.narrow(
            -1,
            start,
            local_output_size,
        ).contiguous()
        grad_flat = grad_local_output.reshape(-1, local_output_size)
        x_flat = x.reshape(-1, x.shape[-1])

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_flat.matmul(weight).reshape(ctx.input_shape)
            grad_x = _DeterministicTPRowReduce.apply(
                grad_x,
                ctx.comm_group,
            )

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = grad_flat.transpose(0, 1).matmul(x_flat)
            grad_weight = grad_weight.to(weight.dtype)
        return grad_x, grad_weight, None, None, None


class _TPColumnShardedProjection(nn.Module):
    def __init__(
        self,
        full_weight: torch.Tensor,
        *,
        comm_group,
        tp_rank: int,
        tp_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        if full_weight.ndim != 2:
            raise ValueError(
                f"{prefix} requires a 2D weight, got {tuple(full_weight.shape)}"
            )
        output_size, input_size = map(int, full_weight.shape)
        if output_size % tp_size != 0:
            raise ValueError(
                f"{prefix} output size {output_size} is not divisible by TP={tp_size}"
            )
        local_output_size = output_size // tp_size
        start = tp_rank * local_output_size
        local_weight = (
            full_weight.detach()
            .narrow(
                0,
                start,
                local_output_size,
            )
            .contiguous()
        )
        self.weight = nn.Parameter(
            local_weight,
            requires_grad=bool(full_weight.requires_grad),
        )
        # DeepSpeed includes rank-local shards from every TP rank when it
        # computes the global gradient norm only if this marker is present.
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = 0
        self.weight.partition_stride = 1
        self.comm_group = comm_group
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.input_size = input_size
        self.output_size = output_size
        self.output_size_per_partition = local_output_size
        self.gather_output = True
        self.prefix = prefix

    def forward(self, x):
        _increment_training_diagnostic("tp_sharded_eh_projection_calls")
        return _TPColumnShardedProjectionFunction.apply(
            x,
            self.weight,
            self.comm_group,
            self.tp_rank,
            self.tp_size,
        )


class _TPFusedEHProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs_embeds,
        previous_hidden_states,
        e_weight,
        h_weight,
        comm_group,
        tp_rank,
        tp_size,
    ):
        ctx.comm_group = comm_group
        ctx.tp_rank = int(tp_rank)
        ctx.tp_size = int(tp_size)
        ctx.inputs_shape = tuple(inputs_embeds.shape)
        ctx.hidden_shape = tuple(previous_hidden_states.shape)
        ctx.save_for_backward(
            inputs_embeds,
            previous_hidden_states,
            e_weight,
            h_weight,
        )
        local_e = F.linear(inputs_embeds, e_weight).unsqueeze(-2)
        local_h = F.linear(previous_hidden_states, h_weight)
        local_hidden = local_e + local_h
        if ctx.tp_size == 1:
            return local_hidden
        return comm_group.all_gather(local_hidden.contiguous(), dim=-1)

    @staticmethod
    @_profile_vllm_ascend_backward("backward_tp_projection")
    def backward(ctx, grad_output):
        inputs_embeds, previous_hidden_states, e_weight, h_weight = ctx.saved_tensors
        local_output_size = int(e_weight.shape[0])
        start = ctx.tp_rank * local_output_size
        grad_local = grad_output.narrow(
            -1,
            start,
            local_output_size,
        ).contiguous()
        grad_e_output = grad_local.sum(dim=-2)

        e_grad_flat = grad_e_output.reshape(-1, local_output_size)
        h_grad_flat = grad_local.reshape(-1, local_output_size)
        e_input_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        h_input_flat = previous_hidden_states.reshape(
            -1,
            previous_hidden_states.shape[-1],
        )
        compute_dtype = (
            torch.float32
            if torch.float32
            in (
                inputs_embeds.dtype,
                previous_hidden_states.dtype,
                e_weight.dtype,
                h_weight.dtype,
            )
            else inputs_embeds.dtype
        )

        grad_inputs = None
        grad_hidden = None
        input_grad_parts = []
        input_grad_sizes = []
        if ctx.needs_input_grad[0]:
            local_grad_inputs = e_grad_flat.to(compute_dtype).matmul(
                e_weight.to(compute_dtype)
            )
            input_grad_parts.append(local_grad_inputs)
            input_grad_sizes.append(local_grad_inputs.shape[0])
        if ctx.needs_input_grad[1]:
            local_grad_hidden = h_grad_flat.to(compute_dtype).matmul(
                h_weight.to(compute_dtype)
            )
            input_grad_parts.append(local_grad_hidden)
            input_grad_sizes.append(local_grad_hidden.shape[0])
        if input_grad_parts:
            combined_local_grad = torch.cat(input_grad_parts, dim=0)
            combined_grad = _DeterministicTPRowReduce.apply(
                combined_local_grad,
                ctx.comm_group,
            )
            offset = 0
            part_index = 0
            if ctx.needs_input_grad[0]:
                size = input_grad_sizes[part_index]
                grad_inputs = (
                    combined_grad[offset : offset + size]
                    .reshape(ctx.inputs_shape)
                    .to(inputs_embeds.dtype)
                )
                offset += size
                part_index += 1
            if ctx.needs_input_grad[1]:
                size = input_grad_sizes[part_index]
                grad_hidden = (
                    combined_grad[offset : offset + size]
                    .reshape(ctx.hidden_shape)
                    .to(previous_hidden_states.dtype)
                )

        grad_e_weight = None
        if ctx.needs_input_grad[2]:
            grad_e_weight = (
                e_grad_flat.transpose(0, 1)
                .to(compute_dtype)
                .matmul(e_input_flat.to(compute_dtype))
                .to(e_weight.dtype)
            )

        grad_h_weight = None
        if ctx.needs_input_grad[3]:
            grad_h_weight = (
                h_grad_flat.transpose(0, 1)
                .to(compute_dtype)
                .matmul(h_input_flat.to(compute_dtype))
                .to(h_weight.dtype)
            )

        return (
            grad_inputs,
            grad_hidden,
            grad_e_weight,
            grad_h_weight,
            None,
            None,
            None,
        )


def _install_tp_sharded_eh_projections(model) -> bool:
    if not _env_flag("MTP_DS4_TP_SHARD_EH_PROJECTIONS"):
        return False
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    tp_size = int(get_tensor_model_parallel_world_size())
    if tp_size <= 1:
        return False
    tp_rank = int(get_tensor_model_parallel_rank())
    comm_group = get_tp_group()
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith((".e_proj", ".h_proj"))
    ]
    if len(matches) != 2:
        raise RuntimeError(
            "Expected exactly one DS4 MTP e_proj and h_proj, found "
            f"{[name for name, _ in matches]}"
        )
    for name, module in matches:
        if isinstance(module, _TPColumnShardedProjection):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError(
                f"Cannot TP-shard {name}: invalid weight "
                f"{None if weight is None else tuple(weight.shape)}"
            )
        if bias is not None:
            raise RuntimeError(f"Cannot TP-shard biased projection {name}")
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            _TPColumnShardedProjection(
                weight,
                comm_group=comm_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                prefix=name,
            ),
        )
    _increment_training_diagnostic("tp_sharded_eh_projection_installations")
    return True


def _find_cv_linear_wrappers(model) -> list[tuple[str, Any]]:
    wrappers: list[tuple[str, Any]] = []
    seen_owners: set[int] = set()
    seen_wrappers: set[int] = set()

    for module_name, module in model.named_modules():
        owners = [(module_name, module)]
        for impl_name in ("impl", "_impl"):
            impl = getattr(module, impl_name, None)
            if impl is not None:
                owners.append((f"{module_name}.{impl_name}", impl))

        for owner_name, owner in owners:
            owner_id = id(owner)
            if owner_id in seen_owners:
                continue
            seen_owners.add(owner_id)
            for attribute_name in ("cv_wq_a", "cv_wkv"):
                wrapper = getattr(owner, attribute_name, None)
                if wrapper is None or id(wrapper) in seen_wrappers:
                    continue
                if getattr(wrapper, "linear", None) is None:
                    continue
                seen_wrappers.add(id(wrapper))
                wrappers.append((f"{owner_name}.{attribute_name}", wrapper))

    return wrappers


def _install_tp_sharded_attention_projections(model) -> bool:
    if not _env_flag("MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS"):
        return False
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    tp_size = int(get_tensor_model_parallel_world_size())
    if tp_size <= 1:
        return False
    tp_rank = int(get_tensor_model_parallel_rank())
    comm_group = get_tp_group()
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(("self_attn.wq_a", "self_attn.wkv"))
    ]
    if len(matches) != 2:
        raise RuntimeError(
            "Expected exactly one DS4 MTP wq_a and wkv module, found "
            f"{[name for name, _ in matches]}"
        )

    cv_wrappers = _find_cv_linear_wrappers(model)
    matched_wrapper_names: set[str] = set()

    for name, module in matches:
        projection_wrappers = [
            (wrapper_name, wrapper)
            for wrapper_name, wrapper in cv_wrappers
            if getattr(wrapper, "linear", None) is module
        ]
        for wrapper_name, wrapper in projection_wrappers:
            wrapper._has_communication = True
            matched_wrapper_names.add(wrapper_name)

        if getattr(
            module,
            "_speco_deepseek_v4_tp_sharded_attention_projection",
            False,
        ):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError(
                f"Cannot TP-shard {name}: invalid weight "
                f"{None if weight is None else tuple(weight.shape)}"
            )
        if not weight.dtype.is_floating_point:
            raise RuntimeError(
                f"Cannot TP-shard non-floating training weight {name}: "
                f"dtype={weight.dtype}"
            )
        if bias is not None:
            raise RuntimeError(f"Cannot TP-shard biased projection {name}")
        output_size, input_size = map(int, weight.shape)
        if output_size % tp_size != 0:
            raise ValueError(
                f"{name} output size {output_size} is not divisible by TP={tp_size}"
            )
        local_output_size = output_size // tp_size
        local_weight = (
            weight.detach()
            .narrow(
                0,
                tp_rank * local_output_size,
                local_output_size,
            )
            .contiguous()
        )
        sharded_weight = nn.Parameter(
            local_weight,
            requires_grad=bool(weight.requires_grad),
        )
        sharded_weight.tensor_model_parallel = True
        sharded_weight.partition_dim = 0
        sharded_weight.partition_stride = 1
        module.weight = sharded_weight
        module._speco_deepseek_v4_original_forward = module.forward
        module._speco_deepseek_v4_tp_comm_group = comm_group
        module._speco_deepseek_v4_tp_rank = tp_rank
        module._speco_deepseek_v4_tp_size = tp_size
        module._speco_deepseek_v4_full_output_size = output_size
        module._speco_deepseek_v4_local_output_size = local_output_size
        module._speco_deepseek_v4_projection_name = name.rsplit(".", 1)[-1]
        module.input_size = input_size
        module.output_size = output_size
        module.output_partition_sizes = [local_output_size]
        module.gather_output = True

        def sharded_forward(layer, x):
            _increment_training_diagnostic("tp_sharded_attention_projection_calls")
            projection_name = layer._speco_deepseek_v4_projection_name
            _increment_training_diagnostic(
                f"tp_sharded_attention_{projection_name}_calls"
            )
            output = _TPColumnShardedProjectionFunction.apply(
                x,
                layer.weight,
                layer._speco_deepseek_v4_tp_comm_group,
                layer._speco_deepseek_v4_tp_rank,
                layer._speco_deepseek_v4_tp_size,
            )
            if not bool(getattr(layer, "return_bias", False)):
                return output
            return output, None

        module.forward = types.MethodType(sharded_forward, module)
        module._speco_deepseek_v4_tp_sharded_attention_projection = True

        # CVLinearWrapper is owned by the attention implementation, which is
        # not necessarily an nn.Module. Force it to delegate to the patched
        # full-output forward instead of applying the rank-local weight.
        if cv_wrappers and not projection_wrappers:
            raise RuntimeError(
                f"Cannot TP-shard {name}: a CVLinearWrapper runtime is active "
                "but no wrapper references this projection"
            )

    unmatched_wrappers = sorted(
        wrapper_name
        for wrapper_name, _ in cv_wrappers
        if wrapper_name not in matched_wrapper_names
    )
    if unmatched_wrappers:
        raise RuntimeError(
            "TP-sharded attention projections left CVLinearWrapper instances "
            f"unpatched: {unmatched_wrappers}"
        )

    _increment_training_diagnostic("tp_sharded_attention_projection_installations")
    return True


def _install_fused_eh_projection_collective(model) -> bool:
    if not _env_flag("MTP_DS4_FUSED_EH_PROJECTION_COLLECTIVE"):
        return False
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "e_proj", None), _TPColumnShardedProjection)
        and isinstance(
            getattr(module, "h_proj", None),
            _TPColumnShardedProjection,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one DS4 MTP layer with TP-sharded e_proj/h_proj, "
            f"found {[name for name, _ in matches]}"
        )
    _, layer = matches[0]
    if getattr(layer, "_speco_deepseek_v4_fused_eh_projection_collective", False):
        return True
    e_proj = layer.e_proj
    h_proj = layer.h_proj
    if (
        e_proj.comm_group is not h_proj.comm_group
        or e_proj.tp_rank != h_proj.tp_rank
        or e_proj.tp_size != h_proj.tp_size
    ):
        raise RuntimeError("e_proj and h_proj do not share the same TP topology")
    original_forward = layer.forward

    def fused_forward(
        current_layer,
        input_ids,
        positions,
        previous_hidden_states,
        inputs_embeds=None,
        spec_step_index=0,
    ):
        if inputs_embeds is None:
            raise ValueError("DS4 MTP fused e/h projection requires inputs_embeds")
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0,
            0,
            inputs_embeds,
        )
        inputs_embeds = current_layer.enorm(inputs_embeds)
        previous_hidden_states = previous_hidden_states.view(
            -1,
            current_layer.hc_mult,
            current_layer.config.hidden_size,
        )
        previous_hidden_states = current_layer.hnorm(previous_hidden_states)
        _increment_training_diagnostic("tp_sharded_eh_projection_calls")
        _increment_training_diagnostic("tp_sharded_eh_projection_calls")
        _increment_training_diagnostic("tp_fused_eh_projection_collective_calls")
        hidden_states = _TPFusedEHProjectionFunction.apply(
            inputs_embeds,
            previous_hidden_states,
            current_layer.e_proj.weight,
            current_layer.h_proj.weight,
            current_layer.e_proj.comm_group,
            current_layer.e_proj.tp_rank,
            current_layer.e_proj.tp_size,
        )
        hidden_states, _residual = current_layer.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        return hidden_states

    layer._speco_deepseek_v4_original_eh_forward = original_forward
    layer.forward = types.MethodType(fused_forward, layer)
    layer._speco_deepseek_v4_fused_eh_projection_collective = True
    _increment_training_diagnostic("tp_fused_eh_projection_collective_installations")
    return True


def _install_deterministic_tp_wo_b_forward(model) -> bool:
    from vllm.distributed import get_tp_group, split_tensor_along_last_dim

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("self_attn.wo_b")
    ]
    if not matches:
        raise RuntimeError(
            "Expected at least one DS4 draft wo_b module, found "
            f"{[name for name, _ in matches]}"
        )

    def deterministic_forward(layer, input_, _original_forward, **kwargs):
        custom_op = getattr(layer, "custom_op", None)
        deterministic_enabled = os.environ.get(
            "MTP_DS4_DETERMINISTIC_TP_ROW_FORWARD",
            "0",
        ).strip().lower() not in {"0", "false", "off", "no"}
        use_deterministic_training_path = (
            deterministic_enabled
            and torch.is_grad_enabled()
            and isinstance(input_, torch.Tensor)
            and input_.is_floating_point()
            and bool(getattr(layer, "reduce_results", False))
            and int(getattr(layer, "tp_size", 1)) > 1
        )
        if not use_deterministic_training_path:
            return _original_forward(input_, **kwargs)

        if custom_op is not None:
            input_parallel = custom_op.get_input_parallel(input_)
            comm_group = custom_op.comm_group
        elif bool(getattr(layer, "input_is_parallel", False)):
            input_parallel = input_
            comm_group = get_tp_group()
        else:
            input_partitions = split_tensor_along_last_dim(
                input_,
                num_partitions=int(layer.tp_size),
            )
            input_parallel = input_partitions[int(layer.tp_rank)].contiguous()
            comm_group = get_tp_group()
        skip_bias_add = bool(getattr(layer, "skip_bias_add", False))
        bias = (
            None
            if int(getattr(layer, "tp_rank", 0)) > 0 or skip_bias_add
            else getattr(layer, "bias", None)
        )
        local_output = layer.quant_method.apply(
            layer,
            input_parallel,
            bias=bias,
        )
        output = _DeterministicTPRowReduce.apply(
            local_output,
            comm_group,
        )
        _increment_training_diagnostic("tp_deterministic_wo_b_forward_calls")
        if bool(getattr(layer, "return_bias", False)):
            output_bias = getattr(layer, "bias", None) if skip_bias_add else None
            return output, output_bias
        return output

    for _, module in matches:
        if getattr(module, "_speco_deepseek_v4_deterministic_tp_forward", False):
            continue
        original_forward = module.forward

        def bound_forward(
            layer,
            input_,
            _original_forward=original_forward,
            **kwargs,
        ):
            return deterministic_forward(
                layer,
                input_,
                _original_forward,
                **kwargs,
            )

        module.forward = types.MethodType(bound_forward, module)
        module._speco_deepseek_v4_deterministic_tp_forward = True
        module._speco_deepseek_v4_original_forward = original_forward
    return True


def _patch_vllm_ascend_dsa_batchmatmul_training_backward() -> bool:
    try:
        import torch_npu
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    original_batchmatmul = getattr(torch_npu, "npu_transpose_batchmatmul", None)
    if original_batchmatmul is None:
        return False
    if getattr(original_batchmatmul, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    class _DSATransposeBatchMatmulGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, op, op_kwargs):
            groups = int(x.shape[1])
            input_dim = int(x.shape[2])
            with torch.no_grad():
                output = op(x, weight, **op_kwargs)
            output_rank = int(output.shape[-1])

            if weight.ndim == 2 and weight.numel() == groups * output_rank * input_dim:
                layout = "flat_grd"
            elif tuple(weight.shape) == (groups, output_rank, input_dim):
                layout = "grd"
            elif tuple(weight.shape) == (groups, input_dim, output_rank):
                layout = "gdr"
            else:
                raise RuntimeError(
                    "Unsupported DSA wo_a weight layout for npu_transpose_batchmatmul: "
                    f"input={tuple(x.shape)}, weight={tuple(weight.shape)}, "
                    f"output={tuple(output.shape)}"
                )

            ctx.layout = layout
            ctx.groups = groups
            ctx.input_dim = input_dim
            ctx.output_rank = output_rank
            ctx.input_dtype = x.dtype
            ctx.weight_dtype = weight.dtype
            ctx.save_for_backward(x, weight)
            return output.clone()

        @staticmethod
        @_profile_vllm_ascend_backward("backward_dsa_projection")
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            compute_dtype = (
                torch.float32
                if torch.float32 in (ctx.input_dtype, ctx.weight_dtype)
                else ctx.input_dtype
            )
            x_compute = x.to(compute_dtype)
            grad_compute = grad_output.to(compute_dtype)

            if ctx.layout == "flat_grd":
                grouped_weight = weight.reshape(
                    ctx.groups,
                    ctx.output_rank,
                    ctx.input_dim,
                )
            elif ctx.layout == "grd":
                grouped_weight = weight
            else:
                grouped_weight = weight.transpose(1, 2)
            grouped_weight = grouped_weight.to(compute_dtype)

            # Treat the DSA head/group dimension as the batch dimension.  The
            # former einsums lower to a generic contraction on TorchNPU and
            # become disproportionately expensive for the long, duplicated
            # In the context-cache path, these BMMs implement the same VJP but
            # dispatch the optimized batched matmul kernels directly:
            #
            #   dx[g] = dout[g] @ W[g]
            #   dW[g] = dout[g].T @ x[g]
            grad_by_group = grad_compute.permute(1, 0, 2).contiguous()

            grad_x = None
            if ctx.needs_input_grad[0]:
                grad_x = (
                    torch.bmm(
                        grad_by_group,
                        grouped_weight.contiguous(),
                    )
                    .permute(1, 0, 2)
                    .contiguous()
                )
                grad_x = grad_x.to(ctx.input_dtype)

            grad_weight = None
            if ctx.needs_input_grad[1]:
                grad_grouped_weight = torch.bmm(
                    grad_by_group.transpose(1, 2),
                    x_compute.permute(1, 0, 2).contiguous(),
                )
                if ctx.layout == "flat_grd":
                    grad_weight = grad_grouped_weight.reshape_as(weight)
                elif ctx.layout == "grd":
                    grad_weight = grad_grouped_weight
                else:
                    grad_weight = grad_grouped_weight.transpose(1, 2)
                grad_weight = grad_weight.to(ctx.weight_dtype)

            return grad_x, grad_weight, None, None

    def _autograd_batchmatmul(x, weight, *args, **kwargs):
        is_dsa_wo_a = (
            not args
            and isinstance(x, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and x.ndim == 3
            and kwargs.get("bias") is None
            and kwargs.get("scale") is None
            and kwargs.get("perm_x1") == (1, 0, 2)
            and kwargs.get("perm_x2") == (0, 1, 2)
            and kwargs.get("perm_y") == (1, 0, 2)
            and kwargs.get("batch_split_factor") == 1
        )
        if not torch.is_grad_enabled() or not is_dsa_wo_a:
            return original_batchmatmul(x, weight, *args, **kwargs)
        return _DSATransposeBatchMatmulGrad.apply(
            x,
            weight,
            original_batchmatmul,
            kwargs,
        )

    _autograd_batchmatmul._speco_deepseek_v4_train_backward_patch = True
    torch_npu.npu_transpose_batchmatmul = _autograd_batchmatmul
    return True


def _chunked_reference_vjp(
    reference,
    replay_inputs: list[torch.Tensor],
    grad_output: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor | None, ...]:
    """Accumulate a reference VJP without retaining its full output graph."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not replay_inputs:
        return ()

    accumulated: list[torch.Tensor | None] = [None] * len(replay_inputs)
    for chunk_start in range(0, int(grad_output.shape[0]), chunk_size):
        chunk_end = min(chunk_start + chunk_size, int(grad_output.shape[0]))
        chunk_output = reference(chunk_start, chunk_end)
        if int(chunk_output.shape[0]) != chunk_end - chunk_start:
            raise RuntimeError(
                "Reference VJP returned an unexpected number of rows: "
                f"range=[{chunk_start}, {chunk_end}), "
                f"output={tuple(chunk_output.shape)}"
            )
        chunk_grads = torch.autograd.grad(
            chunk_output,
            replay_inputs,
            grad_output[chunk_start:chunk_end],
            allow_unused=True,
        )
        for input_index, chunk_grad in enumerate(chunk_grads):
            if chunk_grad is None:
                continue
            if accumulated[input_index] is None:
                accumulated[input_index] = chunk_grad
            else:
                accumulated[input_index].add_(chunk_grad)
        _increment_training_diagnostic("dsa_backward_query_chunks")
    _increment_training_diagnostic("dsa_chunked_backward_calls")
    return tuple(accumulated)


def _patch_vllm_ascend_sparse_attention_training_backward() -> bool:
    global _DSA_TRAINING_SCATTER_WRAPPER
    global _DSA_TRAINING_SPARSE_ATTENTION_WRAPPER
    try:
        ascend_ops = torch.ops._C_ascend
        original_sparse_attention = ascend_ops.npu_sparse_attn_sharedkv
        original_scatter_update = ascend_ops.npu_scatter_nd_update_v2
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        _increment_training_diagnostic("dsa_training_op_patch_unavailable")
        return False
    if getattr(
        original_sparse_attention,
        "_speco_deepseek_v4_train_backward_patch",
        False,
    ):
        return True

    # Scatter is used by more caches than the two consumed by sparse
    # attention.  Entries keyed only by id(cache) used to survive cache
    # replacement and kept an entire autograd graph alive.  Pair every entry
    # with a weak reference so replacing a dynamically-sized cache also drops
    # its pending update.
    cache_updates: dict[
        int,
        tuple[Any, list[torch.Tensor], list[torch.Tensor]],
    ] = {}

    def _remember_cache_update(
        cache: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        cache_id = id(cache)

        def _discard_stale_entry(cache_ref, *, expected_id=cache_id):
            current = cache_updates.get(expected_id)
            if current is not None and current[0] is cache_ref:
                cache_updates.pop(expected_id, None)

        current = cache_updates.get(cache_id)
        if current is not None and current[0]() is cache:
            current[1].append(updates)
            current[2].append(slots)
            _increment_training_diagnostic("dsa_accumulated_cache_scatter_updates")
            return
        cache_ref = weakref.ref(cache, _discard_stale_entry)
        cache_updates[cache_id] = (cache_ref, [updates], [slots])

    def _take_cache_update(
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not isinstance(cache, torch.Tensor):
            _increment_training_diagnostic("dsa_cache_update_non_tensor_misses")
            return None
        entry = cache_updates.pop(id(cache), None)
        if entry is None:
            _increment_training_diagnostic("dsa_cache_update_misses")
            return None
        cache_ref, updates, slots = entry
        # Guard against CPython reusing the id of an already-released tensor.
        if cache_ref() is not cache:
            _increment_training_diagnostic("dsa_stale_cache_update_drops")
            return None
        _increment_training_diagnostic("dsa_cache_update_hits")
        if len(updates) == 1:
            return updates[0], slots[0]
        return torch.cat(updates, dim=0), torch.cat(slots, dim=0)

    def _pytorch_scatter_update_(
        cache: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        if indices.numel() == 0:
            return cache
        if indices.ndim == 1:
            coordinates = indices.reshape(-1, 1).long()
        else:
            coordinates = indices.reshape(-1, indices.shape[-1]).long()
        index_depth = int(coordinates.shape[-1])
        if index_depth not in (1, 2) or index_depth > cache.ndim:
            raise RuntimeError(
                "Unsupported training scatter layout: "
                f"cache={tuple(cache.shape)}, indices={tuple(indices.shape)}, "
                f"updates={tuple(updates.shape)}"
            )

        source = updates.reshape(coordinates.shape[0], *cache.shape[index_depth:])
        if index_depth == 1:
            cache.index_copy_(0, coordinates[:, 0], source)
            return cache

        flat_cache = cache.reshape(
            cache.shape[0] * cache.shape[1],
            *cache.shape[2:],
        )
        flat_indices = coordinates[:, 0] * cache.shape[1] + coordinates[:, 1]
        flat_cache.index_copy_(0, flat_indices, source)
        return cache

    def _autograd_scatter_update(cache, indices, updates, *args, **kwargs):
        if not torch.is_grad_enabled() or not isinstance(updates, torch.Tensor):
            return original_scatter_update(cache, indices, updates, *args, **kwargs)
        with torch.no_grad():
            result = _pytorch_scatter_update_(cache, indices, updates)
        # Cached DSA metadata may reuse and mutate its slot coordinates in a
        # later speculative step. Backward must retain this call's coordinates.
        slot_snapshot = indices.detach().clone()
        _remember_cache_update(cache, updates, slot_snapshot)
        _increment_training_diagnostic("dsa_scatter_slot_snapshots")
        return result

    def _paged_kv_sequence(
        cache: torch.Tensor,
        block_table: torch.Tensor,
        batch_index: int,
        sequence_length: int,
        update_source: torch.Tensor | None = None,
        update_slots: torch.Tensor | None = None,
        update_lookup: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        block_size = int(cache.shape[1])
        positions = torch.arange(sequence_length, device=cache.device, dtype=torch.long)
        logical_blocks = positions // block_size
        offsets = positions % block_size
        physical_blocks = (
            block_table[batch_index].long().index_select(0, logical_blocks)
        )
        sequence = cache[physical_blocks, offsets]
        if update_lookup is None:
            update_lookup = _prepare_update_slot_lookup(
                cache,
                update_source,
                update_slots,
            )
        if update_lookup is None:
            return sequence

        updates, slot_to_update = update_lookup
        physical_slots = physical_blocks * block_size + offsets
        update_indices = slot_to_update.index_select(0, physical_slots)
        replaced = update_indices >= 0
        safe_update_indices = update_indices.clamp_min(0)
        replacement = updates.index_select(0, safe_update_indices)
        replacement_mask = replaced.reshape(
            replaced.shape[0],
            *([1] * (sequence.ndim - 1)),
        )
        _increment_training_diagnostic("dsa_linear_slot_lookup_calls")
        return torch.where(replacement_mask, replacement, sequence)

    def _prepare_update_slot_lookup(
        cache: torch.Tensor,
        update_source: torch.Tensor | None,
        update_slots: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            update_source is None
            or update_slots is None
            or update_source.numel() == 0
            or update_slots.numel() == 0
        ):
            return None

        block_size = int(cache.shape[1])
        num_updates = int(update_source.shape[0])
        slot_coordinates = update_slots.reshape(num_updates, -1).long()
        if slot_coordinates.shape[1] == 1:
            slots = slot_coordinates[:, 0]
        elif slot_coordinates.shape[1] == 2:
            slots = slot_coordinates[:, 0] * block_size + slot_coordinates[:, 1]
        else:
            raise RuntimeError(
                "Unsupported paged-cache slot mapping layout: "
                f"source={tuple(update_source.shape)}, slots={tuple(update_slots.shape)}"
            )
        updates = update_source.reshape(num_updates, *cache.shape[2:])
        total_slots = int(cache.shape[0]) * block_size
        slot_to_update = torch.full(
            (total_slots,),
            -1,
            dtype=torch.long,
            device=cache.device,
        )
        source_indices = torch.arange(
            num_updates,
            dtype=torch.long,
            device=cache.device,
        )
        valid_slots = (slots >= 0) & (slots < total_slots)
        # Cache writes normally have unique slots.  If malformed metadata
        # contains an out-of-range coordinate, preserve the old "no match"
        # behavior instead of indexing outside the lookup table.
        slot_to_update.scatter_(
            0,
            slots[valid_slots],
            source_indices[valid_slots],
        )
        _increment_training_diagnostic("dsa_linear_slot_lookup_builds")
        return updates, slot_to_update

    def _shared_kv_attention_reference(
        q: torch.Tensor,
        sinks: torch.Tensor | None,
        op_kwargs: dict,
        ori_source: torch.Tensor | None = None,
        ori_slots: torch.Tensor | None = None,
        cmp_source: torch.Tensor | None = None,
        cmp_slots: torch.Tensor | None = None,
        host_metadata: tuple[list[int], list[int], list[int] | None] | None = None,
        query_start: int = 0,
        query_end: int | None = None,
    ) -> torch.Tensor:
        if op_kwargs.get("layout_q", "BSND") != "TND":
            raise RuntimeError(
                "Sparse-attention training fallback currently requires layout_q='TND'"
            )
        if op_kwargs.get("layout_kv", "PA_ND") != "PA_ND":
            raise RuntimeError(
                "Sparse-attention training fallback currently requires layout_kv='PA_ND'"
            )

        ori_kv = op_kwargs.get("ori_kv")
        ori_block_table = op_kwargs.get("ori_block_table")
        cu_seqlens_q = op_kwargs.get("cu_seqlens_q")
        seqused_kv = op_kwargs.get("seqused_kv")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (ori_kv, ori_block_table, cu_seqlens_q, seqused_kv)
        ):
            raise RuntimeError(
                "Sparse-attention training fallback requires paged ori_kv metadata"
            )
        if ori_kv.requires_grad:
            raise RuntimeError(
                "Paged ori_kv requires gradients, but the native cache scatter is not differentiable"
            )

        cmp_kv = op_kwargs.get("cmp_kv")
        cmp_block_table = op_kwargs.get("cmp_block_table")
        cmp_sparse_indices = op_kwargs.get("cmp_sparse_indices")
        cu_seqlens_cmp_kv = op_kwargs.get("cu_seqlens_cmp_kv")
        if isinstance(cmp_kv, torch.Tensor) and cmp_kv.requires_grad:
            raise RuntimeError(
                "Paged cmp_kv requires gradients, but the native compressor scatter is not differentiable"
            )

        if host_metadata is None:
            q_offsets = cu_seqlens_q.detach().cpu().tolist()
            kv_lengths = seqused_kv.detach().cpu().tolist()
            cmp_offsets = (
                cu_seqlens_cmp_kv.detach().cpu().tolist()
                if isinstance(cu_seqlens_cmp_kv, torch.Tensor)
                else None
            )
        else:
            q_offsets, kv_lengths, cmp_offsets = host_metadata
        softmax_scale = float(op_kwargs.get("softmax_scale", 0.0))
        cmp_ratio = int(op_kwargs.get("cmp_ratio", 0))
        ori_win_left = int(op_kwargs.get("ori_win_left", 128))
        ori_win_right = int(op_kwargs.get("ori_win_right", 0))
        num_q_heads = int(q.shape[1])
        total_queries = int(q.shape[0])
        query_start = int(query_start)
        query_end = total_queries if query_end is None else int(query_end)
        if not 0 <= query_start < query_end <= total_queries:
            raise ValueError(
                "Invalid DSA reference query range: "
                f"[{query_start}, {query_end}) for {total_queries} queries"
            )
        output_rows = []

        for batch_index in range(len(q_offsets) - 1):
            q_start = int(q_offsets[batch_index])
            q_end = int(q_offsets[batch_index + 1])
            selected_query_start = max(q_start, query_start)
            selected_query_end = min(q_end, query_end)
            if selected_query_start >= selected_query_end:
                continue
            q_length = q_end - q_start
            kv_length = int(kv_lengths[batch_index])
            ori_sequence = _paged_kv_sequence(
                ori_kv,
                ori_block_table,
                batch_index,
                kv_length,
                ori_source,
                ori_slots,
            )
            num_kv_heads = int(ori_sequence.shape[1])
            heads_per_kv = max(1, num_q_heads // num_kv_heads)
            kv_head_ids = (
                torch.arange(num_q_heads, device=q.device, dtype=torch.long)
                // heads_per_kv
            ).clamp_max(num_kv_heads - 1)
            query_position_start = kv_length - q_length

            cmp_sequence = None
            cmp_length = 0
            if isinstance(cmp_kv, torch.Tensor) and isinstance(
                cmp_block_table, torch.Tensor
            ):
                if cmp_offsets is not None and len(cmp_offsets) == len(q_offsets):
                    cmp_length = int(
                        cmp_offsets[batch_index + 1] - cmp_offsets[batch_index]
                    )
                elif cmp_ratio > 0:
                    cmp_length = kv_length // cmp_ratio
                if cmp_length > 0:
                    cmp_sequence = _paged_kv_sequence(
                        cmp_kv,
                        cmp_block_table,
                        batch_index,
                        cmp_length,
                        cmp_source,
                        cmp_slots,
                    )

            for global_query_index in range(
                selected_query_start,
                selected_query_end,
            ):
                local_query_index = global_query_index - q_start
                query_position = query_position_start + local_query_index
                ori_start = max(0, query_position - ori_win_left)
                ori_end = min(kv_length, query_position + ori_win_right + 1)
                selected = ori_sequence[ori_start:ori_end]

                if cmp_sequence is not None and isinstance(
                    cmp_sparse_indices, torch.Tensor
                ):
                    sparse_ids = cmp_sparse_indices[global_query_index]
                    if sparse_ids.ndim == 2:
                        sparse_ids = sparse_ids[0]
                    sparse_ids = sparse_ids.long()
                    sparse_ids = sparse_ids[
                        (sparse_ids >= 0) & (sparse_ids < cmp_length)
                    ]
                    if sparse_ids.numel() > 0:
                        selected = torch.cat(
                            (selected, cmp_sequence.index_select(0, sparse_ids)),
                            dim=0,
                        )

                selected = selected.index_select(1, kv_head_ids).transpose(0, 1)
                query = q[global_query_index]
                selected_float = selected.float()
                scores = torch.einsum(
                    "hd,hkd->hk",
                    query.float(),
                    selected_float,
                )
                scores = scores * softmax_scale
                if sinks is not None and sinks.numel() > 0:
                    scores_with_sink = torch.cat(
                        (scores, sinks.float().reshape(num_q_heads, 1)),
                        dim=-1,
                    )
                    probabilities = torch.softmax(scores_with_sink, dim=-1)[..., :-1]
                else:
                    probabilities = torch.softmax(scores, dim=-1)
                output_rows.append(
                    torch.einsum(
                        "hk,hkd->hd",
                        probabilities,
                        selected_float,
                    ).to(q.dtype)
                )

        return torch.stack(output_rows, dim=0)

    class _FusedAttentionSinkGradient(torch.autograd.Function):
        """Attach the sink gradient to the native FusionAttention output.

        FusionAttention uses a detached synthetic PSE sink when computing its
        q/k/v gradients.  This node adds the exact gradient for the learned
        sink logit from the fused softmax statistics.
        """

        @staticmethod
        def forward(ctx, output, sinks, sink_probability):
            ctx.output_dtype = output.dtype
            ctx.sinks_dtype = sinks.dtype
            ctx.save_for_backward(
                output.detach(),
                sink_probability.detach(),
            )
            return output

        @staticmethod
        @_profile_vllm_ascend_backward("backward_attention_sink")
        def backward(ctx, grad_output):
            output, sink_probability = ctx.saved_tensors
            grad_sinks = -(
                sink_probability.float()
                * (grad_output.float() * output.float()).sum(
                    dim=-1,
                    keepdim=True,
                )
            ).sum(dim=(0, 2, 3))
            return (
                grad_output,
                grad_sinks.to(ctx.sinks_dtype),
                None,
            )

    def _fusion_softmax_stat_scalar(stat: torch.Tensor) -> torch.Tensor:
        # FlashAttentionScore exposes max/sum as [B, N, S, 8] on the
        # supported Ascend training kernels, with the scalar repeated in the
        # final lane.  Accept an already-squeezed representation as well.
        if stat.ndim == 4:
            return stat[..., :1]
        if stat.ndim == 3:
            return stat.unsqueeze(-1)
        raise RuntimeError(
            f"Unexpected FusionAttention softmax statistic shape: {tuple(stat.shape)}"
        )

    def _shared_kv_attention_reference_vectorized(
        q: torch.Tensor,
        sinks: torch.Tensor | None,
        op_kwargs: dict,
        ori_source: torch.Tensor | None = None,
        ori_slots: torch.Tensor | None = None,
        cmp_source: torch.Tensor | None = None,
        cmp_slots: torch.Tensor | None = None,
        grad_output: torch.Tensor | None = None,
        gradient_needs: tuple[bool, bool, bool, bool] | None = None,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ):
        """Replay shared-KV attention in bounded query chunks.

        The serial reference above launches several small operations per query
        token.  This version keeps the same key order and masking semantics but
        batches the score, softmax, and value-reduction work across queries.
        """
        if op_kwargs.get("layout_q", "BSND") != "TND":
            raise RuntimeError(
                "Sparse-attention training fallback currently requires layout_q='TND'"
            )
        if op_kwargs.get("layout_kv", "PA_ND") != "PA_ND":
            raise RuntimeError(
                "Sparse-attention training fallback currently requires "
                "layout_kv='PA_ND'"
            )

        ori_kv = op_kwargs.get("ori_kv")
        ori_block_table = op_kwargs.get("ori_block_table")
        cu_seqlens_q = op_kwargs.get("cu_seqlens_q")
        seqused_kv = op_kwargs.get("seqused_kv")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                ori_kv,
                ori_block_table,
                cu_seqlens_q,
                seqused_kv,
            )
        ):
            raise RuntimeError(
                "Sparse-attention training fallback requires paged ori_kv metadata"
            )
        if ori_kv.requires_grad:
            raise RuntimeError(
                "Paged ori_kv requires gradients, but the native cache "
                "scatter is not differentiable"
            )

        cmp_kv = op_kwargs.get("cmp_kv")
        cmp_block_table = op_kwargs.get("cmp_block_table")
        cmp_sparse_indices = op_kwargs.get("cmp_sparse_indices")
        cu_seqlens_cmp_kv = op_kwargs.get("cu_seqlens_cmp_kv")
        if isinstance(cmp_kv, torch.Tensor) and cmp_kv.requires_grad:
            raise RuntimeError(
                "Paged cmp_kv requires gradients, but the native compressor "
                "scatter is not differentiable"
            )
        ori_update_lookup = _prepare_update_slot_lookup(
            ori_kv,
            ori_source,
            ori_slots,
        )
        if cmp_kv is ori_kv and cmp_source is ori_source and cmp_slots is ori_slots:
            cmp_update_lookup = ori_update_lookup
        else:
            cmp_update_lookup = (
                _prepare_update_slot_lookup(
                    cmp_kv,
                    cmp_source,
                    cmp_slots,
                )
                if isinstance(cmp_kv, torch.Tensor)
                else None
            )

        streaming_vjp = grad_output is not None
        if streaming_vjp:
            if gradient_needs is None or len(gradient_needs) != 4:
                raise RuntimeError(
                    "Streaming sparse-attention VJP requires four gradient flags"
                )
            if tuple(grad_output.shape) != tuple(q.shape):
                raise RuntimeError(
                    "Sparse-attention output gradient has an unexpected shape: "
                    f"q={tuple(q.shape)}, grad={tuple(grad_output.shape)}"
                )
            need_q, need_sinks, need_ori_source, need_cmp_source = gradient_needs
            q = q.detach()
            if sinks is not None:
                sinks = sinks.detach().requires_grad_(need_sinks)
            if ori_source is not None:
                ori_source = ori_source.detach().requires_grad_(need_ori_source)
            if cmp_source is not None:
                cmp_source = cmp_source.detach().requires_grad_(need_cmp_source)
            grad_q = torch.zeros_like(q) if need_q else None
            grad_sinks = (
                torch.zeros_like(sinks) if need_sinks and sinks is not None else None
            )
            grad_ori_source = (
                torch.zeros_like(ori_source)
                if need_ori_source and ori_source is not None
                else None
            )
            grad_cmp_source = (
                torch.zeros_like(cmp_source)
                if need_cmp_source and cmp_source is not None
                else None
            )

        fused_attention_op = None
        if not streaming_vjp and _env_flag(
            "MTP_DS4_FUSED_SPARSE_ATTN_REPLAY",
            True,
        ):
            try:
                import torch_npu

                fused_attention_op = getattr(
                    torch_npu,
                    "npu_fusion_attention",
                    None,
                )
            except ImportError:
                fused_attention_op = None
            if fused_attention_op is None:
                _increment_training_diagnostic("dsa_fused_attention_replay_unavailable")

        q_offsets = cu_seqlens_q.detach().cpu().tolist()
        kv_lengths = seqused_kv.detach().cpu().tolist()
        cmp_offsets = (
            cu_seqlens_cmp_kv.detach().cpu().tolist()
            if isinstance(cu_seqlens_cmp_kv, torch.Tensor)
            else None
        )
        softmax_scale = float(op_kwargs.get("softmax_scale", 0.0))
        cmp_ratio = int(op_kwargs.get("cmp_ratio", 0))
        ori_win_left = int(op_kwargs.get("ori_win_left", 128))
        ori_win_right = int(op_kwargs.get("ori_win_right", 0))
        if ori_win_left < 0 or ori_win_right < 0:
            raise RuntimeError(
                "Vectorized sparse-attention replay requires non-negative "
                f"window sizes, got left={ori_win_left}, right={ori_win_right}"
            )
        checkpoint_chunks = (
            _env_flag(
                "MTP_DS4_CHECKPOINT_SPARSE_ATTN_REPLAY",
                False,
            )
            and not streaming_vjp
        )
        checkpoint_fn = None
        if checkpoint_chunks:
            from torch.utils.checkpoint import checkpoint as checkpoint_fn
        try:
            query_chunk_size = max(
                1,
                int(
                    os.environ.get(
                        "MTP_DS4_SPARSE_ATTN_REPLAY_QUERY_CHUNK_SIZE",
                        ("256" if fused_attention_op is not None else "32"),
                    )
                ),
            )
        except ValueError as exc:
            raise RuntimeError(
                "MTP_DS4_SPARSE_ATTN_REPLAY_QUERY_CHUNK_SIZE must be an integer"
            ) from exc
        try:
            fused_attention_inner_precise = int(
                os.environ.get(
                    "MTP_DS4_FUSED_ATTN_INNER_PRECISE",
                    "1",
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "MTP_DS4_FUSED_ATTN_INNER_PRECISE must be 0 or 1"
            ) from exc
        if fused_attention_inner_precise not in (0, 1):
            raise RuntimeError(
                "MTP_DS4_FUSED_ATTN_INNER_PRECISE must be 0 or 1, "
                f"got {fused_attention_inner_precise}"
            )

        num_q_heads = int(q.shape[1])
        head_dim = int(q.shape[2])
        window_size = max(1, ori_win_left + ori_win_right + 1)
        window_offsets = torch.arange(
            window_size,
            device=q.device,
            dtype=torch.long,
        )
        output_chunks = []

        def _attention_chunk(
            q_chunk: torch.Tensor,
            ori_sequence: torch.Tensor,
            cmp_sequence: torch.Tensor,
            query_positions: torch.Tensor,
            sparse_ids: torch.Tensor,
            sink_values: torch.Tensor,
        ) -> torch.Tensor:
            """Replay one chunk without retaining its large gather tensors."""
            chunk_length = int(q_chunk.shape[0])
            ori_starts = (query_positions - ori_win_left).clamp_min(0)
            ori_ends = (query_positions + ori_win_right + 1).clamp_max(
                int(ori_sequence.shape[0])
            )
            ori_indices = ori_starts.unsqueeze(1) + window_offsets.unsqueeze(0)
            ori_valid = ori_indices < ori_ends.unsqueeze(1)
            safe_ori_indices = ori_indices.clamp_max(int(ori_sequence.shape[0]) - 1)
            num_ori_heads = int(ori_sequence.shape[1])
            ori_heads_per_q = max(1, num_q_heads // num_ori_heads)
            ori_head_ids = (
                torch.arange(
                    num_q_heads,
                    device=q_chunk.device,
                    dtype=torch.long,
                )
                // ori_heads_per_q
            ).clamp_max(num_ori_heads - 1)
            selected_ori = (
                ori_sequence.index_select(
                    0,
                    safe_ori_indices.reshape(-1),
                )
                .reshape(
                    chunk_length,
                    window_size,
                    num_ori_heads,
                    head_dim,
                )
                .index_select(2, ori_head_ids)
                .permute(0, 2, 1, 3)
            )
            ori_scores = torch.einsum(
                "chd,chwd->chw",
                q_chunk.float(),
                selected_ori.float(),
            )
            ori_scores = (ori_scores * softmax_scale).masked_fill(
                ~ori_valid.unsqueeze(1),
                float("-inf"),
            )

            score_parts = [ori_scores]
            selected_cmp = None
            if cmp_sequence.numel() > 0 and sparse_ids.numel() > 0:
                cmp_length = int(cmp_sequence.shape[0])
                sparse_valid = (sparse_ids >= 0) & (sparse_ids < cmp_length)
                safe_sparse_ids = sparse_ids.clamp(
                    min=0,
                    max=cmp_length - 1,
                )
                sparse_count = int(safe_sparse_ids.shape[1])
                num_cmp_heads = int(cmp_sequence.shape[1])
                cmp_heads_per_q = max(1, num_q_heads // num_cmp_heads)
                cmp_head_ids = (
                    torch.arange(
                        num_q_heads,
                        device=q_chunk.device,
                        dtype=torch.long,
                    )
                    // cmp_heads_per_q
                ).clamp_max(num_cmp_heads - 1)
                selected_cmp = (
                    cmp_sequence.index_select(
                        0,
                        safe_sparse_ids.reshape(-1),
                    )
                    .reshape(
                        chunk_length,
                        sparse_count,
                        num_cmp_heads,
                        head_dim,
                    )
                    .index_select(2, cmp_head_ids)
                    .permute(0, 2, 1, 3)
                )
                cmp_scores = torch.einsum(
                    "chd,chsd->chs",
                    q_chunk.float(),
                    selected_cmp.float(),
                )
                cmp_scores = (cmp_scores * softmax_scale).masked_fill(
                    ~sparse_valid.unsqueeze(1),
                    float("-inf"),
                )
                score_parts.append(cmp_scores)

            scores = torch.cat(score_parts, dim=-1)
            if sink_values.numel() > 0:
                sink_scores = (
                    sink_values.float()
                    .reshape(
                        1,
                        num_q_heads,
                        1,
                    )
                    .expand(chunk_length, -1, -1)
                )
                scores = torch.cat((scores, sink_scores), dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            ori_count = int(selected_ori.shape[2])
            output = torch.einsum(
                "chw,chwd->chd",
                probabilities[..., :ori_count],
                selected_ori.float(),
            )
            if selected_cmp is not None:
                cmp_count = int(selected_cmp.shape[2])
                output = output + torch.einsum(
                    "chs,chsd->chd",
                    probabilities[
                        ...,
                        ori_count : ori_count + cmp_count,
                    ],
                    selected_cmp.float(),
                )
            return output.to(q_chunk.dtype)

        def _fused_attention_chunk(
            q_chunk: torch.Tensor,
            ori_sequence: torch.Tensor,
            cmp_sequence: torch.Tensor,
            query_positions: torch.Tensor,
            sparse_ids: torch.Tensor,
            sink_values: torch.Tensor,
        ) -> torch.Tensor:
            """Replay a selected shared-KV chunk with native FA backward."""
            chunk_length = int(q_chunk.shape[0])
            ori_length = int(ori_sequence.shape[0])
            num_kv_heads = int(ori_sequence.shape[1])
            ori_starts = (query_positions - ori_win_left).clamp_min(0)
            ori_ends = (query_positions + ori_win_right + 1).clamp_max(ori_length)
            ori_indices = ori_starts.unsqueeze(1) + window_offsets.unsqueeze(0)
            ori_valid = ori_indices < ori_ends.unsqueeze(1)
            safe_ori_indices = ori_indices.clamp_max(ori_length - 1)
            selected_ori = (
                ori_sequence.index_select(
                    0,
                    safe_ori_indices.reshape(-1),
                )
                .reshape(
                    chunk_length,
                    window_size,
                    num_kv_heads,
                    head_dim,
                )
                .permute(0, 2, 1, 3)
            )

            kv_parts = [selected_ori]
            valid_parts = [ori_valid]
            if cmp_sequence.numel() > 0 and sparse_ids.numel() > 0:
                cmp_length = int(cmp_sequence.shape[0])
                num_cmp_heads = int(cmp_sequence.shape[1])
                if num_cmp_heads != num_kv_heads:
                    raise RuntimeError(
                        "FusionAttention replay requires matching original "
                        "and compressed KV head counts: "
                        f"ori={num_kv_heads}, cmp={num_cmp_heads}"
                    )
                sparse_valid = (sparse_ids >= 0) & (sparse_ids < cmp_length)
                safe_sparse_ids = sparse_ids.clamp(
                    min=0,
                    max=cmp_length - 1,
                )
                sparse_count = int(safe_sparse_ids.shape[1])
                selected_cmp = (
                    cmp_sequence.index_select(
                        0,
                        safe_sparse_ids.reshape(-1),
                    )
                    .reshape(
                        chunk_length,
                        sparse_count,
                        num_kv_heads,
                        head_dim,
                    )
                    .permute(0, 2, 1, 3)
                )
                kv_parts.append(selected_cmp)
                valid_parts.append(sparse_valid)

            selected_kv = torch.cat(kv_parts, dim=2)
            valid = torch.cat(valid_parts, dim=1)
            pse = None
            effective_sink = None
            if sink_values.numel() > 0:
                # TorchNPU 7.3 exposes a native sink argument, but the shipped
                # backward can still dispatch FlashAttentionScoreGradV3 and
                # produce an invalid dSink buffer.  Represent the sink as an
                # extra zero-valued KV token instead.  This preserves native
                # q/k/v backward while avoiding the broken native dSink path.
                sink_kv = selected_kv.new_zeros(
                    (chunk_length, num_kv_heads, 1, head_dim)
                )
                selected_kv = torch.cat(
                    (selected_kv, sink_kv),
                    dim=2,
                )
                valid = torch.cat(
                    (
                        valid,
                        torch.ones(
                            (chunk_length, 1),
                            dtype=torch.bool,
                            device=q_chunk.device,
                        ),
                    ),
                    dim=1,
                )
                # This TorchNPU schema has no pse_type argument and therefore
                # uses score = scale * (QK + PSE).  Divide by scale so the
                # synthetic token receives the intended unscaled sink logit.
                if softmax_scale == 0.0:
                    raise RuntimeError(
                        "FusionAttention sink emulation requires a non-zero "
                        "softmax scale"
                    )
                pse = q_chunk.new_zeros((1, num_q_heads, 1, int(selected_kv.shape[2])))
                pse[..., -1] = (
                    (sink_values.detach().float() / softmax_scale)
                    .to(q_chunk.dtype)
                    .reshape(1, num_q_heads, 1)
                )
                effective_sink = pse[..., -1:].float() * softmax_scale

            attention_mask = (~valid).reshape(
                chunk_length,
                1,
                1,
                int(selected_kv.shape[2]),
            )
            fused_result = fused_attention_op(
                q_chunk.unsqueeze(2),
                selected_kv,
                selected_kv,
                num_q_heads,
                "BNSD",
                pse=pse,
                atten_mask=attention_mask,
                scale=softmax_scale,
                keep_prob=1.0,
                inner_precise=fused_attention_inner_precise,
                # This is a complete per-batch mask, so allMask is the correct
                # mode.  It also avoids the historical pre_tockens/
                # next_tockens spelling difference between TorchNPU releases.
                sparse_mode=1,
            )
            output = fused_result[0]
            if sink_values.numel() > 0:
                softmax_max = _fusion_softmax_stat_scalar(fused_result[1]).float()
                softmax_sum = _fusion_softmax_stat_scalar(fused_result[2]).float()
                sink_probability = torch.exp(effective_sink - softmax_max) / softmax_sum
                output = _FusedAttentionSinkGradient.apply(
                    output,
                    sink_values,
                    sink_probability,
                )
            _increment_training_diagnostic("dsa_fused_attention_replay_chunks")
            return output.squeeze(2)

        for batch_index in range(len(q_offsets) - 1):
            q_start = int(q_offsets[batch_index])
            q_end = int(q_offsets[batch_index + 1])
            q_length = q_end - q_start
            if q_length <= 0:
                continue
            kv_length = int(kv_lengths[batch_index])
            if kv_length <= 0:
                raise RuntimeError(
                    "Vectorized sparse-attention replay received an empty "
                    f"KV sequence for batch index {batch_index}"
                )

            ori_sequence = _paged_kv_sequence(
                ori_kv,
                ori_block_table,
                batch_index,
                kv_length,
                ori_source,
                ori_slots,
                ori_update_lookup,
            )
            query_position_start = kv_length - q_length

            cmp_length = 0
            cmp_sequence = q.new_empty((0, 0, head_dim))
            if isinstance(cmp_kv, torch.Tensor) and isinstance(
                cmp_block_table, torch.Tensor
            ):
                if cmp_offsets is not None and len(cmp_offsets) == len(q_offsets):
                    cmp_length = int(
                        cmp_offsets[batch_index + 1] - cmp_offsets[batch_index]
                    )
                elif cmp_ratio > 0:
                    cmp_length = kv_length // cmp_ratio
                if cmp_length > 0:
                    cmp_sequence = _paged_kv_sequence(
                        cmp_kv,
                        cmp_block_table,
                        batch_index,
                        cmp_length,
                        cmp_source,
                        cmp_slots,
                        cmp_update_lookup,
                    )

            batch_grad_ori_sequence = (
                torch.zeros_like(ori_sequence)
                if (
                    streaming_vjp
                    and need_ori_source
                    and ori_source is not None
                    and ori_sequence.requires_grad
                )
                else None
            )
            batch_grad_cmp_sequence = (
                torch.zeros_like(cmp_sequence)
                if (
                    streaming_vjp
                    and need_cmp_source
                    and cmp_source is not None
                    and cmp_sequence.requires_grad
                )
                else None
            )

            for chunk_start in range(0, q_length, query_chunk_size):
                chunk_end = min(
                    chunk_start + query_chunk_size,
                    q_length,
                )
                chunk_length = chunk_end - chunk_start
                q_chunk = q[q_start + chunk_start : q_start + chunk_end]
                if streaming_vjp:
                    q_chunk = q_chunk.detach().requires_grad_(need_q)
                query_positions = query_position_start + torch.arange(
                    chunk_start,
                    chunk_end,
                    device=q.device,
                    dtype=torch.long,
                )
                sparse_ids = q.new_empty(
                    (chunk_length, 0),
                    dtype=torch.long,
                )
                if cmp_sequence.numel() > 0 and isinstance(
                    cmp_sparse_indices, torch.Tensor
                ):
                    global_query_ids = torch.arange(
                        q_start + chunk_start,
                        q_start + chunk_end,
                        device=q.device,
                        dtype=torch.long,
                    )
                    sparse_ids = cmp_sparse_indices.index_select(
                        0,
                        global_query_ids,
                    )
                    if sparse_ids.ndim == 3:
                        sparse_ids = sparse_ids[:, 0, :]
                    elif sparse_ids.ndim != 2:
                        raise RuntimeError(
                            "Unsupported cmp_sparse_indices layout for "
                            "vectorized replay: "
                            f"{tuple(cmp_sparse_indices.shape)}"
                        )
                    sparse_ids = sparse_ids.long()
                sink_values = (
                    sinks if sinks is not None else q.new_empty(0, dtype=torch.float32)
                )
                chunk_args = (
                    q_chunk,
                    ori_sequence,
                    cmp_sequence,
                    query_positions,
                    sparse_ids,
                    sink_values,
                )
                if fused_attention_op is not None:
                    chunk_output = _fused_attention_chunk(*chunk_args)
                elif checkpoint_fn is not None:
                    chunk_output = checkpoint_fn(
                        _attention_chunk,
                        *chunk_args,
                        use_reentrant=False,
                    )
                else:
                    chunk_output = _attention_chunk(*chunk_args)
                if not streaming_vjp:
                    output_chunks.append(chunk_output)
                    continue

                replay_inputs = []
                replay_input_names = []
                if need_q:
                    replay_inputs.append(q_chunk)
                    replay_input_names.append("q")
                if need_sinks and sinks is not None:
                    replay_inputs.append(sinks)
                    replay_input_names.append("sinks")
                if batch_grad_ori_sequence is not None:
                    replay_inputs.append(ori_sequence)
                    replay_input_names.append("ori_sequence")
                if batch_grad_cmp_sequence is not None:
                    replay_inputs.append(cmp_sequence)
                    replay_input_names.append("cmp_sequence")
                replay_grads = (
                    torch.autograd.grad(
                        chunk_output,
                        replay_inputs,
                        grad_output[q_start + chunk_start : q_start + chunk_end],
                        allow_unused=True,
                    )
                    if replay_inputs
                    else ()
                )
                grad_by_name = dict(zip(replay_input_names, replay_grads))
                q_chunk_grad = grad_by_name.get("q")
                if q_chunk_grad is not None:
                    grad_q[q_start + chunk_start : q_start + chunk_end].copy_(
                        q_chunk_grad
                    )
                sink_chunk_grad = grad_by_name.get("sinks")
                if sink_chunk_grad is not None:
                    grad_sinks.add_(sink_chunk_grad)
                ori_sequence_grad = grad_by_name.get("ori_sequence")
                if ori_sequence_grad is not None:
                    batch_grad_ori_sequence.add_(ori_sequence_grad)
                cmp_sequence_grad = grad_by_name.get("cmp_sequence")
                if cmp_sequence_grad is not None:
                    batch_grad_cmp_sequence.add_(cmp_sequence_grad)

            if batch_grad_ori_sequence is not None:
                ori_source_grad = torch.autograd.grad(
                    ori_sequence,
                    ori_source,
                    batch_grad_ori_sequence,
                    allow_unused=True,
                )[0]
                if ori_source_grad is not None:
                    grad_ori_source.add_(ori_source_grad)
            if batch_grad_cmp_sequence is not None:
                cmp_source_grad = torch.autograd.grad(
                    cmp_sequence,
                    cmp_source,
                    batch_grad_cmp_sequence,
                    allow_unused=True,
                )[0]
                if cmp_source_grad is not None:
                    grad_cmp_source.add_(cmp_source_grad)

        if streaming_vjp:
            return (
                grad_q,
                grad_sinks,
                grad_ori_source,
                grad_cmp_source,
            )
        if not output_chunks:
            return q.new_empty(q.shape)
        return torch.cat(output_chunks, dim=0)

    def _explicit_sparse_attention_vjp(
        ctx,
        q,
        sinks,
        ori_source,
        ori_slots,
        cmp_source,
        cmp_slots,
        grad_output,
    ):
        chunk_size = _env_nonnegative_int(
            "MTP_DS4_DSA_BACKWARD_QUERY_CHUNK_SIZE",
            0,
        )
        if chunk_size == 0:
            chunk_size = max(1, int(grad_output.shape[0]))
        host_metadata = ctx.host_metadata
        if host_metadata is None:
            cu_seqlens_q = ctx.op_kwargs["cu_seqlens_q"]
            seqused_kv = ctx.op_kwargs["seqused_kv"]
            cu_seqlens_cmp_kv = ctx.op_kwargs.get("cu_seqlens_cmp_kv")
            host_metadata = (
                cu_seqlens_q.detach().cpu().tolist(),
                seqused_kv.detach().cpu().tolist(),
                (
                    cu_seqlens_cmp_kv.detach().cpu().tolist()
                    if isinstance(cu_seqlens_cmp_kv, torch.Tensor)
                    else None
                ),
            )
        selected_kv_row_budget = _env_nonnegative_int(
            "MTP_DS4_DSA_BACKWARD_SELECTED_KV_ROW_BUDGET",
            2048,
        )
        max_queries_per_batch = _env_nonnegative_int(
            "MTP_DS4_DSA_BACKWARD_MAX_QUERIES_PER_BATCH",
            8,
        )
        if selected_kv_row_budget == 0 or max_queries_per_batch == 0:
            selected_kv_row_budget = 1
            max_queries_per_batch = 1
        backward_function = explicit_shared_kv_attention_backward
        diagnostic_kwargs = {}
        dsa_host_totals = None
        if _VLLM_ASCEND_BACKWARD_PROFILE_ENABLED:
            from verl_speco.ops.deepseek_v4_dsa_diagnostics import (
                explicit_shared_kv_attention_backward as diagnostic_backward,
            )

            backward_function = diagnostic_backward
            dsa_host_totals = {}
            diagnostic_kwargs["phase_totals"] = dsa_host_totals
        with torch.no_grad():
            result = backward_function(
                q,
                sinks if ctx.has_sinks else None,
                ori_source if ctx.has_ori_source else None,
                ori_slots if ctx.has_ori_source else None,
                cmp_source if ctx.has_cmp_source else None,
                cmp_slots if ctx.has_cmp_source else None,
                ctx.op_kwargs,
                host_metadata,
                grad_output,
                chunk_size,
                selected_kv_row_budget,
                max_queries_per_batch,
                need_q=ctx.needs_input_grad[1],
                need_sinks=ctx.has_sinks and ctx.needs_input_grad[2],
                need_ori_source=(ctx.has_ori_source and ctx.needs_input_grad[3]),
                need_cmp_source=(ctx.has_cmp_source and ctx.needs_input_grad[5]),
                reuse_temporary_structures=_env_flag(
                    "MTP_DS4_REUSE_DSA_BACKWARD_STRUCTURES",
                    False,
                ),
                group_queries_by_kv_length=_env_flag(
                    "MTP_DS4_DSA_BACKWARD_GROUP_BY_KV_LENGTH",
                    False,
                ),
                **diagnostic_kwargs,
            )
        if dsa_host_totals is not None:
            # Aggregate on the host: no per-query device events or syncs.
            # One summary record per category per DSA call, bounded storage.
            for phase_name, (phase_calls, host_seconds) in dsa_host_totals.items():
                _VLLM_ASCEND_BACKWARD_PROFILE_RECORDS.append(
                    (phase_name, host_seconds, None, None)
                )
                _add_training_diagnostic(phase_name + "_calls", phase_calls)
        (
            grad_q,
            grad_sinks,
            grad_ori_source,
            grad_cmp_source,
            query_chunks,
            query_batches,
            max_batched_queries,
            max_batched_selected_kv_rows,
            structure_cache_hits,
            structure_cache_misses,
            contiguous_query_batches,
            structure_cache_evictions,
        ) = result
        _increment_training_diagnostic("dsa_explicit_backward_calls")
        _increment_training_diagnostic("dsa_chunked_backward_calls")
        _add_training_diagnostic("dsa_backward_query_chunks", query_chunks)
        _add_training_diagnostic("dsa_query_batching_batches", query_batches)
        _add_training_diagnostic(
            "dsa_vectorized_selected_kv_batches",
            query_batches,
        )
        _add_training_diagnostic(
            "dsa_query_batching_queries", int(grad_output.shape[0])
        )
        _record_max_training_diagnostic(
            "dsa_query_batching_max_queries", max_batched_queries
        )
        _record_max_training_diagnostic(
            "dsa_query_batching_max_selected_kv_rows",
            max_batched_selected_kv_rows,
        )
        _add_training_diagnostic(
            "dsa_backward_structure_cache_hits", structure_cache_hits
        )
        _add_training_diagnostic(
            "dsa_backward_structure_cache_misses", structure_cache_misses
        )
        _add_training_diagnostic(
            "dsa_contiguous_query_batch_slices", contiguous_query_batches
        )
        _add_training_diagnostic(
            "dsa_backward_structure_cache_evictions", structure_cache_evictions
        )
        return grad_q, grad_sinks, grad_ori_source, grad_cmp_source

    class _SparseAttentionGrad(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            native_output,
            q,
            sinks,
            ori_source,
            ori_slots,
            cmp_source,
            cmp_slots,
            op_kwargs,
            host_metadata,
        ):
            ctx.has_sinks = sinks is not None
            ctx.has_ori_source = ori_source is not None
            ctx.has_cmp_source = cmp_source is not None
            ctx.op_kwargs = op_kwargs
            ctx.host_metadata = host_metadata
            if sinks is None:
                sinks = q.new_empty(0, dtype=torch.float32)
            if ori_source is None:
                ori_source = q.new_empty(0)
                ori_slots = q.new_empty(0, dtype=torch.long)
            if cmp_source is None:
                cmp_source = q.new_empty(0)
                cmp_slots = q.new_empty(0, dtype=torch.long)
            ctx.save_for_backward(
                q,
                sinks,
                ori_source,
                ori_slots,
                cmp_source,
                cmp_slots,
            )
            return native_output.clone()

        @staticmethod
        @_profile_vllm_ascend_backward("backward_dsa")
        def backward(ctx, grad_output):
            # Every sparse-attention call has already consumed its matching
            # scatter entry during forward.  Anything left belongs to caches
            # that never feed this operator and must not pin their update
            # graphs across optimizer steps.
            if cache_updates:
                _increment_training_diagnostic("dsa_unconsumed_cache_updates_cleared")
                cache_updates.clear()
            q, sinks, ori_source, ori_slots, cmp_source, cmp_slots = ctx.saved_tensors
            if _env_flag("MTP_DS4_EXPLICIT_DSA_BACKWARD", True):
                (
                    grad_q,
                    grad_sinks,
                    grad_ori_source,
                    grad_cmp_source,
                ) = _explicit_sparse_attention_vjp(
                    ctx,
                    q,
                    sinks,
                    ori_source,
                    ori_slots,
                    cmp_source,
                    cmp_slots,
                    grad_output,
                )
                return (
                    None,
                    grad_q,
                    grad_sinks,
                    grad_ori_source,
                    None,
                    grad_cmp_source,
                    None,
                    None,
                    None,
                )
            vectorized_replay = _env_flag(
                "MTP_DS4_VECTORIZED_SPARSE_ATTN_REPLAY",
                True,
            )
            streaming_vjp = _env_flag(
                "MTP_DS4_STREAMING_SPARSE_ATTN_VJP",
                False,
            )
            with torch.enable_grad():
                if vectorized_replay and streaming_vjp:
                    (
                        grad_q,
                        grad_sinks,
                        grad_ori_source,
                        grad_cmp_source,
                    ) = _shared_kv_attention_reference_vectorized(
                        q,
                        sinks if ctx.has_sinks else None,
                        ctx.op_kwargs,
                        ori_source if ctx.has_ori_source else None,
                        ori_slots if ctx.has_ori_source else None,
                        cmp_source if ctx.has_cmp_source else None,
                        cmp_slots if ctx.has_cmp_source else None,
                        grad_output=grad_output,
                        gradient_needs=(
                            ctx.needs_input_grad[1],
                            ctx.needs_input_grad[2],
                            ctx.needs_input_grad[3],
                            ctx.needs_input_grad[5],
                        ),
                    )
                    _increment_training_diagnostic(
                        "dsa_streaming_sparse_attention_backward_calls"
                    )
                else:
                    q_replay = q.detach().requires_grad_(ctx.needs_input_grad[1])
                    sinks_replay = (
                        sinks.detach().requires_grad_(ctx.needs_input_grad[2])
                        if ctx.has_sinks
                        else None
                    )
                    ori_source_replay = (
                        ori_source.detach().requires_grad_(ctx.needs_input_grad[3])
                        if ctx.has_ori_source
                        else None
                    )
                    cmp_source_replay = (
                        cmp_source.detach().requires_grad_(ctx.needs_input_grad[5])
                        if ctx.has_cmp_source
                        else None
                    )
                    replay_inputs = []
                    replay_input_names = []
                    for name, tensor, needed in (
                        ("q", q_replay, ctx.needs_input_grad[1]),
                        ("sinks", sinks_replay, ctx.needs_input_grad[2]),
                        (
                            "ori_source",
                            ori_source_replay,
                            ctx.needs_input_grad[3],
                        ),
                        (
                            "cmp_source",
                            cmp_source_replay,
                            ctx.needs_input_grad[5],
                        ),
                    ):
                        if tensor is not None and needed:
                            replay_inputs.append(tensor)
                            replay_input_names.append(name)
                    replay_function = (
                        _shared_kv_attention_reference_vectorized
                        if vectorized_replay
                        else _shared_kv_attention_reference
                    )
                    replay_output = replay_function(
                        q_replay,
                        sinks_replay,
                        ctx.op_kwargs,
                        ori_source_replay,
                        ori_slots if ctx.has_ori_source else None,
                        cmp_source_replay,
                        cmp_slots if ctx.has_cmp_source else None,
                    )
                    replay_grads = torch.autograd.grad(
                        replay_output,
                        replay_inputs,
                        grad_output,
                        allow_unused=True,
                    )
                    grad_by_name = dict(zip(replay_input_names, replay_grads))
                    grad_q = grad_by_name.get("q")
                    grad_sinks = grad_by_name.get("sinks")
                    grad_ori_source = grad_by_name.get("ori_source")
                    grad_cmp_source = grad_by_name.get("cmp_source")
                    _increment_training_diagnostic(
                        "dsa_full_graph_vectorized_sparse_attention_backward_calls"
                        if vectorized_replay
                        else "dsa_serial_sparse_attention_backward_calls"
                    )
            return (
                None,
                grad_q,
                grad_sinks,
                grad_ori_source,
                None,
                grad_cmp_source,
                None,
                None,
                None,
            )

    def _autograd_sparse_attention(q, *args, **kwargs):
        if not torch.is_grad_enabled() or not q.is_floating_point():
            return original_sparse_attention(q, *args, **kwargs)
        if args:
            raise RuntimeError(
                "Sparse-attention training fallback expects keyword-only optional arguments"
            )
        sinks = kwargs.get("sinks")
        ori_cache = kwargs.get("ori_kv")
        cmp_cache = kwargs.get("cmp_kv")
        ori_update = _take_cache_update(ori_cache)
        cmp_update = (
            ori_update if cmp_cache is ori_cache else _take_cache_update(cmp_cache)
        )
        ori_source, ori_slots = ori_update if ori_update is not None else (None, None)
        cmp_source, cmp_slots = cmp_update if cmp_update is not None else (None, None)
        reuse_host_metadata = _env_flag(
            "MTP_DS4_REUSE_DSA_BACKWARD_HOST_METADATA",
            False,
        )
        host_metadata = (
            _lookup_dsa_backward_host_metadata(kwargs) if reuse_host_metadata else None
        )
        if reuse_host_metadata:
            _increment_training_diagnostic(
                "dsa_backward_host_metadata_hits"
                if host_metadata is not None
                else "dsa_backward_host_metadata_fallbacks"
            )
        with torch.no_grad():
            native_output, softmax_lse = original_sparse_attention(q, **kwargs)
        wrapped_output = _SparseAttentionGrad.apply(
            native_output,
            q,
            sinks,
            ori_source,
            ori_slots,
            cmp_source,
            cmp_slots,
            kwargs,
            host_metadata,
        )
        return wrapped_output, softmax_lse

    _autograd_sparse_attention._speco_deepseek_v4_train_backward_patch = True
    _autograd_scatter_update._speco_deepseek_v4_train_backward_patch = True
    _DSA_TRAINING_SCATTER_WRAPPER = _autograd_scatter_update
    _DSA_TRAINING_SPARSE_ATTENTION_WRAPPER = _autograd_sparse_attention
    ascend_ops.npu_scatter_nd_update_v2 = _autograd_scatter_update
    ascend_ops.npu_sparse_attn_sharedkv = _autograd_sparse_attention
    _increment_training_diagnostic("dsa_training_op_patch_installed")
    return True


def _explicit_rms_norm_vjp(
    norm_input: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_float = norm_input.float()
    grad_float = grad_output.float()
    weight_float = weight.float()
    inverse_rms = torch.rsqrt(
        input_float.square().mean(dim=-1, keepdim=True) + float(epsilon)
    )
    normalized_input = input_float * inverse_rms
    grad_scaled_input = grad_float * weight_float
    projection = (grad_scaled_input * normalized_input).mean(
        dim=-1,
        keepdim=True,
    )
    grad_input = (grad_scaled_input - normalized_input * projection) * inverse_rms
    grad_weight = (
        (grad_float * normalized_input).reshape(-1, int(weight.numel())).sum(dim=0)
    )
    return grad_input, grad_weight.reshape_as(weight)


def _remember_bf16_qnorm_source(
    normalized_output: torch.Tensor,
    norm_input: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_module,
    epsilon: float,
    sum_grad_output_across_tp: bool,
) -> None:
    output_id = id(normalized_output)

    def _discard_stale_source(output_ref, *, expected_id=output_id):
        current = _BF16_QNORM_SOURCES.get(expected_id)
        if current is not None and current[0] is output_ref:
            _BF16_QNORM_SOURCES.pop(expected_id, None)

    output_ref = weakref.ref(normalized_output, _discard_stale_source)
    _BF16_QNORM_SOURCES[output_id] = (
        output_ref,
        norm_input,
        norm_weight,
        norm_bias,
        norm_module,
        float(epsilon),
        bool(sum_grad_output_across_tp),
    )


def _take_bf16_qnorm_source(
    normalized_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, float, bool] | None:
    source = _BF16_QNORM_SOURCES.pop(id(normalized_output), None)
    if source is None:
        return None
    output_ref, *values = source
    if output_ref() is not normalized_output:
        _increment_training_diagnostic("stale_bf16_qnorm_source_drops")
        return None
    return tuple(values)


class _NativeRMSNormGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        native_output,
        x,
        weight,
        bias,
        module,
        epsilon,
        sum_grad_output_across_tp,
    ):
        ctx.module = module
        ctx.epsilon = float(epsilon)
        ctx.has_bias = bool(bias.numel())
        ctx.sum_grad_output_across_tp = bool(sum_grad_output_across_tp)
        ctx.save_for_backward(x, weight, bias)
        return native_output.clone()

    @staticmethod
    @_profile_vllm_ascend_backward("backward_rms_norm")
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        ctx.module._speco_deepseek_v4_replay_grad_calls = (
            getattr(ctx.module, "_speco_deepseek_v4_replay_grad_calls", 0) + 1
        )
        norm_grad_output = (
            _tensor_parallel_gradient_sum(grad_output)
            if ctx.sum_grad_output_across_tp
            else grad_output
        )
        grad_x_float, grad_weight_float = _explicit_rms_norm_vjp(
            x,
            weight,
            norm_grad_output,
            ctx.epsilon,
        )
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
            grad_bias = grad_bias.reshape_as(bias).to(bias.dtype)
        return (
            None,
            grad_x_float.to(x.dtype) if ctx.needs_input_grad[1] else None,
            (grad_weight_float.to(weight.dtype) if ctx.needs_input_grad[2] else None),
            grad_bias,
            None,
            None,
            None,
        )


class _NativeResidualRMSNormGrad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        native_output,
        native_residual,
        x,
        residual,
        weight,
        bias,
        module,
        epsilon,
        sum_grad_output_across_tp,
    ):
        ctx.module = module
        ctx.epsilon = float(epsilon)
        ctx.has_bias = bool(bias.numel())
        ctx.sum_grad_output_across_tp = bool(sum_grad_output_across_tp)
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype
        ctx.save_for_backward(native_residual, weight, bias)
        return native_output.clone(), native_residual.clone()

    @staticmethod
    @_profile_vllm_ascend_backward("backward_rms_norm")
    def backward(ctx, grad_output, grad_residual_output):
        norm_input, weight, bias = ctx.saved_tensors
        ctx.module._speco_deepseek_v4_replay_grad_calls = (
            getattr(ctx.module, "_speco_deepseek_v4_replay_grad_calls", 0) + 1
        )
        norm_grad_output = (
            _tensor_parallel_gradient_sum(grad_output)
            if ctx.sum_grad_output_across_tp
            else grad_output
        )
        grad_norm_float, grad_weight_float = _explicit_rms_norm_vjp(
            norm_input,
            weight,
            norm_grad_output,
            ctx.epsilon,
        )
        if grad_residual_output is not None:
            grad_norm_float.add_(grad_residual_output.float())
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[5]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
            grad_bias = grad_bias.reshape_as(bias).to(bias.dtype)
        return (
            None,
            None,
            (grad_norm_float.to(ctx.x_dtype) if ctx.needs_input_grad[2] else None),
            (
                grad_norm_float.to(ctx.residual_dtype)
                if ctx.needs_input_grad[3]
                else None
            ),
            (grad_weight_float.to(weight.dtype) if ctx.needs_input_grad[4] else None),
            grad_bias,
            None,
            None,
            None,
        )


def _patch_vllm_ascend_rms_norm_training_backward() -> bool:
    try:
        from vllm_ascend.ops.layernorm import AscendRMSNorm
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(AscendRMSNorm, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    original_forward = AscendRMSNorm.forward_oot

    def _autograd_forward(self, x, residual=None):
        if not torch.is_grad_enabled():
            return original_forward(self, x, residual)
        self._speco_deepseek_v4_training_forward_calls = (
            getattr(self, "_speco_deepseek_v4_training_forward_calls", 0) + 1
        )
        if self.variance_size_override is not None:
            raise RuntimeError(
                "The RMSNorm training fallback does not support variance_size_override"
            )
        weight_shape = tuple(self.weight.shape)
        input_tail_shape = tuple(x.shape[-self.weight.ndim :])
        if input_tail_shape != weight_shape:
            raise RuntimeError(
                "RMSNorm training input shape mismatch before the native NPU "
                f"kernel: input={tuple(x.shape)}, weight={weight_shape}. "
                "For TP-sharded MTP attention this usually means a CVLinearWrapper "
                "bypassed the projection all-gather."
            )

        with torch.no_grad():
            native_result = original_forward(self, x, residual)
        sum_grad_output_across_tp = getattr(
            self,
            "_speco_deepseek_v4_sum_grad_output_across_tp",
            False,
        )
        if sum_grad_output_across_tp:
            _increment_training_diagnostic("tp_replicated_norm_grad_sum_calls")
        bias = self.bias if getattr(self, "bias_loaded", False) else x.new_empty((0,))

        if residual is None:
            wrapped_output = _NativeRMSNormGrad.apply(
                native_result,
                x,
                self.weight,
                bias,
                self,
                self.variance_epsilon,
                sum_grad_output_across_tp,
            )
            if getattr(self, "_speco_deepseek_v4_fuse_with_wq_b", False) and _env_flag(
                "MTP_DS4_FUSED_QNORM_UNQUANTIZED_BACKWARD",
                True,
            ):
                _remember_bf16_qnorm_source(
                    wrapped_output,
                    x,
                    self.weight,
                    bias,
                    self,
                    self.variance_epsilon,
                    sum_grad_output_across_tp,
                )
                _increment_training_diagnostic("bf16_qnorm_linear_source_registrations")
            return wrapped_output

        native_output, native_residual = native_result
        return _NativeResidualRMSNormGrad.apply(
            native_output,
            native_residual,
            x,
            residual,
            self.weight,
            bias,
            self,
            self.variance_epsilon,
            sum_grad_output_across_tp,
        )

    AscendRMSNorm.forward_oot = _autograd_forward
    AscendRMSNorm._speco_deepseek_v4_train_backward_patch = True
    return True


def _patch_vllm_ascend_tp1_gather_training_backward() -> bool:
    try:
        vllm_ops = torch.ops.vllm
        original_gather = vllm_ops.maybe_all_gather_and_maybe_unpad
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(original_gather, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    def _autograd_gather(x, *args, **kwargs):
        if not torch.is_grad_enabled() or not x.is_floating_point():
            return original_gather(x, *args, **kwargs)

        with torch.no_grad():
            native_output = original_gather(x, *args, **kwargs)
        if native_output.shape != x.shape:
            raise RuntimeError(
                "Grad-enabled maybe_all_gather entered a token-sharded communication "
                "path. DS4 TP training requires enable_flashcomm1=false: "
                f"input={tuple(x.shape)}, output={tuple(native_output.shape)}"
            )
        _increment_training_diagnostic("tp_identity_gather_training_calls")
        return native_output.detach() + x - x.detach()

    _autograd_gather._speco_deepseek_v4_train_backward_patch = True
    vllm_ops.maybe_all_gather_and_maybe_unpad = _autograd_gather
    return True


def _patch_vllm_ascend_tp_all_reduce_training_backward() -> bool:
    try:
        vllm_ops = torch.ops.vllm
        original_pad_reduce = vllm_ops.maybe_pad_and_reduce
        original_moe_reduce = vllm_ops.maybe_all_reduce_tensor_model_parallel
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(
        original_pad_reduce,
        "_speco_deepseek_v4_train_backward_patch",
        False,
    ):
        return True

    class _TensorParallelAllReduce(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, native_output):
            return native_output

        @staticmethod
        @_profile_vllm_ascend_backward("backward_tp_collective")
        def backward(ctx, grad_output):
            return grad_output, None

    def _wrap_standard_tp_all_reduce(
        original_op,
        diagnostic_name,
        deterministic_env=None,
    ):
        def _autograd_reduce(x, *args, **kwargs):
            if not torch.is_grad_enabled() or not x.is_floating_point():
                return original_op(x, *args, **kwargs)
            deterministic_enabled = deterministic_env is not None and os.environ.get(
                deterministic_env, "0"
            ).strip().lower() not in {"0", "false", "off", "no"}
            if deterministic_enabled:
                from vllm.distributed import get_tp_group

                output = _DeterministicTPRowReduce.apply(x, get_tp_group())
                _increment_training_diagnostic(diagnostic_name)
                _increment_training_diagnostic("tp_deterministic_moe_all_reduce_calls")
                return output
            with torch.no_grad():
                native_output = original_op(x, *args, **kwargs)
            if native_output.shape != x.shape:
                raise RuntimeError(
                    f"Grad-enabled {diagnostic_name} entered a reduce-scatter "
                    "path. DS4 TP training requires enable_flashcomm1=false: "
                    f"input={tuple(x.shape)}, output={tuple(native_output.shape)}"
                )
            _increment_training_diagnostic(diagnostic_name)
            return _TensorParallelAllReduce.apply(x, native_output.detach())

        _autograd_reduce._speco_deepseek_v4_train_backward_patch = True
        return _autograd_reduce

    vllm_ops.maybe_pad_and_reduce = _wrap_standard_tp_all_reduce(
        original_pad_reduce, "tp_pad_reduce_training_calls"
    )
    vllm_ops.maybe_all_reduce_tensor_model_parallel = _wrap_standard_tp_all_reduce(
        original_moe_reduce,
        "tp_moe_all_reduce_training_calls",
        "MTP_DS4_DETERMINISTIC_TP_MOE_FORWARD",
    )
    return True


def _patch_vllm_ascend_cv_linear_training_backward() -> bool:
    try:
        from vllm_ascend.ops.cv_linear import CVLinearWrapper
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(CVLinearWrapper, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    original_quantize = CVLinearWrapper.quantize
    original_matmul = CVLinearWrapper.matmul

    class _CVQuantLinearInputGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, native_output, x, weight, weight_scale):
            ctx.input_dtype = x.dtype
            ctx.input_size = int(x.shape[-1])
            ctx.save_for_backward(weight, weight_scale)
            return native_output.clone()

        @staticmethod
        @_profile_vllm_ascend_backward("backward_cv_projection")
        def backward(ctx, grad_output):
            weight, weight_scale = ctx.saved_tensors
            input_size = ctx.input_size
            scale = weight_scale.detach().float().reshape(-1)
            if weight.shape[0] == input_size:
                input_major = True
                output_size = int(weight.shape[1])
            elif weight.shape[1] == input_size:
                input_major = False
                output_size = int(weight.shape[0])
            else:
                raise RuntimeError(
                    f"Unexpected CV quantized weight shape {tuple(weight.shape)} "
                    f"for input size {input_size}"
                )
            if scale.numel() != output_size:
                raise RuntimeError(
                    f"CV weight scale size {scale.numel()} does not match output size {output_size}"
                )

            grad_flat = grad_output.reshape(-1, output_size).float()
            grad_input = torch.zeros(
                grad_flat.shape[0],
                input_size,
                dtype=torch.float32,
                device=grad_output.device,
            )
            chunk_size = 4096
            for start in range(0, output_size, chunk_size):
                end = min(start + chunk_size, output_size)
                grad_chunk = grad_flat[:, start:end] * scale[start:end]
                if input_major:
                    weight_chunk = weight[:, start:end].detach().float()
                    grad_input.add_(grad_chunk.matmul(weight_chunk.transpose(0, 1)))
                else:
                    weight_chunk = weight[start:end, :].detach().float()
                    grad_input.add_(grad_chunk.matmul(weight_chunk))
            return (
                None,
                grad_input.reshape(*grad_output.shape[:-1], input_size).to(
                    ctx.input_dtype
                ),
                None,
                None,
            )

    def _autograd_quantize(self, x):
        if not torch.is_grad_enabled() or not self._is_w8a8_dynamic:
            return original_quantize(self, x)
        _increment_training_diagnostic("cv_quantize_calls")
        self._speco_deepseek_v4_float_input = x
        with torch.no_grad():
            return original_quantize(self, x)

    def _autograd_matmul(self, quantized_x, pertoken_scale=None, bias=None):
        float_input = getattr(self, "_speco_deepseek_v4_float_input", None)
        if hasattr(self, "_speco_deepseek_v4_float_input"):
            del self._speco_deepseek_v4_float_input
        if not torch.is_grad_enabled() or not self._is_w8a8_dynamic:
            return original_matmul(self, quantized_x, pertoken_scale, bias)
        _increment_training_diagnostic("cv_matmul_calls")
        # The custom autograd node owns this input from here on.  Leaving the
        # same tensor on every CVLinearWrapper retained the completed step's
        # activation graph until that module was reached in the next forward.
        with torch.no_grad():
            native_output = original_matmul(self, quantized_x, pertoken_scale, bias)
        if not isinstance(float_input, torch.Tensor):
            raise RuntimeError(  # noqa: TRY004 -- preserve numerical validation exception contract
                "CVLinearWrapper.matmul has no matching float quantize input"
            )
        weight_scale = getattr(self.linear, "weight_scale_fp32", None)
        if weight_scale is None:
            weight_scale = self.linear.weight_scale
        return _CVQuantLinearInputGrad.apply(
            native_output,
            float_input,
            self.linear.weight,
            weight_scale,
        )

    CVLinearWrapper.quantize = _autograd_quantize
    CVLinearWrapper.matmul = _autograd_matmul
    CVLinearWrapper._speco_deepseek_v4_train_backward_patch = True
    return True


def _patch_vllm_ascend_fused_qnorm_quant_training_backward() -> bool:
    try:
        import torch_npu

        ascend_ops = torch.ops._C_ascend
        original_norm_quant = ascend_ops.npu_rms_norm_dynamic_quant
        original_quant_matmul = torch_npu.npu_quant_matmul
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(original_norm_quant, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    quantized_sources: dict[
        int,
        tuple[Any, torch.Tensor, torch.Tensor, float],
    ] = {}

    def _remember_quantized_source(
        quantized: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        quantized_id = id(quantized)

        def _discard_stale_source(
            quantized_ref,
            *,
            expected_id=quantized_id,
        ):
            current = quantized_sources.get(expected_id)
            if current is not None and current[0] is quantized_ref:
                quantized_sources.pop(expected_id, None)

        quantized_ref = weakref.ref(
            quantized,
            _discard_stale_source,
        )
        quantized_sources[quantized_id] = (
            quantized_ref,
            x,
            weight,
            epsilon,
        )

    def _take_quantized_source(
        quantized: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        source = quantized_sources.pop(id(quantized), None)
        if source is None:
            return None
        quantized_ref, x, weight, epsilon = source
        if quantized_ref() is not quantized:
            _increment_training_diagnostic("stale_quantized_source_drops")
            return None
        return x, weight, epsilon

    class _FusedQNormQuantLinearGrad(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            native_output,
            norm_input,
            norm_weight,
            linear_weight,
            linear_scale,
            epsilon,
        ):
            ctx.epsilon = float(epsilon)
            ctx.input_dtype = norm_input.dtype
            ctx.norm_weight_dtype = norm_weight.dtype
            ctx.save_for_backward(
                norm_input,
                norm_weight,
                linear_weight,
                linear_scale,
            )
            return native_output.clone()

        @staticmethod
        @_profile_vllm_ascend_backward("backward_qnorm_projection")
        def backward(ctx, grad_output):
            _increment_training_diagnostic("fused_qnorm_quant_backward_calls")
            norm_input, norm_weight, linear_weight, linear_scale = ctx.saved_tensors
            input_size = int(norm_input.shape[-1])
            scale = linear_scale.detach().float().reshape(-1)
            if linear_weight.shape[0] == input_size:
                input_major = True
                output_size = int(linear_weight.shape[1])
            elif linear_weight.shape[1] == input_size:
                input_major = False
                output_size = int(linear_weight.shape[0])
            else:
                raise RuntimeError(
                    f"Unexpected fused q_norm q_b weight shape {tuple(linear_weight.shape)} "
                    f"for input size {input_size}"
                )
            if scale.numel() != output_size:
                raise RuntimeError(
                    f"Fused q_norm q_b scale size {scale.numel()} "
                    f"does not match output size {output_size}"
                )

            grad_flat = grad_output.reshape(-1, output_size).float()
            grad_normed = torch.zeros(
                grad_flat.shape[0],
                input_size,
                dtype=torch.float32,
                device=grad_output.device,
            )
            chunk_size = 4096
            for start in range(0, output_size, chunk_size):
                end = min(start + chunk_size, output_size)
                grad_chunk = grad_flat[:, start:end] * scale[start:end]
                if input_major:
                    weight_chunk = linear_weight[:, start:end].detach().float()
                    grad_normed.add_(grad_chunk.matmul(weight_chunk.transpose(0, 1)))
                else:
                    weight_chunk = linear_weight[start:end, :].detach().float()
                    grad_normed.add_(grad_chunk.matmul(weight_chunk))

            input_float = norm_input.reshape(-1, input_size).float()
            norm_weight_float = norm_weight.float().reshape(1, input_size)
            inverse_rms = torch.rsqrt(
                input_float.square().mean(dim=-1, keepdim=True) + ctx.epsilon
            )
            grad_scaled_input = grad_normed * norm_weight_float
            correction = (grad_scaled_input * input_float).mean(
                dim=-1, keepdim=True
            ) * inverse_rms.pow(3)
            grad_input = inverse_rms * grad_scaled_input - input_float * correction
            grad_norm_weight = (grad_normed * input_float * inverse_rms).sum(dim=0)

            return (
                None,
                grad_input.reshape_as(norm_input).to(ctx.input_dtype),
                grad_norm_weight.reshape_as(norm_weight).to(ctx.norm_weight_dtype),
                None,
                None,
                None,
            )

    def _autograd_norm_quant(x, weight, *args, **kwargs):
        if not torch.is_grad_enabled() or not x.is_floating_point():
            return original_norm_quant(x, weight, *args, **kwargs)
        _increment_training_diagnostic("fused_qnorm_quant_calls")
        with torch.no_grad():
            quantized, scale = original_norm_quant(x, weight, *args, **kwargs)
        epsilon = kwargs.get("epsilon", args[0] if args else 1e-6)
        _remember_quantized_source(
            quantized,
            x,
            weight,
            float(epsilon),
        )
        return quantized, scale

    def _autograd_quant_matmul(
        quantized_x,
        linear_weight,
        linear_scale,
        *args,
        **kwargs,
    ):
        source = _take_quantized_source(quantized_x)
        if not torch.is_grad_enabled() or source is None:
            if torch.is_grad_enabled():
                _increment_training_diagnostic("direct_quant_matmul_source_misses")
            return original_quant_matmul(
                quantized_x,
                linear_weight,
                linear_scale,
                *args,
                **kwargs,
            )
        _increment_training_diagnostic("direct_quant_matmul_source_hits")
        with torch.no_grad():
            native_output = original_quant_matmul(
                quantized_x,
                linear_weight,
                linear_scale,
                *args,
                **kwargs,
            )
        norm_input, norm_weight, epsilon = source
        return _FusedQNormQuantLinearGrad.apply(
            native_output,
            norm_input,
            norm_weight,
            linear_weight,
            linear_scale,
            epsilon,
        )

    _autograd_norm_quant._speco_deepseek_v4_train_backward_patch = True
    _autograd_quant_matmul._speco_deepseek_v4_train_backward_patch = True
    ascend_ops.npu_rms_norm_dynamic_quant = _autograd_norm_quant
    torch_npu.npu_quant_matmul = _autograd_quant_matmul
    return True


def _patch_vllm_ascend_triton_q_rms_training_backward() -> bool:
    try:
        from vllm_ascend.attention import dsa_v1
        from vllm_ascend.ops.triton import rms_norm as triton_rms_norm
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    original_q_rms = triton_rms_norm.triton_q_rms
    if getattr(original_q_rms, "_speco_deepseek_v4_train_backward_patch", False):
        return True

    def _autograd_q_rms(q, variance_epsilon):
        if not torch.is_grad_enabled() or not q.is_floating_point():
            return original_q_rms(q, variance_epsilon)
        _increment_training_diagnostic("triton_q_rms_training_calls")
        with torch.no_grad():
            native_output = original_q_rms(q, variance_epsilon)
        q_float = q.float()
        replay_output = q_float * torch.rsqrt(
            q_float.square().mean(dim=-1, keepdim=True) + float(variance_epsilon)
        )
        replay_output = replay_output.to(q.dtype)
        return native_output.detach() + replay_output - replay_output.detach()

    _autograd_q_rms._speco_deepseek_v4_train_backward_patch = True
    triton_rms_norm.triton_q_rms = _autograd_q_rms
    dsa_v1.triton_q_rms = _autograd_q_rms
    try:
        from vllm_ascend.attention.context_parallel import dsa_cp

        dsa_cp.triton_q_rms = _autograd_q_rms
    except Exception:  # noqa: BLE001,S110 -- optional accelerator capability fallback
        pass
    return True


def _patch_vllm_ascend_muls_add_training_fallback() -> bool:
    try:
        from verl_speco.backends.vllm_ascend_compat import (
            MODEL_MODULES,
            module_with_symbol,
        )

        deepseek_v4, _ = module_with_symbol(
            MODEL_MODULES["model"], ("muls_add_triton",)
        )
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(deepseek_v4, "_speco_deepseek_v4_muls_add_training_patch", False):
        return True

    original_muls_add = deepseek_v4.muls_add_triton

    def _autograd_muls_add(x: torch.Tensor, y: torch.Tensor, scale: float):
        if not torch.is_grad_enabled():
            return original_muls_add(x, y, scale)
        if x.shape != y.shape:
            raise ValueError(
                f"muls_add inputs must have identical shapes, got {x.shape} and {y.shape}"
            )
        return x * scale + y

    deepseek_v4.muls_add_triton = _autograd_muls_add
    deepseek_v4._speco_deepseek_v4_muls_add_training_patch = True
    return True


def _hc_pre_reference(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = x.dtype
    x_float = x.float()
    x_flat = x_float.flatten(start_dim=-2)
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    logits = F.linear(x_flat, hc_fn.float()) * rsqrt
    pre_logits = logits[..., :hc_mult]
    post_logits = logits[..., hc_mult : 2 * hc_mult]
    comb_logits = logits[..., 2 * hc_mult :]
    pre = (
        torch.sigmoid(hc_scale[0].float() * pre_logits + hc_base[:hc_mult].float())
        + hc_eps
    )
    post = 2 * torch.sigmoid(
        hc_scale[1].float() * post_logits + hc_base[hc_mult : 2 * hc_mult].float()
    )
    comb = (hc_scale[2].float() * comb_logits + hc_base[2 * hc_mult :].float()).reshape(
        *x.shape[:-2], hc_mult, hc_mult
    )
    comb = torch.softmax(comb, dim=-1) + hc_eps
    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = torch.sum(pre.unsqueeze(-1) * x_float, dim=-2)
    return collapsed.to(dtype), post, comb


class _ChunkedHCPre(torch.autograd.Function):
    """Recompute HC pre gradients without retaining full FP32 activations."""

    @staticmethod
    def forward(
        ctx,
        x,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult,
        norm_eps,
        hc_eps,
        sinkhorn_iters,
        chunk_size,
        fused_forward,
    ):
        flat_x = x.reshape(-1, x.shape[-2], x.shape[-1])
        ctx.save_for_backward(x, hc_fn, hc_scale, hc_base)
        ctx.hc_mult = hc_mult
        ctx.norm_eps = norm_eps
        ctx.hc_eps = hc_eps
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.chunk_size = chunk_size
        leading_shape = x.shape[:-2]
        if fused_forward and chunk_size >= flat_x.shape[0]:
            collapsed, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
                flat_x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult,
                sinkhorn_iters,
                norm_eps,
                hc_eps,
            )
            _increment_training_diagnostic("hc_pre_fused_forward_chunks")
            _increment_training_diagnostic("hc_pre_forward_token_chunks")
            _increment_training_diagnostic("hc_pre_chunked_forward_calls")
            return (
                collapsed.reshape(*leading_shape, x.shape[-1]),
                post.reshape(*leading_shape, hc_mult),
                comb.reshape(*leading_shape, hc_mult, hc_mult),
            )
        collapsed = x.new_empty((flat_x.shape[0], flat_x.shape[-1]))
        post = torch.empty(
            (flat_x.shape[0], hc_mult),
            dtype=torch.float32,
            device=x.device,
        )
        comb = torch.empty(
            (flat_x.shape[0], hc_mult, hc_mult),
            dtype=torch.float32,
            device=x.device,
        )
        for start in range(0, flat_x.shape[0], chunk_size):
            end = min(start + chunk_size, flat_x.shape[0])
            if fused_forward:
                chunk_collapsed, chunk_post, chunk_comb = (
                    torch.ops._C_ascend.npu_hc_pre_v2(
                        flat_x[start:end],
                        hc_fn,
                        hc_scale,
                        hc_base,
                        hc_mult,
                        sinkhorn_iters,
                        norm_eps,
                        hc_eps,
                    )
                )
                _increment_training_diagnostic("hc_pre_fused_forward_chunks")
            else:
                chunk_collapsed, chunk_post, chunk_comb = _hc_pre_reference(
                    flat_x[start:end],
                    hc_fn,
                    hc_scale,
                    hc_base,
                    hc_mult,
                    norm_eps,
                    hc_eps,
                    sinkhorn_iters,
                )
            collapsed[start:end].copy_(chunk_collapsed)
            post[start:end].copy_(chunk_post)
            comb[start:end].copy_(chunk_comb)
            _increment_training_diagnostic("hc_pre_forward_token_chunks")

        _increment_training_diagnostic("hc_pre_chunked_forward_calls")
        return (
            collapsed.reshape(*leading_shape, x.shape[-1]),
            post.reshape(*leading_shape, hc_mult),
            comb.reshape(*leading_shape, hc_mult, hc_mult),
        )

    @staticmethod
    @_profile_vllm_ascend_backward("backward_hc_pre")
    def backward(ctx, grad_collapsed, grad_post, grad_comb):
        x, hc_fn, hc_scale, hc_base = ctx.saved_tensors
        needs = ctx.needs_input_grad
        flat_x = x.reshape(-1, x.shape[-2], x.shape[-1])
        flat_grad_collapsed = grad_collapsed.reshape(
            -1,
            grad_collapsed.shape[-1],
        )
        flat_grad_post = grad_post.reshape(-1, grad_post.shape[-1])
        flat_grad_comb = grad_comb.reshape(
            -1,
            grad_comb.shape[-2],
            grad_comb.shape[-1],
        )
        if ctx.chunk_size >= flat_x.shape[0]:
            gradients = hc_pre_explicit_backward_chunk(
                flat_x,
                hc_fn,
                hc_scale,
                hc_base,
                flat_grad_collapsed,
                flat_grad_post,
                flat_grad_comb,
                ctx.hc_mult,
                ctx.norm_eps,
                ctx.hc_eps,
                ctx.sinkhorn_iters,
                need_x=needs[0],
                need_hc_fn=needs[1],
                need_hc_scale=needs[2],
                need_hc_base=needs[3],
            )
            _increment_training_diagnostic("hc_pre_backward_token_chunks")
            _increment_training_diagnostic("hc_pre_explicit_backward_chunks")
            _increment_training_diagnostic("hc_pre_chunked_backward_calls")
            _increment_training_diagnostic("hc_pre_explicit_backward_calls")
            grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base = gradients
            return (
                grad_x.reshape_as(x) if grad_x is not None else None,
                grad_hc_fn,
                grad_hc_scale,
                grad_hc_base,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        grad_x = torch.empty_like(flat_x) if needs[0] else None
        grad_hc_fn = torch.zeros_like(hc_fn) if needs[1] else None
        grad_hc_scale = torch.zeros_like(hc_scale) if needs[2] else None
        grad_hc_base = torch.zeros_like(hc_base) if needs[3] else None

        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, flat_x.shape[0])
            (
                chunk_grad_x,
                chunk_grad_hc_fn,
                chunk_grad_hc_scale,
                chunk_grad_hc_base,
            ) = hc_pre_explicit_backward_chunk(
                flat_x[start:end],
                hc_fn,
                hc_scale,
                hc_base,
                flat_grad_collapsed[start:end],
                flat_grad_post[start:end],
                flat_grad_comb[start:end],
                ctx.hc_mult,
                ctx.norm_eps,
                ctx.hc_eps,
                ctx.sinkhorn_iters,
                need_x=needs[0],
                need_hc_fn=needs[1],
                need_hc_scale=needs[2],
                need_hc_base=needs[3],
            )
            if grad_x is not None:
                grad_x[start:end].copy_(chunk_grad_x)
            if grad_hc_fn is not None:
                grad_hc_fn.add_(chunk_grad_hc_fn)
            if grad_hc_scale is not None:
                grad_hc_scale.add_(chunk_grad_hc_scale)
            if grad_hc_base is not None:
                grad_hc_base.add_(chunk_grad_hc_base)
            _increment_training_diagnostic("hc_pre_backward_token_chunks")
            _increment_training_diagnostic("hc_pre_explicit_backward_chunks")

        _increment_training_diagnostic("hc_pre_chunked_backward_calls")
        _increment_training_diagnostic("hc_pre_explicit_backward_calls")
        return (
            grad_x.reshape_as(x) if grad_x is not None else None,
            grad_hc_fn,
            grad_hc_scale,
            grad_hc_base,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ChunkedHCPost(torch.autograd.Function):
    """Recompute HC post gradients while bounding FP32 token intermediates."""

    @staticmethod
    def forward(ctx, x, residual, post, comb, chunk_size, fused_forward):
        if x.shape[:-1] != residual.shape[:-2]:
            raise ValueError(
                "HC post x/residual batch shapes differ: "
                f"x={tuple(x.shape)}, residual={tuple(residual.shape)}"
            )
        flat_x = x.reshape(-1, x.shape[-1])
        flat_residual = residual.reshape(
            -1,
            residual.shape[-2],
            residual.shape[-1],
        )
        flat_post = post.reshape(-1, post.shape[-1])
        flat_comb = comb.reshape(-1, comb.shape[-2], comb.shape[-1])
        ctx.save_for_backward(x, residual, post, comb)
        ctx.chunk_size = chunk_size
        if fused_forward and chunk_size >= flat_x.shape[0]:
            output = torch.ops._C_ascend.npu_hc_post(
                flat_x.unsqueeze(0),
                flat_residual.unsqueeze(0),
                flat_post.unsqueeze(0),
                flat_comb.unsqueeze(0),
            ).squeeze(0)
            _increment_training_diagnostic("hc_post_fused_forward_chunks")
            _increment_training_diagnostic("hc_post_forward_token_chunks")
            _increment_training_diagnostic("hc_post_chunked_forward_calls")
            return output.reshape_as(residual)
        output = torch.empty_like(flat_residual)
        for start in range(0, flat_x.shape[0], chunk_size):
            end = min(start + chunk_size, flat_x.shape[0])
            if fused_forward:
                chunk_output = torch.ops._C_ascend.npu_hc_post(
                    flat_x[start:end].unsqueeze(0),
                    flat_residual[start:end].unsqueeze(0),
                    flat_post[start:end].unsqueeze(0),
                    flat_comb[start:end].unsqueeze(0),
                ).squeeze(0)
                _increment_training_diagnostic("hc_post_fused_forward_chunks")
            else:
                chunk_output = flat_post[start:end].float().unsqueeze(-1) * (
                    flat_x[start:end].float().unsqueeze(-2)
                )
                chunk_output.add_(
                    torch.matmul(
                        flat_comb[start:end].float().transpose(-1, -2),
                        flat_residual[start:end].float(),
                    )
                )
            output[start:end].copy_(chunk_output.to(residual.dtype))
            _increment_training_diagnostic("hc_post_forward_token_chunks")

        _increment_training_diagnostic("hc_post_chunked_forward_calls")
        return output.reshape_as(residual)

    @staticmethod
    @_profile_vllm_ascend_backward("backward_hc_post")
    def backward(ctx, grad_output):
        x, residual, post, comb = ctx.saved_tensors
        needs = ctx.needs_input_grad
        flat_x = x.reshape(-1, x.shape[-1])
        flat_residual = residual.reshape(
            -1,
            residual.shape[-2],
            residual.shape[-1],
        )
        flat_post = post.reshape(-1, post.shape[-1])
        flat_comb = comb.reshape(-1, comb.shape[-2], comb.shape[-1])
        flat_grad_output = grad_output.reshape_as(flat_residual)
        if ctx.chunk_size >= flat_x.shape[0]:
            gradients = hc_post_explicit_backward_chunk(
                flat_x,
                flat_residual,
                flat_post,
                flat_comb,
                flat_grad_output,
                need_x=needs[0],
                need_residual=needs[1],
                need_post=needs[2],
                need_comb=needs[3],
            )
            _increment_training_diagnostic("hc_post_backward_token_chunks")
            _increment_training_diagnostic("hc_post_chunked_backward_calls")
            grad_x, grad_residual, grad_post, grad_comb = gradients
            return (
                grad_x.reshape_as(x) if grad_x is not None else None,
                (
                    grad_residual.reshape_as(residual)
                    if grad_residual is not None
                    else None
                ),
                grad_post.reshape_as(post) if grad_post is not None else None,
                grad_comb.reshape_as(comb) if grad_comb is not None else None,
                None,
                None,
            )
        grad_x = torch.empty_like(flat_x) if needs[0] else None
        grad_residual = torch.empty_like(flat_residual) if needs[1] else None
        grad_post = torch.empty_like(flat_post) if needs[2] else None
        grad_comb = torch.empty_like(flat_comb) if needs[3] else None

        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, flat_x.shape[0])
            (
                chunk_grad_x,
                chunk_grad_residual,
                chunk_grad_post,
                chunk_grad_comb,
            ) = hc_post_explicit_backward_chunk(
                flat_x[start:end],
                flat_residual[start:end],
                flat_post[start:end],
                flat_comb[start:end],
                flat_grad_output[start:end],
                need_x=needs[0],
                need_residual=needs[1],
                need_post=needs[2],
                need_comb=needs[3],
            )
            if grad_x is not None:
                grad_x[start:end].copy_(chunk_grad_x)
            if grad_residual is not None:
                grad_residual[start:end].copy_(chunk_grad_residual)
            if grad_post is not None:
                grad_post[start:end].copy_(chunk_grad_post)
            if grad_comb is not None:
                grad_comb[start:end].copy_(chunk_grad_comb)
            _increment_training_diagnostic("hc_post_backward_token_chunks")

        _increment_training_diagnostic("hc_post_chunked_backward_calls")
        return (
            grad_x.reshape_as(x) if grad_x is not None else None,
            (grad_residual.reshape_as(residual) if grad_residual is not None else None),
            grad_post.reshape_as(post) if grad_post is not None else None,
            grad_comb.reshape_as(comb) if grad_comb is not None else None,
            None,
            None,
        )


def _patch_vllm_ascend_hc_training_fallback() -> bool:
    try:
        from verl_speco.backends.vllm_ascend_compat import decoder_layer_type

        DeepseekV2DecoderLayer = decoder_layer_type()
    except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
        return False
    if getattr(DeepseekV2DecoderLayer, "_speco_deepseek_v4_hc_training_patch", False):
        return True

    original_hc_pre = DeepseekV2DecoderLayer.hc_pre
    original_hc_post = DeepseekV2DecoderLayer.hc_post
    native_hc_available = (
        hasattr(torch.ops, "_C_ascend")
        and hasattr(torch.ops._C_ascend, "npu_hc_pre")
        and hasattr(torch.ops._C_ascend, "npu_hc_post")
    )
    fused_training_hc_available = native_hc_available and hasattr(
        torch.ops._C_ascend, "npu_hc_pre_v2"
    )

    def use_fused_training_hc(x: torch.Tensor, hc_mult: int) -> bool:
        requested = _env_flag("MTP_DS4_FUSED_HC_FORWARD", False)
        supported = (
            fused_training_hc_available
            and x.device.type == "npu"
            and x.dtype == torch.bfloat16
            and int(hc_mult) == 4
            and int(x.shape[-2]) == 4
            and int(x.shape[-1]) in {4096, 7168}
        )
        if requested and not supported:
            _increment_training_diagnostic("hc_fused_forward_fallback_calls")
        return requested and supported

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        chunk_size = _env_nonnegative_int(
            "MTP_DS4_HC_TOKEN_CHUNK_SIZE",
            0,
        )
        if chunk_size == 0:
            chunk_size = max(1, int(x.numel() // (self.hc_mult * x.shape[-1])))
        fused_forward = use_fused_training_hc(x, self.hc_mult)
        if not torch.is_grad_enabled() and native_hc_available and not fused_forward:
            return original_hc_pre(self, x, hc_fn, hc_scale, hc_base)
        return _ChunkedHCPre.apply(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
            chunk_size,
            fused_forward,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        chunk_size = _env_nonnegative_int(
            "MTP_DS4_HC_TOKEN_CHUNK_SIZE",
            0,
        )
        if chunk_size == 0:
            chunk_size = max(1, int(x.numel() // x.shape[-1]))
        fused_forward = use_fused_training_hc(residual, residual.shape[-2])
        if not torch.is_grad_enabled() and native_hc_available and not fused_forward:
            return original_hc_post(self, x, residual, post, comb)
        return _ChunkedHCPost.apply(
            x,
            residual,
            post,
            comb,
            chunk_size,
            fused_forward,
        )

    DeepseekV2DecoderLayer.hc_pre = hc_pre
    DeepseekV2DecoderLayer.hc_post = hc_post
    DeepseekV2DecoderLayer._speco_deepseek_v4_hc_training_patch = True
    return True
