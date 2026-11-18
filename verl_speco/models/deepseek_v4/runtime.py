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

import copy
import math

# Adapted from training_common/modeling/ds4_runtime.py.
import os
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

from verl_speco.ops.deepseek_v4_cache import (
    cache_shape_has_capacity,
    required_training_cache_blocks,
)
from verl_speco.ops.deepseek_v4_runtime import (
    _build_dsa_common_structure_tensors,
    _build_vllm_ascend_config,
    _configure_vllm_moe_training_dp_replication,
    _ensure_vllm_ascend_noop_weight_prefetch,
    _ensure_vllm_ascend_parallel_initialized,
    _env_flag,
    _increment_training_diagnostic,
    _install_deterministic_tp_wo_b_forward,
    _install_fused_eh_projection_collective,
    _install_tp_sharded_attention_projections,
    _install_tp_sharded_eh_projections,
    _patch_vllm_ascend_cv_linear_training_backward,
    _patch_vllm_ascend_dsa_batchmatmul_training_backward,
    _patch_vllm_ascend_fused_qnorm_quant_training_backward,
    _patch_vllm_ascend_hc_training_fallback,
    _patch_vllm_ascend_linear_training_backward,
    _patch_vllm_ascend_model_internal_router_training_forward,
    _patch_vllm_ascend_moe_forward_training_fallback,
    _patch_vllm_ascend_moe_unpermute_backward,
    _patch_vllm_ascend_muls_add_training_fallback,
    _patch_vllm_ascend_rms_norm_training_backward,
    _patch_vllm_ascend_rope_precompute_device,
    _patch_vllm_ascend_routed_moe_training_backward,
    _patch_vllm_ascend_sparse_attention_training_backward,
    _patch_vllm_ascend_tp1_gather_training_backward,
    _patch_vllm_ascend_tp_all_reduce_training_backward,
    _patch_vllm_ascend_triton_q_rms_training_backward,
    _patch_vllm_ascend_unquantized_linear_training_backward,
    _patch_vllm_ascend_unquantized_moe_training_forward,
    _prepare_vllm_ascend_imports,
    _register_dsa_backward_host_metadata,
    _set_vllm_ascend_env_defaults,
    _TensorParallelCopy,
    _vllm_ascend_moe_runtime_type,
    _vllm_ascend_moe_weights_module,
)


class DeepseekV4ServingTrainingWrapper(nn.Module):
    def _load_draft_model_class(self) -> type[nn.Module]:
        from verl_speco.backends.vllm_ascend_compat import model_symbol

        return model_symbol("mtp", "DeepSeekV4MTP")

    def _router_module_names(self) -> tuple[str, ...]:
        return (
            "vllm_ascend.models.deepseek_v4.model",
            "vllm_ascend.models.deepseek_v4",
        )

    def _install_variant_runtime_adapters(self) -> bool:
        return False

    def __init__(self, config) -> None:
        super().__init__()
        # SparseAttnSharedkvMetadata allocates temporary workspace outside the
        # tensors reported by max_memory_allocated().  Without a process cap,
        # the PyTorch NPU caching allocator can reserve almost the entire card
        # and leave AICPU with only a few MiB even though tens of GiB are
        # logically free inside the allocator.  Keep explicit device headroom
        # before constructing the vLLM-Ascend model and its caches.
        npu_memory_fraction = float(
            os.environ.get(
                "DS4_NPU_PROCESS_MEMORY_FRACTION",
                # Backward compatibility for existing MTP launch commands.
                os.environ.get("MTP_DS4_NPU_PROCESS_MEMORY_FRACTION", "0.90"),
            )
        )
        if not 0.0 < npu_memory_fraction <= 1.0:
            raise ValueError(
                "DS4_NPU_PROCESS_MEMORY_FRACTION must be in (0, 1], "
                f"got {npu_memory_fraction}"
            )
        if hasattr(torch, "npu") and torch.npu.is_available():
            set_memory_fraction = getattr(
                torch.npu,
                "set_per_process_memory_fraction",
                None,
            )
            if set_memory_fraction is not None:
                set_memory_fraction(npu_memory_fraction)
                _increment_training_diagnostic("npu_process_memory_fraction_applied")
            else:
                _increment_training_diagnostic(
                    "npu_process_memory_fraction_unavailable"
                )
        self._weights_postprocessed = False
        self.dequantize_mtp_for_training = bool(
            getattr(config, "dequantize_mtp_for_training", False)
        )
        self.mtp_checkpoint_format = str(
            getattr(config, "mtp_checkpoint_format", "unknown")
        )
        self.mtp_checkpoint_format_report = getattr(
            config,
            "mtp_checkpoint_format_report",
            None,
        )
        self.compact_training_kv_cache = bool(
            getattr(config, "vllm_compact_training_kv_cache", True)
        )
        self.cache_dsa_metadata_structures = _env_flag(
            "MTP_DS4_CACHE_DSA_METADATA_STRUCTURES",
            bool(
                getattr(
                    config,
                    "cache_dsa_metadata_structures",
                    False,
                )
            ),
        )
        self.reuse_dsa_backward_host_metadata = _env_flag(
            "MTP_DS4_REUSE_DSA_BACKWARD_HOST_METADATA",
            bool(
                getattr(
                    config,
                    "reuse_dsa_backward_host_metadata",
                    False,
                )
            ),
        )
        self.fused_qnorm_unquantized_backward = _env_flag(
            "MTP_DS4_FUSED_QNORM_UNQUANTIZED_BACKWARD",
            bool(
                getattr(
                    config,
                    "fused_qnorm_unquantized_backward",
                    False,
                )
            ),
        )
        self._dsa_common_structure_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._dsa_metadata_builder_cache: dict[tuple[Any, ...], Any] = {}
        self._dsa_qli_metadata_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._compact_training_cache_logged: set[str] = set()
        self._dequantized_training_modules: list[str] = []
        self.using_tp_sharded_eh_projections = False
        self.using_fused_eh_projection_collective = False
        self.using_tp_sharded_attention_projections = False
        self._requested_tensor_parallel_size = int(
            getattr(config, "vllm_tensor_parallel_size", 1)
        )
        self._validated_tp_topology_signature: tuple[int, ...] | None = None
        self._tp1_fallback_warned = False
        self._using_tp1_fallback = self._should_use_tp1_fallback(
            self._requested_tensor_parallel_size
        )
        vllm_runtime_config = self._make_vllm_runtime_config(config)
        _set_vllm_ascend_env_defaults(vllm_runtime_config)
        try:
            _prepare_vllm_ascend_imports()
            from vllm.config import set_current_vllm_config
            from vllm_ascend.models import register_model

            DraftModelClass = self._load_draft_model_class()
        except Exception as exc:
            raise RuntimeError(
                "ds4_layout='vllm_ascend' requires vllm_ascend and the Ascend runtime."
            ) from exc

        _patch_vllm_ascend_rope_precompute_device()
        register_model()
        self.vllm_config = _build_vllm_ascend_config(vllm_runtime_config)
        self.using_training_dp_replicated_moe = (
            _configure_vllm_moe_training_dp_replication(
                bool(
                    getattr(
                        config,
                        "vllm_training_dp_replicate_moe",
                        True,
                    )
                )
            )
        )
        from vllm_ascend import ops
        from vllm_ascend.utils import enable_custom_op, register_ascend_customop

        self.custom_op_enabled = bool(enable_custom_op())
        ops.register_dummy_fusion_op()
        register_ascend_customop(self.vllm_config)
        _ensure_vllm_ascend_noop_weight_prefetch()
        self.using_moe_unpermute_backward_fallback = (
            _patch_vllm_ascend_moe_unpermute_backward()
        )
        self.using_moe_forward_training_fallback = (
            _patch_vllm_ascend_moe_forward_training_fallback()
        )
        self.using_model_internal_router_training_forward = (
            _patch_vllm_ascend_model_internal_router_training_forward(
                module_names=self._router_module_names()
            )
        )
        self.using_routed_moe_training_backward = (
            _patch_vllm_ascend_routed_moe_training_backward()
        )
        self.using_unquantized_moe_training_forward = (
            _patch_vllm_ascend_unquantized_moe_training_forward()
        )
        self.using_linear_training_backward = (
            _patch_vllm_ascend_linear_training_backward()
        )
        self.using_unquantized_linear_training_backward = (
            _patch_vllm_ascend_unquantized_linear_training_backward()
        )
        self.using_dsa_batchmatmul_training_backward = (
            _patch_vllm_ascend_dsa_batchmatmul_training_backward()
        )
        self.using_sparse_attention_training_backward = (
            _patch_vllm_ascend_sparse_attention_training_backward()
        )
        self.using_rms_norm_training_backward = (
            _patch_vllm_ascend_rms_norm_training_backward()
        )
        self.using_tp1_gather_training_backward = (
            _patch_vllm_ascend_tp1_gather_training_backward()
        )
        self.using_tp_all_reduce_training_backward = (
            _patch_vllm_ascend_tp_all_reduce_training_backward()
        )
        self.using_cv_linear_training_backward = (
            _patch_vllm_ascend_cv_linear_training_backward()
        )
        self.using_fused_qnorm_quant_training_backward = (
            _patch_vllm_ascend_fused_qnorm_quant_training_backward()
        )
        self.using_triton_q_rms_training_backward = (
            _patch_vllm_ascend_triton_q_rms_training_backward()
        )
        self.using_muls_add_training_fallback = (
            _patch_vllm_ascend_muls_add_training_fallback()
        )
        self.using_hc_training_fallback = _patch_vllm_ascend_hc_training_fallback()
        self.using_hc_fallback = self.using_hc_training_fallback
        _ensure_vllm_ascend_parallel_initialized(self.vllm_config, vllm_runtime_config)
        self._set_current_vllm_config = set_current_vllm_config
        with self._set_current_vllm_config(self.vllm_config):
            self.model = DraftModelClass(vllm_config=self.vllm_config)
            # Some draft models import sparse attention while constructing
            # their layers. Retry when an early patch ran before op registration.
            if not self.using_sparse_attention_training_backward:
                self.using_sparse_attention_training_backward = (
                    _patch_vllm_ascend_sparse_attention_training_backward()
                )
            self.using_dsa_device_operator_training_aliases = (
                self._install_variant_runtime_adapters()
            )
            self.using_deterministic_tp_wo_b_forward = (
                _install_deterministic_tp_wo_b_forward(self.model)
            )
            if hasattr(torch, "npu") and torch.npu.is_available():
                self.model.to(torch.device("npu", torch.npu.current_device()))
            self._keep_hc_parameters_float32()
            for module_name, module in self.model.named_modules():
                if module_name.endswith("self_attn.kv_norm"):
                    module._speco_deepseek_v4_sum_grad_output_across_tp = True
                if module_name.endswith("self_attn.q_norm"):
                    module._speco_deepseek_v4_fuse_with_wq_b = True
        self._ensure_vllm_ascend_tp_equivalent(stage="init")
        self.training_parallel_summary = self._tp_state_summary()
        self.hidden_size = config.hidden_size
        self.hc_mult = getattr(config, "hc_mult", 1)
        self._dummy_dsa_cache_shape_overrides: dict[str, tuple[int, ...]] = {}

    def _should_use_tp1_fallback(self, requested_tp: int) -> bool:
        if requested_tp <= 1:
            return False
        if torch.distributed.is_initialized():
            try:
                return torch.distributed.get_world_size() < requested_tp
            except Exception:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
                return True
        env_world_size = (
            os.environ.get("WORLD_SIZE")
            or os.environ.get("RANK_SIZE")
            or os.environ.get("OMPI_COMM_WORLD_SIZE")
        )
        try:
            return int(env_world_size) < requested_tp if env_world_size else True
        except ValueError:
            return True

    def _make_vllm_runtime_config(self, config):
        if not self._using_tp1_fallback:
            return config
        local_config = copy.copy(config)
        local_config.vllm_tensor_parallel_size = 1
        local_config.vllm_data_parallel_size = 1
        local_config.vllm_enable_expert_parallel = False
        return local_config

    def _tp_state_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "requested_tensor_parallel_size": self._requested_tensor_parallel_size,
            "runtime_tensor_parallel_size": self.vllm_config.parallel_config.tensor_parallel_size,
            "runtime_world_size": self.vllm_config.parallel_config.world_size,
            "using_tp1_fallback": self._using_tp1_fallback,
            "torch_distributed_initialized": torch.distributed.is_initialized(),
        }
        if torch.distributed.is_initialized():
            try:
                summary["torch_world_size"] = torch.distributed.get_world_size()
                summary["torch_rank"] = torch.distributed.get_rank()
            except Exception as exc:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
                summary["torch_distributed_error"] = repr(exc)
        try:
            from vllm.distributed import (
                get_dp_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            summary["effective_tp_world_size"] = get_tensor_model_parallel_world_size()
            summary["effective_tp_rank"] = get_tensor_model_parallel_rank()
            dp_group = get_dp_group()
            summary["effective_dp_world_size"] = dp_group.world_size
            summary["effective_dp_rank"] = dp_group.rank_in_group
        except Exception as exc:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
            summary["effective_tp_error"] = repr(exc)
        try:
            first_layer = next(iter(self.model.model.layers.values()))
            decoder_layer = getattr(first_layer, "mtp_block", first_layer)
            wo_b = decoder_layer.self_attn.wo_b
            summary["wo_b_type"] = type(wo_b).__name__
            summary["wo_b_tp_size"] = getattr(wo_b, "tp_size", None)
            summary["wo_b_tp_rank"] = getattr(wo_b, "tp_rank", None)
            summary["wo_b_input_is_parallel"] = getattr(wo_b, "input_is_parallel", None)
            summary["wo_b_reduce_results"] = getattr(wo_b, "reduce_results", None)
            summary["wo_b_input_size"] = getattr(wo_b, "input_size", None)
            summary["wo_b_input_size_per_partition"] = getattr(
                wo_b, "input_size_per_partition", None
            )
            summary["wo_b_weight_shape"] = (
                tuple(wo_b.weight.shape) if hasattr(wo_b, "weight") else None
            )
            summary["wo_b_custom_op"] = type(getattr(wo_b, "custom_op", None)).__name__
        except Exception as exc:  # noqa: BLE001 -- runtime capability probing and numerical failure propagation
            summary["wo_b_error"] = repr(exc)
        moe_layouts = []
        for name, module in self.model.named_modules():
            moe_config = getattr(module, "moe_parallel_config", None)
            if moe_config is None:
                moe_config = getattr(module, "moe_config", None)
            if moe_config is None or not hasattr(moe_config, "tp_size"):
                continue
            moe_layouts.append(
                {
                    "name": name,
                    "tp_size": int(moe_config.tp_size),
                    "tp_rank": int(moe_config.tp_rank),
                    "dp_size": int(moe_config.dp_size),
                    "dp_rank": int(moe_config.dp_rank),
                    "ep_size": int(moe_config.ep_size),
                    "ep_rank": int(moe_config.ep_rank),
                    "use_ep": bool(moe_config.use_ep),
                }
            )
        summary["moe_layouts"] = moe_layouts
        summary["using_sparse_attention_training_backward"] = (
            self.using_sparse_attention_training_backward
        )
        summary["using_dsa_batchmatmul_training_backward"] = (
            self.using_dsa_batchmatmul_training_backward
        )
        summary["using_dsa_device_operator_training_aliases"] = getattr(
            self,
            "using_dsa_device_operator_training_aliases",
            False,
        )
        return summary

    def _ensure_vllm_ascend_tp_equivalent(
        self,
        *,
        stage: str,
        force: bool = False,
    ) -> None:
        requested_tp = int(self._requested_tensor_parallel_size)
        expected_tp = 1 if self._using_tp1_fallback else requested_tp
        expected_dp = int(
            getattr(self.vllm_config.parallel_config, "data_parallel_size", 1)
        )
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        distributed_world_size = dist.get_world_size() if dist.is_initialized() else 1
        topology_signature = (
            requested_tp,
            expected_tp,
            expected_dp,
            global_rank,
            distributed_world_size,
        )
        if not force and self._validated_tp_topology_signature == topology_signature:
            return
        expected_tp_rank = global_rank % expected_tp
        expected_dp_rank = global_rank // expected_tp
        summary = self._tp_state_summary()
        effective_tp = int(summary.get("effective_tp_world_size") or 1)
        wo_b_tp = int(summary.get("wo_b_tp_size") or 1)
        input_size = summary.get("wo_b_input_size")
        input_partition = summary.get("wo_b_input_size_per_partition")
        partition_is_correct = (
            input_size is not None
            and input_partition is not None
            and int(input_partition) * expected_tp == int(input_size)
        )
        moe_layouts = summary.get("moe_layouts", [])
        moe_layout_is_correct = bool(moe_layouts) and all(
            layout["tp_size"] == expected_tp
            and layout["tp_rank"] == expected_tp_rank
            and layout["dp_size"] == 1
            and layout["dp_rank"] == 0
            and layout["ep_size"] == 1
            and layout["use_ep"] is False
            for layout in moe_layouts
        )
        is_equivalent = (
            effective_tp == expected_tp
            and int(summary.get("effective_tp_rank") or 0) == expected_tp_rank
            and int(summary.get("effective_dp_world_size") or 1) == expected_dp
            and int(summary.get("effective_dp_rank") or 0) == expected_dp_rank
            and wo_b_tp == expected_tp
            and bool(partition_is_correct)
            and summary.get("wo_b_input_is_parallel") is True
            and summary.get("wo_b_reduce_results") is True
            and moe_layout_is_correct
        )
        if not is_equivalent:
            raise RuntimeError(
                "vLLM-Ascend MTP topology does not match the expected training "
                f"TP={expected_tp}, DP={expected_dp} layout at {stage}: {summary}"
            )
        self._validated_tp_topology_signature = topology_signature
        if self._using_tp1_fallback and not self._tp1_fallback_warned:
            print(
                "[MTP warning] requested TP is unavailable in this process; using "
                "native vLLM-Ascend TP=1. RowParallelLinear executes the full "
                "quant_method.apply path and skips all-reduce."
            )
            self._tp1_fallback_warned = True

    def set_dummy_dsa_cache_shape_overrides(
        self, overrides: dict[str, tuple[int, ...]]
    ) -> None:
        self._dummy_dsa_cache_shape_overrides = dict(overrides)

    def _apply(self, fn):
        preserved_float32 = {}
        if hasattr(self, "model"):
            for name, param in self.model.named_parameters():
                if (
                    self._requires_float32_parameter(name)
                    and param.dtype == torch.float32
                ):
                    preserved_float32[name] = param.detach().clone()

        result = super()._apply(fn)

        if preserved_float32:
            current_parameters = dict(self.model.named_parameters())
            for name, value in preserved_float32.items():
                param = current_parameters.get(name)
                if param is None:
                    raise RuntimeError(
                        f"DS4 mixed-precision conversion removed parameter {name!r}"
                    )
                param.data = value.to(device=param.device)
        self._keep_hc_parameters_float32()
        return result

    @staticmethod
    def _requires_float32_parameter(name: str) -> bool:
        return (
            "_hc" in name
            or ".hc_" in name
            or name.startswith("hc_")
            or name.endswith((".attn_sink", "e_score_correction_bias"))
            or name == "attn_sink"
        )

    @torch.no_grad()
    def _keep_hc_parameters_float32(self) -> None:
        if not hasattr(self, "model"):
            return
        for name, param in self.model.named_parameters():
            if self._requires_float32_parameter(name):  # noqa: SIM102 -- keep runtime capability guards separate
                if param.dtype != torch.float32:
                    param.data = param.data.to(dtype=torch.float32)

    def _process_weights_after_loading(self) -> None:
        if getattr(self, "_weights_postprocessed", False):
            return
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        try:
            import torch_npu

            torch_npu.npu.config.allow_internal_format = True
        except Exception:  # noqa: BLE001,S110 -- optional accelerator capability fallback
            pass

        process_weights_after_loading(
            self.model,
            self.vllm_config.model_config,
            self.runtime_device(),
        )
        self._weights_postprocessed = True

    @staticmethod
    def _remove_quantization_state(module: nn.Module, names: tuple[str, ...]) -> None:
        for name in names:
            if hasattr(module, name):
                delattr(module, name)

    @torch.no_grad()
    def _dequantize_weights_for_training(self) -> None:
        """Replace W8A8 MTP parameters with trainable BF16 parameters.

        This runs before vLLM's weight postprocessing, while linear weights still
        use checkpoint layout [out, in] and MoE weights use [expert, out, in].
        """
        if not self.dequantize_mtp_for_training:
            return
        if self._weights_postprocessed:
            raise RuntimeError(
                "MTP weights must be dequantized before vLLM postprocessing"
            )

        from verl_speco.backends.vllm_ascend_compat import unquantized_moe_type

        AscendUnquantizedFusedMoEMethod = unquantized_moe_type()
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
        from vllm_ascend.quantization.method_adapters import (
            AscendEmbeddingMethod,
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        moe_type, _ = _vllm_ascend_moe_runtime_type()
        converted: list[str] = []
        unsupported: list[str] = []
        for module_name, module in self.model.named_modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, AscendEmbeddingMethod):
                continue
            if isinstance(method, AscendLinearMethod):
                weight = getattr(module, "weight", None)
                scale = getattr(module, "weight_scale", None)
                if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8:
                    unsupported.append(
                        f"{module_name}: expected int8 linear weight before postprocess"
                    )
                    continue
                if not isinstance(scale, torch.Tensor):
                    unsupported.append(f"{module_name}: missing linear weight_scale")
                    continue
                output_size = int(weight.shape[0])
                scale_flat = scale.float().reshape(-1)
                if scale_flat.numel() != output_size:
                    unsupported.append(
                        f"{module_name}: scale size {scale_flat.numel()} != output {output_size}"
                    )
                    continue
                dequantized = weight.float() * scale_flat[:, None]
                module.weight = nn.Parameter(
                    dequantized.to(torch.bfloat16), requires_grad=True
                )
                self._remove_quantization_state(
                    module,
                    (
                        "weight_scale",
                        "weight_offset",
                        "weight_scale_fp32",
                        "deq_scale",
                        "quant_bias",
                    ),
                )
                module.quant_method = AscendUnquantizedLinearMethod()
                custom_op = getattr(module, "custom_op", None)
                if custom_op is not None:
                    if hasattr(custom_op, "update_attrs"):
                        custom_op.update_attrs()
                    else:
                        custom_op.quant_method = module.quant_method
                converted.append(module_name)
                continue

            if isinstance(module, moe_type):
                experts = _vllm_ascend_moe_weights_module(module)
                method = getattr(experts, "quant_method", None)
                if not isinstance(method, AscendFusedMoEMethod):
                    continue
                w13 = getattr(experts, "w13_weight", None)
                w2 = getattr(experts, "w2_weight", None)
                w13_scale = getattr(experts, "w13_weight_scale", None)
                w2_scale = getattr(experts, "w2_weight_scale", None)
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (w13, w2, w13_scale, w2_scale)
                ):
                    unsupported.append(
                        f"{module_name}: incomplete routed MoE W8A8 state"
                    )
                    continue
                if w13.dtype != torch.int8 or w2.dtype != torch.int8:
                    unsupported.append(
                        f"{module_name}: routed MoE weights are not int8"
                    )
                    continue
                w13_scale = w13_scale.float().reshape(w13.shape[0], w13.shape[1], 1)
                w2_scale = w2_scale.float().reshape(w2.shape[0], w2.shape[1], 1)
                experts.w13_weight = nn.Parameter(
                    (w13.float() * w13_scale).to(torch.bfloat16), requires_grad=True
                )
                experts.w2_weight = nn.Parameter(
                    (w2.float() * w2_scale).to(torch.bfloat16), requires_grad=True
                )
                self._remove_quantization_state(
                    experts,
                    (
                        "w13_weight_scale",
                        "w13_weight_offset",
                        "w13_weight_scale_fp32",
                        "w2_weight_scale",
                        "w2_weight_offset",
                        "fused_w1_scale",
                        "fused_w2_scale",
                    ),
                )
                unquantized = AscendUnquantizedFusedMoEMethod(
                    module.moe_config, tid2eid=getattr(module, "tid2eid", None)
                )
                experts.quant_method = unquantized
                if hasattr(experts, "base_quant_method"):
                    experts.base_quant_method = unquantized
                module.moe_config.supports_eplb = unquantized.supports_eplb
                converted.append(
                    f"{module_name}.routed_experts"
                    if experts is not module
                    else module_name
                )

        if unsupported:
            details = "\n  ".join(unsupported)
            raise RuntimeError(
                "Cannot convert all quantized MTP modules to BF16 training weights:\n  "
                + details
            )

        # DSA creates CVLinearWrapper objects during module construction. They
        # cache whether the wrapped linear is W8A8, so refresh them after the
        # underlying Linear method has been replaced.
        for module in self.model.modules():
            for value in vars(module).values():
                if value.__class__.__name__ != "CVLinearWrapper":
                    continue
                linear = getattr(value, "linear", None)
                if linear is None:
                    continue
                value._quant_method = linear.quant_method
                value._is_w8a8_dynamic = value._detect_w8a8_dynamic(linear.quant_method)

        stale_linear_methods = []
        stale_cv_wrappers = []
        for module_name, module in self.model.named_modules():
            custom_op = getattr(module, "custom_op", None)
            if custom_op is not None and hasattr(module, "quant_method"):
                cached_method = getattr(custom_op, "quant_method", module.quant_method)
                if cached_method is not module.quant_method:
                    stale_linear_methods.append(module_name)
            for attr_name, value in vars(module).items():
                if value.__class__.__name__ != "CVLinearWrapper":
                    continue
                linear = getattr(value, "linear", None)
                if (
                    linear is not None
                    and value._quant_method is not linear.quant_method
                ):
                    stale_cv_wrappers.append(f"{module_name}.{attr_name}")
                if (
                    linear is not None
                    and isinstance(linear.quant_method, AscendUnquantizedLinearMethod)
                    and value._is_w8a8_dynamic
                ):
                    stale_cv_wrappers.append(f"{module_name}.{attr_name}:w8a8")
        if stale_linear_methods:
            raise RuntimeError(
                "Linear custom ops still cache pre-dequantization quant methods: "
                + ", ".join(stale_linear_methods)
            )
        if stale_cv_wrappers:
            raise RuntimeError(
                "DSA CV linear wrappers still cache pre-dequantization state: "
                + ", ".join(stale_cv_wrappers)
            )
        self._dequantized_training_modules = converted
        if converted:
            print(
                f"Dequantized {len(converted)} MTP linear/MoE modules for BF16 training"
            )

    def _validate_loaded_training_weight_format(self) -> None:
        if self.mtp_checkpoint_format not in {"w8a8", "bf16"}:
            return
        from vllm_ascend.quantization.method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        moe_type, _ = _vllm_ascend_moe_runtime_type()
        integer_weights = []
        quantized_methods = []
        for module_name, module in self.model.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.is_floating_point() and any(
                    token in parameter_name for token in ("weight", "w13", "w2")
                ):
                    integer_weights.append(
                        f"{module_name}.{parameter_name}:{parameter.dtype}"
                    )
            method = getattr(module, "quant_method", None)
            if isinstance(method, AscendLinearMethod):
                quantized_methods.append(f"{module_name}:{type(method).__name__}")
            if isinstance(module, moe_type) and isinstance(
                getattr(
                    _vllm_ascend_moe_weights_module(module),
                    "quant_method",
                    None,
                ),
                AscendFusedMoEMethod,
            ):
                quantized_methods.append(
                    f"{module_name}:{type(_vllm_ascend_moe_weights_module(module).quant_method).__name__}"
                )
        if integer_weights or quantized_methods:
            details = []
            if integer_weights:
                details.append("integer weights=" + ", ".join(integer_weights))
            if quantized_methods:
                details.append("quantized methods=" + ", ".join(quantized_methods))
            raise RuntimeError(
                "DS4 MTP checkpoint did not reach the common BF16 training "
                "contract after loading: " + "; ".join(details)
            )
        _increment_training_diagnostic(
            f"checkpoint_{self.mtp_checkpoint_format}_training_loads"
        )

    def runtime_device(self) -> torch.device:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu", torch.npu.current_device())
        return next(self.model.parameters()).device

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.runtime_device())
        with self._set_current_vllm_config(self.vllm_config):
            return self.model.embed_input_ids(input_ids)

    @staticmethod
    def _needs_cache_rebuild(
        cache: Any,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        *,
        grow_first_dim: bool = False,
    ) -> bool:
        if not isinstance(cache, torch.Tensor):
            return True
        return (
            not cache_shape_has_capacity(
                tuple(cache.shape),
                shape,
                grow_first_dim=grow_first_dim,
            )
            or cache.dtype != dtype
            or cache.device != device
        )

    @staticmethod
    def _unwrap_cache_tensor(cache: Any) -> torch.Tensor | None:
        current = cache
        while isinstance(current, (list, tuple)) and current:
            current = current[0]
        return current if isinstance(current, torch.Tensor) else None

    @staticmethod
    def _wrap_cache_like_runtime(cache: torch.Tensor) -> list[list[torch.Tensor]]:
        return [[cache]]

    @staticmethod
    def _zeros_cache(
        shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    def _build_dummy_common_attn_metadata(
        self,
        *,
        positions: torch.Tensor,
        positions_cpu: torch.Tensor,
        batch_shape: tuple[int, ...],
        block_size: int,
        num_blocks_per_req: int,
        num_blocks: int,
        device: torch.device,
    ):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        if len(batch_shape) == 0:
            num_reqs = 1
            query_len = int(positions.numel())
        else:
            num_reqs = (
                int(math.prod(batch_shape[:-1]))
                if len(batch_shape) > 1
                else int(batch_shape[0])
            )
            query_len = int(batch_shape[-1])
        key = (
            tuple(int(dim) for dim in batch_shape),
            block_size,
            num_blocks_per_req,
            num_blocks,
            str(device),
        )
        structure = self._dsa_common_structure_cache.get(key)
        if structure is not None and self.cache_dsa_metadata_structures:
            _increment_training_diagnostic("dsa_metadata_structure_cache_hits")
        else:
            _increment_training_diagnostic("dsa_metadata_structure_cache_misses")
            structure = _build_dsa_common_structure_tensors(
                num_reqs=num_reqs,
                query_len=query_len,
                block_size=block_size,
                num_blocks_per_req=num_blocks_per_req,
                device=device,
            )
            if self.cache_dsa_metadata_structures:
                if len(self._dsa_common_structure_cache) >= 64:
                    self._dsa_common_structure_cache.clear()
                    _increment_training_diagnostic(
                        "dsa_metadata_structure_cache_evictions"
                    )
                self._dsa_common_structure_cache[key] = structure
        if num_blocks < num_reqs * num_blocks_per_req:
            raise ValueError(
                "dummy DSA cache block allocation is smaller than block table"
            )

        flat_positions = positions.reshape(-1).to(device)
        flat_positions_cpu = positions_cpu.reshape(-1)
        if flat_positions_cpu.device.type != "cpu":
            raise ValueError("positions_cpu must remain on CPU")
        if flat_positions_cpu.numel() != flat_positions.numel():
            raise ValueError(
                "CPU/NPU positions have inconsistent token counts: "
                f"cpu={flat_positions_cpu.numel()}, npu={flat_positions.numel()}"
            )
        query_lens_cpu = structure["query_lens_cpu"]
        query_start_loc_cpu = structure["query_start_loc_cpu"]
        seq_lens = structure["seq_lens"]
        seq_lens_cpu = structure["seq_lens_cpu"]
        num_tokens = num_reqs * query_len
        return AscendCommonAttentionMetadata(
            query_start_loc=structure["query_start_loc"],
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            num_computed_tokens_cpu=structure["num_computed_tokens_cpu"],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=int(query_lens_cpu.max().item()) if num_reqs else 0,
            max_seq_len=int(seq_lens_cpu.max().item()) if num_reqs else 0,
            block_table_tensor=structure["block_table"],
            slot_mapping=structure["slot_mapping"],
            causal=True,
            is_prefilling=structure["is_prefilling"],
            num_input_tokens=num_tokens,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            positions=flat_positions,
            positions_cpu=flat_positions_cpu,
            attn_state=AscendAttentionState.ChunkedPrefill,
            decode_token_per_req=1,
        )

    def _build_dsa_metadata(
        self,
        *,
        kv_cache_spec: Any,
        layer_name: str,
        positions: torch.Tensor,
        positions_cpu: torch.Tensor,
        batch_shape: tuple[int, ...],
        device: torch.device,
        ratio_state: dict[str, Any],
    ) -> Any:
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

        block_size = int(
            getattr(
                kv_cache_spec, "block_size", self.vllm_config.cache_config.block_size
            )
        )
        num_reqs = (
            int(math.prod(batch_shape[:-1]))
            if len(batch_shape) > 1
            else int(batch_shape[0])
        )
        query_len = int(batch_shape[-1]) if batch_shape else int(positions.numel())
        num_blocks_per_req = max(1, math.ceil(max(query_len, 1) / block_size))
        num_blocks = max(1, num_reqs * num_blocks_per_req + 1)
        common_metadata = self._build_dummy_common_attn_metadata(
            positions=positions,
            positions_cpu=positions_cpu,
            batch_shape=batch_shape,
            block_size=block_size,
            num_blocks_per_req=num_blocks_per_req,
            num_blocks=num_blocks,
            device=device,
        )
        builder_key = (
            layer_name,
            block_size,
            int(getattr(kv_cache_spec, "compress_ratio", 1)),
            str(device),
        )
        builder = self._dsa_metadata_builder_cache.get(builder_key)
        if builder is not None and self.cache_dsa_metadata_structures:
            _increment_training_diagnostic("dsa_metadata_builder_cache_hits")
        else:
            _increment_training_diagnostic("dsa_metadata_builder_cache_misses")
            builder = AscendDSAMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=[layer_name],
                vllm_config=self.vllm_config,
                device=device,
            )
            if self.cache_dsa_metadata_structures:
                self._dsa_metadata_builder_cache[builder_key] = builder
        built_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata,
            compress_ratio=getattr(kv_cache_spec, "compress_ratio", 1),
            num_reqs_actual=num_reqs,
            prefill_ratio_to_sas_metadata=ratio_state.setdefault("prefill", {}),
            decode_ratio_to_sas_metadata=ratio_state.setdefault("decode", {}),
            common_ratio_to_sas_metadata=ratio_state.setdefault("common", {}),
            block_size=block_size,
        )
        if self.reuse_dsa_backward_host_metadata:
            _register_dsa_backward_host_metadata(
                built_metadata,
                common_metadata,
                int(getattr(kv_cache_spec, "compress_ratio", 1)),
            )
        return built_metadata

    def _ensure_dummy_dsa_kv_cache_for_layer(
        self,
        *,
        attn_module: Any,
        batch_shape: tuple[int, ...],
        device: torch.device,
        swa_num_blocks: int | None = None,
    ) -> int:
        query_len = int(batch_shape[-1]) if batch_shape else 1
        num_reqs = (
            int(math.prod(batch_shape[:-1]))
            if len(batch_shape) > 1
            else int(batch_shape[0])
        )
        block_size = int(self.vllm_config.cache_config.block_size)
        num_blocks = required_training_cache_blocks(
            num_reqs,
            query_len,
            block_size,
        )

        swa = attn_module.swa_cache_layer
        swa_block_size = int(getattr(swa, "block_size", 64))
        max_model_len = int(
            getattr(self.vllm_config.model_config, "max_model_len", 0) or 0
        )
        if max_model_len <= 0:
            max_model_len = int(
                getattr(self.vllm_config.scheduler_config, "max_model_len", 0) or 0
            )
        if max_model_len <= 0:
            max_model_len = int(max(query_len, 1))
        required_swa_blocks = required_training_cache_blocks(
            num_reqs,
            query_len,
            swa_block_size,
        )
        if swa_num_blocks is not None:
            if swa_num_blocks <= 0:
                raise ValueError("swa_num_blocks must be positive")
            required_swa_blocks = swa_num_blocks
        swa_blocks = (
            required_swa_blocks
            if self.compact_training_kv_cache
            else max(
                required_swa_blocks,
                max(1, math.ceil(max_model_len / max(swa_block_size, 1))),
            )
        )
        cache_dtype = getattr(self.vllm_config.cache_config, "cache_dtype", "auto")
        cache_override = getattr(self, "_dummy_dsa_cache_shape_overrides", {}).get(
            f"{attn_module.prefix}.swa_cache"
        )
        if cache_override is not None:
            swa_shape = tuple(int(dim) for dim in cache_override)
        elif cache_dtype == "fp8_ds_mla":
            swa_shape = (swa_blocks, swa_block_size, 1, 584)
        else:
            swa_shape = (swa_blocks, swa_block_size, 1, int(attn_module.head_dim))
        swa_dtype = getattr(swa, "dtype", torch.bfloat16)
        current_swa = self._unwrap_cache_tensor(getattr(swa, "kv_cache", None))
        grow_swa_cache = self.compact_training_kv_cache and cache_override is None
        if self._needs_cache_rebuild(
            current_swa,
            swa_shape,
            swa_dtype,
            device,
            grow_first_dim=grow_swa_cache,
        ):
            swa.kv_cache = self._wrap_cache_like_runtime(
                self._zeros_cache(swa_shape, swa_dtype, device)
            )
            cache_key = f"{attn_module.prefix}.swa_cache"
            if grow_swa_cache and cache_key not in self._compact_training_cache_logged:
                cache_bytes = (
                    math.prod(swa_shape)
                    * torch.empty((), dtype=swa_dtype).element_size()
                )
                print(
                    "Compact training SWA cache: "
                    f"{cache_key}, shape={swa_shape}, size={cache_bytes / (1024**2):.3f} MiB"
                )
                self._compact_training_cache_logged.add(cache_key)

        if int(getattr(attn_module, "compress_ratio", 1)) > 1:
            compress_shape = (num_blocks, block_size, 1, int(attn_module.head_dim))
            compress_dtype = getattr(
                attn_module.dsa_attn, "kv_cache_dtype", torch.bfloat16
            )
            if isinstance(compress_dtype, str):
                compress_dtype = torch.bfloat16
            current = (
                attn_module.dsa_attn.kv_cache[0]
                if isinstance(attn_module.dsa_attn.kv_cache, list)
                and attn_module.dsa_attn.kv_cache
                else None
            )
            if self._needs_cache_rebuild(
                current,
                compress_shape,
                compress_dtype,
                device,
                grow_first_dim=self.compact_training_kv_cache,
            ):
                attn_module.dsa_attn.kv_cache = [
                    self._zeros_cache(compress_shape, compress_dtype, device)
                ]

            state_cache = attn_module.compressor.state_cache
            state_block_size = int(getattr(state_cache, "block_size", 8))
            state_blocks = max(
                1,
                num_reqs * max(1, math.ceil(max(query_len, 1) / state_block_size)) + 1,
            )
            state_shape = (
                state_blocks,
                state_block_size,
                int(getattr(state_cache, "state_dim", attn_module.head_dim)),
            )
            state_dtype = getattr(state_cache, "dtype", torch.float32)
            if self._needs_cache_rebuild(
                getattr(state_cache, "kv_cache", None),
                state_shape,
                state_dtype,
                device,
                grow_first_dim=self.compact_training_kv_cache,
            ):
                state_cache.kv_cache = self._zeros_cache(
                    state_shape, state_dtype, device
                )

        if (
            int(getattr(attn_module, "compress_ratio", 1)) == 4
            and getattr(attn_module, "indexer", None) is not None
        ):
            indexer = attn_module.indexer
            idx_state_cache = indexer.compressor.state_cache
            idx_state_block_size = int(getattr(idx_state_cache, "block_size", 4))
            idx_state_blocks = max(
                1,
                num_reqs * max(1, math.ceil(max(query_len, 1) / idx_state_block_size))
                + 1,
            )
            idx_state_shape = (
                idx_state_blocks,
                idx_state_block_size,
                int(
                    getattr(
                        idx_state_cache, "state_dim", getattr(indexer, "head_dim", 128)
                    )
                ),
            )
            idx_state_dtype = getattr(idx_state_cache, "dtype", torch.float32)
            if self._needs_cache_rebuild(
                getattr(idx_state_cache, "kv_cache", None),
                idx_state_shape,
                idx_state_dtype,
                device,
                grow_first_dim=self.compact_training_kv_cache,
            ):
                idx_state_cache.kv_cache = self._zeros_cache(
                    idx_state_shape, idx_state_dtype, device
                )

            idx_k_cache = indexer.k_cache
            idx_block_size = int(self.vllm_config.cache_config.block_size)
            idx_blocks = num_blocks
            idx_shape = (
                idx_blocks,
                idx_block_size,
                1,
                int(getattr(indexer, "head_dim", 128)),
            )
            idx_dtype = getattr(idx_k_cache, "dtype", torch.float8_e4m3fn)
            idx_scale_shape = (idx_blocks, idx_block_size, 1, 1)
            idx_scale_dtype = torch.float32
            current_pair = (
                idx_k_cache.kv_cache[0]
                if isinstance(idx_k_cache.kv_cache, list) and idx_k_cache.kv_cache
                else (None, None)
            )
            current_k, current_scale = (
                current_pair
                if isinstance(current_pair, tuple) and len(current_pair) >= 2
                else (None, None)
            )
            if self._needs_cache_rebuild(
                current_k,
                idx_shape,
                idx_dtype,
                device,
                grow_first_dim=self.compact_training_kv_cache,
            ) or self._needs_cache_rebuild(
                current_scale,
                idx_scale_shape,
                idx_scale_dtype,
                device,
                grow_first_dim=self.compact_training_kv_cache,
            ):
                idx_k_cache.kv_cache = [
                    (
                        self._zeros_cache(idx_shape, idx_dtype, device),
                        self._zeros_cache(idx_scale_shape, idx_scale_dtype, device),
                    )
                ]
        return num_blocks

    def _build_dummy_dsa_attn_metadata(
        self,
        *,
        positions: torch.Tensor,
        positions_cpu: torch.Tensor,
        batch_shape: tuple[int, ...],
        device: torch.device,
        module_root: nn.Module | None = None,
    ) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {}
        # DeepSeekV4MTP executes exactly one speculative layer for a given
        # spec_step_idx.  Building AICPU scheduler metadata for every MTP layer
        # made each training forward O(num_mtp_layers) and queued metadata
        # work for layers that were not executed.
        modules = (
            module_root.modules() if module_root is not None else self.model.modules()
        )
        for module in modules:
            if not (
                hasattr(module, "dsa_attn")
                and hasattr(module, "swa_cache_layer")
                and hasattr(module, "prefix")
            ):
                continue
            self._ensure_dummy_dsa_kv_cache_for_layer(
                attn_module=module, batch_shape=batch_shape, device=device
            )
            ratio_state: dict[str, Any] = {}
            prefix = module.prefix
            compress_ratio = int(getattr(module, "compress_ratio", 1))
            # AscendDSAMetadataBuilder currently materializes QLI scheduling
            # metadata for every DSA prefill build, independent of the layer's
            # compression ratio. Its inputs are stable for a fixed training
            # batch shape, so reuse it across speculative forwards.
            qli_cache_key = (
                prefix,
                tuple(int(dim) for dim in batch_shape),
                str(device),
            )
            qli_cache_hit = False
            cached_qli_metadata = self._dsa_qli_metadata_cache.get(qli_cache_key)
            if cached_qli_metadata is not None and self.cache_dsa_metadata_structures:
                ratio_state["prefill"] = {
                    "qli": cached_qli_metadata,
                }
                qli_cache_hit = True
                _increment_training_diagnostic("dsa_qli_metadata_cache_hits")
            else:
                _increment_training_diagnostic("dsa_qli_metadata_cache_misses")
            if compress_ratio > 1:
                dsa_spec = module.dsa_attn.get_kv_cache_spec(self.vllm_config)
                if dsa_spec is not None:
                    metadata[f"{prefix}.attn"] = self._build_dsa_metadata(
                        kv_cache_spec=dsa_spec,
                        layer_name=f"{prefix}.attn",
                        positions=positions,
                        positions_cpu=positions_cpu,
                        batch_shape=batch_shape,
                        device=device,
                        ratio_state=ratio_state,
                    )
                state_spec = module.compressor.state_cache.get_kv_cache_spec(
                    self.vllm_config
                )
                metadata[f"{prefix}.compressor.state_cache"] = self._build_dsa_metadata(
                    kv_cache_spec=state_spec,
                    layer_name=f"{prefix}.compressor.state_cache",
                    positions=positions,
                    positions_cpu=positions_cpu,
                    batch_shape=batch_shape,
                    device=device,
                    ratio_state=ratio_state,
                )
            if compress_ratio == 4 and getattr(module, "indexer", None) is not None:
                indexer_state_spec = (
                    module.indexer.compressor.state_cache.get_kv_cache_spec(
                        self.vllm_config
                    )
                )
                metadata[f"{prefix}.indexer.compressor.state_cache"] = (
                    self._build_dsa_metadata(
                        kv_cache_spec=indexer_state_spec,
                        layer_name=f"{prefix}.indexer.compressor.state_cache",
                        positions=positions,
                        positions_cpu=positions_cpu,
                        batch_shape=batch_shape,
                        device=device,
                        ratio_state=ratio_state,
                    )
                )
                indexer_spec = module.indexer.k_cache.get_kv_cache_spec(
                    self.vllm_config
                )
                metadata[f"{prefix}.indexer.k_cache"] = self._build_dsa_metadata(
                    kv_cache_spec=indexer_spec,
                    layer_name=f"{prefix}.indexer.k_cache",
                    positions=positions,
                    positions_cpu=positions_cpu,
                    batch_shape=batch_shape,
                    device=device,
                    ratio_state=ratio_state,
                )
            swa_spec = module.swa_cache_layer.get_kv_cache_spec(self.vllm_config)
            metadata[f"{prefix}.swa_cache"] = self._build_dsa_metadata(
                kv_cache_spec=swa_spec,
                layer_name=f"{prefix}.swa_cache",
                positions=positions,
                positions_cpu=positions_cpu,
                batch_shape=batch_shape,
                device=device,
                ratio_state=ratio_state,
            )
            if (
                self.cache_dsa_metadata_structures
                and qli_cache_key is not None
                and not qli_cache_hit
            ):
                qli_metadata = ratio_state.get("prefill", {}).get("qli")
                if isinstance(qli_metadata, torch.Tensor):
                    if len(self._dsa_qli_metadata_cache) >= 64:
                        self._dsa_qli_metadata_cache.clear()
                        _increment_training_diagnostic(
                            "dsa_qli_metadata_cache_evictions"
                        )
                    self._dsa_qli_metadata_cache[qli_cache_key] = qli_metadata
        return metadata or None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError(
                "vLLM-Ascend DS4 MTP branch does not accept attention_mask directly."
            )
        self._ensure_vllm_ascend_tp_equivalent(stage="forward")
        device = self.runtime_device()
        input_ids = input_ids.to(device)
        positions = positions.to(device)
        positions_cpu = kwargs.pop("positions_cpu", None)
        if positions_cpu is None:
            positions_cpu = positions.detach().to("cpu")
            _increment_training_diagnostic("dsa_positions_npu_to_cpu_fallbacks")
        else:
            positions_cpu = positions_cpu.detach().to("cpu", copy=False)
            _increment_training_diagnostic("dsa_positions_cpu_direct_calls")
        hidden_states = hidden_states.to(device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device)
        output_prefix_shape = tuple(input_ids.shape)
        input_ids = input_ids.reshape(-1)
        positions = positions.reshape(-1)
        hidden_states = hidden_states.reshape(-1, self.hc_mult * self.hidden_size)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(-1, self.hidden_size)
        num_tokens = int(input_ids.numel())
        current_step_idx = int(spec_step_idx) % int(self.model.model.num_mtp_layers)
        active_mtp_layer = self.model.model.layers[str(current_step_idx)]
        attn_metadata = self._build_dummy_dsa_attn_metadata(
            positions=positions,
            positions_cpu=positions_cpu,
            batch_shape=output_prefix_shape,
            device=device,
            module_root=active_mtp_layer,
        )
        with self._set_current_vllm_config(self.vllm_config):
            from verl_speco.backends.vllm_ascend_compat import (
                ascend_forward_context as set_ascend_forward_context,
            )

            forward_context = set_ascend_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                model_instance=self.model,
                is_draft_model=True,
                input_ids=input_ids,
            )
            with forward_context:
                output = self.model.forward(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    spec_step_idx=spec_step_idx,
                )
        if output.ndim == 3 and output.shape[0] == num_tokens:
            return output.reshape(
                *output_prefix_shape, output.shape[-2], output.shape[-1]
            )
        return output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module | None = None,
        spec_step_idx: int = 0,
    ):
        hidden_states = hidden_states.to(self.runtime_device())
        num_tokens = int(hidden_states.numel() // (self.hc_mult * self.hidden_size))
        with self._set_current_vllm_config(self.vllm_config):
            from verl_speco.backends.vllm_ascend_compat import (
                ascend_forward_context as set_ascend_forward_context,
            )

            forward_context = set_ascend_forward_context(
                attn_metadata=None,
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                model_instance=self.model,
                is_draft_model=True,
            )
            with forward_context:
                return self.model.compute_logits(
                    hidden_states, spec_step_idx=spec_step_idx
                )

    def compute_local_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
        hidden_states = hidden_states.to(self.runtime_device())
        current_step_idx = spec_step_idx % self.model.model.num_mtp_layers
        mtp_layer = self.model.model.layers[str(current_step_idx)]
        hidden_states = hidden_states.view(
            -1,
            mtp_layer.hc_mult,
            mtp_layer.config.hidden_size,
        )
        hidden_states = mtp_layer.hc_head(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
        )
        hidden_states = mtp_layer.shared_head(hidden_states)
        hidden_states = _TensorParallelCopy.apply(hidden_states)
        head = mtp_layer.shared_head.head
        local_logits = head.quant_method.apply(head, hidden_states, bias=head.bias)
        valid_size = int(head.num_org_embeddings_per_partition)
        return local_logits[..., :valid_size]

    def load_weights(self, path: str, mode: str = "full") -> dict[str, Any]:
        if mode == "no":
            return {
                "loaded": 0,
                "skipped": 0,
                "mismatched": [],
                "not_updated": [],
                "unresolved": [],
                "updated_keys": [],
            }
        weights = []
        shared_weights = {}
        dedicated_mtp_targets = set()
        for root, _, files in os.walk(path):
            for f in files:
                if not f.endswith(".safetensors"):
                    continue
                file_path = os.path.join(root, f)
                with safe_open(file_path, framework="pt", device="cpu") as sf:
                    for key in sf.keys():  # noqa: SIM118 -- safetensors/runtime key API is explicit
                        if key.startswith("mtp."):
                            weights.append((key, sf.get_tensor(key)))
                            if key.endswith(".head.weight"):
                                dedicated_mtp_targets.add(
                                    "model.layers.0.shared_head.head.weight"
                                )
                            elif key.endswith(".emb.tok_emb.weight"):
                                dedicated_mtp_targets.add("model.embed_tokens.weight")
                        elif key in (
                            "model.embed_tokens.weight",
                            "embed_tokens.weight",
                            "embed.weight",
                        ):
                            shared_weights["model.embed_tokens.weight"] = sf.get_tensor(
                                key
                            )
                        elif key in (
                            "lm_head.weight",
                            "model.lm_head.weight",
                            "head.weight",
                        ):
                            shared_weights["model.layers.0.shared_head.head.weight"] = (
                                sf.get_tensor(key)
                            )
        with self._set_current_vllm_config(self.vllm_config):
            loaded_params = self.model.load_weights(weights)
            missing_dedicated = dedicated_mtp_targets - loaded_params
            if missing_dedicated:
                raise RuntimeError(
                    "Checkpoint contains dedicated MTP weights that vLLM-Ascend "
                    f"did not load: {sorted(missing_dedicated)}"
                )
            loaded_shared_params = self._load_shared_main_weights(
                shared_weights,
                protected_names=loaded_params,
            )
            loaded_params.update(loaded_shared_params)
            self._dequantize_weights_for_training()
            self._validate_loaded_training_weight_format()
            self._process_weights_after_loading()
            self.using_tp_sharded_eh_projections = _install_tp_sharded_eh_projections(
                self.model
            )
            self.using_fused_eh_projection_collective = (
                _install_fused_eh_projection_collective(self.model)
            )
            self.using_tp_sharded_attention_projections = (
                _install_tp_sharded_attention_projections(self.model)
            )
        self._keep_hc_parameters_float32()
        self._ensure_vllm_ascend_tp_equivalent(
            stage="load_weights",
            force=True,
        )
        total_candidate_weights = len(weights) + len(shared_weights)
        return {
            "loaded": len(loaded_params),
            "skipped": max(total_candidate_weights - len(loaded_params), 0),
            "mismatched": [],
            "not_updated": [],
            "unresolved": [],
            "updated_keys": sorted(loaded_params),
            "checkpoint_format": self.mtp_checkpoint_format,
            "checkpoint_format_report": self.mtp_checkpoint_format_report,
        }

    @torch.no_grad()
    def _load_shared_main_weights(
        self,
        shared_weights: dict[str, torch.Tensor],
        *,
        protected_names: set[str],
    ) -> set[str]:
        if not shared_weights:
            return set()
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params_dict = dict(self.model.named_parameters())
        loaded_params = set()
        for name, loaded_weight in shared_weights.items():
            if name in protected_names:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
