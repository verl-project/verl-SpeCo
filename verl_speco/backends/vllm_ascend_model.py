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
"""Construct the native DeepSeek V4 DSpark model with explicit runtime options."""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
from omegaconf import dictconfig
from torch import nn

from verl_speco.backends.vllm_ascend_options import (
    export_ds4_model_construction_environment,
    resolve_optimization_config,
)
from verl_speco.backends.vllm_ascend_parallel import is_main_process
from verl_speco.models.deepseek_v4.checkpoint import inspect_ds4_checkpoint_format
from verl_speco.models.deepseek_v4.modeling_dspark import DeepseekV4DSparkServingWrapper


def load_mtp_draft_model(
    cfg: dictconfig.DictConfig, logger: logging.Logger
) -> nn.Module:
    draft_model_path = cfg.model.path
    if not os.path.isdir(draft_model_path):
        raise FileNotFoundError(
            f"Draft model path must be a local directory: {draft_model_path}"
        )

    if is_main_process():
        logger.info("Initializing MTP draft model from: %s", draft_model_path)

    model_type = getattr(cfg.model, "type", "ds4_dspark")
    if model_type != "ds4_dspark":
        raise ValueError("This backend factory only constructs DeepSeek V4 DSpark")
    optimizations = resolve_optimization_config(cfg)
    model_construction_optimizations = {}
    if model_type in {"ds4", "ds4_dspark"}:
        model_construction_optimizations = export_ds4_model_construction_environment(
            cfg
        )

    model_architecture = DeepseekV4DSparkServingWrapper
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

    config_type = DeepseekV4Config
    config = config_type.from_pretrained(draft_model_path)
    config._attn_implementation = "sdpa"
    if model_type == "ds4_dspark":
        config.dspark_enable_confidence_head = bool(
            getattr(cfg.train, "dspark_enable_confidence_head", False)
        )
        config.dspark_confidence_weight = float(
            getattr(cfg.train, "dspark_confidence_weight", 0.0)
        )
        config.dspark_anchor_min_spacing = getattr(
            cfg.train,
            "dspark_anchor_min_spacing",
            None,
        )
        if (
            config.dspark_confidence_weight > 0
            and not config.dspark_enable_confidence_head
        ):
            raise ValueError(
                "train.dspark_confidence_weight requires "
                "train.dspark_enable_confidence_head=true"
            )

    if model_type == "glm5":
        config.ep_size = dist.get_world_size() if dist.is_initialized() else 1
    if model_type in {"ds4", "ds4_dspark"}:
        configured_layout = getattr(cfg.model, "ds4_layout", "vllm_ascend")
        if configured_layout != "vllm_ascend":
            raise ValueError(
                "DeepSeek V4 training supports only model.ds4_layout=vllm_ascend"
            )
        checkpoint_report = inspect_ds4_checkpoint_format(
            draft_model_path,
            getattr(cfg.model, "checkpoint_format", "auto"),
        )
        if is_main_process():
            logger.info(
                "Detected DS4 MTP checkpoint format: format=%s, "
                "dtypes=%s, mtp_tensors=%d, quant_description=%s",
                checkpoint_report.resolved_format,
                checkpoint_report.dtype_counts,
                checkpoint_report.mtp_tensors,
                checkpoint_report.quant_description_present,
            )
        environment_flags = {
            "MTP_DS4_GROUPED_ROUTED_MOE_REPLAY": "grouped_routed_moe_replay",
            "MTP_DS4_FP32_ROUTE_ARGSORT": "fp32_route_argsort",
            "MTP_DS4_FUSED_HC_FORWARD": "fused_hc_forward",
            "MTP_DS4_FUSED_ROUTED_MOE_TOKEN_UNPERMUTE": (
                "fused_routed_moe_token_unpermute"
            ),
            "MTP_DS4_REUSE_DSA_BACKWARD_STRUCTURES": ("reuse_dsa_backward_structures"),
            "MTP_DS4_BUFFERED_DETERMINISTIC_REDUCE": ("buffered_deterministic_reduce"),
            "MTP_DS4_TP_SHARD_EH_PROJECTIONS": "tp_shard_eh_projections",
            "MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS": (
                "tp_shard_attention_projections"
            ),
            "MTP_DS4_FUSED_EH_PROJECTION_COLLECTIVE": (
                "fused_eh_projection_collective"
            ),
        }
        for environment_name, option_name in environment_flags.items():
            enabled = bool(getattr(optimizations, option_name, False))
            os.environ.setdefault(environment_name, "1" if enabled else "0")

        config.ds4_layout = "vllm_ascend"
        config.ds4_mtp_impl = "vllm_import_train"
        config.use_ascend_hc_ops = getattr(cfg.model, "use_ascend_hc_ops", False)
        config.vllm_model_path = draft_model_path
        config.vllm_tensor_parallel_size = getattr(
            cfg.model, "vllm_tensor_parallel_size", 1
        )
        config.vllm_max_model_len = getattr(cfg.model, "vllm_max_model_len", 135168)
        config.vllm_max_num_seqs = getattr(cfg.model, "vllm_max_num_seqs", 32)
        config.vllm_max_num_batched_tokens = getattr(
            cfg.model, "vllm_max_num_batched_tokens", 4096
        )
        config.vllm_enable_chunked_prefill = getattr(
            cfg.model, "vllm_enable_chunked_prefill", True
        )
        config.vllm_enable_prefix_caching = getattr(
            cfg.model, "vllm_enable_prefix_caching", True
        )
        config.vllm_async_scheduling = getattr(cfg.model, "vllm_async_scheduling", True)
        config.vllm_gpu_memory_utilization = getattr(
            cfg.model, "vllm_gpu_memory_utilization", 0.92
        )
        config.vllm_safetensors_load_strategy = getattr(
            cfg.model, "vllm_safetensors_load_strategy", "prefetch"
        )
        config.vllm_block_size = getattr(cfg.model, "vllm_block_size", 128)
        config.vllm_cache_dtype = getattr(cfg.model, "vllm_cache_dtype", "auto")
        config.dspark_share_context_kv = bool(
            getattr(cfg.model, "dspark_share_context_kv", False)
        )
        config.vllm_compact_training_kv_cache = getattr(
            cfg.model,
            "vllm_compact_training_kv_cache",
            False,
        )
        config.cache_dsa_metadata_structures = bool(
            model_construction_optimizations["cache_dsa_metadata_structures"]
        )
        config.reuse_dsa_backward_host_metadata = bool(
            model_construction_optimizations["reuse_dsa_backward_host_metadata"]
        )
        config.fused_qnorm_unquantized_backward = bool(
            model_construction_optimizations["fused_qnorm_unquantized_backward"]
        )
        if config.vllm_compact_training_kv_cache and is_main_process():
            logger.warning(
                "Experimental compact DSA training cache is enabled; its "
                "smaller shape can change Ascend sparse-attention BF16 tiling "
                "and training numerics"
            )
        configured_quantization = getattr(cfg.model, "vllm_quantization", "auto")
        quantization_is_auto = (
            isinstance(configured_quantization, str)
            and configured_quantization.strip().lower() == "auto"
        )
        normalized_quantization = None
        if configured_quantization is not None and not quantization_is_auto:
            value = str(configured_quantization).strip()
            normalized_quantization = (
                None if value.lower() in {"none", "null", ""} else value
            )
        if (
            not quantization_is_auto
            and normalized_quantization != checkpoint_report.runtime_quantization
        ):
            raise ValueError(
                "model.vllm_quantization conflicts with the detected DS4 "
                "checkpoint runtime: configured="
                f"{normalized_quantization!r}, detected="
                f"{checkpoint_report.runtime_quantization!r}"
            )
        config.vllm_quantization = checkpoint_report.runtime_quantization
        configured_dequantize = getattr(
            cfg.model,
            "dequantize_mtp_for_training",
            "auto",
        )
        if isinstance(configured_dequantize, str):
            normalized_dequantize = configured_dequantize.strip().lower()
            if normalized_dequantize == "auto":
                dequantize = checkpoint_report.requires_dequantization
            elif normalized_dequantize in {"1", "true", "yes", "on"}:
                dequantize = True
            elif normalized_dequantize in {"0", "false", "no", "off"}:
                dequantize = False
            else:
                raise ValueError(
                    "model.dequantize_mtp_for_training must be auto or a "
                    f"boolean, got {configured_dequantize!r}"
                )
        else:
            dequantize = bool(configured_dequantize)
        if dequantize != checkpoint_report.requires_dequantization:
            raise ValueError(
                "model.dequantize_mtp_for_training conflicts with "
                f"checkpoint_format={checkpoint_report.resolved_format!r}: "
                f"configured={dequantize}, required="
                f"{checkpoint_report.requires_dequantization}"
            )
        config.dequantize_mtp_for_training = dequantize
        config.mtp_checkpoint_format = checkpoint_report.resolved_format
        config.mtp_checkpoint_format_report = checkpoint_report.to_dict()
        config.vllm_dtype = getattr(cfg.model, "vllm_dtype", "auto")
        config.vllm_distributed_init_method = getattr(
            cfg.model, "vllm_distributed_init_method", None
        )
        config.vllm_set_ascend_env_defaults = getattr(
            cfg.model, "vllm_set_ascend_env_defaults", True
        )
        config.vllm_enable_expert_parallel = getattr(
            cfg.model,
            "vllm_enable_expert_parallel",
            False,
        )
        if config.vllm_enable_expert_parallel:
            raise ValueError(
                "DS4 MTP training does not support vLLM expert parallelism or "
                "EPLB; set model.vllm_enable_expert_parallel=false. TP-sharded "
                "MoE training remains supported."
            )
        distributed_world_size = dist.get_world_size() if dist.is_initialized() else 1
        if distributed_world_size % config.vllm_tensor_parallel_size != 0:
            raise ValueError(
                f"DeepSpeed world size {distributed_world_size} must be divisible by "
                f"vllm_tensor_parallel_size={config.vllm_tensor_parallel_size}"
            )
        default_vllm_dp_size = (
            distributed_world_size // config.vllm_tensor_parallel_size
        )
        configured_vllm_dp_size = getattr(cfg.model, "vllm_data_parallel_size", None)
        config.vllm_data_parallel_size = (
            default_vllm_dp_size
            if configured_vllm_dp_size in (None, "auto")
            else int(configured_vllm_dp_size)
        )
        if (
            config.vllm_tensor_parallel_size * config.vllm_data_parallel_size
            != distributed_world_size
        ):
            raise ValueError(
                "Configured vLLM TP*DP topology does not match the DeepSpeed world: "
                f"tp={config.vllm_tensor_parallel_size}, "
                f"dp={config.vllm_data_parallel_size}, "
                f"world={distributed_world_size}"
            )
        if (
            config.ds4_layout == "vllm_ascend"
            and config.vllm_data_parallel_size > 1
            and not bool(
                getattr(
                    cfg.model,
                    "vllm_training_dp_replicate_moe",
                    True,
                )
            )
        ):
            raise ValueError(
                "DS4 TP+DP training requires "
                "model.vllm_training_dp_replicate_moe=true so vLLM MoE "
                "weights remain TP-sharded and are replicated across DP"
            )
        config.vllm_training_dp_replicate_moe = bool(
            getattr(cfg.model, "vllm_training_dp_replicate_moe", True)
        )
        # Keep expert partitioning aligned with model parallelism rather than
        # accidentally treating every DeepSpeed process as an expert shard.
        config.ep_size = config.vllm_tensor_parallel_size
        config.vllm_additional_config = getattr(
            cfg.model, "vllm_additional_config", None
        )
        if config.vllm_tensor_parallel_size > 1:
            additional_config = config.vllm_additional_config or {}
            if bool(additional_config.get("enable_flashcomm1", False)):
                raise ValueError(
                    "DS4 TP training requires "
                    "model.vllm_additional_config.enable_flashcomm1=false; "
                    "the token-sharded FlashComm path is not autograd-enabled"
                )
        config.vllm_compilation_config = getattr(
            cfg.model, "vllm_compilation_config", None
        )
        config.vllm_model_loader_extra_config = getattr(
            cfg.model, "vllm_model_loader_extra_config", None
        )

    draft_model = model_architecture(config=config)
    if model_type == "ds4_dspark":
        for attribute, expected in model_construction_optimizations.items():
            actual = bool(getattr(draft_model, attribute, False))
            if actual != expected:
                raise RuntimeError(
                    "DS4 model-construction optimization mismatch: "
                    f"{attribute} resolved={expected}, model={actual}"
                )
        if is_main_process():
            logger.info(
                "DS4 model-construction optimizations: %s",
                model_construction_optimizations,
            )
    draft_model.train()
    return draft_model.to(torch.bfloat16)
