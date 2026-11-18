# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Build the native MTP wrapper using the existing Ascend configuration rules."""

import os

import torch
import torch.distributed as dist

from verl_speco.backends.vllm_ascend_options import (
    export_ds4_model_construction_environment,
    resolve_optimization_config,
)
from verl_speco.models.deepseek_v4.checkpoint import inspect_ds4_checkpoint_format
from verl_speco.trainer.deepseek_v4_mtp_contract import DeepSeekV4MTPContract


def load_mtp_model(cfg, logger):
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

    from verl_speco.models.deepseek_v4.modeling_mtp import DeepseekV4MTP

    DeepSeekV4MTPContract.from_model(cfg.model.path)
    report = inspect_ds4_checkpoint_format(cfg.model.path, "bf16")
    if report.requires_dequantization:
        raise ValueError("MTP requires direct BF16 checkpoint weights")
    config = DeepseekV4Config.from_pretrained(cfg.model.path)
    config._attn_implementation = "sdpa"
    # Runtime options retain their existing names and interpretation. No
    # alternative attention, HC or MoE formula is introduced by this factory.
    for name, value in cfg.model.items():
        if name.startswith("vllm_") or name in {
            "ds4_layout",
            "ds4_mtp_impl",
            "use_ascend_hc_ops",
            "dspark_share_context_kv",
            "dequantize_mtp_for_training",
        }:
            setattr(config, name, value)
    config.vllm_model_path = cfg.model.path
    config.vllm_quantization = report.runtime_quantization
    config.mtp_checkpoint_format = report.resolved_format
    config.mtp_checkpoint_format_report = report.to_dict()
    world = dist.get_world_size() if dist.is_initialized() else 1
    tp = int(cfg.model.vllm_tensor_parallel_size)
    if tp <= 0 or world % tp:
        raise ValueError("MTP world size must be divisible by TP")
    configured_dp = cfg.model.vllm_data_parallel_size
    if configured_dp not in (None, "auto") and int(configured_dp) != world // tp:
        raise ValueError("MTP configured DP differs from launch world/TP")
    config.vllm_data_parallel_size = world // tp
    config.ep_size = tp
    if config.vllm_enable_expert_parallel or not config.vllm_training_dp_replicate_moe:
        raise ValueError("MTP uses TP-sharded experts replicated over training DP")
    if config.vllm_additional_config.get("enable_flashcomm1", False):
        raise ValueError("MTP training does not support FlashComm1")
    options = resolve_optimization_config(cfg)
    resolved = export_ds4_model_construction_environment(cfg)
    for name, enabled in resolved.items():
        setattr(config, name, enabled)
    for name, option in {
        "MTP_DS4_GROUPED_ROUTED_MOE_REPLAY": "grouped_routed_moe_replay",
        "MTP_DS4_FP32_ROUTE_ARGSORT": "fp32_route_argsort",
        "MTP_DS4_FUSED_HC_FORWARD": "fused_hc_forward",
        "MTP_DS4_FUSED_ROUTED_MOE_TOKEN_UNPERMUTE": "fused_routed_moe_token_unpermute",
        "MTP_DS4_REUSE_DSA_BACKWARD_STRUCTURES": "reuse_dsa_backward_structures",
        "MTP_DS4_BUFFERED_DETERMINISTIC_REDUCE": "buffered_deterministic_reduce",
        "MTP_DS4_TP_SHARD_EH_PROJECTIONS": "tp_shard_eh_projections",
        "MTP_DS4_TP_SHARD_ATTENTION_PROJECTIONS": "tp_shard_attention_projections",
        "MTP_DS4_FUSED_EH_PROJECTION_COLLECTIVE": "fused_eh_projection_collective",
    }.items():
        os.environ.setdefault(
            name, "1" if bool(getattr(options, option, False)) else "0"
        )
    model = DeepseekV4MTP(config)
    logger.info(
        "Constructed native MTP model: format=%s TP=%s DP=%s",
        report.resolved_format,
        tp,
        world // tp,
    )
    return model.train().to(torch.bfloat16)
