# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""MTP configuration and identity kept independent of stable DSpark artifacts."""

import hashlib
from pathlib import Path

from omegaconf import OmegaConf

from verl_speco.backends.training_backend import resolve_training_backend
from verl_speco.integration.vllm_ascend_implementation import implementation_fingerprint
from verl_speco.trainer.deepseek_v4_mtp_contract import DeepSeekV4MTPContract


def mtp_implementation_fingerprint():
    root = Path(__file__).parents[1]
    digest = hashlib.sha256(implementation_fingerprint().encode())
    for name in (
        "models/deepseek_v4/modeling_mtp.py",
        "backends/vllm_ascend_mtp_model.py",
        "ops/deepseek_v4_mtp_loss.py",
        "trainer/deepseek_v4_mtp_training.py",
        "trainer/deepseek_v4_mtp_contract.py",
        "trainer/deepseek_v4_mtp_dataset.py",
        "models/deepseek_v4/mtp_artifact.py",
        "config/deepseek_v4_mtp_backend.yaml",
        "config/deepseek_v4_mtp_sources.json",
        "integration/vllm_ascend_mtp_config.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def load_mtp_config(config):
    training, settings = (
        config.actor_rollout_ref.rollout.drafter.training,
        config.draft_engine,
    )
    if (
        resolve_training_backend(config) != "vllm_ascend"
        or settings.variant != "ds4_mtp"
    ):
        raise ValueError("MTP requires backend=vllm_ascend and variant=ds4_mtp")
    if (
        str(config.actor_rollout_ref.rollout.drafter.speculative_algorithm).upper()
        != "MTP"
    ):
        raise ValueError("MTP configuration requires speculative_algorithm=MTP")
    if training.mode != "offline" or training.feature_store.type != "torch_shard":
        raise ValueError("MTP currently uses offline torch_shard input")
    model = Path(config.actor_rollout_ref.model.path).expanduser().resolve(strict=True)
    DeepSeekV4MTPContract.from_model(model)
    root = Path(__file__).parents[1] / "config"
    selected = settings.get("config_path")
    overlay = (
        root / "deepseek_v4_mtp_backend.yaml"
        if selected is None
        else Path(selected).expanduser().resolve(strict=True)
    )
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "deepseek_v4_dspark_backend.yaml"),
        OmegaConf.load(overlay),
        settings.get("overrides", {}),
    )
    cfg.model.path, cfg.model.resume_ckpt = str(model), None
    cfg.train.hidden_states_dir = cfg.train.prompts = None
    cfg.validation.hidden_states_dir = cfg.validation.prompts = None
    if cfg.model.type != "ds4" or cfg.model.ds4_layout != "vllm_ascend":
        raise ValueError("MTP model must select ds4/vllm_ascend")
    if cfg.model.get("ds4_mtp_impl") != "vllm_import_train":
        raise ValueError("MTP requires ds4_mtp_impl=vllm_import_train")
    if (
        cfg.model.checkpoint_format != "bf16"
        or cfg.model.vllm_quantization is not None
        or cfg.model.dequantize_mtp_for_training
    ):
        raise ValueError(
            "MTP integration requires direct BF16 weights, without dequantization"
        )
    if cfg.train.loss_type != "eagle" or not cfg.optimizations.mtp.vocab_parallel_kl:
        raise ValueError("MTP initial integration supports Eagle vocab-parallel KL")
    if cfg.optimizations.mtp.prune_grad_tree:
        raise ValueError("MTP initial integration uses one backward per microbatch")
    if (
        cfg.model.vllm_enable_expert_parallel
        or int(cfg.deepspeed.zero_optimization.stage) != 1
        or cfg.deepspeed.zero_optimization.overlap_comm
    ):
        raise ValueError(
            "MTP custom backward requires non-EP ZeRO-1 without overlap_comm"
        )
    if not cfg.deepspeed.bf16.enabled or cfg.deepspeed.get("fp16", {}).get(
        "enabled", False
    ):
        raise ValueError("MTP training requires BF16, without FP16 loss scaling")
    if cfg.train.get("load_weights", "full") != "full" or cfg.train.num_workers != 0:
        raise ValueError("MTP requires full weight loading and num_workers=0")
    for value in (
        cfg.train.ttt_length,
        cfg.train.epochs,
        training.max_steps,
        training.save_interval_steps,
        cfg.deepspeed.train_micro_batch_size_per_gpu,
        cfg.deepspeed.gradient_accumulation_steps,
    ):
        if int(value) <= 0:
            raise ValueError(
                "MTP steps, epochs, batch sizes and save interval must be positive"
            )
    fingerprint = mtp_implementation_fingerprint()
    selected_identity = settings.get("implementation_fingerprint") or settings.get(
        "source_fingerprint"
    )
    if selected_identity and selected_identity != fingerprint:
        raise ValueError("MTP implementation fingerprint differs")
    return cfg, fingerprint
