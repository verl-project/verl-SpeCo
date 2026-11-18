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
"""Resolve an packaged backend configuration without native PPO dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from verl_speco.backends.training_backend import resolve_training_backend
from verl_speco.integration.ascend_state_config import DETERMINISTIC_TRAINING_ENV
from verl_speco.integration.vllm_ascend_implementation import (
    compatible_implementation_fingerprint,
    implementation_fingerprint,
)


def load_vllm_ascend_config(config):
    settings = config.draft_engine
    if settings.variant == "ds4_mtp":
        from verl_speco.integration.vllm_ascend_mtp_config import load_mtp_config

        return load_mtp_config(config)
    if (
        resolve_training_backend(config) != "vllm_ascend"
        or settings.variant != "ds4_dspark"
        or str(settings.target_revision) != "0731"
    ):
        raise ValueError("This entrypoint supports only DeepSeek V4 0731 DSpark")
    training = config.actor_rollout_ref.rollout.drafter.training
    if training.mode != "offline":
        raise ValueError(
            "Online execution is not yet validated by this opt-in entrypoint"
        )
    if str(training.feature_store.type) != "torch_shard":
        raise ValueError(
            "Offline backend execution consumes the existing torch_shard feature store"
        )
    fingerprint = implementation_fingerprint()
    selected = settings.get("implementation_fingerprint") or settings.get(
        "source_fingerprint"
    )
    if selected and not compatible_implementation_fingerprint(selected):
        raise ValueError(
            "Selected implementation differs from the packaged DeepSeek V4 backend"
        )
    selected_path = settings.get("config_path")
    # Keep the old default spelling as a configuration alias, not an external
    # checkout lookup. Users may still explicitly supply their own numeric YAML.
    if selected_path in (None, "configs/ds4_dspark_train_smoke.yaml"):
        path = Path(__file__).parents[1] / "config/deepseek_v4_dspark_backend.yaml"
    else:
        path = Path(selected_path).expanduser().resolve(strict=True)
    cfg = OmegaConf.merge(OmegaConf.load(path), settings.get("overrides", {}))
    # Bind paths to the SpeCo invocation, never to AIR's checked-in machine paths.
    cfg.model.path = str(
        Path(config.actor_rollout_ref.model.path).expanduser().resolve(strict=True)
    )
    cfg.model.resume_ckpt = None
    cfg.train.num_workers = 0
    cfg.train.hidden_states_dir = None
    cfg.train.prompts = None
    cfg.validation.hidden_states_dir = None
    cfg.validation.prompts = None
    if cfg.model.type != "ds4_dspark" or cfg.model.ds4_layout != "vllm_ascend":
        raise ValueError(
            "Backend model configuration must select ds4_dspark/vllm_ascend"
        )
    if cfg.model.get("vllm_enable_expert_parallel", False):
        raise ValueError("DS4 backward requires TP-sharded non-EP training")
    zero = cfg.deepspeed.zero_optimization
    if int(zero.stage) != 1 or zero.get("overlap_comm", False):
        raise ValueError("DS4 custom backward requires ZeRO-1 with overlap_comm=false")
    for name in ("train_micro_batch_size_per_gpu", "gradient_accumulation_steps"):
        if int(cfg.deepspeed[name]) <= 0:
            raise ValueError(f"deepspeed.{name} must be positive")
    if int(cfg.train.epochs) <= 0 or int(training.max_steps) <= 0:
        raise ValueError("epochs and SpeCo max_steps must be positive")
    if int(training.save_interval_steps) <= 0:
        raise ValueError("save_interval_steps must be positive")
    if not cfg.deepspeed.get("bf16", {}).get("enabled", False) or cfg.deepspeed.get(
        "fp16", {}
    ).get("enabled", False):
        raise ValueError("The DS4 0731 adapter requires BF16 without FP16 loss scaling")
    if cfg.train.get("load_weights", "full") != "full":
        raise ValueError(
            "The DS4 0731 adapter requires full initial checkpoint loading"
        )
    return cfg, fingerprint


def training_fingerprint(cfg) -> str:
    """Freeze numerical settings independently of paths and a run's step cap."""
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("DSpark training configuration must be a mapping")
    values.pop("paths", None)
    values.pop("logging", None)
    for key in ("path", "resume_ckpt", "initial_draft_artifact"):
        values["model"].pop(key, None)
    # Native max_optimizer_steps is a per-cycle limit, assigned by the adapter.
    values["train"].pop("max_optimizer_steps", None)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def execution_environment(config) -> dict[str, str]:
    """Record numerical overrides that the backend resolves outside its YAML config."""
    names = set(config.draft_engine.environment)
    names.update(DETERMINISTIC_TRAINING_ENV)
    names.update(name for name in os.environ if name.startswith(("MTP_", "DS4_")))
    return {
        name: os.environ.get(name, str(config.draft_engine.environment.get(name, "")))
        for name in sorted(names)
    }
