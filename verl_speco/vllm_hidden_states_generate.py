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
"""Generate vLLM safetensors draft features from token replay samples."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf, open_dict

from verl_speco.trainer.draft_dataset import (
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import build_feature_store_from_config
from verl_speco.trainer.target_feature_replay import TargetFeatureReplayer

logger = logging.getLogger(__name__)


def _plain_config(value: Any) -> dict[str, Any]:
    return dict(OmegaConf.to_container(value, resolve=True) or {})


def generate_vllm_safetensors_features(config) -> dict[str, Any]:
    """Materialize token replay samples through vLLM and save safetensors."""

    draft_config = config.actor_rollout_ref
    training_cfg = draft_config.rollout.drafter.training
    replay_cfg = training_cfg.get("target_feature_replay", {}) or {}
    generation_cfg = replay_cfg.get("offline_generation", {}) or {}
    feature_store_cfg = training_cfg.feature_store

    input_path = generation_cfg.get("input_path", None)
    output_path = generation_cfg.get("output_path", None) or feature_store_cfg.get(
        "path", None
    )
    if not input_path:
        raise ValueError(
            "target_feature_replay.offline_generation.input_path is required"
        )
    if not output_path:
        raise ValueError(
            "target_feature_replay.offline_generation.output_path or "
            "training.feature_store.path is required"
        )

    input_type = str(generation_cfg.get("input_type", "token_replay") or "token_replay")
    input_cfg = _plain_config(feature_store_cfg)
    input_cfg.update({"type": input_type, "path": os.fspath(input_path)})

    output_cfg = _plain_config(feature_store_cfg)
    output_cfg.update({"type": "vllm_safetensors", "path": os.fspath(output_path)})

    max_samples = int(generation_cfg.get("max_samples", 0) or 0)
    batch_size = max(int(generation_cfg.get("batch_size", 1) or 1), 1)
    shuffle = bool(generation_cfg.get("shuffle", False))
    seed = int(training_cfg.get("seed", 0) or 0)

    replay_config = deepcopy(config)
    with open_dict(replay_config):
        target_feature_replay = replay_config.actor_rollout_ref.rollout.drafter.training.target_feature_replay
        target_feature_replay.backend = "vllm_file"

    input_store = build_feature_store_from_config(input_cfg, read_only=True)
    output_store = build_feature_store_from_config(
        output_cfg,
        read_only=False,
        metadata={
            "source_format": input_type,
            "source_path": os.fspath(input_path),
            "target_feature_backend": "vllm_file",
        },
    )
    replayer = TargetFeatureReplayer(
        replay_config,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    written = 0
    try:
        loader = DraftFeatureDataLoader(
            input_store,
            DraftFeatureDataLoaderConfig(
                batch_size=batch_size,
                rank=0,
                world_size=1,
                shuffle=shuffle,
                repeat=False,
                seed=seed,
            ),
        )
        for samples in loader:
            if max_samples > 0 and written >= max_samples:
                break
            if max_samples > 0:
                samples = samples[: max_samples - written]
            features = replayer.materialize(samples)
            output_store.write_many(features)
            written += len(features)
            if written % 100 == 0:
                logger.warning("Generated %s vLLM safetensors samples", written)
    finally:
        input_store.close()
        output_store.close()
        replayer.close()

    metrics = replayer.metrics()
    result = {
        "input_path": os.fspath(input_path),
        "output_path": os.fspath(output_path),
        "written_samples": written,
        "metrics": metrics,
    }
    logger.warning("vLLM safetensors generation finished: %s", result)
    return result


@hydra.main(config_path="config", config_name="draft_trainer", version_base=None)
def main(config):
    logging.basicConfig(level=logging.INFO)
    generate_vllm_safetensors_features(config)


if __name__ == "__main__":
    main()
