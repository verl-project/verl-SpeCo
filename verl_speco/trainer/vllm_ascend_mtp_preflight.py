# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""MTP-specific checks using the shared SpeCo feature store and DP loader."""

import os

from verl_speco.backends.vllm_ascend_runtime import runtime_versions
from verl_speco.integration.vllm_ascend_config import (
    execution_environment,
    training_fingerprint,
)
from verl_speco.integration.vllm_ascend_mtp_config import load_mtp_config
from verl_speco.trainer.deepseek_v4_dataset import build_loader
from verl_speco.trainer.deepseek_v4_mtp_contract import DeepSeekV4MTPContract
from verl_speco.trainer.deepseek_v4_mtp_dataset import MTPStoreDataset
from verl_speco.trainer.deepseek_v4_mtp_training import valid_token_counts


def preflight_mtp(config):
    cfg, source = load_mtp_config(config)
    contract = DeepSeekV4MTPContract.from_model(cfg.model.path)
    dataset = MTPStoreDataset(
        config.actor_rollout_ref.rollout.drafter.training.feature_store.path,
        contract,
        cfg,
    )
    try:
        metadata = dataset.store.get_metadata()
        if metadata.get("source") not in {"trajectory_replay", "mtp_capture_import"}:
            raise ValueError(
                "MTP offline features must declare trajectory_replay or mtp_capture_import provenance"
            )
        world = int(config.speco.draft_training.nproc_per_node) * int(
            config.speco.draft_training.nnodes
        )
        tp = int(cfg.model.vllm_tensor_parallel_size)
        if tp <= 0 or world <= 0 or world % tp:
            raise ValueError("MTP world size must be divisible by TP")
        if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) != world:
            raise ValueError("MTP WORLD_SIZE differs from configured topology")
        dp = world // tp
        if cfg.model.vllm_data_parallel_size not in (None, "auto", dp):
            raise ValueError("MTP configured DP differs from world/TP")
        if not cfg.model.vllm_training_dp_replicate_moe:
            raise ValueError("MTP training requires replicated MoE across DP")
        # Drop windows that cannot supervise even the first recursive step.
        # Keep segmentation itself shared, so overlap ownership is unchanged.
        if any(not any(dataset[i]["loss_mask"][1:]) for i in range(len(dataset))):
            raise ValueError(
                "MTP segment has no supervised future target; adjust data/window"
            )
        lengths = []
        for rank in range(dp):
            _, loader = build_loader(dataset, cfg, dp_rank=rank, dp_size=dp)
            batch = next(iter(loader))
            if not any(valid_token_counts(batch, int(cfg.train.ttt_length))):
                raise ValueError("MTP batch has no supervised future targets")
            lengths.append(len(loader))
        if len(set(lengths)) != 1:
            raise ValueError("MTP DP ranks have unequal batch counts")
        pad = dataset.segment_length if cfg.train.pad_segment_tails else 0
        required = int(cfg.deepspeed.train_micro_batch_size_per_gpu) * max(
            max(dataset.sequence_lengths), int(pad or 0)
        )
        if int(cfg.model.vllm_max_num_batched_tokens) < required:
            raise ValueError(
                f"MTP vllm_max_num_batched_tokens must cover {required} tokens"
            )
        report = {
            "status": "passed",
            "stage": "preflight",
            "algorithm": "MTP",
            "target_revision": config.draft_engine.target_revision,
            "contract": contract.identity(),
            "mask_semantics": "speco_1_supervised_0_ignored",
            "target_runtime_version": None,
            "source_fingerprint": source,
            "training_fingerprint": training_fingerprint(cfg),
            "world_size": world,
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "source_samples": len(dataset.keys),
            "segments": len(dataset),
            "batches_per_dp_rank": lengths[0],
            "runtime_versions": runtime_versions(
                require=bool(config.draft_engine.require_runtime)
                or config.draft_engine.action == "train"
            ),
            "execution_environment": execution_environment(config),
            "hardware_training_validated": False,
        }
        return report, cfg, dataset
    except BaseException:
        dataset.store.close()
        raise
