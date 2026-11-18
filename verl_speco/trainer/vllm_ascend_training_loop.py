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
"""vLLM-Ascend execution; currently supports offline DeepSeek V4 0731 DSpark."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from verl_speco.backends.vllm_ascend_parallel import (
    DeepSpeedParallelTopology,
    resolve_tensor_parallel_size,
)
from verl_speco.backends.vllm_ascend_runtime import (
    build_engine,
    initialize_distributed,
    runtime_versions,
)
from verl_speco.integration.vllm_ascend_config import (
    execution_environment,
    load_vllm_ascend_config,
    training_fingerprint,
)
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract
from verl_speco.trainer.deepseek_v4_dataset import StoreDataset, build_loader

logger = logging.getLogger(__name__)


class SpeCoTracker:
    """Translate DSpark metrics into the existing SpeCo tracking implementations."""

    def __init__(self, config, rank: int):
        from verl_speco.trainer.standalone_tracking import _build_standalone_tracking

        self.trackers = _build_standalone_tracking(config, rank=rank)
        self.rank = rank

    def log_metrics(self, metrics, step=None):
        from verl_speco.trainer.standalone_tracking import (
            _log_standalone_tracking_metrics,
        )

        if any(not math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("DSpark training reported non-finite metrics")
        if self.rank == 0:
            logger.info("DSpark step=%s metrics=%s", step, metrics)
            _log_standalone_tracking_metrics(
                self.trackers, metrics, step=int(step or 0)
            )

    def close(self):
        from verl_speco.trainer.standalone_tracking import _finish_standalone_tracking

        _finish_standalone_tracking(self.trackers, rank=self.rank)


def _atomic_report(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def preflight(config) -> tuple[dict, Any, StoreDataset]:
    """Check every sample and construct actual DSpark batches on CPU."""
    if config.draft_engine.variant == "ds4_mtp":
        from verl_speco.trainer.vllm_ascend_mtp_preflight import preflight_mtp

        return preflight_mtp(config)
    cfg, source = load_vllm_ascend_config(config)
    contract = DeepSeekV4Contract.from_model(cfg.model.path)
    dataset = StoreDataset(
        config.actor_rollout_ref.rollout.drafter.training.feature_store.path,
        contract,
        cfg,
    )
    store_metadata = dataset.store.get_metadata()
    if store_metadata.get("source") == "verl_rollout":
        captured_version = store_metadata.get("runtime_target_version")
        declared_version = config.draft_engine.get("target_runtime_version")
        if declared_version is None or str(declared_version) != str(captured_version):
            dataset.store.close()
            raise ValueError(
                "Rollout features require draft_engine.target_runtime_version matching "
                "the captured version, and model.path must select that target/teacher snapshot"
            )
    tp = resolve_tensor_parallel_size(int(cfg.model.vllm_tensor_parallel_size))
    cfg.model.vllm_tensor_parallel_size = tp
    world = int(config.speco.draft_training.nproc_per_node) * int(
        config.speco.draft_training.nnodes
    )
    # Ray/torchrun own the effective world size; reject conflicting declarations.
    if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) != world:
        raise ValueError(
            "Launcher WORLD_SIZE differs from speco.draft_training topology"
        )
    if tp <= 0 or world % tp:
        raise ValueError(
            f"world_size={world} must be divisible by DSpark training TP={tp}"
        )
    dp = world // tp
    configured_dp = cfg.model.get("vllm_data_parallel_size", "auto")
    if configured_dp not in (None, "auto") and int(configured_dp) != dp:
        raise ValueError("Configured Backend data parallel size differs from world_size/TP")
    if dp > 1 and not cfg.model.get("vllm_training_dp_replicate_moe", True):
        raise ValueError("DS4 TP+DP requires vllm_training_dp_replicate_moe=true")
    batches = []
    for rank in range(dp):
        _, loader = build_loader(dataset, cfg, dp_rank=rank, dp_size=dp)
        batch = next(iter(loader))
        expected = {
            "input_ids",
            "positions",
            "loss_mask",
            "raw_mask",
            "hidden_states",
            "target_head_hidden_states",
        }
        if not expected.issubset(batch):
            raise ValueError("Collate did not retain the full DS4 feature schema")
        batches.append(len(loader))
    if len(set(batches)) != 1:
        raise ValueError("DP replicas have unequal batch counts")
    padded_length = (
        dataset.segment_length if cfg.train.get("pad_segment_tails", False) else None
    )
    required_tokens = int(cfg.deepspeed.train_micro_batch_size_per_gpu) * max(
        max(dataset.sequence_lengths), int(padded_length or 0)
    )
    if int(cfg.model.vllm_max_num_batched_tokens) < required_tokens:
        raise ValueError(
            f"Backend vllm_max_num_batched_tokens must cover {required_tokens} tokens"
        )
    versions = runtime_versions(
        require=bool(config.draft_engine.require_runtime)
        or config.draft_engine.action == "train"
    )
    report = {
        "status": "passed",
        "stage": "preflight",
        "target_revision": "0731",
        "contract": contract.identity(),
        "mask_semantics": "speco_1_supervised_0_ignored",
        "target_runtime_version": store_metadata.get("runtime_target_version"),
        "source_fingerprint": source,
        "training_fingerprint": training_fingerprint(cfg),
        "world_size": world,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "source_samples": len(dataset.keys),
        "segments": len(dataset),
        "batches_per_dp_rank": batches[0],
        "runtime_versions": versions,
        "execution_environment": execution_environment(config),
        "hardware_training_validated": False,
    }
    return report, cfg, dataset


def run_vllm_ascend_training(config) -> dict:
    action = str(config.draft_engine.action)
    if action not in {"preflight", "train"}:
        raise ValueError("draft_engine.action must be preflight or train")
    output = Path(config.draft_engine.output_dir).expanduser().resolve()
    rank = int(os.environ.get("RANK", "0"))
    if action == "train":
        for name, value in config.draft_engine.environment.items():
            os.environ.setdefault(name, str(value))
    tracker = None
    owned_distributed = False
    stage = "preflight"
    dataset = None
    try:
        report, cfg, dataset = preflight(config)
        if rank == 0:
            # Do not silently replace a previous completed or partial run.
            if action == "train" and output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    "Use a fresh output_dir; resume_from points to the previous checkpoint"
                )
            output.mkdir(parents=True, exist_ok=True)
            _atomic_report(output / "preflight.json", report)
            OmegaConf.save(cfg, output / "resolved_air_config.yaml")
        if action == "preflight":
            return report
        stage = "initialize"
        device, owned_distributed = initialize_distributed()
        import torch.distributed as dist

        if dist.get_world_size() != report["world_size"]:
            raise ValueError("Initialized HCCL world size differs from preflight")
        topology = DeepSpeedParallelTopology(report["tensor_parallel_size"])
        sampler, loader = build_loader(
            dataset,
            cfg,
            dp_rank=topology.data_parallel_rank,
            dp_size=topology.data_parallel_size,
        )
        tracker = SpeCoTracker(config, rank)
        checkpoint_dir = output / "checkpoints"
        engine = build_engine(
            cfg=cfg,
            topology=topology,
            loader=loader,
            dataset=dataset,
            device=device,
            tracker=tracker,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
        )
        from verl_speco.trainer.vllm_ascend_checkpoint import (
            restore_checkpoint,
            save_checkpoint,
            write_checkpoint_metadata,
        )

        identity = {
            "contract": report["contract"],
            "source_fingerprint": report["source_fingerprint"],
            "training_fingerprint": report["training_fingerprint"],
            "runtime_versions": report["runtime_versions"],
            "execution_environment": report["execution_environment"],
        }
        if report.get("target_runtime_version") is not None:
            identity["target_runtime_version"] = report["target_runtime_version"]
        if config.draft_engine.resume_from:
            stage = "restore"
            restore_checkpoint(
                engine,
                path=config.draft_engine.resume_from,
                identity=identity,
                store=dataset.store,
            )
        training = config.actor_rollout_ref.rollout.drafter.training
        total = math.ceil(
            len(loader) / int(cfg.deepspeed.gradient_accumulation_steps)
        ) * int(cfg.train.epochs)
        target = min(int(training.max_steps), total)
        if engine.progress.global_step >= target:
            raise ValueError(
                "No updates requested: max_steps must exceed restored progress within configured epochs"
            )
        stage = "train"
        checkpoint = None
        interval = int(training.save_interval_steps)
        while engine.progress.global_step < target:
            budget = min(
                target - engine.progress.global_step,
                interval - engine.progress.global_step % interval,
            )
            engine.train_cycle(loader, sampler, max_steps=budget)
            p = engine.progress
            should_save = p.global_step % interval == 0 or (
                p.global_step == target and bool(training.save_final_checkpoint)
            )
            if should_save:
                checkpoint = save_checkpoint(
                    engine,
                    base_dir=checkpoint_dir,
                    tag=f"step_{p.global_step:08d}",
                    identity=identity,
                    store=dataset.store,
                )
                if rank == 0:
                    write_checkpoint_metadata(
                        checkpoint,
                        engine=engine,
                        identity=identity,
                        store=dataset.store,
                    )
                dist.barrier()
        result = {
            **report,
            "stage": "train",
            "status": "passed",
            "rank": rank,
            "progress": asdict(engine.progress),
            "checkpoint": checkpoint,
            "hardware_training_validated": True,
        }
        if rank == 0:
            _atomic_report(output / "training_report.json", result)
        return result
    except Exception as error:
        # Rank-specific reports must not race with other workers.
        if output.exists():
            _atomic_report(
                output / f"failure_rank_{rank}.json",
                {
                    "status": "failed",
                    "stage": stage,
                    "type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    finally:
        if dataset is not None:
            dataset.store.close()
        if tracker is not None:
            tracker.close()
        if owned_distributed:
            import torch.distributed as dist

            dist.destroy_process_group()
