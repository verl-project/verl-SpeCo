# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare uninterrupted training with full-checkpoint recovery in fresh workers.

Use the standalone trainer Hydra configuration. Without resume_from, save step 1
and continue to step 2; with resume_from, restore step 1 and continue to step 2.
Every rank records complete persistent-state checksums, not sampled values.
"""

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def tensor_chunks(value, chunk_bytes=8 * 1024**2):
    """Copy bounded logical slices, including noncontiguous parameter views."""
    if chunk_bytes < value.element_size():
        raise ValueError("Chunk budget is smaller than one tensor element")
    limit = chunk_bytes // value.element_size()
    if value.numel() <= limit:
        yield value.detach().cpu().contiguous()
    elif value.is_contiguous():
        flat = value.detach().view(-1)
        for start in range(0, flat.numel(), limit):
            yield flat.narrow(0, start, min(limit, flat.numel() - start)).cpu()
    else:
        width = math.prod(value.shape[1:])
        rows = max(1, limit // width)
        for part in value.split(rows, dim=0):
            if part.shape[0] == 1:
                part = part.squeeze(0)
            yield from tensor_chunks(part, chunk_bytes)


def persistent_state(value, *, chunk_bytes=8 * 1024**2):
    """Deterministic JSON tree with exact tensor bytes; never pickle addresses."""
    if torch.is_tensor(value):
        result = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.is_meta:
            return {**result, "meta": True}
        digest = hashlib.sha256()
        for part in tensor_chunks(value, chunk_bytes):
            if part.is_floating_point() and not bool(torch.isfinite(part).all()):
                raise ValueError("Non-finite persistent state")
            raw = part.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(raw)
        return {**result, "sha256": digest.hexdigest()}
    if isinstance(value, np.ndarray):
        return {
            "numpy_dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(key): persistent_state(item, chunk_bytes=chunk_bytes)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [persistent_state(item, chunk_bytes=chunk_bytes) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            "class": f"{type(value).__module__}.{type(value).__name__}",
            "state": persistent_state(vars(value), chunk_bytes=chunk_bytes),
        }
    raise TypeError(f"Unsupported persistent-state type: {type(value)}")


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            location = f"{path}/{key}"
            if key not in left or key not in right:
                result.append(location)
            else:
                result.extend(differences(left[key], right[key], location))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path + "/length"]
        return [
            p
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for p in differences(a, b, f"{path}/{index}")
        ]
    return [] if left == right else [path]


def engine_state(engine, store):
    from verl_speco.trainer.vllm_ascend_checkpoint import rank_rng_state

    model = engine.model_engine.module
    return persistent_state(
        {
            "model": model.state_dict(),
            "router_caches": {
                name: layer.weight_fp32
                for name, layer in model.named_modules()
                if torch.is_tensor(getattr(layer, "weight_fp32", None))
            },
            "teacher": dict(engine.teacher.named_parameters()),
            "optimizer": engine.model_engine.optimizer.state_dict(),
            "scheduler": (
                engine.model_engine.lr_scheduler.state_dict()
                if engine.model_engine.lr_scheduler is not None
                else None
            ),
            "progress": asdict(engine.progress),
            "engine_global_steps": engine.model_engine.global_steps,
            "store": store.checkpoint_state(),
            "rng": rank_rng_state(),
        }
    )


class ReplicaGradientAudit:
    """Observe production backward without changing gradients or optimizer state."""

    def __init__(self, model):
        self.model = model
        self.parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.endswith(
                (".hc_attn_fn", ".self_attn.wq_a.weight", ".mlp.gate.weight")
            )
        }
        if not self.parameters:
            raise ValueError("No replicated gradient audit parameters found")
        self.initial = persistent_state(self.parameters)
        self.gradients = {}
        self.calls = {}
        self.inputs = []
        self.original_prepare = model.prepare_training_inputs
        self.had_instance_prepare = "prepare_training_inputs" in vars(model)
        self.handles = [
            parameter.register_hook(self._hook(name))
            for name, parameter in self.parameters.items()
        ]

        def prepare(*args, **kwargs):
            targets, runtime_inputs = self.original_prepare(*args, **kwargs)
            self.inputs.append(
                persistent_state(
                    {
                        "arguments": kwargs,
                        "targets": targets,
                        "runtime": runtime_inputs,
                    }
                )
            )
            return targets, runtime_inputs

        model.prepare_training_inputs = prepare

    def _hook(self, name):
        def observe(gradient):
            value = gradient.detach().to(device="cpu", dtype=torch.float32)
            if name in self.gradients:
                self.gradients[name].add_(value)
            else:
                self.gradients[name] = value.clone()
            self.calls[name] = self.calls.get(name, 0) + 1
            return gradient

        return observe

    def report(self):
        return {
            "initial_parameters": self.initial,
            "inputs": self.inputs,
            "local_gradient_sum": persistent_state(self.gradients),
            "hook_calls": self.calls,
            "updated_parameters": persistent_state(self.parameters),
            "scope": "local_autograd_contributions_before_deepspeed_reduction",
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.had_instance_prepare:
            self.model.prepare_training_inputs = self.original_prepare
        else:
            del self.model.prepare_training_inputs


def run(config):
    import torch.distributed as dist

    from verl_speco.backends.vllm_ascend_parallel import DeepSpeedParallelTopology
    from verl_speco.backends.vllm_ascend_runtime import (
        build_engine,
        initialize_distributed,
    )
    from verl_speco.trainer.deepseek_v4_dataset import build_loader
    from verl_speco.trainer.vllm_ascend_checkpoint import (
        restore_checkpoint,
        save_checkpoint,
        write_checkpoint_metadata,
    )
    from verl_speco.trainer.vllm_ascend_training_loop import (
        SpeCoTracker,
        _atomic_report,
        preflight,
    )

    for name, value in config.draft_engine.environment.items():
        os.environ.setdefault(name, str(value))
    output = Path(config.draft_engine.output_dir).resolve()
    rank = int(os.environ.get("RANK", "0"))
    dataset = tracker = None
    owned = False
    phase = "preflight"
    try:
        report, cfg, dataset = preflight(config)
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    "Recovery probe requires a fresh output directory"
                )
            output.mkdir(parents=True, exist_ok=True)
            _atomic_report(output / "preflight.json", report)
            OmegaConf.save(config, output / "invocation.yaml")
            OmegaConf.save(cfg, output / "resolved_backend.yaml")
        device, owned = initialize_distributed()
        control = dist.new_group(backend="gloo")
        dist.barrier(group=control)
        topology = DeepSpeedParallelTopology(report["tensor_parallel_size"])
        sampler, loader = build_loader(
            dataset,
            cfg,
            dp_rank=topology.data_parallel_rank,
            dp_size=topology.data_parallel_size,
        )
        tracker = SpeCoTracker(config, rank)
        engine = build_engine(
            cfg=cfg,
            topology=topology,
            loader=loader,
            dataset=dataset,
            device=device,
            tracker=tracker,
            logger=logger,
            checkpoint_dir=output / "checkpoints",
        )
        if config.draft_engine.get("recovery_probe_gradients_only", False):
            if config.draft_engine.resume_from:
                raise ValueError("Gradient audit requires fresh initialization")
            phase = "audit_training_step"
            audit = ReplicaGradientAudit(engine.model_engine.module)
            try:
                engine.train_cycle(loader, sampler, max_steps=1)
                audit_report = audit.report()
                _atomic_report(output / f"gradients_rank_{rank}.json", audit_report)
                if (
                    topology.tensor_parallel_rank in {0, 1}
                    and topology.data_parallel_rank == 0
                ):
                    torch.save(audit.gradients, output / f"gradients_rank_{rank}.pt")
                reports = [None] * dist.get_world_size()
                dist.all_gather_object(reports, audit_report, group=control)
                if rank == 0:
                    comparisons = []
                    for dp in range(topology.data_parallel_size):
                        start = dp * topology.tensor_parallel_size
                        for other in range(
                            start + 1, start + topology.tensor_parallel_size
                        ):
                            comparisons.append(
                                {
                                    "ranks": [start, other],
                                    "different_paths": differences(
                                        reports[start], reports[other]
                                    ),
                                }
                            )
                    _atomic_report(
                        output / "gradient_audit_report.json",
                        {
                            "status": "diagnostic_completed",
                            "comparisons": comparisons,
                            "optimizer_steps": engine.progress.global_step,
                        },
                    )
            finally:
                audit.close()
            return
        identity = {
            name: report[name]
            for name in [
                "contract",
                "source_fingerprint",
                "training_fingerprint",
                "runtime_versions",
                "execution_environment",
            ]
        }
        if report.get("target_runtime_version") is not None:
            identity["target_runtime_version"] = report["target_runtime_version"]
        resume = config.draft_engine.resume_from
        if resume:
            phase = "restore"
            restore_checkpoint(
                engine, path=resume, identity=identity, store=dataset.store
            )
            if engine.progress.global_step != 1:
                raise ValueError("Expected a complete optimizer-step-1 checkpoint")
        else:
            phase = "train_step_1"
            engine.train_cycle(loader, sampler, max_steps=1)
            if not config.draft_engine.get("recovery_probe_skip_save", False):
                phase = "save_step_1"
                checkpoint = save_checkpoint(
                    engine,
                    base_dir=output / "checkpoints",
                    tag="step_00000001",
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
                dist.barrier(group=control)
            else:
                existing = config.draft_engine.get(
                    "recovery_probe_reference_checkpoint"
                )
                if not existing:
                    raise ValueError(
                        "Skipping a new save requires an existing paired checkpoint"
                    )
                metadata = json.loads((Path(existing) / "speco_ds4.json").read_text())
                if (
                    metadata["identity"] != identity
                    or metadata["progress"]["global_step"] != 1
                ):
                    raise ValueError("Existing step-1 checkpoint identity differs")
        phase = "fingerprint_step_1"
        state1 = engine_state(engine, dataset.store)
        _atomic_report(output / f"state_step1_rank_{rank}.json", state1)
        reference = None
        if resume:
            reference = Path(
                config.draft_engine.get("recovery_probe_reference")
                or Path(resume).parent.parent
            )
            expected1 = json.loads(
                (reference / f"state_step1_rank_{rank}.json").read_text()
            )
            changed1 = differences(expected1, state1)
            _atomic_report(
                output / f"restore_step1_rank_{rank}.json",
                {"rank": rank, "different_paths": changed1},
            )
            all_changes = [None] * dist.get_world_size()
            dist.all_gather_object(all_changes, changed1, group=control)
            if any(all_changes):
                raise RuntimeError(
                    "Restored step-1 state differs from the paired uninterrupted state"
                )
        del state1
        phase = "train_step_2"
        metrics = engine.train_cycle(loader, sampler, max_steps=1)
        if engine.progress.global_step != 2:
            raise RuntimeError("Second optimizer step did not complete")
        phase = "fingerprint"
        state = engine_state(engine, dataset.store)
        _atomic_report(output / f"state_rank_{rank}.json", state)
        changed = []
        if resume:
            expected = json.loads((reference / f"state_rank_{rank}.json").read_text())
            changed = differences(expected, state)
        result = {
            "rank": rank,
            "status": "passed" if not changed else "different",
            "identity": identity,
            "progress": asdict(engine.progress),
            "resumed": bool(resume),
            "metrics": metrics,
            "different_paths": changed,
            "comparison": "all_persistent_tensor_bytes",
        }
        _atomic_report(output / f"rank_{rank}.json", result)
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, result, group=control)
        if rank == 0:
            _atomic_report(
                output / "recovery_report.json",
                {
                    "status": "passed"
                    if all(x["status"] == "passed" for x in results)
                    else "different",
                    "resumed": bool(resume),
                    "workers": results,
                },
            )
        if any(row["status"] != "passed" for row in results):
            raise RuntimeError(
                "Recovered state differs from uninterrupted reference; inspect per-rank reports"
            )
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_report(
            output / f"failure_rank_{rank}.json", {"phase": phase, "error": repr(error)}
        )
        raise
    finally:
        if tracker is not None:
            tracker.close()
        if dataset is not None:
            dataset.store.close()
        if owned:
            dist.destroy_process_group()


@hydra.main(
    config_path="config",
    config_name="deepseek_v4_0731_draft_trainer",
    version_base=None,
)
def main(config):
    logging.basicConfig(level=logging.INFO)
    run(config)


if __name__ == "__main__":
    main()
