# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Torchrun worker for the no-RL, two-round resident online trainer probe."""

import argparse
import asyncio
import json
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.pending")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def trainer_config(model, work_dir, tp, dp, *, state_parking="none"):
    from omegaconf import OmegaConf

    overrides = OmegaConf.create(
        {
            "model": {"path": model},
            "rollout": {
                "tensor_model_parallel_size": tp,
                "data_parallel_size": dp,
                "pipeline_model_parallel_size": 1,
                "drafter": {
                    "speculative_algorithm": "DSPARK",
                    "enable": True,
                    "enable_drafter_training": True,
                    "model_path": model,
                    "training": {
                        "backend": "vllm_ascend",
                        "mode": "online",
                        "collect_hidden_states_from_rollout": False,
                        "collect_hidden_states_from_old_logprob": True,
                        "use_data_buffer": False,
                        "batch_size_per_gpu": 1,
                        "step": 1,
                        "max_seq_len": 512,
                        "vllm_ascend": {
                            "state_parking": state_parking,
                            "work_dir": work_dir,
                            "total_optimizer_steps": 2,
                            "overrides": {
                                "model": {
                                    "vllm_tensor_parallel_size": tp,
                                    "vllm_data_parallel_size": dp,
                                    "vllm_max_num_batched_tokens": 512,
                                    "vllm_max_num_seqs": 1,
                                },
                                "train": {
                                    "segment_length": None,
                                    "max_len": None,
                                    "epochs": 2,
                                    "dspark_num_anchors": 4,
                                    "dspark_anchor_min_spacing": 8,
                                },
                                "deepspeed": {
                                    "train_micro_batch_size_per_gpu": 1,
                                    "gradient_accumulation_steps": 1,
                                },
                                "optimizations": {
                                    "dspark": {"anchor_micro_batch_size": 4},
                                    "mtp": {
                                        "tp_shard_eh_projections": False,
                                        "tp_shard_attention_projections": False,
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
    )

    # Inherit the same constructor defaults as the production SpeCo trainer.
    defaults = OmegaConf.load(Path(__file__).parents[1] / "config/speco_base.yaml")
    return OmegaConf.merge(defaults.actor_rollout_ref, overrides)


def collect_window(session, store_path, *, dp_rank, expected_revision):
    """Read one live rollout sample for this DP and retain its final 512-token window."""
    import torch

    from verl_speco.trainer.deepseek_v4_contract import to_engine_tensors
    from verl_speco.trainer.feature_store import TorchShardFeatureStore

    store = TorchShardFeatureStore(store_path, read_only=True)
    try:
        metadata = store.get_metadata()
        keys = list(store.iter_keys())
        if (
            metadata.get("source") != "verl_rollout"
            or metadata.get("runtime_target_version") != 0
        ):
            raise ValueError("Resident probe requires live target-version-0 features")
        if len(keys) != session.dp_group_world_size:
            raise ValueError("Expected one request per DP engine")
        sample = store.read(keys[dp_rank])
        if sample.metadata.get("runtime_drafter_revision") != expected_revision:
            raise ValueError("Features came from the wrong serving draft revision")
        values = to_engine_tensors(sample, session.contract)
        full_ids = values["input_ids"]
        full_mask = values["loss_mask"]
        start = max(0, sample.input_ids.numel() - 512)
        values = {key: value[start:] for key, value in values.items()}
        mask = values["loss_mask"]
        supervised = mask.nonzero().reshape(-1)
        if not len(supervised):
            raise ValueError("No supervised rollout rows")
        boundary = int(supervised[0])
        if not torch.equal(
            mask, torch.cat([torch.zeros(boundary), torch.ones(len(mask) - boundary)])
        ):
            raise ValueError("Resident probe expects the SpeCo prompt/response mask")
        # hidden_positions index the original unpadded sequence and also
        # retain the absolute RoPE positions. Crop feature rows, not token IDs.
        prompt_boundary = int(full_mask.nonzero().reshape(-1)[0])
        session.collect_online_data(
            {
                "input_ids": full_ids.unsqueeze(0),
                "prompts": full_ids[:prompt_boundary].unsqueeze(0),
                "hidden_positions": values["positions"],
                "global_step": 0,
                "target_head_hidden_states": values["target_head_hidden_states"],
            },
            values["hidden_states"].unsqueeze(0),
        )
        if not torch.equal(session.collected_data[-1]["feature"].loss_mask, mask):
            raise RuntimeError("Online collection changed the SpeCo mask")
    finally:
        store.close()


def collect_actor_window(session, features_path, target_path, *, dp_rank):
    """Consume actual actor transport and paired output weights at target zero."""
    import torch

    features = torch.load(
        features_path, map_location="cpu", weights_only=True, mmap=True
    )
    target = torch.load(target_path, map_location="cpu", weights_only=True, mmap=True)
    expected_scope = "full_target_real_verl_fsdp2_oldlogprob_synthetic_short_inputs"
    if (
        features.get("contract") != session.contract.identity()
        or target.get("contract") != session.contract.identity()
        or features.get("scope") != expected_scope
        or target.get("scope") != expected_scope
        or features.get("target_version") != 0
        or target.get("global_step") != 0
        or session.current_rl_step != 0
    ):
        raise ValueError(
            "Resident actor transport requires matching full-target version-zero payloads"
        )
    samples = features.get("samples", [])
    if len(samples) != session.dp_group_world_size or not 0 <= dp_rank < len(samples):
        raise ValueError("Expected one actor sample per training DP rank")
    sample = samples[dp_rank]
    session.sync_target_output(target, 0)
    session.collect_online_data(sample["batch"], sample["hidden_states"])
    return {"source": "actor_oldlogprob", "target_version": 0, "synthetic_tokens": True}


def collect_probe_window(session, command, *, dp_rank, expected_revision):
    """Accept paired actor exports or the existing controller's recorded store."""
    actor_keys = {"actor_features", "actor_target_output"}
    provided = actor_keys.intersection(command)
    if provided:
        if provided != actor_keys or "store" in command:
            raise ValueError(
                "Provide paired actor payloads or a store, without mixing sources"
            )
        return collect_actor_window(
            session,
            command["actor_features"],
            command["actor_target_output"],
            dp_rank=dp_rank,
        )
    if not command.get("store"):
        raise ValueError("Resident probe command has no feature source")
    collect_window(
        session,
        command["store"],
        dp_rank=dp_rank,
        expected_revision=expected_revision,
    )
    return {"source": "recorded_rollout_store", "target_version": 0}


def wait_for_rollout_release(
    needed,
    *,
    free_bytes,
    gather,
    timeout=120.0,
    interval=0.5,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Wait for asynchronous serving teardown before the strict lifecycle wake.

    All ranks settle every observation on the CPU control group. A locally
    ready rank cannot wake while another rank is still waiting or has failed.
    This probe barrier does not relax the production capacity check.
    """
    started = clock()
    observations = 0
    while True:
        elapsed = clock() - started
        sample = {
            "needed": needed,
            "free": None,
            "error": None,
            "expired": elapsed >= timeout,
        }
        try:
            sample["free"] = int(free_bytes())
        except Exception as error:  # noqa: BLE001 -- settle errors on every rank
            sample["error"] = repr(error)
        ranks = gather(sample)
        observations += 1
        if any(row["error"] for row in ranks):
            raise RuntimeError(f"Rollout release observation failed: {ranks}")
        if all(row["free"] >= row["needed"] for row in ranks):
            return {
                "wait_seconds": clock() - started,
                "observations": observations,
                "ranks": ranks,
            }
        if any(row["expired"] for row in ranks):
            raise TimeoutError(f"Rollout device release timed out: {ranks}")
        sleep(interval)


def wake_parked_for_shutdown(parking, engine, *, free_bytes, **wait_options):
    """Map storage for destruction only after this device has been released.

    Cleanup must not require peers to enter a collective: another rank may
    already be unwinding an exception. The same strict capacity rule applies.
    """
    if parking.state != "parked":
        return None
    evidence = wait_for_rollout_release(
        parking.coverage_inventory["managed_bytes"],
        free_bytes=free_bytes,
        gather=lambda row: [row],
        **wait_options,
    )
    parking.wake(engine)
    return evidence


def worker(args):
    import torch
    import torch.distributed as dist

    from verl_speco.backends.vllm_ascend_parallel import DeepSpeedParallelTopology
    from verl_speco.backends.vllm_ascend_runtime import initialize_distributed
    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )
    from verl_speco.trainer.ascend_state_parking import (
        AscendStateParking,
        available_host_memory,
    )
    from verl_speco.trainer.vllm_ascend_online import build_online_trainer

    install_verl_npu_vllm_import_compat()
    _device, owned = initialize_distributed()
    control = dist.new_group(backend="gloo")
    topology = DeepSpeedParallelTopology(args.tensor_parallel_size)
    root = Path(args.output)
    rank = dist.get_rank()
    session = build_online_trainer(
        trainer_config(
            args.model,
            str(root / "trainer_work"),
            args.tensor_parallel_size,
            dist.get_world_size() // args.tensor_parallel_size,
            state_parking="camem" if args.production_lifecycle else "none",
        ),
        world_size=dist.get_world_size(),
        rollout_dp_rank=topology.data_parallel_rank,
        training_device_mesh=None,
        training_process_group=topology.tensor_parallel_group,
        data_parallel_process_group=topology.data_parallel_group,
        _allow_probe_engine_dp=True,
    )
    lifecycle = session._get_state_lifecycle() if args.production_lifecycle else None
    parking = lifecycle.parking if lifecycle else AscendStateParking()
    completed = 0
    teacher_state = None
    identity = None
    try:
        if rank == 0:
            atomic_json(
                root / "trainer_ready.json",
                {"pid": os.getpid(), "world_size": dist.get_world_size()},
            )
        for sequence in range(1, 4):
            command = [None]
            if rank == 0:
                path = root / f"command_{sequence:02d}.json"
                deadline = time.monotonic() + 3600
                while not path.exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Timed out waiting for controller command")
                    time.sleep(0.1)
                command[0] = json.loads(path.read_text())
            dist.broadcast_object_list(command, src=0, group=control)
            command = command[0]
            if command["operation"] == "stop":
                break
            if command["operation"] != "train" or command["round"] != completed + 1:
                raise ValueError("Out-of-order resident training command")
            errors = [None] * dist.get_world_size()
            error = None
            try:
                validation_error = None
                try:
                    release_evidence = None
                    if parking.state == "parked":

                        def gather_capacity(sample):
                            rows = [None] * dist.get_world_size()
                            dist.all_gather_object(rows, sample, group=control)
                            return rows

                        release_evidence = wait_for_rollout_release(
                            parking.coverage_inventory["managed_bytes"],
                            free_bytes=lambda: torch.npu.mem_get_info()[0],
                            gather=gather_capacity,
                        )
                        if rank == 0:
                            atomic_json(
                                root / f"release_round_{completed + 1:02d}.json",
                                release_evidence,
                            )
                        if lifecycle:
                            asyncio.run(session.activate_training_model())
                        else:
                            parking.wake(session.ascend_engine)
                    session.current_rl_step = 0
                    feature_source = collect_probe_window(
                        session,
                        command,
                        dp_rank=topology.data_parallel_rank,
                        expected_revision=completed,
                    )
                except Exception as validation:  # noqa: BLE001 -- all ranks reject together
                    validation_error = repr(validation)
                dist.all_gather_object(errors, validation_error, group=control)
                if any(errors):
                    raise RuntimeError(
                        f"Resident input/state validation failed: {errors}"
                    )
                with nullcontext() if lifecycle else parking.allocation_scope():
                    awaitable = session.activate_training_model()
                    asyncio.run(awaitable)
                    if feature_source["source"] == "recorded_rollout_store":
                        if teacher_state is None:
                            teacher_state = {
                                name: p.detach().cpu().clone()
                                for name, p in session.ascend_engine.teacher.named_parameters()
                            }
                        session.sync_target_output(
                            {"ds4_target_output": teacher_state}, 0
                        )
                    session.reset_training_metrics()
                    if not asyncio.run(session.training_step(0)):
                        raise RuntimeError(
                            "Online trainer did not consume the new collection"
                        )
                metrics = session.get_training_metrics()
                prefix = "drafter/vllm_ascend/"
                if (
                    metrics.get(prefix + "optimizer_probe_delta", 0) <= 0
                    or metrics.get(prefix + "optimizer_lr", 0) <= 0
                ):
                    raise RuntimeError("No measured optimizer update")
                snapshot = Path(command["snapshot"])
                session.write_runtime_publish_shards(snapshot)
                if rank == 0:
                    atomic_json(
                        root / f"train_round_{completed + 1:02d}.json",
                        {
                            "stage": "optimizer_step_and_runtime_save_completed",
                            "global_step": session.optimizer_steps_total,
                            "metrics": metrics,
                            "snapshot": str(snapshot),
                            "park_validated": False,
                        },
                    )
                asyncio.run(session.cleanup_training(clear_data=True))
                if lifecycle:
                    parked_bytes = lifecycle.last_status["managed_bytes"]
                else:
                    torch.npu.synchronize()
                    # Finish HCCL before unmapping managed storage. Only CPU/Gloo control follows.
                    dist.barrier()
                    coverage_error = None
                    try:
                        allocation_bytes = parking.check_coverage(session.ascend_engine)
                    except Exception as coverage:  # noqa: BLE001 -- reject before any rank parks
                        coverage_error = repr(coverage)
                        allocation_bytes = 0
                    dist.all_gather_object(errors, coverage_error, group=control)
                    if any(errors):
                        raise RuntimeError(
                            f"Resident allocation coverage failed: {errors}"
                        )
                    allocations = [None] * dist.get_world_size()
                    dist.all_gather_object(allocations, allocation_bytes, group=control)
                    budget_error = [None]
                    if (
                        rank == 0
                        and sum(allocations) + 16 * 1024**3 > available_host_memory()
                    ):
                        budget_error[0] = (
                            f"Insufficient container memory for {sum(allocations) / 1024**3:.1f} GiB of resident state"
                        )
                    dist.broadcast_object_list(budget_error, src=0, group=control)
                    if budget_error[0]:
                        raise RuntimeError(budget_error[0])
                    parked_bytes = parking.park(session.ascend_engine)
                free_bytes, _ = torch.npu.mem_get_info()
                if free_bytes < 50 * 1024**3:
                    raise RuntimeError(
                        "Resident pool did not release 50 GiB of NPU headroom for rollout"
                    )
                evidence = parking.probe
                if (
                    evidence["global_step"] != completed + 1
                    or not evidence["optimizer_steps"]
                ):
                    raise RuntimeError("Optimizer state/progress was not retained")
                if any(step != completed + 1 for step in evidence["optimizer_steps"]):
                    raise RuntimeError(
                        "Adam optimizer steps differ from the completed round"
                    )
                current_identity = {
                    key: evidence[key]
                    for key in (
                        "engine_id",
                        "optimizer_id",
                        "inner_optimizer_id",
                        "scheduler_id",
                    )
                }
                if identity is not None and current_identity != identity:
                    raise RuntimeError(
                        "Resident engine/optimizer/scheduler was reconstructed"
                    )
                identity = current_identity
                completed += 1
                result = {
                    "rank": rank,
                    "pid": os.getpid(),
                    "round": completed,
                    "state": parking.state,
                    "state_lifecycle": "production_hooks" if lifecycle else "probe",
                    "managed_bytes": parked_bytes,
                    "allocation_inventory": parking.coverage_inventory,
                    "global_step": evidence["global_step"],
                    "optimizer_steps": evidence["optimizer_steps"],
                    "state_probe_sha256": evidence["digest"],
                    "identities": identity,
                    "metrics": metrics,
                    "feature_source": feature_source,
                    "wake_verified": completed > 1,
                    "rollout_release": release_evidence,
                    "deferred_cache_release_calls": parking.deferred_cache_release_calls,
                    "deferred_cache_release_origin": parking.cache_release_origin,
                    "hccl_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "HCCL_NPU_SOCKET_PORT_RANGE",
                            "HCCL_HOST_SOCKET_PORT_RANGE",
                        )
                    },
                }
            except Exception as exc:  # noqa: BLE001 -- settle CPU status before failing the job
                error = repr(exc)
                atomic_json(
                    root / f"trainer_failure_rank_{rank}.json",
                    {
                        "error": error,
                        "original_traceback": traceback.format_exc(),
                        "allocation_inventory": parking.coverage_inventory,
                    },
                )
                result = None
            dist.all_gather_object(errors, error, group=control)
            if any(errors):
                failed_ranks = [rank for rank, error in enumerate(errors) if error]
                raise RuntimeError(
                    f"Resident ranks failed: {failed_ranks}; "
                    "inspect trainer_failure_rank_*.json for original errors"
                )
            results = [None] * dist.get_world_size()
            dist.all_gather_object(results, result, group=control)
            if rank == 0:
                atomic_json(
                    Path(command["snapshot"]) / "speco_runtime_snapshot.json",
                    {
                        "contract": session.contract.identity(),
                        "target_version": 0,
                        "drafter_revision": completed,
                        "training_tp": topology.tensor_parallel_size,
                        "training_dp": topology.data_parallel_size,
                        "available_dp_rank": 0,
                        "recovery_checkpoint": False,
                    },
                )
                atomic_json(
                    root / f"result_{sequence:02d}.json",
                    {"status": "passed", "workers": results},
                )
    finally:
        # A parked pool must be mapped again before torch tears down its storages.
        try:
            evidence = wake_parked_for_shutdown(
                parking,
                session.ascend_engine,
                free_bytes=lambda: torch.npu.mem_get_info()[0],
            )
            if evidence is not None:
                atomic_json(root / f"shutdown_release_rank_{rank}.json", evidence)
        finally:
            if owned:
                dist.destroy_process_group()


def main():
    # torchrun subprocess environment must set this before importing torch.
    if os.environ.get("PYTORCH_NPU_ALLOC_CONF") != "expandable_segments:False":
        raise RuntimeError(
            "Set PYTORCH_NPU_ALLOC_CONF=expandable_segments:False for resident trainer workers"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--production-lifecycle", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    try:
        worker(args)
    except BaseException as error:
        failure = root / f"trainer_failure_rank_{os.environ.get('RANK', '0')}.json"
        details = json.loads(failure.read_text()) if failure.exists() else {}
        details.update(collective_error=repr(error))
        details.setdefault("error", repr(error))
        details.setdefault("original_traceback", traceback.format_exc())
        atomic_json(failure, details)
        raise


if __name__ == "__main__":
    main()
