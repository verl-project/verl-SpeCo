# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare standard-artifact cold loading against saved native TP parameters.

This exercises the actual Ascend loader, postprocessing and forward kernels.
It validates the saved training TP topology, not a different serving EP layout.
"""

import json
import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from verl_speco.probe_deepseek_v4_recovery import differences, persistent_state


def model_state(model):
    result = persistent_state(
        {
            "weights": model.state_dict(),
            "router_caches": {
                name: module.weight_fp32
                for name, module in model.named_modules()
                if torch.is_tensor(getattr(module, "weight_fp32", None))
            },
        }
    )
    import torch_npu

    result["parameter_layouts"] = {
        name: {
            "stride": list(value.stride()),
            "npu_format": int(torch_npu.get_npu_format(value)),
        }
        for name, value in model.named_parameters(remove_duplicate=False)
    }
    return result


def forward(model, packed):
    # Validate the actual training forward. The serving engine has its own
    # metadata executor and requires a separate end-to-end inference test.
    with torch.enable_grad():
        hidden = model.forward_packed_runtime(packed)
        logits = model.apply_markov_bias(
            model.compute_base_logits(hidden), packed.previous_token_ids
        )
    return {"hidden": hidden.detach().cpu(), "logits": logits.detach().cpu()}


def run(config):
    import torch.distributed as dist

    from verl_speco.backends import vllm_ascend_api as api
    from verl_speco.backends.vllm_ascend_parallel import DeepSpeedParallelTopology
    from verl_speco.backends.vllm_ascend_runtime import (
        initialize_distributed,
        load_initial_draft_weights,
        warmup_metadata_collectives,
    )
    from verl_speco.integration.vllm_ascend_implementation import (
        implementation_identity_matches,
    )
    from verl_speco.models.deepseek_v4.modeling_dspark import pack_runtime_block_inputs
    from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches
    from verl_speco.trainer.deepseek_v4_dataset import build_loader
    from verl_speco.trainer.vllm_ascend_mask import install_speco_mask
    from verl_speco.trainer.vllm_ascend_training_loop import _atomic_report, preflight

    for name, value in config.draft_engine.environment.items():
        os.environ.setdefault(name, str(value))
    deterministic = bool(config.draft_engine.get("cold_probe_deterministic", False))
    if deterministic:
        # Diagnostic execution profile for this forward-only comparison. This
        # never changes or aliases the optimizer-recovery identity.
        os.environ["HCCL_DETERMINISTIC"] = "strict"
        os.environ["LCCL_DETERMINISTIC"] = "1"
        torch.use_deterministic_algorithms(True)
    rank = int(os.environ.get("RANK", "0"))
    output = Path(config.draft_engine.output_dir).resolve()
    checkpoint = Path(config.draft_engine.resume_from).resolve(strict=True)
    artifact = Path(config.draft_engine.artifact_path).resolve(strict=True)
    mode = config.draft_engine.get("cold_probe_mode", "artifact")
    if mode not in {"reference", "artifact"}:
        raise ValueError("cold_probe_mode must be reference or artifact")
    reference_root = config.draft_engine.get("cold_probe_reference")
    if mode == "artifact" and not reference_root:
        raise ValueError("Artifact comparison requires a fresh-process reference")
    dataset, owned, metadata_op = None, False, None
    phase = "preflight"
    try:
        report, cfg, dataset = preflight(config)
        metadata = json.loads((checkpoint / "speco_ds4.json").read_text())
        identity = {key: report[key] for key in metadata["identity"]}
        if not implementation_identity_matches(metadata["identity"], identity):
            raise ValueError("Reference checkpoint identity differs")
        for key in ("world_size", "tensor_parallel_size", "data_parallel_size"):
            if metadata["topology"][key] != report[key]:
                raise ValueError(f"Reference topology differs at {key}")
        artifact_metadata = json.loads((artifact / "metadata.json").read_text())
        if artifact_metadata["step"] != metadata["progress"]["global_step"]:
            raise ValueError("Artifact and checkpoint steps differ")
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("Use a fresh cold-load probe output directory")
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, output / "invocation.yaml")
            _atomic_report(output / "preflight.json", report)
        device, owned = initialize_distributed()
        control = dist.new_group(backend="gloo")
        dist.barrier(group=control)
        topology = DeepSpeedParallelTopology(report["tensor_parallel_size"])
        _, loader = build_loader(
            dataset,
            cfg,
            dp_rank=topology.data_parallel_rank,
            dp_size=topology.data_parallel_size,
        )
        torch.set_num_threads(int(cfg.train.get("omp_num_threads", 2)))
        api.set_seed(int(cfg.debug.seed))
        api.export_memory_chunking_environment(api.resolve_memory_chunking(cfg))
        logger = logging.getLogger(__name__)
        phase = "load_" + mode
        model = api.load_mtp_draft_model(cfg, logger)
        warmup_metadata_collectives(topology, device)
        install_speco_mask(model)
        if mode == "reference":
            model.load_weights(path=cfg.model.path, mode="full")
            state = torch.load(
                checkpoint / f"model_states_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            model.load_state_dict(state, strict=True)
            del state
            refresh_fp32_weight_caches(model)
            loading = {"initialization": "runtime_checkpoint"}
        else:
            # Exercise the same optional initializer used by build_engine.
            cfg.model.initial_draft_artifact = str(artifact)
            loading = load_initial_draft_weights(model, cfg)
        model.freeze_params(cfg)
        model.train()
        fresh_metadata_builders = bool(
            config.draft_engine.get("cold_probe_fresh_metadata_builders", False)
        )
        if fresh_metadata_builders:
            model.cache_dsa_metadata_structures = False
            model._dsa_metadata_builder_cache.clear()
        actual = model_state(model)
        changed = []
        if reference_root:
            previous = json.loads(
                (Path(reference_root) / f"weights_rank_{rank}.json").read_text()
            )
            if previous["diagnostic_deterministic_profile"] != deterministic:
                raise ValueError("Reference diagnostic execution profile differs")
            changed = differences(previous["actual"], actual)
        _atomic_report(
            output / f"weights_rank_{rank}.json",
            {
                "actual": actual,
                "different_paths": changed,
                "diagnostic_deterministic_profile": deterministic,
            },
        )
        changes = [None] * dist.get_world_size()
        dist.all_gather_object(changes, changed, group=control)
        if any(changes):
            raise RuntimeError(
                "Cold-loaded persistent parameters differ; see weights_rank files"
            )
        if config.draft_engine.get("cold_probe_weights_only", False):
            result = {
                "rank": rank,
                "status": "passed",
                "mode": mode,
                "loading": loading,
                "different_paths": changed,
                "weights_verified": True,
                "forward_validated": False,
                "serving_engine_validated": False,
            }
            _atomic_report(output / f"rank_{rank}.json", result)
            results = [None] * dist.get_world_size()
            dist.all_gather_object(results, result, group=control)
            if rank == 0:
                _atomic_report(
                    output / "weights_only_report.json",
                    {"status": "passed", "workers": results},
                )
            return
        empty_metadata_as_none = bool(
            config.draft_engine.get("cold_probe_empty_metadata_as_none", False)
        )
        if (
            config.draft_engine.get("cold_probe_log_metadata", False)
            or empty_metadata_as_none
        ):
            metadata_op = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
            calls = [0]

            def logged_metadata(*args, **kwargs):
                if empty_metadata_as_none:
                    kwargs = {
                        key: None
                        if key
                        in {"cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_q"}
                        and torch.is_tensor(value)
                        and value.numel() == 0
                        else value
                        for key, value in kwargs.items()
                    }

                def describe(value):
                    if torch.is_tensor(value):
                        return {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "values": value.detach().cpu().tolist()
                            if value.numel() < 4096
                            else "large",
                        }
                    return (
                        value
                        if isinstance(value, (int, float, bool, str, type(None)))
                        else str(value)
                    )

                calls[0] += 1
                _atomic_report(
                    output / f"metadata_rank_{rank}_call_{calls[0]}.json",
                    {
                        "args": [describe(v) for v in args],
                        "kwargs": {key: describe(v) for key, v in kwargs.items()},
                    },
                )
                return metadata_op(*args, **kwargs)

            torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata = logged_metadata
        phase = "forward"
        batch = {
            name: value.to(device)
            for name, value in next(iter(loader)).items()
            if torch.is_tensor(value)
        }
        targets, inputs = model.prepare_training_inputs(
            input_ids=batch["input_ids"],
            raw_mask=batch["raw_mask"],
            positions=batch["positions"],
            num_anchors=int(cfg.train.dspark_num_anchors),
            generator=torch.Generator().manual_seed(20261111),
        )
        packed = pack_runtime_block_inputs(
            batch["hidden_states"], targets, inputs, positions=batch["positions"]
        )
        input_state = persistent_state(packed)
        _atomic_report(output / f"inputs_rank_{rank}.json", input_state)
        if reference_root:
            previous = json.loads(
                (Path(reference_root) / f"inputs_rank_{rank}.json").read_text()
            )
            if differences(previous, input_state):
                raise RuntimeError("Reference and cold-model packed inputs differ")
        actual_forward = forward(model, packed)
        after_first_inputs = persistent_state(packed)
        _atomic_report(
            output / f"first_forward_rank_{rank}.json", persistent_state(actual_forward)
        )
        torch.save(actual_forward, output / f"forward_rank_{rank}.pt")
        repeated_forward = forward(model, packed)
        input_mutations = differences(input_state, after_first_inputs) + differences(
            input_state, persistent_state(packed)
        )
        expected_forward = (
            torch.load(
                Path(reference_root) / f"forward_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if reference_root
            else actual_forward
        )
        repeat_changes = differences(
            persistent_state(actual_forward), persistent_state(repeated_forward)
        )
        forward_changes = differences(
            persistent_state(expected_forward), persistent_state(actual_forward)
        )
        result = {
            "rank": rank,
            "mode": mode,
            "status": "passed"
            if not repeat_changes and not forward_changes and not input_mutations
            else "different",
            "serialization": artifact_metadata["serialization"],
            "loading": loading,
            "repeat_different_paths": repeat_changes,
            "input_mutations": sorted(set(input_mutations)),
            "repeat_errors": {
                name: {
                    "max_abs": float(
                        (value.float() - repeated_forward[name].float()).abs().max()
                    ),
                    "mean_abs": float(
                        (value.float() - repeated_forward[name].float()).abs().mean()
                    ),
                    "different_elements": int((value != repeated_forward[name]).sum()),
                    "elements": value.numel(),
                }
                for name, value in actual_forward.items()
            },
            "forward_different_paths": forward_changes,
            "reference_forward": persistent_state(expected_forward),
            "cold_forward": persistent_state(actual_forward),
            "max_abs": {
                name: float(
                    (expected_forward[name].float() - actual_forward[name].float())
                    .abs()
                    .max()
                )
                for name in expected_forward
            },
            "scope": "same_training_tp_layout_training_forward_full_vocab_logits",
            "serving_engine_validated": False,
            "diagnostic_deterministic_profile": deterministic,
            "diagnostic_empty_metadata_as_none": empty_metadata_as_none,
            "diagnostic_fresh_metadata_builders": fresh_metadata_builders,
        }
        _atomic_report(output / f"rank_{rank}.json", result)
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, result, group=control)
        if rank == 0:
            _atomic_report(
                output / "cold_load_report.json",
                {
                    "workers": results,
                    "status": "passed"
                    if all(r["status"] == "passed" for r in results)
                    else "different",
                },
            )
        if any(r["status"] != "passed" for r in results):
            raise RuntimeError("Native cold-load forward comparison differs")
    except BaseException as error:
        _atomic_report(
            output / f"failure_rank_{rank}.json", {"phase": phase, "error": repr(error)}
        )
        raise
    finally:
        if metadata_op is not None:
            torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata = metadata_op
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
