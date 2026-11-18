# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process MTP native/standard artifact comparison on fixed replay rows."""

import json
import logging
import os
from pathlib import Path

import hydra
import torch

from verl_speco.probe_deepseek_v4_artifact import model_state
from verl_speco.probe_deepseek_v4_recovery import differences, persistent_state
from verl_speco.trainer.deepseek_v4_mtp_training import step_inputs


def forward_mtp(model, batch, device, ttt_length):
    batch = {
        key: value[:, :32]
        for key, value in batch.items()
        if key in {"input_ids", "positions", "loss_mask", "hidden_states"}
    }
    ids, positions, _ = step_inputs(batch, 1, ttt_length)
    with torch.enable_grad():
        hidden = model(
            input_ids=ids.to(device),
            positions=positions.to(device),
            positions_cpu=positions,
            hidden_states=batch["hidden_states"].to(device).requires_grad_(True),
            attention_mask=None,
            spec_step_idx=0,
        )
        logits = model.compute_logits(hidden, lm_head=None, spec_step_idx=0)
    return {"hidden": hidden.detach().cpu(), "logits": logits.detach().cpu()}


def run(config):
    import torch.distributed as dist

    from verl_speco.backends.vllm_ascend_api import set_seed
    from verl_speco.backends.vllm_ascend_mtp_model import load_mtp_model
    from verl_speco.backends.vllm_ascend_options import (
        export_memory_chunking_environment,
        resolve_memory_chunking,
    )
    from verl_speco.backends.vllm_ascend_parallel import DeepSpeedParallelTopology
    from verl_speco.backends.vllm_ascend_runtime import (
        initialize_distributed,
        load_initial_draft_weights,
        warmup_metadata_collectives,
    )
    from verl_speco.ops.ascend_weight_cache import refresh_fp32_weight_caches
    from verl_speco.trainer.deepseek_v4_dataset import build_loader
    from verl_speco.trainer.vllm_ascend_training_loop import _atomic_report, preflight

    for key, value in config.draft_engine.environment.items():
        os.environ.setdefault(key, str(value))
    output = Path(config.draft_engine.output_dir).resolve()
    rank = int(os.environ.get("RANK", "0"))
    dataset, owned = None, False
    mode = config.draft_engine.get("cold_probe_mode", "artifact")
    phase = "preflight"
    try:
        if mode not in {"reference", "artifact"}:
            raise ValueError("MTP cold mode must be reference or artifact")
        report, cfg, dataset = preflight(config)
        checkpoint = Path(config.draft_engine.resume_from).resolve(strict=True)
        sidecar = json.loads((checkpoint / "speco_ds4.json").read_text())
        if sidecar["identity"] != {name: report[name] for name in sidecar["identity"]}:
            raise ValueError("MTP reference checkpoint identity differs")
        for key in ("world_size", "tensor_parallel_size", "data_parallel_size"):
            if sidecar["topology"][key] != report[key]:
                raise ValueError("MTP reference topology differs")
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("Use a new MTP cold comparison directory")
            output.mkdir(parents=True, exist_ok=True)
            _atomic_report(output / "preflight.json", report)
        device, owned = initialize_distributed()
        control = dist.new_group(backend="gloo")
        dist.barrier(group=control)
        topology = DeepSpeedParallelTopology(report["tensor_parallel_size"])
        torch.set_num_threads(int(cfg.train.omp_num_threads))
        if cfg.debug.get("seed") is not None:
            set_seed(int(cfg.debug.seed))
        export_memory_chunking_environment(resolve_memory_chunking(cfg))
        # Reuse the production model constructor and loader; no second model
        # is constructed in this process and no vendor sources are modified.
        phase = "load_" + mode
        model = load_mtp_model(cfg, logging.getLogger(__name__))
        warmup_metadata_collectives(topology, device)
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
        else:
            artifact = Path(config.draft_engine.artifact_path).resolve(strict=True)
            metadata = json.loads((artifact / "metadata.json").read_text())
            if metadata["step"] != sidecar["progress"]["global_step"]:
                raise ValueError("MTP artifact/checkpoint steps differ")
            cfg.model.initial_draft_artifact = str(artifact)
            load_initial_draft_weights(model, cfg)
        model.freeze_params(cfg)
        model.train()
        phase = "parameter_comparison"
        current = model_state(model)
        changed = []
        reference = config.draft_engine.get("cold_probe_reference")
        if mode == "artifact" and not reference:
            raise ValueError("MTP artifact comparison requires a native reference")
        if reference:
            expected = json.loads(
                (Path(reference) / f"weights_rank_{rank}.json").read_text()
            )
            changed = differences(expected, current)
        _atomic_report(output / f"weights_rank_{rank}.json", current)
        if changed:
            raise ValueError(
                f"MTP artifact persistent weights/cache/layout differ: {changed[:10]}"
            )
        _, loader = build_loader(
            dataset,
            cfg,
            dp_rank=topology.data_parallel_rank,
            dp_size=topology.data_parallel_size,
        )
        batch = next(iter(loader))
        original_inputs = persistent_state(batch)
        phase = "forward"
        first = forward_mtp(model, batch, device, int(cfg.train.ttt_length))
        torch.save(first, output / f"forward_rank_{rank}.pt")
        second = forward_mtp(model, batch, device, int(cfg.train.ttt_length))
        repeated = differences(persistent_state(first), persistent_state(second))
        if repeated or differences(original_inputs, persistent_state(batch)):
            raise ValueError("MTP repeated forward differs or modified input features")
        errors = {key: 0.0 for key in first}
        if reference:
            expected = torch.load(
                Path(reference) / f"forward_rank_{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
            changed = differences(persistent_state(expected), persistent_state(first))
            errors = {
                key: float((expected[key].float() - first[key].float()).abs().max())
                for key in first
            }
            if changed:
                raise ValueError(f"MTP cold forward differs: {errors}")
        row = {
            "rank": rank,
            "status": "passed",
            "mode": mode,
            "max_abs": errors,
            "weights_cache_layout_exact": True,
            "repeat_exact": True,
            "scope": "same_training_tp_fixed_mtp_replay_rows_full_vocab_logits",
            "serving_engine_validated": False,
        }
        _atomic_report(output / f"rank_{rank}.json", row)
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, row, group=control)
        if rank == 0:
            _atomic_report(
                output / "cold_load_report.json", {"status": "passed", "workers": rows}
            )
    except BaseException as error:
        _atomic_report(
            output / f"failure_rank_{rank}.json", {"phase": phase, "error": repr(error)}
        )
        raise
    finally:
        if dataset is not None:
            dataset.store.close()
        if owned:
            dist.destroy_process_group()


@hydra.main(
    config_path="config", config_name="deepseek_v4_mtp_draft_trainer", version_base=None
)
def main(config):
    logging.basicConfig(level=logging.INFO)
    run(config)


if __name__ == "__main__":
    main()
