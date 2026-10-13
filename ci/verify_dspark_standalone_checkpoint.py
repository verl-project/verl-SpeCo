# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Prepare DSpark CI baselines and verify a fresh standalone checkpoint.

Uses the same backend as training. The final synthetic batch checks reload and
forward compatibility, not model quality or vLLM serving compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def read_training_losses(text: str) -> list[float]:
    values = re.findall(r"\[standalone drafter metrics\][^\n]*?avg_loss=([^\s]+)", text)
    losses = [float(value) for value in values]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError(f"Missing or non-finite training losses: {values}")
    return losses


def changed_parameters(before: dict, current: dict) -> list[str]:
    if current.keys() != before.keys():
        raise ValueError("Trainable parameter names changed")
    changed = []
    for name, value in current.items():
        if value["shape"] != before[name]["shape"]:
            raise ValueError(f"Trainable shape changed: {name}")
        if value["sha256"] != before[name]["sha256"]:
            changed.append(name)
    if not changed:
        raise ValueError("No trainable parameter changed in the exported checkpoint")
    return changed


def build_backend(target: str, checkpoint: str):
    import torch
    import torch_npu  # noqa: F401 - register NPU before importing the backend
    from omegaconf import OmegaConf
    from transformers import AutoConfig
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend

    config = OmegaConf.create({
        "model": {"path": target},
        "rollout": {"drafter": {
            "model_path": checkpoint,
            "training": {
                "dspark_block_size": 7, "dspark_num_anchors": 8,
                "dspark_loss_mode": "restricted_ce", "dspark_loss_decay_gamma": 7,
                "dspark_ce_loss_alpha": 0.1, "dspark_l1_loss_alpha": 0.9,
                "dspark_confidence_loss_alpha": 0.0,
                "dspark_distribution_loss_impl": "eager",
            },
        }},
    })
    backend = DSparkTrainerBackend(config, AutoConfig.from_pretrained(target))
    model, model_config = backend.build_model()
    state = backend._normalize_draft_state_dict(backend._load_draft_state_dict(checkpoint))
    parameters = dict(model.draft_model.named_parameters())
    required = {name for name, param in parameters.items() if param.requires_grad}
    missing = required - state.keys()
    unexpected = state.keys() - model.draft_model.state_dict().keys()
    if missing or unexpected:
        raise ValueError(f"Incomplete checkpoint load: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, tensor in state.items():
        expected = model.draft_model.state_dict()[name]
        if tensor.shape != expected.shape:
            raise ValueError(f"Shape mismatch for {name}: {tensor.shape} != {expected.shape}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite saved tensor: {name}")
        # Ensure the backend did not silently drop a checkpoint tensor.
        if name in required and not torch.equal(tensor.to(expected.dtype), expected):
            raise ValueError(f"Checkpoint tensor was not loaded: {name}")
    return backend, model, model_config


def fingerprints(model):
    import torch

    result = {}
    for name, parameter in model.draft_model.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().float().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite parameter: {name}")
        result[name] = {
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if not result:
        raise ValueError("No trainable drafter parameters")
    return result


def prepare(args):
    backend, model, config = build_backend(args.target, args.initial)
    from verl_speco.standalone_layer_ids import normalize_standalone_layer_ids

    target_ids, aux_ids = normalize_standalone_layer_ids("DSPARK", config.target_layer_ids, None)
    final_id = int(config.target_num_hidden_layers)
    if any(layer >= final_id for layer in target_ids):
        raise ValueError("Drafter capture plan exceeds target depth")
    def compact(values):
        return json.dumps(list(values), separators=(",", ":"))
    Path(args.layer_env).write_text(
        f"TARGET_LAYER_IDS='{compact(target_ids)}'\n"
        f"AUX_LAYER_IDS='{compact(aux_ids)}'\n"
        f"CAPTURE_LAYER_IDS='{compact([*aux_ids, final_id])}'\n",
        encoding="utf-8",
    )
    Path(args.baseline).write_text(json.dumps({
        "initial": str(Path(args.initial).resolve()),
        "target": str(Path(args.target).resolve()),
        "parameters": fingerprints(model),
        "target_layer_ids": list(target_ids),
    }, indent=2), encoding="utf-8")
    print(f"Prepared baseline and layer plan: target={target_ids}, vLLM={aux_ids}, final={final_id}")


def verify(args):
    import torch
    import torch_npu  # noqa: F401
    from verl_speco.trainer.checkpoint import (
        get_drafter_checkpoint_step, get_drafter_trainer_state,
        is_pretrained_drafter_checkpoint,
    )

    checkpoint = Path(args.checkpoint)
    if not is_pretrained_drafter_checkpoint(checkpoint):
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    state = get_drafter_trainer_state(checkpoint)
    if get_drafter_checkpoint_step(checkpoint) != args.steps:
        raise ValueError("Saved checkpoint step differs from requested step")
    if int(state.get("optimizer_steps_total", -1)) != args.steps:
        raise ValueError(f"Expected {args.steps} actual optimizer steps: {state}")
    resume = json.loads((checkpoint / "standalone_resume.json").read_text(encoding="utf-8"))
    if int(resume["optimizer_step"]) != args.steps:
        raise ValueError("Standalone resume step mismatch")
    losses = read_training_losses(Path(args.training_log).read_text(encoding="utf-8", errors="replace"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    if baseline["initial"] != str(Path(args.initial).resolve()) or baseline["target"] != str(Path(args.target).resolve()):
        raise ValueError("Baseline belongs to different models")
    backend, model, config = build_backend(args.target, str(checkpoint))
    current = fingerprints(model)
    before = baseline["parameters"]
    if list(config.target_layer_ids) != baseline["target_layer_ids"]:
        raise ValueError("Saved target layer plan changed")
    changed = changed_parameters(before, current)

    # A fresh process builds the full training backend from the saved artifact.
    # Keep this input synthetic: it verifies computation, not data quality.
    torch.manual_seed(0)
    device = torch.device("npu:0")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    backend.target_lm_head = backend.target_lm_head.to(device=device, dtype=torch.bfloat16)
    rows = 16
    hdim = int(config.target_hidden_size)
    batch = {
        "input_ids": torch.randint(0, int(config.vocab_size), (1, rows), device=device),
        "hidden_states": torch.randn(1, rows, hdim * int(config.num_context_layers), device=device, dtype=torch.bfloat16),
        "target_last_hidden_states": torch.randn(1, rows, hdim, device=device, dtype=torch.bfloat16),
        "loss_mask": torch.ones(1, rows, device=device),
    }
    with torch.no_grad():
        result = backend.compute_loss(model, batch, 0)
    count = float(result["local_num_tokens"].item())
    if count <= 0:
        raise ValueError("Reload forward produced no supervised tokens")
    loss = float((result["total_local_ploss"] / result["local_num_tokens"]).item())
    if not math.isfinite(loss) or tuple(result["loss_per_position"].shape) != (7,):
        raise ValueError("Reload forward has invalid loss or output shape")
    report = {
        "checkpoint": str(checkpoint), "optimizer_steps": args.steps,
        "training_losses": [float(value) for value in losses],
        "changed_parameter_count": len(changed), "changed_parameters": changed,
        "reload_forward_loss": loss, "reload_supervised_tokens": count,
        "forward_input": "synthetic", "vllm_checkpoint_reload_tested": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--layer-env")
    parser.add_argument("--checkpoint")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--training-log")
    parser.add_argument("--report")
    args = parser.parse_args()
    required = ("layer_env",) if args.mode == "prepare" else ("checkpoint", "steps", "training_log", "report")
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"{args.mode} requires --{name.replace('_', '-')}")
    (prepare if args.mode == "prepare" else verify)(args)


if __name__ == "__main__":
    main()
