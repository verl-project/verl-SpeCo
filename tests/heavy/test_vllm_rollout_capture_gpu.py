# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in CUDA numerical acceptance test against a same-weight HF target.

Set SPECO_CAPTURE_TARGET and SPECO_CAPTURE_DRAFT to compatible model paths.
SPECO_CAPTURE_ALGORITHM defaults to DSPARK; EAGLE3/DFLASH are also selectable.
Run each pytest parameter node in a fresh process. This test intentionally runs reference forwards; production capture does
not. Run in a fresh process because runner hooks and CUDA graphs are process-wide.
Set SPECO_CAPTURE_TP_SIZE=2 (and expose two CUDA devices) to validate TP capture.
"""

import os
import json
import time
from pathlib import Path

import pytest


def cached_reference_states(model, tokens, prefill_length):
    """Teacher-force the exact token sequence; never sample reference tokens."""
    import torch

    started = time.perf_counter()
    interval = max(1, int(os.getenv("SPECO_CAPTURE_REFERENCE_PROGRESS_INTERVAL", "16")))
    total = tokens.shape[1] - prefill_length
    print(
        f"SPECO_REFERENCE incremental_prefill_start tokens={prefill_length} decode_tokens={total}",
        flush=True,
    )
    output = model(
        tokens[:, :prefill_length], output_hidden_states=True, use_cache=True
    )
    chunks = [[state] for state in output.hidden_states]
    cache = output.past_key_values
    print(
        f"SPECO_REFERENCE incremental_prefill_done elapsed_s={time.perf_counter() - started:.2f}",
        flush=True,
    )
    for position in range(prefill_length, tokens.shape[1]):
        output = model(
            tokens[:, position : position + 1],
            past_key_values=cache,
            output_hidden_states=True,
            use_cache=True,
        )
        cache = output.past_key_values
        for parts, state in zip(chunks, output.hidden_states, strict=True):
            parts.append(state)
        completed = position - prefill_length + 1
        if completed == 1 or completed % interval == 0 or completed == total:
            print(
                f"SPECO_REFERENCE incremental_progress completed={completed}/{total} elapsed_s={time.perf_counter() - started:.2f}",
                flush=True,
            )
    result = tuple(torch.cat(parts, dim=1) for parts in chunks)
    print(
        f"SPECO_REFERENCE incremental_done elapsed_s={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return result


def install_forward_audit(model):
    """Diagnostic only: snapshot target layer inputs before any runner/export reuse."""
    import inspect

    if type(model).__name__ not in {"Qwen2ForCausalLM", "Qwen3ForCausalLM"}:
        raise RuntimeError("Forward audit supports Qwen2/Qwen3 only")
    model._speco_audit = []
    model._speco_audit_handles = []

    def begin(module, args, kwargs):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        values = bound.arguments
        model._speco_audit.append(
            {
                "positions": values["positions"].detach().cpu().clone(),
                "tokens": values["input_ids"].detach().cpu().clone(),
                "layers": {},
            }
        )

    model._speco_audit_handles.append(
        model.register_forward_pre_hook(begin, with_kwargs=True)
    )

    def layer_hook(index, signature):
        def hook(module, args, kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            hidden = bound.arguments["hidden_states"]
            residual = bound.arguments.get("residual")
            full = hidden if residual is None else hidden + residual
            model._speco_audit[-1]["layers"][str(index)] = full.detach().cpu().clone()

        return hook

    # Only retain layers requested by the checkpoint.
    config = model.config
    layers = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
    if layers is None:
        layers = range(len(model.model.layers))
    for index, layer in enumerate(model.model.layers):
        if index in layers:
            model._speco_audit_handles.append(
                layer.register_forward_pre_hook(
                    layer_hook(index, inspect.signature(layer.forward)),
                    with_kwargs=True,
                )
            )

    def final(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        model._speco_audit[-1]["layers"]["last"] = hidden.detach().cpu().clone()

    model._speco_audit_handles.append(model.model.norm.register_forward_hook(final))
    return True


def collect_forward_audit(model):
    for handle in model._speco_audit_handles:
        handle.remove()
    records = model._speco_audit
    del model._speco_audit, model._speco_audit_handles
    return records


def compare_forward_audit(sample, records, prompt, response):
    """Keep repeated-position candidates explicit; never guess acceptance ownership."""
    labels = [str(x) for x in sample["hidden_layer_ids"]]
    if sample["hidden_states_layout"].endswith("plus_last"):
        labels.append("last")
    actual = sample["hidden_states"][0].float()
    width = actual.shape[-1] // len(labels)
    tokens = prompt + response[:-1]
    reports = []
    for row, position in enumerate(sample["hidden_positions"][0].tolist()):
        candidates = []
        for step, record in enumerate(records):
            for index in (record["positions"] == position).nonzero().flatten().tolist():
                if int(record["tokens"][index]) != tokens[position]:
                    continue
                errors = {}
                for part, label in enumerate(labels):
                    a = actual[row, part * width : (part + 1) * width]
                    b = record["layers"][label][index].float()
                    errors[label] = float(
                        (a - b).square().mean().sqrt()
                        / b.square().mean().sqrt().clamp_min(1e-8)
                    )
                candidates.append(
                    {
                        "forward": step,
                        "stage": "prefill" if step == 0 else "decode_or_verification",
                        "forward_positions": record["positions"].tolist(),
                        "errors": errors,
                    }
                )
        reports.append(
            {
                "position": position,
                "input_token": tokens[position],
                "candidates": candidates,
                "matches_any_forward": any(
                    all(v < 0.0001 for v in c["errors"].values()) for c in candidates
                ),
            }
        )
    return reports


@pytest.mark.skipif(
    not (os.getenv("SPECO_CAPTURE_TARGET") and os.getenv("SPECO_CAPTURE_DRAFT")),
    reason="requires CUDA and explicit compatible target/draft model paths",
)
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("runner_v2", [False, True])
def test_rollout_features_match_reference(eager, runner_v2, monkeypatch, tmp_path):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from verl_speco.integration import vllm_rollout_capture as capture

    tp_size = int(os.getenv("SPECO_CAPTURE_TP_SIZE", "1"))
    dcp_size = int(os.getenv("SPECO_CAPTURE_DCP_SIZE", "1"))
    assert tp_size >= 1
    assert 1 <= dcp_size <= tp_size and tp_size % dcp_size == 0
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if runner_v2 else "0")
    if tp_size == 1:
        monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    assert torch.cuda.is_available(), "This acceptance test requires CUDA"
    from verl_speco.integration.vllm_runtime import (
        _speculative_method_from_drafter,
        _normalize_dflash_target_layer_aliases,
        patch_vllm_dflash_config_aliases,
        patch_vllm_dspark_registry_aliases,
        patch_vllm_dspark_draft_load_config,
        patch_vllm_dspark_runtime,
    )
    from verl_speco.integration.oldlogprob_layer_ids import (
        resolve_drafter_hidden_states_layout,
    )

    algorithm = os.getenv("SPECO_CAPTURE_ALGORITHM", "DSPARK").upper()
    assert algorithm in {"DSPARK", "DFLASH", "EAGLE3"}, (
        "Unsupported acceptance-test algorithm"
    )
    training = {
        "dspark_l1_loss_alpha": 0.9,
        "hidden_state_window_tokens_per_sample": 32,
    }
    training.update(json.loads(os.getenv("SPECO_CAPTURE_TRAINING_JSON", "{}")))
    if algorithm in {"DSPARK", "DFLASH"}:
        patch_vllm_dflash_config_aliases()
    if algorithm == "DSPARK":
        patch_vllm_dspark_registry_aliases()
        patch_vllm_dspark_draft_load_config()
        patch_vllm_dspark_runtime()
    capture.install_runner_capture()
    # Spawned TP workers need the same production hook/DSpark initialization.
    from verl_speco.integration.vllm_runtime import SPECO_DRAFTER_CONFIG_ENV

    monkeypatch.setenv(
        SPECO_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                "enable": True,
                "enable_drafter_training": True,
                "speculative_algorithm": algorithm,
                "training": {**training, capture.CAPTURE_FLAG: True},
            }
        ),
    )
    from vllm import LLM, SamplingParams

    target, draft = (
        os.environ["SPECO_CAPTURE_TARGET"],
        os.environ["SPECO_CAPTURE_DRAFT"],
    )
    draft_config = AutoConfig.from_pretrained(draft, trust_remote_code=True)
    if algorithm in {"DSPARK", "DFLASH"}:
        _normalize_dflash_target_layer_aliases(draft_config)
    expected_layers = (
        json.loads(os.environ["SPECO_CAPTURE_EXPECTED_LAYERS"])
        if os.getenv("SPECO_CAPTURE_EXPECTED_LAYERS")
        else getattr(draft_config, "eagle_aux_hidden_state_layer_ids", None)
    )
    assert expected_layers, (
        "Checkpoint must declare auxiliary layers, or set SPECO_CAPTURE_EXPECTED_LAYERS to an independently verified JSON list"
    )
    tokenizer = AutoTokenizer.from_pretrained(target)
    prompt = tokenizer.encode("Explain why the sky appears blue in two sentences.")
    fixture_path = os.getenv("SPECO_CAPTURE_TOKEN_FIXTURE")
    fixture = None
    if fixture_path and Path(fixture_path).exists():
        fixture = json.loads(Path(fixture_path).read_text())
        assert fixture["target"] == target and fixture["draft"] == draft
        prompt = fixture["prompt"]
    max_tokens = int(os.getenv("SPECO_CAPTURE_BENCH_MAX_TOKENS", "32"))
    if fixture is not None:
        assert fixture["max_tokens"] == max_tokens, (
            "Token fixture generation limit differs"
        )
    plan = capture.make_plan(
        training,
        prompt_len=len(prompt),
        max_tokens=max_tokens,
        request_id="gpu",
        global_step=1,
        replica_rank=0,
    )
    plan["layout"] = resolve_drafter_hidden_states_layout(algorithm, training)
    directory = capture.plan_directory(plan)
    directory.mkdir(parents=True)
    (directory / "live").touch()
    engine = LLM(
        model=target,
        dtype="bfloat16",
        speculative_config={
            "method": _speculative_method_from_drafter(
                {"speculative_algorithm": algorithm}
            ),
            "model": draft,
            "num_speculative_tokens": int(
                os.getenv(
                    "SPECO_CAPTURE_SPEC_TOKENS", "7" if algorithm == "DSPARK" else "3"
                )
            ),
        },
        tensor_parallel_size=tp_size,
        decode_context_parallel_size=dcp_size,
        worker_extension_cls="verl_speco.integration.vllm_runtime.SpecoVLLMColocateWorkerExtension",
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=eager,
        gpu_memory_utilization=0.5,
        max_model_len=max(512, len(prompt) + max_tokens),
    )
    audit_enabled = os.getenv("SPECO_CAPTURE_FORWARD_AUDIT") == "1"
    if audit_enabled:
        assert tp_size == 1 and eager, "Forward audit requires TP=1 and eager mode"
        engine.apply_model(install_forward_audit)
    try:
        output = engine.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                temperature=0,
                max_tokens=max_tokens,
                extra_args={capture.CAPTURE_KEY: plan},
            ),
        )[0]
        audit_records = (
            engine.apply_model(collect_forward_audit) if audit_enabled else None
        )
        response = list(output.outputs[0].token_ids)
        if fixture is not None:
            assert response == fixture["response"], (
                "Greedy response differs from token fixture; runs cannot be compared position-wise"
            )
        elif fixture_path:
            fixture_file = Path(fixture_path)
            fixture_file.parent.mkdir(parents=True, exist_ok=True)
            fixture_file.write_text(
                json.dumps(
                    {
                        "target": target,
                        "draft": draft,
                        "prompt": prompt,
                        "response": response,
                        "max_tokens": max_tokens,
                    }
                )
            )
        baseline = engine.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(temperature=0, max_tokens=max_tokens),
        )[0]
        assert response == list(baseline.outputs[0].token_ids), (
            "Capturing features changed greedy generation"
        )
        sample = capture.load_sample(plan, prompt, response)
        assert sample is not None
        artifact_dir = os.getenv("SPECO_CAPTURE_ARTIFACT_DIR")
        artifact = None
        if artifact_dir:
            artifact = Path(artifact_dir) / f"tp{tp_size}-v2{runner_v2}-eager{eager}"
            artifact.mkdir(parents=True, exist_ok=True)
            import hashlib
            import subprocess

            sources = [
                Path(__file__),
                Path(capture.__file__),
                Path(capture.__file__).with_name("vllm_rollout_capture_v2.py"),
            ]
            provenance = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
                if p.exists()
            }
            for name, args in {
                "head": ["rev-parse", "HEAD"],
                "status": ["status", "--porcelain"],
                "diff": ["diff", "HEAD", "--", "verl_speco/integration", "tests/heavy"],
            }.items():
                result = subprocess.run(["git", *args], capture_output=True, text=True)
                provenance[name] = (
                    result.stdout if result.returncode == 0 else result.stderr
                )
            (artifact / "provenance.json").write_text(json.dumps(provenance, indent=2))
            torch.save(sample, artifact / "captured.pt")
            (artifact / "tokens.json").write_text(
                json.dumps(
                    {
                        "target": target,
                        "draft": draft,
                        "prompt": prompt,
                        "response": response,
                    }
                )
            )
        if audit_records is not None:
            assert len(audit_records) == 1
            audit = compare_forward_audit(sample, audit_records[0], prompt, response)
            destination = artifact or directory
            torch.save(audit_records[0], destination / "forward-audit.pt")
            (destination / "forward-audit.json").write_text(json.dumps(audit, indent=2))
            unmatched = [r for r in audit if not r["matches_any_forward"]]
            print(
                "SPECO_FORWARD_AUDIT "
                + json.dumps(
                    {
                        "rows": len(audit),
                        "unmatched": len(unmatched),
                        "first_unmatched": unmatched[:1],
                        "artifact": str(destination),
                    }
                ),
                flush=True,
            )
            assert not unmatched, (
                "Export differs from all same-token target forwards; see forward-audit.json"
            )
        repeats = int(os.getenv("SPECO_CAPTURE_BENCH_REPEATS", "0"))
        if repeats:
            import uuid
            import statistics

            elapsed = {False: [], True: []}
            for iteration in range(repeats + 1):
                for collect in (False, True) if iteration % 2 == 0 else (True, False):
                    trial = {**plan, "sample_id": uuid.uuid4().hex}
                    if collect:
                        folder = capture.plan_directory(trial)
                        folder.mkdir(parents=True)
                        (folder / "live").touch()
                    try:
                        started = time.perf_counter()
                        measured = engine.generate(
                            [{"prompt_token_ids": prompt}],
                            SamplingParams(
                                temperature=0,
                                max_tokens=max_tokens,
                                extra_args={capture.CAPTURE_KEY: trial}
                                if collect
                                else None,
                            ),
                        )[0]
                        assert list(measured.outputs[0].token_ids) == response
                        if collect:
                            received = capture.load_sample(trial, prompt, response)
                            assert received is not None
                            assert torch.equal(
                                received["hidden_positions"], sample["hidden_positions"]
                            )
                            assert torch.equal(
                                received["hidden_states"], sample["hidden_states"]
                            )
                        duration = time.perf_counter() - started
                        if iteration:
                            elapsed[collect].append(duration)
                    finally:
                        if collect:
                            capture.cleanup(trial)
            print(
                "SPECO_FIXED_WEIGHT_BENCH "
                + json.dumps(
                    {
                        "repeats": repeats,
                        "tp": tp_size,
                        "eager": eager,
                        "pack_chunks": os.getenv("SPECO_CAPTURE_PACK_CHUNKS", "0"),
                        "seconds": {str(k): v for k, v in elapsed.items()},
                        "median_seconds": {
                            str(k): statistics.median(v) for k, v in elapsed.items()
                        },
                    }
                ),
                flush=True,
            )
        # Keep the reference on CPU so the engine and HF weights do not compete
        # for GPU memory. Use the same target dtype for a useful comparison.
        reference_previous_threads = torch.get_num_threads()
        available_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        reference_threads = int(
            os.getenv("SPECO_CAPTURE_REFERENCE_THREADS", str(min(8, available_cpus)))
        )
        assert reference_threads > 0, "Reference thread count must be positive"
        torch.set_num_threads(reference_threads)
        print(
            f"SPECO_REFERENCE cpu_threads={torch.get_num_threads()} previous_threads={reference_previous_threads}",
            flush=True,
        )
        reference_dtype_name = os.getenv("SPECO_CAPTURE_REFERENCE_DTYPE", "bfloat16")
        assert reference_dtype_name in {"bfloat16", "float32"}
        reference_started = time.perf_counter()
        print(
            f"SPECO_REFERENCE load_start dtype={reference_dtype_name} device=cpu",
            flush=True,
        )
        reference = AutoModelForCausalLM.from_pretrained(
            target, torch_dtype=getattr(torch, reference_dtype_name)
        ).eval()
        print(
            f"SPECO_REFERENCE load_done elapsed_s={time.perf_counter() - reference_started:.2f}",
            flush=True,
        )
        reference_mode = os.getenv("SPECO_CAPTURE_REFERENCE_MODE", "full")
        assert reference_mode in {"full", "incremental", "compare"}
        tokens = torch.tensor([prompt + response[:-1]])
        with torch.inference_mode():
            full_started = time.perf_counter()
            print(
                f"SPECO_REFERENCE full_start tokens={tokens.shape[1]} mode={reference_mode}",
                flush=True,
            )
            states = reference(
                tokens,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states
            print(
                f"SPECO_REFERENCE full_done elapsed_s={time.perf_counter() - full_started:.2f}",
                flush=True,
            )
            if reference_mode != "full":
                incremental = cached_reference_states(reference, tokens, len(prompt))
                for layer in [*expected_layers, len(states) - 1]:
                    full, cached = states[layer].float(), incremental[layer].float()
                    rmse = (
                        full - cached
                    ).square().mean().sqrt() / full.square().mean().sqrt().clamp_min(
                        1e-8
                    )
                    print(
                        "SPECO_HF_PATH_COMPARE "
                        + json.dumps(
                            {
                                "layer": layer,
                                "dtype": reference_dtype_name,
                                "relative_rmse": float(rmse),
                                "tokens": tokens.shape[1],
                                "prefill_length": len(prompt),
                            }
                        ),
                        flush=True,
                    )
                if reference_mode == "incremental":
                    states = incremental
        positions = sample["hidden_positions"][0]
        assert sample["hidden_layer_ids"] == list(expected_layers), (
            "Captured layers disagree with checkpoint/reference configuration"
        )
        expected_parts = [states[i][0, positions] for i in expected_layers]
        labels = [f"aux_hf_index_{i}" for i in expected_layers]
        if plan["layout"].endswith("plus_last"):
            assert sample["last_hidden_state_norm"] == "post_norm"
            expected_parts.append(states[-1][0, positions])
            labels.append("last_post_norm")
        expected = torch.cat(expected_parts, dim=-1).float()
        actual = sample["hidden_states"][0].float()
        width = states[-1].shape[-1]
        assert actual.shape == expected.shape
        failures, reports = [], []
        for label, a, b in zip(
            labels,
            actual.split(width, dim=-1),
            expected.split(width, dim=-1),
            strict=True,
        ):
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())

            relative_rmse = (
                a - b
            ).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8)
            print(
                f"algorithm={algorithm} tp={tp_size} dcp={dcp_size} runner_v2={runner_v2} eager={eager} layer={label} rows={positions.numel()} relative_rmse={relative_rmse.item():.6f} max_abs_error={(a - b).abs().max().item():.6f}",
                flush=True,
            )
            row_error = (a - b).square().mean(dim=-1).sqrt()
            row_relative = row_error / b.square().mean(dim=-1).sqrt().clamp_min(1e-8)
            worst = (
                row_error.nan_to_num(nan=float("inf")).topk(min(5, a.shape[0])).indices
            )
            report = {
                "layer": label,
                "reference_dtype": reference_dtype_name,
                "relative_rmse": relative_rmse.item() if finite else None,
                "finite": finite,
                "rows_over_3pct": int((row_relative > 0.03).sum()),
                "worst_positions": [
                    {
                        "position": int(positions[i]),
                        "input_token": int((prompt + response[:-1])[int(positions[i])]),
                        "relative_rmse": float(row_relative[i]) if finite else None,
                        "capture_max_abs": float(a[i].abs().max()),
                        "reference_max_abs": float(b[i].abs().max()),
                    }
                    for i in worst
                ],
            }
            reports.append(report)
            report["reference_mode"] = reference_mode
            details = []
            for row in worst[:2]:
                row_index = int(row)
                channels = (
                    (a[row_index] - b[row_index])
                    .abs()
                    .topk(min(3, a.shape[-1]))
                    .indices
                )
                for channel in channels:
                    col = int(channel)
                    neighbors = range(
                        max(0, row_index - 2), min(a.shape[0], row_index + 3)
                    )
                    details.append(
                        {
                            "position": int(positions[row_index]),
                            "channel": col,
                            "neighbors": [
                                {
                                    "position": int(positions[j]),
                                    "capture": float(a[j, col]),
                                    "reference": float(b[j, col]),
                                }
                                for j in neighbors
                            ],
                        }
                    )
            report["worst_channels"] = details
            print("SPECO_LAYER_DIAGNOSTIC " + json.dumps(report), flush=True)
            if not finite or relative_rmse.item() >= 0.03:
                failures.append(label)
        if artifact is not None:
            (
                artifact / f"reference-{reference_dtype_name}-{reference_mode}.json"
            ).write_text(json.dumps(reports, indent=2))
        assert not failures, (
            f"Feature numerical acceptance failed (3% threshold): {failures}"
        )
    finally:
        if "reference_previous_threads" in locals():
            torch.set_num_threads(reference_previous_threads)
        capture.cleanup(plan)
        del engine
