# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Independent target-only vLLM reference; never installed by training code."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.getenv("SPECO_CAPTURE_REFERENCE_ARTIFACT"),
    reason="requires saved capture artifact and CUDA",
)
def test_target_only_features_match_capture(monkeypatch):
    import torch
    from vllm import LLM, SamplingParams

    assert torch.cuda.is_available()
    folder = Path(os.environ["SPECO_CAPTURE_REFERENCE_ARTIFACT"])
    sample = torch.load(folder / "captured.pt", map_location="cpu", weights_only=True)
    token_data = json.loads((folder / "tokens.json").read_text())
    assert token_data.get("target") == os.environ["SPECO_CAPTURE_TARGET"], (
        "Artifact must record matching target checkpoint; regenerate with the updated capture test"
    )
    prompt, response = token_data["prompt"], token_data["response"]
    tokens = prompt + response[:-1]
    positions = sample["hidden_positions"][0].long()
    layers = sample["hidden_layer_ids"]
    assert sample["prompts"].reshape(-1).tolist() == prompt
    assert sample["responses"].reshape(-1).tolist() == response
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    engine = LLM(
        model=os.environ["SPECO_CAPTURE_TARGET"],
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_chunked_prefill=False,
        max_model_len=max(512, len(tokens) + 1),
        max_num_batched_tokens=max(512, len(tokens) + 1),
        gpu_memory_utilization=float(
            os.getenv("SPECO_CAPTURE_REFERENCE_GPU_MEMORY", "0.5")
        ),
    )

    def install(model):
        import inspect

        if type(model).__name__ not in {"Qwen3ForCausalLM", "Qwen2ForCausalLM"}:
            raise RuntimeError("Independent hook currently validates Qwen2/Qwen3 only")
        body = model.model
        model._speco_reference_records = {}
        model._speco_reference_handles = []

        def prehook(index, signature):
            def capture(module, args, kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                hidden, residual = values["hidden_states"], values.get("residual")
                full = hidden if residual is None else hidden + residual
                p = values["positions"].detach().cpu().clone()
                model._speco_reference_positions = p
                if index in layers:
                    model._speco_reference_records.setdefault(str(index), []).append(
                        (p, full.detach().cpu().clone())
                    )

            return capture

        for index, layer in enumerate(body.layers):
            model._speco_reference_handles.append(
                layer.register_forward_pre_hook(
                    prehook(index, inspect.signature(layer.forward)), with_kwargs=True
                )
            )

        def final_hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            model._speco_reference_records.setdefault("last", []).append(
                (model._speco_reference_positions, hidden.detach().cpu().clone())
            )

        model._speco_reference_handles.append(
            body.norm.register_forward_hook(final_hook)
        )
        return True

    def collect(model):
        result = model._speco_reference_records
        for handle in model._speco_reference_handles:
            handle.remove()
        return result

    try:
        engine.apply_model(install)
        engine.generate(
            [{"prompt_token_ids": tokens}], SamplingParams(temperature=0, max_tokens=1)
        )
        records = engine.apply_model(collect)
        assert len(records) == 1
        labels = [str(layer) for layer in layers]
        if sample["hidden_states_layout"].endswith("plus_last"):
            labels.append("last")
        expected = []
        for label in labels:
            rows = {}
            for p, values in records[0][label]:
                assert p.ndim == 1 and values.shape[0] == p.numel()
                for index, position in enumerate(p.tolist()):
                    assert position not in rows, (
                        "Reference forward repeated a token position"
                    )
                    rows[position] = values[index]
            expected.append(torch.stack([rows[int(p)] for p in positions]).float())
        actual = sample["hidden_states"][0].float()
        assert actual.shape == torch.cat(expected, dim=-1).shape
        reports = []
        for label, a, b in zip(
            labels, actual.split(expected[0].shape[-1], dim=-1), expected, strict=True
        ):
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            rmse = float(
                (a - b).square().mean().sqrt()
                / b.square().mean().sqrt().clamp_min(1e-8)
            )
            reports.append({"layer": label, "relative_rmse": rmse, "finite": finite})
            print("SPECO_VLLM_TARGET_REFERENCE " + json.dumps(reports[-1]), flush=True)
        (folder / "vllm-target-only-reference.json").write_text(
            json.dumps(reports, indent=2)
        )
        assert all(r["finite"] and r["relative_rmse"] < 0.03 for r in reports), reports
    finally:
        del engine
