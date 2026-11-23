# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Teacher-force a saved sequence through a non-speculative vLLM engine."""
import argparse
import json
import os
from pathlib import Path


def run(folder, target=None):
    # Set these before importing vLLM. Run in a fresh process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    import torch
    from vllm import LLM, SamplingParams
    from test_vllm_rollout_capture_gpu import install_forward_audit, collect_forward_audit

    folder = Path(folder)
    metadata = json.loads((folder / "tokens.json").read_text())
    sample = torch.load(folder / "captured.pt", map_location="cpu", weights_only=True)
    prompt, response = metadata["prompt"], metadata["response"]
    assert response and prompt
    sequence = prompt + response
    target = target or metadata["target"]
    assert target == metadata["target"], "Use the exact saved target checkpoint"
    assert torch.cuda.is_available()

    def install(model):
        import types
        install_forward_audit(model)
        model._incremental_original_logits = model.compute_logits
        model._incremental_forced = []
        original = model.compute_logits

        def logits(self, *args, **kwargs):
            result = original(*args, **kwargs)
            assert result is not None and result.ndim == 2 and result.shape[0] == 1
            record = self._speco_audit[-1]
            next_position = int(record["positions"].max()) + 1
            token = sequence[next_position]
            # Only replace sampling logits; target hidden states and KV writes
            # have already been computed from the fixed input sequence.
            result.fill_(float("-inf"))
            result[:, token] = 0
            self._incremental_forced.append(token)
            return result

        model.compute_logits = types.MethodType(logits, model)
        return True

    def collect(model):
        model.compute_logits = model._incremental_original_logits
        del model._incremental_original_logits
        return {"records": collect_forward_audit(model),
                "forced": model._incremental_forced}

    engine = LLM(model=target, dtype="bfloat16", tensor_parallel_size=1,
                 enforce_eager=True, enable_prefix_caching=False,
                 async_scheduling=False, gpu_memory_utilization=0.5,
                 max_model_len=max(512, len(sequence) + 1))
    try:
        engine.apply_model(install)
        output = engine.generate([{"prompt_token_ids": prompt}],
                                 SamplingParams(temperature=0, max_tokens=len(response),
                                                ignore_eos=True))[0]
        assert list(output.outputs[0].token_ids) == response, "Forced sequence was not honored"
        result = engine.apply_model(collect)
        assert len(result) == 1
        records = result[0]["records"]
        assert result[0]["forced"] == response
        # Fail closed if vLLM did not execute true one-token cached decoding.
        assert records[0]["positions"].tolist() == list(range(len(prompt)))
        assert len(records) == len(response)
        for step, record in enumerate(records[1:]):
            position = len(prompt) + step
            assert record["positions"].tolist() == [position]
            assert record["tokens"].tolist() == [sequence[position]]
        torch.save(records, folder / "ordinary-incremental.pt")
        labels = [str(x) for x in sample["hidden_layer_ids"]]
        if sample["hidden_states_layout"].endswith("plus_last"):
            labels.append("last")
        positions = sample["hidden_positions"][0].tolist()
        rows = {}
        for record in records:
            for index, position in enumerate(record["positions"].tolist()):
                assert position not in rows
                rows[position] = (record, index)
        actual = sample["hidden_states"][0].float()
        width = actual.shape[-1] // len(labels)
        reports = []
        for part, label in enumerate(labels):
            expected = torch.stack([rows[p][0]["layers"][label][rows[p][1]]
                                    for p in positions]).float()
            observed = actual[:, part * width:(part + 1) * width]
            relative = (observed-expected).square().mean(-1).sqrt() / expected.square().mean(-1).sqrt().clamp_min(1e-8)
            finite = bool(torch.isfinite(observed).all() and torch.isfinite(expected).all())
            bad = ((relative >= 0.03) | ~torch.isfinite(relative)).nonzero().flatten().tolist()
            worst = relative.nan_to_num(nan=float("inf")).topk(min(10,len(positions))).indices.tolist()
            report = {"layer": label, "finite": finite,
                      "relative_rmse": float((observed-expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)) if finite else None,
                      "rows_over_3pct": len(bad),
                      "first_bad_position": positions[bad[0]] if bad else None,
                      "worst": [{"position": positions[i], "input_token": sequence[positions[i]],
                                 "relative_rmse": float(relative[i]) if torch.isfinite(relative[i]) else None} for i in worst]}
            reports.append(report)
            print("SPECO_ORDINARY_INCREMENTAL " + json.dumps(report), flush=True)
        (folder / "ordinary-incremental.json").write_text(json.dumps(reports, indent=2))
        print("Diagnostic completed; inspect JSON for numerical mismatches.", flush=True)
    finally:
        del engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--target")
    args = parser.parse_args()
    run(args.artifact, args.target)
