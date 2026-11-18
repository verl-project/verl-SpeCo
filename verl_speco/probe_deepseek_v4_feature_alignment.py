# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Compare fixed actor features with dedicated native Ascend trajectory replay.

This probe never installs capture in verl rollout workers. No tolerance is
silently selected: reports describe measured differences, not model equivalence.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from verl_speco.integration.standalone_rollout_capture import (
    StandaloneCaptureWorkerExtension,
)


def encode_logits_reply(calls, head_weights):
    tensors = {f"logits/{i}": value.contiguous() for i, value in enumerate(calls)}
    tensors.update(
        {f"head/{name}": value.contiguous() for name, value in head_weights.items()}
    )
    return {"num_calls": len(calls), "tensors": save_tensors(tensors)}


def decode_logits_reply(reply):
    if not isinstance(reply.get("tensors"), bytes):
        raise TypeError("Alignment logits require binary tensor transport")
    fields = load_tensors(reply["tensors"])
    calls = [fields[f"logits/{i}"] for i in range(reply["num_calls"])]
    head = {
        name.removeprefix("head/"): value
        for name, value in fields.items()
        if name.startswith("head/")
    }
    return {"logits_calls": calls, "output_head_weights": head}


class AlignmentCaptureWorkerExtension(StandaloneCaptureWorkerExtension):
    @torch.no_grad()
    def speco_alignment_precision(self, operation):
        """Controlled in-memory experiment; source checkpoint is never written."""
        model = self.model_runner.get_model()
        parameters = dict(model.named_parameters())
        if operation == "round_fp32":
            if hasattr(self, "_speco_alignment_fp32_originals"):
                raise RuntimeError("FP32 rounding already active")
            originals = {
                name: value.detach().cpu().clone()
                for name, value in parameters.items()
                if value.dtype == torch.float32
            }
            self._speco_alignment_fp32_originals = originals
            for name in originals:
                value = parameters[name]
                value.copy_(value.bfloat16().float())
            return {
                "status": "rounded_fp32_values_to_bf16",
                "parameters": sorted(originals),
            }
        if operation == "restore":
            originals = getattr(self, "_speco_alignment_fp32_originals", {})
            for name, value in originals.items():
                parameters[name].copy_(value.to(parameters[name].device))
            if hasattr(self, "_speco_alignment_fp32_originals"):
                del self._speco_alignment_fp32_originals
            return {"status": "restored", "parameters": len(originals)}
        raise ValueError(f"Unknown precision experiment: {operation}")

    def speco_alignment_logits(self, operation):
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return None
        model = self.model_runner.get_model()
        if operation == "begin":
            if hasattr(self, "_speco_alignment_original"):
                raise RuntimeError("Alignment capture already active")
            self._speco_alignment_records = []
            self._speco_alignment_original = model.compute_logits
            self._speco_alignment_had_instance = "compute_logits" in vars(model)

            def compute_logits(*args, **kwargs):
                result = self._speco_alignment_original(*args, **kwargs)
                if result is not None:
                    self._speco_alignment_records.append(result.detach().cpu().clone())
                return result

            model.compute_logits = compute_logits
            return {"status": "started"}
        if operation == "finish":
            records = self._speco_alignment_records
            head_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.named_parameters()
                if name
                in {
                    "model.hc_head_fn",
                    "model.hc_head_base",
                    "model.hc_head_scale",
                    "model.norm.weight",
                }
            }
            if self._speco_alignment_had_instance:
                model.compute_logits = self._speco_alignment_original
            else:
                del model.compute_logits
            del self._speco_alignment_original
            self._speco_alignment_records = []
            return encode_logits_reply(records, head_weights)
        raise ValueError(f"Unknown alignment operation: {operation}")


def error_metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Alignment shapes differ: {reference.shape}, {candidate.shape}"
        )
    x, y = reference.double(), candidate.double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite alignment tensor")
    delta = y - x
    scale = x.square().mean().sqrt()
    denominator = x.norm() * y.norm()
    return {
        "shape": list(x.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(delta.norm() / x.norm()) if x.norm() else None,
        "cosine": float((x.flatten() @ y.flatten()) / denominator)
        if denominator
        else None,
        "reference_rms": float(scale),
        "different_elements": int((reference != candidate).sum()),
        "elements": x.numel(),
    }


def select_prompt_logits(calls, token_count, positions, vocab_size):
    # Dedicated one-request, unchunked prefill. Native prompt-logprob execution
    # computes either the entire prompt or its first N-1 prediction rows.
    matches = [
        x for x in calls if x.ndim == 2 and x.shape[0] in {token_count, token_count - 1}
    ]
    if len(matches) != 1 or matches[0].shape[1] != vocab_size:
        raise ValueError(
            f"Ambiguous native prompt logits: {[list(x.shape) for x in calls]}"
        )
    if int(positions.max()) >= matches[0].shape[0]:
        raise ValueError("Native logits do not cover the actor window")
    return matches[0].index_select(0, positions)


def capture(args):
    from vllm import SamplingParams

    from verl_speco.collect_deepseek_v4_features import (
        build_capture_llm,
        capture_request,
    )
    from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract

    source = torch.load(args.actor_features, weights_only=True, map_location="cpu")
    contract = DeepSeekV4Contract.from_model(args.model)
    if source["contract"] != contract.identity() or source["target_version"] != 0:
        raise ValueError("Actor and native target contracts/versions differ")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    settings = {
        "model_family": "deepseek_v4",
        "target_layer_ids": list(contract.target_layer_ids),
        "hidden_size": contract.hidden_size,
        "hidden_states_layout": contract.identity()["layout"],
        "max_rows_per_request": 512,
        "max_pending_bytes": 128 * 1024**2,
    }
    llm = build_capture_llm(
        args.model,
        contract,
        tensor_parallel_size=8,
        data_parallel_size=2,
        enable_expert_parallel=True,
        max_model_len=512,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.8,
        worker_extension_cls=__name__ + ".AlignmentCaptureWorkerExtension"
        if __name__ != "__main__"
        else "verl_speco.probe_deepseek_v4_feature_alignment.AlignmentCaptureWorkerExtension",
    )
    rows = []
    precision = None
    try:
        if args.round_fp32_weights:
            precision = llm.draft_update_rpc(
                "speco_alignment_precision", kwargs={"operation": "round_fp32"}
            )
        inputs = source["samples"] * (2 if args.repeat_capture else 1)
        for index, sample in enumerate(inputs):
            ids = sample["batch"]["input_ids"].reshape(-1)
            positions = sample["batch"]["hidden_positions"].reshape(-1)
            llm.llm_engine.collective_rpc(
                "speco_alignment_logits", kwargs={"operation": "begin"}
            )
            _, _, payload = capture_request(
                llm.llm_engine,
                ids.tolist(),
                SamplingParams(
                    temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=1
                ),
                settings,
            )
            torch.save(payload, root / f"native_features_{index}.pt")
            outputs = llm.llm_engine.collective_rpc(
                "speco_alignment_logits", kwargs={"operation": "finish"}
            )
            outputs = [x for x in outputs if x is not None]
            if len(outputs) != 1 or not torch.equal(payload["input_ids"], ids):
                raise ValueError("Native capture token IDs or TP ownership differ")
            outputs = [decode_logits_reply(outputs[0])]
            if not torch.equal(payload["positions"], torch.arange(len(ids))):
                raise ValueError("Native absolute positions differ")
            logits = select_prompt_logits(
                outputs[0]["logits_calls"], len(ids), positions, contract.vocab_size
            )
            row = {
                "input_ids": ids,
                "positions": positions,
                "auxiliary": payload["hidden_states"].index_select(0, positions),
                "pre_hc": payload["target_head_hidden_states"].index_select(
                    0, positions
                ),
                "logits": logits,
                "output_head_weights": outputs[0]["output_head_weights"],
            }
            torch.save(row, root / f"native_sample_{index}.pt")
            rows.append(
                {
                    "sample": index,
                    "source_sample": index % len(source["samples"]),
                    "tokens": len(ids),
                    "positions": positions.tolist(),
                    "native_logits_call_shapes": [
                        list(x.shape) for x in outputs[0]["logits_calls"]
                    ],
                }
            )
        (root / "capture_report.json").write_text(
            json.dumps(
                {
                    "status": "captured",
                    "contract": contract.identity(),
                    "samples": rows,
                    "scope": "dedicated_native_replay_tp8_dp2_ep_no_rl_capture",
                    "fp32_values_rounded_to_bf16": args.round_fp32_weights,
                    "precision_experiment": precision,
                },
                indent=2,
            )
        )
    finally:
        try:
            try:
                if precision is not None:
                    llm.draft_update_rpc(
                        "speco_alignment_precision", kwargs={"operation": "restore"}
                    )
            finally:
                llm.llm_engine.collective_rpc(
                    "speco_standalone_capture", kwargs={"operation": "close"}
                )
        finally:
            llm.shutdown_capture_frontends()


def compare(args):
    source = torch.load(args.actor_features, weights_only=True, map_location="cpu")
    root = Path(args.output)
    contract = source["contract"]
    width, layer_ids = contract["hidden_size"], contract["target_layer_ids"]
    report = {
        "status": "measured",
        "equivalence_validated": False,
        "samples": [],
        "scope": "actor_bf16_fsdp2_vs_native_ascend_fixed_tokens",
    }
    actor_logits = None
    if args.actor_logits:
        actor_logits = torch.load(
            args.actor_logits, weights_only=True, map_location="cpu"
        )
        if actor_logits["contract"] != contract:
            raise ValueError("Actor logits and features use different contracts")
    native_metadata = json.loads((root / "capture_report.json").read_text())
    if native_metadata["contract"] != contract:
        raise ValueError("Native capture and actor contracts differ")
    report["native_fp32_values_rounded_to_bf16"] = native_metadata.get(
        "fp32_values_rounded_to_bf16", False
    )
    for index, sample in enumerate(source["samples"]):
        native = torch.load(
            root / f"native_sample_{index}.pt", weights_only=True, map_location="cpu"
        )
        batch = sample["batch"]
        if not torch.equal(
            native["input_ids"], batch["input_ids"].reshape(-1)
        ) or not torch.equal(
            native["positions"], batch["hidden_positions"].reshape(-1)
        ):
            raise ValueError("Comparison token/position mismatch")
        features = sample["hidden_states"].reshape(
            -1, width * (len(layer_ids) + contract["hc_mult"])
        )
        result = {"sample": index, "layers": {}}
        for offset, layer in enumerate(layer_ids):
            part = slice(offset * width, (offset + 1) * width)
            result["layers"][str(layer)] = error_metrics(
                features[:, part], native["auxiliary"][:, part]
            )
        result["pre_hc"] = error_metrics(
            features[:, len(layer_ids) * width :], native["pre_hc"]
        )
        result["pre_hc_streams"] = {
            str(stream): error_metrics(
                features[:, len(layer_ids) * width :].reshape(
                    -1, contract["hc_mult"], width
                )[:, stream],
                native["pre_hc"].reshape(-1, contract["hc_mult"], width)[:, stream],
            )
            for stream in range(contract["hc_mult"])
        }
        if actor_logits is not None:
            if not torch.equal(
                actor_logits["input_ids"][index], native["input_ids"]
            ) or not torch.equal(actor_logits["positions"][index], native["positions"]):
                raise ValueError("Actor logits token/position mismatch")
            x, y = actor_logits["logits"][index].float(), native["logits"].float()
            result["logits"] = error_metrics(x, y)
            logp, logq = x.log_softmax(-1), y.log_softmax(-1)
            tv = (logp.exp() - logq.exp()).abs().sum(-1) / 2
            result["distribution"] = {
                "mean_total_variation": float(tv.mean()),
                "max_total_variation": float(tv.max()),
                "per_position_total_variation": tv.tolist(),
                "mean_kl_actor_to_native": float(
                    (logp.exp() * (logp - logq)).sum(-1).mean()
                ),
                "top1_agreement": float((x.argmax(-1) == y.argmax(-1)).float().mean()),
            }
        report["samples"].append(result)
    report["native_actor_logits_compared"] = actor_logits is not None
    (root / "alignment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["capture", "compare"])
    parser.add_argument("--model")
    parser.add_argument("--actor-features", required=True)
    parser.add_argument("--actor-logits")
    parser.add_argument(
        "--repeat-capture",
        action="store_true",
        help="Replay each fixed input twice to measure native repeatability",
    )
    parser.add_argument(
        "--round-fp32-weights",
        action="store_true",
        help="Dedicated diagnostic engine only: round FP32 parameter values through BF16, then restore",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "capture":
        if not args.model:
            parser.error("capture requires --model")
        capture(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
