"""Inspect SPECO standalone draft feature stores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch

from verl_speco.trainer.feature_store import MANIFEST_NAME


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a torch_shard draft feature store."
    )
    parser.add_argument(
        "path",
        help="Feature store directory containing manifest.jsonl and shard .pt files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Maximum number of samples to inspect.",
    )
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Print valid samples as well as invalid samples.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with code 1 when invalid samples are found.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature store manifest: {manifest_path}")

    entries = _load_manifest(manifest_path)
    shape_counts: Counter[str] = Counter()
    inspected = 0
    invalid = 0

    print(f"feature_store={root}")
    print(f"manifest_entries={len(entries)}")

    for entry in entries:
        if inspected >= args.max_samples:
            break
        shard_name = str(entry.get("path"))
        shard_path = root / shard_name
        shard = _torch_load(shard_path)
        samples = shard.get("samples") or []
        for sample_index, sample in enumerate(samples):
            if inspected >= args.max_samples:
                break
            inspected += 1
            key = f"{shard_name}:{sample_index}"
            issues = _sample_issues(sample)
            summary = _sample_summary(sample)
            shape_counts.update(summary.values())
            if issues:
                invalid += 1
                print(f"[BAD] {key} {summary}")
                for issue in issues:
                    print(f"  - {issue}")
            elif args.show_ok:
                print(f"[OK]  {key} {summary}")

    print(f"inspected_samples={inspected}")
    print(f"invalid_samples={invalid}")
    if shape_counts:
        print("shape_counts:")
        for shape, count in shape_counts.most_common():
            print(f"  {shape}: {count}")
    return 1 if invalid and args.strict_exit else 0


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_shape(item) for item in value) + "]"
    return type(value).__name__


def _tensor(value: Any, name: str, issues: list[str]) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        issues.append(f"{name} is not a tensor: {type(value).__name__}")
        return None
    return value


def _sample_summary(sample: dict[str, Any]) -> dict[str, str]:
    keys = [
        "input_ids",
        "loss_mask",
        "hidden_states",
        "last_hidden_states",
        "target_logprobs",
        "position_ids",
    ]
    return {key: _shape(sample[key]) for key in keys if key in sample}


def _sample_issues(sample: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    input_ids = _tensor(sample.get("input_ids"), "input_ids", issues)
    loss_mask = _tensor(sample.get("loss_mask"), "loss_mask", issues)
    hidden_states = sample.get("hidden_states")
    position_ids = sample.get("position_ids")
    target_logprobs = sample.get("target_logprobs")

    seq_len = None
    if input_ids is not None:
        if input_ids.dim() > 2 or (input_ids.dim() == 2 and 1 not in input_ids.shape):
            issues.append(
                f"input_ids should be 1D or singleton-2D, got {_shape(input_ids)}"
            )
        seq_len = int(input_ids.numel())
    if loss_mask is not None:
        if loss_mask.numel() != seq_len:
            issues.append(
                f"loss_mask length {loss_mask.numel()} does not match input_ids length {seq_len}"
            )
    if torch.is_tensor(hidden_states):
        hidden_states = cast(torch.Tensor, hidden_states)
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_len = int(hidden_states.size(1))
        elif hidden_states.dim() == 2:
            hidden_len = int(hidden_states.size(0))
        else:
            hidden_len = -1
            issues.append(
                f"hidden_states should be [seq, hidden] or [1, seq, hidden], got {_shape(hidden_states)}"
            )
        if seq_len is not None and hidden_len >= 0 and hidden_len < max(seq_len - 1, 1):
            issues.append(
                f"hidden_states length {hidden_len} is too short for input_ids length {seq_len}"
            )
    elif isinstance(hidden_states, (list, tuple)):
        for idx, tensor in enumerate(hidden_states):
            if not torch.is_tensor(tensor):
                issues.append(
                    f"hidden_states[{idx}] is not a tensor: {type(tensor).__name__}"
                )
            elif tensor.dim() not in {2, 3}:
                issues.append(
                    f"hidden_states[{idx}] has unexpected shape {_shape(tensor)}"
                )
    else:
        issues.append(
            f"hidden_states is not a tensor/list: {type(hidden_states).__name__}"
        )

    if position_ids is not None:
        if not torch.is_tensor(position_ids):
            issues.append(
                f"position_ids is not a tensor: {type(position_ids).__name__}"
            )
        else:
            position_ids = cast(torch.Tensor, position_ids)
            if position_ids.numel() != seq_len:
                issues.append(
                    f"position_ids length {position_ids.numel()} does not match input_ids length {seq_len}"
                )
            elif position_ids.dim() > 1:
                issues.append(
                    f"position_ids is normalizable but not stored as 1D: {_shape(position_ids)}"
                )
    if target_logprobs is not None:
        if not torch.is_tensor(target_logprobs):
            issues.append(
                f"target_logprobs is not a tensor: {type(target_logprobs).__name__}"
            )
        else:
            target_logprobs = cast(torch.Tensor, target_logprobs)
            normalized = target_logprobs
            while normalized.dim() > 3 and normalized.size(0) == 1:
                normalized = normalized.squeeze(0)
            if normalized.dim() != 3 or normalized.size(-1) < 2:
                issues.append(
                    f"target_logprobs should be [rows, topk, 2], got {_shape(target_logprobs)}"
                )
            elif target_logprobs.dim() > 3:
                issues.append(
                    f"target_logprobs is normalizable but not stored as 3D: {_shape(target_logprobs)}"
                )
            elif seq_len is not None and int(normalized.size(0)) < max(seq_len - 2, 1):
                issues.append(
                    f"target_logprobs rows {int(normalized.size(0))} may be too short for input_ids length {seq_len}"
                )

    return issues


if __name__ == "__main__":
    raise SystemExit(main())
