# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inspect JSONL draft-training samples without loading model dependencies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_REPLAY_REQUIRED_KEYS = {
    "input_ids",
    "loss_mask",
    "attention_mask",
    "position_ids",
    "feature_positions",
    "draft_position_ids",
}
FEATURE_REQUIRED_KEYS = {"input_ids", "loss_mask", "hidden_states"}
INPUT_LOSS_REQUIRED_KEYS = {"input_ids", "loss_mask"}
VLLM_SAFETENSORS_MANIFEST_KEYS = {"path", "num_samples", "sample"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a JSONL file and report whether it matches SPECO sample schemas."
    )
    parser.add_argument("path", help="JSONL file to inspect.")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=20,
        help="Maximum number of JSONL rows to inspect.",
    )
    parser.add_argument(
        "--show-first",
        action="store_true",
        help="Print the first JSON object with long arrays summarized.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with code 1 when inspected rows do not match a known schema.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    summaries: list[dict[str, Any]] = []
    key_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    issues_by_schema: dict[str, list[str]] = defaultdict(list)
    first_obj: dict[str, Any] | None = None

    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if len(summaries) >= int(args.max_lines):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                summaries.append({"line": line_number, "schema": "invalid_json"})
                schema_counts["invalid_json"] += 1
                issues_by_schema["invalid_json"].append(f"line {line_number}: {exc}")
                continue
            if first_obj is None and isinstance(obj, dict):
                first_obj = obj
            if not isinstance(obj, dict):
                summaries.append({"line": line_number, "schema": type(obj).__name__})
                schema_counts[type(obj).__name__] += 1
                continue
            keys = set(obj)
            key_counts.update(keys)
            schema, issues = _classify(obj)
            schema_counts[schema] += 1
            issues_by_schema[schema].extend(
                f"line {line_number}: {issue}" for issue in issues
            )
            summaries.append(
                {
                    "line": line_number,
                    "schema": schema,
                    "keys": sorted(keys),
                    "shapes": _shape_summary(obj),
                }
            )

    print(f"jsonl={path}")
    print(f"inspected_lines={len(summaries)}")
    print("schema_counts:")
    for schema, count in schema_counts.most_common():
        print(f"  {schema}: {count}")
    print("key_counts:")
    for key, count in key_counts.most_common():
        print(f"  {key}: {count}")
    print("sample_summaries:")
    for summary in summaries[: min(len(summaries), 5)]:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if issues_by_schema:
        print("issues:")
        for schema, issues in issues_by_schema.items():
            for issue in issues[:10]:
                print(f"  [{schema}] {issue}")
    if args.show_first and first_obj is not None:
        print("first_object:")
        print(
            json.dumps(
                _compact_json(first_obj), ensure_ascii=False, indent=2, sort_keys=True
            )
        )

    invalid = any(
        schema
        not in {
            "token_replay_jsonl",
            "feature_jsonl",
            "input_loss_jsonl",
            "vllm_safetensors_manifest",
        }
        for schema in schema_counts
    )
    return 1 if invalid and args.strict_exit else 0


def _classify(obj: dict[str, Any]) -> tuple[str, list[str]]:
    keys = set(obj)
    if VLLM_SAFETENSORS_MANIFEST_KEYS.issubset(keys):
        return "vllm_safetensors_manifest", []
    if TOKEN_REPLAY_REQUIRED_KEYS.issubset(keys):
        return "token_replay_jsonl", _token_replay_issues(obj)
    if FEATURE_REQUIRED_KEYS.issubset(keys):
        return "feature_jsonl", _feature_issues(obj)
    if INPUT_LOSS_REQUIRED_KEYS.issubset(keys):
        return "input_loss_jsonl", _input_loss_issues(obj)
    missing_token = sorted(TOKEN_REPLAY_REQUIRED_KEYS - keys)
    missing_feature = sorted(FEATURE_REQUIRED_KEYS - keys)
    return (
        "unknown",
        [
            f"missing token_replay keys={missing_token}",
            f"missing feature keys={missing_feature}",
        ],
    )


def _token_replay_issues(obj: dict[str, Any]) -> list[str]:
    issues = []
    input_len = _flat_len(obj.get("input_ids"))
    for key in ("loss_mask", "attention_mask", "position_ids"):
        value_len = _flat_len(obj.get(key))
        if input_len is not None and value_len != input_len:
            issues.append(f"{key} length {value_len} != input_ids length {input_len}")
    feature_len = _flat_len(obj.get("feature_positions"))
    draft_len = _flat_len(obj.get("draft_position_ids"))
    if feature_len is not None and draft_len != feature_len:
        issues.append(
            f"draft_position_ids length {draft_len} != feature_positions length {feature_len}"
        )
    return issues


def _feature_issues(obj: dict[str, Any]) -> list[str]:
    issues = []
    input_len = _flat_len(obj.get("input_ids"))
    loss_len = _flat_len(obj.get("loss_mask"))
    if input_len is not None and loss_len != input_len:
        issues.append(f"loss_mask length {loss_len} != input_ids length {input_len}")
    hidden_shape = _shape(obj.get("hidden_states"))
    if not hidden_shape or hidden_shape[0] in {"scalar", "dict", "str", "none"}:
        issues.append("hidden_states is not an array-like value")
    return issues


def _input_loss_issues(obj: dict[str, Any]) -> list[str]:
    issues = []
    input_len = _flat_len(obj.get("input_ids"))
    loss_len = _flat_len(obj.get("loss_mask"))
    if input_len is None:
        issues.append("input_ids is not a JSON list")
    if loss_len is None:
        issues.append("loss_mask is not a JSON list")
    if input_len is not None and loss_len is not None and loss_len != input_len:
        issues.append(f"loss_mask length {loss_len} != input_ids length {input_len}")
    return issues


def _shape_summary(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _shape(value)
        for key, value in obj.items()
        if key
        in {
            "input_ids",
            "loss_mask",
            "attention_mask",
            "position_ids",
            "feature_positions",
            "draft_position_ids",
            "hidden_states",
            "target_logprobs",
            "sample",
            "path",
        }
    }


def _shape(value: Any) -> list[Any]:
    if value is None:
        return ["none"]
    if isinstance(value, dict):
        return ["dict", sorted(value)[:20]]
    if isinstance(value, str):
        return ["str", len(value)]
    if not isinstance(value, list):
        return ["scalar", type(value).__name__]
    shape = []
    current: Any = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else None
    return shape


def _flat_len(value: Any) -> int | None:
    if not isinstance(value, list):
        return None
    current = value
    while isinstance(current, list) and len(current) == 1:
        current = current[0]
    return len(current) if isinstance(current, list) else None


def _compact_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact_json(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 16:
            return {
                "__list__": True,
                "shape": _shape(value),
                "head": [_compact_json(item) for item in value[:4]],
                "tail": [_compact_json(item) for item in value[-4:]],
            }
        return [_compact_json(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
