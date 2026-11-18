# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Existing trajectory inputs with SpeCo supervision and no regeneration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from verl_speco.producer.input_reader import _record_from_payload, tokenize_record
from verl_speco.producer.trajectory_mask import tokenize_role_preserving_messages


@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    input_ids: list[int]
    loss_mask: list[float]
    line_number: int
    source_file: str | None = None


def _ids(value, name):
    if (
        not isinstance(value, list)
        or not value
        or any(type(token) is not int for token in value)
    ):
        raise ValueError(f"{name} must be a nonempty flat list of integer token IDs")
    return value


def normalize_trajectory(payload, tokenizer, *, line_number, vocab_size, max_model_len):
    if not isinstance(payload, dict):
        raise TypeError("Trajectory must be a JSON object")
    token_modes = int("input_ids" in payload) + int(
        "prompt_token_ids" in payload or "response_token_ids" in payload
    )
    if token_modes > 1:
        raise ValueError("Choose input_ids or split prompt/response token IDs")
    if token_modes and any(
        key in payload
        for key in (
            "prompt",
            "response",
            "messages",
            "conversation",
            "conversations",
            "chat",
        )
    ):
        raise ValueError(
            "Choose one trajectory representation; mixed text and token fields are ambiguous"
        )
    sample_id = payload.get(
        "sample_id", payload.get("id", f"trajectory-{line_number:08d}")
    )
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id/id must be a nonempty string")
    role_preserving = (
        payload.get("metadata", {}).get("schema") == "alpaca-role-preserving-v1"
        if isinstance(payload.get("metadata"), dict)
        else False
    )
    if role_preserving and not token_modes:
        if "loss_mask" in payload:
            raise ValueError(
                "Role-preserving messages derive their mask; use input_ids for an explicit mask"
            )
        ids, mask = tokenize_role_preserving_messages(
            payload.get("messages"), tokenizer
        )
    elif "input_ids" in payload:
        ids = _ids(payload["input_ids"], "input_ids")
        if "loss_mask" not in payload:
            raise ValueError(
                "input_ids requires an explicit SpeCo loss_mask; inverted masks are not accepted"
            )
        mask = payload["loss_mask"]
    elif token_modes:
        prompt = _ids(payload.get("prompt_token_ids"), "prompt_token_ids")
        response = _ids(payload.get("response_token_ids"), "response_token_ids")
        ids = prompt + response
        mask = payload.get("loss_mask", [0] * len(prompt) + [1] * len(response))
    elif "chat" in payload:
        if not isinstance(payload["chat"], str) or not payload["chat"]:
            raise ValueError("chat must be the complete serialized trajectory string")
        if "loss_mask" not in payload:
            raise ValueError("A serialized chat requires an explicit SpeCo loss_mask")
        ids = _ids(
            tokenizer.encode(payload["chat"], add_special_tokens=True), "encoded chat"
        )
        mask = payload["loss_mask"]
    else:
        if "loss_mask" in payload:
            raise ValueError(
                "Explicit loss_mask requires input_ids, split token IDs, or serialized chat"
            )
        normalized = dict(payload)
        if "messages" in normalized:
            if (
                "conversation" in normalized
                or "conversations" in normalized
                or "prompt" in normalized
            ):
                raise ValueError("Choose one conversation field")
            normalized["conversation"] = normalized.pop("messages")
        record = _record_from_payload(
            normalized, f"line {line_number}", line_number - 1, True
        )
        original = normalized.get("conversation", normalized.get("conversations"))
        if original is not None and (
            not isinstance(record.prompt, tuple)
            or len(record.prompt) + 1 != len(original)
        ):
            raise ValueError(
                "Conversation roles were altered during parsing; provide exact token IDs instead"
            )
        request = tokenize_record(record, tokenizer, {"preserve_prompt_context": True})
        ids = request.input_ids.tolist()
        mask = request.loss_mask.tolist()
    ids = _ids(ids, "input_ids")
    if any(not 0 <= token < vocab_size for token in ids):
        raise ValueError("Trajectory token ID is outside the target vocabulary")
    if len(ids) < 3 or len(ids) + 1 > max_model_len:
        raise ValueError(
            "Trajectory needs at least three tokens and one free sampling slot; increase max-model-len rather than truncating"
        )
    if (
        not isinstance(mask, list)
        or len(mask) != len(ids)
        or any(value not in (0, 1) for value in mask)
    ):
        raise ValueError("SpeCo loss_mask must match input_ids and contain only 0/1")
    if not any(mask):
        raise ValueError("Trajectory has no supervised tokens")
    return TrajectoryRecord(
        sample_id, ids, [float(value) for value in mask], line_number
    )


def iter_trajectories(path, tokenizer, *, vocab_size, max_model_len, max_samples=None):
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    count = 0
    seen = set()
    selected = Path(path)
    files = sorted(selected.glob("*.jsonl")) if selected.is_dir() else [selected]
    if not files:
        raise ValueError(f"No JSONL trajectories found in {path}")
    for source in files:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if (
                        selected.is_dir()
                        and isinstance(payload, dict)
                        and "sample_id" not in payload
                        and "id" not in payload
                    ):
                        payload["sample_id"] = f"{source.name}:{line_number}"
                    record = normalize_trajectory(
                        payload,
                        tokenizer,
                        line_number=line_number,
                        vocab_size=vocab_size,
                        max_model_len=max_model_len,
                    )
                    record = TrajectoryRecord(
                        record.sample_id,
                        record.input_ids,
                        record.loss_mask,
                        line_number,
                        str(source),
                    )
                    if record.sample_id in seen:
                        raise ValueError(f"Duplicate trajectory ID: {record.sample_id}")
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(
                        f"Invalid trajectory at {source}:{line_number}: {error}"
                    ) from error
                seen.add(record.sample_id)
                count += 1
                yield record
                if max_samples is not None and count >= max_samples:
                    return


def prepare_trajectories(records, output, *, metadata=None):
    """Write canonical tokens/masks using CPU only, without touching the input."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pending = output / "trajectories.jsonl.incomplete"
    summaries = []
    with pending.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "input_ids": record.input_ids,
                        "loss_mask": record.loss_mask,
                        "metadata": {
                            "source_file": record.source_file,
                            "source_line": record.line_number,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            summaries.append(
                {
                    "sample_id": record.sample_id,
                    "source_file": record.source_file,
                    "source_line": record.line_number,
                    "tokens": len(record.input_ids),
                    "supervised_tokens": int(sum(record.loss_mask)),
                }
            )
    if not summaries:
        raise ValueError("No trajectories were prepared")
    pending.replace(output / "trajectories.jsonl")
    report = {
        "status": "passed",
        "samples": summaries,
        "sample_count": len(summaries),
        "max_tokens": max(s["tokens"] for s in summaries),
        "metadata": metadata or {},
        "npu_forward_validated": False,
    }
    (output / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    return report
