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
"""Streaming verl/JSONL input and token preparation for the Producer."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


@dataclass(frozen=True)
class InputRecord:
    sequence_no: int
    sample_id: str
    prompt: str | tuple[dict[str, str], ...]
    response: str | None
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class TokenizedRequest:
    sequence_no: int
    sample_id: str
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    feature_positions: torch.Tensor
    draft_position_ids: torch.Tensor
    source_metadata: dict[str, Any]
    vllm_prompt_token_ids: tuple[int, ...]

    @property
    def prompt_token_ids(self) -> list[int]:
        return list(self.vllm_prompt_token_ids)


@dataclass(frozen=True)
class GenerationRequest:
    """One prompt-only row that needs target-model response generation."""

    sequence_no: int
    sample_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    source_metadata: dict[str, Any]


def iter_input_records(path: str | os.PathLike[str]) -> Iterator[InputRecord]:
    """Yield strict prompt/response records from one JSONL or Parquet file."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Producer input file not found: {input_path}")

    sequence_no = 0
    for location, payload in _iter_payloads(input_path):
        if not isinstance(payload, dict):
            raise ValueError(
                f"Producer input at {location} must be a JSON-style object"
            )
        prompt_value, response = _prompt_response_from_payload(payload, location)
        prompt = _normalize_prompt(prompt_value, location)
        if response is not None and (not isinstance(response, str) or not response):
            raise ValueError(
                f"Producer input at {location} field 'response' must be a non-empty "
                "string when present"
            )
        sample_id = payload.get("sample_id") or _extra_info_index(payload)
        if sample_id is None:
            sample_id = f"train-{sequence_no:06d}"
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Producer input at {location} has invalid sample_id")
        source_metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {"prompt", "response", "conversation", "conversations", "sample_id"}
        }
        yield InputRecord(
            sequence_no=sequence_no,
            sample_id=sample_id,
            prompt=prompt,
            response=response,
            source_metadata=source_metadata,
        )
        sequence_no += 1


def _prompt_response_from_payload(
    payload: Mapping[str, Any], location: str
) -> tuple[Any, Any]:
    """Normalize supported row schemas to Producer ``prompt``/``response``."""

    if "prompt" in payload:
        return payload.get("prompt"), payload.get("response")

    conversation = payload.get("conversation")
    if conversation is not None:
        messages = _normalize_conversation_messages(
            conversation,
            location,
            role_key="role",
            content_key="content",
        )
        return _split_final_assistant(messages)

    conversations = payload.get("conversations")
    if conversations is not None:
        messages = _normalize_conversation_messages(
            conversations,
            location,
            role_key="from",
            content_key="value",
        )
        return _split_final_assistant(messages)

    return None, payload.get("response")


def _normalize_conversation_messages(
    value: Any,
    location: str,
    *,
    role_key: str,
    content_key: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Producer input at {location} conversation must be non-empty")
    role_mapping = {"human": "user", "gpt": "assistant"}
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Producer input at {location} conversation item {index} must be an object"
            )
        role = item.get(role_key)
        content = item.get(content_key)
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"Producer input at {location} conversation item {index} requires "
                f"string field {role_key!r}"
            )
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"Producer input at {location} conversation item {index} requires "
                f"non-empty string field {content_key!r}"
            )
        messages.append(
            {"role": role_mapping.get(role.strip().lower(), role), "content": content}
        )
    return tuple(messages)


def _split_final_assistant(
    messages: tuple[dict[str, str], ...],
) -> tuple[tuple[dict[str, str], ...], str | None]:
    if messages[-1]["role"] != "assistant":
        return messages, None
    prompt = messages[:-1]
    if not prompt:
        raise ValueError("Conversation cannot contain only an assistant response")
    return prompt, messages[-1]["content"]


def _normalize_prompt(value: Any, location: str) -> str | tuple[dict[str, str], ...]:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        messages: list[dict[str, str]] = []
        for index, message in enumerate(value):
            if not isinstance(message, Mapping):
                raise ValueError(
                    f"Producer input at {location} prompt message {index} must be "
                    "an object"
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"Producer input at {location} prompt message {index} requires "
                    "string field 'role'"
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"Producer input at {location} prompt message {index} requires "
                    "string field 'content'"
                )
            messages.append({"role": role, "content": content})
        return tuple(messages)
    raise ValueError(
        f"Producer input at {location} requires 'prompt' as a string or "
        "chat-message list"
    )


def _extra_info_index(payload: Mapping[str, Any]) -> str | None:
    extra_info = payload.get("extra_info")
    if not isinstance(extra_info, Mapping):
        return None
    value = extra_info.get("index")
    return value if isinstance(value, str) and value else None


def _iter_payloads(input_path: Path) -> Iterator[tuple[str, Any]]:
    if _is_parquet(input_path):
        yield from _iter_parquet_payloads(input_path)
        return
    yield from _iter_jsonl_payloads(input_path)


def _is_parquet(input_path: Path) -> bool:
    if input_path.suffix.lower() in {".parquet", ".pq"}:
        return True
    with input_path.open("rb") as input_file:
        return input_file.read(4) == b"PAR1"


def _iter_jsonl_payloads(input_path: Path) -> Iterator[tuple[str, Any]]:
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON object at {input_path}:{line_number}: {exc.msg}"
                    ) from exc
                yield f"{input_path}:{line_number}", payload
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Producer input {input_path} is not UTF-8 JSONL or a Parquet file"
        ) from exc


def _iter_parquet_payloads(input_path: Path) -> Iterator[tuple[str, Any]]:
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeError(
            "Reading a Parquet training file requires pyarrow; install the normal "
            "verl data dependencies in the training environment"
        ) from exc

    parquet_file = parquet.ParquetFile(input_path)
    row_number = 0
    for batch in parquet_file.iter_batches():
        for payload in batch.to_pylist():
            row_number += 1
            yield f"{input_path}:row {row_number}", payload


def build_loss_mask(input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    sequence_length = int(input_ids.numel())
    if prompt_length < 0 or prompt_length > sequence_length:
        raise ValueError(
            f"prompt_length must be within [0, {sequence_length}], got {prompt_length}"
        )
    mask = torch.ones(sequence_length, dtype=torch.float32)
    mask[:prompt_length] = 0
    return mask


def tokenize_record(
    record: InputRecord,
    tokenizer: Any,
    config: Mapping[str, Any] | Any,
) -> TokenizedRequest:
    """Tokenize one row that already contains a response."""

    if record.response is None:
        raise ValueError(
            f"Producer sample {record.sample_id!r} has no response; prepare it for "
            "target-model generation instead"
        )
    prompt_ids = _prompt_ids(record.prompt, tokenizer)
    if isinstance(record.prompt, str):
        full_ids = _token_ids(
            tokenizer(record.prompt + record.response, add_special_tokens=False)
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            # Some tokenizers merge text across the prompt/response boundary.
            # Keep that boundary explicit so the response-only loss mask and the
            # exact token IDs sent to vLLM remain aligned.
            response_ids = _token_ids(
                tokenizer(record.response, add_special_tokens=False)
            )
            full_ids = [*prompt_ids, *response_ids]
    else:
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                [*record.prompt, {"role": "assistant", "content": record.response}],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            prompt_ids, full_ids = _tokenize_chat_response_with_explicit_boundary(
                record.prompt,
                record.response,
                tokenizer,
            )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Producer sample {record.sample_id!r} has an unstable tokenizer boundary "
            "between prompt and response; prompt token IDs are not a prefix of full "
            "token IDs"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            f"Producer sample {record.sample_id!r} produced no response tokens"
        )
    return _build_tokenized_request(
        sequence_no=record.sequence_no,
        sample_id=record.sample_id,
        prompt_length=len(prompt_ids),
        full_ids=full_ids,
        source_metadata=record.source_metadata,
        config=config,
    )


def prepare_generation_request(
    record: InputRecord,
    tokenizer: Any,
    config: Mapping[str, Any] | Any,
) -> GenerationRequest:
    """Tokenize a prompt-only row and bound target-model generation."""

    if record.response is not None:
        raise ValueError(
            f"Producer sample {record.sample_id!r} already contains a response"
        )
    prompt_ids = _prompt_ids(record.prompt, tokenizer)
    max_sequence_length = int(_config_value(config, "max_sequence_length", 0) or 0)
    max_tokens = int(_config_value(config, "generation_max_tokens", 0) or 0)
    if max_tokens <= 0:
        raise ValueError("generation_max_tokens must be positive")
    if max_sequence_length > 0:
        max_tokens = min(max_tokens, max_sequence_length - len(prompt_ids))
    if max_tokens <= 0:
        raise ValueError(
            f"Producer sample {record.sample_id!r} prompt has {len(prompt_ids)} tokens "
            "and leaves no generation capacity within "
            f"max_sequence_length={max_sequence_length}"
        )
    return GenerationRequest(
        sequence_no=record.sequence_no,
        sample_id=record.sample_id,
        prompt_token_ids=tuple(prompt_ids),
        max_tokens=max_tokens,
        source_metadata=dict(record.source_metadata),
    )


def finalize_generated_request(
    request: GenerationRequest,
    hidden_state_token_ids: Any,
    config: Mapping[str, Any] | Any,
    *,
    expected_response_token_ids: Any | None = None,
) -> TokenizedRequest:
    """Build a training request from vLLM generation and hidden-state tokens."""

    hidden_ids = _token_ids(hidden_state_token_ids)
    prompt_ids = list(request.prompt_token_ids)
    if hidden_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"vLLM hidden-state token sequence for sample {request.sample_id!r} "
            "does not "
            "start with the rendered prompt token IDs"
        )
    if expected_response_token_ids is not None:
        response_ids = _token_ids(expected_response_token_ids)
        full_ids = [*prompt_ids, *response_ids]
        # ExampleHiddenStatesConnector deliberately excludes the final sampled
        # token: that token was emitted by the model but was never fed through a
        # subsequent forward pass, so no hidden state exists for it.
        expected_hidden_ids = full_ids[:-1]
        if hidden_ids != expected_hidden_ids:
            raise ValueError(
                f"vLLM hidden-state token IDs for sample {request.sample_id!r} do not "
                "match the prompt plus generated completion excluding its final token "
                f"(hidden={len(hidden_ids)}, expected={len(expected_hidden_ids)}, "
                f"completion={len(response_ids)})"
            )
    else:
        # Prefilled records already contain their complete response, and their
        # caller supplies the full token sequence directly.
        full_ids = hidden_ids
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            f"vLLM generated no response tokens for sample {request.sample_id!r}; "
            "the hidden-state server must enable include_output_tokens"
        )
    return _build_tokenized_request(
        sequence_no=request.sequence_no,
        sample_id=request.sample_id,
        prompt_length=len(prompt_ids),
        full_ids=full_ids,
        source_metadata=request.source_metadata,
        config=config,
        vllm_prompt_token_ids=hidden_ids,
        feature_end_limit=len(hidden_ids),
    )


def prepare_generated_prefill_request(
    request: GenerationRequest,
    response_token_ids: Any,
    config: Mapping[str, Any] | Any,
) -> TokenizedRequest:
    """Build the full-sequence prefill request after target generation.

    The final sampled token has not itself passed through a model forward, so
    target features are requested for ``prompt + completion[:-1]`` while the
    full completion remains in ``input_ids`` as the next-token label sequence.
    This matches the existing non-TQ vLLM replay path and does not require the
    connector to capture decode-step hidden states.
    """

    response_ids = _token_ids(response_token_ids)
    if not response_ids:
        raise ValueError(
            f"vLLM generated no response tokens for sample {request.sample_id!r}"
        )
    prompt_ids = list(request.prompt_token_ids)
    full_ids = [*prompt_ids, *response_ids]
    hidden_input_ids = full_ids[:-1]
    return _build_tokenized_request(
        sequence_no=request.sequence_no,
        sample_id=request.sample_id,
        prompt_length=len(prompt_ids),
        full_ids=full_ids,
        source_metadata=request.source_metadata,
        config=config,
        vllm_prompt_token_ids=hidden_input_ids,
        feature_end_limit=len(hidden_input_ids),
    )


def _prompt_ids(prompt: str | tuple[dict[str, str], ...], tokenizer: Any) -> list[int]:
    if isinstance(prompt, str):
        return _token_ids(tokenizer(prompt, add_special_tokens=False))
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError(
            "Chat-message prompts require a tokenizer with apply_chat_template()"
        )
    return _token_ids(
        apply_chat_template(
            list(prompt),
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def _tokenize_chat_response_with_explicit_boundary(
    prompt: tuple[dict[str, str], ...],
    response: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Render a chat response while keeping its loss boundary deterministic.

    Qwen-family templates may render a generation prompt differently from an
    existing assistant message (for example by inserting a thinking preamble).
    A marker lets us retain the template's assistant header and suffix while
    tokenizing the response as a separate loss-bearing region.
    """

    marker = "__VERL_SPECO_ASSISTANT_RESPONSE_BOUNDARY_8F7C2D91__"
    while marker in response or any(marker in message["content"] for message in prompt):
        marker += "_"
    rendered = tokenizer.apply_chat_template(
        [*prompt, {"role": "assistant", "content": marker}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or rendered.count(marker) != 1:
        raise ValueError(
            "Tokenizer chat template did not preserve the assistant response marker"
        )
    prompt_text, suffix_text = rendered.split(marker, 1)
    explicit_prompt_ids = _token_ids(tokenizer(prompt_text, add_special_tokens=False))
    response_ids = _token_ids(tokenizer(response, add_special_tokens=False))
    suffix_ids = _token_ids(tokenizer(suffix_text, add_special_tokens=False))
    return explicit_prompt_ids, [*explicit_prompt_ids, *response_ids, *suffix_ids]


def _build_tokenized_request(
    *,
    sequence_no: int,
    sample_id: str,
    prompt_length: int,
    full_ids: list[int],
    source_metadata: Mapping[str, Any],
    config: Mapping[str, Any] | Any,
    vllm_prompt_token_ids: list[int] | None = None,
    feature_end_limit: int | None = None,
) -> TokenizedRequest:
    input_ids = torch.tensor(full_ids, dtype=torch.int64)
    if int(input_ids.numel()) <= 0:
        raise ValueError(f"Producer sample {sample_id!r} produced no input tokens")

    loss_mask = build_loss_mask(input_ids, prompt_length)
    position_ids = torch.arange(int(input_ids.numel()), dtype=torch.int64)

    feature_start = max(prompt_length - 1, 0)
    feature_end = int(input_ids.numel())
    if feature_end_limit is not None:
        feature_end = min(feature_end, int(feature_end_limit))
    max_feature_length = int(_config_value(config, "max_feature_length", 0) or 0)
    if max_feature_length == 1:
        raise ValueError("max_feature_length must be 0 or at least 2")
    if max_feature_length > 1:
        feature_end = min(feature_start + max_feature_length, feature_end)
    request_prompt_token_ids = (
        list(vllm_prompt_token_ids)
        if vllm_prompt_token_ids is not None
        else full_ids[:feature_end]
    )
    max_sequence_length = int(_config_value(config, "max_sequence_length", 0) or 0)
    if max_sequence_length > 0 and len(request_prompt_token_ids) > max_sequence_length:
        raise ValueError(
            f"Producer sample {sample_id!r} requires a vLLM prefill of "
            f"{len(request_prompt_token_ids)} tokens after selecting its training "
            f"window, exceeding max_sequence_length={max_sequence_length} "
            f"(full_sequence_length={int(input_ids.numel())}, "
            f"prompt_length={prompt_length})"
        )
    feature_positions = torch.arange(feature_start, feature_end, dtype=torch.int64)
    if int(feature_positions.numel()) <= 0:
        raise ValueError(f"Producer sample {sample_id!r} has an empty feature window")
    draft_position_ids = position_ids[feature_start:feature_end] + 1
    return TokenizedRequest(
        sequence_no=sequence_no,
        sample_id=sample_id,
        input_ids=input_ids,
        loss_mask=loss_mask,
        position_ids=position_ids,
        feature_positions=feature_positions,
        draft_position_ids=draft_position_ids,
        source_metadata=dict(source_metadata),
        vllm_prompt_token_ids=tuple(request_prompt_token_ids),
    )


def _token_ids(encoding: Any) -> list[int]:
    if isinstance(encoding, (list, tuple)) or torch.is_tensor(encoding):
        value = encoding
    else:
        value = (
            encoding.get("input_ids")
            if isinstance(encoding, Mapping)
            else encoding.input_ids
        )
    if value is None:
        raise ValueError("Tokenizer result is missing input_ids")
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tokenizer input_ids must be a tensor, list, or tuple")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Tokenizer returned more than one sequence for one input")
        value = value[0]
    return [int(token_id) for token_id in value]


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


__all__ = [
    "GenerationRequest",
    "InputRecord",
    "TokenizedRequest",
    "build_loss_mask",
    "finalize_generated_request",
    "iter_input_records",
    "prepare_generation_request",
    "prepare_generated_prefill_request",
    "tokenize_record",
]
