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
"""Server-side differential render boundaries and the Producer wiring.

Pure logic plus an injectable vLLM ``/render`` client. No network in tests.
"""

from __future__ import annotations

import pytest

from verl_speco.data.render_boundary import (
    BoundaryUnstableError,
    boundary_from_renders,
    common_prefix_len,
    render_conversation,
)

_ROLE_IDS = {"system": 0, "user": 1, "assistant": 2}


def test_common_prefix_len() -> None:
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([1], [1]) == 1


def test_boundary_from_renders_prefix() -> None:
    assert boundary_from_renders([1, 2, 3], [1, 2, 3, 4, 5]) == 3


def test_boundary_from_renders_scaffold_fallback() -> None:
    # Generation prompt pre-fills scaffold 77, full render has reasoning 99.
    assert boundary_from_renders([1, 2, 77], [1, 2, 99, 5], [1, 2]) == 2


def test_boundary_from_renders_unstable_history_raises() -> None:
    with pytest.raises(BoundaryUnstableError, match="diverge inside history"):
        boundary_from_renders([1, 2, 77], [1, 9, 99, 5], [1, 2])


def test_render_conversation_client_returns_token_ids(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, body, timeout):
        captured["url"] = url
        captured["body"] = body
        return {"token_ids": [1, 2, 3]}

    monkeypatch.setattr("verl_speco.data.render_boundary._post_json", fake_post)
    ids = render_conversation(
        "http://localhost:8000",
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tools=[{"type": "function"}],
        truncate_prompt_tokens=16,
        truncation_side="right",
    )
    assert ids == [1, 2, 3]
    assert captured["url"].endswith("/v1/chat/completions/render")
    assert captured["body"]["add_generation_prompt"] is True
    assert captured["body"]["tools"] == [{"type": "function"}]
    assert captured["body"]["truncate_prompt_tokens"] == 16
    assert captured["body"]["truncation_side"] == "right"


def test_render_conversation_normalizes_v1_endpoint(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, body, timeout):
        captured["url"] = url
        return {"token_ids": [1]}

    monkeypatch.setattr("verl_speco.data.render_boundary._post_json", fake_post)
    for endpoint in ("http://localhost:8000/v1", "http://localhost:8000/v1/"):
        render_conversation(
            endpoint, [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
        assert captured["url"] == "http://localhost:8000/v1/chat/completions/render"


def test_render_conversation_client_missing_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.data.render_boundary._post_json",
        lambda url, body, timeout: {"error": "bad"},
    )
    with pytest.raises(ValueError, match="missing 'token_ids'"):
        render_conversation(
            "http://localhost:8000",
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
        )


# ---------------------------------------------------------------------------
# Producer input path integration
# ---------------------------------------------------------------------------
from verl_speco.producer.input_reader import (  # noqa: E402
    InputRecord,
    tokenize_record_with_render_boundary,
)


def _fake_render(messages, *, add_generation_prompt, tools=None, max_length=None):
    ids: list[int] = []
    for message in messages:
        ids.append(_ROLE_IDS[message["role"]])
        ids.extend(ord(char) % 97 for char in message["content"])
    if add_generation_prompt:
        ids.append(_ROLE_IDS["assistant"])
    return ids


def test_producer_uses_render_boundary_for_loss_mask() -> None:
    record = InputRecord(
        sequence_no=0,
        sample_id="s0",
        prompt=({"role": "user", "content": "hi"},),
        response="yo",
        source_metadata={},
    )
    request = tokenize_record_with_render_boundary(
        record,
        object(),
        {"max_feature_length": 0, "max_sequence_length": 0},
        _fake_render,
    )
    assert request.input_ids.tolist() == [1, 7, 8, 2, 24, 14]
    assert request.loss_mask.tolist() == [0, 0, 0, 0, 1, 1]


def test_producer_render_boundary_renders_history_only_when_needed() -> None:
    record = InputRecord(
        sequence_no=0,
        sample_id="s0",
        prompt=({"role": "user", "content": "hi"},),
        response="yo",
        source_metadata={},
    )
    config = {"max_feature_length": 0, "max_sequence_length": 0}

    stable_calls: list[bool] = []

    def stable_render(messages, *, add_generation_prompt, tools=None, max_length=None):
        stable_calls.append(add_generation_prompt)
        return _fake_render(messages, add_generation_prompt=add_generation_prompt)

    tokenize_record_with_render_boundary(record, object(), config, stable_render)
    # The prompt render already extends into the full render, so the history
    # render is skipped.
    assert len(stable_calls) == 2

    fallback_calls: list[int] = []

    def fallback_render(messages, *, add_generation_prompt, tools=None, max_length=None):
        fallback_calls.append(len(messages))
        if add_generation_prompt:
            return [1, 7, 8, 2]
        if len(messages) > 1:
            return [1, 7, 8, 99, 24, 14]
        return [1, 7, 8]

    tokenize_record_with_render_boundary(record, object(), config, fallback_render)
    # A non-prefix-stable template still needs the history render for validation.
    assert len(fallback_calls) == 3


def test_producer_render_boundary_requires_response() -> None:
    record = InputRecord(
        sequence_no=0,
        sample_id="s0",
        prompt=({"role": "user", "content": "hi"},),
        response=None,
        source_metadata={},
    )
    with pytest.raises(ValueError, match="has no response"):
        tokenize_record_with_render_boundary(
            record, object(), {"max_feature_length": 0}, _fake_render
        )


def test_producer_string_prompt_falls_back_to_local_tokenizer(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        "verl_speco.producer.input_reader.tokenize_record",
        lambda record, tokenizer, config: sentinel,
    )
    record = InputRecord(
        sequence_no=0, sample_id="s0", prompt="plain", response="yo", source_metadata={}
    )
    result = tokenize_record_with_render_boundary(
        record, object(), {"max_feature_length": 0}, _fake_render
    )
    assert result is sentinel
