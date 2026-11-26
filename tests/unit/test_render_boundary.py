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

Pure logic plus an injectable vLLM ``/render`` client. The live-server check
requires SPECO_TEST_VLLM_RENDER_ENDPOINT; other tests do not use the network.
"""

from __future__ import annotations

import json
import os
from io import BytesIO

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
    assert "tool_choice" in captured["body"]
    assert captured["body"]["tool_choice"] is None
    assert captured["body"]["truncate_prompt_tokens"] == 16
    assert captured["body"]["truncation_side"] == "right"


@pytest.mark.parametrize("add_generation_prompt", [True, False])
@pytest.mark.parametrize(
    "tools", [None, [{"type": "function", "function": {"name": "lookup"}}]]
)
def test_render_conversation_serialized_tool_choice(
    monkeypatch, add_generation_prompt, tools
):
    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        assert body["add_generation_prompt"] is add_generation_prompt
        if tools is None:
            assert "tools" not in body
            assert "tool_choice" not in body
        else:
            assert body["tools"] == tools
            assert "tool_choice" in body  # Omission would default to "auto".
            assert body["tool_choice"] is None
        return BytesIO(b'{"token_ids": [1, 2, 3]}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert render_conversation(
        "http://localhost:8000/v1",
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    ) == [1, 2, 3]


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


def test_live_vllm_render_preserves_tool_schema_and_boundary() -> None:
    """Run against servers without auto tools, with exclusion both off and on."""
    endpoint = os.environ.get("SPECO_TEST_VLLM_RENDER_ENDPOINT")
    if not endpoint:
        pytest.skip("Set SPECO_TEST_VLLM_RENDER_ENDPOINT to test a running vLLM server")

    from verl_speco.data.render_boundary import _post_json, _render_url

    marker = "speco_schema_marker_7d3261"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "speco_lookup",
                "description": marker,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    renders = []

    def render(messages, *, add_generation_prompt, tools=None, max_length=None):
        ids = render_conversation(
            endpoint,
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        base = _render_url(endpoint).rsplit("/v1/", 1)[0]
        text = _post_json(f"{base}/detokenize", {"tokens": ids}, 10)["prompt"]
        assert marker in text
        assert "speco_lookup" in text
        renders.append((add_generation_prompt, ids))
        return ids

    record = InputRecord(
        sequence_no=0,
        sample_id="tools",
        prompt=({"role": "user", "content": "Say done."},),
        response="done",
        source_metadata={},
        tools=tools,
    )
    request = tokenize_record_with_render_boundary(
        record,
        object(),
        {"max_feature_length": 0, "max_sequence_length": 0},
        render,
    )
    assert [entry[0] for entry in renders[:2]] == [True, False]
    history_ids = renders[2][1] if len(renders) == 3 else None
    boundary = boundary_from_renders(renders[0][1], renders[1][1], history_ids)
    assert request.input_ids.tolist() == renders[1][1]
    assert 0 < boundary < len(request.input_ids)
    assert request.loss_mask.tolist() == [0] * boundary + [1] * (
        len(request.input_ids) - boundary
    )


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

    def fallback_render(
        messages, *, add_generation_prompt, tools=None, max_length=None
    ):
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
