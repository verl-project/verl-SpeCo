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
"""Tolerant role cleaning in the conversation parser."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from verl_speco.data.parse import GeneralParser  # noqa: E402
from verl_speco.data.template import TEMPLATE_REGISTRY  # noqa: E402


class _ChatTokenizer:
    """Byte tokenizer plus a minimal qwen-style chat template renderer."""

    pad_token_id = None
    unk_token_id = 0
    bos_token = ""

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
        **kwargs,
    ):
        return types.SimpleNamespace(
            input_ids=[self.encode(text, max_length=max_length)]
        )

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = list(text.encode("utf-8"))
        if truncation and max_length is not None:
            return ids[:max_length]
        return ids

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        parts = []
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
            else:
                parts.append(content)
        return "".join(parts)


def _parse(conversation, *, strict_roles=False):
    parser = GeneralParser(
        _ChatTokenizer(),
        TEMPLATE_REGISTRY.get("qwen"),
        parser_strict_roles=strict_roles,
    )
    _, loss_mask = parser.parse(conversation, max_length=512)
    return parser, loss_mask


def test_unknown_role_is_mapped_to_assistant_by_default() -> None:
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello there"},
    ]
    parser, loss_mask = _parse(conversation, strict_roles=False)
    assert parser.role_cleanup_stats["mapped_roles"] == 1
    assert parser.role_cleanup_stats["dropped_role_turns"] == 0
    assert int(loss_mask.sum().item()) > 0


def test_unknown_role_keeps_legacy_behavior_in_strict_mode() -> None:
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello there"},
    ]
    parser, loss_mask = _parse(conversation, strict_roles=True)
    assert parser.role_cleanup_stats["mapped_roles"] == 0
    # The unknown role is not an assistant header, so nothing is supervised.
    assert int(loss_mask.sum().item()) == 0


def test_tool_order_violation_drops_turn_not_conversation() -> None:
    conversation = [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "stray tool output"},
        {"role": "assistant", "content": "answer"},
    ]
    parser, loss_mask = _parse(conversation, strict_roles=False)
    assert parser.role_cleanup_stats["dropped_role_turns"] == 1
    # The later assistant turn survives, so the row is still supervised.
    assert int(loss_mask.sum().item()) > 0


def test_tool_order_violation_truncates_in_strict_mode() -> None:
    conversation = [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "stray tool output"},
        {"role": "assistant", "content": "answer"},
    ]
    parser, loss_mask = _parse(conversation, strict_roles=True)
    assert parser.role_cleanup_stats["dropped_role_turns"] == 0
    assert int(loss_mask.sum().item()) == 0


def test_normal_tool_call_is_unchanged() -> None:
    conversation = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "calling tool"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": "final answer"},
    ]
    tolerant, tolerant_mask = _parse(conversation, strict_roles=False)
    strict, strict_mask = _parse(conversation, strict_roles=True)

    assert tolerant.role_cleanup_stats["dropped_role_turns"] == 0
    assert tolerant.role_cleanup_stats["mapped_roles"] == 0
    assert int(tolerant_mask.sum().item()) > 0
    assert torch.equal(tolerant_mask, strict_mask)


def test_non_user_first_turn_is_dropped_then_recovered() -> None:
    conversation = [
        {"role": "assistant", "content": "preemptive answer"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    parser, loss_mask = _parse(conversation, strict_roles=False)
    assert parser.role_cleanup_stats["dropped_role_turns"] == 1
    assert int(loss_mask.sum().item()) > 0
    assert "dropped_role_turns=1" in parser.format_role_cleanup_stats()


def test_cleaner_preserves_one_leading_system_turn() -> None:
    from verl_speco.data.parse import clean_conversation_roles

    cleaned, stats = clean_conversation_roles(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    assert [message["role"] for message in cleaned] == [
        "system",
        "user",
        "assistant",
    ]
    assert stats["dropped_role_turns"] == 0


def test_cleaner_requires_user_after_leading_system() -> None:
    from verl_speco.data.parse import clean_conversation_roles

    cleaned, stats = clean_conversation_roles(
        [
            {"role": "system", "content": "be terse"},
            {"role": "assistant", "content": "preemptive answer"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    assert [message["role"] for message in cleaned] == [
        "system",
        "user",
        "assistant",
    ]
    assert stats["dropped_role_turns"] == 1

