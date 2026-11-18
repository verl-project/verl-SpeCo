# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Positive supervision for complete tool-use trajectories without role cleanup."""

from copy import deepcopy

from verl_speco.data.render_boundary import boundary_from_renders


def tokenize_role_preserving_messages(messages, tokenizer):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Role-preserving trajectory requires nonempty messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role")
            not in {
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
            }
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(
                "Each message must have a supported role and string content"
            )

    def render(prefix, *, generation=False):
        # DeepSeek's native template normally drops reasoning from older turns.
        # Replay must retain it and must not mutate the original tool-call dicts.
        value = tokenizer.apply_chat_template(
            deepcopy(prefix),
            tokenize=True,
            add_generation_prompt=generation,
            drop_thinking=False,
        )
        if not isinstance(value, list) or any(
            type(token) is not int for token in value
        ):
            raise ValueError("Chat template must return a flat integer token list")
        return value

    full = render(messages)
    mask = [0.0] * len(full)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = render(messages[:index], generation=True)
        until = render(messages[: index + 1])
        history = None if until[: len(before)] == before else render(messages[:index])
        start = boundary_from_renders(before, until, history)
        if full[: len(until)] != until:
            raise ValueError(
                "Chat template rewrites an earlier assistant turn; cannot align its mask"
            )
        if start >= len(until):
            raise ValueError("Assistant turn has no rendered tokens to supervise")
        mask[start : len(until)] = [1.0] * (len(until) - start)
    return full, mask
