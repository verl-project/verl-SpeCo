# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Carry a single-turn proposal bundle across the LLM/agent output boundary."""

from contextvars import ContextVar

from verl_speco.integration.specopd_protocol import PROPOSAL_KEY

PROPOSAL_TURNS: ContextVar[list[str] | None] = ContextVar(
    "specopd_proposal_turns", default=None
)


def collect_proposal_output(output):
    turns = PROPOSAL_TURNS.get()
    fields = getattr(output, "extra_fields", None)
    if turns is not None and isinstance(fields, dict) and fields.get(PROPOSAL_KEY):
        turns.append(fields[PROPOSAL_KEY])


def merge_proposal_output(output):
    turns = PROPOSAL_TURNS.get()
    if not turns:
        return
    if len(turns) != 1:
        raise ValueError(
            "SpecOPD requires exactly one generated request per trajectory"
        )
    if not isinstance(getattr(output, "extra_fields", None), dict):
        raise ValueError("SpecOPD agent output has no extra_fields transport")
    output.extra_fields[PROPOSAL_KEY] = turns[0]
