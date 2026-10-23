# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration.agent_loop_runtime import (
    SPECO_STUDENT_DECODE_TIME_KEY,
    SPECO_TEACHER_PREFILL_TIME_KEY,
    _CURRENT_STUDENT_DECODE_TIMES,
    _compute_teacher_logprobs_with_timing,
    _record_phase_time,
    _run_agent_loop_with_student_timing,
)


def test_student_timing_accumulates_llm_request_times() -> None:
    output = SimpleNamespace(extra_fields={})

    async def run():
        times = _CURRENT_STUDENT_DECODE_TIMES.get()
        times.extend([0.2, 0.3])
        return output

    result = asyncio.run(_run_agent_loop_with_student_timing(run))

    assert result is output
    assert output.extra_fields[SPECO_STUDENT_DECODE_TIME_KEY] == 0.5


def test_nested_student_timing_uses_the_outer_accumulator() -> None:
    output = SimpleNamespace(extra_fields={})

    async def inner():
        _CURRENT_STUDENT_DECODE_TIMES.get().append(0.4)
        return output

    async def outer():
        return await _run_agent_loop_with_student_timing(inner)

    asyncio.run(_run_agent_loop_with_student_timing(outer))

    assert output.extra_fields[SPECO_STUDENT_DECODE_TIME_KEY] == 0.4


def test_teacher_prefill_timing_does_not_count_as_student_decode() -> None:
    worker = SimpleNamespace(distillation_enabled=True)
    output = SimpleNamespace(extra_fields={})
    student_times = []
    token = _CURRENT_STUDENT_DECODE_TIMES.set(student_times)

    async def upstream(*args):
        assert _CURRENT_STUDENT_DECODE_TIMES.get() is None
        return "scored"

    try:
        result = asyncio.run(
            _compute_teacher_logprobs_with_timing(
                worker, upstream, output, [1], [2], False
            )
        )
    finally:
        _CURRENT_STUDENT_DECODE_TIMES.reset(token)

    assert result == "scored"
    assert student_times == []
    assert output.extra_fields[SPECO_TEACHER_PREFILL_TIME_KEY] >= 0.0


def test_teacher_prefill_timing_is_skipped_for_validation() -> None:
    worker = SimpleNamespace(distillation_enabled=True)
    output = SimpleNamespace(extra_fields={})

    async def upstream(*args):
        return None

    asyncio.run(
        _compute_teacher_logprobs_with_timing(worker, upstream, output, [1], [2], True)
    )

    assert SPECO_TEACHER_PREFILL_TIME_KEY not in output.extra_fields


def test_teacher_prefill_timing_accumulates_across_calls() -> None:
    output = SimpleNamespace(extra_fields={})

    _record_phase_time(output, SPECO_TEACHER_PREFILL_TIME_KEY, 0.25)
    _record_phase_time(output, SPECO_TEACHER_PREFILL_TIME_KEY, 0.5)

    assert output.extra_fields[SPECO_TEACHER_PREFILL_TIME_KEY] == 0.75


@pytest.mark.parametrize(
    "key", [SPECO_STUDENT_DECODE_TIME_KEY, SPECO_TEACHER_PREFILL_TIME_KEY]
)
def test_phase_timing_accumulates_from_none_default(key) -> None:
    output = SimpleNamespace(extra_fields={key: None})

    _record_phase_time(output, key, 0.25)
    _record_phase_time(output, key, 0.5)

    assert output.extra_fields[key] == 0.75
