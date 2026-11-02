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
"""Hidden-state delivery in SGLang's stream accumulator after the SPECO rewrite.

The two accumulator shapes below are copied from
``sglang/srt/managers/scheduler_components/output_streamer.py`` (trimmed to the
fields the hidden-state block touches): the SGLang build bundled in the ROCm image and
public SGLang 0.5.16. Each test rewrites the real ``accept`` source, runs a stream
step over a batch, and reads the result the way that version's tokenizer manager does.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from verl_speco.integration.sglang_hidden_state_stream import (
    VERL_HIDDEN_NEXT_POSITION_ATTR,
    VERL_HIDDEN_STATES_STREAM_FINAL_ATTR,
    patch_sglang_accumulator_accept_source,
    sglang_accumulator_patch_globals,
)


@dataclass(slots=True, kw_only=True)
class _ImageAccumulator:
    rids: list = field(default_factory=list)
    output_hidden_states: list = field(default_factory=list)

    def accept(self, *, req) -> None:
        self.rids.append(req.rid)

        if req.return_hidden_states:
            self.output_hidden_states.append(req.hidden_states)
        if req.return_routed_experts:
            self.routed_experts.append(req.routed_experts)


@dataclass(slots=True, kw_only=True)
class _V0516Accumulator:
    return_hidden_states: bool
    return_routed_experts: bool = False
    rids: list = field(default_factory=list)
    output_hidden_states: Optional[list] = None

    def __post_init__(self) -> None:
        if self.return_hidden_states:
            self.output_hidden_states = []

    def accept(self, *, req) -> None:
        self.rids.append(req.rid)

        if self.return_hidden_states:
            if req.return_hidden_states:
                # Mirror output_ids_through_stop: spec verify steps can overshoot finished_len.
                hs = req.hidden_states
                if req.finished_len is not None:
                    hs = hs[: req.finished_len]
                self.output_hidden_states.append(hs)
            else:
                self.output_hidden_states.append(None)
        if self.return_routed_experts:
            self.routed_experts.append(None)


class _Req:
    def __init__(
        self,
        rid: str,
        *,
        return_hidden_states: bool,
        hidden_states: list | None = None,
        finished: bool = False,
        finished_len: int | None = None,
        stream_final: bool = False,
        speco_chunks: bool = False,
    ) -> None:
        self.rid = rid
        self.return_hidden_states = return_hidden_states
        self.return_routed_experts = False
        self.hidden_states = hidden_states if hidden_states is not None else []
        self.finished_len = finished_len
        self._finished = finished
        if stream_final:
            setattr(self, VERL_HIDDEN_STATES_STREAM_FINAL_ATTR, True)
        if speco_chunks:
            setattr(self, VERL_HIDDEN_NEXT_POSITION_ATTR, 128)

    def finished(self) -> bool:
        return self._finished


def _patched_accept(accumulator_cls: type) -> Any:
    source = textwrap.dedent(inspect.getsource(accumulator_cls.accept))
    patched = patch_sglang_accumulator_accept_source(source)
    assert patched is not None
    namespace: dict[str, Any] = {}
    exec(patched, dict(sglang_accumulator_patch_globals()), namespace)  # noqa: S102
    return namespace["accept"]


def _stream_image(reqs: list[_Req]) -> list[Any]:
    accept = _patched_accept(_ImageAccumulator)
    acc = _ImageAccumulator()
    for req in reqs:
        accept(acc, req=req)
    output_hidden_states = acc.output_hidden_states or None
    # tokenizer_manager: ``if output_hidden_states: meta_info["hidden_states"] = ...[i]``
    return [
        output_hidden_states[i] if output_hidden_states else None
        for i in range(len(acc.rids))
    ]


def _stream_v0516(reqs: list[_Req]) -> list[Any]:
    accept = _patched_accept(_V0516Accumulator)
    # _stream_output_generation computes the batch gate before accepting any request.
    acc = _V0516Accumulator(
        return_hidden_states=any(req.return_hidden_states for req in reqs)
    )
    for req in reqs:
        accept(acc, req=req)
    # tokenizer_manager: skip the field when the list is empty or the entry is None.
    return [
        acc.output_hidden_states[i] if acc.output_hidden_states else None
        for i in range(len(acc.rids))
    ]


STREAMERS = pytest.mark.parametrize(
    ("stream", "padding"), [(_stream_image, []), (_stream_v0516, None)]
)


def test_unknown_source_is_not_patched() -> None:
    source = "def accept(self, *, req):\n    self.rids.append(req.rid)\n"
    assert patch_sglang_accumulator_accept_source(source) is None


@STREAMERS
def test_mixed_batch_stays_aligned_with_rids(stream, padding) -> None:
    reqs = [
        _Req("plain-0", return_hidden_states=False),
        _Req("collect-1", return_hidden_states=True, hidden_states=["h1"]),
        _Req("plain-2", return_hidden_states=False),
        _Req("collect-3", return_hidden_states=True, hidden_states=["h3"]),
    ]

    assert stream(reqs) == [padding, ["h1"], padding, ["h3"]]


@STREAMERS
def test_batch_without_collection_has_no_hidden_states(stream, padding) -> None:
    reqs = [_Req(f"plain-{i}", return_hidden_states=False) for i in range(3)]

    assert stream(reqs) == [None, None, None]


@STREAMERS
def test_window_done_request_delivers_on_finish(stream, padding) -> None:
    # The completed window cleared return_hidden_states on every request, so the
    # 0.5.16 batch gate is False; the finished request must still deliver its chunks.
    reqs = [
        _Req("running-0", return_hidden_states=False, stream_final=True),
        _Req(
            "done-1",
            return_hidden_states=False,
            hidden_states=["c0", "c1", "c2"],
            finished=True,
            finished_len=1,
            stream_final=True,
            speco_chunks=True,
        ),
    ]

    assert stream(reqs) == [padding, ["c0", "c1", "c2"]]


@STREAMERS
def test_window_request_waits_for_finish(stream, padding) -> None:
    reqs = [
        _Req(
            "capturing-0",
            return_hidden_states=True,
            hidden_states=["c0"],
            stream_final=True,
        ),
        _Req("plain-1", return_hidden_states=False),
    ]

    assert stream(reqs) == [None, None]


def test_v0516_still_trims_per_token_hidden_states() -> None:
    reqs = [
        _Req(
            "plain-hidden-0",
            return_hidden_states=True,
            hidden_states=["t0", "t1", "t2"],
            finished=True,
            finished_len=2,
        )
    ]

    assert _stream_v0516(reqs) == [["t0", "t1"]]
