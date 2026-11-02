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
"""Source rewrites for SGLang's stream-output hidden-state block.

Newer SGLang builds the per-batch stream output in
``sglang.srt.managers.scheduler_components.output_streamer._GenerationStreamAccumulator``.
Two shapes of its ``accept`` hidden-state block are known:

* the SGLang dev build bundled in the ROCm image (reports "0.5.16")::

      if req.return_hidden_states:
          self.output_hidden_states.append(req.hidden_states)

  with ``output_hidden_states`` defaulting to ``[]`` and the tokenizer manager reading
  ``output_hidden_states[i]`` for every rid as soon as the list is non-empty;

* public SGLang 0.5.16::

      if self.return_hidden_states:
          if req.return_hidden_states:
              hs = req.hidden_states
              if req.finished_len is not None:
                  hs = hs[: req.finished_len]
              self.output_hidden_states.append(hs)
          else:
              self.output_hidden_states.append(None)

  where ``self.return_hidden_states`` is a batch-level gate computed once per stream call as
  ``any(req.return_hidden_states ...)``, and ``None`` entries are skipped by the consumer.

SPECO's capture window clears ``req.return_hidden_states`` as soon as the window is complete
and only delivers the collected chunks when the request finishes. Both upstream shapes then
drop that final delivery (the per-request check, and in 0.5.16 also the batch gate once every
request in the batch has completed its window), and the image shape additionally misaligns
``output_hidden_states`` with ``rids`` in mixed batches. Both shapes are rewritten to
the same logic: decide per request with :func:`sglang_req_should_stream_hidden_states`,
create the list lazily on the first delivering request, and pad every other position so the
list stays aligned with ``rids``.

This module does not import SGLang so the rewrites can be unit-tested on plain sources.
"""

from __future__ import annotations

import re
from typing import Any

VERL_HIDDEN_STATES_STREAM_FINAL_ATTR = "_verl_hidden_states_stream_final"
# Set on every request whose hidden states SPECO captured as chunk payloads.
VERL_HIDDEN_NEXT_POSITION_ATTR = "_verl_hidden_next_position"


def sglang_req_should_stream_hidden_states(req: Any) -> bool:
    """Whether this stream output should carry the request's hidden states.

    Requests capturing with a window or row budget deliver once, when they finish (even
    after the completed window cleared ``return_hidden_states``); all other requests
    follow ``return_hidden_states``.
    """
    if getattr(req, VERL_HIDDEN_STATES_STREAM_FINAL_ATTR, False):
        try:
            return bool(req.finished())
        except Exception:  # noqa: BLE001
            return False
    return bool(getattr(req, "return_hidden_states", False))


def sglang_stream_hidden_states_payload(req: Any) -> Any:
    """Return the hidden states to stream for ``req``.

    Matches SGLang 0.5.16, which trims per-token hidden states to ``finished_len`` because
    speculative verify steps can overshoot it. SPECO chunk payloads are a list of chunks,
    not of tokens, so they are returned untrimmed.
    """
    hidden_states = req.hidden_states
    finished_len = getattr(req, "finished_len", None)
    if (
        finished_len is None
        or getattr(req, VERL_HIDDEN_NEXT_POSITION_ATTR, None) is not None
    ):
        return hidden_states
    return hidden_states[:finished_len]


# Image dev build: per-request block only.
SGLANG_ACCUMULATOR_HIDDEN_STATES_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]+)if\s+req\.return_hidden_states\s*:\r?\n"
    r"(?P=indent)[ \t]+self\.output_hidden_states\.append\(req\.hidden_states\)\r?\n",
)
# Public 0.5.16: the whole batch-gated block, through its ``append(None)`` else branch.
SGLANG_ACCUMULATOR_GATED_HIDDEN_STATES_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]+)if\s+self\.return_hidden_states\s*:\r?\n"
    r"(?P=indent)[ \t]+if\s+req\.return_hidden_states\s*:\r?\n"
    r"(?:(?:(?P=indent)[ \t]+.*)?\r?\n)*?"
    r"(?P=indent)[ \t]+self\.output_hidden_states\.append\(None\)\r?\n",
)


def _render_accumulator_hidden_states(indent: str, payload: str, padding: str) -> str:
    inner = indent + "    "
    return (
        f"{indent}if _sglang_req_should_stream_hidden_states(req):\n"
        f"{inner}if not self.output_hidden_states:\n"
        f"{inner}    self.output_hidden_states = [{padding} for _ in range(len(self.rids) - 1)]\n"
        f"{inner}self.output_hidden_states.append({payload})\n"
        f"{indent}elif self.output_hidden_states:\n"
        f"{inner}self.output_hidden_states.append({padding})\n"
    )


def patch_sglang_accumulator_accept_source(source: str) -> str | None:
    """Rewrite the hidden-state block of a dedented ``accept`` source.

    Returns the patched source, or ``None`` when neither known shape is found. The patched
    code calls ``_sglang_req_should_stream_hidden_states`` (and, for 0.5.16,
    ``_sglang_stream_hidden_states_payload``), which the caller must provide as globals.
    Padding follows each consumer: ``[]`` for the image build, ``None`` (skipped) for 0.5.16.
    """
    patched, count = SGLANG_ACCUMULATOR_GATED_HIDDEN_STATES_PATTERN.subn(
        lambda match: _render_accumulator_hidden_states(
            match.group("indent"),
            "_sglang_stream_hidden_states_payload(req)",
            "None",
        ),
        source,
        count=1,
    )
    if count <= 0:
        patched, count = SGLANG_ACCUMULATOR_HIDDEN_STATES_PATTERN.subn(
            lambda match: _render_accumulator_hidden_states(
                match.group("indent"), "req.hidden_states", "[]"
            ),
            source,
            count=1,
        )
    if count <= 0 or patched == source:
        return None
    return patched


def sglang_accumulator_patch_globals() -> dict[str, Any]:
    """Helpers the patched ``accept`` source refers to."""
    return {
        "_sglang_req_should_stream_hidden_states": sglang_req_should_stream_hidden_states,
        "_sglang_stream_hidden_states_payload": sglang_stream_hidden_states_payload,
    }
