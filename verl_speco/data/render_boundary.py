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
"""Server-side differential rendering for loss-mask boundaries.

Instead of reconstructing assistant spans with a local regex, render each
conversation prefix through vLLM's chat template. For assistant turn ``j`` the
boundary is where the ``conv[:j+1]`` full render extends the ``conv[:j]``
generation-prompt render: earlier tokens are context (mask 0), later ones are
supervised (mask 1). This mirrors speculators' ``_render_boundary_rows``.

The pure functions here take an injected ``render_fn`` and are fully unit
testable; :func:`render_conversation` is the vLLM ``/render`` client used to
build that ``render_fn`` in production.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

__all__ = [
    "BoundaryUnstableError",
    "boundary_from_renders",
    "common_prefix_len",
    "render_conversation",
]

DEFAULT_RENDER_TIMEOUT = 10.0


class BoundaryUnstableError(ValueError):
    """The chat template is not prefix-stable at an assistant turn boundary."""


def common_prefix_len(a: list[int], b: list[int]) -> int:
    length = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        length += 1
    return length


def boundary_from_renders(
    prompt_ids: list[int],
    full_ids: list[int],
    history_ids: list[int] | None = None,
) -> int:
    """Derive the supervised boundary between two renders.

    The generation-prompt render is a prefix of the full render when the
    template is prefix-stable. When it diverges -- a pre-filled ``<think>``
    scaffold vs. recorded reasoning -- fall back to the common prefix, which is
    valid only if history itself agrees (checked against ``history_ids``).
    """
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    boundary = common_prefix_len(prompt_ids, full_ids)
    if history_ids is not None and (
        full_ids[: len(history_ids)] != history_ids or boundary < len(history_ids)
    ):
        raise BoundaryUnstableError(
            "prompt and full renders diverge inside history; cannot derive a "
            "boundary loss mask"
        )
    return boundary


def _render_url(endpoint: str) -> str:
    """Build the vLLM ``/render`` URL, tolerating an endpoint that ends in ``/v1``."""
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/v1/chat/completions/render"


def render_conversation(
    endpoint: str,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    tools: list[dict] | None = None,
    truncate_prompt_tokens: int | None = None,
    truncation_side: str | None = None,
    timeout: float = DEFAULT_RENDER_TIMEOUT,
) -> list[int]:
    """POST to vLLM's ``/v1/chat/completions/render`` and return ``token_ids``."""
    url = _render_url(endpoint)
    body: dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        body["tools"] = tools
    if truncate_prompt_tokens is not None:
        body["truncate_prompt_tokens"] = truncate_prompt_tokens
    if truncation_side is not None:
        body["truncation_side"] = truncation_side

    data = _post_json(url, body, timeout)
    if "token_ids" not in data:
        raise ValueError(f"Render endpoint response missing 'token_ids': {data}")
    return list(data["token_ids"])


def _post_json(url: str, body: Mapping[str, Any], timeout: float) -> dict:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Render endpoint returned {error.code}: {detail}"
        ) from error
