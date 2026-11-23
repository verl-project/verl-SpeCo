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
"""Plan proposal collection before response length is known."""

import hashlib


def make_window(
    config, request_id, actor_revision, drafter_revision, prompt_len, max_tokens
):
    width = min(config.window_response_tokens, max_tokens)
    if width <= 0 or prompt_len <= 0:
        raise ValueError("SpecOPD needs a nonempty prompt and generation budget")
    limit = min(config.random_max_offset, max(0, max_tokens - width))
    offset = 0
    if config.window_mode == "random" and limit:
        key = f"{request_id}:{actor_revision}:{drafter_revision}".encode()
        offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (
            limit + 1
        )
    return {
        "schema": 1,
        "request_id": str(request_id),
        "actor_revision": int(actor_revision),
        "drafter_revision": int(drafter_revision),
        "prompt_len": prompt_len,
        "start": offset,
        "end": offset + width,
        "max_branches": min(
            config.max_branches_per_request,
            config.max_proposal_tokens_per_request // config.proposal_width,
        ),
        "proposal_width": config.proposal_width,
        "reservoir": bool(config.anchor_count),
        "proposal_temperature": config.temperature,
        "max_wire_bytes": config.max_wire_bytes,
    }
