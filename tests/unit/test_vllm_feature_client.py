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

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from verl_speco.producer.vllm_feature_client import (
    VllmEndpoint,
    request_generate,
)


def test_request_generate_only_requests_generated_token_ids() -> None:
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        prompt_token_ids=[1, 2],
                        token_ids=[3, 4],
                    )
                ],
                kv_transfer_params={"hidden_states_path": "/tmp/result.safetensors"},
            )

    client = SimpleNamespace(completions=Completions())

    response = asyncio.run(
        request_generate(
            VllmEndpoint("http://vllm:8000/v1", 1),
            client,
            [1, 2],
            model="target",
            max_tokens=128,
            timeout=30,
        )
    )

    assert response.generated_token_ids == (3, 4)
    assert calls[0]["max_tokens"] == 128
    assert calls[0]["extra_body"] == {"return_token_ids": True}
