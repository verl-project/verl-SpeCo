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
"""DeepSeek templates must match the glyphs the model family actually renders."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from verl_speco.data.parse import GeneralParser  # noqa: E402
from verl_speco.data.template import TEMPLATE_REGISTRY  # noqa: E402

# Full-width vertical bar U+FF5C and lower-one-eighth block U+2581. The
# DeepSeek chat format uses the full-width bar, which is *not* the ASCII "|".
BAR = "\uff5c"
EOS = f"<{BAR}end\u2581of\u2581sentence{BAR}>"
ASSISTANT = f"<{BAR}Assistant{BAR}>"
USER = f"<{BAR}User{BAR}>"

DEEPSEEK_TEMPLATES = ("deepseek-r1-distill", "deepseek-v3", "deepseek-v32")


class _DeepSeekTokenizer:
    """Character-level tokenizer that renders the real DeepSeek chat format."""

    pad_token_id = 0
    unk_token_id = 0
    bos_token = f"<{BAR}begin\u2581of\u2581sentence{BAR}>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kw
    ):
        parts = [self.bos_token]
        for message in messages:
            if message["role"] == "user":
                parts.append(f"{USER}{message['content']}")
            elif message["role"] == "assistant":
                parts.append(f"{ASSISTANT}{message['content']}{EOS}")
        return "".join(parts)

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,
        **kw,
    ):
        return types.SimpleNamespace(
            input_ids=[self.encode(text, max_length=max_length)]
        )

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return torch.tensor(ids, dtype=torch.long)


@pytest.mark.parametrize("name", DEEPSEEK_TEMPLATES)
def test_separators_use_the_deepseek_glyphs(name: str) -> None:
    template = TEMPLATE_REGISTRY.get(name)
    assert template.assistant_header == ASSISTANT
    assert template.user_header == USER
    assert template.end_of_turn_token == EOS


@pytest.mark.parametrize("name", DEEPSEEK_TEMPLATES)
def test_loss_mask_covers_the_answer_and_its_end_of_sentence(name: str) -> None:
    tokenizer = _DeepSeekTokenizer()
    render = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am R1."},
        ]
    )
    parser = GeneralParser(tokenizer, TEMPLATE_REGISTRY.get(name))
    input_ids, loss_mask = parser.parse(
        [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am R1."},
        ],
        512,
    )

    supervised = "".join(
        chr(token)
        for token, keep in zip(input_ids.tolist(), loss_mask.tolist())
        if keep
    )
    # The prompt must stay unsupervised, and the answer must carry the loss.
    assert supervised == f"I am R1.{EOS}"
    assert render.count(EOS) == 1
