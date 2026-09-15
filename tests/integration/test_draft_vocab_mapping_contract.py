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
"""Contract tests for the draft vocabulary mapping buffers.

``d2t`` holds offsets, not absolute ids: the target id of draft id ``i`` is
``i + d2t[i]``. That is what ``process_token_dict_to_mappings`` emits and what the
serving engines apply, so every producer of a ``d2t`` buffer has to agree on it.
"""

from __future__ import annotations

import pytest


def _tiny_eagle3_config(vocab_size: int, draft_vocab_size: int):
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=vocab_size,
        draft_vocab_size=draft_vocab_size,
        target_hidden_size=8,
        num_aux_hidden_states=3,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def _tiny_peagle_config(vocab_size: int, draft_vocab_size: int):
    from verl_speco.models.peagle import PeagleConfig

    return PeagleConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_draft_layers=1,
        target_hidden_size=8,
        num_aux_hidden_states=3,
        vocab_size=vocab_size,
        draft_vocab_size=draft_vocab_size,
        mask_token_id=vocab_size - 1,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def _draft_models(vocab_size: int, draft_vocab_size: int):
    from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3
    from verl_speco.models.peagle import LlamaForCausalLMPeagle

    return {
        "eagle3": LlamaForCausalLMEagle3(
            _tiny_eagle3_config(vocab_size, draft_vocab_size)
        ),
        "peagle": LlamaForCausalLMPeagle(
            _tiny_peagle_config(vocab_size, draft_vocab_size)
        ),
    }


@pytest.mark.parametrize("name", ["eagle3", "peagle"])
def test_default_d2t_is_the_offset_identity(name) -> None:
    """The default mapping must be the identity under the offset convention.

    ``torch.arange`` looks like an identity only if ``d2t`` held absolute ids. It
    does not, so an arange default resolves to ``target_id = 2 * draft_id``.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model = _draft_models(vocab_size=32, draft_vocab_size=32)[name]

    assert torch.equal(model.d2t, torch.zeros(32, dtype=model.d2t.dtype))

    # This is what the serving engine computes.
    target_ids = torch.arange(model.draft_vocab_size) + model.d2t
    assert torch.equal(target_ids, torch.arange(32))
    assert int(target_ids.max()) < model.vocab_size


@pytest.mark.parametrize("name", ["eagle3", "peagle"])
def test_default_d2t_stays_inside_the_target_vocabulary(name) -> None:
    """A reduced draft vocabulary must still resolve to in-range target ids."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model = _draft_models(vocab_size=32, draft_vocab_size=8)[name]

    target_ids = torch.arange(model.draft_vocab_size) + model.d2t
    assert int(target_ids.max()) < model.vocab_size
    # t2d marks the same tokens the offsets resolve to.
    assert torch.equal(torch.nonzero(model.t2d, as_tuple=False).flatten(), target_ids)


def test_generated_mapping_round_trips_through_the_offset_convention() -> None:
    """The repo's own mapping generator emits offsets, so decoding must match."""
    torch = pytest.importorskip("torch")
    from collections import Counter

    from verl_speco.data.preprocessing import process_token_dict_to_mappings

    token_dict = Counter({3: 10, 9: 8, 17: 5, 21: 2})
    d2t, t2d = process_token_dict_to_mappings(
        token_dict, draft_vocab_size=4, target_vocab_size=32
    )

    target_ids = torch.arange(4) + d2t
    assert target_ids.tolist() == [3, 9, 17, 21]
    assert torch.nonzero(t2d, as_tuple=False).flatten().tolist() == [3, 9, 17, 21]
