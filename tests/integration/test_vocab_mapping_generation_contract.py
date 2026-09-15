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
"""Contract tests for draft vocabulary mapping generation.

The mapping is baked into the exported drafter checkpoint, so it has to be both
reproducible and cheap enough to build at production vocabulary sizes.
"""

from __future__ import annotations

import time
from collections import Counter

import pytest


def test_t2d_marks_exactly_the_selected_tokens() -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.data.preprocessing import process_token_dict_to_mappings

    token_dict = Counter({3: 10, 9: 8, 17: 5, 21: 2})
    d2t, t2d = process_token_dict_to_mappings(
        token_dict, draft_vocab_size=4, target_vocab_size=32
    )

    assert t2d.dtype == torch.bool
    assert t2d.numel() == 32
    assert torch.nonzero(t2d, as_tuple=False).flatten().tolist() == [3, 9, 17, 21]
    # d2t holds offsets: the target id of draft id i is i + d2t[i].
    assert (torch.arange(4) + d2t).tolist() == [3, 9, 17, 21]


def test_padding_a_short_corpus_takes_the_lowest_unused_ids() -> None:
    """A corpus with too few distinct tokens is padded with zero-frequency ids.

    Which ids those are ends up in the exported checkpoint, so pin the contract:
    the lowest unused ones, in order, and the same on every run.
    """
    torch = pytest.importorskip("torch")
    from verl_speco.data.preprocessing import process_token_dict_to_mappings

    def build():
        return process_token_dict_to_mappings(
            Counter({40: 9, 41: 7}), draft_vocab_size=6, target_vocab_size=64
        )

    first_d2t, first_t2d = build()
    for _ in range(4):
        d2t, t2d = build()
        assert torch.equal(d2t, first_d2t)
        assert torch.equal(t2d, first_t2d)

    selected = torch.nonzero(first_t2d, as_tuple=False).flatten().tolist()
    assert selected == [0, 1, 2, 3, 40, 41]


def test_mapping_build_scales_to_a_production_vocabulary() -> None:
    """Building t2d must not cost target_vocab_size * draft_vocab_size scans."""
    torch = pytest.importorskip("torch")
    from verl_speco.data.preprocessing import process_token_dict_to_mappings

    target_vocab_size, draft_vocab_size = 65536, 8192
    token_dict = Counter({token: token + 1 for token in range(draft_vocab_size * 2)})

    started = time.perf_counter()
    _, t2d = process_token_dict_to_mappings(
        token_dict, draft_vocab_size, target_vocab_size
    )
    elapsed = time.perf_counter() - started

    assert int(t2d.sum()) == draft_vocab_size
    # Vectorized scatter is milliseconds here; the per-id list scan is minutes.
    assert elapsed < 5.0, f"vocab mapping build took {elapsed:.1f}s"
