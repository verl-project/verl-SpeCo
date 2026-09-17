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
"""Read-side filtering for the non-TQ DraftFeatureDataLoader.

Equivalent to the standalone TQ Producer input filters: bad-sample policy,
min_supervised_tokens, and alignment/NaN validation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.draft_dataset import (  # noqa: E402
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
    validate_draft_sample,
)
from verl_speco.trainer.feature_store import DraftFeatureSample  # noqa: E402


def _sample(*, input_ids_len=4, mask=None, hidden=None):
    input_ids = torch.arange(input_ids_len, dtype=torch.long)
    loss_mask = torch.ones(input_ids_len, dtype=torch.float32) if mask is None else mask
    hidden_states = (
        torch.zeros(input_ids_len, 8, dtype=torch.float32) if hidden is None else hidden
    )
    return DraftFeatureSample(
        input_ids=input_ids,
        loss_mask=loss_mask,
        hidden_states=hidden_states,
    )


# ------------------------------- validator ---------------------------------


def test_valid_sample_passes_validation() -> None:
    assert validate_draft_sample(_sample(), min_supervised_tokens=1) is None


def test_loss_mask_length_mismatch_detected() -> None:
    reason = validate_draft_sample(
        _sample(mask=torch.ones(3, dtype=torch.float32)), min_supervised_tokens=0
    )
    assert reason is not None and "length mismatch" in reason


def test_hidden_row_mismatch_detected() -> None:
    reason = validate_draft_sample(
        _sample(hidden=torch.zeros(3, 8, dtype=torch.float32)),
        min_supervised_tokens=0,
    )
    assert reason is not None and "rows" in reason


def test_non_finite_hidden_detected() -> None:
    hidden = torch.zeros(4, 8, dtype=torch.float32)
    hidden[0, 0] = float("nan")
    reason = validate_draft_sample(_sample(hidden=hidden), min_supervised_tokens=0)
    assert reason is not None and "NaN/Inf" in reason


def test_min_supervised_tokens_detected() -> None:
    reason = validate_draft_sample(
        _sample(mask=torch.zeros(4, dtype=torch.float32)), min_supervised_tokens=1
    )
    assert reason is not None and "min_supervised_tokens" in reason


# -------------------------------- loader -----------------------------------


class _FakeStore:
    def __init__(self, mapping):
        self.mapping = mapping

    def iter_keys(self, *, shuffle=False, seed=0):
        return iter(self.mapping.keys())

    def read(self, key):
        value = self.mapping[key]
        if isinstance(value, Exception):
            raise value
        return value


def _loader(mapping, **overrides):
    config = DraftFeatureDataLoaderConfig(
        batch_size=16, shuffle=False, repeat=False, **overrides
    )
    return DraftFeatureDataLoader(_FakeStore(mapping), config)


def test_loader_skips_unreadable_samples() -> None:
    loader = _loader(
        {
            "a": _sample(),
            "b": RuntimeError("corrupt shard"),
            "c": _sample(),
        }
    )
    samples = [sample for batch in loader for sample in batch]
    assert len(samples) == 2
    assert loader.filter_stats["read_errors"] == 1
    assert loader.filter_stats["kept"] == 2


def test_loader_raises_on_read_error_when_configured() -> None:
    loader = _loader({"a": RuntimeError("corrupt shard")}, on_error="raise")
    with pytest.raises(RuntimeError, match="Failed to read feature sample"):
        list(loader)


def test_loader_circuit_breaker_on_consecutive_errors() -> None:
    loader = _loader(
        {"a": RuntimeError("bad a"), "b": RuntimeError("bad b")},
        on_error="skip",
        max_consecutive_errors=0,
    )
    with pytest.raises(RuntimeError, match="max_consecutive_errors=0"):
        list(loader)


def test_loader_drops_misaligned_sample() -> None:
    loader = _loader(
        {
            "a": _sample(),
            "b": _sample(hidden=torch.zeros(3, 8, dtype=torch.float32)),
        }
    )
    samples = [sample for batch in loader for sample in batch]
    assert len(samples) == 1
    assert loader.filter_stats["dropped_misaligned"] == 1


def test_loader_drops_unsupervised_sample() -> None:
    loader = _loader(
        {
            "a": _sample(),
            "b": _sample(mask=torch.zeros(4, dtype=torch.float32)),
        },
        min_supervised_tokens=1,
    )
    samples = [sample for batch in loader for sample in batch]
    assert len(samples) == 1
    assert loader.filter_stats["dropped_min_supervised"] == 1


def test_loader_strict_alignment_raises() -> None:
    loader = _loader(
        {"a": _sample(hidden=torch.zeros(3, 8, dtype=torch.float32))},
        strict_token_alignment="strict",
    )
    with pytest.raises(ValueError, match="failed validation"):
        list(loader)


def test_loader_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="on_error must be"):
        _loader({}, on_error="ignore")
    with pytest.raises(ValueError, match="strict_token_alignment must be"):
        _loader({}, strict_token_alignment="bogus")
