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
"""Read-side tolerance for the standalone TQ Producer input path.

Role cleanup, truncation/min-supervision filtering, bad-row policy, and
non-finite hidden-state rejection.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.producer.input_reader import (  # noqa: E402
    SampleFilteredError,
    _build_tokenized_request,
    iter_input_records,
)


def _write_jsonl(tmp_path, rows, extra_lines=()):
    path = tmp_path / "input.jsonl"
    lines = [json.dumps(row) for row in rows]
    for index, bad in extra_lines:
        lines.insert(index, bad)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------- role tolerance ---------------------------


def test_unknown_role_is_mapped_to_assistant(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "model", "value": "answer"},
                ]
            }
        ],
    )
    records = list(iter_input_records(path))
    assert len(records) == 1
    assert records[0].response == "answer"


def test_unknown_role_kept_in_strict_mode(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "model", "value": "answer"},
                ]
            }
        ],
    )
    records = list(iter_input_records(path, parser_strict_roles=True))
    # The unknown role is not an assistant, so the last turn is not a response.
    assert records[0].response is None


def test_order_violating_turn_dropped_not_aborted(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "tool", "value": "stray tool output"},
                    {"from": "gpt", "value": "answer"},
                ]
            }
        ],
    )
    records = list(iter_input_records(path))
    assert len(records) == 1
    assert records[0].response == "answer"


# ---------------------------------- bad rows --------------------------------


def test_bad_json_line_is_skipped_and_counted(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "q1"},
                    {"from": "gpt", "value": "a1"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "q2"},
                    {"from": "gpt", "value": "a2"},
                ]
            },
        ],
        extra_lines=[(1, "{not valid json")],
    )
    records = list(iter_input_records(path, on_error="skip"))
    assert [record.response for record in records] == ["a1", "a2"]
    assert [record.sequence_no for record in records] == [0, 1]


def test_bad_json_line_raises_in_raise_mode(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "q"},
                    {"from": "gpt", "value": "a"},
                ]
            }
        ],
        extra_lines=[(0, "{not valid json")],
    )
    with pytest.raises(ValueError, match="is invalid"):
        list(iter_input_records(path, on_error="raise"))


def test_bad_row_circuit_breaker(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "human", "value": "q"},
                    {"from": "gpt", "value": "a"},
                ]
            }
        ],
        extra_lines=[(0, "{bad1"), (1, "{bad2")],
    )
    with pytest.raises(RuntimeError, match="max_consecutive_errors=0"):
        list(iter_input_records(path, on_error="skip", max_consecutive_errors=0))


def test_malformed_record_is_skipped(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {"foo": "bar"},
            {
                "conversations": [
                    {"from": "human", "value": "q"},
                    {"from": "gpt", "value": "a"},
                ]
            },
        ],
    )
    records = list(iter_input_records(path, on_error="skip"))
    assert [record.response for record in records] == ["a"]


def test_invalid_on_error_rejected(tmp_path) -> None:
    path = _write_jsonl(tmp_path, [])
    with pytest.raises(ValueError, match="on_error must be"):
        list(iter_input_records(path, on_error="ignore"))


# ------------------------- truncation / supervision -------------------------


def _build(full_ids, prompt_length, **config):
    return _build_tokenized_request(
        sequence_no=0,
        sample_id="s",
        prompt_length=prompt_length,
        full_ids=full_ids,
        source_metadata={},
        config=config,
    )


def test_min_supervised_tokens_filters_short_window() -> None:
    with pytest.raises(SampleFilteredError, match="min_supervised_tokens"):
        _build(list(range(10)), prompt_length=8, min_supervised_tokens=5)


def test_truncated_supervision_drop() -> None:
    # Window of 5 cuts into the supervised response.
    with pytest.raises(SampleFilteredError, match="cut by the feature window"):
        _build(
            list(range(10)),
            prompt_length=2,
            max_feature_length=5,
            min_supervised_tokens=1,
            on_truncated_supervision="drop",
        )


def test_truncated_supervision_keep_returns_request() -> None:
    request = _build(
        list(range(10)),
        prompt_length=2,
        max_feature_length=5,
        min_supervised_tokens=1,
        on_truncated_supervision="keep",
    )
    assert request.input_ids.numel() == 10
    assert int(request.loss_mask.sum().item()) > 0


def test_invalid_on_truncated_value_rejected() -> None:
    with pytest.raises(ValueError, match="on_truncated_supervision must be"):
        _build(
            list(range(10)),
            prompt_length=2,
            max_feature_length=5,
            min_supervised_tokens=1,
            on_truncated_supervision="bogus",
        )


# --------------------------- non-finite hidden ------------------------------


def test_non_finite_hidden_states_rejected() -> None:
    from verl_speco.trainer.target_feature_replay import (
        FeatureContract,
        HiddenStateAlignmentError,
        feature_from_vllm_payload,
    )

    contract = FeatureContract(
        algorithm="EAGLE3",
        target_layer_ids=[1],
        hidden_states_layout="eagle3_aux_plus_last",
        dtype=torch.bfloat16,
        target_model_id="m",
        target_model_revision=None,
        tokenizer_fingerprint="t",
    )
    request = SimpleNamespace(
        feature_positions=torch.arange(0, 3), prompt_token_ids=[1, 2, 3]
    )
    hidden = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
    hidden[0, 0, 0] = float("nan")
    payload = {"token_ids": torch.tensor([1, 2, 3]), "hidden_states": hidden}
    with pytest.raises(HiddenStateAlignmentError, match="NaN/Inf"):
        feature_from_vllm_payload(payload, request, contract)


def test_leading_system_turn_is_preserved_for_the_producer(tmp_path) -> None:
    path = _write_jsonl(
        tmp_path,
        [
            {
                "conversations": [
                    {"from": "system", "value": "be terse"},
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ]
            }
        ],
    )
    records = list(iter_input_records(path))
    assert len(records) == 1
    assert records[0].response == "answer"
    assert any(message.get("content") == "be terse" for message in records[0].prompt)

