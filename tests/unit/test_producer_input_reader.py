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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.producer import input_reader


def test_iter_input_records_reads_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        json.dumps({"prompt": "Q: ", "response": "A", "source": "test"}) + "\n",
        encoding="utf-8",
    )

    records = list(input_reader.iter_input_records(input_path))

    assert len(records) == 1
    assert records[0].sample_id == "train-000000"
    assert records[0].prompt == "Q: "
    assert records[0].response == "A"
    assert records[0].source_metadata == {"source": "test"}


def test_iter_input_records_reads_parquet_rows(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "train.parquet"
    input_path.write_bytes(b"PAR1-test-fixture")
    rows = [
        {"sample_id": "first", "prompt": "Q1: ", "response": "A1"},
        {"prompt": "Q2: ", "response": "A2", "split": "train"},
    ]

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            assert path == input_path

        def iter_batches(self):
            yield SimpleNamespace(to_pylist=lambda: rows)

    monkeypatch.setattr(
        input_reader.importlib,
        "import_module",
        lambda name: SimpleNamespace(ParquetFile=FakeParquetFile),
    )

    records = list(input_reader.iter_input_records(input_path))

    assert [record.sample_id for record in records] == ["first", "train-000001"]
    assert records[1].source_metadata == {"split": "train"}


def test_iter_input_records_reads_real_parquet_when_available(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    input_path = tmp_path / "train.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist(
            [{"prompt": "real Q: ", "response": "real A", "split": "train"}]
        ),
        input_path,
    )

    records = list(input_reader.iter_input_records(input_path))

    assert len(records) == 1
    assert records[0].prompt == "real Q: "
    assert records[0].response == "real A"
    assert records[0].source_metadata == {"split": "train"}


def test_iter_input_records_reads_real_dapo_style_parquet_when_available(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    input_path = tmp_path / "dapo.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "data_source": "math_dapo",
                    "prompt": [{"role": "user", "content": "Solve Q"}],
                    "reward_model": {
                        "ground_truth": "42",
                        "style": "rule-lighteval/MATH_v2",
                    },
                    "extra_info": {"index": "dapo-real-row"},
                }
            ]
        ),
        input_path,
    )

    record = next(input_reader.iter_input_records(input_path))

    assert record.prompt == ({"role": "user", "content": "Solve Q"},)
    assert record.response is None
    assert record.sample_id == "dapo-real-row"


def test_dapo_parquet_prompt_is_prepared_for_target_generation(
    monkeypatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "train.parquet"
    input_path.write_bytes(b"PAR1-test-fixture")

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            assert path == input_path

        def iter_batches(self):
            yield SimpleNamespace(
                to_pylist=lambda: [
                    {
                        "prompt": [{"role": "user", "content": "Solve Q"}],
                        "reward_model": {"ground_truth": "42"},
                        "extra_info": {"index": "dapo-row-id"},
                    }
                ]
            )

    monkeypatch.setattr(
        input_reader.importlib,
        "import_module",
        lambda name: SimpleNamespace(ParquetFile=FakeParquetFile),
    )

    record = next(input_reader.iter_input_records(input_path))

    class ChatTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "Solve Q"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return [10, 11, 12]

    generation = input_reader.prepare_generation_request(
        record,
        ChatTokenizer(),
        {"max_sequence_length": 16, "generation_max_tokens": 4},
    )
    finalized = input_reader.finalize_generated_request(
        generation,
        [10, 11, 12, 20, 21],
        {"max_sequence_length": 16, "max_feature_length": 8},
    )

    assert record.sample_id == "dapo-row-id"
    assert record.response is None
    assert generation.prompt_token_ids == (10, 11, 12)
    assert generation.max_tokens == 4
    assert finalized.input_ids.tolist() == [10, 11, 12, 20, 21]
    assert finalized.loss_mask.tolist() == [0, 0, 0, 1, 1]
    assert finalized.prompt_token_ids == [10, 11, 12, 20, 21]


def test_finalize_generated_request_aligns_connector_excluding_final_token() -> None:
    request = input_reader.GenerationRequest(
        sequence_no=0,
        sample_id="generated-row",
        prompt_token_ids=(10, 11, 12),
        max_tokens=4,
        source_metadata={},
    )

    finalized = input_reader.finalize_generated_request(
        request,
        [10, 11, 12, 20],
        {"max_sequence_length": 16, "max_feature_length": 8},
        expected_response_token_ids=[20, 21],
    )

    assert finalized.input_ids.tolist() == [10, 11, 12, 20, 21]
    assert finalized.loss_mask.tolist() == [0, 0, 0, 1, 1]
    assert finalized.prompt_token_ids == [10, 11, 12, 20]
    assert finalized.feature_positions.tolist() == [2, 3]
    assert finalized.draft_position_ids.tolist() == [3, 4]


def test_prefilled_response_limits_vllm_prefix_after_selecting_training_window() -> None:
    prepared = input_reader._build_tokenized_request(
        sequence_no=0,
        sample_id="long-response",
        prompt_length=3,
        full_ids=list(range(20)),
        source_metadata={},
        config={"max_sequence_length": 8, "max_feature_length": 4},
    )

    assert prepared.input_ids.numel() == 20
    assert prepared.feature_positions.tolist() == [2, 3, 4, 5]
    assert prepared.prompt_token_ids == [0, 1, 2, 3, 4, 5]


def test_prefilled_response_rejects_prompt_prefix_beyond_vllm_limit() -> None:
    with pytest.raises(
        ValueError,
        match=r"vLLM prefill of 13 tokens.*max_sequence_length=8",
    ):
        input_reader._build_tokenized_request(
            sequence_no=0,
            sample_id="long-prompt",
            prompt_length=10,
            full_ids=list(range(20)),
            source_metadata={},
            config={"max_sequence_length": 8, "max_feature_length": 4},
        )


def test_iter_jsonl_conversation_splits_final_assistant_response(tmp_path: Path) -> None:
    input_path = tmp_path / "conversation.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "conversation": [
                    {"role": "human", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = next(input_reader.iter_input_records(input_path))

    assert record.prompt == ({"role": "user", "content": "Question"},)
    assert record.response == "Answer"


def test_iter_jsonl_legacy_conversations_normalizes_from_value(tmp_path: Path) -> None:
    input_path = tmp_path / "conversations.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "Question"},
                    {"from": "gpt", "value": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = next(input_reader.iter_input_records(input_path))

    assert record.prompt == ({"role": "user", "content": "Question"},)
    assert record.response == "Answer"


def test_prepare_generated_prefill_request_uses_full_sequence_without_final_token() -> None:
    request = input_reader.GenerationRequest(
        sequence_no=0,
        sample_id="generated-row",
        prompt_token_ids=(10, 11, 12),
        max_tokens=4,
        source_metadata={},
    )

    prepared = input_reader.prepare_generated_prefill_request(
        request,
        [20, 21],
        {"max_sequence_length": 16, "max_feature_length": 8},
    )

    assert prepared.input_ids.tolist() == [10, 11, 12, 20, 21]
    assert prepared.loss_mask.tolist() == [0, 0, 0, 1, 1]
    assert prepared.prompt_token_ids == [10, 11, 12, 20]
    assert prepared.feature_positions.tolist() == [2, 3]


def test_finalize_generated_request_rejects_misaligned_connector_tokens() -> None:
    request = input_reader.GenerationRequest(
        sequence_no=0,
        sample_id="generated-row",
        prompt_token_ids=(10, 11),
        max_tokens=4,
        source_metadata={},
    )

    with pytest.raises(ValueError, match="excluding its final token"):
        input_reader.finalize_generated_request(
            request,
            [10, 11, 99],
            {"max_sequence_length": 16, "max_feature_length": 8},
            expected_response_token_ids=[20, 21],
        )


def test_non_utf8_non_parquet_input_has_actionable_error(tmp_path: Path) -> None:
    input_path = tmp_path / "train.data"
    input_path.write_bytes(b"plain-prefix\xc0binary")

    with pytest.raises(ValueError, match="not UTF-8 JSONL or a Parquet file"):
        list(input_reader.iter_input_records(input_path))
