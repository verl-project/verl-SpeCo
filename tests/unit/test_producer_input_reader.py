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

pytest.importorskip("torch")

from verl_speco.producer import input_reader


def test_source_route_waves_keep_resume_routes_and_bind_endpoint_order():
    from verl_speco.producer.sequence_replay import iter_source_route_waves

    requests = [input_reader.GenerationRequest(i, str(i), tuple(range(length)), 1, {})
                for i, length in enumerate([2, 8, 3, 7, 4])]
    settings = dict(wave_samples=4, endpoints=["source-a", "source-b"], max_samples=4,
                    max_tokens=32, max_sequence_tokens=16)
    full = list(iter_source_route_waves(requests, **settings))
    resumed = list(iter_source_route_waves(requests, consumed=[0, 2], **settings))
    assert resumed == [request for request in full if request.sequence_no not in {0, 2}]
    assert len({request.source_plan_sha256 for request in full[:4]}) == 1
    assert full[4].source_plan_sha256 != full[0].source_plan_sha256
    reversed_sources = list(iter_source_route_waves(requests, **dict(settings, endpoints=["source-b", "source-a"])))
    assert reversed_sources[0].source_plan_sha256 != full[0].source_plan_sha256
    with pytest.raises(ValueError, match="increasing"):
        list(iter_source_route_waves([requests[1], requests[0]], **settings))
    def oversized():
        yield input_reader.GenerationRequest(0, "large", tuple(range(17)), 1, {})
        raise AssertionError("Must reject oversized input before reading another request")
    with pytest.raises(ValueError, match="token limits"):
        list(iter_source_route_waves(oversized(), **settings))


def test_source_route_plan_preserves_order_and_enforces_wave_capacity():
    from verl_speco.producer.sequence_replay import plan_source_routes

    limits = dict(source_count=3, max_samples=2, max_tokens=12, max_sequence_tokens=10)
    lengths = [2, 10, 3, 9]
    plan = plan_source_routes(iter(lengths), **limits)
    assert plan == plan_source_routes(lengths, **limits)
    assert plan["routes"] == [2, 0, 2, 1]
    assert plan["lengths"] == lengths
    for source in range(3):
        assigned = [length for length, route in zip(lengths, plan["routes"]) if route == source]
        assert len(assigned) <= 2 and sum(assigned) <= 12
    assert plan_source_routes([10], **limits)["routes"] == [0]
    assert plan_source_routes([], **limits)["routes"] == []
    with pytest.raises(ValueError, match="token limits"):
        plan_source_routes([11], **limits)
    with pytest.raises(ValueError, match="sample capacity"):
        plan_source_routes([1] * 7, **limits)
    with pytest.raises(ValueError, match="cannot fit"):
        plan_source_routes([8] * 4, **limits)
    changed = plan_source_routes(lengths, **dict(limits, max_tokens=13))
    assert changed["sha256"] != plan["sha256"]


@pytest.mark.parametrize("response_budget", [0, 2, 3])
def test_prompt_context_does_not_consume_response_feature_budget(response_budget):
    request = input_reader._build_tokenized_request(
        sequence_no=0,
        sample_id="long-prompt",
        prompt_length=10,
        full_ids=list(range(15)),
        source_metadata={},
        config={
            "preserve_prompt_context": True,
            "max_feature_length": response_budget,
            "min_response_feature_rows": 2,
        },
    )
    end = 15 if response_budget == 0 else 10 + response_budget
    assert request.feature_positions.tolist() == list(range(end))
    assert request.loss_mask[:10].sum().item() == 0
    assert request.loss_mask[10:end].sum().item() == end - 10


def test_prompt_context_still_obeys_sequence_limit():
    with pytest.raises(input_reader.SampleFilteredError, match="max_sequence_length"):
        input_reader._build_tokenized_request(
            sequence_no=0,
            sample_id="too-long",
            prompt_length=10,
            full_ids=list(range(15)),
            source_metadata={},
            config={
                "preserve_prompt_context": True,
                "max_feature_length": 3,
                "max_sequence_length": 12,
            },
        )


@pytest.mark.parametrize("preserve", [False, True])
def test_prompt_context_option_keeps_same_response_budget(preserve):
    request = input_reader._build_tokenized_request(
        sequence_no=0,
        sample_id="optional-context",
        prompt_length=10,
        full_ids=list(range(15)),
        source_metadata={},
        config={
            "preserve_prompt_context": preserve,
            "response_feature_budget": True,
            "max_feature_length": 3,
        },
    )
    start = 0 if preserve else 9
    assert request.feature_positions.tolist() == list(range(start, 13))
    assert request.loss_mask[10:13].sum().item() == 3


def test_legacy_feature_window_keeps_prompt_anchor_budget():
    request = input_reader._build_tokenized_request(
        sequence_no=0,
        sample_id="legacy",
        prompt_length=10,
        full_ids=list(range(15)),
        source_metadata={},
        config={"max_feature_length": 3},
    )
    assert request.feature_positions.tolist() == [9, 10, 11]


@pytest.mark.parametrize("minimum", [2, 4, 16])
@pytest.mark.parametrize("preserve", [False, True])
def test_short_response_filtered_even_with_unlimited_feature_budget(minimum, preserve):
    with pytest.raises(input_reader.SampleFilteredError):
        input_reader._build_tokenized_request(
            sequence_no=0,
            sample_id="short-response",
            prompt_length=10,
            full_ids=list(range(10 + minimum - 1)),
            source_metadata={},
            config={
                "preserve_prompt_context": preserve,
                "response_feature_budget": True,
                "min_response_feature_rows": minimum,
                "max_feature_length": 0,
            },
        )


@pytest.mark.parametrize("minimum", [2, 4, 16])
def test_minimum_response_length_is_retained(minimum):
    request = input_reader._build_tokenized_request(
        sequence_no=0,
        sample_id="minimum-response",
        prompt_length=10,
        full_ids=list(range(10 + minimum)),
        source_metadata={},
        config={
            "preserve_prompt_context": True,
            "response_feature_budget": True,
            "min_response_feature_rows": minimum,
            "max_feature_length": minimum,
        },
    )
    assert request.feature_positions.numel() == 10 + minimum


def test_prompt_context_preserves_generated_hidden_row_limit():
    request = input_reader._build_tokenized_request(
        sequence_no=0,
        sample_id="generated",
        prompt_length=10,
        full_ids=list(range(15)),
        source_metadata={},
        config={"preserve_prompt_context": True, "max_feature_length": 10},
        vllm_prompt_token_ids=list(range(14)),
        feature_end_limit=14,
    )
    assert request.feature_positions.tolist() == list(range(14))
    assert request.input_ids.tolist() == list(range(15))
    assert request.prompt_token_ids == list(range(14))


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


def test_shuffled_jsonl_reader_reads_in_caller_order(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"sample_id": "a", "prompt": "Q1: ", "response": "A1"}),
        "",  # blank lines must not shift the byte offsets
        json.dumps({"sample_id": "b", "prompt": "Q2: ", "response": "A2"}),
        json.dumps({"sample_id": "c", "prompt": "Q3: ", "response": "A3"}),
    ]
    input_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reader = input_reader.ShuffledJsonlReader(input_path)
    assert reader.record_count == 3

    order = [2, 0, 1]
    records = list(reader.iter_records(order))

    sequential = list(input_reader.iter_input_records(input_path))
    expected = [sequential[i] for i in order]
    assert [r.sample_id for r in records] == [r.sample_id for r in expected]
    for shuffled, plain in zip(records, expected):
        assert (shuffled.prompt, shuffled.response) == (plain.prompt, plain.response)


def test_shuffled_jsonl_reader_skips_invalid_lines(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        json.dumps({"sample_id": "a", "prompt": "Q: ", "response": "A"})
        + "\n"
        + "{not json\n"
        + json.dumps({"sample_id": "b", "prompt": "Q2: ", "response": "B"})
        + "\n",
        encoding="utf-8",
    )

    reader = input_reader.ShuffledJsonlReader(input_path, max_consecutive_errors=5)
    assert reader.record_count == 3  # offsets are collected before parsing
    records = list(reader.iter_records([0, 1, 2]))

    assert [r.sample_id for r in records] == ["a", "b"]


def test_shuffled_implicit_ids_can_follow_original_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    input_path.write_text('\n'.join([json.dumps({"prompt": "a", "response": "x"}), "{bad",
                                     json.dumps({"prompt": "b", "response": "y"})]) + '\n', encoding="utf-8")
    reader = input_reader.ShuffledJsonlReader(input_path, stable_source_ids=True)
    first = list(reader.iter_records([2, 1, 0]))
    second = list(reader.iter_records([0, 1, 2]))
    assert [(record.sequence_no, record.sample_id) for record in first] == [(0, "train-000002"), (1, "train-000000")]
    assert {record.prompt: record.sample_id for record in first} == {record.prompt: record.sample_id for record in second}
    legacy = list(input_reader.ShuffledJsonlReader(input_path).iter_records([2, 0]))
    assert [record.sample_id for record in legacy] == ["train-000000", "train-000001"]


def test_shuffled_jsonl_reader_rejects_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "train.parquet"
    input_path.write_bytes(b"PAR1-fixture")

    with pytest.raises(input_reader.ShuffledInputUnsupportedError):
        input_reader.ShuffledJsonlReader(input_path)


def test_jsonl_limits_apply_to_sequential_and_shuffled_readers(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    row = (json.dumps({"prompt": "中文", "response": "x"}, ensure_ascii=False) + "\n").encode()
    path.write_bytes(row * 3)
    with pytest.raises(ValueError, match="max_jsonl_line_bytes"):
        list(input_reader.iter_input_records(path, max_jsonl_line_bytes=len(row) - 1))
    with pytest.raises(ValueError, match="max_jsonl_line_bytes"):
        input_reader.ShuffledJsonlReader(path, max_jsonl_line_bytes=len(row) - 1)
    assert len(list(input_reader.iter_input_records(path, max_jsonl_line_bytes=0))) == 3
    with pytest.raises(ValueError, match="max_indexed_records"):
        input_reader.ShuffledJsonlReader(path, max_indexed_records=2)
    assert input_reader.ShuffledJsonlReader(path, max_indexed_records=0).record_count == 3


def test_shuffled_reader_rejects_mid_epoch_edit_and_metadata_preserving_edit(tmp_path: Path) -> None:
    import os

    path = tmp_path / "train.jsonl"
    row = json.dumps({"prompt": "a", "response": "x"}) + "\n"
    path.write_text(row * 2, encoding="utf-8")
    reader = input_reader.ShuffledJsonlReader(path)
    records = reader.iter_records([0, 1])
    next(records)
    path.write_text(row * 3, encoding="utf-8")
    with pytest.raises(ValueError, match="changed after indexing"):
        next(records)
    reader = input_reader.ShuffledJsonlReader(path, verify_indexed_input=True)
    stat = path.stat()
    path.write_text((row * 3).replace('"prompt": "a"', '"prompt": "b"', 1), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed after indexing"):
        list(reader.iter_records([0]))


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


def test_prefilled_response_limits_vllm_prefix_after_selecting_training_window() -> (
    None
):
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


def test_prefilled_response_filters_prompt_prefix_beyond_vllm_limit() -> None:
    # Over-length samples are data-filtered (skipped upstream), not fatal.
    with pytest.raises(
        input_reader.SampleFilteredError,
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


def test_iter_jsonl_conversation_splits_final_assistant_response(
    tmp_path: Path,
) -> None:
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


def test_prepare_generated_prefill_request_uses_full_sequence_without_final_token() -> (
    None
):
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


@pytest.mark.parametrize("feature_rows", [2, 3, 5])
def test_minimum_feature_rows_applies_after_window_selection(feature_rows):
    kwargs = dict(
        sequence_no=0, sample_id="eagle3", prompt_length=2,
        full_ids=list(range(6)), source_metadata={},
        config={"min_feature_rows": 3, "min_supervised_tokens": 1,
                "max_feature_length": feature_rows},
    )
    if feature_rows < 3:
        with pytest.raises(input_reader.SampleFilteredError, match="2 feature rows"):
            input_reader._build_tokenized_request(**kwargs)
    else:
        request = input_reader._build_tokenized_request(**kwargs)
        assert request.feature_positions.numel() == feature_rows


def _replay_row(index):
    return {"sequence_no": index, "sample_id": f"id-{index}",
            "input_ids": [1, 2, index + 3], "loss_mask": [0, 0, 1],
            "source_metadata": {"source": "test"}}


def test_sequence_replay_commits_segments_and_resumes_idempotently(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayReader, SequenceReplayWriter

    identity = {"target_revision": "model-sha", "tokenizer": "tokenizer-sha"}
    writer = SequenceReplayWriter(tmp_path, identity=identity, segment_rows=2)
    assert writer.append(_replay_row(0))
    assert not (tmp_path / "manifest.json").exists()
    assert writer.append(_replay_row(1))
    reader = SequenceReplayReader(tmp_path, identity=identity, allow_incomplete=True)
    assert [row["sample_id"] for row in reader.iter_rows([1, 0])] == ["id-1", "id-0"]
    writer.append(_replay_row(2))
    writer.close(commit=False)
    with pytest.raises(ValueError, match="has not completed"):
        SequenceReplayReader(tmp_path)
    assert len(SequenceReplayReader(tmp_path, allow_incomplete=True)) == 2
    writer = SequenceReplayWriter(tmp_path, identity=identity, segment_rows=2)
    assert writer.has_record(0, "id-0") and not writer.append(_replay_row(0))
    assert writer.append(_replay_row(2))
    writer.close()
    assert len(SequenceReplayReader(tmp_path)) == 3
    with pytest.raises(ValueError, match="identity differs"):
        SequenceReplayWriter(tmp_path, identity={"target_revision": "other"})


def test_sequence_replay_manifest_failure_keeps_old_committed_snapshot(tmp_path, monkeypatch):
    from verl_speco.producer import sequence_replay

    writer = sequence_replay.SequenceReplayWriter(tmp_path, identity={"target": "v1"}, segment_rows=1)
    writer.append(_replay_row(0))
    original_write = sequence_replay._atomic_write
    def interrupted(path, data):
        if path.name == "manifest.json":
            raise OSError("injected manifest failure")
        return original_write(path, data)
    monkeypatch.setattr(sequence_replay, "_atomic_write", interrupted)
    with pytest.raises(OSError, match="injected"):
        writer.append(_replay_row(1))
    writer.close(commit=False)
    assert [row["sequence_no"] for row in sequence_replay.SequenceReplayReader(tmp_path, allow_incomplete=True).iter_rows()] == [0]


def test_sequence_replay_rejects_concurrent_writer_and_corrupted_segment(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayReader, SequenceReplayWriter

    writer = SequenceReplayWriter(tmp_path, identity={"target": "v1"}, segment_rows=1)
    with pytest.raises(OSError):
        SequenceReplayWriter(tmp_path, identity={"target": "v1"})
    writer.append(_replay_row(0))
    writer.close()
    segment = next(tmp_path.glob("segment-*.jsonl"))
    segment.write_bytes(segment.read_bytes().replace(b'id-0', b'id-x'))
    with pytest.raises(ValueError, match="checksum mismatch"):
        SequenceReplayReader(tmp_path)


def test_sequence_replay_detects_equal_length_row_swap_after_indexing(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayReader, SequenceReplayWriter
    writer = SequenceReplayWriter(tmp_path, identity={"target": "v1"}, segment_rows=2)
    writer.append(_replay_row(0))
    writer.append(_replay_row(1))
    writer.close()
    reader = SequenceReplayReader(tmp_path)
    assert len(reader.row_hashes) == 64 and not isinstance(reader.seen, dict)
    segment = next(tmp_path.glob("segment-*.jsonl"))
    lines = segment.read_bytes().splitlines(keepends=True)
    assert len(lines[0]) == len(lines[1])
    segment.write_bytes(lines[1] + lines[0])
    with pytest.raises(ValueError, match="row changed after indexing"):
        list(reader.iter_rows())


def test_sequence_replay_index_preserves_large_sequence_numbers(tmp_path):
    from verl_speco.producer.sequence_replay import SequenceReplayReader, SequenceReplayWriter
    sequence = 10**30
    row = _replay_row(0)
    row["sequence_no"] = sequence
    writer = SequenceReplayWriter(tmp_path, identity={"target": "v1"})
    writer.append(row)
    writer.close()
    writer = SequenceReplayWriter(tmp_path, identity={"target": "v1"})
    assert writer.has_record(sequence, row["sample_id"])
    writer.close()
    assert next(SequenceReplayReader(tmp_path).iter_rows())["sequence_no"] == sequence


def test_sequence_replay_temporary_index_is_closed_and_rebuildable(tmp_path, monkeypatch):
    from verl_speco.producer import sequence_replay
    index_root = tmp_path / "indices"
    index_root.mkdir()
    original = sequence_replay.tempfile.TemporaryDirectory
    paths = []
    def temporary(*args, **kwargs):
        kwargs["dir"] = index_root
        directory = original(*args, **kwargs)
        paths.append(Path(directory.name))
        return directory
    monkeypatch.setattr(sequence_replay.tempfile, "TemporaryDirectory", temporary)
    data = tmp_path / "data"
    writer = sequence_replay.SequenceReplayWriter(data, identity={"target": "v1"}, segment_rows=1)
    assert paths[-1].exists()
    writer.append(_replay_row(0))
    index = writer.seen
    original_add = index.add
    def fail(sequence, sample, digest):
        if sequence == 1:
            raise OSError("injected temporary index failure")
        return original_add(sequence, sample, digest)
    monkeypatch.setattr(index, "add", fail)
    with pytest.raises(OSError, match="temporary index failure"):
        writer.append(_replay_row(1))
    with pytest.raises(RuntimeError, match="closed or failed"):
        writer.append(_replay_row(2))
    writer.close(commit=False)
    assert not paths[-1].exists()
    recovered = sequence_replay.SequenceReplayWriter(data, identity={"target": "v1"})
    assert recovered.has_record(0, "id-0") and recovered.has_record(1, "id-1")
    recovered.close()
    reader = sequence_replay.SequenceReplayReader(data)
    assert paths[-1].exists() and reader.seen[1][0] == "id-1" and 0 in reader.seen
    reader.close()
    assert not paths[-1].exists()
    assert [row["sequence_no"] for row in reader.iter_rows()] == [0, 1]
    segment = next(data.glob("segment-*.jsonl"))
    segment.write_bytes(segment.read_bytes().replace(b"id-", b"zz-"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        sequence_replay.SequenceReplayReader(data)
    assert not paths[-1].exists()


def test_sequence_replay_index_rejects_duplicate_across_segments(tmp_path):
    import hashlib
    from verl_speco.producer import sequence_replay
    writer = sequence_replay.SequenceReplayWriter(tmp_path, identity={"target": "v1"}, segment_rows=1)
    writer.append(_replay_row(0))
    writer.append(_replay_row(1))
    writer.close()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    descriptor = manifest["segments"][1]
    segment = tmp_path / descriptor["file"]
    row = json.loads(segment.read_text(encoding="utf-8"))
    row["sequence_no"] = 0
    payload = sequence_replay._encoded(row) + b"\n"
    segment.write_bytes(payload)
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(sequence_replay._encoded(manifest) + b"\n")
    with pytest.raises(ValueError, match="Duplicate sequence number"):
        sequence_replay.SequenceReplayReader(tmp_path)


def test_sequence_replay_training_request_preserves_full_labels_and_metadata():
    from verl_speco.producer.sequence_replay import prepare_replay_request

    row = {"sequence_no": 3, "sample_id": "s", "input_ids": [1, 2, 3, 4, 5],
           "loss_mask": [0, 0, 1, 1, 1], "source_metadata": {"replay_sequence_no": "original"}}
    request = prepare_replay_request(row, sequence_no=9, config={"max_sequence_length": 10})
    assert request.sequence_no == 9
    assert request.input_ids.tolist() == [1, 2, 3, 4, 5]
    assert request.prompt_token_ids == [1, 2, 3, 4]
    assert request.loss_mask.tolist() == [0, 0, 1, 1, 1]
    assert request.source_metadata == row["source_metadata"]
    row["prefill_excludes_last"] = False
    response_request = prepare_replay_request(row, sequence_no=9, config={"max_sequence_length": 10})
    assert response_request.prompt_token_ids == [1, 2, 3, 4, 5]


def test_sequence_replay_limits_buffer_by_bytes_not_only_rows(tmp_path, monkeypatch):
    from verl_speco.producer import sequence_replay

    monkeypatch.setattr(sequence_replay, "_MAX_ROW_BYTES", 256)
    writer = sequence_replay.SequenceReplayWriter(
        tmp_path, identity={"model": "v1"}, segment_rows=100, segment_bytes=256,
    )
    for index in range(3):
        row = _replay_row(index)
        row["source_metadata"]["description"] = "x" * 80
        writer.append(row)
        assert writer.pending_bytes <= 256
    writer.close()
    reader = sequence_replay.SequenceReplayReader(tmp_path)
    assert len(reader.manifest["segments"]) > 1
    assert [row["sequence_no"] for row in reader.iter_rows()] == [0, 1, 2]


def test_source_proportional_order_preserves_rows_and_stable_fallback_identity(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"prompt": "q", "response": "a", "source": source}
            for source in ["a", "a", "a", "b", "b", "c"]]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    reader = input_reader.ShuffledJsonlReader(path, source_field="source", on_error="raise")
    order = list(reader.source_proportional_order())
    assert sorted(order) == list(range(6))
    assert [rows[index]["source"] for index in order] == ["a", "b", "a", "c", "b", "a"]
    shuffled = list(reader.source_proportional_order(seed=42))
    assert sorted(shuffled) == list(range(6))
    assert shuffled == list(reader.source_proportional_order(seed=42))
    records = list(reader.iter_records(shuffled))
    assert [item.sample_id for item in records] == [f"train-{index:06d}" for index in shuffled]
    assert [item.sequence_no for item in records] == list(range(len(records)))
    import os
    stat = path.stat()
    path.write_text(path.read_text(encoding="utf-8").replace('"source": "a"', '"source": "z"', 1), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed after indexing"):
        list(reader.iter_records(order))


@pytest.mark.parametrize("value", [None, "", 3, []])
def test_source_proportional_order_requires_valid_source_labels(tmp_path, value):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"prompt": "q", "response": "a", "source": value}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid source label"):
        input_reader.ShuffledJsonlReader(path, source_field="source")
