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

from pathlib import Path

import pytest

pytest.importorskip("torch")

from verl_speco.trainer.standalone_resume import (
    build_input_fingerprint,
    load_standalone_resume,
    parse_bool_setting,
    save_standalone_resume,
    validate_standalone_loader_config,
)


def test_standalone_resume_round_trip_and_input_validation(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    input_path.write_text('{"prompt":"q","response":"a"}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_7"

    save_standalone_resume(
        checkpoint_path,
        [5, 1, 5, 3],
        optimizer_step=7,
        input_path=input_path,
    )

    consumed, metadata = load_standalone_resume(
        checkpoint_path, input_path=input_path
    )
    assert consumed == {1, 3, 5}
    assert metadata is not None
    assert metadata["optimizer_step"] == 7
    assert metadata["consumed_count"] == 3

    input_path.write_text('{"prompt":"changed","response":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="input file changed"):
        load_standalone_resume(checkpoint_path, input_path=input_path)


def test_missing_standalone_resume_is_not_a_resume_checkpoint(
    tmp_path: Path,
) -> None:
    consumed, metadata = load_standalone_resume(tmp_path / "pretrained")
    assert consumed == set()
    assert metadata is None


def test_ray_reference_checkpoint_preserves_upstream_shuffle_resume_settings(tmp_path):
    """Both checkpoint call sites must retain main's shuffle metadata for ray_reference."""
    import ast
    from typing import Any

    import torch

    from verl_speco.standalone_tq_producer import _assert_resume_shuffle_alignment

    source = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_standalone_producer_shuffle", "_save_resume_sidecar"}]
    namespace = {"Any": Any, "torch": torch, "save_standalone_resume": save_standalone_resume}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    settings = namespace["_standalone_producer_shuffle"]({
        "speco": {"standalone_tq_producer": {"shuffle": True, "shuffle_seed": 73}},
    })
    loop = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_run_standalone_draft_training_async")
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_save_standalone_checkpoint"]
    assert len(calls) == 2
    for call in calls:
        expression = next(keyword.value for keyword in call.keywords if keyword.arg == "producer_shuffle")
        assert eval(compile(ast.Expression(expression), str(source), "eval"), {
            "feature_store_type": "ray_reference", "standalone_producer_shuffle": settings,
        }) == settings
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"prompt":"q","response":"a"}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    namespace["_save_resume_sidecar"](str(checkpoint), torch.tensor([1, 5]), step=7,
                                       input_path=str(input_path), producer_shuffle=settings)
    consumed, metadata = load_standalone_resume(checkpoint, input_path=input_path)
    assert metadata["producer_shuffle"] == settings
    _assert_resume_shuffle_alignment(metadata, shuffle=True, shuffle_seed=73,
                                     consumed_sequence_nos=consumed)


def test_standalone_resume_round_trips_file_store_data_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "manifest.jsonl"
    input_path.write_text('{"num_samples": 10}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_4"

    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=4,
        input_path=input_path,
        data_progress={"epoch": 2, "batch_index": 5},
    )

    consumed, metadata = load_standalone_resume(
        checkpoint_path, input_path=input_path
    )
    assert consumed == set()
    assert metadata is not None
    assert metadata["data_progress"] == {"epoch": 2, "batch_index": 5}

    with pytest.raises(ValueError, match="data_progress must be non-negative"):
        save_standalone_resume(
            tmp_path / "bad",
            [],
            optimizer_step=1,
            input_path=input_path,
            data_progress={"epoch": -1, "batch_index": 0},
        )


def test_standalone_resume_persists_loader_config_and_read_error_state(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "manifest.jsonl"
    input_path.write_text('{"num_samples": 10}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_6"
    loader_config = {
        "batch_size": 2,
        "world_size": 2,
        "shuffle": True,
        "seed": 7,
        "min_supervised_tokens": 1,
    }

    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=6,
        input_path=input_path,
        data_progress={
            "epoch": 0,
            "batch_index": 4,
            "read_errors": 0,
            "read_errors_exact": True,
        },
        loader_config=loader_config,
    )

    _, metadata = load_standalone_resume(checkpoint_path, input_path=input_path)
    assert metadata is not None
    assert metadata["data_loader"] == loader_config
    assert metadata["data_progress"] == {
        "epoch": 0,
        "batch_index": 4,
        "read_errors": 0,
        "read_errors_exact": True,
    }


def test_loader_config_validation_rejects_changed_settings() -> None:
    saved = {"batch_size": 2, "world_size": 1, "seed": 0, "shuffle": True}
    validate_standalone_loader_config(saved, dict(saved))
    validate_standalone_loader_config(None, {"batch_size": 4})

    with pytest.raises(ValueError, match="loader configuration changed"):
        validate_standalone_loader_config(saved, {**saved, "batch_size": 4})

    with pytest.raises(ValueError, match="loader configuration changed"):
        validate_standalone_loader_config(saved, {k: v for k, v in saved.items() if k != "seed"})


def test_build_input_fingerprint_handles_plain_jsonl_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "jsonl"
    data_dir.mkdir()
    (data_dir / "a.jsonl").write_text('{"input_ids": [1, 2]}\n', encoding="utf-8")
    (data_dir / "b.jsonl").write_text('{"input_ids": [3, 4]}\n', encoding="utf-8")

    fingerprint = build_input_fingerprint(data_dir)
    assert fingerprint["format"] == "jsonl_token_replay"
    assert set(fingerprint["files"]) == {"a.jsonl", "b.jsonl"}
    assert "sha256" in fingerprint["files"]["a.jsonl"]
    assert "mtime_ns" not in fingerprint["files"]["a.jsonl"]

    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=1,
        input_path=data_dir,
        data_progress={"epoch": 0, "batch_index": 1},
        loader_config={"batch_size": 2},
    )
    _, metadata = load_standalone_resume(checkpoint_path, input_path=data_dir)
    assert metadata is not None
    assert metadata["input_fingerprint"] == fingerprint


def test_build_input_fingerprint_handles_generic_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "shard.bin").write_bytes(b"abc")
    (data_dir / "nested").mkdir()
    (data_dir / "nested" / "index.json").write_text("{}", encoding="utf-8")

    fingerprint = build_input_fingerprint(data_dir)
    assert fingerprint["format"] == "directory"
    assert set(fingerprint["files"]) == {"shard.bin", "nested/index.json"}
    assert "sha256" in fingerprint["files"]["shard.bin"]


def test_input_fingerprint_comparison_ignores_mount_path(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "data.jsonl").write_text('{"input_ids": [1]}\n', encoding="utf-8")
    (dir_b / "data.jsonl").write_text('{"input_ids": [1]}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_1"

    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=1,
        input_path=dir_a,
    )
    _, metadata = load_standalone_resume(checkpoint_path, input_path=dir_b)
    assert metadata is not None

    (dir_b / "data.jsonl").write_text('{"input_ids": [2]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="input file changed"):
        load_standalone_resume(checkpoint_path, input_path=dir_b)


def test_save_standalone_resume_accepts_precomputed_fingerprint(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "manifest.jsonl"
    input_path.write_text('{"num_samples": 1}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_1"
    fingerprint = {"path": "/node/a", "format": "test", "files": {}}

    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=1,
        input_path=input_path,
        input_fingerprint=fingerprint,
    )
    _, metadata = load_standalone_resume(
        checkpoint_path, current_fingerprint=fingerprint
    )
    assert metadata is not None
    assert metadata["input_fingerprint"] == fingerprint

    with pytest.raises(ValueError, match="input file changed"):
        load_standalone_resume(
            checkpoint_path,
            current_fingerprint={**fingerprint, "format": "other"},
        )


def test_load_standalone_resume_accepts_precomputed_current_fingerprint(
    monkeypatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "manifest.jsonl"
    input_path.write_text('{"num_samples": 1}\n', encoding="utf-8")
    checkpoint_path = tmp_path / "draft_step_1"
    save_standalone_resume(
        checkpoint_path,
        [],
        optimizer_step=1,
        input_path=input_path,
    )
    fingerprint = build_input_fingerprint(input_path)

    def fail_recompute(*args, **kwargs):
        raise AssertionError("build_input_fingerprint should not be called")

    monkeypatch.setattr(
        "verl_speco.trainer.standalone_resume.build_input_fingerprint",
        fail_recompute,
    )
    _, metadata = load_standalone_resume(
        checkpoint_path, current_fingerprint=fingerprint
    )
    assert metadata is not None


def test_large_file_fingerprint_detects_head_and_tail_changes(
    monkeypatch, tmp_path: Path
) -> None:
    import verl_speco.trainer.standalone_resume as resume

    monkeypatch.setattr(resume, "_HASH_FULL_LIMIT_BYTES", 16)
    monkeypatch.setattr(resume, "_HASH_CHUNK_BYTES", 4)
    monkeypatch.setattr(resume, "_HASH_SAMPLE_COUNT", 3)
    path = tmp_path / "big.jsonl"
    path.write_bytes(b"0123456789abcdefghij")
    original = build_input_fingerprint(path)

    path.write_bytes(b"X123456789abcdefghij")
    assert build_input_fingerprint(path) != original

    path.write_bytes(b"0123456789abcdefghiX")
    assert build_input_fingerprint(path) != original


def test_build_input_fingerprint_includes_raw_hidden_shard_sizes(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    metadata_path = data_dir / "metadata.json"
    metadata_path.write_text('{"format": "torch_shard"}\n', encoding="utf-8")
    shard_path = data_dir / "shard_000000.pt"
    shard_path.write_bytes(b"abc")
    (data_dir / "manifest.jsonl").write_text(
        '{"path": "shard_000000.pt", "num_samples": 1}\n', encoding="utf-8"
    )

    fingerprint = build_input_fingerprint(data_dir)
    assert fingerprint["format"] == "standalone_raw_hidden"
    assert fingerprint["shards"]["shard_000000.pt"]["size_bytes"] == 3

    metadata_path.write_text('{"format": "other"}\n', encoding="utf-8")
    assert build_input_fingerprint(data_dir) != fingerprint

    metadata_path.write_text('{"format": "torch_shard"}\n', encoding="utf-8")
    shard_path.write_bytes(b"abcdef")
    assert build_input_fingerprint(data_dir) != fingerprint


def test_parse_bool_setting_handles_strings_and_numbers() -> None:
    assert parse_bool_setting(True, default=False) is True
    assert parse_bool_setting("false", default=True) is False
    assert parse_bool_setting("true", default=False) is True
    assert parse_bool_setting("0", default=True) is False
    assert parse_bool_setting("on", default=False) is True
    assert parse_bool_setting("", default=True) is True
    assert parse_bool_setting(None, default=True) is True
    with pytest.raises(ValueError, match="boolean setting"):
        parse_bool_setting("nope", default=False)
