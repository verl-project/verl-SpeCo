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
