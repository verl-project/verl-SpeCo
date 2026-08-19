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

"""Unit-style smoke tests for the SpeCo example naming checker."""

from __future__ import annotations

from pathlib import Path

from tests.special_sanity.check_example_naming import (
    ROLLOUT_BACKENDS,
    check_filename,
    main,
)


def _violations(name: str) -> list[str]:
    return check_filename(Path(f"examples/{name}"))


def test_backend_matrix_names_pass():
    for rollout_backend in ROLLOUT_BACKENDS:
        assert (
            _violations(
                f"run_qwen3-8b_drafter_anystrategy_{rollout_backend}.sh"
            )
            == []
        )


def test_npu_suffix_passes():
    assert _violations("run_qwen3-8b_drafter_eagle3_vllm_npu.sh") == []


def test_actor_backend_with_multiple_tokens_passes():
    assert (
        _violations(
            "run_qwen3-8b_actor_custom_distributed_drafter_eagle3_vllm.sh"
        )
        == []
    )


def test_actor_backend_is_optional_for_separate_training():
    assert (
        _violations("run_qwen3-8b_actor_fsdp2_drafter_eagle3_separate_training.sh")
        == []
    )


def test_arbitrary_actor_backend_passes():
    assert _violations("run_qwen3-8b_actor_newbackend_drafter_eagle3_vllm.sh") == []


def test_missing_actor_backend_rejected():
    errs = _violations("run_qwen3-8b_actor_drafter_eagle3_vllm.sh")
    assert errs and "non-empty actor backend" in errs[0]


def test_separate_training_entrypoint_passes():
    assert _violations("run_qwen3-8b_drafter_separate_training.sh") == []


def test_separate_training_may_name_its_drafter_backends():
    assert _violations("run_qwen3-8b_drafter_newstrategy_separate_training.sh") == []
    assert _violations("run_qwen3-8b_drafter_strategy_a_strategy_b_separate_training.sh") == []


def test_arbitrary_drafter_backend_passes():
    assert _violations("run_qwen3-8b_drafter_newstrategy_vllm.sh") == []


def test_missing_drafter_marker_rejected():
    errs = _violations("run_qwen3-8b_eagle3_vllm.sh")
    assert errs and "drafter" in errs[0]


def test_missing_drafter_backend_rejected():
    errs = _violations("run_qwen3-8b_drafter__vllm.sh")
    assert errs and "non-empty drafter backend" in errs[0]


def test_unknown_rollout_backend_rejected():
    errs = _violations("run_qwen3-8b_drafter_eagle3_unknown.sh")
    assert errs and "unknown rollout backend" in errs[0]


def test_unknown_suffix_rejected():
    errs = _violations("run_qwen3-8b_drafter_eagle3_vllm_fp8.sh")
    assert errs and "unknown optional suffix" in errs[0]


def test_repo_tree_passes():
    assert main(["--root", "examples", "--repo-root", "."]) == 0


def test_synthetic_violation_fails(tmp_path):
    fake = tmp_path / "examples"
    fake.mkdir(parents=True)
    (fake / "run_qwen3-8b_drafter_eagle3_vllm.sh").write_text("#!/bin/bash\n")
    (fake / "run_qwen3-8b_drafter_eagle3_vllm_fp8.sh").write_text("#!/bin/bash\n")

    rc = main(["--root", str(fake), "--repo-root", str(tmp_path)])
    assert rc == 1
