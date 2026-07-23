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
    DRAFTER_BACKENDS,
    ROLLOUT_BACKENDS,
    check_filename,
    main,
)


def _violations(name: str) -> list[str]:
    return check_filename(Path(f"examples/{name}"))


def test_backend_matrix_names_pass():
    for drafter_backend in DRAFTER_BACKENDS:
        for rollout_backend in ROLLOUT_BACKENDS:
            assert (
                _violations(
                    f"run_qwen3-8b_drafter_{drafter_backend}_{rollout_backend}.sh"
                )
                == []
            )


def test_npu_suffix_passes():
    assert _violations("run_qwen3-8b_drafter_eagle3_vllm_npu.sh") == []


def test_separate_training_entrypoint_passes():
    assert _violations("run_qwen3-8b_drafter_separate_training.sh") == []


def test_missing_drafter_marker_rejected():
    errs = _violations("run_qwen3-8b_eagle3_vllm.sh")
    assert errs and "drafter" in errs[0]


def test_unknown_drafter_backend_rejected():
    errs = _violations("run_qwen3-8b_drafter_unknown_vllm.sh")
    assert errs and "unknown drafter backend" in errs[0]


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
