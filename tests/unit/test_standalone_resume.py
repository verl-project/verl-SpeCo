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

from verl_speco.trainer.standalone_resume import (
    load_standalone_resume,
    save_standalone_resume,
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
