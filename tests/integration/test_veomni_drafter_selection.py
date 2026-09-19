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
"""Reject unsupported engines before model initialization or optional imports."""

from types import SimpleNamespace

import pytest

from verl_speco.trainer.base_trainer import DrafterBaseTrainer


@pytest.mark.parametrize(
    ("engine", "algorithm", "mesh", "sp", "message"),
    [
        ("unknown", "peagle", None, False, "Unknown drafter training engine"),
        ("veomni", "eagle3", None, False, "dense P-EAGLE"),
        ("veomni", "peagle", None, False, "requires FSDP2"),
        ("veomni", "peagle", object(), True, "without sequence parallelism"),
        ("veomni", "peagle", object(), False, "dedicated standalone processes"),
    ],
)
def test_invalid_engine_configuration(engine, algorithm, mesh, sp, message):
    trainer = object.__new__(DrafterBaseTrainer)
    trainer.rollout_dp_rank = 0
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training={"engine": engine}))
    )
    trainer.backend = SimpleNamespace(model_type=algorithm)
    trainer.fsdp_device_mesh = mesh
    trainer.use_native_dp_sp = False
    trainer.use_ulysses_sp = sp
    with pytest.raises(ValueError, match=message):
        trainer._build_draft_model()
