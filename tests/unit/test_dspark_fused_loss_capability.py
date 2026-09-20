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

import pytest

torch = pytest.importorskip("torch")

from verl_speco.ops.dspark_fused_loss import fused_loss_capability


def test_fused_loss_capability_rejects_cpu_with_reason() -> None:
    capability = fused_loss_capability(torch.device("cpu"))

    assert capability.available is False
    assert capability.backend is None
    assert capability.reason == "unsupported device type 'cpu'"
