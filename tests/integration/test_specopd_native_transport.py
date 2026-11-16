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
"""Native MRV2 ID transport, without loading a model or creating a Ray cluster.

Run with an explicitly reserved visible NPU, for example:
ASCEND_RT_VISIBLE_DEVICES=8 python -m pytest -q tests/integration/test_specopd_native_transport.py
"""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from verl_speco.integration import vllm_runtime
from verl_speco.integration.specopd_vllm import patch_draft_token_handler


@pytest.mark.parametrize("method", ["draft_opd", "spec_opd"])
def test_native_mrv2_npu_draft_id_transport(monkeypatch, method):
    if not torch.npu.is_available():
        pytest.skip("requires a reserved visible NPU")
    monkeypatch.setenv(vllm_runtime.SPECO_DRAFTER_CONFIG_ENV, json.dumps({
        "training": {"drafter_opd": {"enable": True, "method": method}},
    }))
    vllm_runtime.install_vllm_runtime_observability()
    utils = pytest.importorskip("vllm.v1.worker.gpu.spec_decode.utils")
    ascend_utils = pytest.importorskip("vllm_ascend.worker.v2.utils")
    patch_draft_token_handler(utils.DraftTokensHandler)
    # Match NPUModelRunner.__init__; its setter runs outside this shim.
    with ascend_utils.torch_cuda_wrapper():
        handler = utils.DraftTokensHandler(torch.device("npu:0"))
    batch = SimpleNamespace(req_ids=["native-proof"], has_structured_output_reqs=False)
    handler.set_draft_tokens(batch, torch.tensor([[11, 12, 13]], device="npu:0"))
    result = handler.get_draft_tokens()
    assert result.req_ids == ["native-proof"]
    assert result.draft_token_ids == ([[11, 12, 13]] if method == "spec_opd" else [[-1, -1, -1]])
    assert batch.has_structured_output_reqs is False
