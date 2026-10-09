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

"""CPU contract check against the installed Ascend worker and MRV2 output.

No model, device allocation, or distributed process group is needed. The real
worker dispatch and real AsyncOutput.get_output are used; model computation
and its copy-completion event are supplied as fixtures.
"""

from types import SimpleNamespace

import pytest

from verl_speco.integration import vllm_runtime
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

np = pytest.importorskip("numpy")


def test_native_ascend_split_sampling_to_trainer(monkeypatch, tmp_path, request):
    worker_module = pytest.importorskip("vllm_ascend.worker.worker")
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput

    monkeypatch.delenv(vllm_runtime.SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)
    calls = []
    # Construct only the CPU-ready output state; use the real method that
    # truncates padded token rows after the original synchronization boundary.
    output = AsyncOutput.__new__(AsyncOutput)
    output.copy_event = SimpleNamespace(synchronize=lambda: calls.append("copy_wait"))
    output.model_runner_output = ModelRunnerOutput(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
    output.sampled_token_ids = np.array([[1, 2, 3, -1], [4, -1, -1, -1]])
    output.num_sampled_tokens_np = np.array([3, 1])
    output.num_nans = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}
    output.routed_experts_cpu = None
    output._has_fault = None

    # Match the real vLLM extension injection, including its conflict check.
    worker_cls = worker_module.NPUWorker
    extension = vllm_runtime.SpecoVLLMColocateWorkerExtension
    for name in dir(extension):
        if not name.startswith("__"):
            assert not hasattr(worker_cls, name), name
    original_bases = worker_cls.__bases__
    worker_cls.__bases__ = original_bases + (extension,)
    request.addfinalizer(lambda: setattr(worker_cls, "__bases__", original_bases))
    worker = worker_cls.__new__(worker_cls)
    worker.rank = 0
    worker.local_rank = 0
    worker.log_memory_stats = lambda: None
    worker._pp_send_work = []
    worker.profiler = None
    worker.vllm_config = SimpleNamespace(
        additional_config={vllm_runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(tmp_path / ".spec_decode_stats")},
        parallel_config=SimpleNamespace(world_size=2, tensor_parallel_size=2),
    )
    worker.model_runner = SimpleNamespace(
        execute_model=lambda *args: None,
        sample_tokens=lambda *args: output,
    )
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: SimpleNamespace(msmonitor_use_daemon=False))
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    schedule = SimpleNamespace(
        total_num_scheduled_tokens=7,
        scheduled_spec_decode_tokens={"a": [5, 6, 7], "b": [8, 9]},
    )
    assert worker.execute_model(schedule) is None
    assert calls == []
    returned = worker.sample_tokens(None)
    assert returned is output
    assert isinstance(returned, AsyncModelRunnerOutput)
    assert calls == []
    assert not (tmp_path / ".spec_decode_stats").exists()
    resolved = returned.get_output()
    assert calls == ["copy_wait"]
    assert resolved.sampled_token_ids == [[1, 2, 3], [4]]

    trainer = SpecoV1Mixin.__new__(SpecoV1Mixin)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)))
    assert trainer._speco_v1_spec_decode_sidecar_metrics() == {
        "drafter/spec_decode/mean_acceptance_length": 2.0,
    }
