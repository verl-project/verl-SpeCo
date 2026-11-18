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
"""Reviewed Ascend runtime pairs and the V1 rollout capture boundary.

Version checks and structural checks are both required. Local packaging tags
such as vLLM's ``+empty`` do not change the reviewed public release identity.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, Version

SUPPORTED_ASCEND_RUNTIME_PAIRS = (
    ("0.25.1", "0.25.1rc1"),
    ("0.26.0", "0.26.0rc1"),
    ("0.27.1", "0.27.1rc1"),
)
CAPTURE_STATE_FIELDS = ("scheduler_output", "spec_decode_metadata", "positions")


def require_ascend_runtime(vllm_version, ascend_version):
    try:
        pair = (Version(str(vllm_version)).public, Version(str(ascend_version)).public)
    except InvalidVersion:
        pair = None
    if pair not in SUPPORTED_ASCEND_RUNTIME_PAIRS:
        supported = "; ".join(
            f"vLLM {v} / vLLM-Ascend {a}" for v, a in SUPPORTED_ASCEND_RUNTIME_PAIRS
        )
        raise RuntimeError(
            f"Ascend execution requires {supported}; got {vllm_version!r} / {ascend_version!r}"
        )
    return pair


def validate_capture_runner(runner):
    config = getattr(runner, "vllm_config", None)
    if getattr(config, "use_v2_model_runner", False):
        raise RuntimeError(
            "Rollout feature capture requires the V1 model runner; set VLLM_USE_V2_MODEL_RUNNER=0 before starting workers"
        )
    if not all(
        callable(getattr(runner, name, None))
        for name in ("_sample", "sample_tokens", "get_model")
    ):
        raise TypeError(
            "Rollout runner does not expose the reviewed V1 sampler boundary"
        )


def capture_state(runner):
    """Read named fields, never positional indices of ExecuteModelState.

    Ascend 0.26 extends the upstream named tuple with attention metadata,
    positions and batch_desc. A no-op sample may legitimately have no state.
    """
    state = getattr(runner, "execute_model_state", None)
    if state is None:
        return None
    missing = [name for name in CAPTURE_STATE_FIELDS if not hasattr(state, name)]
    if missing:
        raise TypeError(f"Unsupported Ascend ExecuteModelState fields: {missing}")
    schedule = state.scheduler_output
    if not all(
        hasattr(schedule, name)
        for name in ("total_num_scheduled_tokens", "num_scheduled_tokens")
    ):
        raise TypeError("Unsupported Ascend scheduling metadata")
    return state
