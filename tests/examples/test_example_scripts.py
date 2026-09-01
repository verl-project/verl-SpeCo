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

import subprocess
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").glob("*.sh"))
PPO_EXAMPLES = [
    script
    for script in EXAMPLES
    if not script.name.endswith("_separate_training.sh")
]


def _require_working_bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    probe = subprocess.run([bash, "--version"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("bash is present but not usable in this environment")
    return bash


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_shell_syntax_is_valid(script: Path) -> None:
    bash = _require_working_bash()
    subprocess.run([bash, "-n", str(script)], check=True)


@pytest.mark.parametrize("script", PPO_EXAMPLES, ids=lambda path: path.name)
def test_example_keeps_speco_entrypoint_and_required_drafter_switches(
    script: Path,
) -> None:
    source = script.read_text(encoding="utf-8")

    assert (
        "python3 -m verl_speco.main" in source or "python -m verl_speco.main" in source
    )
    assert "actor_rollout_ref.rollout.drafter.enable=" in source
    assert "actor_rollout_ref.rollout.drafter.enable_drafter_training=" in source
    assert "actor_rollout_ref.rollout.drafter.model_path=" in source
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=" in source
    assert (
        "actor_rollout_ref.rollout.drafter.training.collect_interval_steps=" in source
    )
    assert (
        "actor_rollout_ref.rollout.drafter.training.training_interval_steps=" in source
    )
    assert "actor_rollout_ref.rollout.drafter.training.publish_async=" in source


def test_standalone_tq_training_example_uses_unified_launcher() -> None:
    source = (
        ROOT / "examples" / "run_qwen3-8b_drafter_dspark_separate_training.sh"
    ).read_text(encoding="utf-8")

    assert "-m verl_speco.standalone_tq_training_launcher" in source
    assert "data.train_files=${TRAIN_FILE}" in source
    assert "actor_rollout_ref.rollout.drafter.enable=True" in source
    assert "actor_rollout_ref.rollout.drafter.enable_drafter_training=True" in source
    assert "actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH}" in source
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK" in source
    assert "speco.standalone_tq_producer.max_inflight_requests=" in source
    assert "speco.standalone_tq_producer.per_endpoint_concurrency=" in source
    assert "actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=" in source
    assert "actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=" in source


def test_standalone_tq_hidden_state_vllm_uses_separate_devices() -> None:
    source = (
        ROOT / "tools" / "run_qwen3-8b_drafter_hidden_state_vllm.sh"
    ).read_text(encoding="utf-8")

    assert 'VLLM_DEVICES=${VLLM_DEVICES:-0,1,2,3,4,5}' in source
    assert "service_count=$((device_count / VLLM_TP))" in source
    assert 'env "${DEVICE_ENV}=${devices}" vllm serve "${MODEL_PATH}"' in source
    assert '--tensor-parallel-size "${VLLM_TP}"' in source
    assert '--max-num-seqs "${VLLM_MAX_NUM_SEQS}"' in source
    assert '"${VLLM_HIDDEN_STATE_LAYER_IDS}"' in source
    assert '"kv_connector":"ExampleHiddenStatesConnector"' in source


def test_vllm_eagle3_example_keeps_runtime_agnostic_training_switches() -> None:
    source = (ROOT / "examples" / "run_qwen3-8b_drafter_eagle3_vllm.sh").read_text(
        encoding="utf-8"
    )

    assert "actor_rollout_ref.rollout.name=vllm" in source
    assert 'actor_rollout_ref.rollout.drafter.speculative_algorithm="EAGLE3"' in source
    assert (
        "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True"
        in source
    )
    assert "actor_rollout_ref.rollout.drafter.training.use_logits=False" in source


def test_sglang_examples_request_sglang_rollout() -> None:
    for script in (ROOT / "examples").glob("*sglang*.sh"):
        source = script.read_text(encoding="utf-8")
        assert "actor_rollout_ref.rollout.name=sglang" in source


def test_npu_vllm_example_keeps_explicit_graph_settings() -> None:
    source = (ROOT / "examples" / "run_qwen3-8b_drafter_eagle3_vllm_npu.sh").read_text(
        encoding="utf-8"
    )

    assert 'cudagraph_mode="FULL_DECODE_ONLY"' in source
    assert "cudagraph_capture_sizes=" in source
    assert "max_cudagraph_capture_size=" in source
