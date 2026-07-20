from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").glob("*.sh"))


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


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_keeps_speco_entrypoint_and_required_drafter_switches(script: Path) -> None:
    source = script.read_text(encoding="utf-8")

    assert "python3 -m verl_speco.main" in source or "python -m verl_speco.main" in source
    assert "actor_rollout_ref.rollout.drafter.enable=" in source
    assert "actor_rollout_ref.rollout.drafter.enable_drafter_training=" in source
    assert "actor_rollout_ref.rollout.drafter.model_path=" in source
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=" in source
    assert "actor_rollout_ref.rollout.drafter.training.collect_interval_steps=" in source
    assert "actor_rollout_ref.rollout.drafter.training.training_interval_steps=" in source
    assert "actor_rollout_ref.rollout.drafter.training.publish_async=" in source


def test_vllm_eagle3_example_keeps_runtime_agnostic_training_switches() -> None:
    source = (ROOT / "examples" / "run_qwen3-8b_drafter_eagle3_vllm.sh").read_text(encoding="utf-8")

    assert "actor_rollout_ref.rollout.name=vllm" in source
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=\"EAGLE3\"" in source
    assert "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True" in source
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
