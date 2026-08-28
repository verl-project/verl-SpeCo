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

import os
import shlex
import subprocess
import shutil
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
RUNNER = ROOT / "ci" / "run_example_test.sh"


def _workflow(name: str) -> dict:
    path = WORKFLOWS / name
    assert path.is_file(), f"missing workflow: {name}"
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _workflow_source(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _require_working_bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    probe = subprocess.run([bash, "--version"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("bash is present but not usable in this environment")
    return bash


def _bash_path(path: Path, bash: str) -> str:
    if os.name != "nt":
        return str(path)
    if "system32" in bash.lower():
        drive = path.drive.rstrip(":").lower()
        rest = path.relative_to(path.anchor).as_posix()
        return f"/mnt/{drive}/{rest}"
    return path.as_posix()


def _runner_script() -> str:
    return "\n".join(RUNNER.read_text(encoding="utf-8").splitlines()) + "\n"


def test_ci_layers_match_required_shape() -> None:
    rollout_workflows = {
        "gpu_vllm_unit_tests.yml",
        "gpu_sglang_unit_tests.yml",
        "npu_vllm_unit_tests.yml",
        "npu_sglang_unit_tests.yml",
    }
    expected = rollout_workflows | {
        "cpu_unit_tests.yml",
        "gpu_drafter_training_smoke.yml",
    }

    assert expected <= {path.name for path in WORKFLOWS.glob("*.yml")}
    assert "pull_request" in _workflow("cpu_unit_tests.yml")["on"]

    manual_hardware_workflows = (rollout_workflows - {"npu_vllm_unit_tests.yml"}) | {
        "gpu_drafter_training_smoke.yml",
    }
    for workflow_name in manual_hardware_workflows:
        triggers = _workflow(workflow_name)["on"]
        assert set(triggers) == {"workflow_dispatch"}

    npu_vllm_triggers = _workflow("npu_vllm_unit_tests.yml")["on"]
    assert set(npu_vllm_triggers) == {
        "pull_request",
        "push",
        "workflow_dispatch",
    }


def test_cpu_unit_workflow_is_lightweight_pr_gate() -> None:
    source = _workflow_source("cpu_unit_tests.yml")

    assert "PYTHONPATH:" in source
    assert "VERL_SPECO_UPSTREAM_ROOT:" in source
    assert "REQUIRED_VERL.txt" in source
    assert "pip install -e ." not in source
    assert "uv pip install --system -e ." not in source
    assert "python -m compileall verl_speco" in source
    assert "bash -n examples/*.sh" in source
    assert "tests/compat" in source
    assert "tests/config" in source
    assert "tests/examples" in source
    assert "tests/integration" in source
    assert "tests/ci" not in source


def test_gpu_and_npu_workflows_run_examples_on_self_hosted_runners() -> None:
    for workflow_name, accelerator, backend, expected_drafters in (
        ("gpu_vllm_unit_tests.yml", "gpu", "vllm", {"eagle3", "dflash"}),
        ("gpu_sglang_unit_tests.yml", "gpu", "sglang", {"eagle3", "dflash"}),
        (
            "npu_vllm_unit_tests.yml",
            "npu",
            "vllm",
            {"eagle3", "megatron-eagle3", "dflash", "dspark"},
        ),
        ("npu_sglang_unit_tests.yml", "npu", "sglang", {"eagle3", "dflash"}),
    ):
        source = _workflow_source(workflow_name)
        workflow = _workflow(workflow_name)
        assert "ci/run_example_test.sh" in source
        assert f"bash ci/run_example_test.sh {accelerator} {backend}" in source
        if workflow_name == "npu_vllm_unit_tests.yml":
            assert "SPECO_DEFAULT_MODEL_ROOT" not in source
            assert "SPECO_DEFAULT_DATA_ROOT" not in source
            assert "/root/.cache/huggingface/hub" in source
            assert "dapo-math-17k.parquet" in source
            assert "aime-2024.parquet" in source
        else:
            assert "SPECO_DEFAULT_MODEL_ROOT" in source
            assert "SPECO_DEFAULT_DATA_ROOT" in source
            assert "/home/runner/models" in source
            assert "/home/runner/models/hf_data" in source
        assert "SPECO_TARGET_MODEL" in source
        assert "SPECO_EAGLE3_DRAFT_MODEL" in source
        assert "SPECO_DFLASH_DRAFT_MODEL" in source
        if backend == "vllm" and accelerator == "npu":
            assert "SPECO_DSPARK_DRAFT_MODEL" in source
            assert {
                entry["drafter"]
                for entry in workflow["jobs"]["example"]["strategy"]["matrix"][
                    "include"
                ]
            } == expected_drafters
            assert {
                (entry["drafter"], entry["enable_training"])
                for entry in workflow["jobs"]["example"]["strategy"]["matrix"][
                    "include"
                ]
            } == {
                ("eagle3", "true"),
                ("megatron-eagle3", "true"),
                ("dflash", "true"),
                ("dspark", "true"),
            }
            assert {
                (entry["drafter"], entry["disable_eagle3_torch_compile"])
                for entry in workflow["jobs"]["example"]["strategy"]["matrix"][
                    "include"
                ]
            } == {
                ("eagle3", "true"),
                ("megatron-eagle3", "true"),
                ("dflash", "false"),
                ("dspark", "false"),
            }
            assert workflow["jobs"]["example"]["container"]["image"] == (
                "swr.cn-north-4.myhuaweicloud.com/"
                "mindspeed/verl0.8.0_vllm_910b_speco:v1"
            )
            assert (
                workflow["jobs"]["example"]["container"]["options"]
                == "--shm-size 16g"
            )
            assert "python -m pip install --no-deps -e ." in source
            assert "verl_speco imported from" in source
            assert workflow["jobs"]["example"]["env"]["HF_ENDPOINT"] == (
                "https://hf-mirror.com"
            )
            assert workflow["jobs"]["example"]["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
            assert "find /root/.cache/huggingface/hub" in source
            assert "Verify model paths" in source
            assert "Missing target model directory" in source
            assert "SPECO_DEFAULT_ACCELERATOR_COUNT: ${{ vars.SPECO_ACCELERATOR_COUNT || '8' }}" in source
            assert "SPECO_TOTAL_TRAINING_STEPS: ${{ vars.SPECO_TOTAL_TRAINING_STEPS || '2' }}" in source
            assert "SPECO_MAX_RESPONSE_LENGTH: ${{ vars.SPECO_MAX_RESPONSE_LENGTH || '512' }}" in source
            assert "SPECO_TRAIN_BATCH_SIZE: ${{ vars.SPECO_TRAIN_BATCH_SIZE || '64' }}" in source
            assert "SPECO_TRAIN_MAX_SAMPLES: ${{ vars.SPECO_TRAIN_MAX_SAMPLES || '128' }}" in source
            assert "SPECO_VAL_MAX_SAMPLES: ${{ vars.SPECO_VAL_MAX_SAMPLES || '32' }}" in source
            assert "SPECO_PPO_MINI_BATCH_SIZE: ${{ vars.SPECO_PPO_MINI_BATCH_SIZE || '8' }}" in source
            assert "matrix.enable_training" in source
            assert "SPECO_EAGLE3_DISABLE_TORCH_COMPILE" in source
        assert "SPECO_ACCELERATOR_COUNT" in source
        assert "SPECO_TENSOR_PARALLEL_SIZE" in source
        assert "SPECO_SEQUENCE_PARALLEL_SIZE" in source
        assert "SPECO_ENABLE_TRAINING" in source
        assert "SPECO_EXTRA_HYDRA_ARGS" in source
        for drafter in expected_drafters:
            assert drafter in source
        for job in workflow["jobs"].values():
            labels = set(job["runs-on"])
            if accelerator == "npu":
                assert labels == {"linux-aarch64-a2-8"}
            else:
                assert "self-hosted" in labels
                assert "gpu" in labels

        if accelerator == "npu":
            assert "linux-aarch64-a2-8" in source
            assert "linux-aarch64-a2-4" not in source


def test_gpu_drafter_training_smoke_covers_standalone_backends() -> None:
    workflow_name = "gpu_drafter_training_smoke.yml"
    source = _workflow_source(workflow_name)
    workflow = _workflow(workflow_name)
    matrix = workflow["jobs"]["smoke"]["strategy"]["matrix"]["include"]

    assert {
        (entry["name"], entry["script"], entry["algorithm"]) for entry in matrix
    } == {
        (
            "EAGLE-1",
            "tests/special_standalone/eagle1_gpu_smoke.py",
            "EAGLE1",
        ),
        (
            "EAGLE-2",
            "tests/special_standalone/eagle1_gpu_smoke.py",
            "EAGLE2",
        ),
        (
            "Domino",
            "tests/special_standalone/domino_gpu_smoke.py",
            "",
        ),
        (
            "P-EAGLE",
            "tests/special_standalone/peagle_gpu_smoke.py",
            "",
        ),
    }
    for entry in matrix:
        assert (ROOT / entry["script"]).is_file()

    assert set(workflow["jobs"]["smoke"]["runs-on"]) == {
        "self-hosted",
        "linux",
        "x64",
        "gpu",
    }
    assert "SPECO_TRAINING_SMOKE_TARGET_MODEL" in source
    assert "SPECO_TRAINING_SMOKE_STEPS" in source
    assert "SPECO_TRAINING_SMOKE_LR" in source
    assert "python -m pip install --no-deps -e ." in source


def test_example_runner_shell_syntax_is_valid() -> None:
    bash = _require_working_bash()
    subprocess.run(
        [bash, "-n", "-s"], input=_runner_script().encode("utf-8"), check=True
    )


def test_example_runner_covers_gpu_and_npu_backend_matrix() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "gpu/vllm/eagle3" in source
    assert "gpu/vllm/dflash" in source
    assert "gpu/vllm/dspark" in source
    assert "gpu/sglang/eagle3" in source
    assert "gpu/sglang/dflash" in source
    assert "npu/vllm/eagle3" in source
    assert "npu/vllm/dflash" in source
    assert "npu/vllm/dspark" in source
    assert "npu/sglang/eagle3" in source
    assert "npu/sglang/dflash" in source
    assert "examples/run_qwen3-8b_drafter_eagle3_vllm.sh" in source
    assert "examples/run_qwen3-8b_drafter_eagle3_sglang.sh" in source
    assert "examples/run_qwen3-8b_drafter_eagle3_vllm_npu.sh" in source
    assert "examples/run_qwen3-8b_drafter_eagle3_sglang_npu.sh" in source
    assert "examples/run_qwen3-8b_drafter_dflash_vllm.sh" in source
    assert "examples/run_qwen3-8b_drafter_dflash_vllm_npu.sh" in source
    assert "examples/run_qwen3-8b_drafter_dflash_sglang.sh" in source
    assert "examples/run_qwen3-8b_drafter_dspark_vllm.sh" in source
    assert "examples/run_qwen3-8b_drafter_dspark_vllm_npu.sh" in source


def test_example_runner_exposes_required_hydra_overrides() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "SPECO_TARGET_MODEL" in source
    assert "SPECO_TRAIN_FILE" in source
    assert "SPECO_TEST_FILE" in source
    assert "SPECO_ACCELERATOR_COUNT" in source
    assert "SPECO_TENSOR_PARALLEL_SIZE" in source
    assert "SPECO_SEQUENCE_PARALLEL_SIZE" in source
    assert "SPECO_ENABLE_TRAINING" in source
    assert "SPECO_HIDDEN_STATE_WINDOW_MIN_ROWS" in source
    assert "SPECO_HIDDEN_STATE_WINDOW_TOKENS_PER_SAMPLE" in source
    assert "SPECO_SPEC_STEPS" in source
    assert "SPECO_SPEC_TOPK" in source
    assert "SPECO_SPEC_VERIFY_TOKENS" in source
    assert "SPECO_DFLASH_NUM_ANCHORS" in source
    assert "SPECO_DFLASH_MAX_WINDOW" in source
    assert "SPECO_DSPARK_DRAFT_MODEL" in source
    assert "SPECO_DSPARK_NUM_ANCHORS" in source
    assert "SPECO_DSPARK_MAX_WINDOW" in source
    assert "SPECO_TOTAL_TRAINING_STEPS" in source
    assert "SPECO_TRAIN_MAX_SAMPLES" in source
    assert "SPECO_VAL_MAX_SAMPLES" in source
    assert "SPECO_DATALOADER_NUM_WORKERS" in source
    assert "SPECO_EXTRA_HYDRA_ARGS" in source


def test_example_runner_dry_run_covers_npu_dspark() -> None:
    bash = _require_working_bash()
    env = {
        "SPECO_DRY_RUN": "true",
        "SPECO_TARGET_MODEL": "/models/target",
        "SPECO_DSPARK_DRAFT_MODEL": "/models/dspark",
        "SPECO_TRAIN_FILE": "/data/train.parquet",
        "SPECO_TEST_FILE": "/data/test.parquet",
        "SPECO_CKPT_DIR": "/tmp/speco",
        "SPECO_ACCELERATOR_COUNT": "1",
    }
    script = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in env.items()
    )
    script += _runner_script()
    result = subprocess.run(
        [bash, "-s", "--", "npu", "vllm", "dspark"],
        env=os.environ.copy(),
        input=script.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")

    assert "example=examples/run_qwen3-8b_drafter_dspark_vllm_npu.sh" in stdout
    assert "draft_algorithm=DSPARK" in stdout
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK" in stdout
    assert "actor_rollout_ref.rollout.drafter.training.dspark_block_size=7" in stdout
    assert "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7" in stdout
