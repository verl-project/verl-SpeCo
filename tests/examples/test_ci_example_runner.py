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
    expected = {
        "cpu_unit_tests.yml",
        "gpu_unit_tests.yml",
        "npu_unit_tests.yml",
    }

    assert expected <= {path.name for path in WORKFLOWS.glob("*.yml")}
    assert "pull_request" in _workflow("cpu_unit_tests.yml")["on"]
    assert "pull_request" not in _workflow("gpu_unit_tests.yml")["on"]
    assert "pull_request" in _workflow("npu_unit_tests.yml")["on"]


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
    for workflow_name, label in (
        ("gpu_unit_tests.yml", "gpu"),
        ("npu_unit_tests.yml", "npu"),
    ):
        source = _workflow_source(workflow_name)
        workflow = _workflow(workflow_name)
        assert "ci/run_example_test.sh" in source
        assert "SPECO_DEFAULT_MODEL_ROOT" in source
        assert "SPECO_DEFAULT_DATA_ROOT" in source
        assert "/home/runner/models" in source
        assert "/home/runner/models/hf_data" in source
        assert "SPECO_TARGET_MODEL" in source
        assert "SPECO_EAGLE3_DRAFT_MODEL" in source
        assert "SPECO_DFLASH_DRAFT_MODEL" in source
        if workflow_name == "npu_unit_tests.yml":
            assert "SPECO_DSPARK_DRAFT_MODEL" in source
        assert "SPECO_ACCELERATOR_COUNT" in source
        assert "SPECO_TENSOR_PARALLEL_SIZE" in source
        assert "SPECO_SEQUENCE_PARALLEL_SIZE" in source
        assert "SPECO_ENABLE_TRAINING" in source
        assert "SPECO_EXTRA_HYDRA_ARGS" in source
        if workflow_name == "npu_unit_tests.yml":
            assert "github.event.pull_request.head.repo.full_name == github.repository" in source
            assert "linux-aarch64-a2-8" in source
            assert "linux-aarch64-a2-4" not in source
            assert '"backend":"vllm","drafter":"dspark"' in source
            assert '"backend":"sglang","drafter":"eagle3"' in source
            assert '"backend":"sglang","drafter":"dflash"' in source
        else:
            matrix_entries = {
                (entry["backend"], entry["drafter"])
                for entry in workflow["jobs"]["example"]["strategy"]["matrix"]["include"]
            }
            assert {
                ("vllm", "eagle3"),
                ("vllm", "dflash"),
                ("sglang", "eagle3"),
                ("sglang", "dflash"),
            } <= matrix_entries
        for job in workflow["jobs"].values():
            if workflow_name == "npu_unit_tests.yml":
                assert set(job["runs-on"]) == {"self-hosted", "linux-aarch64-a2-8"}
            else:
                assert {"self-hosted", label} <= set(job["runs-on"])


def test_example_runner_shell_syntax_is_valid() -> None:
    bash = _require_working_bash()
    subprocess.run([bash, "-n", "-s"], input=_runner_script().encode("utf-8"), check=True)


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
    script = "".join(f"export {name}={shlex.quote(value)}\n" for name, value in env.items())
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
