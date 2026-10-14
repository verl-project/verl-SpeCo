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

import functools
import os
import shlex
import subprocess
import shutil
import tempfile
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


@functools.lru_cache(maxsize=None)
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


@functools.lru_cache(maxsize=None)
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
                workflow["jobs"]["example"]["container"]["options"] == "--shm-size 16g"
            )
            assert "repository: verl-project/verl" in source
            assert "ref: release/v0.8.0" in source
            assert "python -m pip install --no-deps -e upstream-verl" in source
            assert "python -m pip install --no-deps -e ." in source
            assert "expected verl 0.8.0 base version" in source
            assert "verl_speco imported from" in source
            assert workflow["jobs"]["example"]["env"]["HF_ENDPOINT"] == (
                "https://hf-mirror.com"
            )
            assert (
                workflow["jobs"]["example"]["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
            )
            assert "find /root/.cache/huggingface/hub" in source
            assert "Verify model paths" in source
            assert "Missing target model directory" in source
            assert (
                "SPECO_DEFAULT_ACCELERATOR_COUNT: ${{ vars.SPECO_ACCELERATOR_COUNT || '8' }}"
                in source
            )
            assert (
                "SPECO_TOTAL_TRAINING_STEPS: ${{ vars.SPECO_TOTAL_TRAINING_STEPS || '2' }}"
                in source
            )
            assert (
                "SPECO_MAX_RESPONSE_LENGTH: ${{ vars.SPECO_MAX_RESPONSE_LENGTH || '512' }}"
                in source
            )
            assert (
                "SPECO_TRAIN_BATCH_SIZE: ${{ vars.SPECO_TRAIN_BATCH_SIZE || '64' }}"
                in source
            )
            assert (
                "SPECO_TRAIN_MAX_SAMPLES: ${{ vars.SPECO_TRAIN_MAX_SAMPLES || '128' }}"
                in source
            )
            assert (
                "SPECO_VAL_MAX_SAMPLES: ${{ vars.SPECO_VAL_MAX_SAMPLES || '32' }}"
                in source
            )
            assert (
                "SPECO_PPO_MINI_BATCH_SIZE: ${{ vars.SPECO_PPO_MINI_BATCH_SIZE || '8' }}"
                in source
            )
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
    assert "gpu/vllm/peagle" in source
    assert "gpu/vllm/domino" in source
    assert (
        "examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh" in source
    )
    # P-EAGLE attends through torch flex_attention, which the NPU runtime does
    # not provide, so the separate-training lane stays GPU-only.
    assert "npu/vllm/peagle" not in source
    assert "npu/vllm/domino" not in source


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
    stdout = _dry_run("dspark", platform="npu")

    assert "example=examples/run_qwen3-8b_drafter_dspark_vllm_npu.sh" in stdout
    assert "draft_algorithm=DSPARK" in stdout
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK" in stdout
    assert "actor_rollout_ref.rollout.drafter.training.dspark_block_size=7" in stdout
    assert "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7" in stdout


def test_example_runner_dry_run_omits_ulysses_overrides_for_npu_megatron() -> None:
    stdout = _dry_run("megatron-eagle3", platform="npu", accelerator_count="8")

    assert "example=examples/run_qwen3-4b_actor_megatron_drafter_eagle3_vllm_npu.sh" in stdout
    assert "draft_algorithm=EAGLE3" in stdout
    assert "ulysses_sequence_parallel_size" not in stdout


@functools.lru_cache(maxsize=None)
def _dry_run(
    drafter: str,
    platform: str = "gpu",
    backend: str = "vllm",
    accelerator_count: str = "1",
    extra_hydra_args: str | None = None,
) -> str:
    """Dump the Hydra overrides the runner would pass, without launching a job."""
    bash = _require_working_bash()
    env = {
        "SPECO_DRY_RUN": "true",
        "SPECO_TARGET_MODEL": "/models/target",
        "SPECO_EAGLE3_DRAFT_MODEL": "/models/eagle3",
        "SPECO_DFLASH_DRAFT_MODEL": "/models/dflash",
        "SPECO_DSPARK_DRAFT_MODEL": "/models/dspark",
        "SPECO_TRAIN_FILE": "/data/train.parquet",
        "SPECO_TEST_FILE": "/data/test.parquet",
        "SPECO_CKPT_DIR": "/tmp/speco",
        "SPECO_ACCELERATOR_COUNT": accelerator_count,
    }
    if extra_hydra_args is not None:
        env["SPECO_EXTRA_HYDRA_ARGS"] = extra_hydra_args
    script = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in env.items()
    )
    script += _runner_script()

    # Run from a file rather than piping the script into `bash -s`: the runner
    # exits early in dry-run mode, and on a script this size that races the
    # writer filling the stdin pipe and can deadlock.
    with tempfile.TemporaryDirectory() as tmp:
        entry = Path(tmp) / "dry_run.sh"
        entry.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [bash, _bash_path(entry, bash), platform, backend, drafter],
            cwd=ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        )
    return result.stdout.decode("utf-8", errors="replace")


def _sections(stdout: str) -> tuple[str, str]:
    """Split the dry-run dump into its collect and offline-train override blocks."""
    marker = "Hydra train overrides:"
    assert marker in stdout, stdout
    collect, _, train = stdout.partition(marker)
    _, _, collect = collect.partition("Hydra overrides:")
    return collect, train


@pytest.mark.parametrize(
    ("drafter", "collect_algorithm", "shrunk_key"),
    (
        ("peagle", "EAGLE3", "training.peagle_num_depths=2"),
        ("domino", "DFLASH", "training.domino_block_size=4"),
    ),
)
def test_example_runner_dry_run_covers_separate_training_stages(
    drafter: str, collect_algorithm: str, shrunk_key: str
) -> None:
    stdout = _dry_run(drafter)

    assert (
        "example=examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh"
        in stdout
    )
    assert "separate_training=true" in stdout
    # Stage 1 rolls out under the engine-servable algorithm whose hidden-state
    # layout this drafter consumes, not under the drafter's own algorithm.
    assert f"draft_algorithm={collect_algorithm}" in stdout

    collect, train = _sections(stdout)

    feature_store = (
        "actor_rollout_ref.rollout.drafter.training.feature_store.path="
        f"/tmp/speco/{drafter}_features"
    )
    assert (
        "actor_rollout_ref.rollout.drafter.speculative_algorithm="
        f"{collect_algorithm}" in collect
    )
    assert feature_store in collect

    # Stage 2 reads that same store through the standalone offline trainer.
    assert feature_store in train
    assert (
        f"actor_rollout_ref.rollout.drafter.model_path=/tmp/speco/{drafter}_draft_init"
        in train
    )
    assert (
        "actor_rollout_ref.rollout.drafter.checkpoint_path="
        f"/tmp/speco/{drafter}_draft_ckpts" in train
    )
    assert "actor_rollout_ref.rollout.drafter.training.max_steps=1" in train
    assert shrunk_key in train

    # The standalone launcher takes the first matching override, so the device
    # count has to reach the example through the environment instead.
    assert "speco.draft_training.num_gpus_per_node" not in train
    assert "DRAFT_TRAIN_GPUS_PER_NODE" in _runner_script()


def test_example_runner_dry_run_keeps_algorithm_knobs_out_of_the_other_lane() -> None:
    peagle = _sections(_dry_run("peagle"))[1]
    domino = _sections(_dry_run("domino"))[1]

    assert "domino_" not in peagle
    assert "peagle_" not in domino


def test_example_runner_dry_run_keeps_extra_hydra_args_on_the_collect_stage() -> None:
    collect, train = _sections(_dry_run("peagle", extra_hydra_args="trainer.nnodes=1"))

    assert "trainer.nnodes=1" in collect
    # Stage 2 is a different entrypoint with a different config tree.
    assert "trainer.nnodes=1" not in train


def test_example_runner_skips_the_offline_stage_without_training() -> None:
    source = _runner_script()

    assert 'if [[ "${enable_training}" == "true" ]]; then' in source
    assert "skipping the offline train stage" in source
    assert 'rm -rf "${feature_store_dir}"' in source


def test_separate_training_example_takes_its_device_count_from_the_env() -> None:
    example = (
        ROOT
        / "examples"
        / "run_qwen3-8b_drafter_domino_peagle_separate_training.sh"
    ).read_text(encoding="utf-8")

    assert "draft_train_gpus_per_node=${DRAFT_TRAIN_GPUS_PER_NODE:-8}" in example
