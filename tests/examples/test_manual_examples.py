# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path

import pytest
import yaml

from ci.run_manual_example import prepare

ROOT = Path(__file__).resolve().parents[2]


def test_manual_workflow_covers_every_example_and_routes_hardware():
    manifest = json.loads((ROOT / "ci/manual_examples.json").read_text())
    scripts = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "examples").rglob("*.sh")
    }
    assert set(manifest) == scripts
    workflow = yaml.load(
        (ROOT / ".github/workflows/examples_e2e_manual.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert (
        set(workflow["on"]["workflow_dispatch"]["inputs"]["example"]["options"])
        == scripts
    )
    assert (
        manifest["examples/dspark_mrv2/run_qwen3-8b_drafter_dspark_vllm_npu.sh"]
        == "npu"
    )
    assert (
        manifest["examples/run_qwen3-8b_actor_veomni_drafter_dspark_vllm.sh"] == "gpu"
    )
    assert (
        manifest["examples/run_qwen3.6-35b-a3b_drafter_dspark_separate_training.sh"]
        == "npu"
    )


def test_manual_online_paths_and_strict_result(tmp_path):
    settings = {
        "MODEL_PATH": "/models/target",
        "TRAIN_FILE": "/data/train",
        "TEST_FILE": "/data/test",
        "DRAFTER_PATH": "/models/draft",
    }
    env, args, online = prepare(
        "examples/run_qwen3-8b_drafter_eagle3_vllm.sh",
        settings,
        "trainer.total_training_steps=2\n",
        tmp_path,
    )
    assert online
    assert args[-1] == "trainer.total_training_steps=2"
    assert env["SPECO_CI_RESULT_PATH"] == str(tmp_path / "result.json")
    assert env["CKPTS_DIR"] == str(tmp_path / "ckpts_dir")
    assert env["MODEL_PATH"] == "/models/target"


@pytest.mark.parametrize("settings", [[], {"MAX_STEPS": 3}, {"invalid name": "x"}])
def test_manual_rejects_invalid_environment(settings, tmp_path):
    with pytest.raises(ValueError, match="environment_json"):
        prepare("examples/run_qwen3-8b_drafter_eagle3_vllm.sh", settings, "", tmp_path)


def test_manual_rejects_unlisted_script(tmp_path):
    with pytest.raises(ValueError, match="Select an example"):
        prepare("../../other.sh", {}, "", tmp_path)


def test_manual_reports_missing_paths(tmp_path, monkeypatch):
    for name in ["MODEL_PATH", "TRAIN_FILE", "TEST_FILE", "DRAFTER_PATH"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="MODEL_PATH"):
        prepare("examples/run_qwen3-8b_drafter_eagle3_vllm.sh", {}, "", tmp_path)


@pytest.mark.parametrize("problem", ["verl_path", "speco_path", "version"])
def test_checkout_verification_rejects_stale_install(tmp_path, problem):
    from ci.verify_checkouts import verify

    args = dict(
        workspace=tmp_path,
        verl_path=tmp_path / "upstream-verl/verl/__init__.py",
        speco_path=tmp_path / "verl_speco/__init__.py",
        version="0.8.0",
        expected_ref="release/v0.8.0",
    )
    verify(**args)
    if problem == "version":
        args[problem] = "0.9.0"
    else:
        args[problem] = tmp_path / "installed/site-packages/module/__init__.py"
    with pytest.raises(RuntimeError):
        verify(**args)


def test_online_workflows_install_checkouts_and_upload_evidence():
    workflow = yaml.load(
        (ROOT / ".github/workflows/examples_e2e_manual.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    for platform in ("gpu", "npu"):
        steps = workflow["jobs"][platform]["steps"]
        commands = "\n".join(step.get("run", "") for step in steps)
        assert "uv pip install --system --no-deps -e upstream-verl -e ." in commands
        assert "python ci/verify_checkouts.py" in commands
        assert "python ci/run_manual_example.py" in commands
        upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
        assert upload["if"] == "always()"
        for evidence in ("example.log", "status.json", "result.json"):
            assert evidence in upload["with"]["path"]



@pytest.mark.parametrize("algorithm", ["eagle3", "dflash", "dspark"])
def test_manual_8b_defaults_match_automatic_ci(algorithm, tmp_path, monkeypatch):
    for key in ["MODEL_PATH", "DRAFTER_PATH", "TRAIN_FILE", "TEST_FILE"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "TARGET_MODEL": "/cache/target",
        "TRAIN_FILE": "/cache/train",
        "TEST_FILE": "/cache/test",
        algorithm.upper() + "_DRAFT_MODEL": "/cache/draft",
    }.items():
        monkeypatch.setenv("SPECO_DEFAULT_" + key, value)
    script = "examples/run_qwen3-8b_drafter_" + algorithm + "_vllm_npu.sh"
    env, _, _ = prepare(script, {}, "", tmp_path)
    assert env["MODEL_PATH"] == "/cache/target"
    assert env["DRAFTER_PATH"] == "/cache/draft"
    assert env["TRAIN_FILE"] == "/cache/train"
    assert env["TEST_FILE"] == "/cache/test"
    env, _, _ = prepare(script, {"MODEL_PATH": "/custom/target"}, "", tmp_path)
    assert env["MODEL_PATH"] == "/custom/target"


def test_manual_workflow_defaults_equal_automatic_workflow():
    workflows = ROOT / ".github/workflows"
    automatic = yaml.load(
        (workflows / "npu_vllm_unit_tests.yml").read_text(), Loader=yaml.BaseLoader
    )
    manual = yaml.load(
        (workflows / "examples_e2e_manual.yml").read_text(), Loader=yaml.BaseLoader
    )
    for key in [
        "TARGET_MODEL",
        "EAGLE3_DRAFT_MODEL",
        "DFLASH_DRAFT_MODEL",
        "DSPARK_DRAFT_MODEL",
        "TRAIN_FILE",
        "TEST_FILE",
    ]:
        assert (
            manual["env"]["SPECO_DEFAULT_" + key]
            == automatic["env"]["SPECO_DEFAULT_" + key]
        )


def test_manual_pr_checkout_and_status_permissions():
    workflow = yaml.load(
        (ROOT / ".github/workflows/examples_e2e_manual.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert "pr_number" in workflow["on"]["workflow_dispatch"]["inputs"]
    for platform in ["gpu", "npu"]:
        checkout = workflow["jobs"][platform]["steps"][0]
        assert checkout["with"]["ref"] == "${{ needs.select.outputs.target_sha }}"
        assert (
            checkout["with"]["repository"]
            == "${{ needs.select.outputs.target_repository }}"
        )
        assert checkout["with"]["persist-credentials"] == "false"
        assert "statuses" not in workflow["jobs"][platform].get("permissions", {})
    assert workflow["permissions"] == {"contents": "read"}
    report = workflow["jobs"]["report-pr"]
    assert report["runs-on"] == "ubuntu-24.04"
    assert report["permissions"] == {"statuses": "write"}
    assert not any("checkout" in step.get("uses", "") for step in report["steps"])


def test_comment_workflow_uses_only_trusted_hosted_controller():
    workflow = yaml.load(
        (ROOT / ".github/workflows/pr_e2e_comments.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"issue_comment"}
    assert workflow["concurrency"]["queue"] == "max"
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert workflow["on"]["issue_comment"]["types"] == ["created"]
    job = workflow["jobs"]["request"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert (
        job["steps"][0]["with"]["ref"]
        == "${{ github.event.repository.default_branch }}"
    )
    assert job["steps"][0]["with"]["persist-credentials"] == "false"
    cases = json.loads((ROOT / "ci/comment_cases.json").read_text())
    examples = json.loads((ROOT / "ci/manual_examples.json").read_text())
    assert set(cases.values()) == set(examples)
    assert cases["dspark-mrv2"].startswith("examples/dspark_mrv2/")
    assert "megatron-eagle3" in cases
    manual = yaml.load(
        (ROOT / ".github/workflows/examples_e2e_manual.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert "expected_sha" in manual["on"]["workflow_dispatch"]["inputs"]


def test_manual_online_smoke_profile_and_override_priority(tmp_path, monkeypatch):
    script = "examples/dspark_mrv2/run_qwen3-8b_drafter_dspark_vllm_npu.sh"
    settings = {
        "MODEL_PATH": "/m/target",
        "DRAFTER_PATH": "/m/draft",
        "TRAIN_FILE": "/d/train",
        "TEST_FILE": "/d/test",
    }
    monkeypatch.setenv(
        "SPECO_CI_VARIABLES_JSON", json.dumps({"SPECO_TOTAL_TRAINING_STEPS": "4"})
    )
    env, args, _ = prepare(script, settings, "trainer.total_training_steps=7", tmp_path)
    assert "trainer.total_training_steps=4" in args
    assert args[-1] == "trainer.total_training_steps=7"
    for field in [
        "collect_interval_steps",
        "training_interval_steps",
        "publish_interval_steps",
    ]:
        assert "actor_rollout_ref.rollout.drafter.training." + field + "=1" in args
    assert "data.max_response_length=512" in args
    settings["SPECO_TOTAL_TRAINING_STEPS"] = "5"
    _, args, _ = prepare(script, settings, "", tmp_path)
    assert "trainer.total_training_steps=5" in args


def test_separate_training_does_not_receive_online_profile(tmp_path):
    settings = {
        "MODEL_PATH": "/m/target",
        "DRAFTER_PATH": "/m/draft",
        "TRAIN_FILE": "/d/train",
        "TEST_FILE": "/d/test",
    }
    _, args, online = prepare(
        "examples/run_qwen3-8b_drafter_separate_training.sh", settings, "", tmp_path
    )
    assert not online and args == []


@pytest.mark.parametrize("algorithm", ["eagle3", "dflash", "dspark"])
def test_standard_npu_manual_reuses_automatic_runner(algorithm, tmp_path, monkeypatch):
    from ci.run_manual_example import command_for

    monkeypatch.setenv("SPECO_CI_VARIABLES_JSON", json.dumps({
        "SPECO_ACCELERATOR_COUNT": "4",
        "SPECO_TENSOR_PARALLEL_SIZE": "2",
        "SPECO_SEQUENCE_PARALLEL_SIZE": "1",
        "SPECO_TOTAL_TRAINING_STEPS": "6",
        "SPECO_TARGET_MODEL": "/models/from-environment",
    }))
    example = f"examples/run_qwen3-8b_drafter_{algorithm}_vllm_npu.sh"
    env, args, online = prepare(example, {
        "DRAFTER_PATH": "/models/draft", "TRAIN_FILE": "/data/train",
        "TEST_FILE": "/data/test", "SPECO_TENSOR_PARALLEL_SIZE": "1",
    }, "trainer.total_training_steps=9", tmp_path)
    assert online
    assert command_for(example, env, args) == [
        "bash", "ci/run_example_test.sh", "npu", "vllm", algorithm,
    ]
    assert env["SPECO_TARGET_MODEL"] == "/models/from-environment"
    assert env["SPECO_ACCELERATOR_COUNT"] == "4"
    assert env["SPECO_TENSOR_PARALLEL_SIZE"] == "1"
    assert env["SPECO_EXTRA_HYDRA_ARGS"].splitlines()[-1] == "trainer.total_training_steps=9"
    assert "trainer.total_training_steps=6" in env["SPECO_EXTRA_HYDRA_ARGS"]


def test_standard_npu_manual_default_parallelism(tmp_path):
    from ci.run_manual_example import command_for

    env = {"MODEL_PATH": "/m", "DRAFTER_PATH": "/d", "CKPTS_DIR": str(tmp_path)}
    command_for("examples/run_qwen3-8b_drafter_eagle3_vllm_npu.sh", env, [])
    assert env["SPECO_ACCELERATOR_COUNT"] == "8"
    assert env["SPECO_TENSOR_PARALLEL_SIZE"] == "1"
    assert env["SPECO_SEQUENCE_PARALLEL_SIZE"] == "1"


def test_ci_variable_resolution_preserves_overrides_and_multiline_values():
    from ci.run_manual_example import configured_environment

    variables = {
        "SPECO_DSPARK_SPEC_STEPS": "2",
        "SPECO_DRAFT_TENSOR_PARALLEL_SIZE": "2",
        "SPECO_EXTRA_HYDRA_ARGS": "trainer.total_epochs=2\ntrainer.total_training_steps=3",
        "SPECO_INVALID-NAME": "bad",
        "OTHER": "ignored",
    }
    env = configured_environment({
        "SPECO_CI_VARIABLES_JSON": json.dumps(variables),
        "SPECO_DRAFT_TENSOR_PARALLEL_SIZE": "1",
    })
    assert env["SPECO_DSPARK_SPEC_STEPS"] == "2"
    assert env["SPECO_DRAFT_TENSOR_PARALLEL_SIZE"] == "1"
    assert env["SPECO_EXTRA_HYDRA_ARGS"] == variables["SPECO_EXTRA_HYDRA_ARGS"]
    assert "OTHER" not in env and "SPECO_INVALID-NAME" not in env


@pytest.mark.parametrize("missing", ["COLLECT_DRAFTER_PATH", "DRAFT_INIT_PATH"])
def test_manual_separate_training_rejects_missing_branch_paths(missing, tmp_path, monkeypatch):
    settings = {
        "MODEL_PATH": "/models/target", "TRAIN_FILE": "/data/train",
        "TEST_FILE": "/data/test", "COLLECT_DRAFTER_PATH": "/models/collect",
        "DRAFT_INIT_PATH": "/models/initial",
    }
    settings.pop(missing)
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ValueError, match=missing):
        prepare("examples/run_qwen3-8b_drafter_domino_peagle_separate_training.sh",
                settings, "", tmp_path)
