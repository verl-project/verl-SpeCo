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
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").rglob("*.sh"))
PPO_EXAMPLES = [
    script
    for script in (ROOT / "examples").glob("*.sh")
    if not script.name.endswith("_separate_training.sh")
    and not script.name.endswith("_sft.sh")
    and "_eval" not in script.name
]


def _launch_assignments(source: str) -> dict[str, str]:
    """Extract Hydra assignments from a shell launch for baseline comparison."""

    assignments = {}
    in_launch = False
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if "python3 -m verl_speco.main" in line or "python -m verl_speco.main" in line:
            in_launch = True
            continue
        if not in_launch or "=" not in line:
            continue
        line = line.removesuffix("\\").strip()
        key, value = line.split("=", 1)
        assignments[key] = value
    return assignments


def _require_working_bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    probe = subprocess.run([bash, "--version"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("bash is present but not usable in this environment")
    return bash


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_shell_syntax_is_valid(script: Path) -> None:
    bash = _require_working_bash()
    path = str(script)
    if os.name == "nt":
        if "system32" in bash.lower():
            path = (
                "/mnt/"
                + script.drive.rstrip(":").lower()
                + "/"
                + script.relative_to(script.anchor).as_posix()
            )
        else:
            path = script.as_posix()
    subprocess.run([bash, "-n", path], check=True)


@pytest.mark.parametrize("script", PPO_EXAMPLES, ids=lambda path: path.name)
def test_example_keeps_speco_entrypoint_and_required_drafter_switches(
    script: Path,
) -> None:
    source = script.read_text(encoding="utf-8")
    if script.name == "run_qwen3-8b_v1_hard_freeze_longrun_drafter_dspark_vllm_npu.sh":
        # This wrapper changes dimensions/policy, then reuses the complete
        # launch so its entrypoint and drafter switches cannot drift.
        launch = "run_qwen3-8b_v1_hard_freeze_smoke_drafter_dspark_vllm_npu.sh"
        assert f'exec bash "${{script_dir}}/{launch}"' in source
        source = (script.parent / launch).read_text(encoding="utf-8")

    if "-vat_drafter_" in script.name or "-opd_drafter_" in script.name:
        backend = "dflash" if "dflash" in script.name else "dspark"
        assert "bash examples/" not in source
        assert f"training.{backend}_loss_mode=full_vocab" in source

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


@pytest.mark.parametrize("backend", ["dflash", "dspark"])
@pytest.mark.parametrize("method", ["opd", "vat"])
def test_standalone_objective_examples_launch_experiment_settings(
    backend: str, method: str, tmp_path: Path,
) -> None:
    bash = _require_working_bash()
    script = ROOT / "examples" / f"run_qwen3-8b-{method}_drafter_{backend}_vllm.sh"
    path = script.as_posix()
    if os.name == "nt" and "system32" in bash.lower():
        path = "/mnt/" + script.drive.rstrip(":").lower() + "/" + script.relative_to(script.anchor).as_posix()
    # Intercept the Python launch: exercise Bash expansion from another cwd,
    # including paths with spaces and the user's trailing Hydra overrides.
    result = subprocess.run(
        [bash, "-c", 'python3() { printf "%s\\n" "$@"; }; source "$1" "${@:2}"',
         "standalone-example", path, "trainer.total_training_steps=3"],
        cwd=tmp_path, env={**os.environ, "MODEL_PATH": "/models/Qwen 3",
                           "DRAFTER_PATH": "/models/drafter with spaces", "RUN_TIMESTAMP": "test"},
        capture_output=True, text=True, check=True,
    )
    args = result.stdout.splitlines()
    assert args[:2] == ["-m", "verl_speco.main"]
    assert args[-1] == "trainer.total_training_steps=3"
    assignments = dict(arg.split("=", 1) for arg in args[2:])
    assert assignments["actor_rollout_ref.model.path"] == "/models/Qwen 3"
    assert assignments["actor_rollout_ref.rollout.drafter.model_path"] == "/models/drafter with spaces"
    assert assignments["data.train_batch_size"] == "32"
    assert assignments["data.max_response_length"] == ("2048" if method == "opd" else "8192")
    assert assignments["trainer.n_gpus_per_node"] == "2"
    assert assignments["trainer.save_freq"] == "-1"
    prefix = "actor_rollout_ref.rollout.drafter.training."
    assert assignments[prefix + "collect_interval_steps"] == "2"
    assert assignments[prefix + "training_interval_steps"] == "2"
    assert assignments[prefix + "step"] == "10"
    assert assignments[prefix + "drafter_opd.enable"] == ("true" if method == "opd" else "false")
    if method == "vat":
        assert assignments[prefix + f"{backend}_per_position_loss_weight"] == "vat"
        assert assignments[prefix + "vat.accept_mode"] == "greedy"


def test_v1_hard_freeze_longrun_uses_full_dimensions_and_normal_policy() -> None:
    source = (
        ROOT / "examples" / "run_qwen3-8b_v1_hard_freeze_longrun_drafter_dspark_vllm_npu.sh"
    ).read_text(encoding="utf-8")
    for setting in (
        "export SPECO_TRAIN_BATCH_SIZE=64",
        "export SPECO_MAX_RESPONSE_LENGTH=8192",
        "export SPECO_DRAFTER_COLLECT_INTERVAL=5",
        "export SPECO_DRAFTER_TRAINING_INTERVAL=5",
        '"${freeze}.gain.practical_gain_all=0.005"',
        '"${freeze}.gain.practical_gain_hard=0.005"',
        '"${freeze}.gain.window_updates=3"',
        '"${freeze}.probe.interval_updates=3"',
    ):
        assert setting in source
    assert '"$@"' in source


def test_v1_hard_freeze_runner_selection_is_explicit_in_ray_environment() -> None:
    source = (
        ROOT / "examples" / "run_qwen3-8b_v1_hard_freeze_smoke_drafter_dspark_vllm_npu.sh"
    ).read_text(encoding="utf-8")
    # An existing Ray head does not inherit the launcher's shell exports.
    for name in ("VLLM_USE_V1", "VLLM_USE_V2_MODEL_RUNNER"):
        assert f"++ray_kwargs.ray_init.runtime_env.env_vars.{name}='\"1\"'" in source


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
    assert (
        "actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha="
        in source
    )
    assert (
        "actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha="
        in source
    )
    assert "DSPARK_MAX_WINDOW=${DSPARK_MAX_WINDOW:-0}" in source
    assert "DSPARK_MAX_WINDOW=${DSPARK_MAX_WINDOW:-512}" not in source
    assert "training.dspark_max_window=${DSPARK_MAX_WINDOW}" in source
    assert "speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids=" in source
    assert "training.dspark_target_layer_ids=" not in source


def test_standalone_tq_hidden_state_vllm_uses_separate_devices() -> None:
    source = (ROOT / "tools" / "run_qwen3-8b_drafter_hidden_state_vllm.sh").read_text(
        encoding="utf-8"
    )

    assert "VLLM_DEVICES=${VLLM_DEVICES:-0,1,2,3,4,5}" in source
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


def test_original_npu_dspark_example_keeps_repository_mrv1_contract() -> None:
    source = (ROOT / "examples" / "run_qwen3-8b_drafter_dspark_vllm_npu.sh").read_text(
        encoding="utf-8"
    )

    assert "export VLLM_USE_V2_MODEL_RUNNER=1" not in source
    assert "no-async-scheduling=True" not in source
    assert "speculative_algorithm=DSPARK" in source
    assert "spec_verify_tokens=7" in source
    assert "actor_rollout_ref.rollout.max_num_seqs=256" in source
    assert "actor_rollout_ref.rollout.max_num_batched_tokens=16384" in source
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" in source

    assignments = _launch_assignments(source)
    assert assignments["data.filter_overlong_prompts"] == "False"
    training_prefix = "actor_rollout_ref.rollout.drafter.training."
    expected_training = {
        "dspark_hard_candidate_ratio": "0",
        "dspark_hard_sample_ratio": "0",
        "collect_interval_steps": "10",
        "training_interval_steps": "10",
        "batch_size_per_gpu": "10",
        "sample_last_n_steps": "10",
    }
    for key, value in expected_training.items():
        assert assignments[training_prefix + key] == value


def test_native_mrv2_example_is_isolated_and_keeps_baseline_parameters() -> None:
    baseline_source = (
        ROOT / "examples" / "run_qwen3-8b_drafter_dspark_vllm_npu.sh"
    ).read_text(encoding="utf-8")
    mrv2_source = (
        ROOT / "examples" / "dspark_mrv2" / "run_qwen3-8b_drafter_dspark_vllm_npu.sh"
    ).read_text(encoding="utf-8")

    baseline = _launch_assignments(baseline_source)
    mrv2 = _launch_assignments(mrv2_source)
    added_keys = {
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.no-async-scheduling",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
        "data.val_batch_size",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode",
        "+actor_rollout_ref.rollout.drafter.vllm.speculative_config_overrides.enable_adaptive_verification",
        "actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha",
        "actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_with_markov",
    }
    baseline_only_keys = {
        "actor_rollout_ref.rollout.drafter.training.dspark_hard_candidate_ratio",
        "actor_rollout_ref.rollout.drafter.training.dspark_hard_sample_ratio",
        "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu",
        "actor_rollout_ref.rollout.drafter.training.sample_last_n_steps",
    }
    changed_keys = {
        "data.filter_overlong_prompts",
        "actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens",
        "actor_rollout_ref.rollout.drafter.training.publish_async",
        "actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha",
        "actor_rollout_ref.rollout.drafter.training.collect_interval_steps",
        "actor_rollout_ref.rollout.drafter.training.training_interval_steps",
    }

    assert set(mrv2) - set(baseline) == added_keys
    assert set(baseline) - set(mrv2) == baseline_only_keys
    assert {
        key: value
        for key, value in mrv2.items()
        if key not in added_keys | baseline_only_keys | changed_keys
    } == {
        key: value
        for key, value in baseline.items()
        if key not in added_keys | baseline_only_keys | changed_keys
    }
    assert (
        baseline["actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens"] == "7"
    )
    assert (
        mrv2["actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens"]
        == "${spec_verify_tokens}"
    )
    assert (
        mrv2["+actor_rollout_ref.rollout.engine_kwargs.vllm.no-async-scheduling"]
        == "False"
    )
    assert "export VLLM_USE_V1=1" in mrv2_source
    assert "export VLLM_USE_V2_MODEL_RUNNER=1" in mrv2_source
    assert (
        "ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
        in mrv2_source
    )
    assert "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES" not in mrv2_source
    assert "SPECO_WORKSPACE_ROOT" not in mrv2_source
    assert "PYTHONPATH" not in mrv2_source
    assert "ppo_gpus_per_node=${SPECO_ACCELERATOR_COUNT:-16}" in mrv2_source
    assert "ray_worker_soft_limit=${SPECO_RAY_WORKER_SOFT_LIMIT:-16}" in mrv2_source
    assert "spec_verify_tokens=${SPECO_DSPARK_VERIFY_TOKENS:-7}" in mrv2_source
    assert "validation_batch_size" not in mrv2_source
    assert "data.filter_overlong_prompts=False" in mrv2_source
    assert "data.filter_overlong_prompts=True" not in mrv2_source
    assert "data.filter_overlong_prompts_workers=256" in mrv2_source
    assert "dspark_confidence_head_alpha=1.0" in mrv2_source
    assert "dspark_confidence_loss_alpha=1.0" in mrv2_source
    assert "dspark_confidence_head_with_markov=True" in mrv2_source
    assert "enable_adaptive_verification=True" in mrv2_source
    assert "additional_config.weight_nz_mode=0" in mrv2_source
    # The V1 dynamic-verify mechanism must not be *configured* on the MRV2 recipe
    # (a header comment may still describe it as the V1 alternative).
    assert not any("dynamic_spec" in key for key in mrv2)
    assert "SPECO MRV2 run directory" not in mrv2_source
    assert "tee -a" not in mrv2_source
    assert "MODEL_PATH=${MODEL_PATH:-/path/to/model}" in mrv2_source
    assert "CKPTS_DIR=${CKPTS_DIR:-/path/to/checkpoint}" in mrv2_source
    assert "TRAIN_FILE=${TRAIN_FILE:-/path/to/train_file}" in mrv2_source
    assert "TEST_FILE=${TEST_FILE:-/path/to/test_file}" in mrv2_source
    assert (
        "DRAFTER_PATH=${DRAFTER_PATH:-/path/to/vllm-compatible-dspark-drafter}"
        in mrv2_source
    )


@pytest.fixture
def two_node_controller(monkeypatch):
    import importlib.util
    import sys
    import types

    # The controller logic under test does not open an SSH connection.
    monkeypatch.setitem(sys.modules, "paramiko", types.ModuleType("paramiko"))
    path = ROOT / "examples" / "dspark_two_node" / "run.py"
    spec = importlib.util.spec_from_file_location("two_node_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "device, selected_busy, expected_error",
    [(2, False, None), (2, True, "Logical NPU 2 is busy"), (3, False, "Cannot map logical NPU 3")],
)
def test_two_node_preflight_checks_only_selected_npu(
    two_node_controller, monkeypatch, tmp_path, device, selected_busy, expected_error
):
    from types import SimpleNamespace

    controller = two_node_controller
    bash = _require_working_bash()
    launcher = tmp_path / "examples/run_qwen3-8b_drafter_dspark_separate_training.sh"
    launcher.parent.mkdir()
    for path in (launcher, tmp_path / "config.json", tmp_path / "data.jsonl"):
        path.touch()

    busy_result = "123 vllm 59000" if selected_busy else "No process in device."
    npu_stub = f"""
npu-smi() {{
    case "$*" in
        'info -m') printf '%s\n' '0 0 0 0 Ascend910' '7 0 2 14 Ascend910' ;;
        'info -t proc-mem -i 0 -c 0') echo '123 vllm 59000' ;;
        'info -t proc-mem -i 7 -c 0') echo '{busy_result}' ;;
        'info -t health -i 7 -c 0') echo 'Health Status : OK' ;;
        *) return 1 ;;
    esac
}}
ss() {{ :; }}
"""
    monkeypatch.setattr(controller, "setup_script", lambda *_: npu_stub)

    results = []

    def capture(_host, script):
        results.append(
            subprocess.run([bash, "-ec", script], cwd=tmp_path, text=True, capture_output=True)
        )

    monkeypatch.setattr(controller, "execute", capture)
    args = SimpleNamespace(root=".", repo=".", model=".", data="data.jsonl", producer_npu=device)
    controller.check(None, args, "producer")
    (result,) = results
    if expected_error is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert expected_error in result.stdout


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("failure", [None, "start", "execute"])
def test_two_node_phase_stops_readers_before_source(
    two_node_controller, monkeypatch, resume, failure
):
    from types import SimpleNamespace

    controller = two_node_controller
    stopped = []
    monkeypatch.setattr(controller, "prepare", lambda host, args, role, *rest: role)
    monkeypatch.setattr(
        controller, "stop", lambda host, directory: stopped.append(host)
    )
    monkeypatch.setattr(controller, "wait_service", lambda *args: None)

    def operation(name):
        def run(*args, **kwargs):
            if name == failure:
                raise RuntimeError("injected phase failure")

        return run

    monkeypatch.setattr(controller, "start", operation("start"))
    monkeypatch.setattr(controller, "execute", operation("execute"))
    args = SimpleNamespace(consumer="10.0.0.1", producer="10.0.0.2", data="/data")
    if failure:
        with pytest.raises(RuntimeError, match="injected phase failure"):
            controller.phase(args, ("reader", "source"), "job", "/checkpoint", resume)
    else:
        controller.phase(args, ("reader", "source"), "job", "/checkpoint", resume)
    assert stopped == ["reader", "source"]


def test_two_node_failed_reader_shutdown_preserves_source(
    two_node_controller, monkeypatch
):
    stopped = []

    def stop(host, directory):
        stopped.append(host)
        if host == "reader":
            raise OSError("reader unreachable")

    monkeypatch.setattr(two_node_controller, "stop", stop)
    with pytest.raises(RuntimeError, match="preserving Producer resources") as error:
        two_node_controller.stop_phase("reader", "/consumer", "source", "/producer")
    assert isinstance(error.value.__cause__, OSError)
    assert stopped == ["reader"]


@pytest.mark.parametrize("width,layers", [(4096, 36), (8192, 64)])
def test_two_node_native_budget_covers_long_samples(two_node_controller, monkeypatch, width, layers):
    import json
    import io
    import shlex
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    staged = {}

    @contextmanager
    def write_remote(path, mode):
        stream = io.BytesIO()
        yield stream
        staged[Path(path).name] = stream.getvalue().decode()

    client = SimpleNamespace(
        open_sftp=lambda: nullcontext(SimpleNamespace(open=write_remote))
    )
    monkeypatch.setattr(two_node_controller, "execute",
        lambda *args: json.dumps({"num_hidden_layers": layers,"hidden_size": width,"torch_dtype":"bfloat16"}))
    args = SimpleNamespace(
        root="/jobs",
        repo="/repo",
        model="/model",
        consumer="10.0.0.1",
        producer="10.0.0.2",
        consumer_npu=0,
        producer_npu=0,
        consumer_env=None,
    )
    two_node_controller.prepare(client, args, "consumer", "test", "token", 0)
    env = dict(
        shlex.split(line.removeprefix("export "))[0].split("=", 1)
        for line in staged["job.env"].splitlines()
        if line.startswith("export ")
    )
    # Full 8k Qwen3-8B capture (five auxiliary layers + final layer), token IDs,
    # and conservative space for both aligned payload and validity registrations.
    payload_bytes = 8192 * width * 6 * 2 + 8192 * 8
    assert json.loads(env["VLLM_AUX_HIDDEN_STATE_LAYER_IDS"])[-1] == layers - 3
    registered_bytes = payload_bytes + 4 * 1024**2
    assert int(env["SPECO_NATIVE_READER_MAX_BYTES"]) >= registered_bytes
    assert int(env["SPECO_NATIVE_SOURCE_MAX_BYTES"]) >= (
        int(env["SPECO_NATIVE_SOURCE_MAX_SLOTS"]) * registered_bytes
    )


def test_two_node_ray_temp_is_unique_and_cleaned(tmp_path):
    import os
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    source = (Path(__file__).resolve().parents[2] / "tools/dspark_two_node/ray.sh").read_text()
    # Exercise the real temp-directory lifecycle without starting a Ray service.
    source = source.replace("python3 -c 'from ray.scripts.scripts import main; main()' start", "true")
    (tmp_path / "job.env").write_text("\n".join([
        "SPECO_NATIVE_RUN_ID=speco-ci-reuse", "ROLE=consumer",
        "HOST_IP=127.0.0.1", "CONSUMER_NPU=0", "",
    ]))
    paths = []
    for _ in range(2):
        subprocess.run([bash, "-c", source], cwd=tmp_path, check=True, env=os.environ.copy())
        paths.append((tmp_path / "ray.tmp").read_text().strip())
    assert paths[0] != paths[1]
    assert all(not Path(path).exists() for path in paths)
