# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Allocator configuration must exist before torchrun imports torch."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from verl_speco import probe_ascend_state_parking as probe


def arguments(tmp_path):
    return SimpleNamespace(
        output=str(tmp_path / "probe"),
        tensor_parallel_size=8,
        data_parallel_size=2,
        rounds=4,
        dry_run=False,
    )


@pytest.mark.parametrize("initial", [None, "", "expandable_segments:True"])
def test_launch_owns_allocator_configuration_before_starting_processes(
    tmp_path, initial
):
    environment = {"KEEP_EXISTING_VARIABLE": "keep"}
    if initial is not None:
        environment["PYTORCH_NPU_ALLOC_CONF"] = initial
    original = dict(environment)
    command, child_env = probe.launch_plan(arguments(tmp_path), environment)
    assert environment == original
    assert child_env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert child_env["KEEP_EXISTING_VARIABLE"] == "keep"
    assert child_env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(map(str, range(16)))
    assert "--nproc_per_node=16" in command and "--worker" in command
    assert "--data-parallel-size" in command
    assert not (tmp_path / "probe").exists()


def test_existing_device_selection_and_ports_are_preserved(tmp_path):
    command, child = probe.launch_plan(
        arguments(tmp_path),
        {
            "ASCEND_RT_VISIBLE_DEVICES": "15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0",
            "HCCL_NPU_SOCKET_PORT_RANGE": "64000-64099",
            "HCCL_HOST_SOCKET_PORT_RANGE": "64100-64199",
        },
    )
    assert child["ASCEND_RT_VISIBLE_DEVICES"].startswith("15,14")
    assert child["HCCL_NPU_SOCKET_PORT_RANGE"] == "64000-64099"
    assert command[0] == sys.executable


def test_actual_child_gets_the_environment_used_by_the_launcher(tmp_path, monkeypatch):
    # Spawn a real CPU child at the same subprocess boundary; no torch/NPU imports.
    output = tmp_path / "observed.json"
    run = subprocess.run

    def cpu_child(command, *, env, check):
        assert "--worker" in command and check
        return run(
            [
                sys.executable,
                "-c",
                (
                    "import json,os,sys;open(sys.argv[1],'w').write(json.dumps("
                    "{k:os.environ.get(k) for k in ('PYTORCH_NPU_ALLOC_CONF','HCCL_NPU_SOCKET_PORT_RANGE')}))"
                ),
                str(output),
            ],
            env=env,
            check=check,
        )

    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("HCCL_NPU_SOCKET_PORT_RANGE", raising=False)
    monkeypatch.delenv("HCCL_HOST_SOCKET_PORT_RANGE", raising=False)
    monkeypatch.setattr(probe.subprocess, "run", cpu_child)
    probe.launch(arguments(tmp_path))
    assert json.loads(output.read_text()) == {
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
        "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61099",
    }


def test_dry_run_imports_no_training_dependencies_and_creates_no_output(tmp_path):
    script = (
        "import runpy,sys;runpy.run_module('verl_speco.probe_ascend_state_parking',run_name='__main__');"
        "assert not any(k in sys.modules for k in ('torch','torch_npu','deepspeed'))"
    )
    child = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        child.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--output",
            str(tmp_path / "new-output"),
            "--dry-run",
        ],
        env=child,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["environment"]["PYTORCH_NPU_ALLOC_CONF"] == probe.ALLOCATOR_CONF
    assert not (tmp_path / "new-output").exists()


def test_worker_rejects_conflicting_environment_and_reports_observed_value(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["probe", "--worker", "--output", str(tmp_path)])
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    assert "expandable_segments:True" in capsys.readouterr().err


def test_launcher_runs_only_once_and_worker_checks_topology(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--worker", "--output", str(tmp_path)])
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", probe.ALLOCATOR_CONF)
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setattr(probe, "worker", lambda args: None)
    monkeypatch.setattr(
        probe, "launch", lambda args: pytest.fail("Worker relaunched torchrun")
    )
    probe.main()
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2


def test_outer_torchrun_worker_bootstraps_missing_configuration_before_torch(tmp_path):
    env = {"CUSTOM": "keep"}
    source = probe.prepare_worker_environment(arguments(tmp_path), env, {})
    assert source == "worker_bootstrap"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == probe.ALLOCATOR_CONF
    assert env["CUSTOM"] == "keep"
    assert env["HCCL_NPU_SOCKET_PORT_RANGE"] == "61000-61099"


@pytest.mark.parametrize("module", ["torch", "torch_npu"])
def test_bootstrap_cannot_reconfigure_an_imported_runtime(tmp_path, module):
    with pytest.raises(RuntimeError, match="after torch/torch_npu import"):
        probe.prepare_worker_environment(arguments(tmp_path), {}, {module: object()})


def test_bootstrap_preserves_explicit_configuration_and_rejects_conflicts(tmp_path):
    args = arguments(tmp_path)
    env = {"PYTORCH_NPU_ALLOC_CONF": probe.ALLOCATOR_CONF}
    assert (
        probe.prepare_worker_environment(args, env, {"torch": object()}) == "inherited"
    )
    with pytest.raises(RuntimeError, match="expandable_segments:True"):
        probe.prepare_worker_environment(
            args, {"PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"}, {}
        )


def test_real_worker_entry_bootstraps_then_exits_without_importing_torch(tmp_path):
    env = os.environ.copy()
    env.pop("PYTORCH_NPU_ALLOC_CONF", None)
    env.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="16")
    script = (
        "import runpy,sys;runpy.run_module('verl_speco.probe_ascend_state_parking',run_name='__main__');"
        "assert not any(k in sys.modules for k in ('torch','torch_npu','deepspeed'))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--output",
            str(tmp_path / "not-created"),
            "--check-environment-only",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    record = json.loads(result.stdout)
    assert record["phase"] == "worker_environment"
    assert record["configuration_source"] == "worker_bootstrap"
    assert record["PYTORCH_NPU_ALLOC_CONF"] == probe.ALLOCATOR_CONF
    assert not (tmp_path / "not-created").exists()
