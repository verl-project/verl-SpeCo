# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Resolve the real shell examples without starting Python training or Ray."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
from omegaconf import OmegaConf

from verl_speco.opd_method import resolve_opd_method, capture_verify_trace_enabled
from verl_speco.specopd_config import validate_specopd_config


@pytest.mark.parametrize(
    "script,method",
    [
        ("run_qwen3-8b-specopd_drafter_dflash_vllm.sh", "spec_opd"),
        ("run_qwen3-8b-specopd_drafter_dflash_vllm.sh", "draft_opd"),
        ("run_qwen3-8b-draftopd_drafter_dflash_vllm.sh", "draft_opd"),
        ("run_qwen3-8b-draftopd_drafter_dspark_vllm.sh", "draft_opd"),
    ],
)
def test_resolved_launch_selects_one_runnable_config(script, method):
    root = Path(__file__).resolve().parents[2]
    bash = shutil.which("bash")
    if bash is None and Path("C:/Program Files/Git/bin/bash.exe").exists():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("bash is unavailable")
    env = {**os.environ, "OPD_METHOD": method, "N_GPUS": "8"}
    for key in ("MODEL_PATH", "DRAFTER_PATH", "TRAIN_FILE", "TEST_FILE", "CKPTS_DIR"):
        env[key] = "/unused/test-path"
    # Follow the existing wrapper/base scripts. Intercept only the final
    # training command and emit its exact argv, including rightmost overrides.
    harness = 'bash() { source "$@"; }; python3() { printf "%s\\0" "$@"; }; source "$1"'
    output = subprocess.run(
        [bash, "-c", harness, "opd-config-test", f"examples/{script}"],
        cwd=root,
        env=env,
        capture_output=True,
        check=True,
    )
    argv = output.stdout.decode().rstrip("\0").split("\0")
    assert argv[:2] == ["-m", "verl_speco.main"]
    config = OmegaConf.merge(
        OmegaConf.load(root / "verl_speco/config/speco_base.yaml"),
        OmegaConf.from_dotlist([arg.lstrip("+") for arg in argv[2:]]),
    )
    training = config.actor_rollout_ref.rollout.drafter.training
    assert resolve_opd_method(training) == method
    spec = validate_specopd_config(config)
    assert spec.enable == (method == "spec_opd")
    assert config.actor_rollout_ref.model.use_remove_padding == (method == "draft_opd")
    assert bool(capture_verify_trace_enabled(training)) == (method == "draft_opd")
    if spec.enable:
        assert (
            config.actor_rollout_ref.model.override_config.attn_implementation
            == "eager"
        )
        assert config.actor_rollout_ref.actor.ulysses_sequence_parallel_size == 1
