# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise the paired recipe without loading models or starting Ray."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_example_scripts import _require_working_bash


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/bubble_time/run_qwen3-8b_drafter_dspark_vllm.sh"
TRAINING = "actor_rollout_ref.rollout.drafter.training."


def _launch(
    mode="bubble", *, keep_hot="false", extra=(), trim_cache=None, report_cache=False
):
    bash = _require_working_bash()
    path = SCRIPT.as_posix()
    if os.name == "nt" and "system32" in bash.lower():
        path = (
            "/mnt/"
            + SCRIPT.drive.rstrip(":").lower()
            + "/"
            + SCRIPT.relative_to(SCRIPT.anchor).as_posix()
        )
    # The real wrapper and base script execute, but their Python entry point
    # prints its exact arguments instead of launching training.
    cache_report = (
        'printf "__trim_cache=%s\\n" "${VERL_DRAFTER_KEEP_HOT_TRIM_CACHE-__unset__}"; '
        if report_cache
        else ""
    )
    cache_environment = (
        "unset VERL_DRAFTER_KEEP_HOT_TRIM_CACHE; "
        if trim_cache is None
        else f"export VERL_DRAFTER_KEEP_HOT_TRIM_CACHE={shlex.quote(trim_cache)}; "
    )
    command = (
        'python3() { printf "%s\\n" "$@"; ' + cache_report + "}; export -f python3; "
        f"export SPECO_TRAINING_MODE={shlex.quote(mode)} "
        f"SPECO_KEEP_HOT={shlex.quote(keep_hot)}; "
        + cache_environment
        + "unset CKPTS_DIR; exec bash "
        + shlex.join([path, *extra])
        + "\n"
    )
    result = subprocess.run(
        [bash, "-s"],
        # Binary input avoids Windows translating shell newlines into CRLF.
        input=command.encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=60,
    )
    result.stdout = result.stdout.decode("utf-8")
    result.stderr = result.stderr.decode("utf-8")
    return result


def _assignments(result):
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["-m", "verl_speco.main"]
    return dict(line.split("=", 1) for line in lines[2:])


def test_bubble_and_sync_keep_the_same_training_workload():
    bubble = _assignments(_launch())
    sync = _assignments(_launch("sync"))
    different = {key for key in bubble if bubble[key] != sync[key]}
    assert different == {
        TRAINING + "scheduler.execution.strategy",
        TRAINING + "scheduler.training_quota.enable",
        "trainer.experiment_name",
        "trainer.default_local_dir",
    }
    assert bubble[TRAINING + "step"] == "20"
    assert bubble[TRAINING + "batch_size_per_gpu"] == "4"
    assert bubble[TRAINING + "collect_interval_steps"] == "4"
    assert bubble[TRAINING + "training_interval_steps"] == "4"
    assert bubble[TRAINING + "scheduler.execution.strategy"] == "rollout_idle_worker"
    assert sync[TRAINING + "scheduler.execution.strategy"] == "sync"
    assert bubble[TRAINING + "scheduler.training_quota.enable"] == "true"
    assert sync[TRAINING + "scheduler.training_quota.enable"] == "false"
    assert bubble[TRAINING + "scheduler.training_quota.target_steps"] == "20"
    assert bubble[TRAINING + "scheduler.training_quota.trigger_mode"] == "interval"
    assert bubble[TRAINING + "scheduler.idle_worker.keep_hot_between_plans"] == "false"
    assert bubble[TRAINING + "scheduler.idle_worker.early_publish_staging"] == "false"


def test_caller_overrides_and_keep_hot_are_forwarded_without_word_splitting():
    arguments = _assignments(
        _launch(
            keep_hot="true",
            extra=(
                "trainer.total_training_steps=100",
                'trainer.logger=["console", "wandb"]',
            ),
        )
    )
    assert arguments["trainer.total_training_steps"] == "100"
    assert arguments["trainer.logger"] == '["console", "wandb"]'
    assert (
        arguments[TRAINING + "scheduler.idle_worker.keep_hot_between_plans"] == "true"
    )


def test_unknown_mode_fails_before_starting_training():
    result = _launch("typo")
    assert result.returncode == 2
    assert "must be bubble or sync" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    "mode,override,expected",
    [
        ("bubble", None, "true"),
        ("bubble", "false", "false"),
        ("sync", None, "__unset__"),
        ("sync", "false", "false"),
    ],
)
def test_cache_trim_default_is_bubble_only_and_can_be_disabled(
    mode, override, expected
):
    arguments = _assignments(_launch(mode, trim_cache=override, report_cache=True))
    assert arguments["__trim_cache"] == expected


def test_scheduler_overrides_match_structured_configuration():
    source = SCRIPT.read_text(encoding="utf-8")
    overrides = [
        line.strip().removesuffix("\\").strip()
        for line in source.splitlines()
        if line.strip().startswith(TRAINING + "scheduler.")
    ]
    overrides = [
        line.replace("${execution_strategy}", "rollout_idle_worker")
        .replace("${quota_enabled}", "true")
        .replace("${keep_hot}", "false")
        for line in overrides
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "verl_speco/config")
    ):
        config = compose(config_name="speco_base", overrides=overrides)
    scheduler = OmegaConf.select(config, TRAINING + "scheduler")
    assert scheduler.execution.strategy == "rollout_idle_worker"
    assert scheduler.training_quota.target_steps == 20
    assert scheduler.training_quota.allow_critical_path_fallback
