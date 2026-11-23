# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import argparse
import importlib.util
import os
import re
import shlex
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
import pytest
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin
from tests.examples.test_example_scripts import _require_working_bash

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "compare_v1", ROOT / "tools/compare_v1_cotrain_draft_opd.py"
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def capture(script, extra=()):
    bash = _require_working_bash()
    path = script.as_posix()
    result = subprocess.run(
        [
            bash,
            "-c",
            'python3() { printf "%s\\0" "$@"; }; source "$1" "${@:2}"',
            "capture",
            path,
            *extra,
        ],
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "MODEL_PATH": "/models/target",
            "DRAFTER_PATH": "/models/draft",
        },
    )
    args = result.stdout.decode().rstrip("\0").split("\0")
    assert args[:2] == ["-m", "verl_speco.main"]
    assert "--config-name=speco_v1_trainer" in args
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "verl_speco/config")
    ):
        cfg = compose(
            config_name="speco_v1_trainer",
            overrides=[arg for arg in args[2:] if not arg.startswith("--")],
        )
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    return cfg


@pytest.mark.parametrize("backend", ["dflash", "dspark"])
def test_new_v1_launch_composes(backend):
    cfg = capture(ROOT / f"examples/run_qwen3-8b-opd_v1_drafter_{backend}_vllm.sh")
    assert cfg.trainer.use_v1 and cfg.transfer_queue.enable
    assert cfg.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable
    assert cfg.trainer.v1.sync.parameter_sync_step == 1


@pytest.mark.parametrize("backend", ["dflash", "dspark"])
@pytest.mark.parametrize("matched", [False, True])
def test_comparison_reuses_examples_and_composes_both_arms(tmp_path, backend, matched):
    args = argparse.Namespace(
        backend=backend,
        model="/models/target",
        drafter="/models/draft",
        train_file="/data/train.parquet",
        val_file="/data/val.parquet",
        seed=42,
        temperature=0.6,
        steps=5,
        matched_full_vocab=matched,
        output=tmp_path,
        bash="bash",
        override=[],
        compose_only=False,
    )
    configs = {}
    for arm, command in compare.commands(args).items():
        configs[arm] = capture(Path(command[1]), command[2:])
        assert configs[arm].trainer.use_v1
        assert configs[arm].actor_rollout_ref.model.path == args.model
        assert configs[arm].data.train_files == args.train_file
    first, second = configs.values()
    assert first.data == second.data
    assert first.actor_rollout_ref.actor == second.actor_rollout_ref.actor
    assert not first.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable
    assert second.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable
    assert first.trainer.default_local_dir != second.trainer.default_local_dir


def test_summary_rejects_incomplete_or_missing_training_evidence():
    raw = "\n".join(
        f"step:{i} - drafter/train_optimizer_step_max:{i} - drafter/published:1 - dspark/opd/trace_blocks:4"
        for i in (1, 2)
    )
    stats = compare.summarize(raw, 2)
    compare.check_training(stats, opd=True, backend="dspark")
    assert stats["dspark/opd/trace_blocks"]["mean"] == 4
    with pytest.raises(ValueError, match="missing steps"):
        compare.summarize(raw, 3)
    del stats["dspark/opd/trace_blocks"]
    with pytest.raises(ValueError, match="trace_blocks"):
        compare.check_training(stats, opd=True, backend="dspark")


def compose_source(script, extra=()):
    """Resolve the simple assignment/continuation recipes without invoking Bash."""
    variables = {"repo_root": ROOT.as_posix(), "RUN_TIMESTAMP": "test"}

    def expand(text):
        for _ in range(10):
            updated = re.sub(
                r"\$\{([^{}]+)\}",
                lambda m: variables.get(m[1].split(":-")[0], m[1].split(":-", 1)[-1]),
                text,
            )
            if updated == text:
                break
            text = updated
        return text

    overrides = []
    launch = False
    for line in script.read_text(encoding="utf-8").splitlines():
        line = line.strip().removesuffix("\\").strip()
        if "python3 -m verl_speco.main" in line:
            launch = True
            continue
        if not launch:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)=(.*)", line)
            if match and "$(" not in match[2]:
                variables[match[1]] = expand(match[2].strip("\"'"))
        else:
            overrides.extend(expand(arg) for arg in shlex.split(line) if "=" in arg)
    overrides.extend(arg for arg in extra if not arg.startswith("--"))
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "verl_speco/config")
    ):
        cfg = compose(config_name="speco_v1_trainer", overrides=overrides)
    SpecoV1Mixin._speco_validate_v1_training_config(cfg)
    return cfg


@pytest.mark.parametrize("backend", ["dflash", "dspark"])
def test_new_v1_config_without_shell(backend):
    cfg = compose_source(
        ROOT / f"examples/run_qwen3-8b-opd_v1_drafter_{backend}_vllm.sh"
    )
    assert cfg.trainer.use_v1 and cfg.transfer_queue.enable
    assert cfg.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable


@pytest.mark.parametrize("backend", ["dflash", "dspark"])
@pytest.mark.parametrize("matched", [False, True])
def test_comparison_configs_without_shell(tmp_path, backend, matched):
    args = argparse.Namespace(
        backend=backend,
        model="/models/target",
        drafter="/models/draft",
        train_file="/data/train.parquet",
        val_file="/data/val.parquet",
        seed=42,
        temperature=0.6,
        steps=5,
        matched_full_vocab=matched,
        output=tmp_path,
        bash="bash",
        override=[],
        compose_only=False,
    )
    configs = {
        arm: compose_source(Path(cmd[1]), cmd[2:])
        for arm, cmd in compare.commands(args).items()
    }
    first, second = configs.values()
    assert first.data == second.data
    assert first.actor_rollout_ref.actor == second.actor_rollout_ref.actor
    assert first.actor_rollout_ref.model.path == args.model
    assert not first.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable
    assert second.actor_rollout_ref.rollout.drafter.training.drafter_opd.enable


@pytest.mark.parametrize(
    "mode, gpu_local_steps", [("colocate_async", 1), ("separate_async", 2)]
)
@pytest.mark.parametrize("matched", [True, False])
def test_async_comparison_composes_native_v1_recipe(
    tmp_path, mode, gpu_local_steps, matched
):
    args = argparse.Namespace(
        mode=mode,
        backend="dspark",
        model="/models/target",
        drafter="/models/draft",
        train_file="/data/train.parquet",
        val_file="/data/val.parquet",
        seed=42,
        temperature=1.0,
        steps=5,
        max_policy_staleness=1,
        max_drafter_staleness=0,
        matched_full_vocab=matched,
        output=tmp_path,
        bash="bash",
        override=[],
        compose_only=False,
    )
    configs = []
    for cmd in compare.commands(args).values():
        script = Path(cmd[1])
        assert (
            script
            == ROOT / f"examples/dspark_mrv2/run_qwen3-4b_drafter_dspark_{mode}_vllm.sh"
        )
        cfg = compose_source(script, cmd[2:])
        assert cfg.trainer.v1.trainer_mode == mode
        assert cfg.trainer.v1[mode].get("parameter_sync_step", 1) == gpu_local_steps
        assert cfg.trainer.use_v1 and cfg.transfer_queue.enable
        configs.append(cfg)
    first, second = configs
    assert first.data == second.data
    assert first.actor_rollout_ref.actor == second.actor_rollout_ref.actor
    baseline = first.actor_rollout_ref.rollout.drafter.training
    treatment = second.actor_rollout_ref.rollout.drafter.training
    assert first.actor_rollout_ref.rollout.drafter.enable_drafter_training
    assert not baseline.drafter_opd.enable
    assert baseline.dspark_loss_mode == ("full_vocab" if matched else "restricted_ce")
    assert treatment.drafter_opd.enable and treatment.drafter_opd.capture_verify_trace
    assert treatment.drafter_opd.max_policy_staleness == 1
    assert treatment.dspark_loss_mode == "full_vocab"
    assert first.trainer.default_local_dir != second.trainer.default_local_dir
