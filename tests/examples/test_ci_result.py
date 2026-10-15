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
import ast
import asyncio
import logging
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from verl_speco.integration.ci_result import CIResult, validate_result


def _method(path, class_name, name, namespace):
    source = Path(__file__).resolve().parents[2] / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    method.decorator_list = []
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


def success():
    result = CIResult(training=True)
    result.record_metrics(
        {
            "drafter/trained": 1,
            "drafter/train_successful_steps_max": 1,
            "drafter/published": 1,
        }
    )
    result.data["completed"] = True
    return result


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.data.update(optimizer_steps=0),
        lambda r: r.data.update(publications=0),
        lambda r: r.record_workers([{"reason": "activation_failed"}]),
        lambda r: r.record_metrics({"drafter/train_worker_results_consistent": 0}),
        lambda r: r.record_workers(
            [{"errors": [{"type": "RuntimeError", "message": "training failed"}]}]
        ),
    ],
)
def test_failures_cannot_pass_after_success(mutation):
    result = success()
    mutation(result)
    with pytest.raises(ValueError):
        validate_result(result.data, training=True)


def test_retry_does_not_clear_error(tmp_path):
    result = CIResult(training=True)
    result.record_workers(
        [{"errors": [{"type": "RuntimeError", "message": "first attempt"}]}]
    )
    result.record_metrics(
        {
            "drafter/trained": 1,
            "drafter/train_successful_steps_max": 1,
            "drafter/published": 1,
        }
    )
    result.data["completed"] = True
    path = tmp_path / "result.json"
    result.write(path)
    with pytest.raises(ValueError, match="first attempt"):
        validate_result(json.loads(path.read_text()), training=True)


def test_generation_has_separate_requirements():
    result = CIResult(training=False)
    result.data["completed"] = True
    with pytest.raises(ValueError):
        validate_result(result.data, training=False)
    result.record_metrics({"training/global_step": 1})
    validate_result(result.data, training=False)
    validate_result(success().data, training=True)


def test_online_training_step_keeps_exception_details():
    # Execute the actual method without importing the accelerator/Ray stack.
    source = Path(__file__).resolve().parents[2] / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DrafterBaseTrainer"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "training_step"
    )
    namespace = {
        "Optional": Optional,
        "torch": SimpleNamespace(enable_grad=nullcontext),
        "logger": logging.getLogger(__name__),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    async def fail(*args, **kwargs):
        raise RuntimeError("optimizer exploded")

    instance = SimpleNamespace(_training_step_impl=fail)
    assert asyncio.run(namespace["training_step"](instance, 7)) is False
    assert instance.last_training_error == {
        "type": "RuntimeError",
        "message": "optimizer exploded",
        "step": 7,
    }


def test_zero_exit_training_process_still_fails_checker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "result.json"
    result = success()
    result.record_workers(
        [{"errors": [{"type": "RuntimeError", "message": "suppressed exception"}]}]
    )
    result.write(path)
    # The training process can report a normal exit; evidence remains decisive.
    assert subprocess.run([sys.executable, "-c", "pass"]).returncode == 0
    check = subprocess.run(
        [
            sys.executable,
            str(root / "ci/check_result.py"),
            str(path),
            "--training",
            "true",
        ],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
    )
    assert check.returncode != 0
    assert b"suppressed exception" in check.stderr


def test_worker_retains_error_when_second_batch_succeeds():
    class Trainer:
        optimizer_steps_total = 0
        last_training_error = None

        def clear_pending_publish_state_dict(self):
            pass

        def reset_training_metrics(self):
            pass

        def get_training_metrics(self):
            return {}

        async def training_step(self, *args, **kwargs):
            if self.last_training_error is None:
                self.last_training_error = {
                    "type": "RuntimeError",
                    "message": "first batch failed",
                }
                return False
            self.last_training_error = None
            self.optimizer_steps_total += 1
            return True

        async def cleanup_training(self, **kwargs):
            pass

        def get_training_data_status(self, **kwargs):
            return {"trainable_samples": 0}

    worker = SimpleNamespace(
        rank=0,
        worker_incarnation="worker-0",
        is_global_publish_leader=True,
        enable_drafter=True,
        in_drafter_train_group=True,
        trainer=Trainer(),
        _prepared_training_plan_id="plan",
        _prepared_training_data_version=1,
        _prepared_training_target_version=1,
        last_global_step=1,
        device_name="cpu",
    )
    method = _method(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        "train_drafter",
        {"time": time, "_preserve_process_rng_state": lambda device: nullcontext()},
    )
    plan = {
        "plan_id": "plan",
        "max_batches": 2,
        "source_global_step": 1,
        "execution_strategy": "sync",
        "worker_snapshots": {"0": {"trainable_samples": 2}},
    }
    result = asyncio.run(method(worker, plan))
    assert result["trained"] is True
    assert result["successful_steps"] == 1
    assert result["errors"] == [
        {"type": "RuntimeError", "message": "first batch failed"}
    ]
    evidence = success()
    evidence.record_workers([result])
    with pytest.raises(ValueError, match="first batch failed"):
        validate_result(evidence.data, training=True)


def test_fit_writes_evidence_after_normal_exit(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    monkeypatch.setenv("SPECO_CI_RESULT_PATH", str(path))
    trainer = SimpleNamespace(
        config={}, is_drafter_training_enabled=lambda config: True
    )

    def run():
        trainer._speco_ci_result.record_workers(
            [{"errors": [{"type": "RuntimeError", "message": "suppressed"}]}]
        )
        trainer._speco_ci_result.record_metrics(
            {
                "drafter/trained": 1,
                "drafter/train_successful_steps_max": 1,
                "drafter/published": 1,
            }
        )
        return "finished"

    trainer._speco_fit_impl = run
    fit = _method(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        "fit",
        {"os": os},
    )
    assert fit(trainer) == "finished"
    data = json.loads(path.read_text())
    assert data["completed"] is True
    with pytest.raises(ValueError, match="suppressed"):
        validate_result(data, training=True)


@pytest.mark.parametrize("contents", [None, "{}", "invalid json"])
def test_checker_rejects_absent_or_invalid_file(tmp_path, contents):
    path = tmp_path / "result.json"
    if contents is not None:
        path.write_text(contents)
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            str(root / "ci/check_result.py"),
            str(path),
            "--training",
            "true",
        ],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
    )
    assert process.returncode != 0
