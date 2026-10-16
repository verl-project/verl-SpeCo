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
"""CPU-light contract tests for SPECO RL-Insight instrumentation.

These pin the observability contract (lane ids, span names, per-rank metric
prefix/labels) that the bridge and the trainer/worker call sites must agree on.
They deliberately avoid importing the Ray/accelerator stack: methods are loaded
from source and executed with stubbed bridge primitives, matching the pattern
used by ``tests/examples/test_ci_result.py``.
"""

import ast
import asyncio
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "verl_speco/workers/speco_worker.py"
TRAINER = ROOT / "verl_speco/trainer/speco_ray_trainer.py"

WORKER_SPANS = ["drafter/worker_train_loop", "drafter/worker_publish_snapshot", "drafter/worker_cleanup"]
TRAINER_SPANS = ["drafter/collect", "drafter/prepare", "drafter/train", "drafter/publish"]


def _load_method(path, class_name, name, globals_=None):
    source = Path(path)
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
    namespace = dict(globals_ or {})
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


class _Trainer:
    optimizer_steps_total = 0
    last_training_error = None

    def clear_pending_publish_state_dict(self):
        pass

    def reset_training_metrics(self):
        pass

    def get_training_metrics(self):
        return {}

    async def training_step(self, *args, **kwargs):
        self.optimizer_steps_total += 1
        return True

    async def cleanup_training(self, **kwargs):
        pass

    def get_training_data_status(self, **kwargs):
        return {"trainable_samples": 0}

    def prepare_model_state_dict_for_publish(self, *args, **kwargs):
        return True

    def record_training_timing(self, *args, **kwargs):
        pass


def _worker(rank, labels, emitted):
    return SimpleNamespace(
        rank=rank,
        replica_rank=rank // 2,
        is_global_publish_leader=(rank == 0),
        worker_incarnation="worker",
        enable_drafter=True,
        in_drafter_train_group=True,
        trainer=_Trainer(),
        _prepared_training_plan_id="plan",
        _prepared_training_data_version=1,
        _prepared_training_target_version=1,
        last_global_step=1,
        device_name="cpu",
        _speco_rl_insight_labels=lambda: labels,
        _speco_emit_rl_insight_worker_metrics=emitted,
    )


def test_worker_labels_are_per_rank_and_step_free():
    labels = _load_method(WORKER, "SpecoWorker", "_speco_rl_insight_labels")
    worker = SimpleNamespace(rank=3, replica_rank=1, is_global_publish_leader=False)

    assert labels(worker) == {
        "worker_rank": 3,
        "replica_rank": 1,
        "publish_leader": 0,
    }


def test_worker_metrics_use_rank_labels_and_prefixed_names():
    calls = []

    def fake_log(prefix, metrics, **labels):
        calls.append((prefix, dict(metrics), dict(labels)))

    emit = _load_method(
        WORKER,
        "SpecoWorker",
        "_speco_emit_rl_insight_worker_metrics",
        {"_rl_insight_log_scalar_metrics": fake_log},
    )
    labels = {"worker_rank": 2, "replica_rank": 1, "publish_leader": 0}
    result = {
        "successful_steps": 2,
        "attempted_steps": 3,
        "optimizer_step": 4,
        "eagle3/loss": 0.5,
        "eagle3/top1_acc": 0.9,
        "errors": [],
    }

    emit(SimpleNamespace(), result, labels)

    assert calls[0][0] == "speco_drafter_worker_"
    assert calls[0][2] == labels
    assert calls[0][1]["successful_steps"] == 2
    assert calls[1][1] == {"eagle3/loss": 0.5, "eagle3/top1_acc": 0.9}


def test_worker_train_drafter_emits_expected_lane_and_spans():
    spans = []

    def fake_trace(state_name, *, state_lane_id=None, **labels):
        spans.append((state_name, state_lane_id, dict(labels)))
        return nullcontext()

    method = _load_method(
        WORKER,
        "SpecoWorker",
        "train_drafter",
        {
            "time": time,
            "_preserve_process_rng_state": lambda device: nullcontext(),
            "_rl_insight_trace_state": fake_trace,
        },
    )
    labels = {"worker_rank": 5, "replica_rank": 2, "publish_leader": 0}
    worker = _worker(5, labels, emitted=lambda *args, **kwargs: None)
    plan = {
        "plan_id": "plan",
        "max_batches": 1,
        "source_global_step": 1,
        "execution_strategy": "sync",
        "publish_after_success": True,
        "worker_snapshots": {"5": {"trainable_samples": 1}},
    }

    result = asyncio.run(method(worker, plan))

    assert result["trained"] is True
    assert [span[0] for span in spans] == WORKER_SPANS
    assert all(span[1] == "drafter_worker_5" for span in spans)
    assert all(span[2]["worker_rank"] == 5 for span in spans)
    assert all(span[2]["replica_rank"] == 2 for span in spans)
    assert all(span[2]["publish_leader"] == 0 for span in spans)
    assert spans[0][2]["max_batches"] == 1


def test_trainer_uses_single_lane_for_all_phase_spans():
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))

    lane_value = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "_SPECO_RL_INSIGHT_TRAINER_LANE"
            for target in node.targets
        ):
            lane_value = ast.literal_eval(node.value)
    assert lane_value == "speco_trainer"

    spans = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_rl_insight_trace_state"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("drafter/")
        ):
            continue
        uses_trainer_lane = any(
            keyword.arg == "state_lane_id"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "_SPECO_RL_INSIGHT_TRAINER_LANE"
            for keyword in node.keywords
        )
        spans[node.args[0].value] = uses_trainer_lane

    assert spans == {name: True for name in TRAINER_SPANS}


def test_forward_rl_insight_worker_env_injects_project_and_experiment(monkeypatch):
    omegaconf = pytest.importorskip("omegaconf")
    from verl_speco import main

    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "http://server:18080")
    config = omegaconf.OmegaConf.create(
        {
            "trainer": {
                "logger": ["console", "rl_insight"],
                "project_name": "proj",
                "experiment_name": "exp",
            },
            "ray_kwargs": {"ray_init": {"num_cpus": None}},
        }
    )

    main._forward_rl_insight_worker_env(config)

    env_vars = config.ray_kwargs.ray_init.runtime_env.env_vars
    assert env_vars["VERL_RL_INSIGHT_PROJECT_NAME"] == "proj"
    assert env_vars["VERL_RL_INSIGHT_EXPERIMENT_NAME"] == "exp"
    assert env_vars["RL_INSIGHT_SERVER_URL"] == "http://server:18080"


def test_forward_rl_insight_worker_env_is_noop_without_logger(monkeypatch):
    omegaconf = pytest.importorskip("omegaconf")
    from verl_speco import main

    config = omegaconf.OmegaConf.create(
        {
            "trainer": {"logger": ["console"], "project_name": "proj"},
            "ray_kwargs": {"ray_init": {}},
        }
    )

    main._forward_rl_insight_worker_env(config)

    assert "runtime_env" not in config.ray_kwargs.ray_init
