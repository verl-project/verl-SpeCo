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
"""Backend selection is independent of algorithms, rollout and native imports."""

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_speco.backends import training_backend as dispatch
from verl_speco.backends.factory import build_trainer_backend


def config(backend="pytorch"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "drafter": {
                        "speculative_algorithm": "DSPARK",
                        "training": {"backend": backend},
                    },
                }
            }
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"rollout": {"name": "vllm"}},
        SimpleNamespace(),
        OmegaConf.create({"rollout": {"drafter": {"training": {}}}}),
    ],
)
def test_absent_backend_retains_pytorch(value):
    assert dispatch.resolve_training_backend(value) == "pytorch"


@pytest.mark.parametrize("backend", ["pytorch", "vllm_ascend"])
def test_selection_accepts_full_and_actor_section_configs(backend):
    cfg = config(backend)
    assert dispatch.resolve_training_backend(cfg) == backend
    assert dispatch.resolve_training_backend(cfg.actor_rollout_ref) == backend


@pytest.mark.parametrize("backend", [None, "fsdp2", "vllm", "unknown"])
def test_invalid_backend_does_not_fall_back(backend):
    with pytest.raises(ValueError, match="training.backend"):
        dispatch.resolve_training_backend(config(backend))


@pytest.mark.parametrize(
    "backend,target",
    [
        ("pytorch", "verl_speco.trainer.draft_training_loop"),
        ("vllm_ascend", "verl_speco.trainer.vllm_ascend_training_loop"),
    ],
)
def test_dispatch_imports_only_selected_implementation(monkeypatch, backend, target):
    imports, calls = [], []
    cfg = config(backend)

    def run(value):
        assert value is cfg
        calls.append(value)
        return {"backend": backend}

    module = SimpleNamespace(
        log_resolved_config=lambda value: None,
        run_standalone_draft_training=run,
        run_vllm_ascend_training=run,
    )

    def load(name):
        imports.append(name)
        return module

    monkeypatch.setattr(dispatch.importlib, "import_module", load)
    assert dispatch.run_standalone_training(cfg) == {"backend": backend}
    assert imports == [target] and calls == [cfg]


def test_existing_ray_worker_arguments_reach_native_loop(monkeypatch):
    commands, events = object(), object()
    cfg = config()

    async def native(value, **kwargs):
        assert value is cfg
        assert kwargs == {"scheduled_commands": commands, "training_events": events}
        return {"native": True}

    monkeypatch.setattr(
        dispatch.importlib,
        "import_module",
        lambda name: SimpleNamespace(_run_standalone_draft_training_async=native),
    )
    assert asyncio.run(
        dispatch.run_standalone_training_async(
            cfg, scheduled_commands=commands, training_events=events
        )
    ) == {"native": True}


def test_scheduled_tq_is_rejected_before_vllm_import(monkeypatch):
    def forbidden(name):
        raise AssertionError("unsupported execution must fail before model imports")

    monkeypatch.setattr(dispatch.importlib, "import_module", forbidden)
    with pytest.raises(ValueError, match="TransferQueue"):
        asyncio.run(
            dispatch.run_standalone_training_async(
                config("vllm_ascend"), scheduled_commands=object()
            )
        )


def test_online_native_factory_cannot_ignore_vllm_ascend_selection():
    with pytest.raises(ValueError, match="online/TQ"):
        build_trainer_backend(config("vllm_ascend").actor_rollout_ref, None)


def test_default_config_and_shared_entrypoint_are_import_light():
    root = Path(__file__).resolve().parents[2]
    source = "from verl_speco.backends.training_backend import resolve_training_backend; import sys; assert resolve_training_backend({}) == 'pytorch'; assert 'torch' not in sys.modules; assert 'torch_npu' not in sys.modules"
    subprocess.run([sys.executable, "-c", source], cwd=root, check=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verl_speco.draft_train",
            "--config-name=deepseek_v4_0731_draft_trainer",
            "--cfg",
            "job",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "backend: vllm_ascend" in result.stdout
    assert "engine: ds4" not in result.stdout
