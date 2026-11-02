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
"""Ensure diagnostics read metadata only and preserve compiler cache settings."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.performance_diagnostics import (
    compilation_metadata,
    optimizer_state_metadata,
)


def test_optimizer_metadata_never_reads_values_or_changes_storage(monkeypatch):
    state = {
        "exp_avg": torch.ones(4),
        "exp_avg_sq": torch.ones(4),
        "step": torch.tensor(2),
    }
    optimizer = SimpleNamespace(state={"p": state})
    before = {key: value.data_ptr() for key, value in state.items()}
    monkeypatch.setattr(
        torch.Tensor,
        "item",
        lambda *args: pytest.fail("device values must not be read"),
    )
    result = optimizer_state_metadata(optimizer)
    assert result["cpu_bytes"] == 32 + state["step"].element_size()
    assert result["accelerator_bytes"] == result["unknown_tensors"] == 0
    assert {key: value.data_ptr() for key, value in state.items()} == before
    assert result["first_blocking_done"] == 0


def test_compile_metadata_reports_paths_and_numeric_counters_only(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/local/inductor")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/local/triton")
    monkeypatch.setitem(
        sys.modules,
        "torch._dynamo.utils",
        SimpleNamespace(counters={"stats": {"unique_graphs": 3, "ignored": object()}}),
    )
    result = compilation_metadata()
    assert result["counters"] == {"stats/unique_graphs": 3}
    assert result["TORCHINDUCTOR_CACHE_DIR"] == "/local/inductor"
    assert result["TRITON_CACHE_DIR"] == "/local/triton"


def test_compile_cache_forwarding_respects_explicit_ray_config(monkeypatch):
    from omegaconf import OmegaConf

    from verl_speco.main import configure_compile_cache_runtime_env

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/shell/inductor")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/shell/triton")
    config = OmegaConf.create(
        {
            "ray_kwargs": {
                "ray_init": {
                    "runtime_env": {
                        "env_vars": {
                            "TORCHINDUCTOR_CACHE_DIR": "/explicit/inductor",
                            "UNCHANGED": "yes",
                        }
                    }
                }
            }
        }
    )
    result = configure_compile_cache_runtime_env(config)
    assert result == {
        "TORCHINDUCTOR_CACHE_DIR": "/explicit/inductor",
        "TRITON_CACHE_DIR": "/shell/triton",
    }
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.UNCHANGED == "yes"


def test_compile_cache_defaults_do_not_create_configuration(monkeypatch):
    from omegaconf import OmegaConf

    from verl_speco.main import configure_compile_cache_runtime_env

    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    config = OmegaConf.create({})
    assert configure_compile_cache_runtime_env(config) == {}
    assert OmegaConf.to_container(config) == {}


def test_allocator_metadata_never_synchronizes_or_reclaims():
    from verl_speco.trainer.performance_diagnostics import allocator_metadata

    class Device:
        def memory_stats(self):
            return {
                "allocated_bytes.all.current": 10,
                "reserved_bytes.all.current": 30,
                "active_bytes.all.current": 20,
                "num_alloc_retries": 2,
            }

        def synchronize(self):
            pytest.fail("diagnostics must not synchronize")

        def empty_cache(self):
            pytest.fail("diagnostics must not reclaim")

    assert allocator_metadata(Device()) == {
        "allocated_bytes": 10,
        "reserved_bytes": 30,
        "active_bytes": 20,
        "inactive_split_bytes": 0,
        "allocation_retries": 2,
        "ooms": 0,
    }
    assert allocator_metadata(object()) == {}


def test_transfer_log_cleanup_preserves_metrics_and_fallback_warnings(caplog):
    import ast
    import json
    import logging
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_record_gpu_transfer"
    )
    logger = logging.getLogger("transfer-cleanup-test")
    logger.setLevel(logging.WARNING)
    namespace = {
        "logging": logging,
        "logger": logger,
        "json": SimpleNamespace(
            dumps=lambda *a, **k: pytest.fail("disabled debug must not serialize")
        ),
    }
    exec(  # noqa: S102 - isolate production method from optional dependencies
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    trainer = SimpleNamespace(rank=0, current_rl_step=4)
    record = namespace[method.name]
    record(
        trainer,
        "optimizer_offload",
        {"elapsed_s": 0.5, "reused_bytes": 64, "fallback_reason": "none"},
    )
    assert trainer._gpu_transfer_metrics == {
        "timing_s/drafter_gpu_optimizer_offload_elapsed": 0.5,
        "drafter/gpu_optimizer_offload_reused_bytes": 64,
    }
    assert not caplog.records
    namespace["json"] = json
    record(
        trainer,
        "optimizer_offload",
        {"used_pool": 0, "fallback_reason": "shared_pool_budget"},
    )
    assert "shared_pool_budget" in caplog.text
    assert trainer._gpu_transfer_metrics["drafter/gpu_optimizer_offload_used_pool"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_stage_clock_only_synchronizes_when_explicitly_armed(enabled):
    from verl_speco.trainer.performance_diagnostics import training_stage_clock

    calls = []
    trainer = SimpleNamespace(
        _diagnostic_gpu_stage_sync=enabled,
        device_module=SimpleNamespace(synchronize=lambda: calls.append("wait")),
    )
    assert isinstance(training_stage_clock(trainer), float)
    assert calls == (["wait"] if enabled else [])


def test_idle_rejection_is_opt_in_bounded_and_uses_cached_state(monkeypatch):
    from verl_speco.integration import generation_diagnostics as diagnostics

    emitted = []
    monkeypatch.setattr(diagnostics, "emit_summary", lambda *a, **k: emitted.append(k))
    worker = SimpleNamespace(
        status="ready", memory_released=False, must_be_ready_at=None
    )
    owner = SimpleNamespace(
        global_steps=4,
        _drafter_scheduler=SimpleNamespace(
            _replica_idle_worker_groups={0: ("0", "1")},
            _idle_workers={"0": worker},
            _training_quota_data_version=4,
        ),
        _speco_ready_target_lm_head_workers={4: ("0",)},
    )
    plan = SimpleNamespace(launch=False, reason="incomplete_training_group")
    monkeypatch.delenv("VERL_SPECO_DIAGNOSTIC_IDLE_REJECTIONS", raising=False)
    diagnostics.report_idle_rejection(owner, plan)
    assert emitted == []
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_IDLE_REJECTIONS", "true")
    diagnostics.report_idle_rejection(owner, plan)
    diagnostics.report_idle_rejection(owner, plan)
    assert len(emitted) == 1
    group = emitted[0]["snapshot"]["replicas"][0]
    assert group["worker_status"] == ["ready", "unknown"]
    assert group["head_ready_workers"] == ["0"]
    owner.global_steps = 5
    diagnostics.report_idle_rejection(owner, plan)
    assert len(emitted) == 2
    diagnostics.report_idle_rejection(owner, SimpleNamespace(launch=True))
    assert len(emitted) == 2


@pytest.mark.parametrize(
    "strategy,idle,device,steps,expected",
    [
        ("rollout_idle_worker", False, "cuda", 0, True),
        ("rollout_idle_worker", False, "cuda", 1, True),
        ("rollout_idle_worker", False, "cuda", 2, False),
        ("rollout_idle_worker", True, "cuda", 0, False),
        ("rollout_idle_worker", False, "npu", 0, False),
        ("sync", False, "cuda", 0, False),
    ],
)
def test_cold_step_diagnostic_scope(
    monkeypatch, strategy, idle, device, steps, expected
):
    from verl_speco.trainer.performance_diagnostics import (
        gpu_first_step_diagnostics_enabled,
    )

    worker = SimpleNamespace(
        device_name=device, trainer=SimpleNamespace(optimizer_steps_total=steps)
    )
    monkeypatch.delenv("VERL_SPECO_DIAGNOSTIC_GPU_FIRST_STEPS", raising=False)
    assert not gpu_first_step_diagnostics_enabled(worker, strategy, idle)
    monkeypatch.setenv("VERL_SPECO_DIAGNOSTIC_GPU_FIRST_STEPS", "true")
    assert gpu_first_step_diagnostics_enabled(worker, strategy, idle) == expected


def test_new_diagnostics_propagate_without_phase_timing(monkeypatch):
    from omegaconf import OmegaConf
    from verl_speco.main import (
        configure_phase_diagnostic_runtime_env,
        drafter_diagnostic_env,
    )

    monkeypatch.delenv("VERL_SPECO_PHASE_TIMING", raising=False)
    names = (
        "VERL_SPECO_DIAGNOSTIC_GPU_FIRST_STEPS",
        "VERL_SPECO_DIAGNOSTIC_IDLE_REJECTIONS",
    )
    for name in names:
        monkeypatch.setenv(name, "true")
    config = OmegaConf.create({})
    configure_phase_diagnostic_runtime_env(config)
    for name in names:
        assert (
            OmegaConf.select(config, f"ray_kwargs.ray_init.runtime_env.env_vars.{name}")
            == "true"
        )
        assert drafter_diagnostic_env()[name] == "true"
