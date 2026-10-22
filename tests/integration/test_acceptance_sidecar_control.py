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
"""Sidecar opt-out must not alter generation counters or Bubble event buses."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.integration import vllm_runtime as runtime


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.setenv(runtime.SPECO_DRAFTER_CONFIG_ENV, "")
    monkeypatch.setenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", "")


def drafter(mode="sync", enabled=False):
    return {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "EAGLE3",
        "model_path": "/models/drafter",
        "rollout": {"spec_steps": 3},
        "training": {"scheduler": {"execution": {"strategy": mode}, "idle_worker": {}}},
        "vllm": {"acceptance_sidecar_enabled": enabled},
    }


@pytest.mark.parametrize("mode", ["sync", "rollout_idle_worker"])
def test_launcher_opt_out_clears_stale_paths_and_preserves_native_config(
    monkeypatch, tmp_path, mode
):
    monkeypatch.setattr(runtime, "install_upstream_vllm_runtime_bridge", lambda: True)
    cfg = drafter(mode)
    cfg[runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY] = str(tmp_path / "stale")
    config = {
        "trainer": {"default_local_dir": str(tmp_path)},
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": cfg,
                "engine_kwargs": {
                    "vllm": {
                        "additional_config": {
                            "existing": 1,
                            runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(
                                tmp_path / "old"
                            ),
                        }
                    }
                },
            }
        },
    }
    spec = runtime.configure_vllm_runtime_from_config(config)
    assert spec["method"] == "eagle3" and spec["num_speculative_tokens"] == 3
    assert config["trainer"]["default_local_dir"] == str(tmp_path)
    assert cfg[runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY] is None
    assert config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"][
        "additional_config"
    ] == {"existing": 1}
    assert runtime._vllm_spec_decode_sidecar_dir() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["sync", "rollout_idle_worker"])
def test_server_restore_preserves_opt_out_without_changing_bus(
    monkeypatch, tmp_path, mode
):
    server = drafter(mode, enabled=True)
    # Simulate upstream retaining stale/default vLLM fields.
    if mode == "rollout_idle_worker":
        server["training"]["scheduler"]["idle_worker"]["event_bus_name"] = (
            "authoritative-bus"
        )
    embedded = drafter(mode, enabled=False)
    monkeypatch.setenv(
        runtime.SPECO_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(tmp_path),
            }
        ),
    )
    resolved, source = runtime._ensure_vllm_server_drafter_env(
        {"rollout": {"drafter": server}}, json.dumps(embedded)
    )
    assert source == "server_config"
    assert resolved["training"] == server["training"]
    assert server["vllm"]["acceptance_sidecar_enabled"] is True  # no mutation
    assert not runtime._vllm_acceptance_sidecar_enabled(resolved)
    assert not runtime._vllm_acceptance_sidecar_enabled()
    # Repeat in the next process layer with private fields stripped again.
    resolved, _ = runtime._ensure_vllm_server_drafter_env(
        {"rollout": {"drafter": {"enable": True}}}, json.dumps(resolved)
    )
    assert not runtime._vllm_acceptance_sidecar_enabled(resolved)


def test_disabled_worker_does_no_file_io_and_keeps_in_memory_counts(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(runtime.SPECO_DRAFTER_CONFIG_ENV, json.dumps(drafter()))
    worker = SimpleNamespace(
        log_stats=False,
        vllm_config=SimpleNamespace(
            additional_config={
                runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(tmp_path / "stale"),
            }
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled sidecar must not access files")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", forbidden)
        guard.setattr(runtime.os, "makedirs", forbidden)
        guard.setattr(runtime.os, "replace", forbidden)
        guard.setattr(runtime.os, "listdir", forbidden)
        runtime._record_vllm_worker_spec_decode_output(
            worker,
            SimpleNamespace(scheduled_spec_decode_tokens={"r": [1, 2, 3, 4]}),
            SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[[1, 2, 3, 4]]),
        )
        assert runtime.read_vllm_spec_decode_sidecar_totals(str(tmp_path)) == {
            "drafts": 0.0,
            "accepted_tokens": 0.0,
            "draft_tokens": 0.0,
        }
    assert worker._speco_spec_decode_total_drafts == 1
    assert worker._speco_spec_decode_total_accepted == 3
    assert worker._speco_spec_decode_total_draft_tokens == 4
    assert list(tmp_path.iterdir()) == []


def test_explicit_enable_and_legacy_default_keep_writing(monkeypatch, tmp_path):
    for cfg in ({}, drafter(enabled=True)):
        monkeypatch.setenv(runtime.SPECO_DRAFTER_CONFIG_ENV, json.dumps(cfg))
        worker = SimpleNamespace(
            vllm_config=SimpleNamespace(
                additional_config={
                    runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(tmp_path),
                }
            ),
            _speco_spec_decode_total_drafts=2,
            _speco_spec_decode_total_accepted=6,
            _speco_spec_decode_total_draft_tokens=8,
        )
        assert runtime._write_vllm_spec_decode_sidecar(worker)
        assert runtime.read_vllm_spec_decode_sidecar_totals(str(tmp_path)) == {
            "drafts": 2.0,
            "accepted_tokens": 6.0,
            "draft_tokens": 8.0,
        }


def test_v1_disabled_sidecar_never_reads_stale_files(monkeypatch):
    path = (
        Path(__file__).parents[2] / "verl_speco" / "trainer" / "v1" / "speco_mixin.py"
    )
    method = next(
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_speco_v1_spec_decode_sidecar_metrics"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - actual repository method under test

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled V1 fallback must not read counters")

    monkeypatch.setattr(runtime, "read_vllm_spec_decode_sidecar_totals", forbidden)
    owner = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout={"drafter": drafter()}),
        )
    )
    assert namespace[method.name](owner) == {}
