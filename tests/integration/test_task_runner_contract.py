# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contracts for SPECO task-runner runtime payload transport."""

from __future__ import annotations

import json
import os

import pytest


def test_worker_drafter_payload_keeps_generated_acceptance_sidecar(monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("verl")
    from omegaconf import OmegaConf

    from verl_speco.integration.drafter_config_env import SPECO_DRAFTER_CONFIG_ENV
    from verl_speco.integration.task_runner import _serialize_drafter_config

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {"enable": True, "model_path": "/models/drafter"}
                }
            }
        }
    )
    monkeypatch.setenv(
        SPECO_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                "enable": True,
                "_speco_acceptance_stats_dir": "/tmp/run/.spec_decode_stats/run-x",
            }
        ),
    )

    payload = json.loads(_serialize_drafter_config(config))

    assert payload["model_path"] == "/models/drafter"
    assert payload["_speco_acceptance_stats_dir"] == "/tmp/run/.spec_decode_stats/run-x"


def test_v1_worker_payload_allocates_and_persists_acceptance_sidecar(
    monkeypatch, tmp_path
):
    """V1 serializes workers before its trainer configures the vLLM runtime."""
    pytest.importorskip("ray")
    pytest.importorskip("verl")
    from omegaconf import OmegaConf

    from verl_speco.integration.drafter_config_env import SPECO_DRAFTER_CONFIG_ENV
    from verl_speco.integration.task_runner import _serialize_drafter_config

    config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path)},
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {"enable": True, "model_path": "/models/drafter"}
                }
            },
        }
    )
    monkeypatch.setenv(SPECO_DRAFTER_CONFIG_ENV, json.dumps({"enable": True}))

    payload = json.loads(_serialize_drafter_config(config))
    persisted = json.loads(os.environ[SPECO_DRAFTER_CONFIG_ENV])

    sidecar_dir = payload["_speco_acceptance_stats_dir"]
    assert sidecar_dir == persisted["_speco_acceptance_stats_dir"]
    assert sidecar_dir.startswith(str(tmp_path / ".spec_decode_stats"))
    assert "/run-" in sidecar_dir.replace("\\", "/")
