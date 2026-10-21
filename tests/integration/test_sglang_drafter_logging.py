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
"""Exercise loss forwarding with Ray imported before the SPECO entrypoint."""

import os
import re

import pytest

pytest.importorskip("ray")
from ray._private import ray_constants, ray_logging

from verl_speco.main import _configure_sglang_drafter_logging


def _config(name="sglang", training=True):
    return {
        "actor_rollout_ref": {
            "rollout": {
                "name": name,
                "drafter": {"enable": True, "enable_drafter_training": training},
            }
        }
    }


@pytest.fixture
def deduplicators(monkeypatch):
    # Record the original environment even when the variable was absent.
    monkeypatch.setenv("RAY_DEDUP_LOGS_ALLOW_REGEX", "")
    monkeypatch.delenv("RAY_DEDUP_LOGS_ALLOW_REGEX", raising=False)
    monkeypatch.setattr(ray_constants, "RAY_DEDUP_LOGS_ALLOW_REGEX", "old-rule")
    monkeypatch.setattr(ray_logging, "RAY_DEDUP_LOGS", True)
    result = []
    for name in ("stdout_deduplicator", "stderr_deduplicator"):
        dedup = ray_logging.LogDeduplicator(
            5, "(?i)existing-rule", None, _timesource=lambda: 0
        )
        monkeypatch.setattr(ray_logging, name, dedup)
        result.append(dedup)
    return result


def test_every_step_survives_preimported_ray(deduplicators):
    _configure_sglang_drafter_logging(_config())
    for dedup in deduplicators:
        forwarded = []
        for step in range(1, 11):
            for pid in (101, 102):
                line = f"INFO:[drafter loss] step={step} loss=0.6900"
                batches = dedup.deduplicate(
                    {"ip": "127.0.0.1", "pid": pid, "lines": [line]}
                )
                forwarded.extend(line for batch in batches for line in batch["lines"])
        assert len(forwarded) == 20
        assert all("repeated" not in line for line in forwarded)
        assert dedup.allow_re.search("EXISTING-RULE")
        # Other repeated logs must still be deduplicated.
        ordinary = []
        for pid in (101, 102):
            batches = dedup.deduplicate(
                {"ip": "127.0.0.1", "pid": pid, "lines": ["ordinary message"]}
            )
            ordinary.extend(line for batch in batches for line in batch["lines"])
        assert ordinary == ["ordinary message"]


def test_preserves_environment_rule_and_is_idempotent(monkeypatch, deduplicators):
    monkeypatch.setenv("RAY_DEDUP_LOGS_ALLOW_REGEX", "custom-rule")
    _configure_sglang_drafter_logging(_config())
    configured = os.environ["RAY_DEDUP_LOGS_ALLOW_REGEX"]
    compiled = [dedup.allow_re for dedup in deduplicators]
    _configure_sglang_drafter_logging(_config())
    assert os.environ["RAY_DEDUP_LOGS_ALLOW_REGEX"] == configured
    assert re.search(configured, "custom-rule")
    assert re.search(configured, "[drafter loss]")
    assert ray_constants.RAY_DEDUP_LOGS_ALLOW_REGEX == configured
    assert [dedup.allow_re for dedup in deduplicators] == compiled


@pytest.mark.parametrize("config", [_config("vllm"), _config(training=False), {}])
def test_other_runs_keep_existing_rules(config, deduplicators):
    _configure_sglang_drafter_logging(config)
    assert "RAY_DEDUP_LOGS_ALLOW_REGEX" not in os.environ
    assert ray_constants.RAY_DEDUP_LOGS_ALLOW_REGEX == "old-rule"
    assert all(not dedup.allow_re.search("[drafter loss]") for dedup in deduplicators)
