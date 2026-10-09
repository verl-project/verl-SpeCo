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

from __future__ import annotations

import json
import os

from verl_speco.integration.drafter_config_env import (
    LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV,
    SPECO_DRAFTER_CONFIG_ENV,
    clear_drafter_config_env,
    get_drafter_config_env,
    serialize_worker_drafter_config,
    set_drafter_config_env,
)


def test_drafter_config_env_reads_the_legacy_name(monkeypatch) -> None:
    monkeypatch.delenv(SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.setenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, "legacy-config")

    assert get_drafter_config_env() == "legacy-config"


def test_drafter_config_env_writes_only_the_canonical_name(monkeypatch) -> None:
    monkeypatch.delenv(SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.setenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, "legacy-config")

    set_drafter_config_env("canonical-config")

    assert get_drafter_config_env() == "canonical-config"
    assert os.environ[SPECO_DRAFTER_CONFIG_ENV] == "canonical-config"
    assert LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV not in os.environ


def test_clear_drafter_config_env_removes_both_names(monkeypatch) -> None:
    monkeypatch.setenv(SPECO_DRAFTER_CONFIG_ENV, "canonical-config")
    monkeypatch.setenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, "legacy-config")

    clear_drafter_config_env()

    assert get_drafter_config_env() == ""


def test_worker_payload_retains_locator_without_overriding_user_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        SPECO_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                "enable": True,
                "model_path": "stale",
                "_speco_acceptance_stats_dir": "/tmp/stats",
            }
        ),
    )
    original = {"enable": False, "model_path": "current"}

    payload = json.loads(serialize_worker_drafter_config(original))

    assert payload == {**original, "_speco_acceptance_stats_dir": "/tmp/stats"}
    assert original == {"enable": False, "model_path": "current"}


def test_worker_payload_allocates_and_reuses_run_locator(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.delenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, raising=False)
    config = {"enable": True, "model_path": "current"}

    first = json.loads(serialize_worker_drafter_config(config, run_dir=tmp_path))
    second = json.loads(serialize_worker_drafter_config(config, run_dir=tmp_path))

    assert first == second == json.loads(get_drafter_config_env())
    assert os.path.dirname(first["_speco_acceptance_stats_dir"]) == str(
        tmp_path / ".spec_decode_stats"
    )


def test_worker_payload_isolates_new_run_and_replaces_stale_config(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        SPECO_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                "enable": False,
                "model_path": "stale",
                "_speco_acceptance_stats_dir": str(
                    tmp_path / "old" / ".spec_decode_stats" / "run-old"
                ),
            }
        ),
    )
    config = {"enable": True, "model_path": "current"}

    payload = json.loads(
        serialize_worker_drafter_config(config, run_dir=tmp_path / "new")
    )

    assert os.path.dirname(payload["_speco_acceptance_stats_dir"]) == str(
        tmp_path / "new" / ".spec_decode_stats"
    )
    assert json.loads(get_drafter_config_env()) == payload


def test_disabled_worker_ignores_invalid_runtime_payload_without_allocating(
    monkeypatch, tmp_path
) -> None:
    for runtime_payload in ("invalid json", "[]", "null"):
        monkeypatch.setenv(SPECO_DRAFTER_CONFIG_ENV, runtime_payload)
        config = {"enable": False}

        assert (
            json.loads(serialize_worker_drafter_config(config, run_dir=tmp_path))
            == config
        )
        assert get_drafter_config_env() == runtime_payload
