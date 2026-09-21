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

import os

from verl_speco.integration.drafter_config_env import (
    LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV,
    SPECO_DRAFTER_CONFIG_ENV,
    clear_drafter_config_env,
    get_drafter_config_env,
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
