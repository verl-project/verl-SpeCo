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

"""Canonical environment transport for SPECO drafter configuration."""

from __future__ import annotations

import os

SPECO_DRAFTER_CONFIG_ENV = "VERL_SPECO_DRAFTER_CONFIG"
LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV = "VERL_SPECO_SGLANG_DRAFTER_CONFIG"


def get_drafter_config_env(default: str = "") -> str:
    """Return the canonical value, falling back to the legacy environment name."""

    return (
        os.getenv(SPECO_DRAFTER_CONFIG_ENV)
        or os.getenv(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV)
        or default
    )


def set_drafter_config_env(value: str) -> None:
    """Publish a drafter config under the canonical name only."""

    os.environ[SPECO_DRAFTER_CONFIG_ENV] = value
    os.environ.pop(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, None)


def clear_drafter_config_env() -> None:
    """Clear both the canonical and legacy names."""

    os.environ.pop(SPECO_DRAFTER_CONFIG_ENV, None)
    os.environ.pop(LEGACY_SPECO_SGLANG_DRAFTER_CONFIG_ENV, None)
