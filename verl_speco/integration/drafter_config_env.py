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

import json
import os
import uuid

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


def serialize_worker_drafter_config(payload: dict, run_dir=None) -> str:
    """Retain only the acceptance locator from runtime state in a worker payload.

    V1 can serialize workers before runtime setup. Allocate and persist the
    locator here when a run directory is available, so later setup reuses it.
    User configuration remains authoritative for all other fields.
    """
    payload = dict(payload)
    try:
        runtime_payload = json.loads(get_drafter_config_env())
    except (TypeError, ValueError):
        runtime_payload = {}
    if not isinstance(runtime_payload, dict):
        runtime_payload = {}
    sidecar_dir = runtime_payload.get("_speco_acceptance_stats_dir")
    if sidecar_dir and run_dir:
        # A driver can launch another run without exiting. Reuse a locator
        # only within its run directory, never counters from another launch.
        sidecar_root = os.path.abspath(
            os.path.join(os.fspath(run_dir), ".spec_decode_stats")
        )
        try:
            if (
                os.path.commonpath([sidecar_root, os.path.abspath(sidecar_dir)])
                != sidecar_root
            ):
                sidecar_dir = None
        except (TypeError, ValueError):
            sidecar_dir = None
    if not sidecar_dir and payload.get("enable") and run_dir:
        sidecar_dir = os.path.abspath(
            os.path.join(
                os.fspath(run_dir), ".spec_decode_stats", f"run-{uuid.uuid4().hex}"
            )
        )
        runtime_payload = {**runtime_payload, **payload}
        runtime_payload["_speco_acceptance_stats_dir"] = sidecar_dir
        set_drafter_config_env(json.dumps(runtime_payload, sort_keys=True))
    if sidecar_dir:
        payload["_speco_acceptance_stats_dir"] = sidecar_dir
    return json.dumps(payload, sort_keys=True)
