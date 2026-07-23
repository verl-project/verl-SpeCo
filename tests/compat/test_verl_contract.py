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

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_VERL = ROOT / "REQUIRED_VERL.txt"
OVERLAY_CONFIG = ROOT / "verl_speco" / "config" / "speco_base.yaml"


def _required_verl_values() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in REQUIRED_VERL.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


def test_required_verl_contract_matches_overlay() -> None:
    required = _required_verl_values()
    overlay = yaml.safe_load(OVERLAY_CONFIG.read_text(encoding="utf-8"))
    base = overlay["speco"]["verl_base"]
    compatibility = overlay["speco"]["compatibility"]

    assert required["VERL_BASE_TAG"] == f"v{base['version']}"
    assert required["VERL_BASE_COMMIT"] == base["commit"]
    assert required["VERL_BASE_TAG"] == compatibility["require_verl_base_tag"]
    assert required["VERL_BASE_COMMIT"] == compatibility["require_verl_base_commit"]
    assert (
        required["VERL_SOURCE_MODIFICATIONS_ALLOWED"]
        == str(base["source_modifications_allowed"]).lower()
    )
    assert required["ENTRYPOINT"] == "python -m verl_speco.main"


def test_overlay_keeps_speco_changes_external_to_verl() -> None:
    required = _required_verl_values()

    assert required["VERL_SOURCE_MODIFICATIONS_ALLOWED"] == "false"
    assert required["REQUIRES"] == "import verl only"
