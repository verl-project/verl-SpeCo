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

    assert required["VERL_BASE_BRANCH"] == base["branch"]
    assert required["VERL_BASE_VERSION"] == base["version"]
    assert required["VERL_BASE_BRANCH"] == compatibility["require_verl_base_branch"]
    assert required["VERL_BASE_VERSION"] == compatibility["require_verl_base_version"]
    assert required["VERL_SOURCE_MODIFICATIONS_ALLOWED"] == str(
        base["source_modifications_allowed"]
    ).lower()
    assert required["ENTRYPOINT"] == "python -m verl_speco.main"


def test_overlay_keeps_speco_changes_external_to_verl() -> None:
    required = _required_verl_values()

    assert required["VERL_SOURCE_MODIFICATIONS_ALLOWED"] == "false"
    assert required["REQUIRES"] == "import verl only"
