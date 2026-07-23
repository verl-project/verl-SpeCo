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
"""Compatibility checks for the import-only verl dependency."""

import json
import logging
import os
from dataclasses import dataclass
from importlib import metadata
from typing import Optional, Sequence

SUPPORTED_VERL_VERSION = "0.8.0"
SUPPORTED_VERL_COMMITS = frozenset({"7aed6b230776f963fa09509c10d9c3a767d1102c"})
ALLOW_UNSUPPORTED_ENV = "VERL_SPECO_ALLOW_UNSUPPORTED_VERL"
STRICT_COMPAT_ENV = "VERL_SPECO_STRICT_VERL"

logger = logging.getLogger(__file__)


@dataclass(frozen=True)
class VerlCompatibility:
    """Resolved metadata for the installed verl package."""

    version: Optional[str]
    commit_id: Optional[str]
    supported: bool
    reason: str


def _read_distribution_commit(distribution_name: str = "verl") -> Optional[str]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None

    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        return None

    try:
        direct_url = json.loads(raw_direct_url)
    except json.JSONDecodeError:
        return None

    vcs_info = direct_url.get("vcs_info") or {}
    commit_id = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
    return str(commit_id) if commit_id else None


def _read_imported_verl_version() -> Optional[str]:
    try:
        return metadata.version("verl")
    except metadata.PackageNotFoundError:
        pass

    try:
        import verl
    except ImportError:
        return None

    version = getattr(verl, "__version__", None)
    return str(version) if version else None


def resolve_verl_compatibility(
    allowed_versions: Sequence[str] = (SUPPORTED_VERL_VERSION,),
    allowed_commits: Sequence[str] = tuple(SUPPORTED_VERL_COMMITS),
) -> VerlCompatibility:
    """Return whether the currently importable verl matches the SPECO base."""

    version = _read_imported_verl_version()
    commit_id = _read_distribution_commit()

    if version in set(allowed_versions):
        return VerlCompatibility(
            version=version,
            commit_id=commit_id,
            supported=True,
            reason="matched version",
        )

    if commit_id in set(allowed_commits):
        return VerlCompatibility(
            version=version,
            commit_id=commit_id,
            supported=True,
            reason="matched commit",
        )

    if os.getenv(ALLOW_UNSUPPORTED_ENV, "").lower() in {"1", "true", "yes"}:
        return VerlCompatibility(
            version=version, commit_id=commit_id, supported=True, reason="env override"
        )

    return VerlCompatibility(
        version=version,
        commit_id=commit_id,
        supported=False,
        reason=(
            "SPECO requires import-only verl v0.8.0 "
            f"or commit {', '.join(sorted(SUPPORTED_VERL_COMMITS))}"
        ),
    )


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def check_compatible_verl(strict: Optional[bool] = None) -> VerlCompatibility:
    """Warn by default when the importable verl is outside the supported base."""

    result = resolve_verl_compatibility()
    if result.supported:
        return result

    message = (
        f"{result.reason}; found version={result.version!r}, commit_id={result.commit_id!r}. "
        f"Set {STRICT_COMPAT_ENV}=1 to fail closed."
    )
    if strict is None:
        strict = _env_flag_enabled(STRICT_COMPAT_ENV)
    if strict:
        raise RuntimeError(message)

    logger.warning(message)
    return result


def assert_compatible_verl() -> VerlCompatibility:
    """Backward-compatible alias for the warning-only compatibility check."""

    return check_compatible_verl()
