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
"""Compatibility checks for the import-only verl dependency.

Release 0.8 and 0.9 use legacy runners. The V1-only layout in 0.10 and later has its
own API contract and does not export the legacy PPO trainer.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata

from packaging.version import InvalidVersion, Version

SUPPORTED_VERL_VERSION = "0.8.0"
SUPPORTED_VERL_BRANCH = "release/v0.8.0"
V1_MIN_VERSION = Version("0.10")
V1_VERSION_RANGE = f">={V1_MIN_VERSION}"
SUPPORTED_VERL_VERSIONS = ("0.8.0", "0.9.0", V1_VERSION_RANGE)
SUPPORTED_VERL_BRANCHES = ("release/v0.8.0", "release/v0.9.0")
SUPPORTED_VERL_RELEASES = {"0.8.0": "release/v0.8.0", "0.9.0": "release/v0.9.0"}
ALLOW_UNSUPPORTED_ENV = "VERL_SPECO_ALLOW_UNSUPPORTED_VERL"
STRICT_COMPAT_ENV = "VERL_SPECO_STRICT_VERL"

logger = logging.getLogger(__name__)

# These are the APIs imported by the core SPECO runner and trainer. Optional
# vLLM/SGLang APIs are checked when their corresponding rollout is enabled.
COMMON_REQUIRED_VERL_API: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("verl.trainer.ppo.utils", ("Role", "need_critic", "need_reference_policy")),
    (
        "verl.utils.device",
        ("auto_set_device", "get_device_id", "get_device_name", "get_torch_device"),
    ),
    (
        "verl.utils.fsdp_utils",
        (
            "get_fsdp_full_state_dict",
            "get_fsdp_wrap_policy",
            "apply_fsdp2",
            "fsdp2_load_full_state_dict",
            "load_fsdp_model_to_gpu",
            "load_fsdp_optimizer",
            "offload_fsdp_model_to_cpu",
            "offload_fsdp_optimizer",
        ),
    ),
    ("verl.utils.dataset.rl_dataset", ("collate_fn",)),
    (
        "verl.utils.ulysses",
        ("get_ulysses_sequence_parallel_group", "set_ulysses_sequence_parallel_group"),
    ),
    (
        "verl.utils.tensordict_utils",
        ("assign_non_tensor", "assign_non_tensor_data", "get", "get_non_tensor_data"),
    ),
    ("verl.workers.engine_workers", ("ActorRolloutRefWorker", "TrainingWorker")),
    ("verl.workers.rollout.replica", ("RolloutReplica", "TokenOutput")),
    ("verl.single_controller.base", ("Worker",)),
    ("verl.single_controller.base.decorator", ("Dispatch", "register")),
    ("verl.single_controller.ray", ("RayClassWithInitArgs",)),
    ("verl.utils.ray_utils", ("auto_await", "parallel_put")),
    (
        "verl.utils.distributed",
        ("initialize_global_process_group_ray", "set_numa_affinity"),
    ),
    ("verl.workers.utils.padding", ("left_right_2_no_padding", "no_padding_2_padding")),
)

REQUIRED_VERL_API_BY_RELEASE: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "0.8.0": COMMON_REQUIRED_VERL_API
    + (
        ("verl.trainer.ppo.ray_trainer", ("RayPPOTrainer",)),
        (
            "verl.trainer.main_ppo",
            (
                "TaskRunner",
                "create_rl_dataset",
                "create_rl_sampler",
                "run_ppo",
                "migrate_legacy_reward_impl",
            ),
        ),
        ("verl.utils.config", ("validate_config",)),
    ),
    "0.9.0": COMMON_REQUIRED_VERL_API
    + (
        ("verl.trainer.ppo.ray_trainer", ("RayPPOTrainer",)),
        ("verl.trainer.main_ppo", ("run_ppo",)),
        ("verl.trainer.main_ppo_v0", ("BaseTaskRunner",)),
        (
            "verl.trainer.ppo.utils",
            ("create_rl_dataset", "create_rl_sampler"),
        ),
        ("verl.utils.config", ("omega_conf_to_dataclass", "validate_config")),
        ("verl.workers.config.model", ("HFModelConfig",)),
    ),
    V1_VERSION_RANGE: COMMON_REQUIRED_VERL_API
    + (
        ("verl.trainer.main_ppo", ("TaskRunnerV1", "run_ppo")),
        (
            "verl.trainer.ppo.v1",
            ("PPOTrainerSync", "PPOTrainerColocateAsync", "PPOTrainerSeparateAsync", "AgentLoopManagerTQ"),
        ),
        ("verl.utils.config", ("omega_conf_to_dataclass", "validate_config")),
    ),
}

# Backward-compatible public name.  It continues to describe the default 0.8
# contract; callers that need release-specific checks should use the mapping.
REQUIRED_VERL_API = REQUIRED_VERL_API_BY_RELEASE[SUPPORTED_VERL_VERSION]


@dataclass(frozen=True)
class VerlCompatibility:
    """Resolved metadata for the installed verl package."""

    version: str | None
    commit_id: str | None
    requested_revision: str | None
    supported: bool
    reason: str
    missing_api: tuple[str, ...] = ()


def _read_distribution_vcs_info(
    distribution_name: str = "verl",
) -> tuple[str | None, str | None]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None, None

    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        return None, None

    try:
        direct_url = json.loads(raw_direct_url)
    except json.JSONDecodeError:
        return None, None

    vcs_info = direct_url.get("vcs_info") or {}
    commit_id = vcs_info.get("commit_id")
    requested_revision = vcs_info.get("requested_revision")
    return (
        str(commit_id) if commit_id else None,
        str(requested_revision) if requested_revision else None,
    )


def _read_distribution_commit(distribution_name: str = "verl") -> str | None:
    """Return the installed commit for diagnostics, never for acceptance."""

    return _read_distribution_vcs_info(distribution_name)[0]


def _read_imported_verl_version() -> str | None:
    """Read the version from the same ``verl`` module Python imported.

    ``PYTHONPATH`` deployments can import verl from a source checkout while an
    older wheel's ``*.dist-info`` remains in site-packages. Prefer the imported
    module so version and API checks describe the dependency SPECO will run.
    """

    try:
        import verl
    except ImportError:
        verl = None

    if verl is not None:
        version = getattr(verl, "__version__", None)
        if version:
            return str(version)

    try:
        return metadata.version("verl")
    except metadata.PackageNotFoundError:
        return None


def _missing_required_api(
    release_version: str = SUPPORTED_VERL_VERSION,
) -> tuple[str, ...]:
    missing: list[str] = []
    required_api = REQUIRED_VERL_API_BY_RELEASE.get(release_version)
    if required_api is None:
        return (f"unsupported verl API contract {release_version}",)
    for module_name, symbols in required_api:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(
                f"{module_name} (import failed: {type(exc).__name__}: {exc})"
            )
            continue
        for symbol in symbols:
            if not hasattr(module, symbol):
                missing.append(f"{module_name}.{symbol}")
    return tuple(missing)


def _version_matches_release(version: str | None, allowed_version: str) -> bool:
    """Match the release line, including its development and prerelease builds."""

    if version is None:
        return False
    try:
        parsed = Version(version)
        if allowed_version == V1_VERSION_RANGE:
            return Version(parsed.base_version) >= V1_MIN_VERSION
        return parsed.base_version == Version(allowed_version).base_version
    except InvalidVersion:
        return False


def has_only_v1_runner(main_ppo) -> bool:
    """0.9 also exports TaskRunnerV1, but still has a separate V0 module."""
    if not hasattr(main_ppo, "TaskRunnerV1") or hasattr(main_ppo, "TaskRunner"):
        return False
    v0_module = "verl.trainer.main_ppo_v0"
    if v0_module in sys.modules:
        return sys.modules[v0_module] is None
    return importlib.util.find_spec(v0_module) is None


def resolve_verl_compatibility(
    allowed_versions: Sequence[str] = SUPPORTED_VERL_VERSIONS,
) -> VerlCompatibility:
    """Return whether the importable verl matches the release API contract."""

    version = _read_imported_verl_version()
    commit_id, requested_revision = _read_distribution_vcs_info()
    allowed_versions = tuple(allowed_versions)
    matched_version = next(
        (
            allowed_version
            for allowed_version in allowed_versions
            if allowed_version in REQUIRED_VERL_API_BY_RELEASE
            and _version_matches_release(version, allowed_version)
        ),
        None,
    )
    matched_branch_version = next(
        (
            allowed_version
            for allowed_version in allowed_versions
            if allowed_version in SUPPORTED_VERL_RELEASES
            and SUPPORTED_VERL_RELEASES[allowed_version] == requested_revision
        ),
        None,
    )
    # Editable installs can retain old VCS metadata after the checkout changes.
    release_version = matched_branch_version or matched_version
    if matched_version == V1_VERSION_RANGE:
        release_version = matched_version
    missing_api = (
        _missing_required_api(release_version) if release_version is not None else ()
    )

    if release_version is not None and not missing_api:
        if release_version == V1_VERSION_RANGE:
            return VerlCompatibility(
                version, commit_id, requested_revision, True, "matched verl >=0.10 V1 API contract"
            )
        release_branch = SUPPORTED_VERL_RELEASES[release_version]
        reason = f"matched {release_branch} version/API contract"
        if matched_branch_version == release_version:
            reason = f"matched {release_branch} branch/API contract"
        return VerlCompatibility(version, commit_id, requested_revision, True, reason)

    if os.getenv(ALLOW_UNSUPPORTED_ENV, "").lower() in {"1", "true", "yes"}:
        return VerlCompatibility(
            version,
            commit_id,
            requested_revision,
            True,
            "env override",
            missing_api,
        )

    reasons = []
    if release_version is None:
        versions = ", ".join(allowed_versions)
        branches = ", ".join(
            SUPPORTED_VERL_RELEASES[item]
            for item in allowed_versions
            if item in SUPPORTED_VERL_RELEASES
        )
        reasons.append(f"expected version in ({versions}) or branch in ({branches})")
    if missing_api:
        reasons.append("missing/incompatible API: " + ", ".join(missing_api[:8]))
    return VerlCompatibility(
        version,
        commit_id,
        requested_revision,
        False,
        "; ".join(reasons) or "unknown compatibility failure",
        missing_api,
    )


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def check_compatible_verl(strict: bool | None = None) -> VerlCompatibility:
    """Warn by default when verl is outside the supported API contracts."""

    result = resolve_verl_compatibility()
    if result.supported:
        return result

    message = (
        "SPECO requires verl release/v0.8.0, release/v0.9.0, or verl >=0.10 with the V1 API; "
        f"found version={result.version!r}, requested_revision={result.requested_revision!r}, "
        f"commit_id={result.commit_id!r}: {result.reason}. "
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
