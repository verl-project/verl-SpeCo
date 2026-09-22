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
import ctypes
import gc
import json
import logging
import os
import re
import sys
import time
from functools import lru_cache
from glob import glob
from typing import Any, Optional, Union


_DRAFT_STEP_PATTERN = re.compile(r"^draft_step_(\d+)$")
_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.index.json",
)
_OPTIMIZER_DCP_METADATA = ".metadata"

logger = logging.getLogger(__name__)

_JEMALLOC_ARENAS_ALL = 4096
_JEMALLOC_RECLAIM_MODE_ENV = "SPECO_JEMALLOC_RECLAIM_MODE"


class DrafterCheckpointMetadataError(ValueError):
    """Raised when a managed drafter checkpoint has invalid metadata."""


def _managed_checkpoint_step(path: str) -> Optional[int]:
    match = _DRAFT_STEP_PATTERN.match(os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match is not None else None


def get_drafter_checkpoint_metadata(
    model_path: Optional[Union[str, os.PathLike]],
) -> dict[str, Any]:
    if not model_path:
        return {}

    metadata_path = os.path.join(os.fspath(model_path), "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint metadata {metadata_path}: expected a JSON object"
        )
    return metadata


def get_drafter_trainer_state(
    model_path: Optional[Union[str, os.PathLike]],
) -> dict[str, Any]:
    metadata = get_drafter_checkpoint_metadata(model_path)
    trainer_state = metadata.get("trainer_state")
    if trainer_state is None:
        return {}
    if not isinstance(trainer_state, dict):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter trainer state metadata: expected a JSON object"
        )
    return dict(trainer_state)


def get_drafter_checkpoint_step(
    model_path: Optional[Union[str, os.PathLike]],
) -> Optional[int]:
    """Return the drafter training step recorded in a saved checkpoint directory."""
    if not model_path:
        return None

    path = os.fspath(model_path)
    metadata = get_drafter_checkpoint_metadata(path)
    if not metadata:
        return None
    try:
        step = metadata.get("step")
        if step is not None:
            return int(step)
    except (TypeError, ValueError) as exc:
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint step in {os.path.join(path, 'metadata.json')}: {step!r}"
        ) from exc
    return None


def get_drafter_optimizer_manifest(
    model_path: Optional[Union[str, os.PathLike]],
) -> dict[str, Any]:
    metadata = get_drafter_checkpoint_metadata(model_path)
    manifest = metadata.get("optimizer")
    if manifest is None:
        return {}
    if not isinstance(manifest, dict):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter optimizer manifest: expected a JSON object"
        )

    checkpoint_format = manifest.get("format")
    relative_path = manifest.get("path")
    if checkpoint_format != "torch_distributed_checkpoint" or not isinstance(
        relative_path, str
    ):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter optimizer manifest: expected torch_distributed_checkpoint format and path"
        )
    normalized_path = os.path.normpath(relative_path)
    if (
        os.path.isabs(normalized_path)
        or normalized_path == ".."
        or normalized_path.startswith(f"..{os.sep}")
    ):
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter optimizer checkpoint path: {relative_path!r}"
        )
    return dict(manifest)


def get_drafter_optimizer_checkpoint_path(
    model_path: Optional[Union[str, os.PathLike]],
) -> Optional[str]:
    if not model_path:
        return None
    manifest = get_drafter_optimizer_manifest(model_path)
    if not manifest:
        return None
    optimizer_path = os.path.join(os.fspath(model_path), manifest["path"])
    if not os.path.isfile(os.path.join(optimizer_path, _OPTIMIZER_DCP_METADATA)):
        raise DrafterCheckpointMetadataError(
            f"Drafter optimizer checkpoint is incomplete: missing {_OPTIMIZER_DCP_METADATA} in {optimizer_path}"
        )
    trainer_state_file = manifest.get("trainer_state_file")
    if (
        not isinstance(trainer_state_file, str)
        or os.path.basename(trainer_state_file) != trainer_state_file
        or not os.path.isfile(os.path.join(optimizer_path, trainer_state_file))
    ):
        raise DrafterCheckpointMetadataError(
            f"Drafter optimizer checkpoint is incomplete: missing trainer state in {optimizer_path}"
        )
    return optimizer_path


def is_pretrained_drafter_checkpoint(
    model_path: Optional[Union[str, os.PathLike]],
) -> bool:
    if not model_path:
        return False
    path = os.fspath(model_path)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "config.json")):
        return False
    if not any(glob(os.path.join(path, pattern)) for pattern in _WEIGHT_PATTERNS):
        return False

    metadata_path = os.path.join(path, "metadata.json")
    managed_step = _managed_checkpoint_step(path)
    if managed_step is not None or os.path.exists(metadata_path):
        metadata = get_drafter_checkpoint_metadata(path)
        if not metadata or metadata.get("complete", True) is not True:
            return False
        recorded_step = get_drafter_checkpoint_step(path)
        if managed_step is not None and recorded_step != managed_step:
            return False
        if metadata.get("optimizer") is not None:
            get_drafter_optimizer_checkpoint_path(path)
    return True


def _read_kib(path: str, keys: set[str]) -> dict[str, int]:
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                name, separator, raw_value = line.partition(":")
                if not separator or name not in keys:
                    continue
                fields = raw_value.strip().split()
                if fields:
                    values[name] = int(fields[0])
    except (OSError, ValueError):
        return {}
    return values


def collect_checkpoint_memory_snapshot() -> dict[str, Optional[int]]:
    """Collect Linux process/system memory counters in KiB."""

    process = _read_kib("/proc/self/status", {"VmRSS", "RssAnon", "RssShmem"})
    system = _read_kib(
        "/proc/meminfo",
        {
            "MemTotal",
            "MemAvailable",
            "Cached",
            "Shmem",
            "Dirty",
            "Writeback",
        },
    )
    dev_shm_used_kib = None
    statvfs = getattr(os, "statvfs", None)
    try:
        if statvfs is not None:
            stat = statvfs("/dev/shm")
            dev_shm_used_kib = ((stat.f_blocks - stat.f_bfree) * stat.f_frsize) // 1024
    except OSError:
        pass
    return {
        "rss_gib": process.get("VmRSS"),
        "anon_gib": process.get("RssAnon"),
        "rss_shmem_gib": process.get("RssShmem"),
        "node_unavailable_gib": (
            system["MemTotal"] - system["MemAvailable"]
            if "MemTotal" in system and "MemAvailable" in system
            else None
        ),
        "available_gib": system.get("MemAvailable"),
        "cached_gib": system.get("Cached"),
        "node_shmem_gib": system.get("Shmem"),
        "dev_shm_used_gib": dev_shm_used_kib,
        "dirty_gib": system.get("Dirty"),
        "writeback_gib": system.get("Writeback"),
    }


def format_checkpoint_memory_snapshot(
    counters: Optional[dict[str, Optional[int]]] = None,
) -> str:
    """Return concise Linux process/system memory counters without extra dependencies."""

    counters = (
        counters if counters is not None else collect_checkpoint_memory_snapshot()
    )
    return " ".join(
        f"{name}={value / (1024**2):.2f}" if value is not None else f"{name}=n/a"
        for name, value in counters.items()
    )


@lru_cache(maxsize=1)
def _jemalloc_is_active() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if "jemalloc" in os.getenv("LD_PRELOAD", "").lower():
        return True
    try:
        with open("/proc/self/maps", encoding="utf-8") as stream:
            return any("jemalloc" in line.lower() for line in stream)
    except OSError:
        return False


@lru_cache(maxsize=1)
def _jemalloc_mallctl_function() -> Any:
    try:
        runtime = ctypes.CDLL(None)
        mallctl = getattr(runtime, "mallctl", None) or getattr(
            runtime, "je_mallctl", None
        )
        if mallctl is None:
            return None
        mallctl.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        mallctl.restype = ctypes.c_int
        return mallctl
    except Exception:  # noqa: BLE001
        return None


def _jemalloc_mallctl(name: str) -> bool:
    try:
        mallctl = _jemalloc_mallctl_function()
        if mallctl is None:
            return False
        return mallctl(name.encode("ascii"), None, None, None, 0) == 0
    except Exception:  # noqa: BLE001
        return False


def _jemalloc_reclaim_mode() -> str:
    mode = os.getenv(_JEMALLOC_RECLAIM_MODE_ENV, "decay").strip().lower()
    return mode if mode in {"decay", "purge"} else "decay"


def _reclaim_jemalloc_heap() -> bool:
    _jemalloc_mallctl("thread.tcache.flush")
    return _jemalloc_mallctl(f"arena.{_JEMALLOC_ARENAS_ALL}.{_jemalloc_reclaim_mode()}")


def _host_allocator_name() -> str:
    if not sys.platform.startswith("linux"):
        return "unsupported"
    return "jemalloc" if _jemalloc_is_active() else "glibc"


def _trim_process_heap() -> bool:
    """Return unused allocator pages to the operating system when available."""

    if not sys.platform.startswith("linux"):
        return False
    if _jemalloc_is_active():
        return _reclaim_jemalloc_heap()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except Exception:  # noqa: BLE001
        return False


def trim_process_host_memory() -> dict[str, Any]:
    """Return unused jemalloc/glibc pages without forcing a Python GC cycle."""

    started = time.perf_counter()
    allocator = _host_allocator_name()
    heap_trimmed = _trim_process_heap()
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": heap_trimmed,
        "allocator": allocator,
        "reclaim_action": (
            _jemalloc_reclaim_mode() if allocator == "jemalloc" else "malloc_trim"
        ),
    }


def _flush_and_drop_checkpoint_file_cache(checkpoint_path: str) -> tuple[int, int]:
    """Flush completed checkpoint files and advise Linux to evict their cache."""

    if not sys.platform.startswith("linux") or not hasattr(os, "posix_fadvise"):
        return 0, 0
    try:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return 0, 0
    except OSError:
        return 0, 1

    advised = 0
    failed = 0
    paths = []
    if os.path.isfile(checkpoint_path):
        paths.append(checkpoint_path)
    else:
        for root, _, filenames in os.walk(checkpoint_path):
            paths.extend(os.path.join(root, filename) for filename in filenames)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    advice = getattr(os, "POSIX_FADV_DONTNEED", 4)
    for path in paths:
        try:
            fd = os.open(path, flags)
        except OSError:
            failed += 1
            continue
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, advice)
            advised += 1
        except OSError:
            failed += 1
        finally:
            try:
                os.close(fd)
            except OSError:
                failed += 1
    return advised, failed


def release_checkpoint_host_memory(
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    *,
    drop_file_cache: bool = False,
) -> dict[str, Any]:
    """Best-effort release of checkpoint staging and file-cache memory."""

    started = time.perf_counter()
    try:
        gc.collect()
    except Exception:  # noqa: BLE001
        pass
    allocator = _host_allocator_name()
    trimmed_before = _trim_process_heap()
    advised = 0
    failed = 0
    if drop_file_cache and checkpoint_path:
        try:
            advised, failed = _flush_and_drop_checkpoint_file_cache(
                os.fspath(checkpoint_path)
            )
        except Exception:  # noqa: BLE001
            failed = 1
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": trimmed_before,
        "allocator": allocator,
        "reclaim_action": (
            _jemalloc_reclaim_mode() if allocator == "jemalloc" else "malloc_trim"
        ),
        "files_advised": advised,
        "files_failed": failed,
    }


def resolve_drafter_checkpoint_path(
    model_path: Optional[Union[str, os.PathLike]],
    checkpoint_path: Optional[Union[str, os.PathLike]],
    global_step: Optional[int],
) -> Optional[str]:
    """Resolve a drafter model path to the checkpoint matching ``global_step`` when available."""
    original_model_path = os.fspath(model_path) if model_path is not None else None
    if global_step is None:
        return original_model_path

    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return original_model_path
    if step <= 0:
        return original_model_path

    if (
        original_model_path is not None
        and get_drafter_checkpoint_step(original_model_path) == step
        and is_pretrained_drafter_checkpoint(original_model_path)
    ):
        return original_model_path

    candidates = []
    if checkpoint_path:
        root = os.fspath(checkpoint_path)
        if os.path.basename(os.path.normpath(root)) == f"draft_step_{step}":
            candidates.append(root)
        candidates.append(os.path.join(root, f"draft_step_{step}"))

    for candidate in candidates:
        if get_drafter_checkpoint_step(
            candidate
        ) == step and is_pretrained_drafter_checkpoint(candidate):
            return candidate
    return original_model_path


def log_drafter_checkpoint_step(
    logger: logging.Logger,
    model_path: Optional[Union[str, os.PathLike]],
    *,
    action: str = "Loading drafter weights",
) -> Optional[int]:
    step = get_drafter_checkpoint_step(model_path)
    step_text = str(step) if step is not None else "unknown"
    logger.info("%s from %s (drafter_step=%s)", action, model_path, step_text)
    return step


# --------------------------------------------------------------------------- #
# Freeze-transition branch checkpoint manifest
#   See freeze_transition_branch_checkpoint_design.md sec. 5.4/7. The manifest
#   lives inside ``global_step_S/`` and only records fork semantics (it never
#   copies model state). ``complete=true`` is committed atomically as the LAST
#   step, so a torn temp file can never be mistaken for a valid fork point.
# --------------------------------------------------------------------------- #
FREEZE_BRANCH_MANIFEST_NAME = "freeze_branch_manifest.json"
FREEZE_POLICY_SIDECAR_NAME = "speco_drafter_freeze_state.json"
FREEZE_BRANCH_FORMAT_VERSION = 1


class FreezeBranchManifestError(ValueError):
    """Raised when a freeze branch checkpoint manifest is missing or invalid."""


def freeze_branch_manifest_path(
    checkpoint_folder: Union[str, os.PathLike],
) -> str:
    return os.path.join(os.fspath(checkpoint_folder), FREEZE_BRANCH_MANIFEST_NAME)


def freeze_policy_sidecar_path(
    checkpoint_folder: Union[str, os.PathLike],
) -> str:
    """Versioned freeze-policy sidecar carried inside a ``global_step_S`` folder."""
    return os.path.join(os.fspath(checkpoint_folder), FREEZE_POLICY_SIDECAR_NAME)


def atomic_write_json(
    path: Union[str, os.PathLike],
    payload: dict[str, Any],
) -> str:
    """Write JSON to ``path`` atomically (tmp file + ``os.replace``)."""
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{target}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return target


def write_freeze_branch_manifest(
    checkpoint_folder: Union[str, os.PathLike],
    manifest: dict[str, Any],
) -> str:
    """Atomically commit a freeze-branch manifest.

    ``complete`` must already be ``True``: the manifest is the final write
    after model/drafter/sidecar state landed, and a non-complete manifest is
    never recognized as a fork point.
    """
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise FreezeBranchManifestError(
            "freeze branch manifest must be a JSON object with complete=true"
        )
    if int(manifest.get("format_version", -1)) != FREEZE_BRANCH_FORMAT_VERSION:
        raise FreezeBranchManifestError(
            f"unsupported freeze branch manifest format_version: "
            f"{manifest.get('format_version')!r}"
        )
    path = freeze_branch_manifest_path(checkpoint_folder)
    return atomic_write_json(path, manifest)


def read_freeze_branch_manifest(
    checkpoint_folder: Union[str, os.PathLike],
) -> dict[str, Any]:
    """Strictly read a committed freeze-branch manifest.

    Raises :class:`FreezeBranchManifestError` when the manifest is missing,
    unreadable, not an object, or ``complete`` is not ``True``.
    """
    path = freeze_branch_manifest_path(checkpoint_folder)
    if not os.path.isfile(path):
        raise FreezeBranchManifestError(
            f"freeze branch manifest not found: {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeBranchManifestError(
            f"invalid freeze branch manifest {path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise FreezeBranchManifestError(
            f"invalid freeze branch manifest {path}: expected a JSON object"
        )
    if manifest.get("complete") is not True:
        raise FreezeBranchManifestError(
            f"freeze branch manifest {path} is not complete"
        )
    if int(manifest.get("format_version", -1)) != FREEZE_BRANCH_FORMAT_VERSION:
        raise FreezeBranchManifestError(
            f"freeze branch manifest {path} has unsupported format_version "
            f"{manifest.get('format_version')!r}"
        )
    return manifest


def validate_freeze_branch_checkpoint(
    checkpoint_folder: Union[str, os.PathLike],
    *,
    drafter_root: Optional[Union[str, os.PathLike]] = None,
    step: Optional[int] = None,
) -> dict[str, Any]:
    """Fail-closed validation of a just-saved freeze branch checkpoint.

    Checks that the actor directory, dataloader state (``data.pt``), the
    versioned freeze-policy sidecar, a complete ``draft_step_S`` drafter
    checkpoint, and a committed ``complete=true`` branch manifest are all
    present together. Returns the parsed manifest; raises
    :class:`FreezeBranchManifestError` listing every problem found.
    """
    folder = os.fspath(checkpoint_folder)
    problems: list[str] = []

    if not os.path.isdir(folder):
        raise FreezeBranchManifestError(
            f"freeze branch checkpoint folder does not exist: {folder}"
        )

    try:
        manifest = read_freeze_branch_manifest(folder)
    except FreezeBranchManifestError as exc:
        problems.append(str(exc))
        manifest = None

    if not os.path.isdir(os.path.join(folder, "actor")):
        problems.append(f"missing actor checkpoint directory under {folder}")
    if not os.path.isfile(os.path.join(folder, "data.pt")):
        problems.append(f"missing dataloader state data.pt under {folder}")

    sidecar = freeze_policy_sidecar_path(folder)
    if not os.path.isfile(sidecar):
        problems.append(f"missing freeze policy sidecar {sidecar}")

    if step is None and manifest is not None:
        step = manifest.get("checkpoint_step")
    if step is not None:
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            problems.append(f"invalid checkpoint step {step!r}")
            step_int = None
        if step_int is not None and drafter_root:
            draft_dir = os.path.join(
                os.fspath(drafter_root), f"draft_step_{step_int}"
            )
            if not is_pretrained_drafter_checkpoint(draft_dir):
                problems.append(
                    f"missing or incomplete drafter checkpoint {draft_dir}"
                )
            elif get_drafter_checkpoint_step(draft_dir) != step_int:
                problems.append(
                    f"drafter checkpoint step mismatch in {draft_dir} "
                    f"(expected {step_int})"
                )
    elif not drafter_root:
        problems.append("drafter checkpoint root not provided for validation")

    if manifest is not None:
        for key in (
            "trigger_step",
            "checkpoint_step",
            "drafter_version",
            "freeze_reason",
            "freeze_mode_at_save",
            "compare_from_step",
        ):
            if key not in manifest:
                problems.append(f"manifest missing key {key!r}")
        if step is not None:
            trigger_step = manifest.get("trigger_step")
            checkpoint_step = manifest.get("checkpoint_step")
            if int(trigger_step) != int(step) or int(checkpoint_step) != int(step):
                problems.append(
                    "manifest step mismatch: "
                    f"trigger_step={trigger_step} checkpoint_step="
                    f"{checkpoint_step} expected {step}"
                )

    if problems:
        raise FreezeBranchManifestError(
            "freeze branch checkpoint validation failed: " + "; ".join(problems)
        )
    return manifest
