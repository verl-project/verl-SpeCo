import ctypes
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
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


class DrafterCheckpointMetadataError(ValueError):
    """Raised when a managed drafter checkpoint has invalid metadata."""


def _managed_checkpoint_step(path: str) -> Optional[int]:
    match = _DRAFT_STEP_PATTERN.match(os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match is not None else None


def get_drafter_checkpoint_metadata(model_path: Optional[Union[str, os.PathLike]]) -> dict[str, Any]:
    if not model_path:
        return {}

    metadata_path = os.path.join(os.fspath(model_path), "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DrafterCheckpointMetadataError(f"Invalid drafter checkpoint metadata {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DrafterCheckpointMetadataError(
            f"Invalid drafter checkpoint metadata {metadata_path}: expected a JSON object"
        )
    return metadata


def get_drafter_trainer_state(model_path: Optional[Union[str, os.PathLike]]) -> dict[str, Any]:
    metadata = get_drafter_checkpoint_metadata(model_path)
    trainer_state = metadata.get("trainer_state")
    if trainer_state is None:
        return {}
    if not isinstance(trainer_state, dict):
        raise DrafterCheckpointMetadataError("Invalid drafter trainer state metadata: expected a JSON object")
    return dict(trainer_state)


def get_drafter_checkpoint_step(model_path: Optional[Union[str, os.PathLike]]) -> Optional[int]:
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


def get_drafter_optimizer_manifest(model_path: Optional[Union[str, os.PathLike]]) -> dict[str, Any]:
    metadata = get_drafter_checkpoint_metadata(model_path)
    manifest = metadata.get("optimizer")
    if manifest is None:
        return {}
    if not isinstance(manifest, dict):
        raise DrafterCheckpointMetadataError("Invalid drafter optimizer manifest: expected a JSON object")

    checkpoint_format = manifest.get("format")
    relative_path = manifest.get("path")
    if checkpoint_format != "torch_distributed_checkpoint" or not isinstance(relative_path, str):
        raise DrafterCheckpointMetadataError(
            "Invalid drafter optimizer manifest: expected torch_distributed_checkpoint format and path"
        )
    normalized_path = os.path.normpath(relative_path)
    if os.path.isabs(normalized_path) or normalized_path == ".." or normalized_path.startswith(f"..{os.sep}"):
        raise DrafterCheckpointMetadataError(f"Invalid drafter optimizer checkpoint path: {relative_path!r}")
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


def is_pretrained_drafter_checkpoint(model_path: Optional[Union[str, os.PathLike]]) -> bool:
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


def prune_drafter_checkpoints(
    checkpoint_root: Optional[Union[str, os.PathLike]],
    max_to_keep: Optional[int],
) -> list[str]:
    """Remove old complete managed checkpoints while preserving incomplete ones."""
    if not checkpoint_root or max_to_keep is None:
        return []
    try:
        keep_count = int(max_to_keep)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_to_keep must be an integer or null, got {max_to_keep!r}") from exc
    if keep_count < 1:
        raise ValueError(f"max_to_keep must be at least 1, got {keep_count}")

    root = os.fspath(checkpoint_root)
    if not os.path.isdir(root):
        return []

    complete_checkpoints: list[tuple[int, str]] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                step = _managed_checkpoint_step(entry.path)
                if step is None:
                    continue
                try:
                    if is_pretrained_drafter_checkpoint(entry.path):
                        complete_checkpoints.append((step, entry.path))
                except DrafterCheckpointMetadataError as exc:
                    logger.warning("Skip pruning invalid drafter checkpoint %s: %s", entry.path, exc)
    except OSError as exc:
        logger.warning("Failed to inspect drafter checkpoint root %s for pruning: %s", root, exc)
        return []

    complete_checkpoints.sort(key=lambda item: item[0], reverse=True)
    removed = []
    for _, checkpoint_path in complete_checkpoints[keep_count:]:
        try:
            shutil.rmtree(checkpoint_path)
            removed.append(checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to prune old drafter checkpoint %s: %s", checkpoint_path, exc)
    return removed


def format_checkpoint_memory_snapshot() -> str:
    """Return concise Linux process/system memory counters without extra dependencies."""

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

    process = _read_kib("/proc/self/status", {"VmRSS"})
    system = _read_kib("/proc/meminfo", {"MemAvailable", "Cached", "Dirty", "Writeback"})
    counters = {
        "rss_gib": process.get("VmRSS"),
        "available_gib": system.get("MemAvailable"),
        "cached_gib": system.get("Cached"),
        "dirty_gib": system.get("Dirty"),
        "writeback_gib": system.get("Writeback"),
    }
    return " ".join(
        f"{name}={value / (1024**2):.2f}" if value is not None else f"{name}=n/a"
        for name, value in counters.items()
    )


def _trim_process_heap() -> bool:
    """Return free glibc heap arenas to the operating system when available."""

    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except Exception:  # noqa: BLE001
        return False


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
    trimmed_before = _trim_process_heap()
    advised = 0
    failed = 0
    if drop_file_cache and checkpoint_path:
        try:
            advised, failed = _flush_and_drop_checkpoint_file_cache(os.fspath(checkpoint_path))
        except Exception:  # noqa: BLE001
            failed = 1
    try:
        gc.collect()
    except Exception:  # noqa: BLE001
        pass
    trimmed_after = _trim_process_heap()
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": bool(trimmed_before or trimmed_after),
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
        if get_drafter_checkpoint_step(candidate) == step and is_pretrained_drafter_checkpoint(candidate):
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
