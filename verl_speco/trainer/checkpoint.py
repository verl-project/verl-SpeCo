import ctypes
import gc
import json
import logging
import os
import re
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

_PROCESS_MEMORY_GROUPS = (
    ("agent_loop", ("agentloopworker", "agent_loop_worker")),
    ("worker_dict", ("workerdict",)),
    ("vllm_http", ("specovllmhttpserver", "vllmhttpserver", "vllm_server")),
    ("engine_core", ("enginecore", "engine_core")),
    ("worker_tp", ("worker_tp",)),
    ("speco_worker", ("specoworker",)),
    ("task_runner", ("specotaskrunner",)),
    ("ray_system", ("raylet", "plasma_store", "gcs_server")),
)


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


def format_checkpoint_memory_snapshot() -> str:
    """Return concise Linux process/system memory counters without extra dependencies."""

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
            "AnonPages",
            "SReclaimable",
            "SUnreclaim",
            "PageTables",
            "KernelStack",
            "Mlocked",
            "Unevictable",
        },
    )
    dev_shm_used_kib = None
    try:
        stat = os.statvfs("/dev/shm")
        dev_shm_used_kib = ((stat.f_blocks - stat.f_bfree) * stat.f_frsize) // 1024
    except (AttributeError, OSError):
        pass
    counters = {
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
        "anon_pages_gib": system.get("AnonPages"),
        "sreclaimable_gib": system.get("SReclaimable"),
        "sunreclaim_gib": system.get("SUnreclaim"),
        "pagetables_gib": system.get("PageTables"),
        "kernel_stack_gib": system.get("KernelStack"),
        "mlocked_gib": system.get("Mlocked"),
        "unevictable_gib": system.get("Unevictable"),
    }
    return " ".join(
        f"{name}={value / (1024**2):.2f}" if value is not None else f"{name}=n/a"
        for name, value in counters.items()
    )


def _classify_process_memory_group(process_text: str) -> str | None:
    normalized = process_text.lower().replace("-", "_")
    for group, patterns in _PROCESS_MEMORY_GROUPS:
        if any(pattern in normalized for pattern in patterns):
            return group
    if "ray::" in normalized or "ray_worker" in normalized:
        return "ray_other"
    return None


def _process_memory_title(comm: str, cmdline: str) -> str:
    ray_title = re.search(r"ray::[^\s]+", f"{comm} {cmdline}", flags=re.IGNORECASE)
    title = ray_title.group(0) if ray_title is not None else comm
    return re.sub(r"[^A-Za-z0-9_.:+-]+", "_", title)[:64] or "unknown"


def collect_node_process_memory_snapshot() -> tuple[dict[int, dict[str, Any]], float]:
    """Collect compact per-process memory data for relevant Ray and vLLM roles."""

    started = time.perf_counter()
    processes: dict[int, dict[str, Any]] = {}
    try:
        process_entries = os.scandir("/proc")
    except OSError:
        return processes, 0.0

    with process_entries:
        for entry in process_entries:
            if not entry.name.isdigit():
                continue
            process_root = os.path.join("/proc", entry.name)
            try:
                with open(os.path.join(process_root, "comm"), encoding="utf-8") as stream:
                    comm = stream.read().strip()
                with open(os.path.join(process_root, "cmdline"), "rb") as stream:
                    cmdline = stream.read(4096).replace(b"\0", b" ").decode("utf-8", errors="replace")
            except OSError:
                continue

            group = _classify_process_memory_group(f"{comm} {cmdline}")
            if group is None:
                continue
            memory = _read_kib(os.path.join(process_root, "status"), {"VmRSS", "RssAnon"})
            processes[int(entry.name)] = {
                "group": group,
                "title": _process_memory_title(comm, cmdline),
                "rss_kib": int(memory.get("VmRSS", 0)),
                "anon_kib": int(memory.get("RssAnon", 0)),
            }
    return processes, (time.perf_counter() - started) * 1000.0


def _aggregate_process_memory_groups(processes: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for process in processes.values():
        values = groups.setdefault(str(process["group"]), [0, 0, 0])
        values[0] += 1
        values[1] += int(process["rss_kib"])
        values[2] += int(process["anon_kib"])
    return groups


def _format_process_memory_deltas(
    processes: dict[int, dict[str, Any]],
    previous_processes: dict[int, dict[str, Any]],
    scope: str,
) -> str:
    groups = _aggregate_process_memory_groups(processes)
    previous_groups = _aggregate_process_memory_groups(previous_processes)
    ordered_groups = [group for group, _ in _PROCESS_MEMORY_GROUPS] + ["ray_other"]
    group_deltas = []
    for group in ordered_groups:
        current = groups.get(group, [0, 0, 0])
        previous = previous_groups.get(group, [0, 0, 0])
        delta = [current[index] - previous[index] for index in range(3)]
        if any(delta):
            group_deltas.append(
                f"{group}:{delta[0]:+d}/{delta[1] / (1024**2):+.2f}/{delta[2] / (1024**2):+.2f}"
            )

    top_deltas = []
    for group in ("ray_other", "worker_tp"):
        candidates = []
        for pid, process in processes.items():
            if process["group"] != group:
                continue
            previous = previous_processes.get(pid)
            previous_anon_kib = (
                int(previous["anon_kib"])
                if previous is not None and previous.get("group") == group
                else 0
            )
            delta_kib = int(process["anon_kib"]) - previous_anon_kib
            if delta_kib > 0:
                candidates.append((delta_kib, pid, str(process["title"])))
        candidates.sort(reverse=True)
        if candidates:
            top_deltas.append(
                f"{group}["
                + ",".join(
                    f"{pid}@{title}:{delta_kib / (1024**2):+.3f}"
                    for delta_kib, pid, title in candidates[:8]
                )
                + "]"
            )
    return (
        f"proc_delta_scope={scope} proc_group_delta={','.join(group_deltas) or 'none'} "
        f"proc_top_anon_delta={';'.join(top_deltas) or 'none'}"
    )


def format_node_process_memory_summary(
    processes: Optional[dict[int, dict[str, Any]]] = None,
    *,
    previous_processes: Optional[dict[int, dict[str, Any]]] = None,
    delta_scope: str = "none",
    scan_ms: Optional[float] = None,
) -> str:
    """Aggregate process memory and optionally report PID-level anonymous deltas."""

    if processes is None:
        processes, scan_ms = collect_node_process_memory_snapshot()
    groups = _aggregate_process_memory_groups(processes)

    ordered_groups = [group for group, _ in _PROCESS_MEMORY_GROUPS] + ["ray_other"]
    formatted = []
    for group in ordered_groups:
        values = groups.get(group)
        if values is None:
            continue
        count, rss_kib, anon_kib = values
        formatted.append(
            f"{group}:{count}/{rss_kib / (1024**2):.2f}/{anon_kib / (1024**2):.2f}"
        )
    delta_summary = ""
    if previous_processes is not None:
        delta_summary = " " + _format_process_memory_deltas(
            processes,
            previous_processes,
            delta_scope,
        )
    return (
        f"proc_groups={','.join(formatted) or 'none'}{delta_summary} "
        f"proc_scan_ms={(scan_ms or 0.0):.1f}"
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


def trim_process_host_memory() -> dict[str, Any]:
    """Return free glibc arenas without forcing a Python GC cycle."""

    started = time.perf_counter()
    heap_trimmed = _trim_process_heap()
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": heap_trimmed,
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
    trimmed_before = _trim_process_heap()
    advised = 0
    failed = 0
    if drop_file_cache and checkpoint_path:
        try:
            advised, failed = _flush_and_drop_checkpoint_file_cache(os.fspath(checkpoint_path))
        except Exception:  # noqa: BLE001
            failed = 1
    return {
        "elapsed_sec": time.perf_counter() - started,
        "heap_trimmed": trimmed_before,
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
