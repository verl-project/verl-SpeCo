import ctypes
import gc
import json
import logging
import os
import re
import sys
import time
import weakref
from collections.abc import Mapping
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

_SMAPS_HEADER_PATTERN = re.compile(
    r"^[0-9A-Fa-f]+-[0-9A-Fa-f]+\s+\S+\s+\S+\s+\S+\s+\S+\s*(.*)$"
)
_SMAPS_MAPPING_GROUPS = (
    "heap",
    "anonymous",
    "npu",
    "torch",
    "shmem",
    "file",
    "stack",
    "special",
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
        "anon_pages_gib": system.get("AnonPages"),
        "sreclaimable_gib": system.get("SReclaimable"),
        "sunreclaim_gib": system.get("SUnreclaim"),
        "pagetables_gib": system.get("PageTables"),
        "kernel_stack_gib": system.get("KernelStack"),
        "mlocked_gib": system.get("Mlocked"),
        "unevictable_gib": system.get("Unevictable"),
    }


def format_checkpoint_memory_snapshot(
    counters: Optional[dict[str, Optional[int]]] = None,
) -> str:
    """Return concise Linux process/system memory counters without extra dependencies."""

    counters = counters if counters is not None else collect_checkpoint_memory_snapshot()
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
    command_prefix = " ".join(cmdline.strip().split()[:3])
    title = ray_title.group(0) if ray_title is not None else (command_prefix or comm)
    return re.sub(r"[^A-Za-z0-9_.:+-]+", "_", title)[:64] or "unknown"


def collect_node_process_memory_snapshot() -> tuple[dict[int, dict[str, Any]], float]:
    """Collect compact memory data for this user's Ray and related processes."""

    started = time.perf_counter()
    processes: dict[int, dict[str, Any]] = {}
    getuid = getattr(os, "getuid", None)
    current_uid = getuid() if callable(getuid) else None
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
                if current_uid is not None and entry.stat(follow_symlinks=False).st_uid != current_uid:
                    continue
                with open(os.path.join(process_root, "comm"), encoding="utf-8") as stream:
                    comm = stream.read().strip()
                with open(os.path.join(process_root, "cmdline"), "rb") as stream:
                    cmdline = stream.read(4096).replace(b"\0", b" ").decode("utf-8", errors="replace")
            except OSError:
                continue

            group = _classify_process_memory_group(f"{comm} {cmdline}")
            if group is None:
                if current_uid is None:
                    continue
                group = "user_other"
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
    ordered_groups = [group for group, _ in _PROCESS_MEMORY_GROUPS] + ["ray_other", "user_other"]
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
    for group in ("worker_dict", "agent_loop", "vllm_http", "worker_tp", "ray_other", "user_other"):
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
            limit = 3 if group in {"worker_dict", "agent_loop"} else 1
            top_deltas.append(
                f"{group}["
                + ",".join(
                    f"{pid}@{title}:{delta_kib / (1024**2):+.3f}"
                    for delta_kib, pid, title in candidates[:limit]
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
) -> str:
    """Aggregate process memory and optionally report PID-level anonymous deltas."""

    if processes is None:
        processes, _ = collect_node_process_memory_snapshot()
    groups = _aggregate_process_memory_groups(processes)

    ordered_groups = [group for group, _ in _PROCESS_MEMORY_GROUPS] + ["ray_other", "user_other"]
    formatted = []
    for group in ordered_groups:
        values = groups.get(group)
        if values is None:
            continue
        count, rss_kib, anon_kib = values
        formatted.append(
            f"{group}:{count}/{rss_kib / (1024**2):.2f}/{anon_kib / (1024**2):.2f}"
        )
    if previous_processes is not None:
        return _format_process_memory_deltas(
            processes,
            previous_processes,
            delta_scope,
        )
    return f"proc_groups={','.join(formatted) or 'none'}"


def _memory_counter_delta(
    current: dict[str, Optional[int]],
    previous: dict[str, Optional[int]],
    key: str,
) -> Optional[int]:
    current_value = current.get(key)
    previous_value = previous.get(key)
    if current_value is None or previous_value is None:
        return None
    return int(current_value) - int(previous_value)


def _format_kib_delta(value: Optional[int]) -> str:
    return "n/a" if value is None else f"{value / (1024**2):+.3f}"


def format_node_memory_delta_summary(
    memory: dict[str, Optional[int]],
    previous_memory: Optional[dict[str, Optional[int]]],
    processes: dict[int, dict[str, Any]],
    previous_processes: Optional[dict[int, dict[str, Any]]],
    *,
    delta_scope: str,
) -> str:
    """Explain node growth using process RSS/anonymous and kernel counters."""

    if previous_memory is None or previous_processes is None:
        return f"memory_delta_scope={delta_scope}"

    process_anon_delta = sum(int(item["anon_kib"]) for item in processes.values()) - sum(
        int(item["anon_kib"]) for item in previous_processes.values()
    )
    unavailable_delta = _memory_counter_delta(memory, previous_memory, "node_unavailable_gib")
    anon_pages_delta = _memory_counter_delta(memory, previous_memory, "anon_pages_gib")
    anon_residual = anon_pages_delta - process_anon_delta if anon_pages_delta is not None else None

    fields = {
        "node_unavailable_delta_gib": unavailable_delta,
        "node_anon_pages_delta_gib": anon_pages_delta,
        "proc_anon_delta_gib": process_anon_delta,
        "anon_pages_minus_proc_anon_delta_gib": anon_residual,
    }
    return " ".join(
        [f"memory_delta_scope={delta_scope}"]
        + [f"{name}={_format_kib_delta(value)}" for name, value in fields.items()]
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
        mallctl = getattr(runtime, "mallctl", None) or getattr(runtime, "je_mallctl", None)
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


def _jemalloc_refresh_stats() -> bool:
    try:
        mallctl = _jemalloc_mallctl_function()
        if mallctl is None:
            return False
        epoch = ctypes.c_uint64(1)
        epoch_size = ctypes.c_size_t(ctypes.sizeof(epoch))
        return (
            mallctl(
                b"epoch",
                ctypes.byref(epoch),
                ctypes.byref(epoch_size),
                ctypes.byref(epoch),
                ctypes.sizeof(epoch),
            )
            == 0
        )
    except Exception:  # noqa: BLE001
        return False


def _jemalloc_read_size(name: str) -> Optional[int]:
    try:
        mallctl = _jemalloc_mallctl_function()
        if mallctl is None:
            return None
        value = ctypes.c_size_t()
        value_size = ctypes.c_size_t(ctypes.sizeof(value))
        if mallctl(name.encode("ascii"), ctypes.byref(value), ctypes.byref(value_size), None, 0) != 0:
            return None
        return int(value.value)
    except Exception:  # noqa: BLE001
        return None


def collect_host_allocator_stats() -> dict[str, Any]:
    """Read allocator counters without allocating or reclaiming model buffers."""

    allocator = _host_allocator_name()
    stats = {
        "allocator": allocator,
        "allocated": None,
        "active": None,
        "resident": None,
        "retained": None,
    }
    if allocator != "jemalloc" or not _jemalloc_refresh_stats():
        return stats
    for key in ("allocated", "active", "resident", "retained"):
        stats[key] = _jemalloc_read_size(f"stats.{key}")
    return stats


def _classify_smaps_mapping(path: str) -> str:
    normalized = path.strip().lower()
    if normalized == "[heap]":
        return "heap"
    if normalized.startswith("[stack"):
        return "stack"
    if not normalized or normalized.startswith("[anon") or normalized.startswith("/dev/zero"):
        return "anonymous"
    if any(token in normalized for token in ("/dev/shm", "/sysv", "memfd:", "plasma")):
        return "shmem"
    if any(token in normalized for token in ("ascend", "torch_npu", "cann", "davinci", "hccl", "libacl")):
        return "npu"
    if any(token in normalized for token in ("site-packages/torch", "libtorch", "torch/lib")):
        return "torch"
    if normalized.startswith("["):
        return "special"
    return "file"


def collect_process_mapping_stats() -> dict[str, Any]:
    """Split this process's RSS and anonymous pages by Linux memory mapping."""

    stats = {
        "available": False,
        "anonymous_kib": 0,
        "anon_huge_pages_kib": 0,
        "private_dirty_kib": 0,
        "groups": {group: 0 for group in _SMAPS_MAPPING_GROUPS},
    }
    current_group = "special"
    try:
        with open("/proc/self/smaps", "r", encoding="utf-8") as stream:
            for line in stream:
                header = _SMAPS_HEADER_PATTERN.match(line)
                if header is not None:
                    current_group = _classify_smaps_mapping(header.group(1))
                    continue
                name, separator, raw_value = line.partition(":")
                if not separator or name not in {
                    "Anonymous",
                    "AnonHugePages",
                    "Private_Dirty",
                }:
                    continue
                fields = raw_value.strip().split()
                if not fields:
                    continue
                value = int(fields[0])
                if name == "Anonymous":
                    stats["anonymous_kib"] += value
                    stats["groups"][current_group] += value
                elif name == "AnonHugePages":
                    stats["anon_huge_pages_kib"] += value
                else:
                    stats["private_dirty_kib"] += value
        stats["available"] = True
    except (OSError, ValueError):
        pass
    return stats


def _iter_memory_buffers(value: Any, seen: set[int]):
    if value is None:
        return
    identifier = id(value)
    if identifier in seen:
        return
    seen.add(identifier)

    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            yield value, int(numel()) * int(element_size())
        except Exception:  # noqa: BLE001
            pass
        return

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            yield value, int(nbytes)
        except (TypeError, ValueError):
            pass
        return

    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (list, tuple, set)):
        children = value
    elif type(value).__module__.startswith("tensordict") and callable(getattr(value, "values", None)):
        try:
            children = value.values()
        except Exception:  # noqa: BLE001
            children = ()
    else:
        children = (
            child
            for child in (
                getattr(value, "batch", None),
                getattr(value, "non_tensor_batch", None),
            )
            if child is not None and child is not value
        )
    for child in children:
        yield from _iter_memory_buffers(child, seen)


def _weakref_or_none(value: Any):
    try:
        return weakref.ref(value)
    except TypeError:
        return None


def _format_allocator_stat(
    stats: dict[str, Any],
    previous: dict[str, Any],
    key: str,
    *,
    include_absolute: bool,
) -> str:
    value = stats.get(key)
    previous_value = previous.get(key)
    if value is None:
        prefix = f"jemalloc_{key}_gib=n/a " if include_absolute else ""
        return f"{prefix}jemalloc_{key}_delta_gib=n/a"
    delta = value - previous_value if previous_value is not None else None
    delta_text = f"{delta / (1024**3):+.3f}" if delta is not None else "n/a"
    prefix = f"jemalloc_{key}_gib={value / (1024**3):.3f} " if include_absolute else ""
    return f"{prefix}jemalloc_{key}_delta_gib={delta_text}"


def _format_mapping_stats(stats: dict[str, Any], previous: dict[str, Any]) -> str:
    if not stats.get("available"):
        return "smaps_available=0"

    previous_available = bool(previous.get("available"))
    group_values = stats.get("groups", {})
    previous_groups = previous.get("groups", {})
    if previous_available:
        group_delta_text = "|".join(
            f"{group}:{(int(group_values.get(group, 0)) - int(previous_groups.get(group, 0))) / (1024**2):+.3f}"
            for group in _SMAPS_MAPPING_GROUPS
        )
    else:
        group_delta_text = "n/a"

    fields = []
    for key in ("anonymous_kib", "anon_huge_pages_kib", "private_dirty_kib"):
        value = int(stats.get(key, 0))
        delta = value - int(previous.get(key, 0)) if previous_available else None
        delta_text = "n/a" if delta is None else f"{delta / (1024**2):+.3f}"
        fields.append(
            f"smaps_{key.removesuffix('_kib')}_gib={value / (1024**2):.3f} "
            f"smaps_{key.removesuffix('_kib')}_delta_gib={delta_text}"
        )
    return (
        "smaps_available=1 "
        + " ".join(fields)
        + f" smaps_anon_group_delta_gib={group_delta_text}"
    )


def _lifetime_state(
    state: dict[str, Any],
    prefix: str,
) -> tuple[int, tuple[Any, ...], int, int, int]:
    reference = state.get(f"{prefix}_ref")
    alive = int(reference() is not None) if callable(reference) else -1
    buffer_refs = state.get(f"{prefix}_buffer_refs", ())
    tracked_bytes = sum(size for _, size in buffer_refs)
    alive_buffers = sum(1 for reference, _ in buffer_refs if reference() is not None)
    alive_bytes = sum(size for reference, size in buffer_refs if reference() is not None)
    return alive, buffer_refs, tracked_bytes, alive_buffers, alive_bytes


def _collect_process_memory_diagnostics() -> dict[str, Any]:
    return {
        "allocator": collect_host_allocator_stats(),
        "mapping": collect_process_mapping_stats(),
        "process": _read_kib(
            "/proc/self/status",
            {"VmRSS", "RssAnon", "VmData"},
        ),
        "python_blocks": int(getattr(sys, "getallocatedblocks", lambda: 0)()),
    }


def _log_process_memory_diagnostics(
    *,
    role: str,
    method: str,
    call_index: int,
    phase: str,
    delta_scope: str,
    current: dict[str, Any],
    previous: dict[str, Any],
) -> None:
    allocator_stats = current["allocator"]
    previous_allocator_stats = previous.get("allocator", {})
    mapping_stats = current["mapping"]
    previous_mapping_stats = previous.get("mapping", {})
    process = current["process"]
    previous_process = previous.get("process", {})

    def process_delta(key: str) -> Optional[int]:
        value = process.get(key)
        previous_value = previous_process.get(key)
        if value is None or previous_value is None:
            return None
        return int(value) - int(previous_value)

    python_blocks = int(current["python_blocks"])
    previous_python_blocks = previous.get("python_blocks")
    python_blocks_delta = (
        python_blocks - int(previous_python_blocks) if previous_python_blocks is not None else None
    )
    include_allocator_absolute = not bool(previous_allocator_stats)
    allocator_text = " ".join(
        _format_allocator_stat(
            allocator_stats,
            previous_allocator_stats,
            name,
            include_absolute=include_allocator_absolute,
        )
        for name in ("allocated", "active", "resident", "retained")
    )
    print(
        f"[speco process memory] role={role} method={method} call={call_index} "
        f"phase={phase} pid={os.getpid()} memory_delta_scope={delta_scope} "
        f"rss_gib={process.get('VmRSS', 0) / (1024**2):.3f} "
        f"rss_delta_gib={_format_kib_delta(process_delta('VmRSS'))} "
        f"anon_gib={process.get('RssAnon', 0) / (1024**2):.3f} "
        f"anon_delta_gib={_format_kib_delta(process_delta('RssAnon'))} "
        f"vm_data_gib={process.get('VmData', 0) / (1024**2):.3f} "
        f"vm_data_delta_gib={_format_kib_delta(process_delta('VmData'))} "
        f"python_blocks_delta={python_blocks_delta if python_blocks_delta is not None else 'n/a'} "
        f"allocator={allocator_stats['allocator']} {allocator_text} "
        f"{_format_mapping_stats(mapping_stats, previous_mapping_stats)}",
        flush=True,
    )


def log_previous_output_lifetime(owner: Any, key: str, *, role: str, method: str) -> int:
    """Log whether prior Ray-call inputs or outputs remain live without retaining them."""

    states = getattr(owner, "_speco_output_lifetime_states", None)
    if not isinstance(states, dict):
        states = {}
        setattr(owner, "_speco_output_lifetime_states", states)
    state = states.setdefault(key, {})
    call_index = int(state.get("call", 0)) + 1
    state["call"] = call_index
    if call_index > 8 and call_index % 10 != 0:
        return call_index

    input_alive, input_buffer_refs, input_tracked_bytes, input_alive_buffers, input_alive_bytes = (
        _lifetime_state(state, "input")
    )
    output_alive, output_buffer_refs, output_tracked_bytes, output_alive_buffers, output_alive_bytes = (
        _lifetime_state(state, "output")
    )

    print(
        f"[speco output lifetime] role={role} method={method} call={call_index} "
        f"pid={os.getpid()} previous_input_call={state.get('input_tracked_call', 'none')} "
        f"previous_input_alive={input_alive} input_tracked_buffers={len(input_buffer_refs)} "
        f"input_alive_buffers={input_alive_buffers} "
        f"input_tracked_mib={input_tracked_bytes / (1024**2):.2f} "
        f"input_alive_mib={input_alive_bytes / (1024**2):.2f} "
        f"previous_call={state.get('output_tracked_call', 'none')} "
        f"previous_output_alive={output_alive} tracked_buffers={len(output_buffer_refs)} "
        f"alive_buffers={output_alive_buffers} tracked_mib={output_tracked_bytes / (1024**2):.2f} "
        f"alive_mib={output_alive_bytes / (1024**2):.2f}",
        flush=True,
    )
    process_memory = _collect_process_memory_diagnostics()
    previous_process_memory = state.get("after_process_memory")
    previous_after_call = state.get("after_process_memory_call")
    if not isinstance(previous_process_memory, dict):
        previous_process_memory = state.get("entry_process_memory", {})
        previous_after_call = state.get("entry_process_memory_call")
        previous_scope = (
            f"since_before_call_{previous_after_call}"
            if previous_after_call is not None
            else "none"
        )
    else:
        previous_scope = f"since_after_call_{previous_after_call}"
    _log_process_memory_diagnostics(
        role=role,
        method=method,
        call_index=call_index,
        phase="before",
        delta_scope=previous_scope,
        current=process_memory,
        previous=previous_process_memory,
    )
    state["entry_process_memory"] = process_memory
    state["entry_process_memory_call"] = call_index
    return call_index


def log_process_memory_after_call(
    owner: Any,
    key: str,
    call_index: int,
    *,
    role: str,
    method: str,
    phase: str = "after",
) -> None:
    """Log process growth inside one instrumented worker call."""

    if call_index > 8 and call_index % 10 != 0:
        return
    states = getattr(owner, "_speco_output_lifetime_states", None)
    if not isinstance(states, dict):
        return
    state = states.get(key)
    if not isinstance(state, dict):
        return

    process_memory = _collect_process_memory_diagnostics()
    entry_process_memory = state.get("entry_process_memory", {})
    entry_call = state.get("entry_process_memory_call")
    _log_process_memory_diagnostics(
        role=role,
        method=method,
        call_index=call_index,
        phase=phase,
        delta_scope=f"call_{entry_call}_entry" if entry_call is not None else "none",
        current=process_memory,
        previous=entry_process_memory,
    )
    state["after_process_memory"] = process_memory
    state["after_process_memory_call"] = call_index


def _remember_lifetime_value(
    state: dict[str, Any],
    prefix: str,
    call_index: int,
    value: Any,
) -> None:
    if call_index < int(state.get(f"{prefix}_tracked_call", 0)):
        return
    buffer_refs = []
    for buffer, size in _iter_memory_buffers(value, set()):
        reference = _weakref_or_none(buffer)
        if reference is not None:
            buffer_refs.append((reference, max(0, int(size))))
    state[f"{prefix}_tracked_call"] = call_index
    state[f"{prefix}_ref"] = _weakref_or_none(value)
    state[f"{prefix}_buffer_refs"] = tuple(buffer_refs)


def remember_input_lifetime(owner: Any, key: str, call_index: int, value: Any) -> None:
    """Store weak references to a Ray-call input for the next call check."""

    states = getattr(owner, "_speco_output_lifetime_states", None)
    if not isinstance(states, dict):
        return
    state = states.get(key)
    if isinstance(state, dict):
        _remember_lifetime_value(state, "input", call_index, value)


def remember_output_lifetime(owner: Any, key: str, call_index: int, output: Any) -> None:
    """Store only weak references for the next output-lifetime check."""

    states = getattr(owner, "_speco_output_lifetime_states", None)
    if not isinstance(states, dict):
        return
    state = states.get(key)
    if not isinstance(state, dict):
        return
    _remember_lifetime_value(state, "output", call_index, output)


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
            advised, failed = _flush_and_drop_checkpoint_file_cache(os.fspath(checkpoint_path))
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
