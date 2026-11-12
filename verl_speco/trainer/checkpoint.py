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
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import sys
import time
from functools import lru_cache
from glob import glob
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Union


_DRAFT_STEP_PATTERN = re.compile(r"^draft_step_(\d+)$")
_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.index.json",
)
_OPTIMIZER_DCP_METADATA = ".metadata"
_OPTIMIZER_LAYOUT = "standalone_optimizer_layout.json"

logger = logging.getLogger(__name__)

_JEMALLOC_ARENAS_ALL = 4096
_JEMALLOC_RECLAIM_MODE_ENV = "SPECO_JEMALLOC_RECLAIM_MODE"


class DrafterCheckpointMetadataError(ValueError):
    """Raised when a managed drafter checkpoint has invalid metadata."""


_STANDALONE_INTEGRITY = "standalone_integrity.json"


class StandaloneCheckpointDirectoryLease:
    """Persistent cooperative directory ownership; crashes leave a stale marker."""

    def __init__(self, directory, token):
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("Invalid checkpoint directory ownership token")
        self.directory = Path(directory)
        self.path = self.directory / ".standalone-checkpoint-owner"
        self.token = token

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"Checkpoint directory already has an owner marker: {self.path}") from exc
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(self.token)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "posix":
            parent = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)

    def verify(self):
        if self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_size != 32:
            raise RuntimeError("Checkpoint directory ownership marker is missing or invalid")
        if self.path.read_text(encoding="ascii") != self.token:
            raise RuntimeError("Checkpoint directory ownership token differs")

    def release(self):
        self.verify()
        self.path.unlink()


def write_standalone_trainable_snapshot(directory, *, step, tensors, training_plan):
    """Publish a separate trainable-only export; it cannot restore optimizer state."""
    import torch
    from safetensors.torch import save_file

    if type(step) is not int or step < 0 or not isinstance(tensors, dict) or not tensors or not isinstance(training_plan, dict):
        raise ValueError("Trainable snapshot requires a step and nonempty tensor mapping")
    if any(not isinstance(name, str) or not name or not torch.is_tensor(value)
           for name, value in tensors.items()):
        raise ValueError("Invalid trainable snapshot tensor mapping")
    identity = json.loads(json.dumps(training_plan, sort_keys=True, allow_nan=False))
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / "trainable.safetensors"
    temporary = destination / "trainable.safetensors.incomplete"
    try:
        weights = {name: value.detach().to(device="cpu").contiguous() for name, value in tensors.items()}
        save_file(weights, str(temporary))
        with temporary.open("rb+") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
            os.fsync(source.fileno())
        os.replace(temporary, payload)
        metadata = {
            "schema_version": 1, "kind": "trainable_parameters_only", "optimizer_step": step,
            "training_plan": identity, "weights_file": payload.name, "weights_sha256": checksum,
            "parameters": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for name, value in weights.items()},
        }
        temp_manifest = destination / "manifest.json.incomplete"
        with temp_manifest.open("x", encoding="utf-8") as target:
            json.dump(metadata, target, sort_keys=True, allow_nan=False)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_manifest, destination / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def read_standalone_trainable_snapshot(directory, *, expected_training_plan=None):
    """Validate a trainable export before a caller applies it to a matching model."""
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest_path, payload = directory / "manifest.json", directory / "trainable.safetensors"
    if (directory.is_symlink() or manifest_path.is_symlink() or payload.is_symlink()
            or not manifest_path.is_file() or manifest_path.stat().st_size > 16 * 1024 * 1024
            or not payload.is_file()):
        raise ValueError("Invalid trainable snapshot files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("kind") != "trainable_parameters_only"
            or manifest.get("weights_file") != payload.name
            or type(manifest.get("optimizer_step")) is not int or manifest["optimizer_step"] < 0):
        raise ValueError("Invalid trainable snapshot manifest")
    if expected_training_plan is not None and manifest.get("training_plan") != expected_training_plan:
        raise ValueError("Trainable snapshot training identity differs")
    before = payload.stat()
    with payload.open("rb") as source:
        opened = os.fstat(source.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ValueError("Trainable snapshot changed while opening")
        if hashlib.file_digest(source, "sha256").hexdigest() != manifest.get("weights_sha256"):
            raise ValueError("Trainable snapshot checksum differs")
    weights = load_file(str(payload), device="cpu")
    after = payload.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Trainable snapshot changed while reading")
    schema = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in weights.items()}
    if not weights or schema != manifest.get("parameters"):
        raise ValueError("Trainable snapshot parameter schema differs")
    return weights, manifest


def _standalone_integrity_run_id(path: Path) -> str | None:
    manifest_path = path / _STANDALONE_INTEGRITY
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 16 * 1024 * 1024:
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest.get("run_id") if isinstance(manifest, dict) else None
        return run_id if isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id) else None
    except (OSError, ValueError):
        return None


def _standalone_checkpoint_inventory(path: Path) -> dict[str, dict[str, Any]]:
    """Hash regular checkpoint files without following symlinks or loading tensors."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Checkpoint must be a real directory")
    inventory = {}
    def fail(error):
        raise error
    for root, directories, files in os.walk(path, followlinks=False, onerror=fail):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError("Checkpoint contains a directory symlink")
        for name in files:
            item = Path(root) / name
            relative = item.relative_to(path).as_posix()
            if item.is_symlink() or not item.is_file():
                raise ValueError("Checkpoint contains a non-regular file")
            if relative in {"metadata.json", _STANDALONE_INTEGRITY}:
                continue
            if name.endswith((".incomplete", ".tmp")):
                raise ValueError("Checkpoint contains incomplete files")
            before = item.stat()
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = item.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            ):
                raise ValueError(f"Checkpoint file changed during verification: {relative}")
            inventory[relative] = {"size": before.st_size, "sha256": digest.hexdigest()}
            if len(inventory) > 100_000:
                raise ValueError("Checkpoint contains too many files")
    return inventory


def _standalone_metadata_digest(metadata: dict[str, Any]) -> str:
    stable = {key: value for key, value in metadata.items() if key != "complete"}
    data = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _optimizer_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive:
        raise ValueError("Invalid optimizer checkpoint dependency path")
    path = Path(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Invalid optimizer checkpoint dependency path")
    return path.as_posix()


def write_standalone_optimizer_layout(directory: str | os.PathLike, metadata: Any) -> None:
    """Record DCP dependencies returned by the save operation without unpickling metadata."""
    root = Path(directory)
    entries = getattr(metadata, "storage_data", None)
    if not isinstance(entries, dict) or not entries or len(entries) > 1_000_000:
        raise ValueError("Optimizer checkpoint has no supported DCP storage layout")
    files: dict[str, int] = {}
    for item in entries.values():
        relative = _optimizer_relative_path(getattr(item, "relative_path", None))
        offset, length = getattr(item, "offset", None), getattr(item, "length", None)
        if type(offset) is not int or offset < 0 or type(length) is not int or length <= 0:
            raise ValueError("Invalid optimizer checkpoint storage extent")
        files[relative] = max(files.get(relative, 0), offset + length)
    metadata_path = root / _OPTIMIZER_DCP_METADATA
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("Optimizer checkpoint metadata is missing")
    with metadata_path.open("rb") as stream:
        metadata_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    for relative, end in files.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size < end:
            raise ValueError("Optimizer checkpoint dependency is missing or truncated")
    atomic_write_json(root / _OPTIMIZER_LAYOUT, {"schema_version": 1, "metadata_sha256": metadata_hash, "files": files})


def _validate_standalone_optimizer_layout(path: Path, inventory: dict[str, dict[str, Any]]) -> None:
    optimizer = get_drafter_optimizer_checkpoint_path(path)
    if not optimizer:
        raise ValueError("Standalone integrity requires optimizer state")
    relative_root = Path(optimizer).relative_to(path)
    layout_path = Path(optimizer) / _OPTIMIZER_LAYOUT
    if layout_path.is_symlink() or not layout_path.is_file() or layout_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Standalone optimizer dependency layout is missing")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(layout, dict) or set(layout) != {"schema_version", "metadata_sha256", "files"}:
        raise ValueError("Invalid standalone optimizer dependency layout")
    files = layout["files"]
    metadata_key = (relative_root / _OPTIMIZER_DCP_METADATA).as_posix()
    if (type(layout["schema_version"]) is not int or layout["schema_version"] != 1
            or not isinstance(files, dict) or not 1 <= len(files) <= 100_000
            or inventory.get(metadata_key, {}).get("sha256") != layout["metadata_sha256"]):
        raise ValueError("Optimizer checkpoint dependency metadata differs")
    for name, end in files.items():
        relative = _optimizer_relative_path(name)
        record = inventory.get((relative_root / relative).as_posix())
        if type(end) is not int or end <= 0 or not isinstance(record, dict) or record["size"] < end:
            raise ValueError("Optimizer checkpoint dependency is missing or truncated")


def write_standalone_integrity_manifest(
    checkpoint_path: str | os.PathLike, *, run_id: str, require_resume: bool = False,
) -> None:
    """Record the finished files before standalone marks the checkpoint complete."""
    path = Path(checkpoint_path)
    if (not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
            or type(require_resume) is not bool):
        raise ValueError("Standalone integrity requires a unique run ID")
    metadata = get_drafter_checkpoint_metadata(path)
    step = _managed_checkpoint_step(str(path))
    if step is None or metadata.get("step") != step or metadata.get("complete") is not False:
        raise ValueError("Standalone checkpoint is not ready for integrity sealing")
    if not get_drafter_optimizer_checkpoint_path(path):
        raise ValueError("Standalone integrity requires a complete optimizer checkpoint")
    if require_resume:
        from verl_speco.trainer.standalone_resume import load_standalone_resume

        _, resume = load_standalone_resume(path)
        if resume is None or resume.get("optimizer_step") != step:
            raise ValueError("Standalone checkpoint is missing matching replay progress")
    inventory = _standalone_checkpoint_inventory(path)
    _validate_standalone_optimizer_layout(path, inventory)
    if "config.json" not in inventory or not any(
        name.endswith((".safetensors", ".bin")) for name in inventory
    ):
        raise ValueError("Standalone checkpoint is missing model config or weights")
    atomic_write_json(str(path / _STANDALONE_INTEGRITY), {
        "schema_version": 2, "step": step, "run_id": run_id,
        "resume_required": bool(require_resume),
        "metadata_sha256": _standalone_metadata_digest(metadata), "files": inventory,
    })


def verify_standalone_checkpoint_integrity(checkpoint_path: str | os.PathLike, *, require_optimizer_layout: bool = False) -> bool:
    """Validate the complete checkpoint and all files against its sealed inventory."""
    path = Path(checkpoint_path)
    manifest_path = path / _STANDALONE_INTEGRITY
    if path.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        return False
    try:
        if manifest_path.stat().st_size > 16 * 1024 * 1024:
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False
        metadata = get_drafter_checkpoint_metadata(path)
        step = _managed_checkpoint_step(str(path))
        version = manifest.get("schema_version")
        if version not in {1, 2} or (require_optimizer_layout and version != 2):
            return False
        inventory = _standalone_checkpoint_inventory(path)
        if version == 2:
            _validate_standalone_optimizer_layout(path, inventory)
        if manifest.get("resume_required") is True:
            from verl_speco.trainer.standalone_resume import load_standalone_resume

            _, resume = load_standalone_resume(path)
            if resume is None or resume.get("optimizer_step") != step:
                return False
        return bool(
            isinstance(manifest, dict)
            and type(version) is int and version in {1, 2}
            and type(manifest.get("resume_required")) is bool
            and isinstance(manifest.get("run_id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", manifest["run_id"]) is not None
            and step is not None
            and manifest.get("step") == step == metadata.get("step")
            and metadata.get("complete") is True
            and manifest.get("metadata_sha256") == _standalone_metadata_digest(metadata)
            and is_pretrained_drafter_checkpoint(path)
            and get_drafter_optimizer_checkpoint_path(path)
            and manifest.get("files") == inventory
        )
    except (OSError, ValueError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
        return False


def standalone_retention_resume_identity(root, checkpoint_path, *, optimizer_step):
    """Validate a same-directory resume before inheriting its retention identity.

    The caller must hold the cooperative directory lease throughout training.
    Existing later step names are rejected, even if damaged, to avoid treating a
    rollback or a conflicting writer as a continuation of the same lineage.
    """
    root, checkpoint = Path(root), Path(checkpoint_path)
    if (type(optimizer_step) is not int or optimizer_step <= 0 or root.is_symlink()
            or not root.is_dir() or checkpoint.is_symlink()
            or checkpoint.name != f"draft_step_{optimizer_step}"
            or checkpoint.resolve().parent != root.resolve()
            or not verify_standalone_checkpoint_integrity(checkpoint)):
        raise ValueError("Retention inheritance requires a verified checkpoint in the owned directory")
    manifest = json.loads((checkpoint / _STANDALONE_INTEGRITY).read_text(encoding="utf-8"))
    if manifest.get("resume_required") is not True:
        raise ValueError("Retention inheritance requires sealed replay progress")
    if any((step := _managed_checkpoint_step(str(path))) is not None and step > optimizer_step
           for path in root.iterdir()):
        raise ValueError("Retention inheritance cannot resume before an existing later checkpoint")
    return manifest["run_id"]


def retire_standalone_checkpoints(
    root: str | os.PathLike, *, current_step: int, keep_last: int,
    protected_paths: tuple[str, ...] = (), delete: bool = False, ownership_check=None,
) -> list[str]:
    """Plan, or explicitly remove, old sealed checkpoints after two verified successors."""
    if (type(keep_last) is not int or keep_last < 2 or type(current_step) is not int
            or current_step < 0 or type(delete) is not bool
            or (ownership_check is not None and not callable(ownership_check))):
        raise ValueError("Checkpoint retention requires keep_last >= 2, a valid current step and boolean delete")
    original_root = Path(root)
    if original_root.is_symlink() or not original_root.is_dir():
        raise ValueError("Checkpoint retention root must be a real directory")
    base = original_root.resolve()
    protected = [Path(path).resolve() for path in protected_paths if path]
    current = base / f"draft_step_{current_step}"
    if not verify_standalone_checkpoint_integrity(current, require_optimizer_layout=True):
        raise ValueError("Current checkpoint is not verified for retention")
    run_id = _standalone_integrity_run_id(current)
    valid = []
    for path in base.iterdir():
        step = _managed_checkpoint_step(str(path))
        if (step is not None and not path.is_symlink()
                and _standalone_integrity_run_id(path) == run_id
                and verify_standalone_checkpoint_integrity(path, require_optimizer_layout=True)):
            valid.append((step, path))
    valid.sort(key=lambda item: item[0])
    if not any(step == current_step for step, _ in valid):
        raise ValueError("Current checkpoint is not verified for retention")
    protected_steps = {step for step, _ in valid[-keep_last:]}
    candidates = []
    for step, path in valid:
        if step >= current_step or step in protected_steps:
            continue
        if any(p == path or path in p.parents for p in protected):
            continue
        candidates.append(path)
    if delete:
        for path in candidates:
            if ownership_check is not None:
                ownership_check()
            # Recheck the candidate and all selected successors before each deletion.
            survivors = [(step, item) for step, item in valid if step > _managed_checkpoint_step(str(path))]
            if (len(survivors) < 2 or path.is_symlink() or path.resolve().parent != base
                    or not verify_standalone_checkpoint_integrity(path, require_optimizer_layout=True)
                    or not all(verify_standalone_checkpoint_integrity(item, require_optimizer_layout=True)
                               for _, item in survivors)):
                raise RuntimeError("Checkpoint retention revalidation failed")
            # Make an interrupted removal unambiguously non-resumable before
            # deleting any payload file. Never change the sealed run identity.
            before = path.stat()
            metadata = get_drafter_checkpoint_metadata(path)
            if (metadata.get("complete") is not True
                    or metadata.get("step") != _managed_checkpoint_step(str(path))
                    or _standalone_integrity_run_id(path) != run_id):
                raise RuntimeError("Checkpoint retention identity changed before removal")
            atomic_write_json(path / "metadata.json", {**metadata, "complete": False})
            if os.name == "posix":
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            after = path.stat()
            if (path.is_symlink() or path.resolve().parent != base
                    or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                    or _standalone_integrity_run_id(path) != run_id):
                raise RuntimeError("Checkpoint retention identity changed after invalidation")
            if ownership_check is not None:
                ownership_check()
            shutil.rmtree(path)
    return [str(path) for path in candidates]


def resolve_verified_standalone_checkpoint(
    requested: str | os.PathLike, *, input_path: str | os.PathLike | None = None,
) -> str:
    """Return the newest verified checkpoint no later than the requested one.

    Rollback remains within the requested checkpoint's original audited run.
    Replay input must match when the caller supplies one.
    """
    path = Path(requested)
    requested_step = _managed_checkpoint_step(str(path))
    if requested_step is None or path.is_symlink() or not path.is_dir():
        raise ValueError("Automatic rollback requires a managed checkpoint directory")
    run_id = _standalone_integrity_run_id(path)
    if run_id is None:
        raise ValueError("Requested checkpoint has no readable integrity identity")
    candidates = []
    for candidate in path.parent.iterdir():
        step = _managed_checkpoint_step(str(candidate))
        if (step is None or candidate.name != f"draft_step_{step}" or step > requested_step
                or candidate.is_symlink() or _standalone_integrity_run_id(candidate) != run_id):
            continue
        candidates.append((step, candidate))
    input_fingerprints: dict[bool, dict[str, Any]] = {}
    for step, candidate in sorted(candidates, key=lambda value: value[0], reverse=True):
        if not verify_standalone_checkpoint_integrity(candidate):
            continue
        if input_path is not None:
            from verl_speco.trainer.standalone_resume import (
                build_input_fingerprint, input_content_identity, load_standalone_resume, validate_input_identity,
            )

            try:
                _, resume = load_standalone_resume(candidate)
                if resume is None:
                    continue
                saved = resume.get("input_fingerprint")
                has_identity = "input_content_identity" in resume
                saved_identity = resume.get("input_content_identity")
                if has_identity:
                    saved_identity = validate_input_identity(saved_identity)
                    if input_content_identity(saved) != saved_identity:
                        continue
                has_hash = saved_identity is not None or (isinstance(saved, dict) and "sha256" in saved)
                if has_hash not in input_fingerprints:
                    input_fingerprints[has_hash] = build_input_fingerprint(
                        input_path, include_content_hash=has_hash,
                    )
                current = input_fingerprints[has_hash]
                if (input_content_identity(current) != saved_identity if saved_identity is not None else saved != current):
                    continue
            except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
                continue
            if resume.get("optimizer_step") != step:
                continue
        return str(candidate)
    raise ValueError("No verified compatible checkpoint is available for rollback")


def validate_checkpoint_buffer_ownership(state: dict[str, Any]) -> None:
    """Legacy partial Bubble checkpoints cannot restore their replay owners."""
    for sample in state.get("data_buffer", []):
        if not isinstance(sample, dict):
            continue
        if sample.get("_drafter_reserved_by") is not None or sample.get(
            "_drafter_replay_pinned", False
        ):
            raise DrafterCheckpointMetadataError(
                "Checkpoint contains unfinished Bubble replay ownership. "
                "Resume from a checkpoint saved after quota publication; "
                "clearing replay pins would silently discard unfinished training."
            )


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
        or bool(PureWindowsPath(relative_path).drive)
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
    *, compact: bool = False,
) -> str:
    """Write JSON to ``path`` atomically (tmp file + ``os.replace``)."""
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{target}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=None if compact else 2,
                      separators=(",", ":") if compact else None, allow_nan=not compact)
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
        raise FreezeBranchManifestError(f"freeze branch manifest not found: {path}")
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
            draft_dir = os.path.join(os.fspath(drafter_root), f"draft_step_{step_int}")
            if not is_pretrained_drafter_checkpoint(draft_dir):
                problems.append(f"missing or incomplete drafter checkpoint {draft_dir}")
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
            if (
                trigger_step is None
                or checkpoint_step is None
                or int(trigger_step) != int(step)
                or int(checkpoint_step) != int(step)
            ):
                problems.append(
                    "manifest step mismatch: "
                    f"trigger_step={trigger_step} checkpoint_step="
                    f"{checkpoint_step} expected {step}"
                )

    if problems:
        raise FreezeBranchManifestError(
            "freeze branch checkpoint validation failed: " + "; ".join(problems)
        )
    assert manifest is not None
    return manifest
