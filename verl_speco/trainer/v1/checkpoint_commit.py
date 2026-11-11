# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Commit a V1 joint recovery point before exposing it or pruning its predecessor."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from omegaconf import OmegaConf, open_dict


_LATEST_TRACKER = "latest_checkpointed_iteration.txt"
_PENDING_MARKER = ".speco_checkpoint_pending"


def validate_checkpoint_step(root, step):
    """Keep a rewind from overwriting the recovery point published in this root."""
    tracker = Path(root) / _LATEST_TRACKER
    try:
        contents = tracker.read_text(encoding="utf-8")
    except FileNotFoundError:
        latest = None
    else:
        try:
            latest = int(contents)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid V1 Bubble checkpoint tracker: {tracker}"
            ) from exc
    reason = None
    if latest is not None and step <= latest:
        reason = f"latest already points to step {latest}"
    else:
        committed = max(committed_steps(root, through_step=None), default=None)
        if committed is not None and step <= committed:
            reason = f"a complete joint manifest exists at step {committed}"
    if reason is not None:
        raise RuntimeError(
            f"Cannot save V1 Bubble checkpoint at step {step} in {root}: "
            f"{reason}. "
            "Resume into a new checkpoint output directory to save after a rewind."
        )


@contextmanager
def temporary_values(mapping, values):
    missing = object()
    previous = {key: mapping.get(key, missing) for key in values}
    scope = open_dict(mapping) if OmegaConf.is_config(mapping) else nullcontext()
    with scope:
        try:
            mapping.update(values)
            yield
        finally:
            for key, value in previous.items():
                if value is missing:
                    mapping.pop(key, None)
                else:
                    mapping[key] = value


def validate_synchronous_checkpoint_config(config, *, use_critic=False):
    """Require worker saves to complete before publishing a joint checkpoint."""
    actor = config.actor_rollout_ref.actor
    if (actor.get("checkpoint") or {}).get("async_save", False):
        raise ValueError(
            "V1 Bubble joint checkpoint requires synchronous actor checkpoint saves"
        )
    if use_critic and (config.critic.get("checkpoint") or {}).get("async_save", False):
        raise ValueError(
            "V1 Bubble joint checkpoint requires synchronous critic checkpoint saves"
        )


@contextmanager
def defer_native_checkpoint_commit(config, *, use_critic=False, root=None):
    """Suppress only driver-side latest publication and premature retention.

    V1's native save checks the driver's async_save setting only when writing
    latest. Worker actors retain their already-serialized synchronous configs;
    no async-save argument is sent by that RPC.
    """
    validate_synchronous_checkpoint_config(config, use_critic=use_critic)
    actor = config.actor_rollout_ref.actor
    with ExitStack() as stack:
        if root is not None:
            stack.enter_context(_protect_latest_tracker(Path(root)))
        if actor.get("checkpoint") is None:
            stack.enter_context(temporary_values(actor, {"checkpoint": {}}))
        stack.enter_context(temporary_values(actor["checkpoint"], {"async_save": True}))
        stack.enter_context(
            temporary_values(
                config.trainer,
                {
                    "remove_previous_ckpt_in_save": False,
                    "max_actor_ckpt_to_keep": None,
                    "max_critic_ckpt_to_keep": None,
                },
            )
        )
        yield


def _fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _replace_latest_tracker(root, contents):
    fd, temporary = tempfile.mkstemp(prefix=".speco_latest.", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / _LATEST_TRACKER)
        _fsync_directory(root)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_tracker_bytes(tracker):
    try:
        return tracker.read_bytes()
    except FileNotFoundError:
        return None


@contextmanager
def _protect_latest_tracker(root):
    tracker = root / _LATEST_TRACKER
    previous = _read_tracker_bytes(tracker)
    try:
        yield
    finally:
        if _read_tracker_bytes(tracker) != previous:
            if previous is None:
                tracker.unlink(missing_ok=True)
                _fsync_directory(root)
            else:
                _replace_latest_tracker(root, previous)
            raise RuntimeError(
                "Native V1 checkpoint changed latest before the joint commit; "
                "the previous tracker state was restored"
            )


def begin_checkpoint_commit(root, step):
    """Mark a joint save as unpublished before any component writes."""
    root = Path(root).resolve()
    folder = root / f"global_step_{int(step)}"
    if folder.resolve() != folder:
        raise RuntimeError(f"V1 Bubble checkpoint directory is an alias: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    try:
        (folder / _PENDING_MARKER).touch(exist_ok=False)
    except FileExistsError:
        # A failed attempt can be retried at the same step.
        pass
    _fsync_directory(folder)
    _fsync_directory(root)


def is_unpublished_checkpoint(folder):
    folder = Path(folder)
    if not (folder / _PENDING_MARKER).exists():
        return False
    try:
        latest = int((folder.parent / _LATEST_TRACKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return folder.name != f"global_step_{latest}"


def saved_drafter_entries(entries):
    if not isinstance(entries, list):
        return []
    return [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("saved") is True
        and isinstance(entry.get("path"), str)
        and entry["path"].strip()
    ]


def _is_complete_bubble_manifest(manifest, step):
    if not isinstance(manifest, dict):
        return False
    bubble = manifest.get("bubble")
    return (
        manifest.get("format") == "speco_v1_joint_checkpoint"
        and manifest.get("version") == 1
        and manifest.get("global_step") == int(step)
        and isinstance(bubble, dict)
        and bubble.get("quota_complete") is True
        and bool(saved_drafter_entries(manifest.get("drafter_checkpoints")))
    )


def commit_latest(root, step):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"global_step_{int(step)}"
    if folder.resolve() != folder:
        raise RuntimeError(f"V1 Bubble checkpoint directory is an alias: {folder}")
    manifest = json.loads(
        (folder / "speco_v1_manifest.json").read_text(encoding="utf-8")
    )
    if not _is_complete_bubble_manifest(manifest, step):
        raise RuntimeError("Cannot publish an incomplete V1 Bubble recovery point")
    # If a previous process stopped after publishing latest, that pointer
    # proves its checkpoint committed. Clear its marker before replacing it.
    tracker = root / _LATEST_TRACKER
    try:
        previous = int(tracker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    else:
        previous_folder = root / f"global_step_{previous}"
        marker = previous_folder / _PENDING_MARKER
        if marker.exists():
            if previous_folder.resolve() != previous_folder:
                raise RuntimeError(
                    f"V1 Bubble checkpoint directory is an alias: {previous_folder}"
                )
            marker.unlink()
            _fsync_directory(previous_folder)
    _fsync_directory(folder)
    _replace_latest_tracker(root, str(int(step)).encode("utf-8"))
    try:
        (folder / _PENDING_MARKER).unlink(missing_ok=True)
        _fsync_directory(folder)
    except OSError:
        logging.getLogger(__name__).warning(
            "V1 Bubble checkpoint committed but marker cleanup failed: %s",
            folder,
            exc_info=True,
        )


def committed_steps(root, through_step):
    steps = []
    for folder in Path(root).resolve().glob("global_step_*"):
        suffix = folder.name.removeprefix("global_step_")
        if (
            not suffix.isdecimal()
            or (through_step is not None and int(suffix) > through_step)
            or folder.resolve() != folder
        ):
            continue
        try:
            manifest = json.loads(
                (folder / "speco_v1_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if _is_complete_bubble_manifest(
            manifest, int(suffix)
        ) and not is_unpublished_checkpoint(folder):
            steps.append(int(suffix))
    return sorted(steps)


@contextmanager
def _checkpoint_prune_lock(target):
    # Keep the lock file in place so all workers lock the same inode.
    lock_path = target.with_name(f".{target.name}.speco-prune.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            yield False
        else:
            yield True
    finally:
        os.close(fd)


def prune_checkpoint_roles(worker, root, steps, role, keep=None, through_step=None):
    """Run on every owning worker, including nodes with private local storage."""
    del worker
    root = Path(root).resolve()
    if not role or Path(role).name != role or role in {".", ".."}:
        raise ValueError("Invalid checkpoint role")
    if keep is not None:
        if keep <= 0 or through_step is None:
            raise ValueError("Checkpoint retention requires a positive count and step")
        # Driver manifests certify components on worker-private filesystems.
        # A worker may also see complete manifests absent from the driver's view.
        visible_steps = set(steps).union(committed_steps(root, through_step))
        steps = sorted(visible_steps)[:-keep]
    targets = []
    for step in steps:
        expected_folder = root / f"global_step_{int(step)}"
        folder = expected_folder.resolve()
        target = (folder / role).resolve()
        if folder != expected_folder or target != folder / role:
            raise ValueError(
                "Checkpoint retention target escapes its recovery directory"
            )
        targets.append(target)
    pruned = failed = 0
    for target in targets:
        if target.is_dir():
            try:
                with _checkpoint_prune_lock(target) as acquired:
                    if acquired and target.is_dir():
                        shutil.rmtree(target)
                        pruned += 1
            except FileNotFoundError:
                continue
            except OSError:
                failed += 1
                logging.getLogger(__name__).warning(
                    "Checkpoint retention failed: %s", target, exc_info=True
                )
    return {"pruned": pruned, "failed": failed}
