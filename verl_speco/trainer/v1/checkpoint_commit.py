# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Commit a V1 joint recovery point before exposing it or pruning its predecessor."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from omegaconf import OmegaConf, open_dict


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


@contextmanager
def defer_native_checkpoint_commit(config, *, use_critic=False):
    """Suppress only driver-side latest publication and premature retention.

    V1's native save checks the driver's async_save setting only when writing
    latest. Worker actors retain their already-serialized synchronous configs;
    no async-save argument is sent by that RPC. Reject genuinely async workers
    because their completion cannot be established by this transaction.
    """
    actor = config.actor_rollout_ref.actor
    if (actor.get("checkpoint") or {}).get("async_save", False):
        raise ValueError(
            "V1 Bubble joint checkpoint requires synchronous actor checkpoint saves"
        )
    if use_critic and (config.critic.get("checkpoint") or {}).get("async_save", False):
        raise ValueError(
            "V1 Bubble joint checkpoint requires synchronous critic checkpoint saves"
        )
    with ExitStack() as stack:
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


def commit_latest(root, step):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"global_step_{int(step)}"
    manifest = json.loads(
        (folder / "speco_v1_manifest.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "speco_v1_joint_checkpoint"
        or manifest.get("version") != 1
        or manifest.get("global_step") != int(step)
        or manifest.get("bubble", {}).get("quota_complete") is not True
    ):
        raise RuntimeError("Cannot publish an incomplete V1 Bubble recovery point")
    _fsync_directory(folder)
    fd, temporary = tempfile.mkstemp(prefix=".speco_latest.", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(int(step)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "latest_checkpointed_iteration.txt")
        _fsync_directory(root)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def committed_steps(root, through_step):
    steps = []
    for folder in Path(root).glob("global_step_*"):
        suffix = folder.name.removeprefix("global_step_")
        if not suffix.isdecimal() or int(suffix) > through_step:
            continue
        try:
            manifest = json.loads(
                (folder / "speco_v1_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        if manifest.get("format") == "speco_v1_joint_checkpoint" and manifest.get(
            "global_step"
        ) == int(suffix):
            steps.append(int(suffix))
    return sorted(steps)


def prune_checkpoint_roles(worker, root, steps, role):
    """Run on every owning worker, including nodes with private local storage."""
    del worker
    root = Path(root).resolve()
    if not role or Path(role).name != role or role in {".", ".."}:
        raise ValueError("Invalid checkpoint role")
    targets = []
    for step in steps:
        folder = (root / f"global_step_{int(step)}").resolve()
        target = (folder / role).resolve()
        if folder.parent != root or target.parent != folder or target.name != role:
            raise ValueError(
                "Checkpoint retention target escapes its recovery directory"
            )
        targets.append(target)
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
    return {"pruned": len(targets)}
