# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Bound host serialization memory without serializing distributed collectives."""

import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def checkpoint_io_slot(base_dir):
    """Serialize local writes on the shared checkpoint filesystem (Linux/NPU)."""
    path = Path(base_dir) / ".speco_checkpoint_io.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def bounded_checkpoint_writes(model_engine, base_dir):
    """Wrap only synchronous Torch serialization; DS barriers remain outside."""
    checkpoint_engine = model_engine.checkpoint_engine
    if type(checkpoint_engine).__name__ != "TorchCheckpointEngine":
        raise RuntimeError("Bounded checkpoint writes require TorchCheckpointEngine")
    original = checkpoint_engine.save
    had_instance_save = "save" in vars(checkpoint_engine)
    instance_save = vars(checkpoint_engine).get("save")
    zipfile = checkpoint_engine.zipfile_serialization

    def save(*args, **kwargs):
        with checkpoint_io_slot(base_dir):
            return original(*args, **kwargs)

    checkpoint_engine.save = save
    checkpoint_engine.zipfile_serialization = True
    try:
        yield
    finally:
        checkpoint_engine.zipfile_serialization = zipfile
        if had_instance_save:
            checkpoint_engine.save = instance_save
        else:
            del checkpoint_engine.save
