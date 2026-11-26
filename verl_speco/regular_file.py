# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Descriptor-based access to regular files in trusted directory trees."""

import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path


@contextmanager
def regular_parent(path, *, root=None, identity=None):
    path = Path(os.path.abspath(path))
    with ExitStack() as stack:
        if os.open in os.supports_dir_fd:
            fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, fd)
            current = Path(path.anchor)
            for part in path.parent.parts[1:]:
                current /= part
                fd = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                stack.callback(os.close, fd)
                if root is not None and current == root:
                    info = os.fstat(fd)
                    if (info.st_dev, info.st_ino) != identity:
                        raise ValueError("Configured shared file root changed")
            yield path, fd
        else:
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError("Regular file paths must not contain symlinks")
            if root is not None:
                info = root.stat(follow_symlinks=False)
                if (info.st_dev, info.st_ino) != identity:
                    raise ValueError("Configured shared file root changed")
            yield path, None


@contextmanager
def regular_file(path, *, root=None, identity=None):
    with regular_parent(path, root=root, identity=identity) as (path, parent):
        kwargs = {"dir_fd": parent} if parent is not None else {}
        before = os.stat(
            path.name if parent is not None else path, follow_symlinks=False, **kwargs
        )
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Expected a regular file")
        fd = os.open(
            path.name if parent is not None else path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            **kwargs,
        )
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise ValueError("Regular file identity changed or is a symlink")
            yield stream


@contextmanager
def regular_path(path, *, root=None, identity=None):
    with regular_file(path, root=root, identity=identity) as stream:
        if os.name == "posix" and Path("/proc/self/fd").is_dir():
            yield Path(f"/proc/self/fd/{stream.fileno()}")
        else:
            yield Path(path)
            current = Path(path).stat(follow_symlinks=False)
            original = os.fstat(stream.fileno())
            if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                raise ValueError("Regular file identity changed during reading")


def unlink_regular(path, *, root=None, identity=None):
    with regular_parent(path, root=root, identity=identity) as (path, parent):
        kwargs = {"dir_fd": parent} if parent is not None else {}
        try:
            info = os.stat(
                path.name if parent is not None else path,
                follow_symlinks=False,
                **kwargs,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Refusing to remove a non-regular file")
        os.unlink(path.name if parent is not None else path, **kwargs)
