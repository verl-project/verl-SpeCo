# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""A local, single-output route for Bubble driver lifecycle diagnostics."""

import logging


def configure_bubble_logger(logger: logging.Logger) -> None:
    """Bypass inherited handler filtering without changing the root logger.

    Called only when Bubble is enabled, before its event thread starts. Keep
    explicit INFO/DEBUG levels, but make lifecycle warnings visible even when
    a parent logger is disabled or configured above WARNING by the launcher.
    """
    # An inherited INFO root is not an explicit request for verbose Bubble
    # records. Default to WARNING so repairing the route cannot flood polling.
    level = logger.level
    logger.disabled = False
    logger.setLevel(level if 0 < level < logging.WARNING else logging.WARNING)
    if not any(
        getattr(handler, "_speco_bubble_handler", False) for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        setattr(handler, "_speco_bubble_handler", True)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        logger.addHandler(handler)
    # This dedicated route must not also print the same record at the root.
    logger.propagate = False
