# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Ray worker adapter for the existing standalone drafter training loop."""

from __future__ import annotations

import logging
import os
from typing import Any

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register


class StandaloneDrafterWorker(Worker):
    """Expose standalone training through the standard verl WorkerGroup RPC.

    RayWorkerGroup supplies RANK/LOCAL_RANK/WORLD_SIZE and rendezvous variables.
    The called loop is intentionally the same loop used by ``torchrun``.
    """

    def __init__(
        self,
        config: Any,
        scheduled_commands: Any | None = None,
        training_events: Any | None = None,
    ):
        # draft_train.main() used to configure the root logger. This worker
        # enters the training loop directly, so restore the same INFO level
        # while retaining Ray's existing output handler and worker prefix.
        level_name = os.environ.get("SPECO_STANDALONE_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, None)
        if not isinstance(level, int):
            raise ValueError(
                "SPECO_STANDALONE_LOG_LEVEL must be a standard Python logging level"
            )
        logging.basicConfig(level=level)
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)
        Worker.__init__(self)
        self.config = config
        self.scheduled_commands = scheduled_commands
        self.training_events = training_events

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def run_standalone_training(self):
        from verl_speco.trainer.draft_training_loop import (
            _run_standalone_draft_training_async,
        )

        return await _run_standalone_draft_training_async(
            self.config,
            scheduled_commands=self.scheduled_commands,
            training_events=self.training_events,
        )


__all__ = ["StandaloneDrafterWorker"]
