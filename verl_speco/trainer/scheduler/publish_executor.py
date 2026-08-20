# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Infrastructure boundary for publishing drafter weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class DrafterPublishExecutor(Protocol):
    def wait_pending(self) -> int: ...

    def fetch_snapshot(self) -> Any | None: ...

    def update_rollout_weights(
        self, payload: Any, *, global_step: object, asynchronous: bool
    ) -> None: ...


@dataclass
class CallbackDrafterPublishExecutor:
    """Adapt drafter/rollout RPCs without exposing them to the scheduler."""

    wait: Callable[[], int]
    fetch: Callable[[], Any | None]
    update: Callable[[Any, object, bool], Any]
    normalize_payload: Callable[[Any], Any]

    def wait_pending(self) -> int:
        return self.wait()

    def fetch_snapshot(self) -> Any | None:
        snapshot = self.fetch()
        return None if snapshot is None else self.normalize_payload(snapshot)

    def update_rollout_weights(
        self, payload: Any, *, global_step: object, asynchronous: bool
    ) -> None:
        self.update(payload, global_step, asynchronous)
