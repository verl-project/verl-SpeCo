# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""One namespace for optional owner-local diagnostic counters."""

from types import SimpleNamespace


def diagnostic_state(owner):
    state = getattr(owner, "_speco_diagnostics", None)
    if state is None:
        state = SimpleNamespace()
        owner._speco_diagnostics = state
    return state
