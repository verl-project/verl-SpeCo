# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Select exactly one online drafter objective, independently of its backend."""


def resolve_opd_method(training):
    """Keep PR #1 defaults and accept the former SpecOPD opt-in for compatibility.

    A disabled selector does not enable either method. The legacy SpecOPD
    flag remains supported, but cannot coexist with an enabled Draft-OPD arm.
    """
    selection = training.get("drafter_opd", {}) or {}
    method = selection.get("method", "draft_opd")
    if method not in {"draft_opd", "spec_opd"}:
        raise ValueError(f"unknown drafter_opd.method: {method!r}")
    enabled = selection.get("enable", False)
    legacy = (training.get("specopd", {}) or {}).get("enable", False)
    if type(enabled) is not bool or type(legacy) is not bool:
        raise ValueError("drafter OPD enable flags must be boolean")
    if legacy and enabled and method != "spec_opd":
        raise ValueError("Draft-OPD and legacy specopd.enable cannot both be enabled")
    if enabled:
        return method
    return "spec_opd" if legacy else "none"


def draft_opd_enabled(training):
    return resolve_opd_method(training) == "draft_opd"


def spec_opd_enabled(training):
    return resolve_opd_method(training) == "spec_opd"


def capture_verify_trace_enabled(training):
    # PR #1 also permits capture-only diagnostics. Its trace must not install
    # a second collection path when full-proposal SpecOPD is selected.
    return resolve_opd_method(training) != "spec_opd" and (
        training.get("drafter_opd", {}) or {}
    ).get("capture_verify_trace", False)
