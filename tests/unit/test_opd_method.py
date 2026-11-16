# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Mutually exclusive method selection and compatibility with both opt-ins."""

import pytest

from verl_speco.opd_method import (
    capture_verify_trace_enabled,
    draft_opd_enabled,
    resolve_opd_method,
    spec_opd_enabled,
)
from verl_speco.specopd_config import SpecOPDConfig


@pytest.mark.parametrize(
    "training,expected",
    [
        ({}, "none"),
        ({"drafter_opd": {"enable": True}}, "draft_opd"),
        ({"drafter_opd": {"enable": True, "method": "spec_opd"}}, "spec_opd"),
        ({"drafter_opd": {"enable": False, "method": "spec_opd"}}, "none"),
        ({"specopd": {"enable": True}}, "spec_opd"),
        (
            {
                "drafter_opd": {"enable": True, "method": "spec_opd"},
                "specopd": {"enable": True},
            },
            "spec_opd",
        ),
    ],
)
def test_one_method_selected(training, expected):
    assert resolve_opd_method(training) == expected
    assert draft_opd_enabled(training) == (expected == "draft_opd")
    assert spec_opd_enabled(training) == (expected == "spec_opd")
    assert SpecOPDConfig.from_training(training).enable == (expected == "spec_opd")


@pytest.mark.parametrize(
    "training",
    [
        {"drafter_opd": {"enable": True}, "specopd": {"enable": True}},
        {"drafter_opd": {"method": "typo"}},
        {"drafter_opd": {"enable": "true"}},
        {"specopd": {"enable": 1}},
    ],
)
def test_ambiguous_or_invalid_selector_fails(training):
    with pytest.raises(ValueError):
        SpecOPDConfig.from_training(training)


def test_specopd_ignores_baseline_trace_switch():
    training = {
        "drafter_opd": {
            "enable": True,
            "method": "spec_opd",
            "capture_verify_trace": True,
        }
    }
    assert not capture_verify_trace_enabled(training)
    training["drafter_opd"]["method"] = "draft_opd"
    assert capture_verify_trace_enabled(training)
