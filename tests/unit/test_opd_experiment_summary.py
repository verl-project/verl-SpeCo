# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Check completeness rules for the checked-in experiment audit script."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/experiments/opd-cotrain/20261013_005359/summarize_results.py"
)
spec = importlib.util.spec_from_file_location("opd_experiment_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def _log(*, fixed=False):
    metrics = [m for m in summary.METRICS if not (fixed and m == "timing_s/drafter")]
    return "\n".join(
        f"step:{step} - " + " ".join(f"{metric}:{step}.0" for metric in metrics)
        for step in (1, 2)
    ).encode()


def test_summary_means_and_missing_baseline_timer():
    values = summary.summarize(_log(fixed=True), 2, fixed=True)
    assert values["critic/rewards/mean"] == {"mean": 1.5, "count": 2}
    assert values["timing_s/drafter"] == {"mean": None, "count": 0}


def test_summary_requires_every_step():
    with pytest.raises(ValueError, match="every training step"):
        summary.summarize(_log().splitlines()[0], 2, fixed=False)


def test_summary_requires_cotrain_timer():
    with pytest.raises(ValueError, match="Incomplete metric timing_s/drafter"):
        summary.summarize(_log(fixed=True), 2, fixed=False)


def test_summary_rejects_partial_required_metrics():
    raw = _log().replace(b"critic/rewards/mean:2.0", b"unrelated:2.0")
    with pytest.raises(ValueError, match="Incomplete metric critic/rewards/mean"):
        summary.summarize(raw, 2, fixed=False)
