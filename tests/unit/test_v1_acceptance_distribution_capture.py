"""Execute deployed-method ASTs: existing JSONL pipeline and V1 integration."""
import ast
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest



def subject(values):
    import torch

    root = Path(__file__).resolve().parents[2]
    trainer = ast.parse((root / "verl_speco/trainer/speco_ray_trainer.py").read_text())
    mixin = ast.parse((root / "verl_speco/trainer/v1/speco_mixin.py").read_text())
    functions = {
        "_speco_optional_float", "_speco_accept_len_quantile",
        "_speco_accept_len_hist_bin_width", "_speco_accept_len_histogram",
        "_speco_request_accept_len_step_payload", "_speco_append_jsonl_batch",
    }
    methods = {
        "_speco_log_request_accept_len_variance_from_batch",
        "_speco_log_request_accept_lens", "_speco_flush_request_accept_len_payloads",
        "_speco_v1_log_request_accept_len_distribution",
    }
    namespace = dict(os=os, math=math, json=json, torch=torch, Any=Any, cast=cast,
                     logger=logging.getLogger(__name__), _speco_diag_enabled=lambda *a: False)
    for n in trainer.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_SPECO_REQUEST_"):
                    try:
                        namespace[target.id] = ast.literal_eval(n.value)
                    except ValueError:
                        pass
    nodes = [n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    selected = [n for tree in [trainer, mixin] for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert len(selected) == len(methods)
    nodes.append(ast.ClassDef(name="Subject", bases=[], keywords=[], decorator_list=[], body=selected))
    nodes.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "capture_contract", "exec"), namespace)
    obj = namespace["Subject"]()
    obj.global_steps = 10
    obj._speco_pending_request_accept_len_payloads = []
    obj._speco_last_request_accept_len_records = [{"original": True}]
    obj._speco_last_request_accept_len_var = 123.0
    obj._speco_v1_batch_data = lambda batch: (None, values)
    obj._speco_batch_size_from_request_stats = len
    obj._speco_request_accept_lengths = lambda batch, size: batch
    obj._speco_request_ids = lambda batch, size, **kw: list(range(size))
    obj._speco_request_completion_indices = lambda batch, size: list(range(size))
    obj._speco_request_elapsed_secs = lambda batch, size: [None] * size
    obj._speco_should_log_request_accept_len_variance = lambda: False
    return obj


@pytest.mark.parametrize("shape", ["bimodal", "single"])
def test_real_jsonl_round_trip(tmp_path, monkeypatch, shape):
    values = ([*np.linspace(1.9000000123, 2.1000000456, 96), *np.linspace(3.9, 4.1, 224)]
              if shape == "bimodal" else list(np.linspace(3.0, 4.0, 320)))
    path = tmp_path / "request-distributions.jsonl"
    monkeypatch.setenv("VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH", str(path))
    obj = subject(values)
    original_records = obj._speco_last_request_accept_len_records
    obj._speco_v1_log_request_accept_len_distribution(object())
    obj._speco_v1_log_request_accept_len_distribution(object())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["step"] == 10 and row["source"] == "v1_rollout" and row["count"] == 320
    assert row["accept_lens_full_precision"] == values
    assert len(row["accept_lens"]) == 320
    assert sum(bin_["count"] for bin_ in row["histogram"]["bins"]) == 320
    assert obj._speco_last_request_accept_len_records is original_records
    assert obj._speco_last_request_accept_len_var == 123.0
    assert not obj._speco_pending_request_accept_len_payloads
    obj.global_steps = 20
    obj._speco_v1_log_request_accept_len_distribution(object())
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [10, 20]


def test_disabled_capture_does_not_touch_batch(monkeypatch):
    monkeypatch.delenv("VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH", raising=False)
    obj = subject([3.0] * 320)
    obj._speco_v1_batch_data = lambda batch: pytest.fail("disabled capture extracted batch")
    obj._speco_v1_log_request_accept_len_distribution(object())
    assert not obj._speco_pending_request_accept_len_payloads


def test_no_requests_does_not_mark_step_written(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_SPECO_REQUEST_ACCEPT_LEN_HIST_LOG_PATH", str(tmp_path / "empty.jsonl"))
    obj = subject([])
    obj._speco_v1_log_request_accept_len_distribution(object())
    assert not hasattr(obj, "_speco_v1_accept_len_logged_step")


def test_v1_oldlogprob_entrypoint_calls_capture():
    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/v1/speco_mixin.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_old_log_prob")
    assert isinstance(method.body[0], ast.Expr)
    assert method.body[0].value.func.attr == "_speco_v1_log_request_accept_len_distribution"
