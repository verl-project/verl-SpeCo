# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_speco.trainer.tq_sample_source import TQLocalBatch


@pytest.mark.parametrize("valid,remote_error", [(True, False), (False, False), (True, True)])
def test_file_prepared_validation_shares_fetch_collective(valid, remote_error):
    path = Path(__file__).resolve().parents[2] / "verl_speco/trainer/draft_training_loop.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_next_batch_across_ranks")
    calls = []

    def reduce(state, **kwargs):
        calls.append(state.clone())
        if remote_error:
            state[0] = 1

    namespace = {
        "torch": torch, "TQLocalBatch": TQLocalBatch, "logger": logging.getLogger(__name__),
        "dist": SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 2,
                                all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="max")),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)} if valid else None)
    operation = namespace[function.name]
    if valid and not remote_error:
        assert operation(iter([batch]), rank=0, device="cpu", require_prepared=True) is batch
    else:
        with pytest.raises(RuntimeError, match="failed"):
            operation(iter([batch]), rank=0, device="cpu", require_prepared=True)
    assert len(calls) == 1
    assert calls[0].tolist() == [int(not valid), 0]
