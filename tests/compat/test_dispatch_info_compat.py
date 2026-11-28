# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cover legacy and structured verl worker routing metadata."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_nodes(path, names, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    if class_name:
        nodes = [
            ast.ClassDef(
                name="Subject", bases=[], keywords=[], body=nodes, decorator_list=[]
            )
        ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {
        "_DRAFTER_TARGET_SYNC_MESH": "target",
        "os": SimpleNamespace(cpu_count=lambda: 2),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("structured", [False, True])
def test_owner_and_target_sync_routing(structured):
    ranks = [0, 0, 1, 1, 2, 2]
    mask = [True, False, True, False, False, False]
    dispatch = (
        [SimpleNamespace(dp_rank=n, is_mp_src=i % 2 == 0) for i, n in enumerate(ranks)]
        if structured
        else ranks
    )
    collect = [SimpleNamespace(is_collect=v) for v in mask] if structured else mask
    names = {
        "_speco_owner_route_mapping",
        "_speco_owner_route_collect_mask",
        "_speco_dispatch_bucket_count",
        "_speco_owner_bucket_count",
        "_speco_build_drafter_target_lm_head_sync_args",
    }
    cls = load_nodes(
        "verl_speco/trainer/speco_ray_trainer.py", names, "SpecoRayPPOTrainer"
    )["Subject"]
    trainer = cls()
    trainer.global_steps = 5
    trainer.drafter_wg = SimpleNamespace(
        _dispatch_info={},
        _collect_info={},
        _query_dispatch_info=lambda _: dispatch,
        _query_collect_info=lambda _: collect,
    )
    assert trainer._speco_dispatch_bucket_count() == 3
    assert trainer._speco_owner_bucket_count() == 2
    assert trainer._speco_owner_route_collect_mask() == mask
    payload = {"weight": "tensor"}
    assert trainer._speco_build_drafter_target_lm_head_sync_args(payload) == (
        [payload] * 3,
        [5] * 3,
        3,
    )
    assert trainer.drafter_wg._dispatch_info["drafter_owner_route"] is dispatch
    assert trainer.drafter_wg._collect_info["drafter_owner_route"] is collect


@pytest.mark.parametrize("structured", [False, True])
def test_worker_dispatch_and_collect(structured):
    class WorkerGroup:
        world_size = 4

    group = WorkerGroup()
    ranks = [0, 0, 1, 1]
    mask = [True, False, True, False]
    dispatch = (
        [SimpleNamespace(dp_rank=n, is_mp_src=i % 2 == 0) for i, n in enumerate(ranks)]
        if structured
        else ranks
    )
    collect = [SimpleNamespace(is_collect=v) for v in mask] if structured else mask
    group._dispatch_info = {}
    group._collect_info = {}
    group._query_dispatch_info = lambda _: dispatch
    group._query_collect_info = lambda _: collect
    namespace = load_nodes(
        "verl_speco/workers/speco_worker.py",
        {
            "_dispatch_nd_compute",
            "_collect_nd_compute",
            "_dispatch_lazy_compute",
            "_collect_lazy_compute",
        },
    )
    worker_module = types.ModuleType("verl.single_controller.base.worker_group")
    worker_module.WorkerGroup = WorkerGroup
    ray_module = types.ModuleType("verl.utils.ray_utils")
    ray_module.parallel_put = lambda value, max_workers: value
    with patch.dict(
        sys.modules,
        {worker_module.__name__: worker_module, ray_module.__name__: ray_module},
    ):
        args, kwargs = namespace["_dispatch_lazy_compute"](
            "route", group, [10, 20], token=["a", "b"]
        )
        assert args == ([10, 10, 20, 20],)
        assert kwargs == {"token": ["a", "a", "b", "b"]}
        assert namespace["_collect_lazy_compute"]("route", group, [11, 12, 21, 22]) == [
            11,
            21,
        ]
    assert group._dispatch_info["route"] is dispatch
    assert group._collect_info["route"] is collect
