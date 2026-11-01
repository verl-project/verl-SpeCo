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
"""Bound producer reclaim and preserve serialized snapshots and failure cleanup."""

import ast
import logging
import pickle
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.integration import rollout_publish
from verl_speco.integration.publish_reclaim import PublishReclaimPolicy


def memory(rss=10 << 20, available=100 << 20):
    return {
        "rss_gib": rss,
        "available_gib": available,
        "node_unavailable_gib": 100 << 20,
    }


@pytest.mark.parametrize(
    "reason",
    [
        "released_bytes_limit",
        "release_count_limit",
        "rss_growth",
        "memory_pressure",
        "unknown_memory",
    ],
)
def test_reclaim_policy_forces_bounded_or_pressure_reclaim(reason):
    policy = PublishReclaimPolicy(max_bytes=100, max_releases=3, rss_growth_kib=100)
    assert policy.decide(10, memory()) == "deferred"
    if reason == "released_bytes_limit":
        result = policy.decide(90, memory())
    elif reason == "release_count_limit":
        assert policy.decide(10, memory()) == "deferred"
        result = policy.decide(10, memory())
    elif reason == "rss_growth":
        result = policy.decide(10, memory(rss=(10 << 20) + 100))
    elif reason == "memory_pressure":
        result = policy.decide(10, memory(), cgroup_pressure=True)
    else:
        result = policy.decide(10, {})
    assert result == reason
    policy.reset()
    assert policy.pending_bytes == policy.releases == 0


def test_payload_always_cleared_and_exit_reclaim_registered(monkeypatch):
    from verl_speco.integration import publish_reclaim
    from verl_speco.trainer import checkpoint

    calls, exit_handlers = [], []
    monkeypatch.setattr(rollout_publish, "_publish_reclaim_policy", None)
    monkeypatch.setattr(rollout_publish.atexit, "register", exit_handlers.append)
    monkeypatch.setattr(checkpoint, "collect_checkpoint_memory_snapshot", memory)
    monkeypatch.setattr(publish_reclaim, "cgroup_memory_pressure", lambda: False)
    monkeypatch.setattr(
        rollout_publish, "_release_publish_host_memory", lambda: calls.append(1) or {}
    )
    for index in range(3):
        payload = {"weight": object()}
        result = rollout_publish.release_draft_weights_payload(
            payload, defer_host_reclaim=True, released_bytes=3 << 30
        )
        assert payload == {} and result["payload_cleared"] == 1
        assert result["reclaim_reason"] == (
            "deferred" if index < 2 else "released_bytes_limit"
        )
    assert len(calls) == 1 and len(exit_handlers) == 1
    # Existing unconditional paths (including failures) still force reclamation.
    rollout_publish.release_draft_weights_payload({"w": object()})
    assert len(calls) == 2
    exit_handlers[0]()
    assert len(calls) == 3


@pytest.mark.parametrize(
    "device,strategy,fail,bubble",
    [
        ("cuda", "sync", False, False),
        ("cuda", "sync", False, True),
        ("cuda", "sync", True, False),
        ("cuda", "rollout_idle_worker", False, True),
        ("npu", "sync", False, False),
    ],
)
def test_producer_drops_wire_before_clear_and_preserves_ray_snapshot(
    monkeypatch, device, strategy, fail, bubble
):
    torch = pytest.importorskip("torch")
    path = Path(__file__).parents[2] / "verl_speco/workers/speco_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "maybe_publish"
    )
    method.decorator_list = []
    payload = {"w": torch.arange(8, dtype=torch.float32)}
    wire_ref, release_options, serialized = [], [], []

    def put(value):
        if "_speco_cpu_tensor_wire_v1" in value:
            wire_ref.append(
                weakref.ref(value["_speco_cpu_tensor_wire_v1"]["w"]["data"])
            )
        if fail:
            raise RuntimeError("put failed")
        serialized.append(pickle.loads(pickle.dumps(value)))
        return "independent-object-ref"

    def release(weights, **options):
        if wire_ref and not fail:
            assert wire_ref[0]() is None
        release_options.append(options)
        weights.clear()
        return {"num_weights": 1, "payload_cleared": 1}

    namespace = {
        "time": time,
        "ray": SimpleNamespace(put=put),
        "logger": logging.getLogger(__name__),
        "release_draft_weights_payload": release,
    }
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - production method
    worker = SimpleNamespace(
        enable_drafter=True,
        in_drafter_train_group=True,
        last_global_step=2,
        last_trained_step=2,
        trainer=SimpleNamespace(
            pop_model_state_dict_for_publish=lambda step: (True, payload),
            _bubble_time_enabled=bubble,
        ),
        _deferred_publish_snapshot_step=None,
        is_global_publish_leader=True,
        is_drafter_group_leader=True,
        rank=0,
        replica_rank=0,
        _last_trained_execution_strategy=strategy,
        _last_trained_target_worker_ids=[],
        device_name=device,
        config={
            "rollout": {"drafter": {"training": {"gpu_transfer_optimizations": True}}}
        },
    )
    if fail:
        with pytest.raises(RuntimeError, match="put failed"):
            namespace["maybe_publish"](worker)
    else:
        assert namespace["maybe_publish"](worker) == {
            "weights_ref": "independent-object-ref"
        }
        from verl_speco.integration.tensor_wire import decode_cpu_tensors

        torch.testing.assert_close(
            decode_cpu_tensors(serialized[0])["w"], torch.arange(8, dtype=torch.float32)
        )
    assert payload == {}
    assert bool(release_options[0].get("defer_host_reclaim")) == (
        device == "cuda" and strategy == "sync" and not fail and not bubble
    )


@pytest.mark.parametrize(
    "used,limit,pressure", [(91, "100", True), (50, "100", False), (999, "max", False)]
)
def test_cgroup_pressure_respects_container_limit(monkeypatch, used, limit, pressure):
    from verl_speco.integration import publish_reclaim

    def read(path):
        return str(used) if path.name == "memory.current" else limit

    monkeypatch.setattr(publish_reclaim.Path, "read_text", read)
    assert publish_reclaim.cgroup_memory_pressure() is pressure
