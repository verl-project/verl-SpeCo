# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
import torch
from test_deepseek_v4_expert_weights import Draft, payload

from verl_speco.integration.deepseek_v4_update_probe import worker_update_probe
from verl_speco.probe_deepseek_v4_draft_update import require_phase


def checkpoint(tmp_path):
    (tmp_path / "parallel_topology.json").write_text(
        json.dumps(
            {
                "world_size": 4,
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "rank_order": "dp_major_tp_minor",
            }
        )
    )
    w13 = torch.arange(4 * 3 * 16).reshape(4, 3, 16).to(torch.bfloat16)
    w2 = torch.arange(4 * 8 * 3).reshape(4, 8, 3).to(torch.bfloat16)
    states = [{}, {}]
    for name, value in payload(2, w13, w2):
        rank = int(name.split("_tp_")[1].split("_")[0])
        states[rank][name.split("/", 1)[1]] = value
    for dp in range(2):
        for tp, state in enumerate(states):
            torch.save(state, tmp_path / f"model_states_rank_{dp * 2 + tp}.pt")
    return w13, w2


def test_prepare_commit_and_global_rollback_retains_old_weights(tmp_path):
    w13, w2 = checkpoint(tmp_path)
    model = Draft([1, -1, -1, 0])
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(drafter=SimpleNamespace(model=model))
    )

    def phase(name):
        return worker_update_probe(worker, name, str(tmp_path), tp_rank=0, tp_size=2)

    assert phase("prepare")["status"] == "prepared"
    assert torch.count_nonzero(model.experts.w13_weight) == 0
    assert phase("commit")["status"] == "committed"
    assert torch.equal(model.experts.w13_weight[1], w13[0])
    assert torch.equal(model.experts.w2_weight[0], w2[3])
    assert phase("rollback")["status"] == "rolled_back"
    assert torch.count_nonzero(model.experts.w13_weight) == 0
    assert phase("prepare")["status"] == "prepared"
    assert phase("commit")["status"] == "committed"
    assert phase("finalize")["status"] == "finalized"
    assert worker._speco_draft_update_probe is None


def test_bad_checkpoint_preparation_reports_failure_without_mutation(tmp_path):
    checkpoint(tmp_path)
    state = torch.load(tmp_path / "model_states_rank_1.pt", weights_only=True)
    state.pop("norm")
    torch.save(state, tmp_path / "model_states_rank_1.pt")
    model = Draft([0, -1, -1, 1])
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(drafter=SimpleNamespace(model=model))
    )
    result = worker_update_probe(worker, "prepare", str(tmp_path), tp_rank=0, tp_size=2)
    assert result["status"] == "failed"
    assert torch.count_nonzero(model.experts.w13_weight) == 0
    assert getattr(worker, "_speco_draft_update_probe", None) is None


def test_coordinator_requires_every_tp_dp_acknowledgement():
    results = [{"status": "prepared", "tp_rank": r} for _ in range(2) for r in range(2)]
    require_phase(results, "prepared", 2, 2)
    with pytest.raises(RuntimeError):
        require_phase(results[:-1], "prepared", 2, 2)
    results[-1]["tp_rank"] = 0
    with pytest.raises(RuntimeError, match="every TP/DP"):
        require_phase(results, "prepared", 2, 2)
