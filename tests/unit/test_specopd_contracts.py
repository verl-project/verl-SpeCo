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
import json
from types import SimpleNamespace

import pytest
import torch

from verl_speco.specopd_config import SpecOPDConfig, validate_specopd_config
from verl_speco.transport.specopd_protocol import (
    ProposalRecord,
    record_dict,
    encode_bundle,
    decode_bundle,
    token_digest,
)
from verl_speco.integration.specopd_window import make_window
from verl_speco.integration.specopd_vllm import (
    snapshot_scheduler_round,
    capture_scheduler_round,
)
from verl_speco.integration.specopd_driver import build_requests
from verl_speco.integration.specopd_driver import plan_microbatches, unwrap_rank_result
from verl_speco.joint_actor import partition_joint_requests
from verl_speco.joint_scoring import JointScoringRequest, DraftBranch
from verl_speco.trainer.scheduler.specopd_cycle import SpecOPDCycle
from verl_speco.backends.specopd_loss import e2e_tv_loss


def config():
    return {
        "trainer": {"use_v1": False},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp2"},
            "model": {"use_remove_padding": False},
            "rollout": {
                "name": "vllm",
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "speculative_algorithm": "DFLASH",
                    "vllm": {"draft_sample_method": "probabilistic"},
                    "training": {
                        "dflash_block_size": 8,
                        "step": 1,
                        "collect_interval_steps": 1,
                        "training_interval_steps": 1,
                        "specopd": {"enable": True},
                    },
                },
            },
        },
    }


def test_configuration_requires_exact_block_and_sampling():
    assert validate_specopd_config(config()).enable
    cfg = config()
    cfg["actor_rollout_ref"]["rollout"]["drafter"]["training"]["dflash_block_size"] = 16
    with pytest.raises(ValueError, match="block size"):
        validate_specopd_config(cfg)
    cfg = config()
    cfg["actor_rollout_ref"]["rollout"]["drafter"]["vllm"]["draft_sample_method"] = (
        "greedy"
    )
    with pytest.raises(ValueError, match="probabilistic"):
        validate_specopd_config(cfg)


def test_config_unknown_and_non_joint_modes_rejected():
    with pytest.raises(ValueError, match="unknown"):
        SpecOPDConfig.from_training({"specopd": {"enable": True, "typo": 1}})
    with pytest.raises(ValueError, match="contract"):
        SpecOPDConfig.from_training(
            {"specopd": {"enable": True, "capture_rejected_tail": False}}
        )


def test_window_is_preplanned_deterministic_and_uses_response_offsets():
    spec = SpecOPDConfig(window_mode="random", window_response_tokens=32)
    a = make_window(spec, "r", 2, 3, 100, 1000)
    assert a == make_window(spec, "r", 2, 3, 100, 1000)
    assert 0 <= a["start"] <= 512 and a["end"] - a["start"] == 32
    assert make_window(SpecOPDConfig(), "r", 2, 3, 100, 1000)["start"] == 0


def capture(raw=(11, 12, 13, 14, 15, 16, 17), generated=(11, 20), offset=0):
    plan = make_window(SpecOPDConfig(), "r", 2, 3, 3, 1000)
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args={"specopd_plan": plan}),
        all_token_ids=[1, 2, 3] + [4] * offset,
        spec_token_ids=list(raw),
        num_output_placeholders=0,
    )
    scheduler = SimpleNamespace(requests={"r": request})
    snapshot = snapshot_scheduler_round(scheduler)
    request.spec_token_ids.clear()  # Real Scheduler.schedule consumes this list.
    output = SimpleNamespace(scheduled_spec_decode_tokens={"r": list(raw)[:2]})
    result = SimpleNamespace(
        req_id_to_index={"r": 0}, sampled_token_ids=[list(generated)]
    )
    capture_scheduler_round(output, result, snapshot)
    return request, snapshot, output, result


def test_full_rejected_tail_survives_scheduler_shortening_and_mutation():
    request, _, _, _ = capture()
    record = request._specopd_records[0]
    assert record["proposal_ids"] == (11, 12, 13, 14, 15, 16, 17)
    assert record["accepted_count"] == 1
    assert record["prefix_digest"] == token_digest([1, 2, 3])


@pytest.mark.parametrize(
    "generated,accepted",
    [((20,), 0), ((11, 12, 20), 2), ((11, 12, 13, 14, 15, 16, 17, 20), 7)],
)
def test_acceptance_does_not_control_saved_tail(generated, accepted):
    request, _, _, _ = capture(generated=generated)
    assert len(request._specopd_records[0]["proposal_ids"]) == 7
    assert request._specopd_records[0]["accepted_count"] == accepted


def test_window_boundary_keeps_whole_proposal():
    request, _, _, _ = capture(offset=511)
    assert len(request._specopd_records[0]["proposal_ids"]) == 7
    request, _, _, _ = capture(offset=512)
    assert not request._specopd_records


def test_async_placeholder_rejected():
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args={"specopd_plan": {"prompt_len": 1}}),
        all_token_ids=[1],
        spec_token_ids=[2],
        num_output_placeholders=1,
    )
    with pytest.raises(RuntimeError, match="placeholders"):
        snapshot_scheduler_round(SimpleNamespace(requests={"r": request}))


def test_first_eos_keeps_rejected_prefix_and_masks_only_later_tokens():
    request, snapshot, scheduled, result = capture()
    request._specopd_records.clear()
    snapshot["r"][1]["eos_token_ids"] = [13]
    capture_scheduler_round(scheduled, result, snapshot)
    assert request._specopd_records[0]["proposal_ids"] == (11, 12, 13)


@pytest.mark.parametrize(
    "key,value",
    [
        ("actor_revision", -1),
        ("drafter_revision", True),
        ("proposal_temperature", float("nan")),
        ("request_id", ""),
    ],
)
def test_wire_rejects_invalid_identity_version_and_sampler(key, value):
    source = bundle()
    source[key] = value
    with pytest.raises(ValueError):
        decode_bundle(json.dumps(source))


def bundle():
    record = ProposalRecord("r:0", 3, (11, 12), token_digest((1, 2, 3)), 0, 0)
    return {
        "schema": 1,
        "request_id": "r",
        "actor_revision": 2,
        "drafter_revision": 3,
        "proposal_temperature": 1.0,
        "records": [record_dict(record)],
    }


def test_wire_bounds_duplicates_and_prefix_validation():
    source = bundle()
    wire = encode_bundle(source, 10000)
    assert decode_bundle(wire)["records"][0]["proposal_ids"] == [11, 12]
    with pytest.raises(ValueError, match="wire"):
        encode_bundle(source, 5)
    source["records"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        decode_bundle(json.dumps(source))


def test_batch_restores_real_unpadded_trunk_and_keeps_rejected_tokens():
    data = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[0, 1, 2, 3]]),
            "responses": torch.tensor([[4, 5, 0]]),
            "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0]]),
        },
        non_tensor_batch={"_specopd_proposals": [encode_bundle(bundle(), 10000)]},
    )
    requests, version = build_requests(data, SpecOPDConfig(), 3)
    assert requests[0].trunk_ids == (1, 2, 3, 4, 5)
    assert requests[0].response_start == 3
    assert requests[0].branches[0].token_ids == (11, 12)
    assert version == 2
    data.batch["input_ids"][0, 2] = 20
    with pytest.raises(ValueError, match="prefix"):
        build_requests(data, SpecOPDConfig(), 3)


def test_cycle_consumes_only_once_and_waits_for_publication():
    cycle = SpecOPDCycle(actor_revision=2, drafter_revision=3)
    cycle.begin(10)
    cycle.scored("data", 2, 3)
    assert cycle.begin_update() == "data"
    with pytest.raises(RuntimeError):
        cycle.begin_update()
    cycle.updated(True)
    with pytest.raises(RuntimeError):
        cycle.begin(11)
    cycle.published()
    assert cycle.drafter_revision == 4 and cycle.state == "ready"


def test_e2e_tv_hand_computed_loss_and_differentiable_prefixes():
    # overlap=(0.8, 0.6); survival=(0.8, 0.48); mean loss=0.36.
    q = (
        torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float64)
        .log()
        .requires_grad_()
    )
    p = torch.tensor([[[0.7, 0.3], [0.9, 0.1]]], dtype=torch.float64).log()
    mask = torch.ones((1, 2), dtype=torch.bool)
    numerator, count, info = e2e_tv_loss(q, p, mask)
    torch.testing.assert_close(
        numerator / count, torch.tensor(0.36), atol=1e-6, rtol=1e-6
    )
    numerator.backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    assert not p.requires_grad
    assert info["survival"][0, 1] < info["survival"][0, 0]


def test_candidate_support_keeps_teacher_mass_outside_support():
    q = torch.tensor([[[0.5, 0.5]]]).log().requires_grad_()
    p = torch.tensor([[[0.4, 0.4, 0.2]]]).log()
    loss, count, _ = e2e_tv_loss(
        q, p, torch.tensor([[True]]), candidate_ids=torch.tensor([[[0, 1]]])
    )
    torch.testing.assert_close(loss / count, torch.tensor(0.2))


def test_loss_padding_is_neutral_and_holes_rejected():
    q = torch.randn(2, 3, 5, requires_grad=True)
    p = torch.randn_like(q)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    numerator, count, _ = e2e_tv_loss(q, p, mask)
    numerator.backward()
    assert count == 4 and q.grad[1, 1:].abs().sum() == 0
    with pytest.raises(ValueError, match="prefix"):
        e2e_tv_loss(q, p, torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.bool))


def test_microbatch_planning_keeps_each_tree_once():
    trees = [
        JointScoringRequest(str(i), (1, 2, 3), 2, (DraftBranch(str(i), 1, (4, 5, 6)),))
        for i in range(3)
    ]
    groups = plan_microbatches(
        trees, SpecOPDConfig(microbatch_trajectories=3, scoring_token_budget=10)
    )
    assert [len(g) for g in groups] == [2, 1]
    assert [tree.sample_id for group in groups for tree in group] == ["0", "1", "2"]
    with pytest.raises(ValueError, match="complete trajectory"):
        plan_microbatches(trees, SpecOPDConfig(scoring_token_budget=4))


def test_distributed_dispatch_partitions_whole_trees_and_pads_empty_rank():
    trees = [
        JointScoringRequest(str(i), (1, 2, 3), 2, (DraftBranch(str(i), 1, (4, 5)),))
        for i in range(3)
    ]
    first, dummy = partition_joint_requests(trees, 0, 2)
    assert not dummy and [t.sample_id for t in first] == ["0", "2"]
    assert first[0].branches == trees[0].branches
    padded, dummy = partition_joint_requests(trees, 3, 4)
    assert dummy and len(padded[0].trunk_ids) == 1 and not padded[0].branches


def test_distributed_rank_outputs_are_merged_without_counting_dummy_rows_as_real():
    a, b = SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")
    results, metrics = unwrap_rank_result(
        [
            ((a,), {"model_forward_calls": 1, "logical_rows": 7}),
            ((b,), {"model_forward_calls": 1, "logical_rows": 9}),
        ]
    )
    assert (
        results == (a, b)
        and metrics["logical_rows"] == 16
        and metrics["model_forward_calls"] == 1
    )
