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
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_speco.backends.specopd_loss import specopd_distribution_loss
from verl_speco.integration.specopd_anchors import add_trunk_anchors
from verl_speco.integration.specopd_driver import (
    build_requests,
)
from verl_speco.integration.specopd_vllm import (
    capture_scheduler_round,
    snapshot_scheduler_round,
)
from verl_speco.integration.specopd_window import make_window
from verl_speco.joint_scoring import DraftBranch, JointScoringRequest, parallel_microbatch_sizes
from verl_speco.specopd_config import (
    SpecOPDConfig,
    specopd_loss_weights,
    validate_specopd_config,
)
from verl_speco.integration.specopd_protocol import (
    ProposalRecord,
    encode_bundle,
    record_dict,
    token_digest,
)


def config():
    return {
        "trainer": {"use_v1": True},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp2"},
            "model": {"use_remove_padding": False},
            "rollout": {
                "name": "vllm",
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "speculative_algorithm": "DSPARK",
                    "vllm": {"draft_sample_method": "greedy"},
                    "training": {
                        "dspark_block_size": 7,
                        "step": 1,
                        "collect_interval_steps": 1,
                        "training_interval_steps": 1,
                        "specopd": {"enable": True},
                    },
                },
            },
        },
    }


@pytest.mark.parametrize("prompt,response", [(512, 4096), (512, 8192)])
def test_default_budgets_allow_long_trajectory(prompt, response):
    cfg = config()
    cfg["data"] = {"max_prompt_length": prompt, "max_response_length": response}
    spec = validate_specopd_config(cfg)
    branches = tuple(
        DraftBranch(str(i), prompt - 1 + i, (1,) * spec.proposal_width)
        for i in range(spec.max_branches_per_request)
    )
    request = JointScoringRequest("long", (1,) * (prompt + response), prompt, branches)
    assert parallel_microbatch_sizes([request], (spec.microbatch_trajectories,), 1, 1,
        dict(max_packed_tokens=spec.scoring_token_budget, max_mask_elements=spec.max_dense_mask_elements)) == (1,)
    training = cfg["actor_rollout_ref"]["rollout"]["drafter"]["training"]
    training["specopd"].update(scoring_token_budget=4096, max_dense_mask_elements=4096**2)
    with pytest.raises(ValueError, match=r"scoring_token_budget>=\d+ and max_dense_mask_elements>=\d+"):
        validate_specopd_config(cfg)


def test_specopd_inherits_width_temperature_and_mixed_anchors_from_drafter():
    cfg = config()
    rollout = cfg["actor_rollout_ref"]["rollout"]
    rollout["temperature"] = 0.6
    drafter = rollout["drafter"]
    drafter["speculative_algorithm"] = "DSPARK"
    drafter["rollout"] = {"spec_verify_tokens": 5}
    training = drafter["training"]
    training.update(dspark_block_size=5, dspark_num_anchors=64, dspark_loss_decay_gamma=0)
    training["specopd"]["anchor_count"] = None
    spec = validate_specopd_config(cfg)
    assert spec == SpecOPDConfig.from_drafter(drafter, rollout_temperature=0.6)
    assert spec.proposal_width == 5 and spec.max_proposal_tokens_per_request == 48 * 5
    assert spec.temperature == 0.6
    training["specopd"].update(
        actor_temperature=0.6, proposal_temperature=0.6, target_temperature=0.6,
        optimizer_updates_per_collection=training["step"],
    )
    assert validate_specopd_config(cfg) == spec
    assert (spec.anchor_count, spec.random_anchor_count, spec.random_position_decay_gamma) == (64, 16, 0)

    training["specopd"].update(anchor_count=0, random_position_decay_gamma=3)
    legacy = validate_specopd_config(cfg)
    assert legacy.anchor_count == 0 and legacy.max_proposal_tokens_per_request == 8 * 5
    assert legacy.random_position_decay_gamma == 3
    training["specopd"]["proposal_temperature"] = 1.0
    with pytest.raises(ValueError, match="must match rollout.temperature"):
        validate_specopd_config(cfg)


def test_v1_sync_config_requires_one_publication_per_group():
    cfg = config()
    cfg["trainer"]["use_v1"] = False
    with pytest.raises(ValueError, match="trainer.use_v1=true"):
        validate_specopd_config(cfg)
    cfg["trainer"] = {"use_v1": True, "v1": {"trainer_mode": "sync"}}
    assert validate_specopd_config(cfg).enable
    cfg["trainer"]["v1"]["trainer_mode"] = "colocate_async"
    with pytest.raises(ValueError, match="trainer_mode=sync"):
        validate_specopd_config(cfg)
    cfg["trainer"]["v1"] = {"trainer_mode": "sync", "separate_async": {"parameter_sync_step": 2}}
    with pytest.raises(ValueError, match="parameter_sync_step=1"):
        validate_specopd_config(cfg)
    cfg["trainer"]["v1"]["separate_async"]["parameter_sync_step"] = 1
    cfg["algorithm"] = {"filter_groups": {"enable": True}}
    with pytest.raises(ValueError, match="group-filter"):
        validate_specopd_config(cfg)


@pytest.mark.parametrize("weight", ["vat", " VAT "])
def test_specopd_rejects_vat_without_restricting_baseline(weight):
    cfg = config()
    training = cfg["actor_rollout_ref"]["rollout"]["drafter"]["training"]
    training["dspark_per_position_loss_weight"] = weight
    with pytest.raises(ValueError, match="SpecOPD.*VAT"):
        validate_specopd_config(cfg)
    training["specopd"]["enable"] = False
    assert not validate_specopd_config(cfg).enable


def capture(raw=(11, 12, 13, 14, 15, 16, 17), generated=(11, 20), offset=0, verified=2):
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
    output = SimpleNamespace(scheduled_spec_decode_tokens={"r": list(raw)[:verified]})
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
    "ignore_eos,effective_eos,explicit,expected",
    [(False, [13], [], 13), (True, [13], [], 17), (True, [13, 15], [15], 15),
     (False, [], [], 14)],
)
def test_first_effective_stop_preserves_tail_through_stop_token(
    ignore_eos, effective_eos, explicit, expected,
):
    request, snapshot, scheduled, result = capture()
    request._specopd_records.clear()
    request.sampling_params.all_stop_token_ids = effective_eos
    request.sampling_params.stop_token_ids = explicit
    request.sampling_params.ignore_eos = ignore_eos
    request.eos_token_id = 14
    capture_scheduler_round(scheduled, result, snapshot)
    assert request._specopd_records[0]["proposal_ids"] == tuple(range(11, expected + 1))


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


def test_e2e_tv_hand_computed_loss_and_differentiable_prefixes():
    # overlap=(0.8, 0.6); survival=(0.8, 0.48); mean loss=0.36.
    q = (
        torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float64)
        .log()
        .requires_grad_()
    )
    p = torch.tensor([[[0.7, 0.3], [0.9, 0.1]]], dtype=torch.float64).log()
    mask = torch.ones((1, 2), dtype=torch.bool)
    numerator, count, info = specopd_distribution_loss(q, p, mask)
    torch.testing.assert_close(
        numerator / count, torch.tensor(0.36), atol=1e-6, rtol=1e-6
    )
    numerator.backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    assert not p.requires_grad
    assert info["survival"][0, 1] < info["survival"][0, 0]


def test_specopd_l1_is_positionwise_on_the_same_histories():
    q = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]).log().requires_grad_()
    p = torch.tensor([[[0.7, 0.3], [0.9, 0.1]]]).log()
    valid = torch.ones((1, 2), dtype=torch.bool)
    numerator, count, _ = specopd_distribution_loss(q, p, valid, use_l1=True)
    torch.testing.assert_close(numerator / count, torch.tensor(0.6))
    numerator.backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0




def test_microbatch_planning_keeps_each_tree_once():
    trees = [
        JointScoringRequest(str(i), (1, 2, 3), 2, (DraftBranch(str(i), 1, (4, 5, 6)),))
        for i in range(3)
    ]
    def plan(spec, sp=1):
        return parallel_microbatch_sizes(trees, (spec.microbatch_trajectories,), 1, sp,
            dict(max_packed_tokens=spec.scoring_token_budget, max_mask_elements=spec.max_dense_mask_elements))
    groups = plan(SpecOPDConfig(microbatch_trajectories=3, scoring_token_budget=10))
    assert groups == (2, 1) and sum(groups) == len(trees)
    with pytest.raises(ValueError, match="complete trajectory"):
        plan(SpecOPDConfig(scoring_token_budget=4))
    with pytest.raises(ValueError, match="complete trajectory"):
        plan(SpecOPDConfig(max_dense_mask_elements=63), 4)
    assert plan(SpecOPDConfig(microbatch_trajectories=3, max_dense_mask_elements=128), 4) == (2, 1)


def _mixed_anchor_config(anchor_count=32, **kw):
    proposal_count = anchor_count * 3 // 4
    return SpecOPDConfig(anchor_count=anchor_count, random_anchor_count=anchor_count // 4,
                         max_branches_per_request=proposal_count,
                         max_proposal_tokens_per_request=proposal_count * 7, **kw)


def _mixed_anchor_bundle():
    return dict(request_id="request", actor_revision=5, drafter_revision=0, window=[0, 512])


def test_collection_cap_optout_cannot_reenter_training_as_random_trunk_anchors():
    assert add_trunk_anchors([], tuple(range(600)), 10,
                             dict(_mixed_anchor_bundle(), collection_enabled=False), _mixed_anchor_config()) == []


def proposals(n):
    return [DraftBranch(f"p{i}", 9 + i, (2, 3, 4), accepted_count=1) for i in range(n)]


@pytest.mark.parametrize("anchor_count", [32, 64])
def test_full_and_short_proposal_groups_fill_unique_positions(anchor_count):
    ids = tuple(range(600))
    spec = _mixed_anchor_config(anchor_count)
    spec.validate()
    for n in [0, 3, spec.max_branches_per_request]:
        mixed = add_trunk_anchors(proposals(n), ids, 10, _mixed_anchor_bundle(), spec)
        assert len(mixed) == len({b.anchor for b in mixed}) == anchor_count
        assert sum(b.source == "proposal" for b in mixed) == n
        assert all(9 <= b.anchor < 521 for b in mixed)
        for b in mixed[n:]:
            assert b.source == "trunk"
            assert b.token_ids == ids[b.anchor + 1:min(b.anchor + 8, 522)]
        assert mixed == add_trunk_anchors(proposals(n), ids, 10, _mixed_anchor_bundle(), spec)


def test_dspark_specopd_launch_defaults_to_pure_tv():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    hydra_parser = pytest.importorskip("hydra.core.override_parser.overrides_parser")
    root = Path(__file__).resolve().parents[2]
    bash = shutil.which("bash")
    if bash is None and Path("C:/Program Files/Git/bin/bash.exe").exists():
        bash = "C:/Program Files/Git/bin/bash.exe"
    if bash is None:
        pytest.skip("bash is unavailable")
    env = {**os.environ, **dict.fromkeys(("MODEL_PATH", "DRAFTER_PATH", "TRAIN_FILE", "TEST_FILE"), "/unused/test-path")}
    harness = 'python3() { printf "%s\\0" "$@"; }; script=$1; shift; source "$script"'
    script = "examples/run_qwen3-8b-specopd_drafter_dspark_vllm_npu.sh"
    result = subprocess.run([bash, "-c", harness, "specopd-config-test", script], cwd=root,
                            env=env, capture_output=True, check=True)
    argv = result.stdout.decode().rstrip("\0").split("\0")
    assert argv[:3] == ["-m", "verl_speco.main", "--config-name=speco_v1_trainer"]
    hydra_parser.OverridesParser.create().parse_overrides(argv[3:])
    config = OmegaConf.merge(
        OmegaConf.load(root / "verl_speco/config/speco_base.yaml"),
        OmegaConf.load(root / "verl_speco/config/speco_v1_trainer.yaml"),
        OmegaConf.from_dotlist([arg.lstrip("+") for arg in argv[3:]]),
    )
    training = config.actor_rollout_ref.rollout.drafter.training
    spec = validate_specopd_config(config)
    assert spec.enable and spec.objective == "tv"
    assert not training.drafter_opd.enable
    assert specopd_loss_weights(training, "DSPARK") == (1.0, 0.0)
