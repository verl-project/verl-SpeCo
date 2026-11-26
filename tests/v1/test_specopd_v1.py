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
"""One-use SpecOPD lifecycle contracts for V1 sync, independent of hardware."""

from types import SimpleNamespace

import pytest

from verl_speco.trainer.v1 import speco_mixin, specopd_adapter
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


def test_v1_sync_bypasses_legacy_replay_and_publishes_after_actor(monkeypatch):
    events = []
    ticks = iter([10.0, 30.0, 80.0])
    monkeypatch.setattr(speco_mixin, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    value = SimpleNamespace(
        update_due=True,
        metrics={"specopd/trained": 1},
        before_actor_update=lambda: events.append("draft-update"),
        finish_step=lambda: events.append("draft-publish-ack"),
    )
    monkeypatch.setattr(specopd_adapter, "enabled", lambda trainer: True)
    monkeypatch.setattr(specopd_adapter, "driver", lambda trainer: value)
    monkeypatch.setattr(specopd_adapter, "prepare_rollout", lambda trainer, batch: events.append("bind-rollout") or batch)
    monkeypatch.setattr(specopd_adapter, "old_log_prob", lambda trainer, batch, metrics: events.append("joint-tq-oldlogprob") or batch)

    class Base:
        def _submit_batch_to_rollout(self, batch):
            events.append("rollout")
            return batch

        def _compute_old_log_prob(self, batch, metrics):
            raise AssertionError("normal oldlogprob must be replaced")

        def _update_actor(self, batch, metrics):
            events.append("actor-update")
            return batch

        def on_step_end(self):
            events.append("actor-publish")

    class Trainer(SpecoV1Mixin, Base):
        pass

    trainer = Trainer()
    batch, metrics = object(), {}
    assert trainer._submit_batch_to_rollout(batch) is batch
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert trainer._update_actor(batch, metrics) is batch
    assert metrics["timing_s/actor_update_without_drafter"] == 50.0
    assert metrics["timing_s/specopd_before_actor_update"] == 20.0
    trainer.on_step_end()
    assert trainer._pending_sync_metrics["specopd/trained"] == 1
    assert events == ["bind-rollout", "rollout", "joint-tq-oldlogprob", "draft-update", "actor-update", "actor-publish", "draft-publish-ack"]


def test_noncollection_v1_step_uses_normal_oldlogprob(monkeypatch):
    monkeypatch.setattr(specopd_adapter, "enabled", lambda trainer: True)
    monkeypatch.setattr(specopd_adapter, "driver", lambda trainer: SimpleNamespace(update_due=False))
    monkeypatch.setattr(specopd_adapter, "old_log_prob", lambda *args: pytest.fail("unexpected joint scoring"))
    class Base:
        def _compute_old_log_prob(self, batch, metrics):
            metrics["normal_oldlogprob"] = 1
            return batch
    class Trainer(SpecoV1Mixin, Base):
        pass
    batch, metrics = object(), {}
    assert Trainer()._compute_old_log_prob(batch, metrics) is batch
    assert metrics == {"normal_oldlogprob": 1}


@pytest.mark.parametrize("branch_indices,calculate_entropy", [
    ((1, 3), True),
    ((1,), False),
    ((), True),
    ((0, 1, 2, 3), True),
])
def test_v1_native_trunk_scores_disjoint_keys(monkeypatch, branch_indices, calculate_entropy):
    import sys
    from types import ModuleType

    import torch

    from verl_speco.integration import specopd_driver
    from verl_speco.joint_scoring import DraftBranch, JointScores, JointScoringRequest

    monkeypatch.setenv("VERL_SPECO_V1_NATIVE_TRUNK_OLD_LOGPROB", "1")
    keys = ["a", "b", "c", "d"]
    masks = [torch.tensor(row) for row in ([1, 1], [1, 0, 1], [0], [1, 1])]
    log_probs = [-torch.arange(1, len(mask) + 1).float() - i * 10 for i, mask in enumerate(masks)]
    entropies = [row.abs() * 0.25 for row in log_probs]
    response_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    requests = [
        JointScoringRequest(
            key, tuple(range(1, 3 + len(mask))), 2,
            (DraftBranch(f"{key}-branch", 1, (8, 9)),) if i in branch_indices else (),
        )
        for i, (key, mask) in enumerate(zip(keys, masks, strict=True))
    ]
    monkeypatch.setattr(specopd_driver, "build_requests", lambda *_: (requests, 7))

    class Fields(dict):
        def select(self, *names):
            return Fields({name: self[name] for name in names})

    class Meta:
        partition_id = "train"

        def __init__(self, keys, tags, extra_info):
            self.keys, self.tags, self.extra_info = list(keys), list(tags), dict(extra_info)

        def select_keys(self, subset):
            return Meta(subset, [self.tags[self.keys.index(key)] for key in subset], self.extra_info)

        def reorder(self, indices):
            self.keys = [self.keys[i] for i in indices]
            self.tags = [self.tags[i] for i in indices]

        def __len__(self):
            return len(self.keys)

    batch = Meta(keys, [{"seq_len": len(r.trunk_ids)} for r in requests], {"temperature": 0.6})
    padded = SimpleNamespace(batch={"response_mask": response_mask, "responses": response_mask})
    calls = {}

    def stub(name, **values):
        module = ModuleType(name)
        for key, value in values.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    def get(*, keys, partition_id, select_fields):
        assert partition_id == "train"
        assert keys == [key for key in batch.keys if key in calls["native"]]
        assert set(select_fields) == (
            {"log_probs", "response_mask", "entropy"} if calculate_entropy else {"log_probs", "response_mask"}
        )
        indices = [batch.keys.index(key) for key in keys]
        data = {"response_mask": [masks[i] for i in indices]}
        for field, values in (("log_probs", log_probs), ("entropy", entropies)):
            data[field] = [torch.cat((torch.zeros(1), values[i], torch.zeros(1))) for i in indices]
        return data

    def put(*, keys, partition_id, fields):
        assert keys == batch.keys and partition_id == "train"
        calls["put"] = fields
        return batch

    def pad(subset, multiple, eos):
        assert multiple == 2 and eos == 2
        if len(subset) % multiple:
            return Meta(subset.keys + ["temporary-pad"], subset.tags + [{"seq_len": 4}], subset.extra_info)
        return subset

    stub("transfer_queue", kv_batch_get=get, kv_batch_put=put, kv_clear=lambda **kw: calls.update(clear=kw))
    stub("verl.trainer.ppo.padding_utils", upsample_batch_to_divisible_size=pad)
    stub("verl.utils.seqlen_balancing", calculate_workload=lambda lengths: lengths,
         get_seqlen_balanced_partitions=lambda lengths, **kw: [list(reversed(range(len(lengths))))])
    stub("verl.workers.utils.padding",
         response_from_nested=lambda rows, mask: torch.nested.as_nested_tensor(
             [row[-len(m) - 1:-1] for row, m in zip(rows, mask, strict=True)], layout=torch.jagged,
         ),
         response_to_nested=lambda values, mask: [row[:len(m)] for row, m in zip(values, mask, strict=True)])
    stub("verl.trainer.ppo.core_algos",
         agg_loss=lambda **kw: (kw["loss_mat"] * kw["loss_mask"]).sum() / kw["loss_mask"].sum())
    stub("verl", DataProto=SimpleNamespace(from_dict=lambda tensors: SimpleNamespace(batch=tensors)))

    def score_native(subset):
        assert subset.extra_info == dict(temperature=0.6, compute_loss=False, calculate_entropy=calculate_entropy)
        assert len(subset) % 2 == 0
        calls["native"] = subset.keys
        return subset

    def score_joint(batch, **options):
        assert options["actor_temperature"] == 0.6 and options["calculate_entropy"] == calculate_entropy
        assert options["return_target_logz"] is False
        calls["joint"] = [req.sample_id for req in batch]
        entries = []
        for req in reversed(batch):
            i = keys.index(req.sample_id)
            entries.append(JointScores(req.sample_id, log_probs[i], (), {}, entropies[i] if calculate_entropy else None))
        return [(entries, dict(model_forward_calls=1, logical_rows=16, independent_rows=16))]

    trainer = SimpleNamespace(
        _speco_v1_batch_data=lambda batch: (Fields(response_mask=masks), padded),
        _speco_oldlogprob_aux_layer_ids=lambda: (0,),
        _speco_oldlogprob_aux_layer_id_space=lambda: "decoder",
        _speco_oldlogprob_calculate_entropy=lambda: calculate_entropy,
        _ray_get_if_needed=lambda output: output,
        actor_rollout_wg=SimpleNamespace(
            _dispatch_info={"actor": [0, 0, 1, 1]},
            compute_log_prob=score_native, compute_joint_old_log_prob=score_joint,
        ),
        tokenizer=SimpleNamespace(eos_token_id=2),
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(loss_agg_mode="token-mean", loss_scale_factor=None),
            rollout=SimpleNamespace(temperature=0.6, get={}.get),
        )),
    )
    value = object.__new__(specopd_driver.SpecOPDDriver)
    value.trainer, value.metrics = trainer, {}
    value.actor_revision = value.drafter_revision = 0
    value._acknowledged_actor_revision, value._bound_actor_revision = 7, False
    value.scoring_sp_size = 1
    value.spec = SimpleNamespace(
        scoring_token_budget=64, max_dense_mask_elements=1024, microbatch_trajectories=4,
        temperature=0.6,
    )
    value.state, value.step, value.pending = "collecting", 5, None
    monkeypatch.setattr(specopd_driver, "build_specopd_training_batch", lambda req, *args: {"_specopd_request_id": req.sample_id})
    monkeypatch.setattr(specopd_adapter, "driver", lambda trainer: value)

    metrics = {}
    assert specopd_adapter.old_log_prob(trainer, batch, metrics) is batch
    branch_keys = [keys[i] for i in branch_indices]
    trunk_keys = set(keys) - set(branch_keys)
    assert calls.get("joint", []) == branch_keys
    assert set(calls.get("native", [])) - {"temporary-pad"} == trunk_keys
    assert calls.get("clear") == (
        dict(keys=["temporary-pad"], partition_id="train") if len(trunk_keys) % 2 else None
    )
    assert batch.keys == keys and batch.extra_info == {"temperature": 0.6}
    assert value.state == "scored"
    assert [b["_specopd_request_id"] for b in value.pending] == branch_keys
    for actual, expected in zip(calls["put"]["old_log_probs"], log_probs, strict=True):
        torch.testing.assert_close(actual, expected)
    masked_entropy = sum((entropy * mask).sum().item() for entropy, mask in zip(entropies, masks, strict=True))
    expected_entropy = masked_entropy / response_mask.sum().item() if calculate_entropy else 0.0
    assert metrics["actor/entropy"] == pytest.approx(expected_entropy)
