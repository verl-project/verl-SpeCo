# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")
from verl_speco.trainer.base_trainer import DrafterBaseTrainer


def trainer(applied=2, queued=2, **overrides):
    t = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    training = {"mode": "online", "use_logits": False,
                "collect_hidden_states_from_old_logprob": True,
                "dspark_l1_loss_alpha": 0., "dspark_lk_loss_alpha": .9,
                "dspark_confidence_loss_alpha": .1}
    training.update(overrides)
    t.config = NS(rollout=NS(drafter=NS(training=training)))
    t.backend = NS(model_type="dspark")
    t.model_config = NS(l1_loss_alpha=.9)
    t._applied_target_lm_head_weight_step = applied
    t._target_lm_head_weight_step = queued
    return t


@pytest.mark.parametrize("applied,queued,item", [(2, 2, 2), (2, 3, 2), (2, 1, 2)])
def test_uses_applied_not_queued_version(applied, queued, item):
    trainer(applied, queued)._validate_dspark_target_head_revision([{"target_version": item, "step": 99}])


@pytest.mark.parametrize("applied,queued,items", [
    (1, 2, [{"target_version": 2, "step": 2}]),
    (None, 2, [{"target_version": 2}]),
    (2, 2, [{"step": 2}]),
    (2, 2, [{"target_version": 2}, {"target_version": 1}]),
])
def test_missing_or_stale_versions_fail(applied, queued, items):
    with pytest.raises(RuntimeError, match="DSpark teacher"):
        trainer(applied, queued)._validate_dspark_target_head_revision(items)


@pytest.mark.parametrize("cfg", [
    {"mode": "offline"}, {"use_logits": True},
    {"collect_hidden_states_from_old_logprob": False},
    {"dspark_lk_loss_alpha": 0., "dspark_confidence_loss_alpha": 0.},
])
def test_unrelated_paths_unchanged(cfg):
    trainer(None, None, **cfg)._validate_dspark_target_head_revision([{}])


def test_actual_batch_preparation_checks_before_backend():
    class ReachedPreprocess(Exception):
        pass
    t = trainer()
    t.batch_size = 1
    t.current_rl_step = 2
    t.training_steps = 0
    t._bubble_time_enabled = False
    t.use_data_buffer = False
    t.collected_data = [{"step": 2, "target_version": 1, "hidden_states": torch.zeros(3, 4)}]
    t.rank = 0
    t.model = torch.nn.Linear(1, 1)
    t.use_ulysses_sp = False
    t._is_block_drafter_backend = lambda: True
    def preprocess(*args):
        raise ReachedPreprocess
    t.backend.preprocess_individual_items = preprocess
    with pytest.raises(RuntimeError, match="revision mismatch"):
        t._prepare_training_batch()
    t.collected_data[0]["target_version"] = 2
    with pytest.raises(ReachedPreprocess):
        t._prepare_training_batch()


def test_real_apply_copies_then_updates_applied_revision():
    t = trainer(applied=1, queued=2)
    head = torch.nn.Linear(3, 2, bias=False)
    t.backend.target_lm_head = NS(fc=head)
    t._pending_target_lm_head_weight = torch.full_like(head.weight, .25)
    t._pending_target_lm_head_row_indices = None
    t._pending_target_lm_head_source_vocab_size = None
    t._pending_target_lm_head_chunked_apply = False
    with pytest.raises(RuntimeError, match="revision mismatch"):
        t._validate_dspark_target_head_revision([{"target_version": 2}])
    assert t._apply_pending_target_lm_head_weight()
    assert torch.equal(head.weight, torch.full_like(head.weight, .25))
    assert t._applied_target_lm_head_weight_step == 2
    t._validate_dspark_target_head_revision([{"target_version": 2}])
