# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verl_speco.integration.opd_cotrain import (
    TEACHER_TENSOR_KEYS,
    detach_student_feature,
    ensure_student_only_feature_payload,
    student_policy_batch_keys,
    validate_sync_opd_cotrain_config,
)


def _cotrain_config(**overrides):
    config = {
        "distillation": {
            "enabled": True,
            "teacher_models": {"teacher_model": {"inference": {}}},
        },
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp2"},
            "rollout": {
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "training": {
                        "mode": "online",
                        "collect_hidden_states_from_old_logprob": True,
                    },
                }
            },
        },
    }
    config.update(overrides)
    return config


def test_teacher_tensor_keys_match_upstream_contract() -> None:
    assert TEACHER_TENSOR_KEYS == {"teacher_ids", "teacher_logprobs"}


def test_student_policy_batch_keys_drop_only_teacher_tensors() -> None:
    keys = ["input_ids", "teacher_ids", "responses", "teacher_logprobs"]

    assert student_policy_batch_keys(keys) == ["input_ids", "responses"]


def test_student_only_payload_accepts_student_features() -> None:
    ensure_student_only_feature_payload({"hidden_states": 1, "input_ids": 2})


@pytest.mark.parametrize("key", sorted(TEACHER_TENSOR_KEYS))
def test_student_only_payload_rejects_teacher_tensors(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        ensure_student_only_feature_payload({"hidden_states": 1, key: 2})


def test_detach_student_feature_detaches_and_rejects_plain_values() -> None:
    detached = object()
    feature = SimpleNamespace(detach=lambda: detached)

    assert detach_student_feature(feature) is detached
    with pytest.raises(TypeError, match="detach"):
        detach_student_feature(object())


def test_select_policy_model_batch_excludes_teacher_tensors() -> None:
    pytest.importorskip("verl")
    from verl_speco.trainer.speco_ray_trainer import _select_policy_model_batch

    seen = {}

    class Batch:
        batch = {"input_ids": 1, "teacher_ids": 2, "teacher_logprobs": 3}
        non_tensor_batch = {}

        def select(self, batch_keys, non_tensor_batch_keys):
            seen["keys"] = batch_keys
            return self

    _select_policy_model_batch(Batch())

    assert seen["keys"] == ["input_ids"]


def test_teacher_drafter_rejected_only_for_cotrain() -> None:
    teacher_with_drafter = {
        "teacher_model": {"inference": {"drafter": {"enable": True}}}
    }
    config = _cotrain_config()
    config["distillation"]["teacher_models"] = teacher_with_drafter
    with pytest.raises(ValueError, match="inference.drafter"):
        validate_sync_opd_cotrain_config(config)

    opd_only = _cotrain_config()
    opd_only["distillation"]["teacher_models"] = teacher_with_drafter
    opd_only["actor_rollout_ref"]["rollout"]["drafter"]["enable"] = False
    opd_only["actor_rollout_ref"]["rollout"]["drafter"]["enable_drafter_training"] = (
        False
    )
    validate_sync_opd_cotrain_config(opd_only)