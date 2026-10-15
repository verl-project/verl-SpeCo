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
"""Contracts for synchronous OPD and SPECO drafter co-training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


TEACHER_TENSOR_KEYS = frozenset({"teacher_ids", "teacher_logprobs"})


def _get_nested(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = config
    for key in path:
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def distillation_enabled(config: Any) -> bool:
    """Return whether upstream verl OPD is enabled."""

    return _as_bool(_get_nested(config, ("distillation", "enabled"), False))


def opd_cotrain_enabled(config: Any) -> bool:
    """Return whether OPD and online drafter training are both enabled."""

    return bool(
        distillation_enabled(config)
        and _as_bool(
            _get_nested(
                config,
                ("actor_rollout_ref", "rollout", "drafter", "enable"),
                False,
            )
        )
        and _as_bool(
            _get_nested(
                config,
                (
                    "actor_rollout_ref",
                    "rollout",
                    "drafter",
                    "enable_drafter_training",
                ),
                False,
            )
        )
    )


def _validate_teacher_configs(config: Any) -> None:
    teacher_models = _get_nested(config, ("distillation", "teacher_models"), {})
    items = teacher_models.items() if hasattr(teacher_models, "items") else ()
    for teacher_name, teacher_config in items:
        inference_config = _get_nested(teacher_config, ("inference",), {})
        if isinstance(inference_config, Mapping):
            has_drafter = "drafter" in inference_config
        else:
            has_drafter = hasattr(inference_config, "drafter")
        if has_drafter:
            raise ValueError(
                "Teacher speculative decoding is outside synchronous OPD Co-Train: "
                f"remove distillation.teacher_models.{teacher_name}.inference.drafter"
            )


def validate_sync_opd_cotrain_config(config: Any) -> None:
    """Fail closed on unsupported synchronous OPD Co-Train configurations.

    Native OPD, fixed-Drafter OPD, and non-OPD SPECO modes are deliberately left
    unchanged. The stricter checks apply only when upstream OPD and online
    Drafter training are enabled together.
    """

    if not distillation_enabled(config):
        return

    _validate_teacher_configs(config)

    drafter_path = ("actor_rollout_ref", "rollout", "drafter")
    rollout_enabled = _as_bool(_get_nested(config, drafter_path + ("enable",), False))
    training_enabled = _as_bool(
        _get_nested(config, drafter_path + ("enable_drafter_training",), False)
    )
    if training_enabled and not rollout_enabled:
        raise ValueError(
            "Synchronous OPD Co-Train requires "
            "actor_rollout_ref.rollout.drafter.enable=true when "
            "enable_drafter_training=true"
        )
    if not (rollout_enabled and training_enabled):
        return

    actor_strategy = str(
        _get_nested(config, ("actor_rollout_ref", "actor", "strategy"), "") or ""
    ).lower()
    if actor_strategy != "fsdp2":
        raise ValueError(
            "Synchronous OPD Co-Train supports only "
            "actor_rollout_ref.actor.strategy=fsdp2, "
            f"got {actor_strategy!r}"
        )

    training_path = drafter_path + ("training",)
    training_mode = str(
        _get_nested(config, training_path + ("mode",), "online") or "online"
    ).lower()
    if training_mode != "online":
        raise ValueError(
            "Synchronous OPD Co-Train requires "
            "actor_rollout_ref.rollout.drafter.training.mode=online, "
            f"got {training_mode!r}"
        )
    if not _as_bool(
        _get_nested(
            config,
            training_path + ("collect_hidden_states_from_old_logprob",),
            False,
        )
    ):
        raise ValueError(
            "Synchronous OPD Co-Train reuses the Student old-logprob forward; set "
            "actor_rollout_ref.rollout.drafter.training."
            "collect_hidden_states_from_old_logprob=true"
        )
    if _as_bool(
        _get_nested(
            config,
            training_path + ("collect_hidden_states_from_sgl",),
            False,
        )
    ):
        raise ValueError(
            "Synchronous OPD Co-Train old-logprob collection requires "
            "actor_rollout_ref.rollout.drafter.training."
            "collect_hidden_states_from_sgl=false"
        )
    if _as_bool(_get_nested(config, training_path + ("use_logits",), False)):
        raise ValueError(
            "Synchronous OPD Co-Train old-logprob collection supports "
            "actor_rollout_ref.rollout.drafter.training.use_logits=false only"
        )
    if _as_bool(
        _get_nested(
            config,
            ("algorithm", "rollout_correction", "bypass_mode"),
            False,
        )
    ):
        raise ValueError(
            "Synchronous OPD Co-Train requires the Student old-logprob forward; "
            "algorithm.rollout_correction.bypass_mode must be false"
        )


def student_policy_batch_keys(batch_keys: Iterable[Any]) -> list[Any]:
    """Exclude upstream Teacher tensors from Student feature forwards."""

    return [key for key in batch_keys if key not in TEACHER_TENSOR_KEYS]


def ensure_student_only_feature_payload(payload: Mapping[str, Any]) -> None:
    """Reject Teacher supervision at the Drafter worker boundary."""

    forbidden = sorted(TEACHER_TENSOR_KEYS.intersection(payload))
    if forbidden:
        raise ValueError(
            "Drafter features must be supervised by Student outputs only; "
            f"forbidden Teacher tensors: {', '.join(forbidden)}"
        )


def detach_student_feature(value: Any) -> Any:
    """Detach a Student feature before it crosses into Drafter training."""

    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise TypeError("Student feature must provide detach()")
    return detach()
