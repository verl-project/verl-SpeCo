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
"""Configuration contract for synchronous rollout-proposal distillation."""

from dataclasses import dataclass, fields

from verl_speco.opd_method import spec_opd_enabled


@dataclass(frozen=True)
class SpecOPDConfig:
    enable: bool = False
    collection_source: str = "rollout_proposals"
    window_mode: str = "front"
    window_response_tokens: int = 512
    random_max_offset: int = 512
    max_branches_per_request: int = 8
    proposal_width: int = 7
    max_proposal_tokens_per_request: int = 56
    scoring_token_budget: int = 8192
    max_dense_mask_elements: int = 16777216
    max_wire_bytes: int = 1048576
    min_scored_positions_per_update: int = 1
    microbatch_trajectories: int = 1
    actor_temperature: float = 1.0
    proposal_temperature: float = 1.0
    target_temperature: float = 1.0
    optimizer_updates_per_collection: int = 1
    replay: bool = False
    mix_trunk_loss: bool = False
    allow_bubble_training: bool = False
    capture_rejected_tail: bool = True
    scoring_phase: str = "before_actor_update"
    scorer: str = "actor"
    scoring_impl: str = "joint_tree_old_logprob"
    replace_normal_old_logprob_forward: bool = True
    one_forward_per_trajectory: bool = True
    independent_branches_for_validation_only: bool = True
    objective: str = "e2e_tv"
    objective_weight: float = 1.0
    confidence_loss_weight: float = 0.0
    strict_versions: bool = True
    max_drafter_revision_lag: int = 0

    @classmethod
    def from_training(cls, training):
        raw = training.get("specopd", {}) or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown SpecOPD options: {sorted(unknown)}")
        result = cls(**{**dict(raw), "enable": spec_opd_enabled(training)})
        if result.enable:
            result.validate()
        return result

    def validate(self):
        import math

        if self.window_mode not in {"front", "random"}:
            raise ValueError("SpecOPD window_mode must be front or random")
        for name in (
            "window_response_tokens",
            "max_branches_per_request",
            "proposal_width",
            "max_proposal_tokens_per_request",
            "scoring_token_budget",
            "max_dense_mask_elements",
            "max_wire_bytes",
            "min_scored_positions_per_update",
            "microbatch_trajectories",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"SpecOPD {name} must be a positive integer")
        if type(self.random_max_offset) is not int or self.random_max_offset < 0:
            raise ValueError("SpecOPD random_max_offset must be nonnegative")
        for name in ("actor_temperature", "proposal_temperature", "target_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"SpecOPD {name} must be finite and positive")
        if (
            self.optimizer_updates_per_collection != 1
            or self.replay
            or self.mix_trunk_loss
            or self.allow_bubble_training
        ):
            raise ValueError(
                "SpecOPD requires one update, no replay, no trunk loss and no Bubble"
            )
        if (
            self.collection_source != "rollout_proposals"
            or self.scoring_phase != "before_actor_update"
            or self.scorer != "actor"
            or self.scoring_impl != "joint_tree_old_logprob"
            or self.objective != "e2e_tv"
            or self.objective_weight != 1.0
            or self.confidence_loss_weight != 0.0
            or not self.capture_rejected_tail
            or not self.replace_normal_old_logprob_forward
            or not self.one_forward_per_trajectory
            or not self.strict_versions
            or self.max_drafter_revision_lag != 0
        ):
            raise ValueError(
                "SpecOPD requires the strict joint-forward E2E-TV contract"
            )
        if self.max_proposal_tokens_per_request < self.proposal_width:
            raise ValueError("proposal token budget cannot contain one complete branch")
        if self.max_wire_bytes > 1048576:
            raise ValueError("SpecOPD wire must fit the 1 MiB frontend transport limit")


def validate_specopd_config(config):
    actor = config.get("actor_rollout_ref", {})
    rollout = actor.get("rollout", {})
    drafter = rollout.get("drafter", {}) or {}
    training = drafter.get("training", {}) or {}
    spec = SpecOPDConfig.from_training(training)
    if not spec.enable:
        return spec
    if training.get("mode", "online") != "online" or training.get("use_logits", False):
        raise ValueError(
            "SpecOPD requires online hidden-state supervision and a synchronized target head"
        )
    if training.get("target_lm_head_row_restricted_sync", False):
        raise ValueError(
            "SpecOPD requires target_lm_head_row_restricted_sync=false for full target mass"
        )
    import math

    rate = training.get("collection_sample_rate", 1.0)
    if (
        not isinstance(rate, (float, int))
        or not math.isfinite(rate)
        or not 0 <= rate <= 1
    ):
        raise ValueError("SpecOPD collection_sample_rate must be in [0, 1]")
    limit = training.get("max_collect_samples_per_step_per_replica", 16)
    if type(limit) is not int or limit < 0:
        raise ValueError("SpecOPD request collection limit must be nonnegative")
    if actor.get("actor", {}).get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("SpecOPD actor sequence parallelism is unsupported")
    temperature = float(rollout.get("temperature", 1.0))
    if any(
        t != temperature
        for t in (
            spec.actor_temperature,
            spec.proposal_temperature,
            spec.target_temperature,
        )
    ):
        raise ValueError(
            "SpecOPD actor/proposal/target temperatures must match rollout.temperature"
        )
    if config.get("algorithm", {}).get("adv_estimator", "gae") not in {"gae", "grpo"}:
        raise ValueError(
            "SpecOPD joint output currently supports PPO/GAE and GRPO only"
        )
    if not drafter.get("enable") or not drafter.get("enable_drafter_training"):
        raise ValueError(
            "SpecOPD requires enabled rollout drafter and drafter training"
        )
    if rollout.get("name") != "vllm" or drafter.get(
        "speculative_algorithm", ""
    ).upper() not in {"DFLASH", "DSPARK", "DFLASH2"}:
        raise ValueError("SpecOPD supports vLLM DFlash/DSpark/DFlash2 only")
    if drafter.get("vllm", {}).get("draft_sample_method", "greedy") != "probabilistic":
        raise ValueError("SpecOPD requires vllm.draft_sample_method=probabilistic")
    trainer = config.get("trainer", {})
    if trainer.get("use_v1", False):
        v1 = trainer.get("v1", {})
        if v1.get("trainer_mode", "sync") != "sync":
            raise ValueError("SpecOPD V1 requires trainer_mode=sync")
        if v1.get("separate_async", {}).get("parameter_sync_step", 1) != 1:
            raise ValueError("SpecOPD V1 requires parameter_sync_step=1")
        if config.get("algorithm", {}).get("filter_groups", {}).get("enable", False):
            raise ValueError("SpecOPD V1 does not support group-filter refill")
    if actor.get("actor", {}).get("strategy", "fsdp2") not in {"fsdp", "fsdp2"}:
        raise ValueError("SpecOPD requires the FSDP actor engine")
    model = actor.get("model", {})
    if model.get("use_remove_padding", True) or model.get("use_fused_kernels", False):
        raise ValueError(
            "SpecOPD requires model.use_remove_padding=false and use_fused_kernels=false"
        )
    if model.get("lora_rank", 0) or model.get("lora_adapter_path"):
        raise ValueError("SpecOPD LoRA actor scoring is not supported")
    if (
        training.get("step", 1) != 1
        or training.get("collect_interval_steps", 1) != 1
        or training.get("training_interval_steps", 1) != 1
    ):
        raise ValueError("SpecOPD requires step=1 and collection/training intervals=1")
    if training.get("collect_hidden_states_from_sgl", False) or training.get(
        "use_data_buffer", False
    ):
        raise ValueError(
            "SpecOPD uses its own one-use scored group, not legacy collectors/replay"
        )
    if training.get("adaptive_schedule", {}).get("enable", False) or training.get(
        "freeze_policy", {}
    ).get("enable", False):
        raise ValueError("SpecOPD does not combine with adaptive/freeze scheduling")
    if training.get("freeze", {}).get("enabled", False):
        raise ValueError("SpecOPD does not combine with the legacy freeze policy")
    if training.get("publish_async", False) or training.get(
        "publish_interval_steps", 0
    ) not in {0, 1}:
        raise ValueError(
            "SpecOPD requires an acknowledged synchronous publish after each update"
        )
    if training.get("scheduler", {}).get("training_quota", {}).get("enable", False):
        raise ValueError("SpecOPD does not execute legacy training quotas")
    strategy = (
        training.get("scheduler", {}).get("execution", {}).get("strategy", "sync")
    )
    if strategy not in {"sync", "synchronous"}:
        raise ValueError("SpecOPD requires synchronous drafter execution")
    if (
        rollout.get("agent", {}).get("default_agent_loop", "single_turn_agent")
        != "single_turn_agent"
    ):
        raise ValueError(
            "SpecOPD currently supports exact single-turn token histories only"
        )
    if (
        config.get("algorithm", {})
        .get("rollout_correction", {})
        .get("bypass_mode", False)
    ):
        raise ValueError(
            "SpecOPD requires actor old-logprob inference; bypass_mode must be false"
        )
    if config.get("distillation", {}).get("enabled", False):
        raise ValueError(
            "SpecOPD joint old-logprob does not yet support external teacher distillation outputs"
        )
    name = drafter.get("speculative_algorithm", "").lower()
    capacity = training.get(
        f"{name}_block_size", {"dspark": 7, "dflash2": 8}.get(name, 16)
    )
    if int(capacity) != spec.proposal_width + (0 if name == "dspark" else 1):
        raise ValueError(
            "SpecOPD training block size must equal the actual rollout proposal block"
        )
    if (
        drafter.get("rollout", {}).get("spec_verify_tokens", spec.proposal_width)
        != spec.proposal_width
    ):
        raise ValueError("SpecOPD proposal_width must match vLLM spec_verify_tokens")
    overrides = drafter.get("vllm", {}).get("speculative_config_overrides", {}) or {}
    if any(
        key in overrides
        for key in (
            "draft_sample_method",
            "num_speculative_tokens",
            "dynamic_spec_config",
        )
    ):
        raise ValueError(
            "SpecOPD sampling/width/dynamic settings cannot be replaced by speculative overrides"
        )
    return spec
