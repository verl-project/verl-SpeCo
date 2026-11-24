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

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SpecOPDConfig:
    enable: bool = False
    window_mode: str = "front"
    window_response_tokens: int = 512
    random_max_offset: int = 512
    max_branches_per_request: int = 8
    proposal_width: int = 7
    max_proposal_tokens_per_request: int = 56
    max_collect_samples_per_step: int | None = None
    scoring_token_budget: int = 9216
    max_dense_mask_elements: int = 84934656
    max_wire_bytes: int = 1048576
    min_scored_positions_per_update: int = 1
    microbatch_trajectories: int = 1
    training_microbatch_trajectories: int = 4
    training_microbatch_token_budget: int = 16384
    training_input_cache_mib: int = 64
    temperature: float = 1.0
    optimizer_updates_per_collection: int = 1
    # Zero retains the legacy branch-only path. Mixed modes reserve a quarter for trunk anchors.
    anchor_count: int = 0
    random_anchor_count: int = 8
    first_reject_weight: float = 2.0
    random_position_decay_gamma: float = 7.0
    objective: str = "e2e_tv"

    @property
    def batch_options(self):
        """One sampling/loss contract for every trajectory in a prepared group."""
        return dict(
            specopd=True,
            objective=self.objective,
            proposal_temperature=self.temperature,
            target_temperature=self.temperature,
            proposal_block_width=self.proposal_width,
        )

    @classmethod
    def from_drafter(cls, drafter, *, rollout_temperature):
        return cls.from_training(
            drafter.get("training") or {},
            algorithm=drafter.get("speculative_algorithm") or "",
            verify_tokens=(drafter.get("rollout") or {}).get("spec_verify_tokens"),
            rollout_temperature=rollout_temperature,
        )

    @classmethod
    def from_training(
        cls, training, *, algorithm="", verify_tokens=None, rollout_temperature=None
    ):
        options = dict(training.get("specopd", {}) or {})
        aliases = ("actor_temperature", "proposal_temperature", "target_temperature")
        temperatures = [options.pop(name, None) for name in aliases]
        unknown = options.keys() - ({f.name for f in fields(cls)} - {"temperature"})
        if unknown:
            raise ValueError(f"unknown SpecOPD options: {sorted(unknown)}")
        enabled = spec_opd_enabled(training)
        if enabled:
            algorithm = algorithm.lower()
            temperature = (
                cls.temperature
                if rollout_temperature is None
                else float(rollout_temperature)
            )
            if any(
                value is not None and value != temperature for value in temperatures
            ):
                raise ValueError(
                    "SpecOPD actor/proposal/target temperatures must match rollout.temperature"
                )
            options["temperature"] = temperature
            anchors = options.get("anchor_count", cls.anchor_count)
            if anchors is None:
                anchors = training.get(f"{algorithm}_num_anchors")
                if anchors not in (32, 64):
                    raise ValueError(
                        f"SpecOPD anchor_count=null requires {algorithm}_num_anchors=32 or 64"
                    )
            if type(anchors) is not int or anchors not in (0, 32, 64):
                raise ValueError("SpecOPD anchor_count must be 0 (legacy), 32 or 64")
            trunk_slots = anchors // 4 if anchors else cls.random_anchor_count
            branches = options.get("max_branches_per_request")
            if branches is None:
                branches = (
                    anchors - trunk_slots if anchors else cls.max_branches_per_request
                )
            width = options.get("proposal_width")
            if width is None:
                width = cls.proposal_width if verify_tokens is None else verify_tokens
            defaults = {
                "anchor_count": anchors,
                "proposal_width": width,
                "window_mode": training.get("hidden_state_window_mode")
                or cls.window_mode,
                "window_response_tokens": (
                    training.get("hidden_state_window_tokens_per_sample")
                    or cls.window_response_tokens
                ),
                "random_max_offset": training.get("hidden_state_random_max_offset"),
                "first_reject_weight": (training.get("drafter_opd") or {}).get(
                    "first_reject_weight", cls.first_reject_weight
                ),
                "random_anchor_count": trunk_slots,
                "random_position_decay_gamma": training.get(
                    f"{algorithm}_loss_decay_gamma"
                ),
                "max_branches_per_request": branches,
                "max_proposal_tokens_per_request": (
                    branches * width
                    if type(branches) is int and type(width) is int
                    else cls.max_proposal_tokens_per_request
                ),
            }
            for name, default in defaults.items():
                if options.get(name) is None:
                    options[name] = getattr(cls, name) if default is None else default
        else:
            # Shared YAML uses null for values resolved only by enabled SpecOPD.
            options = {
                name: value for name, value in options.items() if value is not None
            }
        updates = training.get("step", 1)
        options.setdefault("optimizer_updates_per_collection", updates)
        if enabled and options["optimizer_updates_per_collection"] != updates:
            raise ValueError(
                "SpecOPD optimizer_updates_per_collection must match training.step"
            )
        result = cls(**{**options, "enable": enabled})
        if enabled:
            result.validate()
        return result

    def validate(self):
        if type(self.anchor_count) is not int or self.anchor_count not in (0, 32, 64):
            raise ValueError("SpecOPD anchor_count must be 0 (legacy), 32 or 64")
        trunk_slots = self.anchor_count // 4
        proposal_slots = self.anchor_count - trunk_slots
        if self.anchor_count and (
            self.random_anchor_count != trunk_slots
            or self.max_branches_per_request != proposal_slots
            or self.max_proposal_tokens_per_request
            < proposal_slots * self.proposal_width
        ):
            raise ValueError(
                f"{self.anchor_count}-anchor SpecOPD requires {proposal_slots} proposal slots "
                f"and {trunk_slots} random trunk slots"
            )
        if not math.isfinite(self.first_reject_weight) or self.first_reject_weight <= 0:
            raise ValueError("SpecOPD first_reject_weight must be finite and positive")
        gamma = self.random_position_decay_gamma
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError(
                "SpecOPD random_position_decay_gamma must be finite and nonnegative"
            )
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
            "training_microbatch_trajectories",
            "training_microbatch_token_budget",
            "optimizer_updates_per_collection",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"SpecOPD {name} must be a positive integer")
        if type(self.random_max_offset) is not int or self.random_max_offset < 0:
            raise ValueError("SpecOPD random_max_offset must be nonnegative")
        if self.max_collect_samples_per_step is not None and (
            type(self.max_collect_samples_per_step) is not int
            or self.max_collect_samples_per_step < 1
        ):
            raise ValueError(
                "SpecOPD max_collect_samples_per_step must be a positive integer or null"
            )
        if (
            type(self.training_input_cache_mib) is not int
            or self.training_input_cache_mib < 0
        ):
            raise ValueError(
                "SpecOPD training_input_cache_mib must be a nonnegative integer"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("SpecOPD temperature must be finite and positive")
        if self.objective not in {"e2e_tv", "tv", "l1"}:
            raise ValueError("SpecOPD objective must be e2e_tv, tv or l1")
        if self.max_proposal_tokens_per_request < self.proposal_width:
            raise ValueError("proposal token budget cannot contain one complete branch")
        if self.max_wire_bytes > 1048576:
            raise ValueError("SpecOPD wire must fit the 1 MiB frontend transport limit")


def specopd_training_config(config):
    for key in ("actor_rollout_ref", "rollout", "drafter", "training"):
        config = (
            config.get(key, {}) if hasattr(config, "get") else getattr(config, key, {})
        )
    return config or {}


def spec_opd_enabled(training):
    return (training.get("specopd") or {}).get("enable", False)


def specopd_loss_weights(training, algorithm):
    """Read the existing drafter CE and distribution-loss coefficients."""
    if algorithm.lower() != "dspark":
        raise ValueError("SpecOPD requires DSpark")
    distribution = float(training.get("dspark_l1_loss_alpha", 0.9))
    ce = float(training.get("dspark_ce_loss_alpha", 0.1))
    if not all(
        math.isfinite(weight) and weight >= 0 for weight in (distribution, ce)
    ) or not (distribution or ce):
        raise ValueError(
            "SpecOPD CE and distribution weights must be finite, nonnegative and not both zero"
        )
    return distribution, ce


def validate_specopd_config(config):
    actor = config.get("actor_rollout_ref", {})
    rollout = actor.get("rollout", {})
    drafter = rollout.get("drafter", {}) or {}
    training = drafter.get("training", {}) or {}
    temperature = rollout.get("temperature", 1.0)
    spec = SpecOPDConfig.from_drafter(drafter, rollout_temperature=temperature)
    if not spec.enable:
        return spec
    if (training.get("drafter_opd") or {}).get("enable", False):
        raise ValueError("Enable only one of SpecOPD and Draft-OPD")
    if (
        str(training.get("dspark_per_position_loss_weight", "")).strip().lower()
        == "vat"
    ):
        raise ValueError(
            "SpecOPD cannot be combined with VAT; choose only one training objective"
        )
    # Reject unsupported features before initializing model workers.
    for path in (
        "actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync",
        "actor_rollout_ref.model.lora_rank",
        "actor_rollout_ref.model.lora_adapter_path",
        "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl",
        "actor_rollout_ref.rollout.drafter.training.use_data_buffer",
        "actor_rollout_ref.rollout.drafter.training.adaptive_schedule.enable",
        "actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze.enabled",
        "actor_rollout_ref.rollout.drafter.training.publish_async",
        "actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.enable",
        "algorithm.rollout_correction.bypass_mode",
        "distillation.enabled",
    ):
        value = config
        for key in path.split("."):
            value = (value or {}).get(key, {})
        if value:
            raise ValueError(f"SpecOPD requires {path} to be disabled")
    if training.get("mode", "online") != "online" or training.get("use_logits", False):
        raise ValueError(
            "SpecOPD requires online hidden-state supervision and a synchronized target head"
        )
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
    sp_size = actor.get("actor", {}).get("ulysses_sequence_parallel_size", 1)
    if type(sp_size) is not int or sp_size < 1:
        raise ValueError("SpecOPD Ulysses size must be a positive integer")
    data = config.get("data", {})
    prompt = data.get("max_prompt_length")
    response = data.get("max_response_length")
    if type(prompt) is int and prompt > 0 and type(response) is int and response > 0:
        # Mixed groups fill unused proposal slots with trunk anchors.
        branches = spec.anchor_count or min(
            spec.max_branches_per_request,
            spec.max_proposal_tokens_per_request // spec.proposal_width,
        )
        branches = min(branches, response, spec.window_response_tokens)
        tokens = prompt + response + branches * (spec.proposal_width - 1)
        width = ((tokens + sp_size - 1) // sp_size) * sp_size
        mask_elements = width * width
        if (
            tokens > spec.scoring_token_budget
            or mask_elements > spec.max_dense_mask_elements
        ):
            raise ValueError(
                f"SpecOPD worst-case trajectory requires scoring_token_budget>={tokens} "
                f"and max_dense_mask_elements>={mask_elements} (SP={sp_size}); "
                "increase the scoring limits or reduce data.max_prompt_length, "
                "data.max_response_length, or the branch budget"
            )
    if config.get("algorithm", {}).get("adv_estimator", "gae") not in {"gae", "grpo"}:
        raise ValueError(
            "SpecOPD joint output currently supports PPO/GAE and GRPO only"
        )
    if not drafter.get("enable") or not drafter.get("enable_drafter_training"):
        raise ValueError(
            "SpecOPD requires enabled rollout drafter and drafter training"
        )
    if (
        rollout.get("name") != "vllm"
        or drafter.get("speculative_algorithm", "").upper() != "DSPARK"
    ):
        raise ValueError("SpecOPD requires DSpark on vLLM")
    if drafter.get("vllm", {}).get("draft_sample_method", "greedy") not in {
        "greedy",
        "probabilistic",
    }:
        raise ValueError("SpecOPD draft_sample_method must be greedy or probabilistic")
    trainer = config.get("trainer", {})
    if not trainer.get("use_v1", False):
        raise ValueError("SpecOPD requires trainer.use_v1=true")
    if trainer.get("critic_warmup", 0) > 0:
        raise ValueError(
            "SpecOPD requires critic_warmup=0 so every collected step updates the actor"
        )
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
    if model.get("use_remove_padding", True) != (sp_size > 1) or model.get(
        "use_fused_kernels", False
    ):
        raise ValueError(
            "SpecOPD requires use_remove_padding=(SP>1) and use_fused_kernels=false"
        )
    interval = training.get("collect_interval_steps", 1)
    if (
        type(interval) is not int
        or interval < 1
        or training.get("training_interval_steps", 1) != interval
    ):
        raise ValueError(
            "SpecOPD requires equal positive collection/training intervals"
        )
    if training.get("publish_interval_steps", 0) not in {0, 1}:
        raise ValueError(
            "SpecOPD requires an acknowledged synchronous publish after each update"
        )
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
    name = drafter.get("speculative_algorithm", "").lower()
    if spec.objective in {"tv", "l1"}:
        specopd_loss_weights(training, name)
    capacity = training.get("dspark_block_size", 7)
    if int(capacity) != spec.proposal_width:
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
