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
"""Legacy PPO/GRPO lifecycle adapter: replace old-logprob, train, then publish."""

from contextlib import contextmanager
from types import MethodType
import time

import torch

from verl_speco.joint_scoring import JointScoringRequest, DraftBranch
from verl_speco.specopd_config import validate_specopd_config
from verl_speco.transport.specopd_protocol import (
    PROPOSAL_KEY,
    decode_bundle,
    token_digest,
)
from verl_speco.trainer.scheduler import SpecOPDCycle


def build_requests(batch, spec, revision):
    tensors = batch.batch
    wires = batch.non_tensor_batch.get(PROPOSAL_KEY)
    if wires is None or len(wires) != len(tensors["input_ids"]):
        raise ValueError("SpecOPD proposal records are missing from the rollout batch")
    prompt_width = tensors["prompts"].shape[1]
    requests, actor_versions = [], set()
    seen = set()
    for index, wire in enumerate(wires):
        bundle = decode_bundle(wire, spec.max_wire_bytes)
        if (
            bundle["drafter_revision"] != revision
            or bundle["proposal_temperature"] != spec.proposal_temperature
        ):
            raise ValueError(
                "SpecOPD drafter revision/sampling configuration differs from the rollout"
            )
        actor_versions.add(int(bundle["actor_revision"]))
        mask = tensors["attention_mask"][index].bool()
        ids = tuple(tensors["input_ids"][index][mask].cpu().tolist())
        prompt_len = int(mask[:prompt_width].sum())
        if (
            len(bundle["records"]) > spec.max_branches_per_request
            or sum(len(r["proposal_ids"]) for r in bundle["records"])
            > spec.max_proposal_tokens_per_request
        ):
            raise ValueError(
                "SpecOPD collected proposals exceed the preplanned request budget"
            )
        branches = []
        for record in bundle["records"]:
            prefix = record["prefix_len"]
            if (
                not prompt_len <= prefix <= len(ids)
                or token_digest(ids[:prefix]) != record["prefix_digest"]
            ):
                raise ValueError(
                    "SpecOPD exact sampled prefix does not match its final rollout trunk"
                )
            if len(record["proposal_ids"]) > spec.proposal_width:
                raise ValueError("SpecOPD proposal exceeds the rollout block width")
            branch_id = record["branch_id"]
            if branch_id in seen:
                raise ValueError(
                    "SpecOPD duplicate request/branch in the rollout batch"
                )
            seen.add(branch_id)
            branches.append(
                DraftBranch(branch_id, prefix - 1, tuple(record["proposal_ids"]))
            )
        sample_id = str(bundle["request_id"])
        if any(r.sample_id == sample_id for r in requests):
            raise ValueError("SpecOPD request IDs are not unique after rollout merging")
        requests.append(
            JointScoringRequest(sample_id, ids, prompt_len, tuple(branches))
        )
    if len(actor_versions) != 1:
        raise ValueError("SpecOPD synchronous group contains multiple actor revisions")
    return requests, actor_versions.pop()


def unwrap_rank_result(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result
    active = [item for item in result if item is not None]
    if not active:
        raise RuntimeError("joint actor RPC returned no participating ranks")
    results = tuple(sample for local, _ in active for sample in local)
    metrics = {
        key: sum(values.get(key, 0) for _, values in active)
        for key in {key for _, values in active for key in values}
    }
    metrics["model_forward_calls"] = max(
        values["model_forward_calls"] for _, values in active
    )
    return results, metrics


def make_oldlogprob(batch, results):
    from verl import DataProto

    log_probs = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    entropy = torch.zeros_like(log_probs)
    for index, result in enumerate(results):
        n = result.response_log_probs.numel()
        if n > log_probs.shape[1]:
            raise ValueError("joint response output exceeds original response layout")
        log_probs[index, :n] = result.response_log_probs.to(log_probs.device)
        if result.response_entropy is not None:
            entropy[index, :n] = result.response_entropy.to(entropy.device)
    return DataProto.from_dict(
        tensors={"old_log_probs": log_probs, "entropys": entropy}
    )


def plan_microbatches(requests, spec):
    """Bound the entire schedule before any model call; never split a tree."""
    groups, current, lengths = [], [], []
    for request in requests:
        size = len(request.trunk_ids) + sum(
            len(b.token_ids) - 1 for b in request.branches
        )
        if (
            size > spec.scoring_token_budget
            or size * size > spec.max_dense_mask_elements
        ):
            raise ValueError(
                f"SpecOPD complete trajectory {request.sample_id} exceeds the joint budget"
            )
        proposed = lengths + [size]
        if current and (
            len(proposed) > spec.microbatch_trajectories
            or sum(proposed) > spec.scoring_token_budget
            or len(proposed) * max(proposed) ** 2 > spec.max_dense_mask_elements
        ):
            groups.append(current)
            current, lengths = [], []
        current.append(request)
        lengths.append(size)
    if current:
        groups.append(current)
    return groups


class SpecOPDDriver:
    def __init__(self, trainer):
        self.trainer = trainer
        self.spec = validate_specopd_config(trainer.config)
        self.cycle = SpecOPDCycle()
        self.feature_indices = ()
        self.trained = False
        self.metrics = {}
        self._bound_actor_revision = False
        self.bootstrapped = False
        self._acknowledged_actor_revision = None

    def acknowledge_serving_revision(self, revision):
        target = self.trainer._speco_rollout_generation_target()
        server_manager = getattr(self.trainer, "llm_server_manager", None)
        servers = getattr(server_manager, "server_handles", None) or getattr(target, "server_handles", None)
        if not servers:
            raise RuntimeError("SpecOPD requires rollout server handles for serving revision ACKs")
        replies = self.trainer._ray_get_if_needed([
            server.speco_set_serving_drafter_revision.remote(revision) for server in servers
        ])
        if not replies or any(
            not isinstance(reply, dict)
            or reply.get("drafter_revision") != revision
            or type(reply.get("actor_revision")) is not int
            or reply["actor_revision"] < 0
            for reply in replies
        ):
            raise RuntimeError("SpecOPD serving revision was not acknowledged by every replica")
        actors = {reply["actor_revision"] for reply in replies}
        if len(actors) != 1:
            raise RuntimeError("SpecOPD replicas acknowledge different actor revisions")
        self._acknowledged_actor_revision = actors.pop()
        self.metrics["specopd/ack_actor_revision"] = self._acknowledged_actor_revision
        self.metrics["specopd/ack_drafter_revision"] = revision

    def bootstrap(self):
        if self.bootstrapped:
            return
        self.trainer._speco_set_drafter_global_step()
        self.trainer._speco_sync_target_lm_head_weight()
        results = self.trainer._require_speco_worker_group().prepare_specopd_bootstrap(
            int(self.trainer.global_steps)
        )
        statuses = [
            s for s in self.trainer._ray_get_if_needed(results) if s is not None
        ]
        if not statuses or len({s["revision"] for s in statuses}) != 1:
            raise RuntimeError(
                "SpecOPD restored drafter ranks have different revisions"
            )
        metrics = self.trainer._speco_publish_drafter_weights(
            True, after_weight_update=True
        )
        if not metrics.get("drafter/published", False):
            raise RuntimeError(
                "SpecOPD initial/restored drafter state was not acknowledged by rollout"
            )
        self.cycle.drafter_revision = statuses[0]["revision"]
        self.acknowledge_serving_revision(self.cycle.drafter_revision)
        self.bootstrapped = True
        self.metrics["specopd/bootstrap_published"] = 1

    def old_log_prob(self, batch):
        if self.cycle.state != "collecting":
            raise RuntimeError(
                "SpecOPD old-logprob must follow exactly one rollout collection"
            )
        requests, actor_revision = build_requests(
            batch, self.spec, self.cycle.drafter_revision
        )
        if actor_revision != self._acknowledged_actor_revision:
            raise RuntimeError("SpecOPD rollout differs from the acknowledged serving actor revision")
        if self._bound_actor_revision and actor_revision < self.cycle.actor_revision:
            raise RuntimeError("SpecOPD rollout actor revision regressed")
        # Under synchronous legacy hooks the actor cannot update between this
        # rollout and old-logprob. Bind the actual serving revision (not prompt
        # submission global_step, which can be off by one at startup/resume).
        self.cycle.actor_revision = actor_revision
        self._bound_actor_revision = True
        raw_indices = self.trainer._speco_oldlogprob_aux_layer_ids()
        space = self.trainer._speco_oldlogprob_aux_layer_id_space()
        if any(i < 0 for i in raw_indices):
            raise ValueError(
                "SpecOPD currently requires resolved nonnegative auxiliary layer IDs"
            )
        self.feature_indices = tuple(
            i + 1 if space == "decoder" else i for i in raw_indices
        )
        results, metrics = (
            [],
            {
                "specopd/model_forward_calls": 0,
                "specopd/logical_rows": 0,
                "specopd/independent_rows": 0,
                "specopd/mfu_unavailable": 1,
                "specopd/actor_revision": actor_revision,
                "specopd/drafter_revision": self.cycle.drafter_revision,
            },
        )
        started = time.perf_counter()
        for chunk in plan_microbatches(requests, self.spec):
            output = self.trainer.actor_rollout_wg.compute_joint_old_log_prob(
                chunk,
                actor_temperature=self.spec.actor_temperature,
                target_temperature=self.spec.target_temperature,
                hidden_state_indices=self.feature_indices,
                calculate_entropy=(
                    self.trainer._speco_oldlogprob_calculate_entropy()
                    if hasattr(self.trainer, "_speco_oldlogprob_calculate_entropy")
                    else True
                ),
                return_target_logits=False,
                capture_empty_trunk_features=False,
                max_packed_tokens=self.spec.scoring_token_budget,
                max_mask_elements=self.spec.max_dense_mask_elements,
            )
            scored, measured = unwrap_rank_result(
                self.trainer._ray_get_if_needed(output)
            )
            indexed = {r.sample_id: r for r in scored}
            if len(indexed) != len(scored) or set(indexed) != {
                r.sample_id for r in chunk
            }:
                raise RuntimeError(
                    "joint actor RPC lost or duplicated trajectory results"
                )
            scored = tuple(indexed[r.sample_id] for r in chunk)
            results.extend(scored)
            for key in ("model_forward_calls", "logical_rows", "independent_rows"):
                metrics[f"specopd/{key}"] += measured[key]
        metrics["timing_s/specopd_joint_oldlogprob"] = time.perf_counter() - started
        self.metrics.update(metrics)
        self.cycle.scored(
            tuple(zip(requests, results, strict=True)),
            actor_revision,
            self.cycle.drafter_revision,
        )
        return make_oldlogprob(batch, results), 0.0  # No fabricated MFU estimate.

    def before_actor_update(self):
        samples = self.cycle.begin_update()
        positions = sum(
            len(b.token_ids) for request, _ in samples for b in request.branches
        )
        self.trained = False
        if positions >= self.spec.min_scored_positions_per_update:
            self.trainer._speco_set_drafter_global_step()
            self.metrics.update(self.trainer._speco_sync_target_lm_head_weight())
            output = self.trainer._require_speco_worker_group().train_specopd_group(
                samples,
                int(self.cycle.step),
                self.cycle.drafter_revision,
                self.feature_indices,
            )
            statuses = [
                s for s in self.trainer._ray_get_if_needed(output) if s is not None
            ]
            if (
                not statuses
                or len({(s["trained"], s["revision"]) for s in statuses}) != 1
            ):
                raise RuntimeError(
                    "SpecOPD drafter ranks disagree about the optimizer update"
                )
            self.trained = bool(statuses[0]["trained"])
            expected = self.cycle.drafter_revision + int(self.trained)
            if statuses[0]["revision"] != expected:
                raise RuntimeError(
                    "SpecOPD serving/training revision did not advance exactly once"
                )
        self.metrics["specopd/scored_positions"] = positions
        self.metrics["specopd/trained"] = int(self.trained)
        self.cycle.updated(self.trained)

    def publish(self):
        if self.cycle.state != "awaiting_publish":
            return
        if self.trained:
            previous_step = self.trainer.global_steps
            try:
                self.trainer.global_steps = self.cycle.step
                metrics = self.trainer._speco_publish_drafter_weights(
                    True, after_weight_update=True
                )
                if not metrics.get("drafter/published", False):
                    raise RuntimeError(
                        "SpecOPD updated drafter snapshot was not published"
                    )
                self.metrics.update(metrics)
            finally:
                self.trainer.global_steps = previous_step
        self.acknowledge_serving_revision(self.cycle.drafter_revision)
        self.cycle.published()
        self.metrics["specopd/published"] = int(self.trained)
        self.metrics["specopd/published_drafter_revision"] = self.cycle.drafter_revision
        if not self.trained:
            self.metrics["drafter/published"] = 0


@contextmanager
def specopd_fit_hooks(trainer):
    """Replace, rather than stack on top of, the legacy replay training hooks."""
    driver = SpecOPDDriver(trainer)
    target = trainer._speco_rollout_generation_target()
    generate, oldlog, update = (
        target.generate_sequences,
        trainer._compute_old_log_prob,
        trainer._update_actor,
    )
    manager = getattr(trainer, "checkpoint_manager", None)
    weight_update = getattr(manager, "update_weights", None)
    last_actor_output = None

    def generated(_manager, *args, **kwargs):
        driver.bootstrap()
        driver.publish()  # Previous actor weight update has completed before new rollout.
        batch = args[0] if args else kwargs.get("batch", kwargs.get("prompts"))
        if batch is None:
            raise ValueError("SpecOPD rollout requires a DataProto input")
        validation = batch.meta_info.get("validate", False) or getattr(
            trainer, "_speco_validation_in_progress", False
        )
        previous = dict(batch.meta_info)
        try:
            if validation:
                batch.meta_info["skip_drafter_collection"] = True
            else:
                driver.cycle.begin(trainer.global_steps)
                batch.meta_info["specopd_drafter_revision"] = (
                    driver.cycle.drafter_revision
                )
            result = generate(*args, **kwargs)
            trainer._speco_store_rollout_metrics(result)
            return result
        finally:
            batch.meta_info.clear()
            batch.meta_info.update(previous)

    def old_log_prob(_trainer, batch):
        return driver.old_log_prob(batch)

    def updated(_trainer, *args, **kwargs):
        nonlocal last_actor_output
        driver.before_actor_update()
        result = update(*args, **kwargs)
        trainer._speco_update_output_metrics(result, driver.metrics)
        last_actor_output = result
        return result

    def weights_updated(_manager, *args, **kwargs):
        result = weight_update(*args, **kwargs)
        driver.publish()
        if last_actor_output is not None:
            # Upstream logs actor_output only after weight publication. Include
            # the completed ACK instead of leaving its metrics one step behind.
            trainer._speco_update_output_metrics(last_actor_output, driver.metrics)
        return result

    target.generate_sequences = MethodType(generated, target)
    trainer._compute_old_log_prob = MethodType(old_log_prob, trainer)
    trainer._update_actor = MethodType(updated, trainer)
    if callable(weight_update):
        manager.update_weights = MethodType(weights_updated, manager)
    try:
        yield
        driver.publish()
    finally:
        (
            target.generate_sequences,
            trainer._compute_old_log_prob,
            trainer._update_actor,
        ) = generate, oldlog, update
        if callable(weight_update):
            manager.update_weights = weight_update
