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
"""Synchronous V1 lifecycle: score the rollout, train, then publish."""

import time

import torch

from verl_speco.joint_scoring import (
    JointScoringRequest,
    DraftBranch,
)
from verl_speco.specopd_config import validate_specopd_config
from verl_speco.integration.specopd_protocol import (
    PROPOSAL_KEY,
    decode_bundle,
    token_digest,
)
from verl_speco.backends.specopd_forward import build_specopd_training_batch


def build_requests(batch, spec, revision):
    tensors = batch.batch
    wires = batch.non_tensor_batch.get(PROPOSAL_KEY)
    if wires is None or len(wires) != len(tensors["input_ids"]):
        raise ValueError("SpecOPD proposal records are missing from the rollout batch")
    prompt_width = tensors["prompts"].shape[1]
    requests, actor_versions = [], set()
    seen = set()
    for index, wire in enumerate(wires):
        if batch.non_tensor_batch.get("_specopd_padding", [False] * len(wires))[index]:
            mask = tensors["attention_mask"][index].bool()
            ids = tuple(tensors["input_ids"][index][mask].cpu().tolist())
            prompt_len = int(mask[:prompt_width].sum())
            requests.append(
                JointScoringRequest(f"__specopd_padding_{index}", ids, prompt_len)
            )
            continue
        bundle = decode_bundle(wire, spec.max_wire_bytes)
        if (
            bundle["drafter_revision"] != revision
            or bundle["proposal_temperature"] != spec.temperature
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
                DraftBranch(
                    branch_id,
                    prefix - 1,
                    tuple(record["proposal_ids"]),
                    accepted_count=record["accepted_count"],
                    verified_count=record.get("verified_count"),
                )
            )
        sample_id = str(bundle["request_id"])
        if spec.anchor_count:
            from verl_speco.integration.specopd_anchors import add_trunk_anchors

            branches = add_trunk_anchors(branches, ids, prompt_len, bundle, spec)
        if any(r.sample_id == sample_id for r in requests):
            raise ValueError("SpecOPD request IDs are not unique after rollout merging")
        requests.append(
            JointScoringRequest(sample_id, ids, prompt_len, tuple(branches))
        )
    if len(actor_versions) != 1:
        raise ValueError("SpecOPD synchronous group contains multiple actor revisions")
    return requests, actor_versions.pop()


def unwrap_rank_result(result):
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
    sp_sizes = {values.get("sp_size", 1) for _, values in active}
    if len(sp_sizes) != 1:
        raise RuntimeError("joint actor ranks disagree about the active SP size")
    metrics["sp_size"] = sp_sizes.pop()
    return results, metrics


def make_oldlogprob(batch, results):
    log_probs = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    entropy = torch.zeros_like(log_probs)
    for index, result in enumerate(results):
        n = result.response_log_probs.numel()
        if n > log_probs.shape[1]:
            raise ValueError("joint response output exceeds original response layout")
        log_probs[index, :n] = result.response_log_probs.to(log_probs.device)
        if result.response_entropy is not None:
            entropy[index, :n] = result.response_entropy.to(entropy.device)
    return {"old_log_probs": log_probs, "entropy": entropy}


class SpecOPDDriver:
    def __init__(self, trainer):
        self.trainer = trainer
        self.spec = validate_specopd_config(trainer.config)
        self.state, self.pending, self.step = "ready", None, None
        self.actor_revision = self.drafter_revision = 0
        self.feature_indices = ()
        self.trained = False
        self.metrics = {}
        self._bound_actor_revision = False
        self.bootstrapped = False
        self._acknowledged_actor_revision = None
        training = trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"]
        self.collection_interval = int(training.get("collect_interval_steps", 1))
        self.update_due = True

    def begin_step(self, step):
        if self.state != "ready":
            raise RuntimeError(
                "previous SpecOPD collection has not completed publication"
            )
        self.update_due = int(step) % self.collection_interval == 0
        self.metrics = {
            "specopd/update_due": int(self.update_due),
            "specopd/trained": 0,
            "specopd/published": 0,
            "specopd/optimizer_updates": 0,
            "specopd/training_trajectories": 0,
            "specopd/bootstrap_published": int(self.bootstrapped),
        }
        if self.update_due:
            self.step, self.state = int(step), "collecting"
        return self.update_due

    def finish_step(self):
        if self.state == "awaiting_publish":
            self.publish()
        elif self.state == "ready":
            self.acknowledge_serving_revision(self.drafter_revision)
        else:
            raise RuntimeError("SpecOPD step ended with an unfinished collection")

    def acknowledge_serving_revision(self, revision):
        server_manager = getattr(self.trainer, "llm_server_manager", None)
        servers = getattr(server_manager, "server_handles", None)
        if not servers:
            raise RuntimeError(
                "SpecOPD requires rollout server handles for serving revision ACKs"
            )
        total = self.spec.max_collect_samples_per_step
        quotas = None
        if total is not None:
            quota, remainder = divmod(total, len(servers))
            quotas = [quota + (i < remainder) for i in range(len(servers))]
        pending = [
            server.speco_set_serving_drafter_revision.remote(
                revision, **({} if quotas is None else {"collection_quota": quotas[i]})
            )
            for i, server in enumerate(servers)
        ]
        replies = self.trainer._ray_get_if_needed(pending)
        if not replies or any(
            not isinstance(reply, dict)
            or reply.get("drafter_revision") != revision
            or type(reply.get("actor_revision")) is not int
            or reply["actor_revision"] < 0
            for reply in replies
        ):
            raise RuntimeError(
                "SpecOPD serving revision was not acknowledged by every replica"
            )
        if (
            quotas is not None
            and [reply.get("collection_quota") for reply in replies] != quotas
        ):
            raise RuntimeError(
                "SpecOPD collection quotas were not acknowledged by every replica"
            )
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
        self.drafter_revision = statuses[0]["revision"]
        self.acknowledge_serving_revision(self.drafter_revision)
        self.bootstrapped = True
        self.metrics["specopd/bootstrap_published"] = 1

    def old_log_prob(self, batch, *, native_trunk=None):
        if self.state != "collecting":
            raise RuntimeError(
                "SpecOPD old-logprob must follow exactly one rollout collection"
            )
        requests, actor_revision = build_requests(
            batch, self.spec, self.drafter_revision
        )
        if actor_revision != self._acknowledged_actor_revision:
            raise RuntimeError(
                "SpecOPD rollout differs from the acknowledged serving actor revision"
            )
        if self._bound_actor_revision and actor_revision < self.actor_revision:
            raise RuntimeError("SpecOPD rollout actor revision regressed")
        # In synchronous V1 the actor cannot update between this
        # rollout and old-logprob. Bind the actual serving revision (not prompt
        # submission global_step, which can be off by one at startup/resume).
        self.actor_revision = actor_revision
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
        metrics = {
            "specopd/model_forward_calls": 0,
            "specopd/logical_rows": 0,
            "specopd/independent_rows": 0,
            "specopd/mfu_unavailable": 1,
            "specopd/actor_revision": actor_revision,
            "specopd/drafter_revision": self.drafter_revision,
        }
        calculate_entropy = self.trainer._speco_oldlogprob_calculate_entropy()
        trunk = (
            tuple(
                (i, request)
                for i, request in enumerate(requests)
                if not request.branches
            )
            if native_trunk is not None
            else ()
        )
        joint_requests = [r for r in requests if r.branches] if trunk else requests
        scored = ()
        if joint_requests:
            started = time.perf_counter()
            output = self.trainer.actor_rollout_wg.compute_joint_old_log_prob(
                joint_requests,
                microbatch_trajectories=self.spec.microbatch_trajectories,
                actor_temperature=self.spec.temperature,
                target_temperature=self.spec.temperature,
                hidden_state_indices=self.feature_indices,
                calculate_entropy=calculate_entropy,
                return_target_logits=False,
                return_target_logz=False,
                capture_empty_trunk_features=False,
                trunk_feature_prefix_only=True,
                max_packed_tokens=self.spec.scoring_token_budget,
                max_mask_elements=self.spec.max_dense_mask_elements,
            )
            scored, measured = unwrap_rank_result(
                self.trainer._ray_get_if_needed(output)
            )
            indexed = {r.sample_id: r for r in scored}
            if len(indexed) != len(scored) or set(indexed) != {
                r.sample_id for r in joint_requests
            }:
                raise RuntimeError(
                    "joint actor RPC lost or duplicated trajectory results"
                )
            metrics["specopd/actor_sp_size"] = measured.get("sp_size", 1)
            for key in ("model_forward_calls", "logical_rows", "independent_rows"):
                metrics[f"specopd/{key}"] = measured[key]
            metrics["timing_s/specopd_joint_oldlogprob"] = time.perf_counter() - started
        else:
            indexed = {}
        if trunk:
            started = time.perf_counter()
            native_results = native_trunk(trunk, calculate_entropy=calculate_entropy)
            if len(native_results) != len(trunk) or {
                r.sample_id for r in native_results
            } != {req.sample_id for _, req in trunk}:
                raise RuntimeError(
                    "native actor RPC lost or duplicated trajectory results"
                )
            indexed.update((result.sample_id, result) for result in native_results)
            metrics["specopd/native_trunk_trajectories"] = len(trunk)
            metrics["timing_s/specopd_native_trunk_oldlogprob"] = (
                time.perf_counter() - started
            )
        results = tuple(indexed[r.sample_id] for r in requests)
        self.metrics.update(metrics)
        self.pending = tuple(
            build_specopd_training_batch(
                req, indexed[req.sample_id], "cpu", self.spec, self.feature_indices
            )
            for req in joint_requests
            if req.branches
        )
        self.state = "scored"
        return make_oldlogprob(batch, results)

    def before_actor_update(self):
        if not self.update_due:
            return
        if self.state != "scored":
            raise RuntimeError(
                "SpecOPD group is not scored or has already been consumed"
            )
        samples, self.pending, self.state = self.pending, None, "updating"
        positions = sum(b["_speco_position_count"] for b in samples)
        proposals = sum(b["_specopd_proposal_anchors"] for b in samples)
        self.metrics["specopd/proposal_anchors"] = proposals
        self.metrics["specopd/trunk_anchors"] = (
            sum(b["_speco_sample_count"] for b in samples) - proposals
        )
        self.trained = False
        if positions >= self.spec.min_scored_positions_per_update:
            self.trainer._speco_set_drafter_global_step()
            self.metrics.update(self.trainer._speco_sync_target_lm_head_weight())
            training_started = time.perf_counter()
            from verl_speco.integration.specopd_training_transport import (
                dispatch_prepared_group,
            )

            statuses = dispatch_prepared_group(
                self.trainer,
                samples,
                int(self.step),
                self.drafter_revision,
                self.feature_indices,
                self.metrics,
                self.spec,
            )
            self.metrics["timing_s/specopd_train_rpc"] = (
                time.perf_counter() - training_started
            )
            for key in (
                "timing_s/specopd_prepare",
                "timing_s/specopd_train_loop",
                "timing_s/specopd_snapshot",
                "timing_s/specopd_release",
                "timing_s/specopd_activation",
                "specopd/prepared_batches",
                "specopd/input_cache_bytes",
                "specopd/train_dp_size",
                "specopd/train_microbatch_trajectories",
            ):
                values = [s.get(key) for s in statuses]
                if values and all(v is not None for v in values):
                    self.metrics[key] = max(values)
            if not statuses:
                raise RuntimeError("SpecOPD drafter ranks returned no optimizer result")
            self.trained = bool(statuses[0]["trained"])
            updates = self.spec.optimizer_updates_per_collection if self.trained else 0
            expected = self.trained, self.drafter_revision + int(self.trained), updates
            if any(
                (s["trained"], s["revision"], s["optimizer_updates"]) != expected
                for s in statuses
            ):
                raise RuntimeError(
                    "SpecOPD drafter ranks disagree about optimizer updates/revision"
                )
            self.metrics["specopd/optimizer_updates"] = updates
        self.metrics["specopd/scored_positions"] = positions
        self.metrics["specopd/trained"] = int(self.trained)
        self.metrics["specopd/training_trajectories"] = (
            len(samples) if self.trained else 0
        )
        self.drafter_revision += int(self.trained)
        self.state = "awaiting_publish"

    def publish(self):
        if self.state != "awaiting_publish":
            return
        if self.trained:
            previous_step = self.trainer.global_steps
            try:
                self.trainer.global_steps = self.step
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
        self.acknowledge_serving_revision(self.drafter_revision)
        self.actor_revision += 1
        self.state = "ready"
        self.metrics["specopd/published"] = int(self.trained)
        ci_result = getattr(self.trainer, "_speco_ci_result", None)
        if ci_result is not None:
            ci_result.record_metrics(self.metrics)
        self.metrics["specopd/published_drafter_revision"] = self.drafter_revision
        if not self.trained:
            self.metrics["drafter/published"] = 0
