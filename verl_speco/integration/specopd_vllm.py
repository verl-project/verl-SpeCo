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
"""vLLM V1 scheduler proposal capture and explicit trace-header transport.

Only token/identity metadata crosses this channel. Requests opt in through the
native SamplingParams.extra_args field. MRV2 captures verified proposals in the
worker result, so async scheduler placeholders never become training tokens.
"""

from functools import wraps

from verl_speco.integration.specopd_protocol import (
    PLAN_KEY,
    TRACE_KEY,
    SCHEMA,
    ProposalRecord,
    token_digest,
    record_dict,
    encode_bundle,
)


def patch_draft_token_request_tracking(runner):
    """Track opt-in requests on each runner's own handler, including eviction."""
    if getattr(runner, "_specopd_request_tracking_installed", False):
        return True
    add = runner.add_requests
    remove = runner._remove_request

    @wraps(add)
    def added(self, scheduler_output, *args, **kwargs):
        result = add(self, scheduler_output, *args, **kwargs)
        handler = self.draft_tokens_handler
        plans = getattr(handler, "_specopd_plans", None)
        if plans is None:
            plans = handler._specopd_plans = {}
        for request in scheduler_output.scheduled_new_reqs:
            params = getattr(request, "sampling_params", None)
            plan = (getattr(params, "extra_args", None) or {}).get(PLAN_KEY)
            plans.pop(request.req_id, None)
            getattr(handler, "_specopd_closed_windows", set()).discard(request.req_id)
            if plan and plan.get("max_branches", 0) > 0:
                plans[request.req_id] = plan
        return result

    @wraps(remove)
    def removed(self, request_id, *args, **kwargs):
        result = remove(self, request_id, *args, **kwargs)
        getattr(self.draft_tokens_handler, "_specopd_plans", {}).pop(request_id, None)
        getattr(self.draft_tokens_handler, "_specopd_closed_windows", set()).discard(
            request_id
        )
        return result

    runner.add_requests, runner._remove_request = added, removed
    runner._specopd_request_tracking_installed = True
    return True


def install_specopd_draft_token_transport_patch():
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return False  # Legacy runners retain synchronous scheduler capture.
    from verl_speco.integration.specopd_async import patch_result_transport

    patch_draft_token_request_tracking(GPUModelRunner)
    return patch_result_transport(GPUModelRunner)


def _plan(request):
    params = getattr(request, "sampling_params", None)
    return (getattr(params, "extra_args", None) or {}).get(PLAN_KEY)


def _stop_ids(request):
    params = request.sampling_params
    explicit = set(getattr(params, "stop_token_ids", None) or ())
    if getattr(params, "ignore_eos", False):
        return explicit
    explicit.update(getattr(params, "all_stop_token_ids", None) or ())
    eos = getattr(request, "eos_token_id", None)
    if eos is not None:
        explicit.add(eos)
    return explicit


def _capture_active(request, plan, offset):
    limit = plan["max_branches"]
    if limit <= 0 or not plan["start"] <= offset < plan["end"]:
        return False
    return (
        bool(plan.get("reservoir"))
        or len(getattr(request, "_specopd_records", ())) < limit
    )


def snapshot_scheduler_round(scheduler):
    snapshots = {}
    for request_id, request in getattr(scheduler, "requests", {}).items():
        plan = _plan(request)
        if not plan:
            continue
        if getattr(request, "num_output_placeholders", 0):
            raise RuntimeError(
                "SpecOPD capture requires synchronous scheduling without placeholders"
            )
        prefix_len = len(request.all_token_ids)
        if prefix_len < plan["prompt_len"]:
            raise RuntimeError("SpecOPD vLLM prefix is shorter than the prompt")
        selected = _capture_active(request, plan, prefix_len - plan["prompt_len"])
        snapshots[request_id] = (
            request,
            plan,
            list(request.all_token_ids) if selected else [],
            list(getattr(request, "spec_token_ids", ()) or ()) if selected else [],
        )
    return snapshots


def capture_scheduler_round(scheduler_output, model_output, snapshots):
    """Capture before update_from_output mutates requests or releases them."""
    sampled = getattr(model_output, "sampled_token_ids", None)
    mapping = getattr(model_output, "req_id_to_index", {})
    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", {})
    invalid = getattr(scheduler_output, "num_invalid_spec_tokens", None) or {}
    for request_id, scheduled_ids in scheduled.items():
        entry = snapshots.get(request_id)
        if not entry:
            continue
        request, plan, prefix, raw = entry
        if (
            request_id not in mapping
            or sampled is None
            or not len(sampled[mapping[request_id]])
        ):
            continue
        records = getattr(request, "_specopd_records", None)
        if records is None:
            records = request._specopd_records = []
        round_id = int(getattr(request, "_specopd_round", 0))
        request._specopd_round = round_id + 1
        offset = len(prefix) - plan["prompt_len"]
        if not _capture_active(request, plan, offset):
            continue
        if not raw:
            # vLLM may expose only the scheduled list; do not claim an unseen
            # tail was captured if the scheduler has shortened this proposal.
            raw = list(scheduled_ids)
        if len(raw) != plan["proposal_width"] or any(
            type(i) is not int or i < 0 for i in raw
        ):
            raise RuntimeError(
                "SpecOPD requires a complete fixed-width proposal before scheduler truncation: "
                f"raw_width={len(raw)}, scheduled_width={len(scheduled_ids)}, "
                f"expected_width={plan['proposal_width']}, "
                f"invalid_types={sorted({type(i).__name__ for i in raw if type(i) is not int})}, "
                f"negative_tokens={sum(1 for i in raw if i < 0)}, offset={offset}"
            )
        stops = _stop_ids(request)
        for index, token in enumerate(raw):
            if token in stops:
                raw = raw[: index + 1]
                break
        generated = sampled[mapping[request_id]]
        verified = min(len(raw), len(scheduled_ids) - invalid.get(request_id, 0))
        accepted = min(len(raw), len(generated) - 1)
        if (
            not 0 <= accepted <= verified
            or list(generated[:accepted]) != raw[:accepted]
        ):
            raise RuntimeError(
                "SpecOPD accepted tokens differ from the verified proposal"
            )
        record = ProposalRecord(
            f"{plan['request_id']}:{round_id}",
            len(prefix),
            tuple(raw),
            token_digest(prefix),
            round_id,
            accepted,
            verified,
        )
        if plan.get("reservoir"):
            from verl_speco.integration.specopd_anchors import retain_proposal

            retain_proposal(request, plan, record_dict(record))
        else:
            records.append(record_dict(record))


def install_specopd_vllm_patch():
    """Install idempotent wrappers; disabled requests keep upstream behavior."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from verl_speco.integration.vllm_runtime import patch_vllm_request_acceptance_stats

    if getattr(Scheduler, "_specopd_capture_installed", False):
        return True
    if not patch_vllm_request_acceptance_stats():
        raise RuntimeError("SpecOPD requires the request-output metadata transport")
    initialize = Scheduler.__init__
    schedule = Scheduler.schedule
    update = Scheduler.update_from_output

    @wraps(initialize)
    def initialized(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        from verl_speco.specopd_config import spec_opd_enabled
        from verl_speco.integration.vllm_runtime import _load_env_drafter_config

        if (
            getattr(self.scheduler_config, "async_scheduling", False)
            and not getattr(self, "use_v2_model_runner", False)
            and spec_opd_enabled(_load_env_drafter_config().get("training", {}))
        ):
            raise RuntimeError(
                "SpecOPD async scheduling requires the active MRV2 model runner; "
                "enable VLLM_USE_V2_MODEL_RUNNER=1 or disable async scheduling"
            )

    @wraps(schedule)
    def scheduled(self, *args, **kwargs):
        if getattr(getattr(self, "scheduler_config", None), "async_scheduling", False):
            return schedule(self, *args, **kwargs)
        snapshots = snapshot_scheduler_round(self)
        result = schedule(self, *args, **kwargs)
        if snapshots and getattr(result, "scheduled_spec_decode_tokens", None):
            # SchedulerOutput is local to the scheduler; retain the snapshot
            # here rather than sending mutable request objects to a worker.
            pending = getattr(self, "_specopd_schedules", None)
            if pending is None:
                pending = self._specopd_schedules = {}
            pending[id(result)] = snapshots
            if len(pending) > 8:
                raise RuntimeError("SpecOPD schedule snapshots were not consumed")
        return result

    @wraps(update)
    def updated(self, scheduler_output, model_output):
        from verl_speco.integration.specopd_async import (
            VERIFIED_KEY,
            snapshot_result_round,
        )

        snapshots = getattr(self, "_specopd_schedules", {}).pop(
            id(scheduler_output), {}
        )
        result_transport = hasattr(model_output, VERIFIED_KEY) or getattr(
            getattr(self, "scheduler_config", None), "async_scheduling", False
        )
        if result_transport:
            snapshots = snapshot_result_round(self, scheduler_output, model_output)
        # Include newly admitted requests so even no-proposal responses carry
        # an explicit empty bundle instead of silently losing the channel.
        for rid, request in getattr(self, "requests", {}).items():
            if not result_transport and rid not in snapshots and _plan(request):
                snapshots[rid] = (
                    request,
                    _plan(request),
                    list(request.all_token_ids),
                    [],
                )
        capture_scheduler_round(scheduler_output, model_output, snapshots)
        results = update(self, scheduler_output, model_output)
        for outputs in results.values():
            for output in getattr(outputs, "outputs", ()) or ():
                if not (
                    getattr(output, "finished", False)
                    or getattr(output, "finish_reason", None) is not None
                ):
                    continue
                entry = snapshots.get(output.request_id)
                if entry:
                    request, plan, _, _ = entry
                    bundle = {
                        "schema": SCHEMA,
                        "request_id": plan["request_id"],
                        "actor_revision": plan["actor_revision"],
                        "drafter_revision": plan["drafter_revision"],
                        "proposal_temperature": plan["proposal_temperature"],
                        "records": getattr(request, "_specopd_records", []),
                        "collection_enabled": plan["max_branches"] > 0,
                        "window": [plan["start"], plan["end"]],
                    }
                    headers = dict(getattr(output, "trace_headers", None) or {})
                    headers[TRACE_KEY] = encode_bundle(bundle, plan["max_wire_bytes"])
                    output.trace_headers = headers
        return results

    Scheduler.__init__ = initialized
    Scheduler.schedule, Scheduler.update_from_output = scheduled, updated
    Scheduler._specopd_capture_installed = True
    return True
