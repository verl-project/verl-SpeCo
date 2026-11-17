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
native SamplingParams.extra_args field. Async placeholder scheduling is rejected
because a committed prefix cannot be inferred from its speculative placeholders.
"""

from functools import wraps
import time

from verl_speco.transport.specopd_protocol import (
    PLAN_KEY,
    TRACE_KEY,
    SCHEMA,
    ProposalRecord,
    token_digest,
    record_dict,
    encode_bundle,
    decode_bundle,
)

_COMPLETED = {}


class _DraftCopyBatch:
    """Select native token copying without changing the real batch's grammar flag."""

    has_structured_output_reqs = True

    def __init__(self, batch):
        self._batch = batch

    def __getattr__(self, name):
        return getattr(self._batch, name)


def patch_draft_token_handler(handler):
    """MRV2 otherwise sends only -1 placeholders to the synchronous scheduler."""
    if getattr(handler, "_specopd_copy_installed", False):
        return True
    original = handler.set_draft_tokens
    original_get = handler.get_draft_tokens

    @wraps(original)
    def copied(self, input_batch, draft_tokens):
        from verl_speco.integration.vllm_runtime import _load_env_drafter_config
        from verl_speco.opd_method import spec_opd_enabled

        self._specopd_real_draft_ids = None
        collecting = getattr(self, "_specopd_collecting_requests", set())
        if (spec_opd_enabled(_load_env_drafter_config().get("training", {}))
                and any(rid in collecting for rid in input_batch.req_ids)):
            if getattr(getattr(draft_tokens, "device", None), "type", None) == "npu":
                # Ascend MRV2 constructs this CUDA helper under a native shim,
                # but its transport setter runs outside that shim. Copy these
                # small ID rows synchronously rather than calling CUDA streams.
                result = original(self, input_batch, draft_tokens)
                self._specopd_real_draft_ids = draft_tokens.detach().cpu().tolist()
                return result
            # Reuse the backend's existing async copy/event/record_stream path.
            # This proxy is consumed only by the token transport setter; neither
            # the scheduler nor sampling/grammar code sees a modified batch.
            input_batch = _DraftCopyBatch(input_batch)
        return original(self, input_batch, draft_tokens)

    @wraps(original_get)
    def retrieved(self):
        result = original_get(self)
        ids = getattr(self, "_specopd_real_draft_ids", None)
        if ids is not None and result is not None:
            result.draft_token_ids = ids
        return result

    handler.set_draft_tokens = copied
    handler.get_draft_tokens = retrieved
    handler._specopd_copy_installed = True
    return True


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
        collecting = getattr(handler, "_specopd_collecting_requests", None)
        if collecting is None:
            collecting = handler._specopd_collecting_requests = set()
        for request in scheduler_output.scheduled_new_reqs:
            params = getattr(request, "sampling_params", None)
            plan = (getattr(params, "extra_args", None) or {}).get(PLAN_KEY)
            collecting.discard(request.req_id)
            if plan and plan.get("max_branches", 0) > 0:
                collecting.add(request.req_id)
        return result

    @wraps(remove)
    def removed(self, request_id, *args, **kwargs):
        result = remove(self, request_id, *args, **kwargs)
        getattr(self.draft_tokens_handler, "_specopd_collecting_requests", set()).discard(request_id)
        return result

    runner.add_requests, runner._remove_request = added, removed
    runner._specopd_request_tracking_installed = True
    return True


def install_specopd_draft_token_transport_patch():
    try:
        from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
    except ImportError:
        # MRV1 already returns real draft IDs through take_draft_token_ids.
        return False
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    patch_draft_token_request_tracking(GPUModelRunner)
    return patch_draft_token_handler(DraftTokensHandler)


def stage_completed(request_id, wire):
    now = time.monotonic()
    for key, (_, timestamp) in list(_COMPLETED.items()):
        if now - timestamp > 300:
            _COMPLETED.pop(key, None)
    if len(_COMPLETED) >= 4096:
        raise RuntimeError("SpecOPD completed request transport is not being drained")
    _COMPLETED[str(request_id)] = (wire, now)


def pop_completed(request_id):
    item = _COMPLETED.pop(str(request_id), None)
    return item[0] if item else None


def _plan(request):
    params = getattr(request, "sampling_params", None)
    return (getattr(params, "extra_args", None) or {}).get(PLAN_KEY)


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
        selected = (
            plan["start"] <= prefix_len - plan["prompt_len"] < plan["end"]
            and len(getattr(request, "_specopd_records", ())) < plan["max_branches"]
        )
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
        if (
            not plan["start"] <= offset < plan["end"]
            or len(records) >= plan["max_branches"]
        ):
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
        stops = set(plan.get("eos_token_ids", []))
        for index, token in enumerate(raw):
            if token in stops:
                raw = raw[: index + 1]
                break
        accepted = min(len(raw), max(0, len(sampled[mapping[request_id]]) - 1))
        record = ProposalRecord(
            f"{plan['request_id']}:{round_id}",
            len(prefix),
            tuple(raw),
            token_digest(prefix),
            round_id,
            accepted,
        )
        records.append(record_dict(record))


def install_specopd_vllm_patch():
    """Install idempotent wrappers; disabled requests keep upstream behavior."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState

    if getattr(Scheduler, "_specopd_capture_installed", False):
        return True
    schedule = Scheduler.schedule
    update = Scheduler.update_from_output
    process = OutputProcessor.process_outputs
    make_output = RequestState.make_request_output

    @wraps(schedule)
    def scheduled(self, *args, **kwargs):
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
        snapshots = getattr(self, "_specopd_schedules", {}).pop(
            id(scheduler_output), {}
        )
        # Include newly admitted requests so even no-proposal responses carry
        # an explicit empty bundle instead of silently losing the channel.
        for rid, request in getattr(self, "requests", {}).items():
            if rid not in snapshots and _plan(request):
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
                    }
                    headers = dict(getattr(output, "trace_headers", None) or {})
                    headers[TRACE_KEY] = encode_bundle(bundle, plan["max_wire_bytes"])
                    output.trace_headers = headers
        return results

    @wraps(process)
    def processed(self, engine_core_outputs, *args, **kwargs):
        for output in engine_core_outputs:
            wire = (getattr(output, "trace_headers", None) or {}).get(TRACE_KEY)
            state = getattr(self, "request_states", {}).get(output.request_id)
            if wire is not None and state is not None:
                decode_bundle(wire)
                state._specopd_wire = wire
        return process(self, engine_core_outputs, *args, **kwargs)

    @wraps(make_output)
    def made(self, *args, **kwargs):
        output = make_output(self, *args, **kwargs)
        wire = getattr(self, "_specopd_wire", None)
        if output is not None and wire is not None:
            bundle = decode_bundle(wire)
            stage_completed(bundle["request_id"], wire)
            del self._specopd_wire
        return output

    Scheduler.schedule, Scheduler.update_from_output = scheduled, updated
    OutputProcessor.process_outputs, RequestState.make_request_output = processed, made
    Scheduler._specopd_capture_installed = True
    return True
