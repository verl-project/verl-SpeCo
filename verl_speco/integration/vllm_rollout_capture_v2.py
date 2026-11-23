# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Adapt V2 request indexing and native acceptance counts to feature export."""

import functools
from types import SimpleNamespace as NS


def capture_v2_rows(runner, state, result):
    from verl_speco.integration.vllm_rollout_capture import (
        capture_sampled_rows,
        capture_detail,
    )
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    batch = state.input_batch
    capture_detail(
        "v2_runner_sample requests=%s aux_layers=%s",
        len(batch.req_ids),
        len(state.aux_hidden_states or []),
    )
    plans = runner._speco_capture_plans
    # Skip adapter allocation when no live request intersects its capture window.
    selected = []
    for index, request_id in enumerate(batch.req_ids):
        plan = plans.get(request_id)
        if not plan:
            continue
        start = int(batch.num_computed_tokens_np[index])
        scheduled = int(batch.num_scheduled_tokens[index])
        if (
            scheduled > 0
            and start < int(plan["end"])
            and start + scheduled > int(plan["start"])
        ):
            selected.append((index, request_id, plan))
    if not selected:
        return
    drafts = batch.num_draft_tokens_per_req
    adapter = NS(
        vllm_config=runner.vllm_config,
        input_batch=NS(
            req_ids=batch.req_ids,
            vocab_size=runner.vocab_size,
            num_computed_tokens_cpu=batch.num_computed_tokens_np,
        ),
        query_start_loc=NS(np=batch.query_start_loc_np),
        input_ids=NS(gpu=batch.input_ids),
        use_async_scheduling=runner.vllm_config.scheduler_config.async_scheduling,
        model_config=runner.model_config,
        _get_eagle3_aux_layers_from_config=lambda: get_eagle3_aux_layers_from_config(
            runner.speculative_config
        ),
        get_model=lambda: runner.model,
    )
    adapter._speco_capture_copy_stream = getattr(
        runner, "_speco_capture_copy_stream", None
    )
    adapter._speco_capture_buffer_pool = getattr(
        runner, "_speco_capture_buffer_pool", None
    )
    captured = NS(
        scheduler_output=NS(
            num_scheduled_tokens={
                rid: batch.num_scheduled_tokens[i] for i, rid, _ in selected
            },
            scheduled_spec_decode_tokens={
                rid: range(int(drafts[i]) if drafts is not None else 0)
                for i, rid, _ in selected
            },
        ),
        hidden_states=state.hidden_states,
        aux_hidden_states=state.aux_hidden_states,
    )
    sampler, num_sampled, _ = result
    capture_sampled_rows(
        adapter,
        captured,
        NS(sampled_token_ids=sampler.sampled_token_ids, speco_num_sampled=num_sampled),
        selected=selected,
    )
    runner._speco_capture_copy_stream = getattr(
        adapter, "_speco_capture_copy_stream", None
    )
    runner._speco_capture_buffer_pool = getattr(
        adapter, "_speco_capture_buffer_pool", None
    )


def install_v2_capture():
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ModuleNotFoundError as exc:
        if exc.name in {"vllm.v1.worker.gpu", "vllm.v1.worker.gpu.model_runner"}:
            return
        raise
    if getattr(GPUModelRunner, "_speco_capture_v2_installed", False):
        return
    required = ("add_requests", "_remove_request", "sample_tokens", "sample")
    missing = [
        name for name in required if not callable(getattr(GPUModelRunner, name, None))
    ]
    if missing:
        raise RuntimeError(f"Incompatible V2 capture hooks: {missing}")
    add, remove = GPUModelRunner.add_requests, GPUModelRunner._remove_request
    tokens, sample = GPUModelRunner.sample_tokens, GPUModelRunner.sample

    @functools.wraps(add)
    def wrapped_add(self, scheduler_output, *args, **kwargs):
        result = add(self, scheduler_output, *args, **kwargs)
        if not hasattr(self, "_speco_capture_plans"):
            self._speco_capture_plans = {}
        for request in scheduler_output.scheduled_new_reqs:
            plan = (getattr(request.sampling_params, "extra_args", None) or {}).get(
                "speco_rollout_capture"
            )
            if plan:
                self._speco_capture_plans[request.req_id] = plan
        return result

    @functools.wraps(remove)
    def wrapped_remove(self, req_id, *args, **kwargs):
        getattr(self, "_speco_capture_plans", {}).pop(req_id, None)
        return remove(self, req_id, *args, **kwargs)

    @functools.wraps(tokens)
    def wrapped_tokens(self, *args, **kwargs):
        self._speco_capture_v2_state = self.execute_model_state
        try:
            return tokens(self, *args, **kwargs)
        finally:
            self._speco_capture_v2_state = None

    @functools.wraps(sample)
    def wrapped_sample(self, *args, **kwargs):
        result = sample(self, *args, **kwargs)
        state = getattr(self, "_speco_capture_v2_state", None)
        if state is not None and getattr(self, "_speco_capture_plans", None):
            capture_v2_rows(self, state, result)
        return result

    GPUModelRunner.add_requests = wrapped_add
    GPUModelRunner._remove_request = wrapped_remove
    GPUModelRunner.sample_tokens = wrapped_tokens
    GPUModelRunner.sample = wrapped_sample
    GPUModelRunner._speco_capture_v2_installed = True
    from verl_speco.integration.vllm_rollout_capture import capture_trace

    capture_trace("v2_runner_hooks_installed module=%s", GPUModelRunner.__module__)
