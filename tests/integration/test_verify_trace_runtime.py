# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import json
import sys
import types
from types import SimpleNamespace

import pytest

import verl_speco.integration.vllm_runtime as runtime
from verl_speco.integration.verify_trace import TRACE_KEY, validate_verify_trace


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("config_source", ["environment", "server_class"])
@pytest.mark.parametrize("invalid_tokens", [None, {}, {"r": 1}])
@pytest.mark.parametrize("layout", ["dflash", "dspark"])
def test_scheduler_patch_transports_trace_on_finished_request(
    monkeypatch, capture, config_source, invalid_tokens, layout
):
    request = SimpleNamespace(
        prompt_token_ids=[1, 2], _output_token_ids=[], is_finished=lambda: False
    )

    class Scheduler:
        num_sampled_tokens_per_step = 1
        num_spec_tokens = 2

        def update_from_output(self, scheduled, output):
            self.requests["r"]._output_token_ids.extend(output.sampled_token_ids[0][:2])
            self.make_spec_decoding_stats(None, 2, 2, {}, "r")
            self.requests.pop("r")
            return {
                0: SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            request_id="r", finished=True, trace_headers=None
                        )
                    ]
                )
            }

        def make_spec_decoding_stats(self, *args):
            return None

    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler
    processor_module = types.ModuleType("vllm.v1.engine.output_processor")
    processor_module.OutputProcessor = type("OutputProcessor", (), {})
    processor_module.RequestState = type("RequestState", (), {})
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, processor_module.__name__, processor_module)
    config = {
        "speculative_algorithm": layout.upper(),
        "training": {"drafter_opd": {"capture_verify_trace": capture}},
    }
    monkeypatch.delenv(runtime.SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)
    if config_source == "server_class":
        resolved, source = runtime._ensure_vllm_server_drafter_env(
            {"rollout": {}}, json.dumps(config)
        )
        assert source == config_source
        assert resolved == config
    else:
        monkeypatch.setenv(runtime.SPECO_DRAFTER_CONFIG_ENV, json.dumps(config))
    assert runtime.patch_vllm_request_acceptance_stats()
    scheduler = Scheduler()
    scheduler.requests = {"r": request}
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"r": 3},
        scheduled_spec_decode_tokens={"r": [-1] * (3 if invalid_tokens else 2)},
        num_invalid_spec_tokens=invalid_tokens,
    )
    output = SimpleNamespace(req_id_to_index={"r": 0}, sampled_token_ids=[[3, 4, 5]])
    result = scheduler.update_from_output(scheduled, output)[0].outputs[0]
    summary = json.loads(
        result.trace_headers[runtime.SPECO_VLLM_REQUEST_STATS_TRACE_HEADER]
    )
    if capture:
        trace = summary[TRACE_KEY]
        validate_verify_trace(trace, layout=layout, capacity=2)
        assert trace["steps"][0]["drafts"] == 2
        assert trace["steps"][0]["accepted"] == 2
        assert trace["steps"][0]["commit"] == 2
        assert trace["steps"][0]["truncated"]
        fields = runtime._vllm_request_accept_stats_to_scalar_extra_fields(
            SimpleNamespace(
                request_id="r", _speco_vllm_request_accept_stats={"r": summary}
            )
        )
        assert fields[TRACE_KEY] == trace
    else:
        assert TRACE_KEY not in summary
