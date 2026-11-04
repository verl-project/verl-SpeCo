# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Existing CPU counters produce weighted acceptance without I/O or tensor reads."""

import sys
from types import SimpleNamespace

import pytest

from verl_speco.integration.acceptance_metrics import (
    bridge_request_acceptance_counters,
    request_acceptance_counters,
)
from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin


def private(rounds, accepted):
    return {
        "_speco_vllm_request_verify_rounds": rounds,
        "_speco_vllm_request_accepted_tokens": accepted,
    }


@pytest.mark.parametrize(
    "rounds,accepted",
    [
        (None, 3),
        (0, 2),
        (-1, 2),
        (2, -1),
        (float("nan"), 1),
        (2, float("inf")),
        ("bad", 2),
    ],
)
def test_invalid_counts_do_not_create_false_measurements(rounds, accepted):
    fields = private(rounds, accepted)
    assert request_acceptance_counters(fields) is None
    assert bridge_request_acceptance_counters(fields) is False
    assert "spec_num_verify_steps" not in fields


def test_native_multiturn_totals_take_precedence_without_double_counting():
    fields = {
        **private(3, 4),
        "spec_num_verify_steps": 10,
        "spec_num_accepted_tokens": 9,
    }
    assert request_acceptance_counters(fields) == (10, 9)


def test_server_bridge_replaces_aggregate_fallback_with_exact_request():
    fields = {
        **private(3, 4),
        "spec_num_verify_steps": 99,
        "spec_num_accepted_tokens": 99,
        "_speco_vllm_request_draft_tokens": 21,
        "global_steps": 12,
    }
    assert bridge_request_acceptance_counters(fields)
    assert request_acceptance_counters(fields) == (3, 4)
    assert fields["spec_num_draft_tokens"] == 21
    assert fields["_verl_request_mean_accept_len"] == pytest.approx(1 + 4 / 3)
    assert fields["global_steps"] == 12


def test_scalar_like_object_is_not_read():
    class DeviceValue:
        def __float__(self):
            pytest.fail("must not read a device scalar")

    assert request_acceptance_counters(private(DeviceValue(), 3)) is None


def test_trainer_weights_rounds_excludes_padding_and_reports_coverage(monkeypatch):
    fields = [
        private(2, 8),  # mean=5
        SimpleNamespace(data=private(8, 0)),  # mean=1
        {**private(1, 99), "spec_num_verify_steps": 4, "spec_num_accepted_tokens": 4},
        {"_verl_request_mean_accept_len": 7},  # mean alone cannot reconstruct weight
        private(1000, 1000),  # padding must not be counted
    ]
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(
            kv_batch_get=lambda **kw: {"extra_fields": fields},
        ),
    )
    trainer = SimpleNamespace(
        _speco_v1_spec_decode_sidecar_metrics=lambda: pytest.fail("no file fallback")
    )
    batch = SimpleNamespace(
        keys=list(range(5)),
        partition_id="p",
        tags=[{}, {}, {}, {}, {"is_padding": True}],
    )
    result = SpecoV1Mixin._speco_v1_spec_decode_metrics(trainer, batch)
    assert result["drafter/spec_decode/mean_acceptance_length"] == pytest.approx(
        1 + 12 / 14
    )
    assert result["drafter/spec_decode/num_verify_steps"] == 14
    assert result["drafter/spec_decode/num_accepted_tokens"] == 12
    assert result["drafter/spec_decode/request_stats_coverage"] == 0.75
    assert result["drafter/spec_decode/requests_missing_stats"] == 1


def test_native_zero_counters_fall_back_to_request_counters():
    assert request_acceptance_counters(
        {**private(2, 0), "spec_num_verify_steps": 0, "spec_num_accepted_tokens": 0}
    ) == (2, 0)


def test_no_valid_request_stats_preserves_existing_fallback(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(
            kv_batch_get=lambda **kw: {"extra_fields": [{}]},
        ),
    )
    marker = {"sidecar": 1}
    trainer = SimpleNamespace(_speco_v1_spec_decode_sidecar_metrics=lambda: marker)
    assert (
        SpecoV1Mixin._speco_v1_spec_decode_metrics(
            trainer, SimpleNamespace(keys=[1], partition_id="p", tags=[{}])
        )
        is marker
    )


def test_engine_aggregate_is_not_counted_as_an_exact_request():
    fields = {
        "spec_num_verify_steps": 100,
        "spec_num_accepted_tokens": 300,
        "_speco_acceptance_engine_fallback": True,
    }
    assert request_acceptance_counters(fields) is None
    fields.update(private(2, 3))
    assert bridge_request_acceptance_counters(fields)
    assert "_speco_acceptance_engine_fallback" not in fields
    assert request_acceptance_counters(fields) == (2, 3)


def test_gpu_style_staged_summary_reaches_native_fields_and_trainer(monkeypatch):
    import asyncio

    from verl_speco.integration import vllm_runtime as runtime

    class Upstream:
        async def generate(self, *args, **kwargs):
            request = kwargs["request_id"]
            runtime._stage_vllm_request_stats_for_rollout_output(
                request,
                {
                    request: {
                        "verify_rounds": 4,
                        "draft_tokens": 28,
                        "accepted_tokens": 6,
                        "invalid_spec_tokens": 0,
                        "completion_index": 0,
                    },
                },
                [request],
            )
            return SimpleNamespace(extra_fields={"global_steps": 12}, token_ids=[1, 2])

    class Server(runtime._SpecoVLLMHttpServerMixin, Upstream):
        pass

    output = asyncio.run(Server().generate(request_id="acceptance-e2e"))
    assert output.extra_fields["spec_num_verify_steps"] == 4
    assert output.extra_fields["spec_num_accepted_tokens"] == 6
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(
            kv_batch_get=lambda **kw: {
                "extra_fields": [SimpleNamespace(data=output.extra_fields)]
            },
        ),
    )
    trainer = SimpleNamespace(
        _speco_v1_spec_decode_sidecar_metrics=lambda: pytest.fail("unexpected file I/O")
    )
    metrics = SpecoV1Mixin._speco_v1_spec_decode_metrics(
        trainer, SimpleNamespace(keys=[1], partition_id="p", tags=[{}])
    )
    assert metrics["drafter/spec_decode/mean_acceptance_length"] == 2.5
    assert metrics["drafter/spec_decode/request_stats_coverage"] == 1
