# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""CPU contracts for vLLM's opt-in rollout hidden-state bridge."""

import asyncio
import sys
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from verl_speco.integration import vllm_rollout_capture as capture
from verl_speco.trainer.scheduler import (
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterScheduleConfig,
    DrafterScheduler,
)


def test_tp_nonleader_does_not_access_requests_or_export(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", NS(get_tp_group=lambda: NS(rank_in_group=1))
    )
    runner = NS(vllm_config=NS(parallel_config=NS(tensor_parallel_size=2)))
    capture.capture_sampled_rows(runner, None, None)


def test_tp_leader_rejects_sharded_hidden(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", NS(get_tp_group=lambda: NS(rank_in_group=0))
    )
    runner, state = runner_and_state([plan()], [2], [[1, 2]], [0])
    runner.vllm_config = NS(parallel_config=NS(tensor_parallel_size=2))
    runner.model_config.hf_config.hidden_size = 4
    with pytest.raises(RuntimeError, match="full-width"):
        capture.capture_sampled_rows(runner, state, None)


def test_tp_configuration_allowed():
    capture.validate_config(
        {
            "enable": True,
            "enable_drafter_training": True,
            "training": {capture.CAPTURE_FLAG: True},
        },
        {"tensor_model_parallel_size": 2, "enable_prefix_caching": False},
        {"async_scheduling": False},
    )


@pytest.mark.parametrize("tp,dp,dcp", [(2, 1, 1), (1, 2, 1), (2, 2, 1), (2, 1, 2)])
def test_tp_leader_exports_full_hidden_once(directory, monkeypatch, tp, dp, dcp):
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", NS(get_tp_group=lambda: NS(rank_in_group=0))
    )
    p = plan(end=3)
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3]], [0])
    runner.vllm_config = NS(
        parallel_config=NS(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
        )
    )
    runner.model_config.hf_config.hidden_size = 2
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[4]]))
    )
    sample = capture.load_sample(p, [1, 2, 3], [4])
    assert sample["hidden_states"].shape == (1, 1, 6)
    assert sample["hidden_positions"].tolist() == [[2]]


@pytest.mark.parametrize("shared", [False, True])
def test_cross_host_export_requires_shared_root(directory, monkeypatch, shared):
    monkeypatch.delenv("SPECO_CAPTURE_EXPORT_ROOT", raising=False)
    if shared:
        monkeypatch.setenv("SPECO_CAPTURE_EXPORT_ROOT", str(directory))
    p = plan(
        end=3,
        export_host="another-host",
        shared_export=shared,
        export_root=str(directory.resolve()),
    )
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3]], [0])
    sampler = NS(sampled_token_ids=torch.tensor([[4]]))
    if not shared:
        with pytest.raises(RuntimeError, match="shared filesystem"):
            capture.capture_sampled_rows(runner, state, sampler)
    else:
        capture.capture_sampled_rows(runner, state, sampler)
        assert capture.load_sample(p, [1, 2, 3], [4]) is not None


def test_export_root_mismatch_fails_before_write(directory):
    p = plan(export_root="/wrong/shared/root")
    runner, state = runner_and_state([p], [2], [[3]], [0])
    with pytest.raises(RuntimeError, match="roots do not match"):
        capture.capture_sampled_rows(runner, state, None)


@pytest.mark.parametrize(
    "key", ["pipeline_model_parallel_size", "prefill_context_parallel_size"]
)
def test_unsupported_sharding_rejected_at_configuration(key):
    with pytest.raises(ValueError, match=key):
        capture.validate_config(
            {
                "enable": True,
                "enable_drafter_training": True,
                "training": {capture.CAPTURE_FLAG: True},
            },
            {key: 2, "enable_prefix_caching": False},
            {"async_scheduling": False},
        )


@pytest.fixture
def directory(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    yield tmp_path
    if capture._EXPORT_QUEUE is not None:
        capture._EXPORT_QUEUE.close()
        capture._EXPORT_QUEUE = None


def plan(**overrides):
    return {
        "start": 2,
        "end": 6,
        "min_rows": 1,
        "sample_id": "a" * 32,
        "layout": "eagle3_aux_plus_last",
        "global_step": 1,
        "replica_rank": 0,
        **overrides,
    }


def test_rollout_export_has_explicit_staging_limits(directory):
    queue = capture.export_queue()
    assert queue.max_bytes == 512 << 20
    assert queue.max_pending == 256


def runner_and_state(plans, starts, inputs, drafts):
    requests = [f"r{i}" for i in range(len(plans))]
    offsets = [0]
    for item in inputs:
        offsets.append(offsets[-1] + len(item))
    hidden = torch.arange(offsets[-1] * 2, dtype=torch.float32).reshape(-1, 2)
    runner = NS(
        input_batch=NS(
            req_ids=requests, vocab_size=100, num_computed_tokens_cpu=starts
        ),
        requests={
            rid: NS(
                sampling_params=NS(extra_args={capture.CAPTURE_KEY: p} if p else {})
            )
            for rid, p in zip(requests, plans)
        },
        query_start_loc=NS(np=offsets),
        input_ids=NS(gpu=torch.tensor(sum(inputs, []))),
        use_async_scheduling=False,
        model_config=NS(hf_config=NS(num_hidden_layers=8)),
        _get_eagle3_aux_layers_from_config=lambda: (1, 3),
    )
    state = NS(
        hidden_states=hidden + 200,
        aux_hidden_states=[hidden, hidden + 100],
        scheduler_output=NS(
            num_scheduled_tokens={r: len(v) for r, v in zip(requests, inputs)},
            scheduled_spec_decode_tokens={r: [0] * n for r, n in zip(requests, drafts)},
        ),
    )
    return runner, state


def open_plan(p):
    path = capture.plan_directory(p)
    path.mkdir()
    (path / "live").touch()


@pytest.mark.parametrize(
    "count,expected",
    [(1, 1), (2, 2), (4, 4)],
)
def test_rejection_rows(count, expected):
    assert capture.confirmed_row_count(4, 3, count) == expected


def test_chunked_prefill_rows_are_inputs():
    assert capture.confirmed_row_count(12, 0, 0) == 12


def test_partial_acceptance_and_buffer_ownership(directory):
    p = plan(end=4)
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3, 4, 99, 98]], [3])
    expected = torch.cat(
        [h[:2] for h in state.aux_hidden_states] + [state.hidden_states[:2]], -1
    ).clone()
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[4, 5, -1, -1]]))
    )
    for h in state.aux_hidden_states + [state.hidden_states]:
        h.fill_(-999)
    result = capture.load_sample(p, [1, 2, 3], [4, 5])
    assert torch.equal(result["hidden_states"][0], expected)
    assert result["hidden_positions"].tolist() == [[2, 3]]
    assert result["hidden_layer_ids"] == [1, 3]
    capture.cleanup(p)
    assert not capture.plan_directory(p).exists()


def test_published_reference_window_is_not_recaptured(directory, monkeypatch):
    p = plan(end=3, window_store=True)
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3]], [0])
    state.hidden_states = state.hidden_states.bfloat16()
    state.aux_hidden_states = [tensor.bfloat16() for tensor in state.aux_hidden_states]
    sampler = NS(sampled_token_ids=torch.tensor([[4]]))
    capture.capture_sampled_rows(runner, state, sampler)
    root = capture.plan_directory(p)
    from verl_speco.producer.feature_export import wait_for_exports

    wait_for_exports(root)
    assert (root / "window.ready").exists()
    monkeypatch.setattr(
        capture.export_queue(),
        "reserve",
        lambda _: pytest.fail("recaptured published window"),
    )
    state.aux_hidden_states[0].fill_(-999)
    capture.capture_sampled_rows(runner, state, sampler)


def test_mixed_requests_do_not_cross_rows(directory):
    first, second = (
        plan(end=3),
        plan(sample_id="b" * 32, start=4, end=5, layout="dflash_aux"),
    )
    for p in (first, second):
        open_plan(p)
    runner, state = runner_and_state(
        [first, None, second], [2, 0, 4], [[3], [90, 91], [5]], [0, 0, 0]
    )
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[4], [92], [6]]))
    )
    a = capture.load_sample(first, [1, 2, 3], [4])
    b = capture.load_sample(second, [1, 2, 3, 4, 5], [6])
    assert a["hidden_states"].shape == (1, 1, 6)
    assert b["hidden_states"].shape == (1, 1, 4)
    assert b["hidden_states"][0, 0, 0] == state.aux_hidden_states[0][3, 0]


def test_stop_truncation_excludes_extra_verified_rows(directory):
    p = plan()
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3, 4, 5, 6]], [3])
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[4, 5, 6, 7]]))
    )
    sample = capture.load_sample(p, [1, 2, 3], [4])
    assert sample["hidden_positions"].tolist() == [[2]]


def test_missing_or_wrong_tokens_fail_closed(directory):
    p = plan(end=4)
    open_plan(p)
    with pytest.raises(RuntimeError, match="missing rows"):
        capture.load_sample(p, [1, 2, 3], [4, 5])
    runner, state = runner_and_state([p], [2], [[3, 99]], [0])
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[5]]))
    )
    with pytest.raises(RuntimeError, match="token alignment"):
        capture.load_sample(p, [1, 2, 3], [4, 5])


def test_cancelled_request_does_not_write(directory):
    p = plan()
    runner, state = runner_and_state([p], [2], [[3]], [0])
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[4]]))
    )
    assert list(directory.iterdir()) == []


def test_window_minimum_and_determinism():
    training = {
        "hidden_state_window_mode": "random",
        "hidden_state_window_tokens_per_sample": 4,
    }
    kwargs = dict(
        prompt_len=3, max_tokens=10, request_id="r", global_step=5, replica_rank=0
    )
    a, b = capture.make_plan(training, **kwargs), capture.make_plan(training, **kwargs)
    assert (a["start"], a["end"]) == (b["start"], b["end"])
    assert 2 <= a["start"] < a["end"] <= 12
    assert a["sample_id"] != b["sample_id"]


def test_inactive_window_does_not_access_sampler_or_hidden(directory):
    p = plan(start=100, end=104)
    runner, state = runner_and_state([p], [2], [[3, 4]], [1])
    state.aux_hidden_states = None
    capture.capture_sampled_rows(runner, state, object())
    assert list(directory.iterdir()) == []


def test_short_response_skips_pending_exports(directory, monkeypatch):
    from verl_speco.producer import feature_export

    p = plan(end=6, min_rows=4)
    open_plan(p)
    (capture.plan_directory(p) / "unfinished.pending").touch()

    def unexpected_wait(*args):
        pytest.fail("A short response must not wait for export")

    monkeypatch.setattr(feature_export, "wait_for_exports", unexpected_wait)
    assert capture.load_sample(p, [1, 2, 3], [4, 5]) is None


def test_capacity_exhaustion_discards_whole_sample(directory):
    from verl_speco.producer.feature_export import FeatureExportQueue

    capture._EXPORT_QUEUE = FeatureExportQueue(max_bytes=1)
    p = plan(end=4)
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3, 4]], [1])
    capture.capture_sampled_rows(
        runner, state, NS(sampled_token_ids=torch.tensor([[4, 5]]))
    )
    assert (capture.plan_directory(p) / "skipped").exists()
    assert capture.load_sample(p, [1, 2, 3], [4, 5]) is None


@pytest.mark.parametrize(
    "engine",
    [
        {"speculative_config": {"method": "extract_hidden_states"}},
        {"kv_transfer_config": {"kv_connector": "ExampleHiddenStatesConnector"}},
    ],
)
def test_connector_extractor_conflicts_are_rejected(engine):
    with pytest.raises(ValueError, match="cannot use"):
        capture.validate_config({"training": {capture.CAPTURE_FLAG: True}}, {}, engine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_cuda_copy_survives_immediate_graph_buffer_reuse(directory):
    import threading
    from verl_speco.producer.feature_export import FeatureExportQueue

    gate = threading.Event()

    class GatedQueue(FeatureExportQueue):
        def submit_reserved(self, directory, payload, *, size, wait, owners):
            def delayed():
                assert gate.wait(10)
                wait()

            super().submit_reserved(
                directory, payload, size=size, wait=delayed, owners=owners
            )

    capture._EXPORT_QUEUE = GatedQueue()
    p = plan(end=4)
    open_plan(p)
    runner, state = runner_and_state([p], [2], [[3, 4, 99]], [2])
    expected = torch.cat(
        [h[:2] for h in state.aux_hidden_states] + [state.hidden_states[:2]], -1
    )
    state.hidden_states = state.hidden_states.cuda()
    state.aux_hidden_states = [h.cuda() for h in state.aux_hidden_states]
    runner.input_ids.gpu = runner.input_ids.gpu.cuda()
    try:
        capture.capture_sampled_rows(
            runner,
            state,
            NS(sampled_token_ids=torch.tensor([[4, 5, -1]], device="cuda")),
        )
        # No synchronize before overwrite: this reproduces graph/proposer reuse.
        for h in state.aux_hidden_states + [state.hidden_states]:
            h.fill_(-999)
        assert list(capture.plan_directory(p).glob("*.pending"))
        assert not list(capture.plan_directory(p).glob("*.pt"))
    finally:
        gate.set()
    sample = capture.load_sample(p, [1, 2, 3], [4, 5])
    assert torch.equal(sample["hidden_states"][0], expected)


def test_configuration_is_explicit_and_exclusive():
    drafter = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "EAGLE3",
        "training": {capture.CAPTURE_FLAG: True},
    }
    capture.validate_config(
        drafter, {"enable_prefix_caching": False}, {"async_scheduling": False}
    )
    with pytest.raises(ValueError, match="async_scheduling"):
        capture.validate_config(drafter, {"enable_prefix_caching": False}, {})
    with pytest.raises(ValueError, match="prefix"):
        capture.validate_config(drafter, {}, {"async_scheduling": False})
    drafter["training"]["collect_hidden_states_from_old_logprob"] = True
    with pytest.raises(ValueError, match="exclusive"):
        capture.validate_config(drafter, {}, {})


def test_vllm_source_uses_shared_scheduler_and_routing():
    scheduler = DrafterScheduler()
    source = DrafterCollectionSource.VLLM
    decision = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=5, source=source, drafter_enabled=True, source_enabled=True
        ),
        DrafterScheduleConfig.from_mapping({"collect_interval_steps": 5}),
    )
    assert decision.collect
    assert decision.metrics()["drafter/collection_plan_source"] == 4
    payload = scheduler.prepare_collection_payload(
        source=source,
        samples=[{"replica_rank": 1}],
        owner_count=2,
        dispatch_bucket_count=2,
        raw_samples=1,
        collection_id=decision.collection_id,
    )
    assert not payload.buckets[0]
    assert payload.buckets[1] == [{"replica_rank": 1}]


@pytest.mark.parametrize("serving_step", [4, 5])
def test_frontend_collects_once_without_replay(directory, serving_step):
    calls = []
    server = NS(global_steps=serving_step, replica_rank=0, config=NS(response_length=4))
    drafter = {
        "speculative_algorithm": "EAGLE3",
        "training": {
            capture.CAPTURE_FLAG: True,
            "collect_interval_steps": 5,
            "hidden_state_window_tokens_per_sample": 2,
            "hidden_state_window_min_rows": 1,
        },
    }

    async def generate(prompt_ids, sampling_params, request_id):
        calls.append(request_id)
        p = sampling_params["extra_args"][capture.CAPTURE_KEY]
        runner, state = runner_and_state([p], [2], [[3, 4]], [1])
        capture.capture_sampled_rows(
            runner, state, NS(sampled_token_ids=torch.tensor([[4, 5]]))
        )
        return NS(token_ids=[4, 5], stop_reason="completed", extra_fields={})

    original = {"max_tokens": 2}
    output = asyncio.run(
        capture.generate_with_capture(
            server, generate, ([1, 2, 3], original, "r"), {}, drafter, collection_step=5
        )
    )
    assert calls == ["r"]
    assert output.extra_fields["drafter_sample"]["hidden_positions"].tolist() == [
        [2, 3]
    ]
    assert original == {"max_tokens": 2}
    assert list(directory.iterdir()) == []


def test_frontend_failure_cleans_up(directory):
    server = NS(global_steps=5, replica_rank=0, config=NS(response_length=2))
    drafter = {
        "speculative_algorithm": "EAGLE3",
        "training": {capture.CAPTURE_FLAG: True, "hidden_state_window_min_rows": 1},
    }

    async def generate(prompt_ids, sampling_params, request_id):
        raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        asyncio.run(
            capture.generate_with_capture(server, generate, ([1], {}, "r"), {}, drafter)
        )
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    "version,method",
    [
        ("0.22.0", "eagle3"),
        ("0.23.0", "dspark"),
        ("0.24.0", "dspark"),
        ("0.25.0", "dflash"),
    ],
)
def test_real_hook_order_preserves_sampler_and_proposer(
    directory, monkeypatch, version, method
):
    import importlib.metadata
    import sys
    import types

    p = plan(end=4)
    open_plan(p)
    fake, state = runner_and_state([p], [2], [[3, 4, 99]], [2])
    order = []

    class FakeRunner:
        def __init__(self):
            self.__dict__.update(fake.__dict__)
            self.execute_model_state = state
            self.device = NS(type="cuda")
            self.supports_mm_inputs = False
            self.use_aux_hidden_state_outputs = True
            self.vllm_config = NS(
                speculative_config=NS(method=method),
                model_config=NS(hf_config=NS(model_type="qwen3")),
                parallel_config=NS(
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    use_ubatching=False,
                ),
                cache_config=NS(enable_prefix_caching=False),
            )

        def _sample(self, logits, metadata):
            order.append("sample")
            return NS(sampled_token_ids=torch.tensor([[4, 5, -1]]))

        def sample_tokens(self):
            self.execute_model_state = None
            result = self._sample(None, None)
            order.append("propose")
            for hidden in state.aux_hidden_states + [state.hidden_states]:
                hidden.fill_(-999)
            return result

    module = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    module.GPUModelRunner = FakeRunner
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: version)
    capture.install_runner_capture()
    capture.install_runner_capture()  # installation is idempotent
    runner = FakeRunner()
    result = runner.sample_tokens()
    assert result.sampled_token_ids.tolist() == [[4, 5, -1]]
    assert order == ["sample", "propose"]
    assert runner._speco_capture_state is None
    sample = capture.load_sample(p, [1, 2, 3], [4, 5])
    assert torch.all(sample["hidden_states"] >= 0)

    incompatible = FakeRunner()
    incompatible.execute_model_state = NS(hidden_states=state.hidden_states)
    with pytest.raises(RuntimeError, match="capture execution state"):
        incompatible.sample_tokens()

    def fail_export(*args):
        raise OSError("storage full")

    monkeypatch.setattr(capture, "capture_sampled_rows", fail_export)
    # A failed feature export does not change sampler/proposer behavior or
    # crash unrelated requests, but the frontend must not train on partial data.
    another = FakeRunner()
    assert another.sample_tokens().sampled_token_ids.tolist() == [[4, 5, -1]]
    assert order == ["sample", "propose", "sample", "propose"]
    with pytest.raises(RuntimeError, match="export failure"):
        capture.load_sample(p, [1, 2, 3], [4, 5])


@pytest.mark.parametrize("algorithm", ["DSPARK", "DOMINO", "EAGLE3", "DFLASH"])
def test_capture_config_does_not_gate_algorithm_names(algorithm):
    capture.validate_config(
        {
            "enable": True,
            "enable_drafter_training": True,
            "speculative_algorithm": algorithm,
            "training": {capture.CAPTURE_FLAG: True},
        },
        {"enable_prefix_caching": False},
        {"async_scheduling": False},
    )


def test_v2_adapter_uses_native_acceptance_counts(directory, monkeypatch):
    import sys
    import types
    from verl_speco.integration.vllm_rollout_capture_v2 import capture_v2_rows

    module = types.ModuleType("vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils")
    module.get_eagle3_aux_layers_from_config = lambda _: (1, 3)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    p = plan(end=4)
    open_plan(p)
    old, state = runner_and_state([p], [2], [[3, 4, 99]], [2])
    batch = NS(
        req_ids=["r0"],
        num_draft_tokens_per_req=[2],
        num_computed_tokens_np=[2],
        query_start_loc_np=[0, 3],
        num_scheduled_tokens=[3],
        input_ids=old.input_ids.gpu,
    )
    state.input_batch = batch
    runner = NS(
        vocab_size=100,
        model_config=old.model_config,
        speculative_config=NS(),
        model=NS(),
        vllm_config=NS(scheduler_config=NS(async_scheduling=False)),
        _speco_capture_plans={"r0": p},
    )
    # Padding is deliberately in-vocabulary: only the native count is correct.
    result = (
        NS(sampled_token_ids=torch.tensor([[4, 5, 0]])),
        torch.tensor([2]),
        torch.tensor([1]),
    )
    capture_v2_rows(runner, state, result)
    sample = capture.load_sample(p, [1, 2, 3], [4, 5])
    assert sample["hidden_positions"].tolist() == [[2, 3]]


def test_final_request_score_is_attached_before_idle_event(monkeypatch):
    from verl_speco.integration import rollout_idle_events

    events = []

    def emit(name, sample, **kwargs):
        events.append(dict(sample))
        return True

    monkeypatch.setattr(rollout_idle_events, "emit_rollout_drafter_sample", emit)
    sample = {
        "hidden_source": "vllm_rollout",
        "_speco_sample_id": "s",
        "global_step": 2,
        "replica_rank": 0,
        "hidden_states": torch.zeros(1, 2, 3),
    }
    output = NS(
        extra_fields={
            "drafter_sample": sample,
            "_verl_request_mean_accept_len": 2.5,
            "_speco_drafter_revision_min": 1,
            "_speco_drafter_revision_max": 1,
        }
    )
    capture.finalize_rollout_sample(output, {})
    capture.finalize_rollout_sample(output, {})
    assert len(events) == 1
    assert events[0]["_verl_accept_len"] == 2.5
    assert events[0]["_verl_hard_score"] == -2.5
    assert events[0]["_speco_drafter_revision_min"] == 1


@pytest.mark.parametrize(
    "scheduled,drafts,accepted,lo,hi",
    [
        (8, 4, 1, 0, 8),
        (8, 4, 3, 2, 8),
        (8, 4, 5, 0, 8),
        (8, 0, 1, 1, 6),
        (8, 4, 0, 6, 8),
    ],
)
def test_device_acceptance_compaction_matches_cpu_filter(
    scheduled, drafts, accepted, lo, hi
):
    import torch
    from verl_speco.integration.vllm_rollout_capture import (
        filter_accepted_snapshot,
        confirmed_row_count,
    )

    hidden = torch.arange((hi - lo) * 3).reshape(hi - lo, 3)
    tokens = torch.arange(lo, hi)
    actual_h, actual_t = filter_accepted_snapshot(
        hidden,
        tokens,
        lo=lo,
        start=0,
        scheduled=scheduled,
        drafts=drafts,
        accepted=torch.tensor(accepted),
    )
    expected = tokens < confirmed_row_count(scheduled, drafts, accepted)
    assert torch.equal(actual_h, hidden[expected])
    assert torch.equal(actual_t, tokens[expected])
