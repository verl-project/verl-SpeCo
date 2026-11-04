# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise production async collection methods without importing Ray/verl."""

import ast
import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
from verl_speco.integration.opd_cotrain import ensure_student_only_feature_payload
from verl_speco.integration.tensor_wire import encode_cpu_tensors


class Ref:
    def __init__(self, value, identity=1):
        self.value, self.identity, self.awaits = value, identity, 0

    def __hash__(self):
        return hash(self.identity)

    def __eq__(self, other):
        return isinstance(other, Ref) and self.identity == other.identity

    def __await__(self):
        async def resolve():
            self.awaits += 1
            return self.value

        return resolve().__await__()


@pytest.fixture
def production():
    path = Path(__file__).resolve().parents[2] / "verl_speco/workers/speco_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_resolve_ray_object_ref",
        "_resolve_hidden_state_chunks",
        "_resolve_feature_chunks",
        "_commit_rollout_features",
    }
    methods = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name in names
    ]
    namespace = {
        "torch": torch,
        "asyncio": asyncio,
        "time": time,
        "Any": Any,
        "cast": cast,
        "logger": logging.getLogger("collection-test"),
        "_is_ray_object_ref": lambda ref: isinstance(ref, Ref),
        "ensure_student_only_feature_payload": ensure_student_only_feature_payload,
    }
    exec(  # noqa: S102 - run isolated production methods without optional dependencies
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return SimpleNamespace(**namespace)


def profile():
    return dict.fromkeys(
        (
            "resolve_s",
            "decode_s",
            "reconstruct_s",
            "unique_chunk_resolves",
            "chunk_cache_hits",
        ),
        0,
    )


def chunk(ref, start=0, length=3, rows=None):
    return {
        "ref": ref,
        "chunk_start": start,
        "chunk_length": length,
        "chunk_row_indices": rows,
    }


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [None, [3, 0, 2], [0, 0, 2]])
def test_wire_and_legacy_reconstruction_match_bits_and_are_independent(
    production, dtype, rows
):
    source = torch.arange(24, dtype=dtype).reshape(6, 4)
    wire = encode_cpu_tensors({"hidden": source})
    wire["_speco_cpu_tensor_wire_v1"]["hidden"]["data"].flags.writeable = False
    legacy = asyncio.run(
        production._resolve_hidden_state_chunks(
            [chunk(Ref(source), rows=rows)], expected_rows=4
        )
    )
    result = asyncio.run(
        production._resolve_hidden_state_chunks(
            [chunk(Ref(wire), rows=rows)], expected_rows=4
        )
    )
    assert torch.equal(result.view(torch.uint8), legacy.view(torch.uint8))
    snapshot = source.clone()
    result.fill_(999)
    assert torch.equal(source, snapshot)


def test_shared_refs_resolve_once_across_samples_but_new_commit_refreshes(production):
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    first = Ref(encode_cpu_tensors({"hidden": source}))
    equal_ref = Ref(first.value)
    cache, stats = {}, profile()

    async def run():
        a = await production._resolve_hidden_state_chunks(
            [chunk(first)], resolved_cache=cache, profile=stats
        )
        b = await production._resolve_hidden_state_chunks(
            [chunk(equal_ref, start=3)], resolved_cache=cache, profile=stats
        )
        return a, b

    a, b = asyncio.run(run())
    assert torch.equal(a[0], source[:3]) and torch.equal(b[0], source[3:])
    assert first.awaits == 1 and equal_ref.awaits == 0
    assert stats["unique_chunk_resolves"] == 1 and stats["chunk_cache_hits"] == 1
    assert stats["full_window_clones"] == 2
    asyncio.run(
        production._resolve_hidden_state_chunks([chunk(equal_ref)], resolved_cache={})
    )
    assert equal_ref.awaits == 1


def test_corrupt_wire_raises_before_storage(production):
    wire = encode_cpu_tensors({"hidden": torch.ones(3, 4)})
    wire["_speco_cpu_tensor_wire_v1"]["hidden"]["shape"] = (100, 4)
    with pytest.raises(ValueError, match="layout"):
        asyncio.run(production._resolve_hidden_state_chunks([chunk(Ref(wire))]))


@pytest.mark.parametrize("teacher_key", ["teacher_ids", "teacher_logprobs"])
def test_commit_rejects_teacher_tensors_before_resolving_or_storing(
    production, teacher_key
):
    ref = Ref(encode_cpu_tensors({"hidden": torch.ones(3, 4)}))
    stored = []
    worker = SimpleNamespace(
        trainer=SimpleNamespace(buffer_version=4),
        _collection_worker_result=lambda **kwargs: dict(
            accepted_samples=0, rejected_samples=0, **kwargs
        ),
        _store_rollout_sample=lambda **kwargs: stored.append(kwargs),
    )
    sample = {
        teacher_key: torch.ones(3),
        "hidden_states_ref_chunks": [chunk(ref)],
    }

    with pytest.raises(ValueError, match="forbidden Teacher tensors: " + teacher_key):
        asyncio.run(production._commit_rollout_features(worker, "teacher", [sample]))

    assert ref.awaits == 0
    assert stored == []


def test_commit_cache_is_local_and_keeps_sample_order_and_accounting(
    production, caplog
):
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    ref = Ref(encode_cpu_tensors({"hidden": source}))
    stored = []

    def store(**kwargs):
        stored.append(kwargs["hidden_states"])
        return True

    worker = SimpleNamespace(
        trainer=SimpleNamespace(buffer_version=4),
        replica_rank=0,
        _collection_worker_result=lambda **kwargs: dict(
            accepted_samples=0, rejected_samples=0, **kwargs
        ),
        _store_rollout_sample=store,
    )
    samples = [
        {
            "input_ids": None,
            "prompts": None,
            "responses": None,
            "hidden_states_ref_chunks": [chunk(ref, start=start)],
        }
        for start in (0, 3)
    ]
    result = asyncio.run(
        production._commit_rollout_features(worker, "collection-1", samples)
    )
    assert result["collected"] and result["accepted_samples"] == 2
    assert result["buffer_version_after"] == 4
    assert result["collection_profile"]["chunk_cache_hits"] == 1
    assert ref.awaits == 1
    assert torch.equal(stored[0][0], source[:3])
    assert torch.equal(stored[1][0], source[3:])
    assert "SpecoCollectionReceive" in caplog.text
    asyncio.run(production._commit_rollout_features(worker, "collection-2", samples))
    assert ref.awaits == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_rpc_diagnostics_preserve_submit_resolve_order_and_results(
    monkeypatch, caplog, enabled
):
    from verl_speco.integration import phase_timing
    from verl_speco.trainer.scheduler import collection_executor as executor

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true" if enabled else "false")
    monkeypatch.setattr(phase_timing, "configure_phase_logger", lambda: None)
    monkeypatch.setattr(
        phase_timing, "logger", logging.getLogger("collection-test.phase")
    )
    calls = []

    def submit(request):
        calls.append(("submit", request))
        return "pending"

    def resolve(ref):
        calls.append(("resolve", ref))
        return {"accepted_samples": 2}

    callback = executor.CallbackDrafterCollectionExecutor(
        set_step=submit,
        stage_submit=submit,
        commit_submit=submit,
        abort_submit=submit,
        rollback_submit=submit,
        finalize_submit=submit,
        resolve=resolve,
    )
    assert callback._call("commit", submit, ["request"], collection_id="c") == {
        "accepted_samples": 2
    }
    assert calls == [("submit", ["request"]), ("resolve", "pending")]
    records = [r for r in caplog.records if "[SpecoPhase]" in r.message]
    assert len(records) == int(enabled)
    assert "SpecoCollectionRPC" not in caplog.text
    if enabled:
        payload = json.loads(records[0].message.split("[SpecoPhase] ", 1)[1])
        assert payload["phase"] == "collection_commit"
        assert set(payload["stages"]) == {"submit_s", "wait_s"}
        assert not payload["failed"]


def test_sp_direct_rows_avoid_padding_and_return_independent_results(
    monkeypatch, caplog
):
    import torch.distributed as dist

    from verl_speco.integration import oldlogprob_runtime as runtime
    from verl_speco.integration import phase_timing

    monkeypatch.setenv("VERL_SPECO_PHASE_TIMING", "true")
    monkeypatch.setattr(phase_timing, "configure_phase_logger", lambda: None)
    monkeypatch.setattr(phase_timing, "logger", logging.getLogger("collection-test.sp"))
    monkeypatch.setattr(
        runtime._gather_cuda_sparse_sp_selected_to_source,
        "_logged_modes",
        set(),
        raising=False,
    )
    rows = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    selected = {
        "rows": rows,
        "batch_indices": torch.tensor([0, 1]),
        "row_indices": torch.tensor([0, 0]),
    }
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "get_global_rank", lambda *args: 0)
    allocations = []

    def gather(value, *, gather_list, **kwargs):
        if value.dtype == torch.float32:
            assert value is rows
        else:
            allocations.append(
                [tensor.untyped_storage().data_ptr() for tensor in gather_list]
            )
        for destination in gather_list:
            destination.copy_(value)

    monkeypatch.setattr(dist, "gather", gather)
    for _ in range(3):
        result = runtime._gather_cuda_sparse_sp_selected_to_source(
            {"sp_group": object(), "sp_rank": 0, "sp_size": 2}, selected
        )
    assert len([r for r in caplog.records if "SpecoSparseSP" in r.message]) == 1
    assert allocations[0][0] == allocations[0][1]
    assert torch.equal(result[0], torch.cat([rows, rows]))
    before = result[0].clone()
    rows.zero_()
    assert torch.equal(result[0], before)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("rows", [None, [2, 0, 1], [0, 0, 2]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_feature_pair_roundtrip_keeps_alignment_fp32_logz_and_cache(
    production, wire, rows, dtype
):
    source = torch.arange(12, dtype=dtype).reshape(3, 4)
    logz = torch.tensor([[1.125], [2.25], [3.5]], dtype=torch.float32)
    payload = (
        encode_cpu_tensors({"hidden": source, "target_logz": logz})
        if wire
        else {"hidden_states": source, "target_logz": logz}
    )
    ref = Ref(payload)
    cache, stats = {}, profile()

    async def resolve():
        return await production._resolve_feature_chunks(
            [chunk(ref, rows=rows)],
            expected_rows=3,
            resolved_cache=cache,
            profile=stats,
        )

    hidden, target_logz = asyncio.run(resolve())
    indices = torch.tensor(rows if rows is not None else [0, 1, 2])
    expected_h = torch.zeros_like(source)
    expected_z = torch.zeros(3, dtype=torch.float32)
    expected_h[indices] = source
    expected_z[indices] = logz.reshape(-1)
    assert torch.equal(hidden[0].view(torch.uint8), expected_h.view(torch.uint8))
    assert target_logz.dtype == torch.float32
    assert torch.equal(target_logz[0], expected_z)
    hidden.zero_()
    target_logz.zero_()
    again_h, again_z = asyncio.run(resolve())
    assert torch.equal(again_h[0], expected_h)
    assert torch.equal(again_z[0], expected_z)
    assert ref.awaits == 1 and stats["chunk_cache_hits"] == 1


def test_commit_forwards_target_logz_with_hidden_features(production):
    source = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    logz = torch.arange(6, dtype=torch.float32).reshape(6, 1) + 0.25
    ref = Ref(encode_cpu_tensors({"hidden": source, "target_logz": logz}))
    stored = []

    def store(**kwargs):
        stored.append((kwargs["hidden_states"], kwargs["target_logz"]))
        return True

    worker = SimpleNamespace(
        trainer=SimpleNamespace(buffer_version=4),
        replica_rank=0,
        _collection_worker_result=lambda **kwargs: dict(
            accepted_samples=0, rejected_samples=0, **kwargs
        ),
        _store_rollout_sample=store,
    )
    samples = [
        {
            "input_ids": None,
            "prompts": None,
            "responses": None,
            "hidden_states_ref_chunks": [chunk(ref, start=start)],
        }
        for start in (0, 3)
    ]
    result = asyncio.run(production._commit_rollout_features(worker, "paired", samples))
    assert result["accepted_samples"] == 2 and ref.awaits == 1
    for index, start in enumerate((0, 3)):
        assert torch.equal(stored[index][0][0], source[start : start + 3])
        assert torch.equal(stored[index][1][0], logz[start : start + 3].reshape(-1))
