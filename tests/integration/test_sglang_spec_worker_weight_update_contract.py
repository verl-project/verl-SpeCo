# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Routed tensor weight updates on sglang's DFlash-family speculative workers.

sglang's scheduler hands every ``update_weights_from_tensor`` that does not set
``disable_draft_model`` to the speculative worker. ``DFlashWorkerV2`` and
``DSparkWorkerV2`` implement none and forward through ``__getattr__`` to the
target worker, so without the patch a draft publish overwrote the target and
left the served draft stale.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

# sglang_patch imports torch at module level; the CPU CI image has none.
pytest.importorskip("torch")

DRAFT_LOADER = "verl_speco.integration.sglang_runtime.speco_sglang_draft_weight_loader"
TARGET_LOADER = (
    "verl_speco.integration.sglang_runtime.speco_sglang_target_weight_loader"
)

_STUBBED = (
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.engine",
    "sglang.srt.utils",
    "sglang.srt.utils.patch_torch",
    "sglang.srt.speculative",
    "sglang.srt.speculative.dspark_components",
)


@pytest.fixture
def sglang_patch(monkeypatch):
    """Import sglang_patch against stub sglang modules (it imports sglang eagerly)."""
    stubs = {name: types.ModuleType(name) for name in _STUBBED}
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(stubs[parent], child, module)
    stubs["sglang.srt.entrypoints.engine"].run_scheduler_process = lambda *a, **k: None
    sys.modules["sglang.srt.utils"].MultiprocessingSerializer = types.SimpleNamespace(
        deserialize=lambda payload: payload
    )
    sys.modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = (
        lambda: (None)
    )
    for name in (
        "sglang.srt.speculative.eagle_worker",
        "sglang.srt.speculative.eagle_worker_v2",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.sglang_patch", raising=False
    )
    module = importlib.import_module("verl_speco.integration.sglang_patch")
    monkeypatch.setattr(module, "_SGLANG_EAGLE_UPDATE_PATCHED", False)
    monkeypatch.setattr(module, "_target_weight_loader", TARGET_LOADER)
    monkeypatch.setattr(module, "_draft_weight_loader", DRAFT_LOADER)
    monkeypatch.setenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", '{"enable": true}')
    yield module
    monkeypatch.delitem(
        sys.modules, "verl_speco.integration.sglang_patch", raising=False
    )


class _FakeModel:
    def __init__(self):
        self.weights = {}


class _FakeWeightUpdater:
    """sglang main's ModelRunner.weight_updater (no runner-level method)."""

    def __init__(self, model):
        self.model = model
        self.load_formats = []

    def update_weights_from_tensor(self, named_tensors, load_format=None):
        self.load_formats.append(load_format)
        if load_format is not None:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        self.model.weights.update(dict(named_tensors))
        return True, "ok"


class _FakeRunner:
    def __init__(self):
        self.model = _FakeModel()
        self.weight_updater = _FakeWeightUpdater(self.model)


class _FakeTargetWorker:
    def __init__(self, tp_rank=0):
        self.tp_rank = tp_rank
        self.model_runner = _FakeRunner()

    def update_weights_from_tensor(self, recv_req):
        return self.model_runner.weight_updater.update_weights_from_tensor(
            recv_req.serialized_named_tensors[self.tp_rank],
            load_format=recv_req.load_format,
        )


def _spec_worker_class(name):
    class SpecWorker:
        def __init__(self, tp_rank=0):
            self._target_worker = _FakeTargetWorker(tp_rank)
            self.draft_model_runner = _FakeRunner()

        @property
        def target_worker(self):
            return self._target_worker

        def __getattr__(self, attr):
            # sglang's delegation: anything unimplemented reaches the target.
            if attr == "_target_worker":
                raise AttributeError(attr)
            return getattr(self.target_worker, attr)

    SpecWorker.__name__ = SpecWorker.__qualname__ = name
    return SpecWorker


@pytest.fixture
def spec_workers(monkeypatch, sglang_patch):
    dflash = types.ModuleType("sglang.srt.speculative.dflash_worker_v2")
    dflash.DFlashWorkerV2 = _spec_worker_class("DFlashWorkerV2")
    dflash.DFlashWorkerV2Helper = object  # not a worker: must stay untouched
    dspark = types.ModuleType(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2"
    )
    dspark.DSparkWorkerV2 = _spec_worker_class("DSparkWorkerV2")
    monkeypatch.setitem(sys.modules, dflash.__name__, dflash)
    monkeypatch.setitem(sys.modules, dspark.__name__, dspark)
    return dflash.DFlashWorkerV2, dspark.DSparkWorkerV2


def _request(named_tensors, tp_size=1, **fields):
    fields.setdefault("load_format", None)
    return types.SimpleNamespace(
        serialized_named_tensors=[list(named_tensors)] * tp_size, **fields
    )


def test_unpatched_worker_sends_draft_tensors_to_the_target(spec_workers) -> None:
    """The bug being fixed: __getattr__ forwards the draft publish to the target."""
    worker = spec_workers[0]()
    worker.update_weights_from_tensor(_request([("fc.weight", 1)]))
    assert worker.target_worker.model_runner.model.weights == {"fc.weight": 1}
    assert worker.draft_model_runner.model.weights == {}


@pytest.mark.parametrize("index", [0, 1], ids=["dflash", "dspark"])
def test_draft_publish_changes_the_draft_and_leaves_the_target(
    sglang_patch, spec_workers, index
) -> None:
    sglang_patch.patch_sglang_eagle_update_weights_from_tensor()
    worker = spec_workers[index](tp_rank=1)
    draft = worker.draft_model_runner
    target = worker.target_worker.model_runner
    draft.model.weights["fc.weight"] = 0

    ok, _ = worker.update_weights_from_tensor(
        _request([("fc.weight", 7)], tp_size=2, load_format=DRAFT_LOADER)
    )
    assert ok
    assert draft.model.weights == {"fc.weight": 7}
    assert target.model.weights == {}
    # The route marker is consumed by the wrapper; the runner gets a plain load.
    assert draft.weight_updater.load_formats == [None]
    assert target.weight_updater.load_formats == []


def test_target_publish_leaves_the_draft(sglang_patch, spec_workers) -> None:
    sglang_patch.patch_sglang_eagle_update_weights_from_tensor()
    worker = spec_workers[0]()
    worker.update_weights_from_tensor(_request([("a", 1)], load_format=TARGET_LOADER))
    worker.update_weights_from_tensor(_request([("b", 2)], disable_draft_model=True))
    assert worker.draft_model_runner.model.weights == {}
    assert worker.target_worker.model_runner.model.weights == {"a": 1, "b": 2}


def test_unrouted_publish_updates_draft_then_target(
    monkeypatch, sglang_patch, spec_workers
) -> None:
    """With no route marker and SPECO off, the EAGLEWorkerV2 default applies."""
    sglang_patch._target_weight_loader = None
    sglang_patch._draft_weight_loader = None
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG")
    monkeypatch.delenv("VERL_SPECO_DRAFTER_CONFIG", raising=False)
    sglang_patch.patch_sglang_eagle_update_weights_from_tensor()
    worker = spec_workers[0]()
    ok, _ = worker.update_weights_from_tensor(_request([("c", 3)]))
    assert ok
    assert worker.draft_model_runner.model.weights == {"c": 3}
    assert worker.target_worker.model_runner.model.weights == {"c": 3}


def test_default_update_fails_loud_on_a_missing_runner(sglang_patch) -> None:
    default = sglang_patch._default_spec_worker_update_weights_from_tensor
    ok, message = default(types.SimpleNamespace(), _request([("a", 1)]))
    assert not ok and "draft" in message
    ok, message = default(
        types.SimpleNamespace(), types.SimpleNamespace(serialized_named_tensors=[])
    )
    assert ok and "No tensor" in message
    ok, message = default(
        types.SimpleNamespace(tp_rank=3), _request([("a", 1)], tp_size=2)
    )
    assert not ok and "tp_rank=3" in message


def test_patch_skips_native_support_and_non_workers(sglang_patch, spec_workers) -> None:
    dflash_cls, dspark_cls = spec_workers
    native = lambda self, recv_req: (True, "native")  # noqa: E731
    dspark_cls.update_weights_from_tensor = native
    sglang_patch.patch_sglang_eagle_update_weights_from_tensor()

    patched = dflash_cls.update_weights_from_tensor
    assert getattr(patched, "_verl_patched_eagle_update_weights", False)
    # A native method is wrapped (routing still applies), not replaced.
    assert dspark_cls.update_weights_from_tensor.__wrapped__ is native
    assert not hasattr(
        sys.modules["sglang.srt.speculative.dflash_worker_v2"].DFlashWorkerV2Helper,
        "update_weights_from_tensor",
    )

    # Re-applying in the same process is a no-op.
    sglang_patch._SGLANG_EAGLE_UPDATE_PATCHED = False
    sglang_patch.patch_sglang_eagle_update_weights_from_tensor()
    assert dflash_cls.update_weights_from_tensor is patched


def test_scheduler_dispatch_patches_a_late_loaded_dspark_worker(
    monkeypatch, sglang_patch
) -> None:
    scheduler_components = types.ModuleType("sglang.srt.managers.scheduler_components")
    weight_updater = types.ModuleType(
        "sglang.srt.managers.scheduler_components.weight_updater"
    )

    class SchedulerWeightUpdaterManager:
        def __init__(self, draft_worker):
            # The manager can hold a stale None across SGLang lifecycle changes;
            # the owning scheduler remains the source of truth.
            self.draft_worker = None
            self.scheduler = types.SimpleNamespace(
                draft_worker=draft_worker,
                model_worker=draft_worker,
                spec_algorithm="DSPARK",
            )
            self.tp_worker = types.SimpleNamespace()
            self.tp_cpu_group = object()

        def _observe_weight_load(self, _kind):
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return False

            return Context()

        def flush_cache_after_weight_update(self, recv_req):
            pass

        def record_weight_version_after_update(self, weight_version):
            pass

        def update_weights_from_tensor(self, recv_req):
            return self.draft_worker.update_weights_from_tensor(recv_req)

    weight_updater.SchedulerWeightUpdaterManager = SchedulerWeightUpdaterManager
    weight_updater.UpdateWeightsFromTensorReqOutput = (
        lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler_components",
        scheduler_components,
    )
    monkeypatch.setitem(sys.modules, weight_updater.__name__, weight_updater)
    monkeypatch.setattr(
        sglang_patch.torch.distributed, "barrier", lambda **kwargs: None
    )

    # This worker was not importable during the initial startup scan.
    late_dspark_cls = _spec_worker_class("DSparkWorkerV2")
    sglang_patch._SGLANG_EAGLE_UPDATE_PATCHED = True
    sglang_patch.patch_sglang_eagle_update_weights_from_tensor()

    worker = late_dspark_cls()
    manager = SchedulerWeightUpdaterManager(worker)
    result = manager.update_weights_from_tensor(
        _request(
            [("markov_head.weight", 9)],
            load_format=DRAFT_LOADER,
            weight_version=None,
        )
    )

    assert result.success
    assert worker.draft_model_runner.model.weights == {"markov_head.weight": 9}
    assert worker.target_worker.model_runner.model.weights == {}
    assert worker.draft_model_runner.weight_updater.load_formats == [None]


def test_runner_update_prefers_the_runner_level_method(sglang_patch) -> None:
    calls = []
    runner = types.SimpleNamespace(
        update_weights_from_tensor=lambda named_tensors, load_format: (
            calls.append(("runner", load_format)) or (True, "ok")
        ),
        weight_updater=types.SimpleNamespace(
            update_weights_from_tensor=lambda **kw: calls.append(("updater", None))
        ),
    )
    assert sglang_patch._runner_update_weights_from_tensor(runner, [], "x") == (
        True,
        "ok",
    )
    assert calls == [("runner", "x")]


def _fake_dspark_model(torch, *, with_stacked, with_bias=False):
    """Minimal DSparkDraftMixin stand-in holding the cached KV-write copies."""
    q_size, kv_size, hidden, n_layers = 4, 2, 3, 2

    def layer():
        attn = types.SimpleNamespace(
            q_size=q_size,
            kv_size=kv_size,
            qkv_proj=types.SimpleNamespace(
                weight=torch.zeros(q_size + 2 * kv_size, hidden),
                bias=torch.zeros(q_size + 2 * kv_size) if with_bias else None,
            ),
            k_norm=types.SimpleNamespace(weight=torch.ones(kv_size)),
        )
        return types.SimpleNamespace(self_attn=attn)

    model = types.SimpleNamespace(
        layers=[layer() for _ in range(n_layers)],
        _fused_kv_write_cache=None,
        _stacked_ctx_kv_cache=None,
    )
    kv_rows = n_layers * 2 * kv_size
    # The real writer builds these caches under torch.inference_mode(), which
    # makes them inference tensors that reject in-place updates elsewhere.
    with torch.inference_mode():
        model._fused_kv_write_cache = (
            1,
            (torch.zeros(kv_rows, hidden), None, torch.ones(n_layers, kv_size)),
        )
        if with_stacked:
            model._stacked_ctx_kv_cache = {
                "weight": torch.zeros(kv_rows, hidden),
                "bias": torch.zeros(kv_rows) if with_bias else None,
                "k_norm_weight": torch.ones(n_layers, kv_size),
            }
    return model


@pytest.mark.parametrize("with_stacked", [False, True], ids=["fused", "fused+stacked"])
def test_draft_publish_refreshes_dspark_ctx_kv_caches(
    sglang_patch, with_stacked
) -> None:
    """The cached cat/stack copies must follow an in-place weight update."""
    import torch

    model = _fake_dspark_model(torch, with_stacked=with_stacked, with_bias=with_stacked)
    w_all, _, knw = model._fused_kv_write_cache[1]
    w_all_ptr, knw_ptr = w_all.data_ptr(), knw.data_ptr()

    def update(named_tensors, load_format):
        for layer in model.layers:
            attn = layer.self_attn
            attn.qkv_proj.weight.fill_(5.0)
            if attn.qkv_proj.bias is not None:
                attn.qkv_proj.bias.fill_(6.0)
            attn.k_norm.weight.fill_(7.0)
        return True, "ok"

    runner = types.SimpleNamespace(update_weights_from_tensor=update, model=model)
    assert sglang_patch._runner_update_weights_from_tensor(runner, [], None) == (
        True,
        "ok",
    )

    # K/V rows only (Q rows are not cached), refreshed without moving the buffers.
    assert torch.all(w_all == 5.0) and torch.all(knw == 7.0)
    assert w_all.data_ptr() == w_all_ptr and knw.data_ptr() == knw_ptr
    if with_stacked:
        stacked = model._stacked_ctx_kv_cache
        assert torch.all(stacked["weight"] == 5.0)
        assert torch.all(stacked["bias"] == 6.0)
        assert torch.all(stacked["k_norm_weight"] == 7.0)


def test_failed_update_and_cacheless_models_skip_the_refresh(sglang_patch) -> None:
    import torch

    model = _fake_dspark_model(torch, with_stacked=False)
    w_all = model._fused_kv_write_cache[1][0]
    failing = types.SimpleNamespace(
        update_weights_from_tensor=lambda named_tensors, load_format: (False, "no"),
        model=model,
    )
    assert sglang_patch._runner_update_weights_from_tensor(failing, [], None) == (
        False,
        "no",
    )
    assert torch.all(w_all == 0.0)

    # Target models and runners without a model carry no such cache.
    sglang_patch._refresh_dspark_ctx_kv_caches(_FakeModel())
    sglang_patch._refresh_dspark_ctx_kv_caches(None)


@pytest.fixture
def confidence_model():
    import torch

    class DSparkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Parameter(torch.zeros(2))
            self.confidence_head = torch.nn.Module()
            self.confidence_head.proj = torch.nn.Linear(2, 1)

        def _load_confidence_weights(self, *, confidence_weights, params_dict):
            # SGLang 76e06fe validates completeness on every load_weights call.
            loaded_names = set()
            for name, weight in confidence_weights:
                if name not in params_dict:
                    raise ValueError(f"unexpected confidence weight {name}")
                param = params_dict[name]
                if param.shape != weight.shape:
                    raise ValueError("confidence shape mismatch")
                param.data.copy_(weight)
                loaded_names.add(name)
            required = {n for n in params_dict if n.startswith("confidence_head.")}
            if required - loaded_names:
                raise ValueError("checkpoint is missing confidence weights")

        def load_weights(self, named_tensors):
            confidence_weights = []
            for name, weight in named_tensors:
                if name.startswith("confidence_head."):
                    confidence_weights.append((name, weight))
                else:
                    self.backbone.data.copy_(weight)
            self._load_confidence_weights(
                confidence_weights=confidence_weights,
                params_dict=dict(self.named_parameters()),
            )

    return DSparkModel()


@pytest.mark.parametrize("runner_level", [False, True])
def test_dspark_confidence_head_supports_multiple_publish_buckets(
    sglang_patch, confidence_model, runner_level
) -> None:
    import torch

    model = confidence_model
    original_load = model._load_confidence_weights

    def update(named_tensors, load_format):
        model.load_weights(named_tensors)
        return True, "ok"

    runner = types.SimpleNamespace(model=model)
    if runner_level:
        runner.update_weights_from_tensor = update
    else:
        runner.weight_updater = types.SimpleNamespace(update_weights_from_tensor=update)

    # Even the head itself may straddle two buckets. Repeat the publish to
    # ensure the first bucket retains the previously served head until updated.
    for value in (3.0, 7.0):
        old_weight = model.confidence_head.proj.weight.detach().clone()
        for index, (name, param) in enumerate(model.named_parameters()):
            assert sglang_patch._runner_update_weights_from_tensor(
                runner, [(name, torch.full_like(param, value))], None
            ) == (True, "ok")
            assert model._load_confidence_weights == original_load
            assert "_load_confidence_weights" not in vars(model)
            if index == 0:
                assert torch.equal(model.confidence_head.proj.weight, old_weight)
        assert all(torch.all(p == value) for p in model.parameters())

    # A subsequent checkpoint load must still reject missing head parameters.
    with pytest.raises(ValueError, match="missing confidence weights"):
        model.load_weights([("backbone", torch.zeros_like(model.backbone))])


@pytest.mark.parametrize("failure", ["unknown_name", "shape", "exception", "false"])
def test_dspark_partial_confidence_load_restores_validation_on_failure(
    sglang_patch, confidence_model, failure
) -> None:
    import torch

    model = confidence_model
    # Also exercise a pre-existing instance override, not just a class method.
    original_load = model._load_confidence_weights
    model._load_confidence_weights = original_load

    def update(named_tensors, load_format):
        if failure == "exception":
            raise RuntimeError("weight update failed")
        if failure == "false":
            return False, "weight update failed"
        model.load_weights(named_tensors)
        return True, "ok"

    runner = types.SimpleNamespace(model=model, update_weights_from_tensor=update)
    name = (
        "confidence_head.unknown"
        if failure == "unknown_name"
        else "confidence_head.proj.weight"
    )
    bucket = [(name, torch.zeros(9))]
    if failure == "false":
        assert sglang_patch._runner_update_weights_from_tensor(
            runner, bucket, None
        ) == (False, "weight update failed")
    else:
        error = RuntimeError if failure == "exception" else ValueError
        match = {
            "unknown_name": "unexpected",
            "shape": "shape",
            "exception": "weight update failed",
        }[failure]
        with pytest.raises(error, match=match):
            sglang_patch._runner_update_weights_from_tensor(runner, bucket, None)
    assert vars(model)["_load_confidence_weights"] is original_load
