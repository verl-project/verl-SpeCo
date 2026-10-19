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

"""Exercise bucketed reloads without requiring a CUDA/Ascend installation."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl_speco.integration.vllm_runtime import _speco_canonicalize_target_tied_weights


EMBED = "model.embed_tokens.weight"
HEAD = "lm_head.weight"


class StrictLoaderModel:
    def __init__(self, tied=True):
        self.aliases = {HEAD: EMBED} if tied else {}
        self.loaded = {}
        self.calls = []

    def load_weights(self, weights):
        weights = dict(weights)
        # Reproduce the vLLM check on this call, not on earlier buckets.
        for alias, canonical in self.aliases.items():
            if alias in weights and canonical not in weights:
                raise ValueError("tied weight is uninitialized")
        loaded = {n: t for n, t in weights.items() if n not in self.aliases}
        self.loaded.update(loaded)
        self.calls.append(loaded)
        return set(loaded)


@pytest.fixture
def loader_utils(monkeypatch):
    utils = ModuleType("vllm.model_executor.models.utils")
    utils._get_tied_embedding_params = lambda model: model.aliases
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return utils


@pytest.mark.parametrize("device", ["cuda:0", "npu:0"])
@pytest.mark.parametrize("buckets", [[EMBED, HEAD], [HEAD, EMBED], [HEAD]])
def test_separate_buckets_and_alias_only_load(loader_utils, device, buckets):
    model = StrictLoaderModel()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model), device=device)
    tensor = object()
    original = model.load_weights
    with pytest.raises(ValueError, match="uninitialized"):
        original([(HEAD, tensor)])

    with _speco_canonicalize_target_tied_weights(worker):
        for name in buckets:
            assert model.load_weights(iter([(name, tensor)])) == {EMBED}
    assert model.loaded == {EMBED: tensor}
    assert all(call == {EMBED: tensor} for call in model.calls)
    assert "load_weights" not in vars(model)
    assert model.load_weights == original
    # Full checkpoint validation still applies outside hot reloads.
    with pytest.raises(ValueError, match="uninitialized"):
        model.load_weights([(HEAD, tensor)])


@pytest.mark.parametrize("alias_first", [True, False])
def test_same_bucket_prefers_canonical(loader_utils, alias_first):
    model = StrictLoaderModel()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    canonical, alias = object(), object()
    weights = [(HEAD, alias), (EMBED, canonical)]
    if not alias_first:
        weights.reverse()
    with _speco_canonicalize_target_tied_weights(worker):
        model.load_weights(weights)
    assert model.loaded == {EMBED: canonical}


@pytest.mark.parametrize("mode", ["untied", "old_ascend", "peft"])
def test_unsupported_paths_are_unchanged(loader_utils, mode):
    model = StrictLoaderModel(tied=False)
    if mode == "old_ascend":
        del loader_utils._get_tied_embedding_params
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    original = model.load_weights
    peft = {} if mode == "peft" else None
    with _speco_canonicalize_target_tied_weights(worker, peft_config=peft):
        assert model.load_weights == original
        model.load_weights([(HEAD, 1), (EMBED, 2)])
    assert model.loaded == {HEAD: 1, EMBED: 2}


def test_loader_failure_restores_existing_override(loader_utils):
    model = StrictLoaderModel()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))

    def failing_loader(weights):
        raise RuntimeError("loader failure")

    model.load_weights = failing_loader
    with pytest.raises(RuntimeError, match="loader failure"):
        with _speco_canonicalize_target_tied_weights(worker):
            model.load_weights([(HEAD, object())])
    assert vars(model)["load_weights"] is failing_loader


@pytest.mark.parametrize(
    "extension_name",
    [
        "SpecoVLLMColocateWorkerExtension",
        "SpecoVLLMWeightSyncCompatExtension",
    ],
)
@pytest.mark.parametrize("is_npu", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_target_sync_preserves_staging_and_restores_loader(
    loader_utils,
    extension_name,
    is_npu,
    fail,
):
    from contextlib import contextmanager
    from inspect import getsource

    from verl_speco.integration import vllm_runtime

    events = []
    model = StrictLoaderModel()
    original = model.load_weights

    @contextmanager
    def staging(worker, **kwargs):
        events.append("staging_enter")
        try:
            yield
        finally:
            events.append("staging_exit")

    class Base:
        def update_weights_from_ipc(self, **kwargs):
            assert kwargs == dict(peft_config=None, base_sync_done=True, use_shm=is_npu)
            model.load_weights([(EMBED, 3)])
            model.load_weights([(HEAD, 3)])
            events.append("loaded")
            if fail:
                raise RuntimeError("sync failed")
            return "synced"

    # Compile the production method against a fake transport base, avoiding
    # dependencies on the installed vLLM worker and actual GPU/NPU hardware.
    namespace = dict(vars(vllm_runtime), Base=Base)
    namespace.update(
        patch_verl_bucketed_weight_transfer_rebuild_ipc=lambda: False,
        patch_verl_bucketed_weight_transfer_shm_reuse=lambda: None,
        patch_verl_bucketed_weight_transfer_npu_staging=lambda: None,
        _speco_is_npu_vllm_worker=lambda worker: is_npu,
        _speco_npu_target_staging=staging,
        trim_process_host_memory=lambda: events.append("trim"),
    )
    method = getattr(vllm_runtime, extension_name).update_weights_from_ipc
    exec("class ReloadExtension(Base):\n" + getsource(method), namespace)
    worker = namespace["ReloadExtension"]()
    worker.model_runner = SimpleNamespace(model=model)
    worker._speco_diag_draft_state = lambda phase: None

    def sync_head():
        assert model.load_weights == original
        events.append("draft_head")
        return 0

    worker._speco_sync_dspark_lm_head_from_target = sync_head
    if fail:
        with pytest.raises(RuntimeError, match="sync failed"):
            worker.update_weights_from_ipc(base_sync_done=True, use_shm=is_npu)
    else:
        assert (
            worker.update_weights_from_ipc(base_sync_done=True, use_shm=is_npu)
            == "synced"
        )
    assert model.loaded == {EMBED: 3}
    assert model.load_weights == original
    assert ("trim" in events) == is_npu
    expects_staging = is_npu or extension_name == "SpecoVLLMColocateWorkerExtension"
    assert ("staging_enter" in events) == expects_staging
    if expects_staging:
        assert (
            events.index("staging_enter")
            < events.index("loaded")
            < events.index("staging_exit")
        )
    assert ("draft_head" in events) == (
        not fail and extension_name == "SpecoVLLMColocateWorkerExtension"
    )
