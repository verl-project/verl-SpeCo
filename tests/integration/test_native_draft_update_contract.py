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
"""Contract for the native vLLM draft weight-update selection.

``speco.runtime.weight_update_mode`` selects the draft update path:
``auto`` (default) uses the vLLM native draft session when capability detection
succeeds, otherwise falls back to SPECO's draft-only compat implementation;
``native`` fails fast when the backend cannot do a draft update; ``compat``
always forces the SPECO compat path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from verl_speco.integration import native_draft_update as ndu


def _config(mode=None):
    speco = {} if mode is None else {"runtime": {"weight_update_mode": mode}}
    return {"speco": speco}


def test_resolve_weight_update_mode_defaults_and_normalizes() -> None:
    assert ndu.resolve_weight_update_mode({}) == "auto"
    assert ndu.resolve_weight_update_mode({"speco": {}}) == "auto"
    assert (
        ndu.resolve_weight_update_mode(
            {"speco": {"runtime": {"weight_update_mode": "NATIVE"}}}
        )
        == "native"
    )
    assert (
        ndu.resolve_weight_update_mode(
            {"speco": {"runtime": {"weight_update_mode": "compat"}}}
        )
        == "compat"
    )


def test_resolve_weight_update_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="auto|native|compat"):
        ndu.resolve_weight_update_mode(
            {"speco": {"runtime": {"weight_update_mode": "fast"}}}
        )


def test_native_draft_update_available_requires_full_capability() -> None:
    assert ndu.native_draft_update_available(None) is False
    assert ndu.native_draft_update_available(SimpleNamespace()) is False
    assert ndu.native_draft_update_available(_full_native_rollout()) is True


def test_stock_verl_rollout_adapter_lacks_native_draft_update_api() -> None:
    """Stock verl ServerAdapter does not expose the native draft-update session.

    Documents that on verl 0.8/0.9 the rollout adapter has ``update_weights``
    (actor session) but lacks ``start_draft_weight_update`` etc.  Therefore
    ``auto`` correctly falls back to ``compat`` and ``native`` correctly
    fails fast.  The native path is provisioned for future vLLM V1 backends
    or runtime injection that expose these APIs.
    """
    pytest.importorskip("verl", reason="stock adapter contract needs verl")
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    adapter = ServerAdapter.__new__(ServerAdapter)
    assert callable(getattr(adapter, "update_weights", None))  # actor API exists
    assert not callable(getattr(adapter, "start_draft_weight_update", None))
    assert not callable(getattr(adapter, "finish_weight_update", None))
    assert ndu.native_draft_update_available(adapter) is False


def test_patch_vllm_server_adapter_update_does_not_inject_update_draft_weights(
    monkeypatch,
) -> None:
    """Production init must not pre-inject compat ``update_draft_weights`` on
    the ``ServerAdapter`` class.  Otherwise ``attach_draft_weight_updater``'s
    early-return guard would short-circuit ``select_draft_update_strategy``,
    making the native/compat selector and its startup log dead code.
    """
    pytest.importorskip("verl", reason="needs verl ServerAdapter")
    from verl.workers.rollout.vllm_rollout import vllm_rollout as vllm_rollout_mod

    from verl_speco.integration.vllm_runtime import patch_vllm_server_adapter_update

    ServerAdapter = getattr(vllm_rollout_mod, "ServerAdapter", None)
    if ServerAdapter is None:
        pytest.skip("ServerAdapter not importable")
    # Clean any pre-existing method from prior test runs / imports.
    if hasattr(ServerAdapter, "update_draft_weights"):
        monkeypatch.delattr(ServerAdapter, "update_draft_weights")

    patch_vllm_server_adapter_update()

    assert not callable(
        getattr(ServerAdapter, "update_draft_weights", None)
    ), "patch_vllm_server_adapter_update must NOT inject update_draft_weights"


def test_production_attach_after_init_selects_compat_on_stock_adapter(
    monkeypatch, caplog
) -> None:
    """After production init (IPC patches only, no compat pre-injection),
    ``attach_draft_weight_updater`` on a stock adapter must run the strategy
    selector, log the decision, and bind the compat method."""
    pytest.importorskip("verl", reason="needs verl ServerAdapter")
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    from verl_speco.integration import native_draft_update as ndu_mod

    compat_calls: list = []
    monkeypatch.setattr(ndu_mod, "_default_compat_attacher", lambda r: compat_calls.append(r))

    caplog.set_level("WARNING", logger="verl_speco.integration.native_draft_update")
    adapter = ServerAdapter.__new__(ServerAdapter)

    ndu_mod.attach_draft_weight_updater({"speco": {"runtime": {"weight_update_mode": "auto"}}}, adapter)

    # The selector ran (not short-circuited by a pre-injected method).
    assert len(compat_calls) == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("backend=" in m and "path=compat" in m for m in messages)


def test_select_compat_mode_forces_compat_path() -> None:
    decision = ndu.select_draft_update_strategy(
        _config("compat"), _full_native_rollout()
    )
    assert decision.path == "compat"
    assert decision.native_available is False
    assert "compat" in decision.reason


def test_select_auto_falls_back_when_native_unavailable() -> None:
    decision = ndu.select_draft_update_strategy(_config("auto"), SimpleNamespace())
    assert decision.path == "compat"
    assert decision.native_available is False
    assert "unavailable" in decision.reason


def test_select_auto_uses_native_when_available() -> None:
    decision = ndu.select_draft_update_strategy(
        _config("auto"), _full_native_rollout()
    )
    assert decision.path == "native"
    assert decision.native_available is True


def test_select_native_fails_fast_when_unavailable() -> None:
    with pytest.raises(
        RuntimeError, match="native draft update requested, but backend"
    ):
        ndu.select_draft_update_strategy(_config("native"), SimpleNamespace())


def test_select_native_uses_native_when_available() -> None:
    decision = ndu.select_draft_update_strategy(
        _config("native"), _full_native_rollout()
    )
    assert decision.path == "native"
    assert decision.native_available is True


class _RemoteMethod:
    """Record a vLLM server remote call and resolve as a no-op awaitable."""

    def __init__(self, calls: list, name: str):
        self._calls = calls
        self._name = name

    def remote(self, *args, **kwargs):
        self._calls.append((self._name, args, kwargs))

        async def _completed():
            return None

        return _completed()

    def __call__(self, *args, **kwargs):
        return self.remote(*args, **kwargs)


class _FullNativeRollout:
    """Fake vLLM rollout exposing the native draft-update session + server RPC."""

    def __init__(self, *, fail_update: bool = False):
        self.calls: list = []
        self.rollout_rank = 0
        self.replica_rank = 0
        self.fail_update = fail_update
        self.server_handle = SimpleNamespace(
            abort_all_requests=_RemoteMethod(self.calls, "abort_all_requests"),
            clear_kv_cache=_RemoteMethod(self.calls, "clear_kv_cache"),
            set_global_steps=_RemoteMethod(self.calls, "set_global_steps"),
            resume_generation=_RemoteMethod(self.calls, "resume_generation"),
            start_draft_weight_update=_RemoteMethod(
                self.calls, "server.start_draft_weight_update"
            ),
        )

    async def start_draft_weight_update(self):
        self.calls.append(("start_draft_weight_update", (), {}))

    async def start_weight_update(self):
        self.calls.append(("start_weight_update", (), {}))

    async def update_weights(self, weights, global_steps=None):
        self.calls.append(
            ("update_weights", (weights,), {"global_steps": global_steps})
        )
        if self.fail_update:
            raise RuntimeError("native update_weights failed")

    async def finish_weight_update(self):
        self.calls.append(("finish_weight_update", (), {}))

    @property
    def supports_draft_weight_update(self) -> bool:
        return True

    def supports_draft_weight_updates(self) -> bool:
        return True


def _full_native_rollout(**kwargs) -> _FullNativeRollout:
    return _FullNativeRollout(**kwargs)


def _expected_lifecycle(global_steps):
    return [
        ("abort_all_requests", (), {"reset_prefix_cache": True}),
        ("start_draft_weight_update", (), {}),
        ("update_weights", ({"w": 1},), {"global_steps": global_steps}),
        ("finish_weight_update", (), {}),
        ("clear_kv_cache", (), {}),
        ("set_global_steps", (global_steps,), {}),
        ("resume_generation", (), {}),
    ]


def test_native_client_update_runs_full_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._load_env_drafter_config",
        lambda: {"training": {}},
    )
    rollout = _full_native_rollout()
    client = ndu.NativeVLLMDraftWeightSyncClient(rollout)

    asyncio.run(client.update({"w": 1}, global_steps=7))

    assert rollout.calls == _expected_lifecycle(7)


def test_native_client_update_skips_empty_weights(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._load_env_drafter_config",
        lambda: {"training": {}},
    )
    rollout = _full_native_rollout()
    client = ndu.NativeVLLMDraftWeightSyncClient(rollout)

    asyncio.run(client.update({}, global_steps=1))

    assert rollout.calls == []


def test_native_client_update_finishes_session_and_resumes_on_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._load_env_drafter_config",
        lambda: {"training": {}},
    )
    rollout = _full_native_rollout(fail_update=True)
    client = ndu.NativeVLLMDraftWeightSyncClient(rollout)

    with pytest.raises(RuntimeError, match="native update_weights failed"):
        asyncio.run(client.update({"w": 1}, global_steps=5))

    assert rollout.calls == [
        ("abort_all_requests", (), {"reset_prefix_cache": True}),
        ("start_draft_weight_update", (), {}),
        ("update_weights", ({"w": 1},), {"global_steps": 5}),
        ("finish_weight_update", (), {}),
        ("resume_generation", (), {}),
    ]


def test_attach_auto_unavailable_uses_compat_attacher(monkeypatch) -> None:
    compat_calls: list = []

    def fake_compat(rollout):
        compat_calls.append(rollout)

    monkeypatch.setattr(ndu, "_default_compat_attacher", fake_compat)
    rollout = SimpleNamespace()

    ndu.attach_draft_weight_updater(_config("auto"), rollout)

    assert compat_calls == [rollout]
    assert not callable(getattr(rollout, "update_draft_weights", None))


def test_attach_auto_available_installs_native_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._load_env_drafter_config",
        lambda: {"training": {}},
    )
    monkeypatch.setattr(ndu, "_default_compat_attacher", lambda _: pytest.fail(
        "compat attacher must not run when native is available"
    ))
    rollout = _full_native_rollout()

    ndu.attach_draft_weight_updater(_config("auto"), rollout)

    assert callable(rollout.update_draft_weights)
    asyncio.run(rollout.update_draft_weights({"w": 1}, global_steps=3))
    assert rollout.calls == _expected_lifecycle(3)


def test_attach_native_unavailable_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(ndu, "_default_compat_attacher", lambda _: pytest.fail(
        "compat attacher must not run when native mode fails fast"
    ))
    with pytest.raises(RuntimeError, match="native draft update requested"):
        ndu.attach_draft_weight_updater(_config("native"), SimpleNamespace())


def test_attach_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(
        ndu,
        "select_draft_update_strategy",
        lambda *_: pytest.fail("strategy must not be re-resolved"),
    )
    sentinel = lambda *_a, **_k: None  # noqa: E731
    rollout = SimpleNamespace(update_draft_weights=sentinel)

    assert ndu.attach_draft_weight_updater(_config("native"), rollout) is rollout
    assert rollout.update_draft_weights is sentinel


def test_attach_logs_decision_once(caplog) -> None:
    caplog.set_level("WARNING", logger="verl_speco.integration.native_draft_update")
    rollout = _full_native_rollout()

    ndu.attach_draft_weight_updater(_config("native"), rollout)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "backend=" in m
        and "native_draft_update=enabled" in m
        and "weight_update_mode=native" in m
        for m in messages
    )


# ---------------------------------------------------------------------------
# Phase 3: vLLM-Ascend capability probe, fallback and speculation-only gating.
# ---------------------------------------------------------------------------


def test_resolve_draft_update_backend_detects_ascend_signals(monkeypatch) -> None:
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "0")
    assert ndu.resolve_draft_update_backend({}, None) == "vllm-ascend"

    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "npu")
    assert ndu.resolve_draft_update_backend({}, None) == "vllm-ascend"

    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    monkeypatch.setenv("VLLM_PLATFORM", "ascend")
    assert ndu.resolve_draft_update_backend({}, None) == "vllm-ascend"

    monkeypatch.delenv("VLLM_PLATFORM", raising=False)
    assert (
        ndu.resolve_draft_update_backend({"trainer": {"device": "npu"}}, None)
        == "vllm-ascend"
    )


def test_ascend_native_draft_update_available_requires_full_contract() -> None:
    assert ndu.ascend_native_draft_update_available(SimpleNamespace()) is False
    assert ndu.ascend_native_draft_update_available(_full_native_rollout()) is True


class _AscendMissingWorkerContract(_FullNativeRollout):
    """Ascend backend that ships the engine flag but not the Worker contract."""

    @property
    def supports_draft_weight_updates(self):
        raise AttributeError("vLLM-Ascend worker has not implemented the contract")


def test_ascend_probe_rejects_engine_without_worker_contract() -> None:
    rollout = _AscendMissingWorkerContract()

    assert ndu.ascend_native_draft_update_available(rollout) is False
    assert ndu.native_draft_update_available(rollout, backend="vllm-ascend") is False


@pytest.mark.parametrize("mode", ["auto", "native"])
def test_ascend_select_fails_fast_or_falls_back_when_native_unavailable(
    mode,
) -> None:
    config = {"speco": {"runtime": {"weight_update_mode": mode}}}
    if mode == "native":
        with pytest.raises(
            RuntimeError, match="native draft update requested, but backend"
        ):
            ndu.select_draft_update_strategy(config, SimpleNamespace())
    else:
        decision = ndu.select_draft_update_strategy(config, SimpleNamespace())
        assert decision.path == "compat"
        assert decision.native_available is False
        assert "unavailable" in decision.reason


def test_ascend_auto_uses_native_when_full_contract_present() -> None:
    decision = ndu.select_draft_update_strategy(
        {"speco": {"runtime": {"weight_update_mode": "auto"}}},
        _full_native_rollout(),
    )
    assert decision.path == "native"
    assert decision.native_available is True


def test_ascend_compat_mode_forces_compat_even_when_native_available() -> None:
    decision = ndu.select_draft_update_strategy(
        {"speco": {"runtime": {"weight_update_mode": "compat"}}},
        _full_native_rollout(),
    )
    assert decision.path == "compat"
    assert decision.native_available is False
    assert "compat" in decision.reason


def test_native_client_does_not_touch_actor_weight_session(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._load_env_drafter_config",
        lambda: {"training": {}},
    )
    rollout = _full_native_rollout()
    client = ndu.NativeVLLMDraftWeightSyncClient(rollout)

    asyncio.run(client.update({"w": 1}, global_steps=9))

    assert ("start_draft_weight_update", (), {}) in rollout.calls
    assert ("start_weight_update", (), {}) not in rollout.calls


def test_attach_ascend_auto_unavailable_uses_compat_attacher(monkeypatch) -> None:
    compat_calls: list = []

    def fake_compat(rollout):
        compat_calls.append(rollout)

    monkeypatch.setattr(ndu, "_default_compat_attacher", fake_compat)
    rollout = SimpleNamespace()

    ndu.attach_draft_weight_updater(
        {"speco": {"runtime": {"weight_update_mode": "auto"}}}, rollout
    )

    assert compat_calls == [rollout]


def test_draft_update_fallback_only_triggers_in_speculation_mode() -> None:
    from verl_speco.integration.rollout_publish import DraftWeightPublishMixin

    worker = DraftWeightPublishMixin()
    worker.config = {"rollout": {"drafter": {"enable": False}}}
    worker._attach_update_draft_weights_to_rollout = lambda: pytest.fail(
        "draft updater must not attach when the drafter is disabled"
    )

    rollout_calls: list = []

    class _Rollout:
        async def update_draft_weights(self, *args, **kwargs):
            rollout_calls.append("called")

    worker.rollout = _Rollout()

    asyncio.run(worker.update_draft_weights({"w": 1}, global_steps=1))

    assert rollout_calls == []
