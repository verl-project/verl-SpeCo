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
from __future__ import annotations

import asyncio
import json
import re
import sys
import types
from contextlib import nullcontext
from inspect import getsource
from pathlib import Path
from types import SimpleNamespace

import pytest

import verl_speco.integration.vllm_runtime as vllm_runtime

from verl_speco.integration.vllm_runtime import (
    SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX,
    SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    SPECO_VLLM_WORKER_EXTENSION_CLS,
    SpecoVLLMColocateWorkerExtension,
    SpecoVLLMWeightSyncCompatExtension,
    _SpecoVLLMHttpServerMixin,
    _describe_vllm_draft_logits,
    _invoke_bucket_received_callback,
    _new_vllm_spec_decode_stats,
    _normalize_dflash_target_layer_aliases,
    _record_vllm_spec_decode_scheduler_stats,
    _record_vllm_worker_spec_decode_output,
    _speco_can_use_npu_target_staging,
    _speco_npu_target_staging,
    _speco_npu_target_staging_decision,
    _speco_persistent_weight_shm_name,
    _validate_vllm_dflash_drafter_config,
    _vllm_ascend_has_dspark_pr11153_k_query_runtime,
    _vllm_spec_decode_stats_to_metrics,
    attach_update_draft_weights_to_rollout,
    build_vllm_speculative_config_from_drafter,
    configure_vllm_runtime_from_config,
    patch_transformers_attention_layer_type_constants,
    patch_verl_bucketed_weight_transfer_npu_staging,
    patch_verl_bucketed_weight_transfer_shm_reuse,
    patch_vllm_dspark_draft_load_config,
    patch_vllm_dspark_registry_aliases,
    patch_vllm_dspark_runtime,
    speco_vllm_update_draft_weights,
)


def _drafter(**overrides):
    config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "EAGLE3",
        "model_path": "/models/drafter",
        "rollout": {"spec_steps": 3},
        "training": {},
        "vllm": {},
    }
    config.update(overrides)
    return config


def test_vllm_speculative_config_maps_eagle3_contract() -> None:
    config = build_vllm_speculative_config_from_drafter(_drafter())

    assert config == {
        "draft_sample_method": "greedy",
        "method": "eagle3",
        "model": "/models/drafter",
        "num_speculative_tokens": 3,
    }


def test_vllm_fresh_training_does_not_load_checkpoint_output_root() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(checkpoint_path="/checkpoints/run/drafter")
    )

    assert config["model"] == "/models/drafter"


def test_vllm_checkpoint_path_remains_a_fallback_without_model_path() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(model_path=None, checkpoint_path="/checkpoints/draft_step_10")
    )

    assert config["model"] == "/checkpoints/draft_step_10"


def test_vllm_worker_extension_constructs_without_wake_up_fallback() -> None:
    extension = SpecoVLLMColocateWorkerExtension()

    assert isinstance(extension, SpecoVLLMColocateWorkerExtension)


def test_vllm_worker_extension_installs_dspark_loader_before_construction(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        vllm_runtime,
        "patch_vllm_dspark_draft_load_config",
        lambda: calls.append("draft_load") or True,
    )

    SpecoVLLMColocateWorkerExtension()

    assert calls == ["draft_load"]
def test_vllm_initial_drafter_initialization_is_serialized() -> None:
    class Server(_SpecoVLLMHttpServerMixin):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self._speco_initial_draft_weights_required = True
            self._speco_initial_draft_weights_ready = False
            self._speco_initial_draft_weights_lock = None

        async def collective_rpc(self, method: str) -> None:
            self.calls.append(method)

    server = Server()

    async def ensure_twice() -> None:
        await asyncio.gather(
            server._speco_ensure_initial_draft_weights(),
            server._speco_ensure_initial_draft_weights(),
        )

    asyncio.run(ensure_twice())

    assert server.calls == ["speco_ensure_draft_initialized"]
    assert server._speco_initial_draft_weights_ready is True


def test_vllm_worker_initializes_dflash_drafter_once(monkeypatch) -> None:
    extension = SpecoVLLMColocateWorkerExtension()
    reload_calls: list[bool] = []

    def reload_from_checkpoint() -> int:
        reload_calls.append(True)
        extension._speco_draft_weight_source = "checkpoint"
        return 23

    monkeypatch.setattr(extension, "_speco_is_dflash_draft", lambda: True)
    monkeypatch.setattr(
        extension,
        "_speco_reload_draft_from_checkpoint",
        reload_from_checkpoint,
    )

    assert extension.speco_ensure_draft_initialized() == {
        "initialized": True,
        "source": "checkpoint",
        "loaded_params": 23,
    }
    assert extension.speco_ensure_draft_initialized() == {
        "initialized": True,
        "source": "checkpoint",
        "loaded_params": 0,
    }
    assert reload_calls == [True]


def _revision_runtime_extension():
    torch = pytest.importorskip("torch")
    draft = torch.nn.Module()
    draft.model = torch.nn.Module()
    draft.model.online = torch.nn.Parameter(torch.tensor([3.0]))
    proposer = SimpleNamespace(
        model=draft,
        speculative_config=SimpleNamespace(method="dspark"),
    )
    extension = SpecoVLLMColocateWorkerExtension()
    extension.model_runner = SimpleNamespace(speculator=proposer)
    return extension, draft, torch


def test_vllm_npu_memory_snapshot_reports_allocator_and_device_values() -> None:
    extension = SpecoVLLMColocateWorkerExtension()
    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(
            memory_allocated=lambda: 2 << 20,
            memory_reserved=lambda: 3 << 20,
            mem_get_info=lambda: (4 << 20, 64 << 20),
        )
    )

    snapshot = extension._speco_npu_memory_snapshot(fake_torch)

    assert snapshot == {
        "allocated": 2 << 20,
        "reserved": 3 << 20,
        "free": 4 << 20,
        "total": 64 << 20,
    }
    assert extension._speco_format_npu_memory_snapshot(snapshot) == (
        "allocated=2.0,reserved=3.0,free=4.0,total=64.0"
    )


def test_vllm_npu_memory_snapshot_tolerates_missing_apis() -> None:
    extension = SpecoVLLMColocateWorkerExtension()

    assert extension._speco_npu_memory_snapshot(SimpleNamespace()) == {}
    assert extension._speco_format_npu_memory_snapshot({}) == "unavailable"


def test_vllm_level1_wake_keeps_online_draft_revision(monkeypatch) -> None:
    extension, _, _ = _revision_runtime_extension()
    extension._speco_draft_runtime_revision = 3
    reload_calls = []
    monkeypatch.setattr(
        extension,
        "_speco_reload_draft_from_checkpoint",
        lambda: reload_calls.append(True) or 1,
    )

    assert extension._speco_prepare_draft_for_sleep(1) == 0
    assert extension._speco_restore_draft_for_wake(["weights"]) == (None, 0)
    assert extension._speco_draft_runtime_revision == 3
    assert reload_calls == []


def test_vllm_level2_wake_restores_matching_online_revision() -> None:
    extension, draft, torch = _revision_runtime_extension()
    extension._speco_draft_runtime_revision = 4
    expected = {
        name: tensor.detach().clone()
        for name, tensor in (
            list(draft.named_parameters()) + list(draft.named_buffers())
        )
    }

    assert extension._speco_prepare_draft_for_sleep(2) == len(expected)
    with torch.no_grad():
        for tensor in draft.parameters():
            tensor.zero_()

    assert extension._speco_restore_draft_for_wake(["weights"]) == (
        "snapshot",
        len(expected),
    )
    assert extension._speco_draft_runtime_revision == 4
    for name, tensor in draft.named_parameters():
        torch.testing.assert_close(tensor, expected[name])


def test_vllm_level2_missing_online_snapshot_refuses_checkpoint_rollback(
    monkeypatch,
) -> None:
    extension, _, _ = _revision_runtime_extension()
    extension._speco_draft_runtime_revision = 2
    extension._speco_draft_level2_restore_pending = True
    extension._speco_draft_level2_snapshot = None
    monkeypatch.setattr(extension, "_speco_reload_draft_from_checkpoint", lambda: 64)

    with pytest.raises(RuntimeError, match="Refusing to roll back"):
        extension._speco_restore_draft_for_wake(["weights"])


def test_vllm_weight_sync_extension_has_stable_runtime_path() -> None:
    assert SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS.endswith(
        ".SpecoVLLMWeightSyncCompatExtension"
    )
    source = getsource(SpecoVLLMWeightSyncCompatExtension.update_weights_from_ipc)
    assert source.index(
        "patch_verl_bucketed_weight_transfer_rebuild_ipc()"
    ) < source.index("super().update_weights_from_ipc(")
    assert source.index(
        "patch_verl_bucketed_weight_transfer_shm_reuse()"
    ) < source.index("super().update_weights_from_ipc(")
    assert source.index(
        "patch_verl_bucketed_weight_transfer_npu_staging()"
    ) < source.index("super().update_weights_from_ipc(")
    assert "with _speco_npu_target_staging(" in source


def test_vllm_npu_staging_is_guarded_and_preserves_upstream_fallback() -> None:
    guard_source = getsource(_speco_npu_target_staging_decision)
    context_source = getsource(_speco_npu_target_staging)
    patch_source = getsource(patch_verl_bucketed_weight_transfer_npu_staging)

    assert "not use_shm" in guard_source
    assert "peft_config is not None" in guard_source
    assert "not _speco_is_npu_vllm_worker(worker)" in guard_source
    assert 'getattr(vllm_config, "quant_config", None)' in guard_source
    assert "quant_config is not None" in guard_source
    assert "return original_receive(self, on_bucket_received)" in patch_source
    assert "_invoke_bucket_received_callback(" in patch_source
    assert "SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES" in patch_source
    assert "staging_buffer[start:end].copy_(" in patch_source
    assert "self.buffer[start:end], non_blocking=False" in patch_source
    assert "get_torch_device().synchronize()" in patch_source
    assert "NPU staging decision" in context_source
    assert "flush=True" in context_source
    assert "return enabled" in getsource(_speco_can_use_npu_target_staging)


def test_bucket_callback_adapter_supports_verl080_and_verl090_signatures() -> None:
    calls = []

    def callback_v080(weights):
        calls.append(("0.8", weights))

    def callback_v090(weights, is_last):
        calls.append(("0.9", weights, is_last))

    weights = [("weight", object())]
    _invoke_bucket_received_callback(callback_v080, weights, False)
    _invoke_bucket_received_callback(callback_v090, weights, True)

    assert calls == [("0.8", weights), ("0.9", weights, True)]


def test_vllm_weight_shm_name_is_stable_and_channel_scoped() -> None:
    handle = "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"

    assert _speco_persistent_weight_shm_name(
        handle, 2048 << 20
    ) == _speco_persistent_weight_shm_name(handle, 2048 << 20)
    assert _speco_persistent_weight_shm_name(
        handle, 2048 << 20
    ) != _speco_persistent_weight_shm_name(handle, 512 << 20)
    assert _speco_persistent_weight_shm_name(
        handle, 2048 << 20
    ) != _speco_persistent_weight_shm_name(
        handle.replace("rank-0", "rank-1"), 2048 << 20
    )


def test_vllm_weight_shm_patch_reuses_mapping_and_preserves_ipc_path() -> None:
    created = []

    class FakeShm:
        def __init__(self, size: int):
            self.size = size
            self.buf = bytearray(size)
            self.close_count = 0
            self.unlink_count = 0

        def close(self):
            self.close_count += 1

        def unlink(self):
            self.unlink_count += 1

    class FakeTorch:
        uint8 = "uint8"

        @staticmethod
        def frombuffer(buffer, dtype):
            assert dtype == FakeTorch.uint8
            return buffer

    class FakeSocket:
        def __init__(self, incoming=None):
            self.metadata = []
            self.incoming = incoming

        def send_pyobj(self, value):
            self.metadata.append(value)

        def recv(self):
            return b""

        def recv_pyobj(self):
            return self.incoming

        def send(self, value):
            self.metadata.append(value)

    class FakeSender:
        def __init__(self, *, use_shm: bool):
            self.use_shm = use_shm
            self.zmq_handle = "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"
            self.bucket_size = 64
            self.socket = FakeSocket()
            self.buffer = None
            self.shm = None
            self.upstream_init_called = False
            self.upstream_cleanup_called = False

        def _init_buffer(self):
            self.upstream_init_called = True

        def _cleanup(self):
            self.upstream_cleanup_called = True
            self.buffer = None
            self.shm = None

    class FakeReceiver:
        def __init__(self, *, use_shm: bool, metadata):
            self.use_shm = use_shm
            self.socket = FakeSocket(metadata)
            self.buffer = None
            self.shm = None
            self.upstream_init_called = False
            self.upstream_cleanup_called = False

        def _init_buffer(self):
            self.upstream_init_called = True

        def _cleanup(self):
            self.upstream_cleanup_called = True
            self.buffer = None
            self.shm = None

    def create_shared_memory(size, name):
        shm = FakeShm(size)
        created.append((name, shm))
        return shm

    module = SimpleNamespace(
        BucketedWeightSender=FakeSender,
        BucketedWeightReceiver=FakeReceiver,
        create_shared_memory=create_shared_memory,
        rebuild_shared_memory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected attach")
        ),
        torch=FakeTorch,
    )

    assert patch_verl_bucketed_weight_transfer_shm_reuse(module) is True
    assert patch_verl_bucketed_weight_transfer_shm_reuse(module) is False
    first = FakeSender(use_shm=True)
    first._init_buffer()
    first_buffer = first.buffer
    first._cleanup()
    second = FakeSender(use_shm=True)
    second._init_buffer()

    assert len(created) == 1
    assert second.buffer is first_buffer
    assert created[0][1].close_count == 0
    assert created[0][1].unlink_count == 0
    assert first.upstream_cleanup_called is True

    receiver = FakeReceiver(use_shm=True, metadata=second.socket.metadata[0])
    receiver._init_buffer()
    assert receiver.buffer is first_buffer
    receiver._cleanup()
    assert receiver.upstream_cleanup_called is True

    ipc_sender = FakeSender(use_shm=False)
    ipc_sender._init_buffer()
    assert ipc_sender.upstream_init_called is True

    second._cleanup()
    module._speco_cleanup_persistent_weight_shm()
    assert created[0][1].close_count == 1
    assert created[0][1].unlink_count == 1


def test_vllm_draft_logits_diagnostic_handles_missing_and_non_tensor_values() -> None:
    assert _describe_vllm_draft_logits(None, missing=True) == "missing"
    assert _describe_vllm_draft_logits(None) == "None(greedy)"
    assert _describe_vllm_draft_logits("MISSING") == "str"


def test_vllm_speculative_config_maps_dflash_contract() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DFLASH",
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": "/models/drafter",
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_maps_dspark_to_native_gpu_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint",
        lambda: False,
    )

    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "draft_load_config": {"load_format": "auto"},
        "method": "dspark",
        "model": str(model_path),
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_maps_dspark_to_dflash_on_npu_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": str(model_path),
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_keeps_native_dspark_on_npu_mrv2(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 1, "spec_verify_tokens": 5},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "draft_load_config": {"load_format": "auto"},
        "method": "dspark",
        "model": str(model_path),
        "num_speculative_tokens": 5,
    }


def test_vllm_mrv2_dspark_rejects_legacy_k_plus_one_alignment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "model_type": "qwen3",
          "markov_head_type": "vanilla",
          "sample_from_anchor": false,
          "block_size": 7
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_from_anchor=true"):
        build_vllm_speculative_config_from_drafter(
            _drafter(
                speculative_algorithm="DSPARK",
                model_path=str(model_path),
                rollout={"spec_steps": 1, "spec_verify_tokens": 5},
            )
        )


def test_vllm_mrv2_dspark_rejects_k_beyond_training_block(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "model_type": "qwen3",
          "markov_head_type": "vanilla",
          "block_size": 7
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exceeds the positions"):
        build_vllm_speculative_config_from_drafter(
            _drafter(
                speculative_algorithm="DSPARK",
                model_path=str(model_path),
                rollout={"spec_steps": 1, "spec_verify_tokens": 8},
            )
        )


@pytest.mark.parametrize(
    ("override", "field_name"),
    [
        ({"method": "dflash"}, "method"),
        ({"model": "/models/other-drafter"}, "model"),
        ({"num_speculative_tokens": 7}, "num_speculative_tokens"),
        ({"draft_sample_method": "probabilistic"}, "draft_sample_method"),
    ],
)
def test_vllm_mrv2_native_dspark_rejects_canonical_overrides(
    tmp_path, monkeypatch, override, field_name
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"does not allow.*canonical {field_name}="):
        build_vllm_speculative_config_from_drafter(
            _drafter(
                speculative_algorithm="DSPARK",
                model_path=str(model_path),
                rollout={"spec_steps": 1, "spec_verify_tokens": 5},
                vllm={"speculative_config_overrides": override},
            )
        )


def test_vllm_mrv2_skips_legacy_dspark_runtime_and_registry_alias(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._VLLM_DSPARK_REGISTRY_ALIAS_PATCHED",
        False,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._vllm_ascend_has_dspark_pr11153_k_query_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("MRV1 probe must not run")),
    )

    assert patch_vllm_dspark_runtime() is False
    assert patch_vllm_dspark_registry_aliases() is False


def test_vllm_dspark_loads_draft_checkpoint_without_changing_target_policy() -> None:
    target_load_config = SimpleNamespace(load_format="dummy")
    draft_load_config = SimpleNamespace(load_format="auto")
    observed_load_configs = []

    def upstream_load_dspark_model(target_model, vllm_config):
        observed_load_configs.append(vllm_config.load_config)
        return target_model

    utils_module = SimpleNamespace(load_dspark_model=upstream_load_dspark_model)
    speculator_module = SimpleNamespace(
        load_dspark_model=upstream_load_dspark_model
    )
    vllm_config = SimpleNamespace(
        load_config=target_load_config,
        speculative_config=SimpleNamespace(
            draft_load_config=draft_load_config,
        ),
    )
    target_model = object()

    assert (
        patch_vllm_dspark_draft_load_config(utils_module, speculator_module) is True
    )
    assert speculator_module.load_dspark_model(target_model, vllm_config) is target_model
    assert observed_load_configs == [draft_load_config]
    assert vllm_config.load_config is target_load_config
    assert utils_module.load_dspark_model is speculator_module.load_dspark_model


def test_vllm_dspark_restores_target_load_policy_when_draft_load_fails() -> None:
    target_load_config = SimpleNamespace(load_format="dummy")
    draft_load_config = SimpleNamespace(load_format="auto")

    def upstream_load_dspark_model(_target_model, vllm_config):
        assert vllm_config.load_config is draft_load_config
        raise RuntimeError("draft checkpoint failed")

    utils_module = SimpleNamespace(load_dspark_model=upstream_load_dspark_model)
    speculator_module = SimpleNamespace(
        load_dspark_model=upstream_load_dspark_model
    )
    vllm_config = SimpleNamespace(
        load_config=target_load_config,
        speculative_config=SimpleNamespace(
            draft_load_config=draft_load_config,
        ),
    )

    patch_vllm_dspark_draft_load_config(utils_module, speculator_module)
    with pytest.raises(RuntimeError, match="draft checkpoint failed"):
        speculator_module.load_dspark_model(object(), vllm_config)

    assert vllm_config.load_config is target_load_config


def test_vllm_dspark_gpu_probabilistic_sampling_requires_override(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint",
        lambda: False,
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
            vllm={
                "speculative_config_overrides": {"draft_sample_method": "probabilistic"}
            },
        )
    )

    assert config["method"] == "dspark"
    assert config["draft_sample_method"] == "probabilistic"


def test_vllm_dflash_validator_rejects_dspark_when_algorithm_is_dflash(
    tmp_path,
) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vLLM DFlash requires"):
        _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH")


def test_vllm_dflash_validator_accepts_a_dflash2_checkpoint(tmp_path) -> None:
    """DFLASH2 is rejected as an engine method and served as a DFlash checkpoint.

    That advice is only followable if the DFlash path accepts the DFlash2
    architecture.
    """
    model_path = tmp_path / "dflash2-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["DFlash2DraftModel"]}', encoding="utf-8"
    )

    _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH")


def test_vllm_dspark_validator_accepts_markov_head_config(tmp_path) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    _validate_vllm_dflash_drafter_config(model_path, algorithm="DSPARK")


def test_vllm_dspark_config_aliases_are_dflash_compatible() -> None:
    config = {
        "architectures": ["DFlashDSparkDraftModel"],
        "markov_head_type": "vanilla",
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
    }

    assert _normalize_dflash_target_layer_aliases(config) is True

    assert config["dflash_config"] == {
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
    }
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]


def _install_fake_vllm_ascend_modules(monkeypatch, dflash_cls, proposer_cls) -> None:
    root_module = types.ModuleType("vllm_ascend")
    spec_decode_module = types.ModuleType("vllm_ascend.spec_decode")
    dflash_module = types.ModuleType("vllm_ascend.spec_decode.dflash_proposer")
    proposer_module = types.ModuleType("vllm_ascend.spec_decode.llm_base_proposer")

    dflash_module.AscendDflashProposer = dflash_cls
    proposer_module.AscendSpecDecodeBaseProposer = proposer_cls
    spec_decode_module.dflash_proposer = dflash_module
    spec_decode_module.llm_base_proposer = proposer_module
    root_module.spec_decode = spec_decode_module

    monkeypatch.setitem(sys.modules, "vllm_ascend", root_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode", spec_decode_module)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.spec_decode.dflash_proposer", dflash_module
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.spec_decode.llm_base_proposer", proposer_module
    )


class _FakePR11153DflashProposer:
    def _num_query_per_req(self):
        return (
            self.num_speculative_tokens
            if self._is_dspark
            else 1 + self.num_speculative_tokens
        )

    def set_inputs_first_pass(self):
        return self._num_query_per_req(), "IS_DSPARK"


class _FakePR11153SpecDecodeBaseProposer:
    def _run_merged_draft(self):
        if hasattr(
            self.speculative_config.draft_model_config.hf_config, "markov_head_type"
        ):
            blk = self.num_speculative_tokens
            draft_token_ids = self.model.model.markov_head
            return draft_token_ids[:, 1:] if blk else None
        return None


class _FakeOldDSparkDflashProposer:
    def set_inputs_first_pass(self):
        return 1 + self.num_speculative_tokens


class _FakeOldDSparkSpecDecodeBaseProposer:
    def _run_merged_draft(self):
        if hasattr(
            self.speculative_config.draft_model_config.hf_config, "markov_head_type"
        ):
            blk = self.num_speculative_tokens + 1
            draft_token_ids = self.model.model.markov_head
            return draft_token_ids[:, 1:] if blk else None
        return None


def test_vllm_ascend_dspark_runtime_detector_accepts_pr11153_k_query(
    monkeypatch,
) -> None:
    _install_fake_vllm_ascend_modules(
        monkeypatch,
        _FakePR11153DflashProposer,
        _FakePR11153SpecDecodeBaseProposer,
    )

    assert _vllm_ascend_has_dspark_pr11153_k_query_runtime() is True


def test_vllm_ascend_dspark_runtime_detector_rejects_old_full_block_layout(
    monkeypatch,
) -> None:
    _install_fake_vllm_ascend_modules(
        monkeypatch,
        _FakeOldDSparkDflashProposer,
        _FakeOldDSparkSpecDecodeBaseProposer,
    )

    assert _vllm_ascend_has_dspark_pr11153_k_query_runtime() is False


def test_vllm_runtime_injects_native_config_and_worker_extension(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    config = {
        "trainer": {"default_local_dir": str(tmp_path)},
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(),
                "engine_kwargs": {"vllm": {"additional_config": {"existing_option": 1}}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"]["method"] == "eagle3"
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS
    assert engine_kwargs["additional_config"]["existing_option"] == 1
    sidecar_dir = engine_kwargs["additional_config"][
        vllm_runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY
    ]
    assert sidecar_dir.startswith(str(tmp_path / ".spec_decode_stats"))
    assert "/run-" in sidecar_dir.replace("\\", "/")

    # Reconfiguration in the same driver process must reuse its run-scoped
    # directory rather than splitting counters between two locations.
    configure_vllm_runtime_from_config(config)
    assert (
        engine_kwargs["additional_config"][
            vllm_runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY
        ]
        == sidecar_dir
    )


def test_vllm_runtime_installs_replica_bridge_without_drafter(monkeypatch) -> None:
    bridge_calls: list[bool] = []
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: bridge_calls.append(True) or True,
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {"name": "vllm", "drafter": {"enable": False}}
        }
    }

    assert configure_vllm_runtime_from_config(config) == {}
    assert bridge_calls == [True]


def test_vllm_http_actor_installs_import_guard_before_deserialization() -> None:
    captured = {}

    class FakeRemoteActorClass:
        def options(self, **options):
            captured["options"] = options
            return "configured"

    class FakeRay:
        @staticmethod
        def remote(actor_cls):
            captured["actor_cls"] = actor_cls
            return FakeRemoteActorClass()

    class Server:
        pass

    actor_class = vllm_runtime._remote_speco_vllm_http_server(FakeRay, Server)
    assert (
        actor_class.options(
            name="server",
            runtime_env={"env_vars": {"EXISTING": "1"}},
        )
        == "configured"
    )
    assert captured["actor_cls"] is Server
    assert captured["options"]["name"] == "server"
    runtime_env = captured["options"]["runtime_env"]
    setup_hook_path = (
        "verl_speco.integration.verl_npu_vllm_compat."
        "install_verl_npu_vllm_worker_process_compat"
    )
    assert runtime_env["env_vars"] == {
        "EXISTING": "1",
        "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": setup_hook_path,
    }
    assert runtime_env["worker_process_setup_hook"] == setup_hook_path
    assert json.loads(json.dumps(runtime_env)) == runtime_env


def test_vllm_runtime_injects_dspark_as_dflash_on_npu_and_worker_extension(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(
                    speculative_algorithm="DSPARK",
                    model_path=str(model_path),
                    rollout={"spec_steps": 3, "spec_verify_tokens": 16},
                ),
                "engine_kwargs": {"vllm": {}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"]["method"] == "dflash"
    assert engine_kwargs["speculative_config"]["num_speculative_tokens"] == 16
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS


def test_vllm_runtime_injects_native_dspark_on_npu_mrv2(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(
                    speculative_algorithm="DSPARK",
                    model_path=str(model_path),
                    rollout={"spec_steps": 1, "spec_verify_tokens": 5},
                ),
                "engine_kwargs": {"vllm": {}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"] == {
        "draft_sample_method": "greedy",
        "draft_load_config": {"load_format": "auto"},
        "method": "dspark",
        "model": str(model_path),
        "num_speculative_tokens": 5,
    }
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS


def test_vllm_mrv2_rejects_final_engine_speculative_override(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(
                    speculative_algorithm="DSPARK",
                    model_path=str(model_path),
                    rollout={"spec_steps": 1, "spec_verify_tokens": 5},
                ),
                "engine_kwargs": {"vllm": {"speculative_config": {"method": "dflash"}}},
            }
        }
    }

    with pytest.raises(ValueError, match="final speculative config"):
        configure_vllm_runtime_from_config(config)


def test_transformers_attention_layer_type_constants_compat(monkeypatch) -> None:
    transformers_module = types.ModuleType("transformers")
    configuration_utils_module = types.ModuleType("transformers.configuration_utils")
    transformers_module.configuration_utils = configuration_utils_module
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(
        sys.modules, "transformers.configuration_utils", configuration_utils_module
    )

    assert patch_transformers_attention_layer_type_constants() is True
    assert configuration_utils_module.ALLOWED_LAYER_TYPES
    assert (
        configuration_utils_module.ALLOWED_LAYER_TYPES
        == configuration_utils_module.ALLOWED_ATTENTION_LAYER_TYPES
    )
    assert patch_transformers_attention_layer_type_constants() is False


def test_import_compat_runs_before_vllm_worker_extension_import() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "verl_speco"
        / "integration"
        / "vllm_runtime.py"
    ).read_text(encoding="utf-8")

    extension_import_match = re.search(
        r"from\s+verl\.workers\.rollout\.vllm_rollout\.utils\s+import\s+"
        r"\(?\s*vLLMColocateWorkerExtension",
        source,
    )
    assert extension_import_match is not None
    extension_import = extension_import_match.start()
    assert (
        source.index("\npatch_transformers_attention_layer_type_constants()\n")
        < extension_import
    )
    assert source.index("\ninstall_verl_npu_vllm_import_compat()\n") < extension_import


def test_vllm_acceptance_stats_keep_stable_transport_keys() -> None:
    stats = _new_vllm_spec_decode_stats()
    scheduler_stats = SimpleNamespace(
        spec_decoding_stats=SimpleNamespace(
            num_drafts=4, num_draft_tokens=24, num_accepted_tokens=7
        )
    )

    _record_vllm_spec_decode_scheduler_stats(stats, scheduler_stats)

    assert _vllm_spec_decode_stats_to_metrics(stats) == {
        "spec_num_draft_tokens": 24.0,
        "spec_num_accepted_tokens": 7.0,
        "spec_num_verify_steps": 4.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": 4.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": 7.0,
    }


def test_vllm_http_server_transports_final_request_acceptance_stats() -> None:
    class Engine:
        def generate(self, *args, **kwargs):
            del args, kwargs

            async def outputs():
                yield SimpleNamespace(
                    metrics=SimpleNamespace(
                        request_spec_decode_stats=SimpleNamespace(
                            num_verify_steps=4,
                            num_draft_tokens=28,
                            num_accepted_tokens=9,
                        )
                    )
                )

            return outputs()

    class Upstream:
        async def generate(self, *args, **kwargs):
            async for _ in self.engine.generate(
                request_id=kwargs["request_id"], prompt="prompt", sampling_params={}
            ):
                pass
            return SimpleNamespace(extra_fields={"global_steps": 1})

    class Server(_SpecoVLLMHttpServerMixin, Upstream):
        pass

    server = Server()
    server.engine = Engine()
    output = asyncio.run(Server.generate(server, request_id="request-1"))

    assert output.extra_fields == {
        "global_steps": 1,
        "spec_num_draft_tokens": 28.0,
        "spec_num_accepted_tokens": 9.0,
        "spec_num_verify_steps": 4.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": 4.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": 9.0,
    }
    assert server._speco_vllm_request_spec_decode_stats == {}


def test_vllm_worker_output_records_acceptance_once_per_tp_group(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._record_vllm_spec_decode_acceptance",
        lambda worker, **values: recorded.append((worker.local_rank, values)),
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"a": [1, 2, 3], "b": [4, 5]},
        num_invalid_spec_tokens={"b": 1},
    )
    runner_output = SimpleNamespace(
        sampled_token_ids=[[8, 9, 10], [11]],
        req_id_to_index={"a": 0, "b": 1},
    )

    _record_vllm_worker_spec_decode_output(
        SimpleNamespace(local_rank=1), scheduler_output, runner_output
    )
    assert recorded == []
    _record_vllm_worker_spec_decode_output(
        SimpleNamespace(local_rank=0), scheduler_output, runner_output
    )
    assert [item[1]["num_accepted_tokens"] for item in recorded] == [2, 0]
    assert [item[1]["num_draft_tokens"] for item in recorded] == [3, 2]


def test_vllm_acceptance_sidecar_is_independent_of_log_stats(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        vllm_runtime.SPECO_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                vllm_runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(tmp_path),
            }
        ),
    )
    runtime = SimpleNamespace(log_stats=False)

    vllm_runtime._record_vllm_worker_spec_decode_output(
        runtime,
        SimpleNamespace(scheduled_spec_decode_tokens={"request-0": [1, 2, 3, 4]}),
        SimpleNamespace(req_id_to_index={"request-0": 0}, sampled_token_ids=[[1, 2, 3, 4]]),
    )

    assert vllm_runtime.read_vllm_spec_decode_sidecar_totals(str(tmp_path)) == {
        "drafts": 1.0,
        "accepted_tokens": 3.0,
        "draft_tokens": 4.0,
    }


@pytest.mark.parametrize("split_sampling", [False, True])
@pytest.mark.parametrize("async_output", [False, True])
def test_vllm_worker_execution_boundary_to_trainer_metric(
    monkeypatch, tmp_path, split_sampling, async_output
):
    """Exercise extension MRO, split sampling, deferred D2H and real sidecar IO."""
    from verl_speco.trainer.v1.speco_mixin import SpecoV1Mixin

    monkeypatch.delenv(vllm_runtime.SPECO_DRAFTER_CONFIG_ENV, raising=False)
    monkeypatch.delenv("VERL_SPECO_SGLANG_DRAFTER_CONFIG", raising=False)
    stats_dir = tmp_path / ".spec_decode_stats"
    resolved = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1, "prefill": 2},
        sampled_token_ids=[[7, 8, 9], [10], []],
    )

    class Deferred:
        calls = 0

        def get_output(self):
            self.calls += 1
            assert self.calls == 1
            return resolved

    # Mirror WorkerWrapperBase: append the extension to the worker's bases.
    class Base:
        pass

    class Worker(Base):
        def execute_model(self, scheduler_output):
            return None if split_sampling else self.output

        def sample_tokens(self, grammar_output):
            return self.output

    Worker.__bases__ += (SpecoVLLMColocateWorkerExtension,)
    worker = Worker()
    worker.rank = 0
    worker.log_stats = False
    worker.vllm_config = SimpleNamespace(
        additional_config={vllm_runtime.SPECO_VLLM_SPEC_DECODE_SIDECAR_KEY: str(stats_dir)},
        parallel_config=SimpleNamespace(world_size=2, tensor_parallel_size=2),
    )
    worker.output = Deferred() if async_output else resolved
    scheduled = SimpleNamespace(
        scheduled_spec_decode_tokens={"a": [1, 2, 3], "b": [4, 5], "prefill": [6, 7]},
        num_invalid_spec_tokens={"b": 1},
    )
    output = worker.execute_model(scheduled)
    if split_sampling:
        assert output is None
        assert not stats_dir.exists()
        output = worker.sample_tokens(None)
    assert output is worker.output
    if async_output:
        assert output.calls == 0  # No eager synchronization by the observer.
        assert not stats_dir.exists()
        # A following forward may mutate/reuse state before D2H completes.
        scheduled.scheduled_spec_decode_tokens.clear()
        assert output.get_output() is resolved

    assert vllm_runtime.read_vllm_spec_decode_sidecar_totals(str(stats_dir)) == {
        "drafts": 2.0, "accepted_tokens": 2.0, "draft_tokens": 4.0,
    }
    # Verify all rows (not just the first/every 32nd) were flushed. Never add
    # legacy Scheduler counters on top of the worker source.
    (stats_dir / "engine-legacy.counters").write_text("100 100 100\n", encoding="ascii")
    trainer = SpecoV1Mixin.__new__(SpecoV1Mixin)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)))
    assert trainer._speco_v1_spec_decode_sidecar_metrics() == {
        "drafter/spec_decode/mean_acceptance_length": 2.0,
    }
    assert trainer._speco_v1_spec_decode_sidecar_metrics() == {}


def test_vllm_worker_acceptance_uses_executor_output_rank(monkeypatch):
    recorded = []
    monkeypatch.setattr(vllm_runtime, "_record_vllm_spec_decode_acceptance", lambda *a, **kw: recorded.append(kw))
    output = SimpleNamespace(req_id_to_index={"a": 0}, sampled_token_ids=[[1, 2]])
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"a": [3, 4]})
    parallel = SimpleNamespace(world_size=4, tensor_parallel_size=2)
    for rank in range(4):
        worker = SimpleNamespace(rank=rank, local_rank=rank, vllm_config=SimpleNamespace(parallel_config=parallel))
        vllm_runtime._record_vllm_worker_spec_decode_output(worker, scheduled, output)
    assert len(recorded) == 1


def test_vllm_http_server_transports_stats_with_positional_request_id() -> None:
    """verl 0.9 calls server.generate with request_id in positional slot 3."""

    class Engine:
        def generate(self, *args, **kwargs):
            del args, kwargs

            async def outputs():
                yield SimpleNamespace(
                    metrics=SimpleNamespace(
                        request_spec_decode_stats=SimpleNamespace(
                            num_verify_steps=3,
                            num_draft_tokens=21,
                            num_accepted_tokens=5,
                        )
                    )
                )

            return outputs()

    class Upstream:
        async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
            del prompt_ids, sampling_params, kwargs
            async for _ in self.engine.generate(
                request_id=request_id, prompt="prompt", sampling_params={}
            ):
                pass
            return SimpleNamespace(extra_fields={"global_steps": 1})

    class Server(_SpecoVLLMHttpServerMixin, Upstream):
        pass

    server = Server()
    server.engine = Engine()
    output = asyncio.run(Server.generate(server, [1], {}, "request-1"))

    assert output.extra_fields == {
        "global_steps": 1,
        "spec_num_draft_tokens": 21.0,
        "spec_num_accepted_tokens": 5.0,
        "spec_num_verify_steps": 3.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": 3.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": 5.0,
    }
    assert server._speco_vllm_request_spec_decode_stats == {}


def test_trainer_keeps_public_acceptance_metric_name() -> None:
    trainer_source = (
        Path(__file__).resolve().parents[2]
        / "verl_speco"
        / "trainer"
        / "speco_ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert '"drafter/spec_decode/mean_acceptance_length"' in trainer_source


def test_trainer_drains_async_publish_before_checkpoint_and_validation() -> None:
    trainer_source = (
        Path(__file__).resolve().parents[2]
        / "verl_speco"
        / "trainer"
        / "speco_ray_trainer.py"
    ).read_text(encoding="utf-8")
    save_source = trainer_source.split("    def _save_checkpoint(self):", 1)[1].split(
        "    def _validate(", 1
    )[0]
    validate_source = trainer_source.split("    def _validate(", 1)[1]

    assert save_source.index("_speco_wait_pending_drafter_publish()") < save_source.index(
        "_speco_save_drafter_checkpoint(wait=True)"
    )
    assert validate_source.index(
        "_speco_wait_pending_drafter_publish()"
    ) < validate_source.index("super()._validate(")


def test_vllm_draft_update_attachment_is_idempotent() -> None:
    rollout = SimpleNamespace()

    assert attach_update_draft_weights_to_rollout(rollout) is rollout
    first = rollout.update_draft_weights
    assert first.__func__ is speco_vllm_update_draft_weights
    assert attach_update_draft_weights_to_rollout(rollout).update_draft_weights == first


def test_vllm_failed_draft_update_does_not_resume_generation(monkeypatch) -> None:
    import verl_speco.integration.vllm_runtime as runtime

    calls = []

    async def call_server(_adapter, method_name, *args, **kwargs):
        del args, kwargs
        calls.append(method_name)

    async def fail_update(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("worker update failed")

    sender_module = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    sender_module.BucketedWeightSender = object
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        sender_module,
    )
    monkeypatch.setattr(runtime, "_maybe_call_vllm_server_method", call_server)
    monkeypatch.setattr(
        runtime,
        "_load_env_drafter_config",
        lambda: {
            "training": {
                "draft_update_pause_generation": True,
                "draft_update_flush_before": False,
                "draft_update_flush_after": False,
            }
        },
    )
    monkeypatch.setattr(
        runtime, "_resolve_vllm_draft_update_use_shm", lambda *args: False
    )
    monkeypatch.setattr(
        runtime, "patch_verl_bucketed_weight_transfer_shm_reuse", lambda: False
    )
    adapter = SimpleNamespace(
        rollout_rank=0,
        replica_rank=0,
        config=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)
        ),
        _execute_method=fail_update,
    )

    with pytest.raises(RuntimeError, match="worker update failed"):
        asyncio.run(speco_vllm_update_draft_weights(adapter, {"weight": object()}))

    assert calls == ["abort_all_requests"]


def test_vllm_draft_update_pauses_flushes_and_resumes_after_commit(monkeypatch) -> None:
    import verl_speco.integration.vllm_runtime as runtime

    calls = []

    async def call_server(_adapter, method_name, *args, **kwargs):
        calls.append((method_name, args, kwargs))

    async def finish_update():
        calls.append(("finish_update", (), {}))

    async def execute_method(method_name, **kwargs):
        calls.append((method_name, (), kwargs))
        return finish_update()

    class Sender:
        def __init__(self, **kwargs):
            del kwargs

        async def async_send_weights(self, weights):
            del weights
            calls.append(("send_weights", (), {}))

    sender_module = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    sender_module.BucketedWeightSender = Sender
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        sender_module,
    )
    monkeypatch.setattr(runtime, "_maybe_call_vllm_server_method", call_server)
    monkeypatch.setattr(
        runtime,
        "_load_env_drafter_config",
        lambda: {
            "training": {
                "draft_update_pause_generation": True,
                "draft_update_flush_before": True,
                "draft_update_flush_after": True,
            }
        },
    )
    monkeypatch.setattr(runtime, "_resolve_vllm_draft_update_use_shm", lambda *args: False)
    monkeypatch.setattr(runtime, "patch_verl_bucketed_weight_transfer_shm_reuse", lambda: False)
    adapter = SimpleNamespace(
        rollout_rank=0,
        replica_rank=0,
        zmq_handle="ipc:///tmp/weights",
        config=SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)),
        _execute_method=execute_method,
    )

    asyncio.run(speco_vllm_update_draft_weights(adapter, {"weight": object()}, global_steps=9))

    assert [name for name, *_ in calls] == [
        "abort_all_requests",
        "update_draft_weights_from_ipc",
        "send_weights",
        "finish_update",
        "clear_kv_cache",
        "set_global_steps",
        "resume_generation",
    ]
    assert calls[0][2] == {"reset_prefix_cache": True}
    assert calls[-2][1] == (9,)


def test_vllm_draft_ipc_streams_buckets_without_cloning(monkeypatch) -> None:
    import verl_speco.integration.vllm_runtime as runtime

    cache_events = []

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            raise AssertionError("streamed draft updates must not detach bucket tensors")

        def clone(self):
            raise AssertionError("streamed draft updates must not clone bucket tensors")

    class InnerModel:
        def __init__(self):
            self.loaded = []
            self.rebuilds = 0

        def load_weights(self, weights):
            materialized = list(weights)
            self.loaded.append(
                [(name, tensor.value) for name, tensor in materialized]
            )
            return {name for name, _ in materialized}

        def _build_fused_kv_buffers(self):
            cache_events.append("rebuild")
            self.rebuilds += 1

    first_tensor = FakeTensor("first")
    second_tensor = FakeTensor("second")

    class FakeReceiver:
        def __init__(self, *, zmq_handle, device, use_shm):
            assert zmq_handle == "ipc://draft"
            assert device == "npu:0"
            assert use_shm is True

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("model.fc.weight", first_tensor)], False)
            first_tensor.value = "overwritten"
            on_bucket_received(
                [("_orig_mod.model.midlayer.norm.weight", second_tensor)], True
            )

    receiver_module = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    receiver_module.BucketedWeightReceiver = FakeReceiver
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        receiver_module,
    )
    platform_module = types.ModuleType("vllm.platforms")
    platform_module.current_platform = SimpleNamespace(device_type="npu")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_module)
    fake_torch = types.ModuleType("torch")
    fake_torch.npu = SimpleNamespace(
        synchronize=lambda: cache_events.append("synchronize"),
        empty_cache=lambda: cache_events.append("empty_cache"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runtime, "patch_verl_bucketed_weight_transfer_rebuild_ipc", lambda: False
    )
    monkeypatch.setattr(
        runtime, "patch_verl_bucketed_weight_transfer_shm_reuse", lambda: False
    )
    monkeypatch.setattr(runtime, "trim_process_host_memory", lambda: None)

    inner_model = InnerModel()
    draft_model = SimpleNamespace(
        model=inner_model,
        named_parameters=lambda: [],
    )
    extension = SpecoVLLMColocateWorkerExtension()
    extension.device = "npu:0"
    extension._get_speco_draft_zmq_handle = lambda: "ipc://draft"
    extension._speco_resolve_draft_model = lambda: (draft_model, None)
    extension._speco_draft_method = lambda: "dspark"
    extension._speco_diag_draft_state = lambda *_args, **_kwargs: None
    extension._speco_resolve_draft_proposer = lambda: None

    result = extension.update_draft_weights_from_ipc(use_shm=True)

    assert result == {"loaded_params": 2, "has_draft_model": True}
    assert inner_model.loaded == [
        [("fc.weight", "first")],
        [("layers.0.norm.weight", "second")],
    ]
    assert inner_model.rebuilds == 1
    assert extension._speco_draft_runtime_revision == 1
    assert cache_events == ["rebuild", "synchronize", "empty_cache"]


def test_vllm_fullgraph_storage_guard_allows_in_place_weight_update() -> None:
    class Parameter:
        shape = (3, 4)
        dtype = "bf16"
        device = "npu:0"

        def data_ptr(self):
            return 1234

        def stride(self):
            return (4, 1)

    parameter = Parameter()
    model = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
    before = SpecoVLLMColocateWorkerExtension._speco_parameter_storage_signatures(model)

    SpecoVLLMColocateWorkerExtension._speco_assert_parameter_storage_unchanged(
        model, before
    )


def test_vllm_fullgraph_storage_guard_rejects_parameter_replacement() -> None:
    class Parameter:
        shape = (3, 4)
        dtype = "bf16"
        device = "npu:0"

        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

        def stride(self):
            return (4, 1)

    holder = {"parameter": Parameter(1234)}
    model = SimpleNamespace(named_parameters=lambda: [("weight", holder["parameter"])])
    before = SpecoVLLMColocateWorkerExtension._speco_parameter_storage_signatures(model)
    holder["parameter"] = Parameter(5678)

    with pytest.raises(RuntimeError, match="replaced Parameter storage"):
        SpecoVLLMColocateWorkerExtension._speco_assert_parameter_storage_unchanged(
            model, before
        )


def test_vllm_fullgraph_metadata_refresh_preserves_captured_storage() -> None:
    class Buffer:
        shape = (2, 3)
        dtype = "bf16"
        device = "npu:0"

        def __init__(self, pointer, value):
            self.pointer = pointer
            self.value = value

        def copy_(self, other, non_blocking):
            assert non_blocking is False
            self.value = other.value
            return self

    old_kv = Buffer(100, 1.0)
    old_norm_0 = Buffer(200, 2.0)
    old_norm_1 = Buffer(201, 2.5)
    old_norm = [old_norm_0, old_norm_1]
    inner = SimpleNamespace(
        _fused_kv_weight=old_kv,
        _fused_kv_bias=None,
        _k_norm_weights=old_norm,
    )

    def rebuild():
        inner._fused_kv_weight = Buffer(300, 3.0)
        inner._fused_kv_bias = None
        inner._k_norm_weights = [Buffer(400, 4.0), Buffer(401, 4.5)]

    inner._build_fused_kv_buffers = rebuild

    SpecoVLLMColocateWorkerExtension._speco_rebuild_draft_metadata_buffers(
        SimpleNamespace(model=inner)
    )

    assert inner._fused_kv_weight is old_kv
    assert inner._fused_kv_weight.value == 3.0
    assert inner._k_norm_weights is old_norm
    assert inner._k_norm_weights[0] is old_norm_0
    assert inner._k_norm_weights[1] is old_norm_1
    assert inner._k_norm_weights[0].value == 4.0
    assert inner._k_norm_weights[1].value == 4.5


def test_vllm_fullgraph_metadata_refresh_supports_inference_buffers() -> None:
    torch = pytest.importorskip("torch")

    with torch.inference_mode():
        old_kv = torch.zeros((2, 3))
        old_norm = [torch.zeros((2, 3))]
    inner = SimpleNamespace(
        _fused_kv_weight=old_kv,
        _fused_kv_bias=None,
        _k_norm_weights=old_norm,
    )

    def rebuild() -> None:
        with torch.inference_mode():
            inner._fused_kv_weight = torch.full((2, 3), 3.0)
            inner._fused_kv_bias = None
            inner._k_norm_weights = [torch.full((2, 3), 4.0)]

    inner._build_fused_kv_buffers = rebuild

    SpecoVLLMColocateWorkerExtension._speco_rebuild_draft_metadata_buffers(
        SimpleNamespace(model=inner)
    )

    assert inner._fused_kv_weight is old_kv
    assert inner._k_norm_weights is old_norm
    assert torch.allclose(old_kv, torch.full((2, 3), 3.0))
    assert torch.allclose(old_norm[0], torch.full((2, 3), 4.0))


def test_vllm_fullgraph_metadata_refresh_rejects_sequence_length_change() -> None:
    class Buffer:
        shape = (2, 3)
        dtype = "bf16"
        device = "npu:0"

        def copy_(self, other, non_blocking):
            del other
            assert non_blocking is False
            return self

    inner = SimpleNamespace(
        _fused_kv_weight=Buffer(),
        _fused_kv_bias=None,
        _k_norm_weights=[Buffer()],
    )

    def rebuild():
        inner._fused_kv_weight = Buffer()
        inner._fused_kv_bias = None
        inner._k_norm_weights = [Buffer(), Buffer()]

    inner._build_fused_kv_buffers = rebuild

    with pytest.raises(
        RuntimeError,
        match=r"sequence length for _k_norm_weights: old=1, new=2",
    ):
        SpecoVLLMColocateWorkerExtension._speco_rebuild_draft_metadata_buffers(
            SimpleNamespace(model=inner)
        )


def test_vllm_dspark_target_sync_updates_only_outer_lm_head() -> None:
    class Weight:
        shape = (4, 3)
        device = "npu:0"
        dtype = "bf16"

        def __init__(self, value):
            self.value = value

        def to(self, *, device, dtype):
            assert device == self.device
            assert dtype == self.dtype
            return self

        def copy_(self, other, non_blocking):
            assert non_blocking is False
            self.value = other.value
            return self

    draft = SimpleNamespace(
        model=SimpleNamespace(online=Weight(7.0)),
        lm_head=SimpleNamespace(weight=Weight(0.0)),
    )
    target = SimpleNamespace(lm_head=SimpleNamespace(weight=Weight(5.0)))
    proposer = SimpleNamespace(
        model=draft,
        speculative_config=SimpleNamespace(method="dspark"),
    )
    extension = SpecoVLLMColocateWorkerExtension()
    extension.model_runner = SimpleNamespace(
        speculator=proposer,
        get_model=lambda: target,
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    online_before = draft.model.online.value

    try:
        assert extension._speco_sync_dspark_lm_head_from_target() == 1
    finally:
        monkeypatch.undo()
    assert draft.lm_head.weight.value == target.lm_head.weight.value
    assert draft.model.online.value == online_before


def test_vllm_target_sync_source_never_reloads_draft_checkpoint() -> None:
    source = getsource(SpecoVLLMColocateWorkerExtension.update_weights_from_ipc)

    assert "_speco_reload_draft_from_checkpoint" not in source
    assert "_speco_sync_dspark_lm_head_from_target" in source


def test_vllm_draft_loader_rejects_partial_online_update() -> None:
    requested = ["layers.0.self_attn.q_proj.weight", "norm.weight"]

    with pytest.raises(RuntimeError, match="complete online update"):
        SpecoVLLMColocateWorkerExtension._speco_validate_loaded_draft_weights(
            requested,
            {"layers.0.self_attn.qkv_proj.weight"},
            draft_method="dspark",
        )
