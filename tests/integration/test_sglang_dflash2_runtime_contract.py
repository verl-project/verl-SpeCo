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
"""SGLang co-training contract for the DFlash2 drafter (served through DFLASH)."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import sys
import types

import pytest

from verl_speco.integration import sglang_runtime
from verl_speco.integration.sglang_adapter import (
    _call_flashinfer_plan_with_abi_compat,
)
from verl_speco.integration.sglang_runtime import (
    _assert_sglang_supports_dflash2,
    _drafter_uses_dflash_aux_hidden,
    _is_sglang_draft_model,
    _server_args_overrides_from_drafter,
    _sglang_draft_param_name,
    _sglang_spec_decode_extra_fields,
    _validate_sglang_dflash2_block_size,
    speco_sglang_draft_weight_loader,
    speco_sglang_target_weight_loader,
)

_SUPPORTED_FIELDS = {
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_draft_attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "enable_return_hidden_states",
    "enable_weights_cpu_backup",
    "enable_draft_weights_cpu_backup",
}

_DFLASH2_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "model_type": "qwen3",
    "dflash_config": {
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "mask_token_id": 151669,
        "target_layer_ids": [0, 8, 16, 24, 32],
    },
}


def test_sglang_acceptance_stats_use_vllm_compatible_transport_keys() -> None:
    assert _sglang_spec_decode_extra_fields(
        {"spec_verify_ct": 4, "spec_num_correct_drafts": 7}
    ) == {
        "_speco_vllm_spec_decode_drafts": 4.0,
        "_speco_vllm_spec_decode_accepted_tokens": 7.0,
    }
    assert _sglang_spec_decode_extra_fields({"spec_verify_ct": 0}) == {}


@pytest.mark.parametrize(
    ("name", "algorithm", "expected"),
    [
        (
            "module.draft_model.markov_head.markov_w2.weight",
            "DSPARK",
            "markov_head.markov_w2.weight",
        ),
        (
            "model.draft_model.midlayer.self_attn.q_proj.weight",
            "DSPARK",
            "layers.0.self_attn.q_proj.weight",
        ),
        (
            "_orig_mod.draft_model.candidate_selector.predecessor_codebook.weight",
            "DFLASH2",
            "candidate_selector.predecessor_codebook",
        ),
    ],
)
def test_sglang_draft_publish_uses_vllm_compatible_parameter_names(
    name, algorithm, expected
) -> None:
    assert _sglang_draft_param_name(name, algorithm) == expected


def test_sglang_weight_update_preserves_http_error_message_and_route_fields() -> None:
    posted = {}

    class Response:
        status = 400

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"success": False, "message": "draft shard 1 is missing"}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, *, json):
            posted.update(url=url, json=json)
            return Response()

    request = types.SimpleNamespace(
        serialized_named_tensors=[b"tensor payload"],
        load_format=sglang_runtime.SPECO_DRAFT_WEIGHT_LOADER,
        flush_cache=False,
        abort_all_requests=False,
        disable_draft_model=False,
        weight_version="step-2",
        torch_empty_cache=False,
    )
    engine = types.SimpleNamespace(
        _get_session=lambda: Session(),
        server_args=types.SimpleNamespace(host="127.0.0.1", port=30000),
    )

    result = asyncio.run(
        sglang_runtime._sgl_http_update_weights_from_tensor(
            engine,
            request,
            request_fields=frozenset(
                {
                    "abort_all_requests",
                    "disable_draft_model",
                    "weight_version",
                    "torch_empty_cache",
                }
            ),
        )
    )

    assert result == {"success": False, "message": "draft shard 1 is missing"}
    assert posted["url"] == "http://127.0.0.1:30000/update_weights_from_tensor"
    assert posted["json"]["serialized_named_tensors"] == [
        base64.b64encode(b"tensor payload").decode("utf-8")
    ]
    assert posted["json"]["disable_draft_model"] is False
    assert posted["json"]["weight_version"] == "step-2"
    assert "disable_target_model" not in posted["json"]


@pytest.mark.parametrize("enable_drafter_training", [False, True])
def test_sglang_generate_exports_acceptance_stats_without_hidden_collection(
    monkeypatch, enable_drafter_training
) -> None:
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class GenerateReqInput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    io_struct.GenerateReqInput = GenerateReqInput
    for parent in (
        "sglang",
        "sglang.srt",
        "sglang.srt.managers",
        "verl",
        "verl.workers",
        "verl.workers.rollout",
        "verl.workers.rollout.replica",
        "verl.workers.rollout.sglang_rollout",
    ):
        monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)

    replica = sys.modules["verl.workers.rollout.replica"]

    class TokenOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    replica.TokenOutput = TokenOutput
    rollout_utils = types.ModuleType("verl.workers.rollout.sglang_rollout.utils")
    rollout_utils.SGLANG_LORA_NAME = "adapter"
    monkeypatch.setitem(
        sys.modules, "verl.workers.rollout.sglang_rollout.utils", rollout_utils
    )

    class TokenizerManager:
        def generate_request(self, request, _):
            async def responses():
                yield {
                    "output_ids": [21, 22, 23],
                    "meta_info": {
                        "finish_reason": {"type": "length"},
                        "spec_verify_ct": 2,
                        "spec_num_correct_drafts": 3,
                    },
                }

            return responses()

    class UpstreamServer:
        async def generate(self, *args, **kwargs):
            raise AssertionError("enabled speculative rollout must retain meta_info")

    class Server(sglang_runtime._SpecoSGLangHttpServerMixin, UpstreamServer):
        pass

    server = Server()
    server._speco_drafter_config = {
        "enable": True,
        "enable_drafter_training": enable_drafter_training,
        "speculative_algorithm": "DSPARK",
        "training": {"collect_hidden_states_from_old_logprob": True},
    }
    server.config = types.SimpleNamespace(
        max_model_len=32,
        response_length=8,
        prompt_length=8,
        enable_rollout_routing_replay=False,
    )
    server.model_config = types.SimpleNamespace(lora_rank=0)
    server.global_steps = 1
    server.replica_rank = 0
    server.tokenizer_manager = TokenizerManager()

    output = asyncio.run(
        server.generate(
            sglang_runtime.torch.tensor([1, 2]),
            {"max_tokens": 3},
            "request-1",
        )
    )

    assert output.extra_fields == {
        "global_steps": 1,
        "_speco_vllm_spec_decode_drafts": 2.0,
        "_speco_vllm_spec_decode_accepted_tokens": 3.0,
    }


def _write_drafter(tmp_path):
    model_path = tmp_path / "dflash2-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(_DFLASH2_CONFIG), encoding="utf-8"
    )
    return model_path


def _drafter(model_path, **overrides):
    config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "DFLASH2",
        "model_path": str(model_path),
        "rollout": {"spec_steps": 1, "spec_topk": 1, "spec_verify_tokens": 8},
        "training": {
            "dflash2_block_size": 8,
            "collect_hidden_states_from_old_logprob": True,
        },
    }
    config.update(overrides)
    return config


@pytest.fixture
def sglang_has_dflash2(monkeypatch):
    monkeypatch.setattr(sglang_runtime, "_sglang_supports_dflash2", lambda: True)


def test_dflash2_maps_to_the_dflash_server_algorithm(
    tmp_path, sglang_has_dflash2
) -> None:
    overrides = _server_args_overrides_from_drafter(
        _drafter(_write_drafter(tmp_path)), _SUPPORTED_FIELDS
    )

    assert overrides["speculative_algorithm"] == "DFLASH"
    assert overrides["speculative_num_draft_tokens"] == 8
    # sglang rejects return_hidden_states for the DFLASH worker; DFlash2 runs
    # collect hidden states from the old-logprob pass instead.
    assert overrides["enable_return_hidden_states"] is False


def test_dflash2_rejects_engine_hidden_state_collection(
    tmp_path, sglang_has_dflash2
) -> None:
    """sglang rejects return_hidden_states for DFLASH, so fail in the overlay."""
    with pytest.raises(ValueError, match="collect_hidden_states_from_old_logprob"):
        _server_args_overrides_from_drafter(
            _drafter(
                _write_drafter(tmp_path),
                training={
                    "dflash2_block_size": 8,
                    "collect_hidden_states_from_sgl": True,
                },
            ),
            _SUPPORTED_FIELDS,
        )


def test_dflash2_refuses_an_sglang_without_the_draft_class(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sglang_runtime, "_sglang_supports_dflash2", lambda: False)

    with pytest.raises(ValueError, match="sglang main"):
        _server_args_overrides_from_drafter(
            _drafter(_write_drafter(tmp_path)), _SUPPORTED_FIELDS
        )


def test_dflash2_capability_probe_is_skipped_without_sglang(monkeypatch) -> None:
    monkeypatch.setattr(sglang_runtime, "_sglang_supports_dflash2", lambda: None)
    _assert_sglang_supports_dflash2()


def test_dflash2_capability_probe_reads_the_installed_sglang(monkeypatch) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert sglang_runtime._sglang_supports_dflash2() is None

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    for parent in ("sglang", "sglang.srt", "sglang.srt.models"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    dflash_module = types.ModuleType(sglang_runtime._SGLANG_DFLASH_MODULE)
    monkeypatch.setitem(
        sys.modules, sglang_runtime._SGLANG_DFLASH_MODULE, dflash_module
    )
    assert sglang_runtime._sglang_supports_dflash2() is False

    dflash_module.DFlash2DraftModel = object
    assert sglang_runtime._sglang_supports_dflash2() is True


def test_dflash2_rejects_a_plain_dflash_checkpoint(
    tmp_path, sglang_has_dflash2
) -> None:
    model_path = tmp_path / "dflash-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["DFlashDraftModel"]}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="DFlash2 drafter checkpoint"):
        _server_args_overrides_from_drafter(_drafter(model_path), _SUPPORTED_FIELDS)


def test_dflash2_block_size_equals_sglang_num_draft_tokens(
    tmp_path, sglang_has_dflash2
) -> None:
    """SGLang uses num_draft_tokens AS the block size (no bonus-token offset)."""
    with pytest.raises(ValueError, match="spec_verify_tokens=7 but block_size=8"):
        _server_args_overrides_from_drafter(
            _drafter(
                _write_drafter(tmp_path),
                rollout={"spec_steps": 1, "spec_topk": 1, "spec_verify_tokens": 7},
            ),
            _SUPPORTED_FIELDS,
        )


def test_dflash2_block_size_falls_back_to_the_checkpoint(tmp_path) -> None:
    drafter = _drafter(_write_drafter(tmp_path), training={})
    _validate_sglang_dflash2_block_size(drafter, 8)
    with pytest.raises(ValueError, match="block_size=8"):
        _validate_sglang_dflash2_block_size(drafter, 4)
    # Training config wins; nothing to compare against is accepted.
    _validate_sglang_dflash2_block_size(
        {**drafter, "training": {"dflash2_block_size": 4}}, 4
    )
    _validate_sglang_dflash2_block_size({"model_path": None, "training": {}}, 4)
    _validate_sglang_dflash2_block_size(drafter, None)


def test_dflash2_uses_the_dflash_aux_hidden_layout() -> None:
    assert _drafter_uses_dflash_aux_hidden(
        {
            "enable": True,
            "enable_drafter_training": True,
            "speculative_algorithm": "DFLASH2",
            "training": {"collect_hidden_states_from_sgl": True},
        }
    )
    assert not _drafter_uses_dflash_aux_hidden(
        {
            "enable": True,
            "enable_drafter_training": True,
            "speculative_algorithm": "DFLASH2",
            "training": {"collect_hidden_states_from_sgl": True, "use_logits": True},
        }
    )


class _FakeConfig:
    def __init__(self, architectures=None, draft_vocab_size=None):
        self.architectures = architectures or []
        self.draft_vocab_size = draft_vocab_size


def test_draft_model_detection_covers_the_dflash_family() -> None:
    """SGLang's DFlash draft classes spell neither "eagle" nor draft_vocab_size.

    Without the dflash checks the draft weight loader silently dropped every
    published DFlash-family tensor.
    """

    class DFlash2DraftModel:
        config = _FakeConfig(architectures=["DFlash2DraftModel"])

    class DFlashDraftModel:
        config = _FakeConfig(architectures=["DFlashDraftModel"])

    class Qwen3DSparkModel:
        config = _FakeConfig(architectures=["Qwen3DSparkModel"])

    class EagleDraft:
        config = _FakeConfig(architectures=["Eagle3LlamaForCausalLM"])

    class Qwen3ForCausalLM:
        config = _FakeConfig(architectures=["Qwen3ForCausalLM"])

    assert _is_sglang_draft_model(DFlash2DraftModel())
    assert _is_sglang_draft_model(DFlashDraftModel())
    assert _is_sglang_draft_model(Qwen3DSparkModel())
    assert _is_sglang_draft_model(EagleDraft())
    assert not _is_sglang_draft_model(Qwen3ForCausalLM())


def test_dspark_maps_gamma_to_sglang_verify_window(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3DSparkModel"], "block_size": 7}),
        encoding="utf-8",
    )
    overrides = _server_args_overrides_from_drafter(
        {
            "enable": True,
            "speculative_algorithm": "DSPARK",
            "model_path": str(tmp_path),
            "rollout": {"spec_steps": 1, "spec_topk": 1, "spec_verify_tokens": 7},
        },
        _SUPPORTED_FIELDS,
    )

    assert overrides["speculative_algorithm"] == "DSPARK"
    assert overrides["speculative_num_draft_tokens"] == 8
    assert "speculative_draft_attention_backend" not in overrides
    assert "prefill_attention_backend" not in overrides
    assert "decode_attention_backend" not in overrides
    assert "flashinfer_plan_abi" in sglang_runtime._default_sglang_verl_patches(
        {"enable": True, "speculative_algorithm": "DSPARK"}
    )


def test_flashinfer_plan_compat_drops_only_new_uniform_q_len_argument() -> None:
    calls = []

    def legacy_plan(*args):
        calls.append(args)
        if len(args) == 20:
            raise TypeError("Expected 19 but got 20 arguments")
        return "legacy-plan"

    assert (
        _call_flashinfer_plan_with_abi_compat(legacy_plan, *range(20)) == "legacy-plan"
    )
    assert [len(args) for args in calls] == [20, 19]
    assert calls[-1] == tuple(range(19))


def test_flashinfer_plan_compat_preserves_unrelated_type_errors() -> None:
    def broken_plan(*args):
        raise TypeError("different failure")

    with pytest.raises(TypeError, match="different failure"):
        _call_flashinfer_plan_with_abi_compat(broken_plan, *range(20))


@pytest.mark.parametrize(
    ("checkpoint_overrides", "error"),
    [
        ({"architectures": ["DFlashDraftModel"]}, "DSpark checkpoint"),
        ({"sample_from_anchor": False}, "sample_from_anchor=true"),
        ({"block_size": 8}, "checkpoint block_size"),
    ],
)
def test_dspark_rejects_incompatible_checkpoint(
    tmp_path, checkpoint_overrides, error
) -> None:
    config = {"architectures": ["Qwen3DSparkModel"], "block_size": 7}
    config.update(checkpoint_overrides)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        _server_args_overrides_from_drafter(
            {
                "enable": True,
                "speculative_algorithm": "DSPARK",
                "model_path": str(tmp_path),
                "rollout": {"spec_verify_tokens": 7},
            },
            _SUPPORTED_FIELDS,
        )


def test_dspark_weight_updates_reach_only_the_selected_model() -> None:
    class DSparkDraftModel:
        config = _FakeConfig(architectures=["DSparkDraftModel"])

        def __init__(self):
            self.loaded = []

        def load_weights(self, weights):
            self.loaded.extend(weights)

    class Qwen3ForCausalLM(DSparkDraftModel):
        config = _FakeConfig(architectures=["Qwen3ForCausalLM"])

    draft = DSparkDraftModel()
    target = Qwen3ForCausalLM()
    weights = [("markov_head.weight", object())]
    speco_sglang_draft_weight_loader(draft, weights)
    speco_sglang_draft_weight_loader(target, weights)
    speco_sglang_target_weight_loader(draft, weights)
    speco_sglang_target_weight_loader(target, weights)

    assert draft.loaded == weights
    assert target.loaded == weights


# --- record field names across sglang's dataclass and msgspec eras -----------


def _install_fake_sglang_modules(monkeypatch, **modules):
    for parent in (
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.speculative",
    ):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_record_field_names_prefer_sglang_record_fields(monkeypatch) -> None:
    """sglang main's ServerArgs is a msgspec Struct: dataclasses.fields raises."""

    class _Field:
        def __init__(self, name):
            self.name = name

    arg_utils = types.ModuleType("sglang.srt.arg_groups.arg_utils")
    arg_utils.record_fields = lambda cls: [
        _Field("custom_weight_loader"),
        _Field("tp_size"),
    ]
    for parent in ("sglang", "sglang.srt", "sglang.srt.arg_groups"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.arg_utils", arg_utils)

    class ServerArgs:  # neither a dataclass nor a Struct
        pass

    assert sglang_runtime._record_field_names(ServerArgs) == {
        "custom_weight_loader",
        "tp_size",
    }


def test_record_field_names_fall_back_without_the_helper(monkeypatch) -> None:
    import dataclasses

    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.arg_utils", None)

    class StructLike:
        __struct_fields__ = ("load_format", "disable_draft_model")

    @dataclasses.dataclass
    class Legacy:
        load_format: str = "auto"

    assert sglang_runtime._record_field_names(StructLike) == {
        "load_format",
        "disable_draft_model",
    }
    assert sglang_runtime._record_field_names(Legacy) == {"load_format"}
    assert sglang_runtime._record_field_names(object) == frozenset()


def test_custom_weight_loader_probe_reads_struct_server_args(monkeypatch) -> None:
    """sglang main's ServerArgs has no __dataclass_fields__, so the old probe
    answered False and the draft publish lost its route marker."""
    server_args = types.ModuleType("sglang.srt.server_args")

    class ServerArgs:
        __struct_fields__ = ("model_path", "custom_weight_loader")

    server_args.ServerArgs = ServerArgs
    _install_fake_sglang_modules(monkeypatch, **{"sglang.srt.server_args": server_args})
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.arg_utils", None)
    assert sglang_runtime._supports_sglang_custom_weight_loader() is True

    ServerArgs.__struct_fields__ = ("model_path",)
    assert sglang_runtime._supports_sglang_custom_weight_loader() is False


def test_native_weight_route_does_not_leak_marker_to_model_loader(monkeypatch) -> None:
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class UpdateWeightsFromTensorReqInput:
        __struct_fields__ = ("load_format", "disable_draft_model")

    io_struct.UpdateWeightsFromTensorReqInput = UpdateWeightsFromTensorReqInput
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)
    monkeypatch.setattr(
        sglang_runtime, "_supports_sglang_custom_weight_loader", lambda: True
    )

    assert (
        sglang_runtime._sglang_route_load_format(
            "disable_draft_model", sglang_runtime.SPECO_TARGET_WEIGHT_LOADER
        )
        is None
    )
    assert (
        sglang_runtime._sglang_route_load_format(
            "disable_target_model", sglang_runtime.SPECO_DRAFT_WEIGHT_LOADER
        )
        == sglang_runtime.SPECO_DRAFT_WEIGHT_LOADER
    )


def test_verl_server_args_probe_accepts_current_sglang_records(monkeypatch) -> None:
    class _Field:
        def __init__(self, name):
            self.name = name

    arg_utils = types.ModuleType("sglang.srt.arg_groups.arg_utils")
    arg_utils.record_fields = lambda cls: [
        _Field("model_path"),
        _Field("enable_weights_cpu_backup"),
    ]
    for parent in ("sglang", "sglang.srt", "sglang.srt.arg_groups"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.arg_utils", arg_utils)

    class ServerArgs:
        pass

    upstream = types.SimpleNamespace(
        dataclasses=dataclasses,
        ServerArgs=ServerArgs,
    )
    sglang_runtime._install_verl_server_args_fields_compat(upstream)

    assert {field.name for field in upstream.dataclasses.fields(ServerArgs)} == {
        "enable_weights_cpu_backup",
        "model_path",
    }
    with pytest.raises(TypeError):
        dataclasses.fields(ServerArgs)


def test_http_server_installs_fields_compat_inside_actor(monkeypatch) -> None:
    import dataclasses

    class ServerArgs:
        __struct_fields__ = ("model_path", "enable_weights_cpu_backup")

    upstream = types.ModuleType("fake_verl_async_sglang_server")
    upstream.dataclasses = dataclasses
    upstream.ServerArgs = ServerArgs
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)

    class UpstreamServer:
        async def launch_server(self):
            return {
                field.name for field in upstream.dataclasses.fields(upstream.ServerArgs)
            }

    UpstreamServer.launch_server.__module__ = upstream.__name__
    monkeypatch.setattr(
        sglang_runtime, "install_sglang_server_actor_runtime", lambda: {}
    )
    monkeypatch.setattr(
        sglang_runtime, "_install_verl_launch_subprocesses_compat", lambda: None
    )

    class SpecoServer(sglang_runtime._SpecoSGLangHttpServerMixin, UpstreamServer):
        pass

    fields = asyncio.run(SpecoServer().launch_server())
    assert fields == {"model_path", "enable_weights_cpu_backup"}


def test_verl_legacy_launcher_uses_current_sglang_engine_shape(monkeypatch) -> None:
    entrypoints = types.ModuleType("sglang.srt.entrypoints")
    http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
    callback_attrs = {
        "init_tokenizer_manager_func": ("init_tokenizer_manager", object()),
        "run_scheduler_process_func": ("run_scheduler_process", object()),
        "run_detokenizer_process_func": ("run_detokenizer_process", object()),
    }
    for attr_name, callback in callback_attrs.values():
        setattr(http_server, attr_name, callback)

    received = {}

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *args, **kwargs):
            received.update(kwargs)
            scheduler_init_result = types.SimpleNamespace(
                scheduler_infos=[{"max_total_num_tokens": 1024}]
            )
            return (
                "tokenizer",
                "template",
                "port_args",
                scheduler_init_result,
                "watchdog",
            )

    for name, (_, callback) in callback_attrs.items():
        setattr(Engine, name, callback)
    http_server.run_scheduler_process = object()
    http_server.Engine = Engine
    entrypoints.http_server = http_server
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", http_server)

    sglang_runtime._install_verl_launch_subprocesses_compat()
    result = http_server._launch_subprocesses(server_args="args")

    assert result[:3] == (
        "tokenizer",
        "template",
        {"max_total_num_tokens": 1024},
    )
    assert received == {
        "server_args": "args",
        **{name: callback for name, (_, callback) in callback_attrs.items()},
    }


def test_verl_legacy_launcher_applies_drafter_config_at_scheduler_boundary(
    monkeypatch,
) -> None:
    entrypoints = types.ModuleType("sglang.srt.entrypoints")
    http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
    received = {}

    class ServerArgs:
        __struct_fields__ = (
            "speculative_algorithm",
            "speculative_draft_model_path",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "speculative_draft_attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
            "enable_return_hidden_states",
            "enable_weights_cpu_backup",
            "enable_draft_weights_cpu_backup",
        )

        def __init__(self):
            for field in self.__struct_fields__:
                setattr(self, field, None)

    class Engine:
        init_tokenizer_manager_func = object()
        run_scheduler_process_func = object()
        run_detokenizer_process_func = object()

        @classmethod
        def _launch_subprocesses(cls, *, server_args, **kwargs):
            received["server_args"] = server_args
            return ("tokenizer", "template", "port_args", {}, "watchdog")

    http_server.Engine = Engine
    http_server.init_tokenizer_manager = object()
    http_server.run_scheduler_process = object()
    http_server.run_detokenizer_process = object()
    entrypoints.http_server = http_server
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", http_server)
    monkeypatch.setenv(
        sglang_runtime.SPECO_SGLANG_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                "enable": True,
                "speculative_algorithm": "DSPARK",
                "model_path": "/models/dspark",
                "rollout": {
                    "spec_steps": 1,
                    "spec_topk": 1,
                    "spec_verify_tokens": 7,
                },
            }
        ),
    )

    sglang_runtime._install_verl_launch_subprocesses_compat()
    http_server._launch_subprocesses(server_args=ServerArgs())

    server_args = received["server_args"]
    assert server_args.speculative_algorithm == "DSPARK"
    assert server_args.speculative_draft_model_path == "/models/dspark"
    assert server_args.speculative_num_draft_tokens == 8
    assert server_args.speculative_draft_attention_backend is None
    assert server_args.prefill_attention_backend is None
    assert server_args.decode_attention_backend is None
    assert server_args.enable_draft_weights_cpu_backup is True


def test_engine_launcher_applies_drafter_config_when_constructor_bypasses_init(
    monkeypatch,
) -> None:
    entrypoints = types.ModuleType("sglang.srt.entrypoints")
    http_server = types.ModuleType("sglang.srt.entrypoints.http_server")
    received = {}

    class StructLikeMeta(type):
        def __call__(cls):
            instance = object.__new__(cls)
            for field in cls.__struct_fields__:
                setattr(instance, field, None)
            return instance

    class ServerArgs(metaclass=StructLikeMeta):
        __struct_fields__ = tuple(_SUPPORTED_FIELDS)

        def __init__(self):
            raise AssertionError("Struct-like construction must bypass __init__")

    class Engine:
        @classmethod
        def _launch_subprocesses(cls, *, server_args, **kwargs):
            received["server_args"] = server_args
            return ("tokenizer", "template", "port_args", {}, "watchdog")

    http_server.Engine = Engine
    http_server._launch_subprocesses = lambda **kwargs: None
    entrypoints.http_server = http_server
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", http_server)
    monkeypatch.setenv(
        sglang_runtime.SPECO_SGLANG_DRAFTER_CONFIG_ENV,
        json.dumps(
            {
                "enable": True,
                "speculative_algorithm": "DSPARK",
                "model_path": "/models/dspark",
                "rollout": {
                    "spec_steps": 1,
                    "spec_topk": 1,
                    "spec_verify_tokens": 7,
                },
            }
        ),
    )

    sglang_runtime._install_verl_launch_subprocesses_compat()
    Engine._launch_subprocesses(server_args=ServerArgs())

    server_args = received["server_args"]
    assert server_args.speculative_algorithm == "DSPARK"
    assert server_args.speculative_draft_model_path == "/models/dspark"
    assert server_args.speculative_num_draft_tokens == 8
    assert server_args.enable_draft_weights_cpu_backup is True
