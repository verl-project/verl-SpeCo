from __future__ import annotations

import importlib
import os

import pytest


BACKEND = os.environ.get("SPECO_RUNTIME_BACKEND")
PLATFORM = os.environ.get("SPECO_RUNTIME_PLATFORM")


def _import(path: str):
    return importlib.import_module(path)


def test_selected_runtime_backend_is_explicit() -> None:
    assert BACKEND in {"vllm", "sglang"}
    assert PLATFORM in {"gpu", "npu"}


@pytest.mark.skipif(BACKEND != "vllm", reason="vLLM contract job only")
def test_vllm_eagle3_api_surface() -> None:
    vllm = _import("vllm")

    if PLATFORM == "gpu":
        eagle = _import("vllm.v1.spec_decode.eagle")
        eagle3_model = _import("vllm.model_executor.models.llama_eagle3")
        assert hasattr(eagle, "EagleProposer")
        assert hasattr(eagle3_model, "Eagle3LlamaForCausalLM")

    runtime = _import("verl_speco.integration.vllm_runtime")
    config = runtime.build_vllm_speculative_config_from_drafter(
        {
            "enable": True,
            "enable_drafter_training": True,
            "speculative_algorithm": "EAGLE3",
            "model_path": "/models/eagle3",
            "rollout": {"spec_steps": 3},
            "vllm": {"draft_tensor_parallel_size": 1},
        },
        runtime_version=vllm.__version__,
    )
    assert config["method"] == "eagle3"
    assert config["draft_tensor_parallel_size"] == 1


@pytest.mark.skipif(BACKEND != "sglang", reason="SGLang contract job only")
def test_sglang_supported_api_surface() -> None:
    sglang = _import("sglang")
    _import("sglang.srt.entrypoints.engine")
    server_args = _import("sglang.srt.server_args")
    io_struct = _import("sglang.srt.managers.io_struct")
    model_runner = _import("sglang.srt.model_executor.model_runner")
    sampler = _import("sglang.srt.layers.sampler")

    assert hasattr(server_args, "ServerArgs")
    assert hasattr(io_struct, "UpdateWeightsFromTensorReqInput")
    assert hasattr(model_runner, "LocalSerializedTensor")
    assert hasattr(sampler, "Sampler")

    adapter = _import("verl_speco.integration.sglang_adapter")
    expected_rope_patch = os.environ.get("SPECO_EXPECT_QWEN3_ROPE_COMPAT_PATCH") == "1"
    assert (
        adapter.sglang_needs_qwen3_rope_compat_patch(sglang.__version__)
        is expected_rope_patch
    )
