# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from verl_speco.probe_deepseek_v4_feature_alignment import (
    AlignmentCaptureWorkerExtension,
    decode_logits_reply,
    encode_logits_reply,
    error_metrics,
    select_prompt_logits,
)


def test_precision_control_restores_values_and_preserves_bf16_and_storage():
    model = torch.nn.Module()
    model.register_parameter(
        "sensitive", torch.nn.Parameter(torch.tensor([1 / 3, -1 / 7]))
    )
    model.register_parameter(
        "ordinary", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    )
    original = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    worker = AlignmentCaptureWorkerExtension()
    worker.model_runner = SimpleNamespace(get_model=lambda: model)
    assert worker.speco_alignment_precision("round_fp32")["parameters"] == ["sensitive"]
    assert torch.equal(model.sensitive, original["sensitive"].bfloat16().float())
    assert torch.equal(model.ordinary, original["ordinary"])
    with pytest.raises(RuntimeError, match="already active"):
        worker.speco_alignment_precision("round_fp32")
    assert worker.speco_alignment_precision("restore")["parameters"] == 1
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name]) and value.data_ptr() == pointers[name]


def test_binary_rpc_roundtrip_preserves_logits_dtype_and_head_weights():
    values = [torch.arange(8, dtype=torch.bfloat16).reshape(2, 4), torch.ones(1, 4)]
    head = {"model.hc_head_fn": torch.ones(2, 3, dtype=torch.float32)}
    reply = encode_logits_reply(values, head)
    assert isinstance(reply["tensors"], bytes)
    decoded = decode_logits_reply(reply)
    for expected, actual in zip(values, decoded["logits_calls"], strict=True):
        assert expected.dtype == actual.dtype and torch.equal(expected, actual)
    assert torch.equal(
        decoded["output_head_weights"]["model.hc_head_fn"], head["model.hc_head_fn"]
    )
    with pytest.raises(TypeError, match="binary tensor"):
        decode_logits_reply({"tensors": [], "num_calls": 2})


def test_metrics_distinguish_scale_error_from_direction_agreement():
    x = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16)
    exact = error_metrics(x, x.clone())
    assert exact["max_abs"] == exact["relative_l2"] == 0
    doubled = error_metrics(x, 2 * x)
    assert doubled["relative_l2"] == 1
    assert doubled["cosine"] == pytest.approx(1)
    assert doubled["different_elements"] == x.numel()
    zeros = error_metrics(torch.zeros(2), torch.ones(2))
    assert zeros["relative_l2"] is None and zeros["cosine"] is None


def test_bad_alignment_tensors_are_rejected():
    with pytest.raises(ValueError, match="shapes differ"):
        error_metrics(torch.zeros(2), torch.zeros(3))
    for invalid in [float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="Non-finite"):
            error_metrics(torch.ones(2), torch.tensor([1.0, invalid]))


def test_native_prompt_logits_select_prefix_rows_and_reject_ambiguity():
    prompt = torch.arange(44).reshape(11, 4)
    sampling = torch.zeros(1, 4)
    positions = torch.tensor([3, 10])
    assert torch.equal(
        select_prompt_logits([sampling, prompt], 12, positions, 4), prompt[positions]
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        select_prompt_logits([prompt, prompt], 12, positions, 4)
    with pytest.raises(ValueError, match="cover"):
        select_prompt_logits([prompt], 12, torch.tensor([11]), 4)
    with pytest.raises(ValueError, match="Ambiguous"):
        select_prompt_logits([prompt], 12, positions, 5)
