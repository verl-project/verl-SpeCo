# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Actual installed vLLM utility-result serialization, without a model or NPU."""

import pytest
import torch

from verl_speco.transport.rollout_feature_payload import (
    decode_feature_payload,
    encode_feature_payload,
)


def test_actual_utility_reply_preserves_alignment_logits(monkeypatch):
    pytest.importorskip("vllm")
    from vllm import envs
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    from verl_speco.probe_deepseek_v4_feature_alignment import (
        decode_logits_reply,
        encode_logits_reply,
    )

    monkeypatch.setattr(envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", False)
    logits = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    encoded = MsgpackEncoder().encode(
        UtilityResult([encode_logits_reply([logits], {}), None])
    )
    reply = MsgpackDecoder(UtilityResult).decode(encoded).result
    assert reply[1] is None
    actual = decode_logits_reply(reply[0])["logits_calls"][0]
    assert actual.dtype == logits.dtype and torch.equal(actual, logits)


def test_actual_utility_reply_preserves_feature_tensor_types(monkeypatch):
    pytest.importorskip("vllm")
    from vllm import envs
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    monkeypatch.setattr(envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", False)
    original = {
        "input_ids": torch.tensor([2**40, 2**40 + 1]),
        "positions": torch.arange(2),
        "hidden_states": torch.arange(64, dtype=torch.bfloat16).reshape(2, 32),
        "target_head_hidden_states": torch.arange(64, dtype=torch.float32).reshape(
            2, 32
        ),
    }
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(UtilityResult)
    # Reproduce the exact untyped utility boundary behind collective_rpc.
    bare = decoder.decode(encoder.encode(UtilityResult([original, None]))).result[0]
    assert isinstance(bare["input_ids"], list)
    assert bare["input_ids"][0] == "int64"
    response = decoder.decode(
        encoder.encode(UtilityResult([encode_feature_payload(original), None]))
    ).result
    assert response[1] is None
    actual = decode_feature_payload(response[0])
    for name, expected in original.items():
        assert actual[name].dtype == expected.dtype
        assert torch.equal(actual[name], expected)
