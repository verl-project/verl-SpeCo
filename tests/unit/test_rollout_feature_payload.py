# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from verl_speco.transport.rollout_feature_payload import (
    decode_feature_payload,
    encode_feature_payload,
)


def sample():
    ids = torch.tensor([2**40, 2**40 + 1, 2**40 + 2], dtype=torch.int64)
    return {
        "input_ids": ids,
        "positions": torch.arange(3),
        "hidden_states": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "target_head_hidden_states": torch.arange(24, dtype=torch.float32).reshape(
            3, 8
        ),
        "metadata": {"alias": ids, "sequence": (True, None, "context")},
    }


def test_wire_roundtrip_preserves_int64_bf16_fp32_and_nested_tree():
    original = sample()
    wire = encode_feature_payload(original)
    assert isinstance(wire["tensors"], bytes)
    actual = decode_feature_payload(wire)
    for name in (
        "input_ids",
        "positions",
        "hidden_states",
        "target_head_hidden_states",
    ):
        assert actual[name].dtype == original[name].dtype
        assert torch.equal(actual[name], original[name])
    assert torch.equal(actual["metadata"]["alias"], original["input_ids"])
    assert actual["metadata"]["sequence"] == (True, None, "context")
    original["hidden_states"].zero_()
    assert actual["hidden_states"].sum() > 0


def test_native_in_process_reply_and_none_remain_usable():
    original = sample()
    assert decode_feature_payload(original)["input_ids"] is original["input_ids"]
    assert encode_feature_payload(None) is None


def test_untyped_vllm_tensor_descriptor_is_not_guessed_as_values():
    payload = sample()
    payload["input_ids"] = ["int64", [3], b"serialized buffer"]
    with pytest.raises(ValueError, match="typed feature codec"):
        decode_feature_payload(payload)


@pytest.mark.parametrize("fault", ["version", "binary", "truncated"])
def test_malformed_wire_payload_is_rejected(fault):
    value = encode_feature_payload(sample())
    if fault == "version":
        value["speco_feature_payload_version"] = 999
    elif fault == "binary":
        value["tensors"] = [1, 2, 3]
    else:
        value["tensors"] = b"broken"
    with pytest.raises((ValueError, TypeError)):
        decode_feature_payload(value)
