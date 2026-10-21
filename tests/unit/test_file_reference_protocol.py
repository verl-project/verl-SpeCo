# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
import pytest
import torch

from verl_speco.transport.drafter_sample_protocol import PROTOCOL_SCHEMA_VERSION, SampleMetadata
from verl_speco.transport.drafter_sample_protocol import make_reference_tag, validate_reference_tag


def _tag():
    return make_reference_tag(
        SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run-a", "sample-a", 0),
        path="/shared/sample.safetensors",
        request={
            "input_ids": [1, 2, 3], "loss_mask": [0, 1, 1],
            "position_ids": [0, 1, 2], "feature_positions": [1, 2],
            "draft_position_ids": [1, 2], "vllm_prompt_token_ids": [1, 2, 3],
            "source_metadata": {},
        },
        feature_contract={"algorithm": "DSPARK", "hidden_states_layout": "dflash_aux_plus_last"},
    )


def test_plain_reference_metadata_roundtrip_and_owned_values():
    tag = _tag()
    validate_reference_tag(tag)
    copied = make_reference_tag(
        SampleMetadata.from_dict(tag), path=tag["hidden_states_path"],
        request=tag["request"], feature_contract=tag["feature_contract"],
    )
    tag["request"]["loss_mask"][0] = 1
    assert copied["request"]["loss_mask"] == [0, 1, 1]


@pytest.mark.parametrize("field,value", [
    ("reference_version", 99), ("backend", "mooncake"),
    ("hidden_states_path", ""), ("payload_kind", "unknown"),
])
def test_reject_incompatible_reference(field, value):
    tag = _tag()
    tag[field] = value
    with pytest.raises(ValueError):
        validate_reference_tag(tag)


def test_reject_tensor_metadata_and_invalid_positions():
    tag = _tag()
    tag["request"]["source_metadata"]["hidden"] = torch.ones(1)
    with pytest.raises(ValueError, match="plain metadata"):
        validate_reference_tag(tag)
    tag = _tag()
    tag["request"]["feature_positions"] = [1, 9]
    with pytest.raises(ValueError, match="out of range"):
        validate_reference_tag(tag)


def test_mooncake_metadata_preserves_manifest_and_rejects_ambiguity():
    tag = _tag()
    params = {"handle": "hs-1", "version": 1, "tensors": {"hidden_states": {"shape": [3, 2, 4], "dtype": "bfloat16"}}}
    copied = make_reference_tag(
        SampleMetadata.from_dict(tag), backend="mooncake", handle="hs-1",
        transfer_params=params, request=tag["request"], feature_contract=tag["feature_contract"],
    )
    params["tensors"]["hidden_states"]["shape"][0] = 99
    assert copied["transfer_params"]["tensors"]["hidden_states"]["shape"][0] == 3
    copied["transfer_params"]["handle"] = "wrong"
    with pytest.raises(ValueError, match="disagree"):
        validate_reference_tag(copied)
    copied["transfer_params"]["handle"] = "hs-1"
    copied["hidden_states_path"] = "/wrong"
    with pytest.raises(ValueError, match="exactly one"):
        validate_reference_tag(copied)
