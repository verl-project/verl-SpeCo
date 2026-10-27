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
    for name in (
        "input_ids", "loss_mask", "position_ids", "feature_positions",
        "draft_position_ids", "vllm_prompt_token_ids",
    ):
        expected = list(tag["request"][name])
        tag["request"][name][0] = 99
        assert copied["request"][name] == expected


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


def test_reference_tag_owns_nested_source_metadata_and_feature_contract():
    tag = _tag()
    tag["request"]["source_metadata"] = {"nested": {"values": [1, 2]}}
    tag["feature_contract"]["target_layer_ids"] = [0, 1]
    copied = make_reference_tag(
        SampleMetadata.from_dict(tag), path=tag["hidden_states_path"],
        request=tag["request"], feature_contract=tag["feature_contract"],
    )
    tag["request"]["source_metadata"]["nested"]["values"][0] = 99
    tag["feature_contract"]["target_layer_ids"][0] = 99
    assert copied["request"]["source_metadata"]["nested"]["values"] == [1, 2]
    assert copied["feature_contract"]["target_layer_ids"] == [0, 1]


@pytest.mark.parametrize("field,value", [
    ("input_ids", [False, 2, 3]),
    ("position_ids", [0.5, 1, 2]),
    ("loss_mask", [0, float("nan"), 1]),
    ("feature_positions", [1, 9]),
])
def test_value_validation_is_deferred_until_full_validation(field, value):
    from verl_speco.transport.drafter_sample_protocol import validate_reference_envelope

    tag = _tag()
    tag["request"][field] = value
    copied = make_reference_tag(
        SampleMetadata.from_dict(tag), path=tag["hidden_states_path"],
        request=tag["request"], feature_contract=tag["feature_contract"],
    )
    validate_reference_envelope(copied)
    with pytest.raises(ValueError):
        validate_reference_tag(copied)
