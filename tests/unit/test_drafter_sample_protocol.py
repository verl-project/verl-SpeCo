# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ExpectedFeatureConfig,
    SampleMetadata,
    decode_sample,
    encode_sample,
    is_ready_sample_tag,
    make_eos_record,
    make_ready_tag,
    make_sample_key,
    parse_ready_tag,
)


def _metadata() -> SampleMetadata:
    return SampleMetadata(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        run_id="run-a",
        sample_id="train-000017",
        sequence_no=17,
    )


def _sample(*, algorithm: str = "DSPARK") -> DraftFeatureSample:
    return DraftFeatureSample(
        algorithm=algorithm,
        input_ids=torch.tensor([10, 11, 12, 13]),
        loss_mask=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        position_ids=torch.tensor([6, 7, 8, 9]),
        hidden_states=torch.arange(64, dtype=torch.bfloat16).reshape(4, 16),
        last_hidden_states=torch.arange(32, dtype=torch.float32).reshape(4, 8),
        metadata={
            "target_model_path": "/models/target",
            "target_revision": "target-rev-a",
            "tokenizer_fingerprint": "tokenizer-sha256-a",
            "hidden_states_layout": "dflash_aux_plus_last",
            "target_layer_ids": [2, 8, 14],
            "hidden_positions": torch.tensor([6, 7, 8, 9]),
            "nested": {"pair": ("a", 2)},
        },
    )


def test_sample_round_trip_preserves_complete_sample() -> None:
    meta = _metadata()
    key = make_sample_key(meta)
    fields = encode_sample(_sample(), meta)
    restored = decode_sample(
        key,
        make_ready_tag(meta),
        fields,
        ExpectedFeatureConfig(run_id=meta.run_id),
    )

    assert key == "drafter:v2:run-a:000000000017:train-000017"
    assert restored.algorithm == "DSPARK"
    assert torch.equal(restored.input_ids, _sample().input_ids)
    assert torch.equal(restored.hidden_states, _sample().hidden_states)
    assert torch.equal(restored.last_hidden_states, _sample().last_hidden_states)
    assert torch.equal(
        restored.metadata["hidden_positions"], _sample().metadata["hidden_positions"]
    )
    assert restored.metadata["nested"]["pair"] == ("a", 2)
    assert fields["sample__manifest_json"].dtype == torch.uint8


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("algorithm", "EAGLE3"),
        ("target_model_path", "/models/wrong"),
        ("target_revision", "wrong-revision"),
        ("tokenizer_fingerprint", "wrong-tokenizer"),
        ("target_layer_ids", (1, 3)),
        ("hidden_states_layout", "dflash_aux"),
        ("hidden_dtype", "float32"),
    ],
)
def test_decode_rejects_consumer_feature_contract_mismatch(
    field: str, wrong_value: object
) -> None:
    meta = _metadata()
    expected = ExpectedFeatureConfig(
        run_id="run-a",
        algorithm="DSPARK",
        target_model_path="/models/target",
        target_revision="target-rev-a",
        tokenizer_fingerprint="tokenizer-sha256-a",
        target_layer_ids=(2, 8, 14),
        hidden_states_layout="dflash_aux_plus_last",
        hidden_dtype="bfloat16",
    )
    with pytest.raises(ValueError, match=field):
        decode_sample(
            make_sample_key(meta),
            make_ready_tag(meta),
            encode_sample(_sample(), meta),
            replace(expected, **{field: wrong_value}),
        )


def test_hidden_state_tensor_list_round_trip() -> None:
    sample = replace(
        _sample(algorithm="EAGLE3"),
        hidden_states=[torch.ones(4, 3), torch.zeros(4, 5)],
    )
    meta = _metadata()
    restored = decode_sample(
        make_sample_key(meta),
        make_ready_tag(meta),
        encode_sample(sample, meta),
        ExpectedFeatureConfig(run_id="run-a"),
    )
    assert restored.algorithm == "EAGLE3"
    assert isinstance(restored.hidden_states, list)
    assert [tuple(value.shape) for value in restored.hidden_states] == [(4, 3), (4, 5)]


def test_ready_parser_is_the_shared_discovery_contract() -> None:
    meta = _metadata()
    tag = make_ready_tag(meta)
    assert parse_ready_tag(tag, run_id="run-a") == meta
    assert is_ready_sample_tag(tag, run_id="run-a")
    assert not is_ready_sample_tag(tag, run_id="another-run")
    assert parse_ready_tag({**tag, "sequence_no": "bad"}, run_id="run-a") is None
    assert parse_ready_tag({**tag, "schema_version": 1}, run_id="run-a") is None


def test_decode_rejects_identity_mismatch() -> None:
    meta = _metadata()
    fields = encode_sample(_sample(), meta)
    bad_tag = {**make_ready_tag(meta), "sample_id": "wrong"}
    with pytest.raises(ValueError, match="key mismatch"):
        decode_sample(
            make_sample_key(meta),
            bad_tag,
            fields,
            ExpectedFeatureConfig(run_id="run-a"),
        )


@pytest.mark.parametrize("hidden_kind", ["tensor", "list"])
def test_decode_rejects_hidden_state_row_mismatch(hidden_kind: str) -> None:
    meta = _metadata()
    sample = _sample()
    if hidden_kind == "list":
        sample = replace(
            sample,
            hidden_states=[torch.ones(4, 8), torch.ones(4, 8)],
        )
    fields = encode_sample(sample, meta)
    malformed_field = (
        "sample__hidden_states"
        if hidden_kind == "tensor"
        else "sample__hidden_states__000001"
    )
    fields[malformed_field] = fields[malformed_field][:-1]
    with pytest.raises(ValueError, match="input_ids/hidden_states row mismatch"):
        decode_sample(
            make_sample_key(meta),
            make_ready_tag(meta),
            fields,
            ExpectedFeatureConfig(run_id="run-a"),
        )


def test_metadata_codec_rejects_lossy_unknown_values() -> None:
    sample = replace(_sample(), metadata={"unsupported": object()})
    with pytest.raises(TypeError, match="metadata.unsupported"):
        encode_sample(sample, _metadata())


@pytest.mark.parametrize("algorithm", ["EAGLE3", "DFLASH", "DSPARK", "DOMINO"])
def test_protocol_algorithm_is_not_hardcoded(algorithm: str) -> None:
    meta = _metadata()
    sample = _sample(algorithm=algorithm)
    restored = decode_sample(
        make_sample_key(meta),
        make_ready_tag(meta),
        encode_sample(sample, meta),
        ExpectedFeatureConfig(run_id="run-a"),
    )
    assert restored.algorithm == algorithm
    assert "algorithm" not in make_ready_tag(meta)


def test_eos_record_is_control_only() -> None:
    key, fields, tag = make_eos_record("run-a", 18)
    assert key == "control:v2:run-a:eos"
    assert fields["marker"].tolist() == [1]
    assert tag == {
        "record_type": "control",
        "status": "eos",
        "schema_version": 2,
        "run_id": "run-a",
        "total_samples": 18,
    }
