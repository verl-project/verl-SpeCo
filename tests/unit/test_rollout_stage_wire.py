# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
import torch

from verl_speco.integration.tensor_wire import (
    encode_rollout_requests,
    decode_rollout_sample,
)


def test_rollout_wire_preserves_bits_metadata_and_actor_identity():
    hidden = torch.randn(1, 16, 32).bfloat16()
    sample = {
        "hidden_states": hidden,
        "hidden_positions": torch.arange(16),
        "_speco_sample_id": "sample",
        "target_version": 4,
    }
    for source in ("vllm", "oldlogprob"):
        request = {"collection_id": "collection-2-" + source, "samples": [sample]}
        encoded = encode_rollout_requests([[request]])[0][0]
        if source == "oldlogprob":
            assert encoded is request
        restored = decode_rollout_sample(encoded["samples"][0])
        assert restored["_speco_sample_id"] == "sample"
        assert restored["target_version"] == 4
        assert torch.equal(
            restored["hidden_states"].view(torch.int16), hidden.view(torch.int16)
        )
        assert torch.equal(restored["hidden_positions"], sample["hidden_positions"])
    assert "_speco_rollout_tensor_wire" not in sample
