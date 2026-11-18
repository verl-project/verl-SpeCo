# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Verify that the MTP artifact probe uses the native output-head contract."""

import torch

from verl_speco.probe_deepseek_v4_mtp_artifact import forward_mtp


def test_mtp_cold_probe_preserves_inputs_and_compares_full_head():
    class NativeMTP(torch.nn.Module):
        def forward(self, *, input_ids, hidden_states, positions, positions_cpu, **kw):
            assert torch.equal(positions, positions_cpu)
            assert hidden_states.requires_grad
            return hidden_states + input_ids.unsqueeze(-1)

        def compute_logits(self, hidden, lm_head, spec_step_idx=0):
            assert lm_head is None and spec_step_idx == 0
            return hidden.sum(-1, keepdim=True).expand(*hidden.shape[:-1], 17)

    batch = {
        "input_ids": torch.arange(40).unsqueeze(0),
        "positions": torch.arange(100, 140).unsqueeze(0),
        "loss_mask": torch.ones(1, 40),
        "hidden_states": torch.randn(1, 40, 8),
    }
    original = {key: value.clone() for key, value in batch.items()}
    first = forward_mtp(NativeMTP(), batch, "cpu", 2)
    second = forward_mtp(NativeMTP(), batch, "cpu", 2)
    assert first["hidden"].shape == (1, 32, 8)
    assert first["logits"].shape == (1, 32, 17)
    for key in first:
        assert not first[key].requires_grad
        assert torch.equal(first[key], second[key])
    for key, value in batch.items():
        assert torch.equal(value, original[key])
