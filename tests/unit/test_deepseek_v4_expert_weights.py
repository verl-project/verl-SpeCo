# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for TP intermediate slicing -> EP local-expert placement."""

import pytest
import torch

from verl_speco.integration.deepseek_v4_online import (
    EmbeddedDraftLoader,
    runtime_weight_name,
)


class Experts(torch.nn.Module):
    def __init__(self, mapping, hidden=3, intermediate=8):
        super().__init__()
        self.register_buffer("expert_map", torch.tensor(mapping))
        count = sum(i >= 0 for i in mapping)
        self.w13_weight = torch.nn.Parameter(
            torch.zeros(count, hidden, 2 * intermediate, dtype=torch.bfloat16)
        )
        self.w2_weight = torch.nn.Parameter(
            torch.zeros(count, intermediate, hidden, dtype=torch.bfloat16)
        )


class Draft(torch.nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.experts = Experts(mapping)
        self.norm = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))


def payload(tp_size, w13, w2):
    width = w2.shape[1] // tp_size
    half = w13.shape[2] // 2
    for rank in reversed(range(tp_size)):
        start, end = rank * width, (rank + 1) * width
        yield (
            runtime_weight_name(rank, tp_size, "experts.w13_weight"),
            torch.cat(
                [w13[:, :, start:end], w13[:, :, half + start : half + end]], dim=2
            ),
        )
        yield (
            runtime_weight_name(rank, tp_size, "experts.w2_weight"),
            w2[:, start:end, :],
        )
        yield runtime_weight_name(rank, tp_size, "norm"), torch.ones(3)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("mapping", [[0, 1, 2, 3], [1, -1, -1, 0], [-1, 0, -1, 1]])
def test_ep_weights_and_forward_match_full_expert_oracle(tp_size, mapping):
    generator = torch.Generator().manual_seed(7)
    w13 = torch.randn(4, 3, 16, generator=generator).to(torch.bfloat16)
    w2 = torch.randn(4, 8, 3, generator=generator).to(torch.bfloat16)
    model = Draft(mapping)
    signatures = {name: value.data_ptr() for name, value in model.named_parameters()}
    loader = EmbeddedDraftLoader(model, tp_rank=0, tp_size=tp_size)
    for item in payload(tp_size, w13, w2):
        loader.receive([item])
    assert torch.count_nonzero(model.experts.w13_weight) == 0
    assert loader.finish() == 3
    for global_id, local_id in enumerate(mapping):
        if local_id < 0:
            continue
        assert torch.equal(model.experts.w13_weight[local_id], w13[global_id])
        assert torch.equal(model.experts.w2_weight[local_id], w2[global_id])
        x = torch.tensor([[0.25, -0.5, 1.0]])

        def forward(a, b, x=x):
            gate, up = (x @ a.float()).chunk(2, dim=-1)
            return (torch.nn.functional.silu(gate) * up) @ b.float()

        assert torch.equal(
            forward(
                model.experts.w13_weight[local_id], model.experts.w2_weight[local_id]
            ),
            forward(w13[global_id], w2[global_id]),
        )
    assert signatures == {
        name: value.data_ptr() for name, value in model.named_parameters()
    }


def test_changed_expert_map_rejects_before_live_write():
    model = Draft([0, -1, -1, 1])
    loader = EmbeddedDraftLoader(model, tp_size=2)
    loader.receive(
        list(
            payload(
                2,
                torch.ones(4, 3, 16, dtype=torch.bfloat16),
                torch.ones(4, 8, 3, dtype=torch.bfloat16),
            )
        )
    )
    model.experts.expert_map.copy_(torch.tensor([1, -1, -1, 0]))
    with pytest.raises(RuntimeError, match="expert_map changed"):
        loader.finish()
    assert torch.count_nonzero(model.experts.w13_weight) == 0


@pytest.mark.parametrize("mapping", [[0, 0, -1, -1], [0, 2, -1, -1], [0, -2, -1, 1]])
def test_invalid_local_expert_map_is_rejected(mapping):
    with pytest.raises(ValueError, match="every local slot"):
        EmbeddedDraftLoader(Draft(mapping), tp_size=2)


def test_incomplete_expert_lane_never_writes_live_parameters():
    model = Draft([0, -1, -1, 1])
    loader = EmbeddedDraftLoader(model, tp_size=2)
    items = list(
        payload(
            2,
            torch.ones(4, 3, 16, dtype=torch.bfloat16),
            torch.ones(4, 8, 3, dtype=torch.bfloat16),
        )
    )
    loader.receive(items[:-1])
    with pytest.raises(RuntimeError, match="Incomplete"):
        loader.finish()
    assert torch.count_nonzero(model.experts.w13_weight) == 0


def test_cached_expert_lists_rejected_instead_of_silently_left_stale():
    model = Draft([0, -1, -1, 1])
    model.experts.w13_weight_list = [model.experts.w13_weight[0].clone()]
    with pytest.raises(ValueError, match="weight-list caches"):
        EmbeddedDraftLoader(model, tp_size=2)


def test_inference_parameters_can_be_updated_in_place():
    with torch.inference_mode():
        model = Draft([0, -1, -1, 1])
    loader = EmbeddedDraftLoader(model, tp_size=2)
    loader.receive(
        list(
            payload(
                2,
                torch.ones(4, 3, 16, dtype=torch.bfloat16),
                torch.ones(4, 8, 3, dtype=torch.bfloat16),
            )
        )
    )
    loader.finish()
    assert torch.all(model.experts.w13_weight == 1)


def test_runner_forwarded_parameters_do_not_duplicate_expert_specs():
    class Runner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_experts = Experts([0, -1, -1, 1])

        @property
        def expert_map(self):
            return self.routed_experts.expert_map

        @property
        def w13_weight(self):
            return self.routed_experts.w13_weight

        @property
        def w2_weight(self):
            return self.routed_experts.w2_weight

    model = torch.nn.Module()
    model.experts = Runner()
    loader = EmbeddedDraftLoader(model, tp_size=2)
    assert set(loader.experts.specs) == {
        "experts.routed_experts.w13_weight",
        "experts.routed_experts.w2_weight",
    }


def test_copy_failure_restores_parameters_already_written(monkeypatch):
    model = Draft([0, -1, -1, 1])
    loader = EmbeddedDraftLoader(model, tp_size=2)
    loader.receive(
        list(
            payload(
                2,
                torch.ones(4, 3, 16, dtype=torch.bfloat16),
                torch.ones(4, 8, 3, dtype=torch.bfloat16),
            )
        )
    )
    original = torch.Tensor.copy_
    fail = [True]
    pointer = model.experts.w2_weight.data_ptr()

    def copy(target, source, *args, **kwargs):
        if target.data_ptr() == pointer and fail[0]:
            fail[0] = False
            raise RuntimeError("injected device copy failure")
        return original(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError, match="injected"):
        loader.finish()
    assert all(torch.count_nonzero(value) == 0 for value in model.parameters())
    assert loader.failed


def test_eplb_is_rejected_before_staging():
    model = Draft([0, -1, -1, 1])
    model.enable_eplb = True
    with pytest.raises(ValueError, match="EPLB"):
        EmbeddedDraftLoader(model, tp_size=2)


def test_quantized_or_non_bf16_experts_are_rejected():
    model = Draft([0, -1, -1, 1])
    model.experts.w13_weight = torch.nn.Parameter(model.experts.w13_weight.half())
    with pytest.raises(ValueError, match="BF16"):
        EmbeddedDraftLoader(model, tp_size=2)
