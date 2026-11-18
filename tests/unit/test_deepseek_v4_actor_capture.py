# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Actor boundaries, existing transport, and window/mask alignment on CPU."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.backends.training_backend import actor_capture_enabled
from verl_speco.integration.deepseek_v4_actor_capture import (
    feature_source,
    install_contract_hooks,
    padded_selection_context,
)
from verl_speco.integration.oldlogprob_runtime import (
    OLD_LOGPROB_HIDDEN_STATES_KEY,
    _cleanup_oldlogprob_hidden_capture,
    _consume_oldlogprob_hidden_capture,
)
from verl_speco.integration.vllm_rollout_capture import (
    capture_enabled,
    configure_capture,
    embedded_speculative_config,
)
from verl_speco.trainer.deepseek_v4_contract import LAYOUT, DeepSeekV4Contract
from verl_speco.trainer.vllm_ascend_online import VllmAscendOnlineMixin


@pytest.fixture
def contract():
    return DeepSeekV4Contract(2, 2, 64, 2, (0, 2), "cpu-fixture")


def selection(start=2):
    return {
        "batch_size": 1,
        "hidden_rows": 3,
        "local_positions": [start, start + 1, start + 2],
        "local_batch_indices": [0, 0, 0],
        "local_row_indices": [0, 1, 2],
        "max_local_position": start + 2,
        "sparse_sp_merge": False,
        "sp_group": None,
    }


@pytest.mark.parametrize(
    "layout", ["packed", "batch_first", "sequence_first", "flattened"]
)
@pytest.mark.parametrize("start", [2, 3])
def test_real_hooks_transport_and_training_window_match_native_contract(
    contract, layout, start
):
    class Layer(torch.nn.Module):
        def forward(self, x):
            return x + 10

    layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
    engine = SimpleNamespace()
    install_contract_hooks(
        engine, layers, selection(start), contract.identity(), [0, 2]
    )
    raw = torch.arange(24).float().reshape(6, 2, 2)
    if layout == "batch_first":
        raw = raw.unsqueeze(0)
    elif layout == "sequence_first":
        raw = raw.unsqueeze(1)
    elif layout == "flattened":
        raw = raw.flatten(-2)
    for layer in layers:
        raw = layer(raw)
    expected_raw = torch.arange(24).float().reshape(6, 2, 2)
    raw.zero_()  # Runtime buffers can be reused before transport/consumption.
    output = _consume_oldlogprob_hidden_capture(engine)
    packed = output[OLD_LOGPROB_HIDDEN_STATES_KEY]
    assert packed.shape == (1, 3, 8)
    assert torch.equal(
        packed[0, :, :2], (expected_raw[start : start + 3] + 10).mean(dim=1)
    )
    assert torch.equal(
        packed[0, :, 2:4], (expected_raw[start : start + 3] + 30).mean(dim=1)
    )
    assert torch.equal(
        packed[0, :, 4:], (expected_raw[start : start + 3] + 30).flatten(-2)
    )
    assert not hasattr(engine, "_speco_oldlogprob_hidden_context")
    assert all(not layer._forward_hooks for layer in layers)

    session = VllmAscendOnlineMixin.__new__(VllmAscendOnlineMixin)
    session.contract = contract
    session.config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {"max_seq_len": 8}}}}
    )
    session.current_rl_step = 7
    session.collected_data = deque()
    session._mark_buffer_changed = lambda: None
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]]),
        "prompts": torch.tensor([[10, 11, 12]]),
        "hidden_positions": torch.arange(start, start + 3).unsqueeze(0),
        "hidden_states_layout": LAYOUT,
        "global_step": 7,
    }
    session.collect_online_data(batch, packed)
    feature = session.collected_data[0]["feature"]
    assert feature.input_ids.tolist() == list(range(10 + start, 13 + start))
    assert feature.position_ids.tolist() == list(range(start, start + 3))
    assert feature.loss_mask.tolist() == ([0, 1, 1] if start == 2 else [1, 1, 1])
    contract.validate(feature)
    batch["global_step"] = 6
    with pytest.raises(ValueError, match="target version"):
        session.collect_online_data(batch, packed)


def test_post_hc_states_rejected_and_failed_capture_can_be_cleaned(contract):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    engine = SimpleNamespace()
    install_contract_hooks(engine, layers, selection(), contract.identity(), [0, 2])
    with pytest.raises(ValueError, match="complete pre-HC"):
        layers[0](torch.ones(6, 2))
    _cleanup_oldlogprob_hidden_capture(engine)
    assert all(not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize("lengths", [[6, 3], [6], [1, 1]])
def test_padded_actor_hooks_select_unpadded_positions_and_ignore_padding(
    contract, lengths
):
    from verl_speco.integration import oldlogprob_runtime as runtime

    ids = torch.nested.nested_tensor(
        [torch.arange(length) for length in lengths], layout=torch.jagged
    )
    positions = torch.tensor([[0, length - 1, length] for length in lengths])
    batch = {
        "input_ids": ids,
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: torch.ones(
            len(lengths), dtype=torch.bool
        ),
        runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY: positions,
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: torch.ones_like(
            positions, dtype=torch.bool
        ),
    }
    engine = SimpleNamespace(use_ulysses_sp=False)
    context = padded_selection_context(engine, batch)
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    install_contract_hooks(engine, layers, context, contract.identity(), [0, 2])
    raw = (
        torch.arange(len(lengths) * max(lengths) * 4)
        .float()
        .reshape(len(lengths), max(lengths), 2, 2)
    )
    for layer in layers:
        layer(raw)
    result = _consume_oldlogprob_hidden_capture(engine)[OLD_LOGPROB_HIDDEN_STATES_KEY]
    for b, length in enumerate(lengths):
        for row, position in enumerate((0, length - 1)):
            assert torch.equal(result[b, row, :2], raw[b, position].mean(dim=0))
            assert torch.equal(result[b, row, 2:4], raw[b, position].mean(dim=0))
            assert torch.equal(result[b, row, 4:], raw[b, position].flatten())
        assert result[b, 2].eq(0).all()
    assert all(not layer._forward_hooks for layer in layers)


def test_padded_actor_selection_rejects_sequence_parallelism():
    with pytest.raises(ValueError, match="SP=1"):
        padded_selection_context(SimpleNamespace(use_ulysses_sp=True), {})


@pytest.mark.parametrize("layout", [LAYOUT, "eagle3_aux_plus_last", "dflash_aux"])
def test_padded_dispatch_is_limited_to_new_feature_contract(layout):
    pytest.importorskip("verl")
    from verl_speco.integration import oldlogprob_runtime as runtime

    ids = torch.nested.nested_tensor([torch.arange(3)], layout=torch.jagged)
    batch = {
        "input_ids": ids,
        "use_remove_padding": False,
        runtime.OLD_LOGPROB_HIDDEN_LAYOUT_KEY: layout,
        runtime.OLD_LOGPROB_COLLECT_MASK_KEY: torch.tensor([True]),
        runtime.OLD_LOGPROB_HIDDEN_POSITIONS_KEY: torch.tensor([[1]]),
        runtime.OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY: torch.tensor([[True]]),
    }
    engine = SimpleNamespace(use_ulysses_sp=False)
    if layout == LAYOUT:
        context = runtime._build_selection_context(engine, {}, batch)
        assert context["local_positions"] == [1]
    else:
        with pytest.raises(NotImplementedError, match="use_remove_padding=True"):
            runtime._build_selection_context(engine, {}, batch)


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {
            "collect_hidden_states_from_rollout": True,
            "collect_hidden_states_from_old_logprob": True,
        },
        {
            "collect_hidden_states_from_old_logprob": True,
            "collect_hidden_states_from_sgl": True,
        },
        {
            "collect_hidden_states_from_old_logprob": True,
            "old_logprob_hidden_capture_impl": "output_hidden_states",
        },
    ],
)
def test_invalid_source_combinations_fail_closed(updates):
    with pytest.raises(ValueError):
        feature_source(updates)


def test_source_selection_is_separate_from_execution_backend():
    training = {
        "backend": "vllm_ascend",
        "collect_hidden_states_from_old_logprob": True,
    }
    drafter = {"enable_drafter_training": True, "training": training}
    assert feature_source(training) == "actor"
    assert actor_capture_enabled(drafter)
    assert not capture_enabled(drafter)
    training["backend"] = "pytorch"
    assert not actor_capture_enabled(drafter)
    with pytest.raises(ValueError, match="removed"):
        feature_source({"collect_hidden_states_from_rollout": True})


def test_actor_keeps_embedded_runtime_publication_without_sampler_capture(tmp_path):
    import json

    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 3,
        "hidden_size": 2,
        "hc_mult": 2,
        "vocab_size": 64,
        "dspark_block_size": 2,
        "dspark_target_layer_ids": [0, 2],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    drafter = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "DSPARK",
        "model_path": str(tmp_path),
        "rollout": {"spec_verify_tokens": 2},
        "training": {
            "backend": "vllm_ascend",
            "mode": "online",
            "collect_hidden_states_from_old_logprob": True,
        },
    }
    full = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": str(tmp_path)},
                "rollout": {"drafter": drafter, "enable_prefix_caching": True},
            }
        }
    )
    kwargs = {}
    configure_capture(full, drafter, drafter, kwargs)
    assert not capture_enabled(drafter)
    assert kwargs == {"load_format": "auto"}
    assert full.actor_rollout_ref.rollout.enable_prefix_caching
    from verl_speco.integration.vllm_runtime import (
        build_vllm_speculative_config_from_drafter,
    )

    assert build_vllm_speculative_config_from_drafter(
        drafter
    ) == embedded_speculative_config(drafter)
    assert embedded_speculative_config(drafter) == {
        "method": "dspark",
        "num_speculative_tokens": 2,
        "enforce_eager": True,
    }


def test_actor_pipeline_parallel_is_rejected_before_hook_installation(contract):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    with pytest.raises(ValueError, match="actor PP=1"):
        install_contract_hooks(
            SimpleNamespace(),
            layers,
            {**selection(), "pp_size": 2},
            contract.identity(),
            [0, 2],
        )
    assert all(not layer._forward_hooks for layer in layers)
