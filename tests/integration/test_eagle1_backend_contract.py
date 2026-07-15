"""Contract tests for the EAGLE-1 / EAGLE-2 drafter backend.

These are CPU-light: they exercise the draft model forward, the loss math
(SmoothL1 feature regression + full-vocab soft-CE distillation against a frozen
target head), the algorithm dispatch, the vLLM method mapping, and the
old-logprob aux-layer selection. No GPU or real target checkpoint required.
"""

from __future__ import annotations

import pytest


def _tiny_eagle1_config():
    from verl_speco.models.eagle1 import Eagle1Config

    return Eagle1Config(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        draft_num_hidden_layers=1,
        target_hidden_size=8,
        num_aux_hidden_states=1,
        vocab_size=32,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def test_eagle1_draft_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.eagle1 import LlamaForCausalLMEagle1

    config = _tiny_eagle1_config()
    model = LlamaForCausalLMEagle1(config).eval()

    # fc fuses [embed(H) ; target_feature(target_H)] -> H
    assert model.fc.in_features == config.hidden_size + config.target_hidden_size
    assert model.fc.out_features == config.hidden_size
    assert len(model.layers) == 1

    batch, seq = 1, 5
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    hidden_states = torch.randn(batch, seq, config.target_hidden_size)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    out = model(input_ids=input_ids, hidden_states=hidden_states, attention_mask=attention_mask)
    assert out.shape == (batch, seq, config.hidden_size)


def _make_backend(training_overrides=None):
    from omegaconf import OmegaConf

    from verl_speco.backends.eagle1_trainer_backend import Eagle1TrainerBackend

    training = {
        "eagle1_hidden_loss_weight": 1.0,
        "eagle1_token_loss_weight": 0.1,
        "eagle1_feature_noise": 0.0,
        "use_logits": False,
    }
    if training_overrides:
        training.update(training_overrides)
    config = OmegaConf.create(
        {"rollout": {"drafter": {"model_path": "/tmp/none", "training": training}}, "model": {"path": "/tmp/none"}}
    )
    return Eagle1TrainerBackend(config, OmegaConf.create({}))


def test_eagle1_backend_reports_eagle3_data_plumbing() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    assert _make_backend().model_type == "eagle3"


def test_eagle1_compute_loss_matches_reference_formula() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.eagle1 import LlamaForCausalLMEagle1

    torch.manual_seed(0)
    config = _tiny_eagle1_config()
    model = LlamaForCausalLMEagle1(config).eval()
    target_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False).eval()

    backend = _make_backend()
    backend.target_model = target_head

    batch_n, seq = 1, 6
    input_ids = torch.randint(0, config.vocab_size, (batch_n, seq))
    hidden_states = torch.randn(batch_n, seq, config.target_hidden_size)
    last_hidden_states = torch.randn(batch_n, seq, config.hidden_size)
    loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0, 1.0]])
    batch = {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "last_hidden_states": last_hidden_states,
        "attention_mask": torch.ones(batch_n, seq, dtype=torch.long),
        "loss_mask": loss_mask,
        "position_ids": torch.arange(seq).unsqueeze(0),
    }

    out = backend.compute_loss(model, batch, 0)

    # Reference: recompute the Automodel EAGLE-1 loss terms directly.
    with torch.no_grad():
        predicted_hidden = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        )
        predicted_logits = target_head(predicted_hidden).float()
        target_probs = torch.softmax(target_head(last_hidden_states).float(), dim=-1)
        valid = loss_mask.bool()
        num_tokens = valid.float().sum()

        hidden_pt = torch.nn.functional.smooth_l1_loss(
            predicted_hidden.float(), last_hidden_states.float(), reduction="none"
        ).mean(dim=-1)
        ref_vloss = torch.where(valid, hidden_pt, torch.zeros_like(hidden_pt)).sum()
        token_pt = -(target_probs * torch.log_softmax(predicted_logits, dim=-1)).sum(dim=-1)
        ref_ploss = torch.where(valid, token_pt, torch.zeros_like(token_pt)).sum()

    assert out["v_weight"] == pytest.approx(1.0)
    assert out["p_weight"] == pytest.approx(0.1)
    assert float(out["local_num_tokens"]) == pytest.approx(float(num_tokens))
    assert float(out["total_local_vloss"]) == pytest.approx(float(ref_vloss), rel=1e-5, abs=1e-6)
    assert float(out["total_local_ploss"]) == pytest.approx(float(ref_ploss), rel=1e-5, abs=1e-6)


def test_eagle1_compute_loss_requires_last_hidden_states() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.eagle1 import LlamaForCausalLMEagle1

    config = _tiny_eagle1_config()
    model = LlamaForCausalLMEagle1(config).eval()
    backend = _make_backend()
    backend.target_model = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    seq = 4
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (1, seq)),
        "hidden_states": torch.randn(1, seq, config.target_hidden_size),
        "last_hidden_states": None,
        "attention_mask": torch.ones(1, seq, dtype=torch.long),
        "loss_mask": torch.ones(1, seq),
        "position_ids": torch.arange(seq).unsqueeze(0),
    }
    with pytest.raises(ValueError, match="last_hidden_states"):
        backend.compute_loss(model, batch, 0)


@pytest.mark.parametrize("algorithm", ["EAGLE1", "EAGLE2"])
def test_eagle1_eagle2_vllm_method_is_eagle(algorithm) -> None:
    from verl_speco.integration.vllm_runtime import _speculative_method_from_drafter

    assert _speculative_method_from_drafter({"speculative_algorithm": algorithm}) == "eagle"


@pytest.mark.parametrize("algorithm", ["EAGLE1", "EAGLE2"])
def test_eagle1_eagle2_collect_final_aux_layer(algorithm) -> None:
    from verl_speco.integration.oldlogprob_layer_ids import resolve_oldlogprob_aux_layer_ids

    layer_ids = resolve_oldlogprob_aux_layer_ids(
        {"speculative_algorithm": algorithm, "training": {}},
        target_num_hidden_layers=32,
    )
    assert layer_ids == [31]
