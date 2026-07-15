"""Contract tests for the P-EAGLE (parallel-drafting) drafter backend.

CPU-light: they exercise the COD sampling, the flex-attention mask predicate, the
draft-model modules, the KL loss, the algorithm routing, and the vLLM guardrail.
The full parallel-drafting forward is validated on GPU by ``ci/peagle_gpu_smoke.py``.
"""

from __future__ import annotations

import pytest


def _tiny_peagle_config():
    from verl_speco.models.peagle import PeagleConfig

    return PeagleConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_draft_layers=2,
        target_hidden_size=8,
        num_aux_hidden_states=3,
        vocab_size=32,
        draft_vocab_size=32,
        num_depths=4,
        mask_token_id=31,
        pad_token_id=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def test_cod_sampling_structure() -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.models.peagle.cod_sampling import generate_cod_sample_indices

    torch.manual_seed(0)
    seq_len = 16
    loss_mask = torch.ones(1, seq_len)
    anchor_pos, depth = generate_cod_sample_indices(seq_len, loss_mask, num_depths=4, down_sample_ratio=0.7)

    assert anchor_pos.shape == depth.shape
    # Depth 0 keeps every position; deeper depths shrink geometrically.
    assert int((depth == 0).sum()) == seq_len
    assert int((depth == 1).sum()) <= seq_len
    assert int((depth == 1).sum()) >= int((depth == 2).sum())
    # Reference position anchor+depth stays in range.
    assert int((anchor_pos + depth).max()) < seq_len
    assert int(anchor_pos.min()) >= 0


def test_peagle_mask_predicate() -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.models.peagle.peagle_mask import create_peagle_mask_mod

    # One document of length 4, depth-0 positions [0,1,2,3] + one depth-1 element.
    anchor_pos = torch.tensor([0, 1, 2, 3, 1])
    depth = torch.tensor([0, 0, 0, 0, 1])
    lengths = torch.tensor([4])
    mod = create_peagle_mask_mod(anchor_pos, depth, lengths, total_seq_len=4)

    z = torch.tensor(0)
    # depth-0 query 2 attends to earlier depth-0 kv 1 (committed causal context).
    assert bool(mod(z, z, torch.tensor(2), torch.tensor(1)))
    # depth-0 query 1 does NOT attend to later depth-0 kv 2.
    assert not bool(mod(z, z, torch.tensor(1), torch.tensor(2)))
    # depth-1 element (idx 4, anchor 1) attends to its own rollout's depth-0 (idx 1).
    assert bool(mod(z, z, torch.tensor(4), torch.tensor(1)))


def test_peagle_mask_isolates_documents() -> None:
    # Regression guard: with per-document lengths, a depth-0 query in one document
    # must NOT attend to a depth-0 key in another, even though the flat sequence
    # is causal. A single all-ones length (the old attention_mask.sum() behaviour)
    # would merge both documents and leak across them.
    torch = pytest.importorskip("torch")
    from verl_speco.models.peagle.peagle_mask import create_peagle_mask_mod

    # Two documents of length 2 concatenated: positions [0,1] and [2,3], all depth 0.
    anchor_pos = torch.tensor([0, 1, 2, 3])
    depth = torch.tensor([0, 0, 0, 0])
    mod = create_peagle_mask_mod(anchor_pos, depth, torch.tensor([2, 2]), total_seq_len=4)
    z = torch.tensor(0)

    # Same document: causal depth-0 attention allowed (query 3 -> key 2).
    assert bool(mod(z, z, torch.tensor(3), torch.tensor(2)))
    # Cross document: query 2 (doc B) must NOT attend to key 1 (doc A), despite 2 >= 1.
    assert not bool(mod(z, z, torch.tensor(2), torch.tensor(1)))
    # Merging both into one document (the bug) would have leaked here.
    merged = create_peagle_mask_mod(anchor_pos, depth, torch.tensor([4]), total_seq_len=4)
    assert bool(merged(z, z, torch.tensor(2), torch.tensor(1)))


def test_peagle_model_modules() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.peagle import LlamaForCausalLMPeagle

    config = _tiny_peagle_config()
    model = LlamaForCausalLMPeagle(config)

    # fc fuses num_aux * target_hidden -> hidden; mask_hidden lives at the pre-fc width.
    assert model.fc.in_features == config.num_aux_hidden_states * config.target_hidden_size
    assert model.fc.out_features == config.hidden_size
    assert tuple(model.mask_hidden.shape) == (1, 1, model.fc.in_features)
    assert len(model.layers) == config.num_draft_layers
    assert model.lm_head.out_features == config.draft_vocab_size
    # masked_projected_hidden runs the placeholder through fc -> [1, H].
    assert tuple(model.masked_projected_hidden().shape) == (1, config.hidden_size)
    # identity vocab mapping (draft_vocab == vocab).
    assert int(model.selected_token_ids().numel()) == config.vocab_size


def test_peagle_kl_loss() -> None:
    torch = pytest.importorskip("torch")
    from verl_speco.backends.peagle_trainer_backend import _kl_div_loss

    logits = torch.randn(5, 32)
    # KL(p||p) == 0.
    assert float(_kl_div_loss(logits, logits).abs().max()) < 1e-5
    kl = _kl_div_loss(torch.randn(5, 32), torch.randn(5, 32))
    assert kl.shape == (5,)
    assert bool((kl >= -1e-5).all())


def test_peagle_backend_metadata() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf

    from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

    backend = PEagleTrainerBackend(
        OmegaConf.create({"rollout": {"drafter": {"training": {}}}, "model": {"path": "/tmp/none"}}),
        OmegaConf.create({}),
    )
    assert backend.model_type == "peagle"
    assert backend.supports_ulysses_sp is False


def test_peagle_vllm_guardrail() -> None:
    from verl_speco.integration.vllm_runtime import _speculative_method_from_drafter

    with pytest.raises(ValueError, match="parallel-drafting runtime"):
        _speculative_method_from_drafter({"speculative_algorithm": "PEAGLE"})
