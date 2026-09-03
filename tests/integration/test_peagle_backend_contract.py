# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    anchor_pos, depth = generate_cod_sample_indices(
        seq_len, loss_mask, num_depths=4, down_sample_ratio=0.7
    )

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
    mod = create_peagle_mask_mod(
        anchor_pos, depth, torch.tensor([2, 2]), total_seq_len=4
    )
    z = torch.tensor(0)

    # Same document: causal depth-0 attention allowed (query 3 -> key 2).
    assert bool(mod(z, z, torch.tensor(3), torch.tensor(2)))
    # Cross document: query 2 (doc B) must NOT attend to key 1 (doc A), despite 2 >= 1.
    assert not bool(mod(z, z, torch.tensor(2), torch.tensor(1)))
    # Merging both into one document (the bug) would have leaked here.
    merged = create_peagle_mask_mod(
        anchor_pos, depth, torch.tensor([4]), total_seq_len=4
    )
    assert bool(merged(z, z, torch.tensor(2), torch.tensor(1)))


def test_peagle_model_modules() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.peagle import LlamaForCausalLMPeagle

    config = _tiny_peagle_config()
    model = LlamaForCausalLMPeagle(config)

    # fc fuses num_aux * target_hidden -> hidden; mask_hidden lives at the pre-fc width.
    assert (
        model.fc.in_features == config.num_aux_hidden_states * config.target_hidden_size
    )
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
        OmegaConf.create(
            {"rollout": {"drafter": {"training": {}}}, "model": {"path": "/tmp/none"}}
        ),
        OmegaConf.create({}),
    )
    assert backend.model_type == "peagle"
    assert backend.supports_ulysses_sp is False


def test_peagle_vllm_guardrail() -> None:
    from verl_speco.integration.vllm_runtime import _speculative_method_from_drafter

    with pytest.raises(ValueError, match="parallel-drafting runtime"):
        _speculative_method_from_drafter({"speculative_algorithm": "PEAGLE"})


def _peagle_backend_and_target(draft_vocab_size=None):
    from omegaconf import OmegaConf
    from transformers import LlamaConfig

    from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

    training: dict = {
        "peagle_num_draft_layers": 1,
        "peagle_num_aux_hidden_states": 3,
        "peagle_num_depths": 2,
    }
    if draft_vocab_size is not None:
        training["peagle_draft_vocab_size"] = draft_vocab_size

    target_hf_config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=32,
        max_position_embeddings=64,
    )
    backend = PEagleTrainerBackend(
        OmegaConf.create(
            {
                "rollout": {"drafter": {"model_path": None, "training": training}},
                "model": {"path": "/tmp/none"},
            }
        ),
        target_hf_config,
    )
    return backend, target_hf_config


def test_peagle_rejects_reduced_draft_vocab_without_mapping() -> None:
    """A reduced draft vocabulary needs a real frequency-derived t2d/d2t pair.

    Without one the model constructor falls back to "the first draft_vocab_size
    target ids", which is an arbitrary slice of any real tokenizer and leaves the
    draft unable to ever emit the rest. EAGLE-3 already refuses this
    configuration; P-EAGLE used to accept it silently.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    backend, _ = _peagle_backend_and_target(draft_vocab_size=16)

    with pytest.raises(ValueError) as excinfo:
        backend.build_model()

    message = str(excinfo.value)
    assert "draft_vocab_size differs from target vocab_size" in message
    # Name the algorithm the user configured, not the EAGLE-3 backend it reuses.
    assert message.startswith("PEAGLE")


def test_peagle_identity_vocab_mapping_passes_validation() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.peagle import LlamaForCausalLMPeagle

    backend, target_hf_config = _peagle_backend_and_target()
    draft_config = backend._build_draft_config(None, target_hf_config)
    assert draft_config.draft_vocab_size == draft_config.vocab_size

    # Must not raise: the full-vocabulary default needs no frequency mapping.
    backend._validate_vocab_mapping(LlamaForCausalLMPeagle(draft_config))


def test_vocab_mapping_validation_names_the_failing_algorithm() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend
    from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

    empty_config = OmegaConf.create(
        {"rollout": {"drafter": {"training": {}}}, "model": {"path": "/tmp/none"}}
    )
    drafter_without_mapping = SimpleNamespace(vocab_size=32, draft_vocab_size=32)

    for backend_cls, expected_label in (
        (Eagle3TrainerBackend, "EAGLE3"),
        (PEagleTrainerBackend, "PEAGLE"),
    ):
        backend = backend_cls(empty_config, OmegaConf.create({}))
        with pytest.raises(AttributeError) as excinfo:
            backend._validate_vocab_mapping(drafter_without_mapping)
        assert str(excinfo.value).startswith(expected_label)


def test_peagle_batch_assembly_matches_reference_shift() -> None:
    """base_trainer must apply the reference target-wrapper shift for P-EAGLE:
    row p pairs unshifted aux[p] with token x[p+1], supervised by the
    distribution from last_hidden[p+1] and gated by loss_mask[p+1]."""
    pytest.importorskip("torch")
    import torch
    from omegaconf import OmegaConf

    base_trainer_mod = pytest.importorskip("verl_speco.trainer.base_trainer")

    seq_len, hidden = 10, 4
    ids = torch.arange(seq_len, dtype=torch.long)
    aux = torch.randn(seq_len, hidden)
    last_h = torch.randn(seq_len, hidden)
    loss_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    position_ids = torch.arange(seq_len, dtype=torch.long)

    class _FakeBackend:
        model_type = "peagle"

        def preprocess_individual_items(self, items, dev, model_config):
            return {
                "ids": [ids],
                "h_states": [aux],
                "masks": [loss_mask],
                "position_ids": [position_ids],
                "last_h_states": [last_h],
            }

    trainer = object.__new__(base_trainer_mod.DrafterBaseTrainer)
    trainer.backend = _FakeBackend()
    trainer.batch_size = 2
    trainer.current_rl_step = 0
    trainer.training_steps = 0
    trainer.use_data_buffer = False
    trainer.collected_data = [{"step": 0, "hidden_states": aux}]
    trainer.config = OmegaConf.create({"rollout": {"drafter": {"training": {}}}})
    trainer.use_ulysses_sp = False
    trainer.rank = 0
    trainer.model_config = None
    trainer.model = torch.nn.Linear(2, 2)

    batch = trainer._prepare_training_batch()

    assert batch is not None
    train_len = seq_len - 1
    assert torch.equal(batch["input_ids"][0], ids[1 : 1 + train_len])
    assert torch.equal(batch["hidden_states"][0], aux[:train_len])
    assert torch.equal(batch["last_hidden_states"][0], last_h[1 : 1 + train_len])
    assert torch.equal(batch["loss_mask"][0], loss_mask[1 : 1 + train_len])
    assert batch["seq_lengths"].tolist() == [train_len]


def test_peagle_layers_are_callable_fsdp_wrap_targets() -> None:
    """The draft layers are the FSDP wrap targets, so their entry point must be
    ``forward``; see ``PEagleTrainingModel`` for why."""
    pytest.importorskip("torch")
    from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel
    from verl_speco.models.peagle.modeling_peagle import PeagleFusedLayer, PeagleVanillaLayer

    wrap_targets = set(PEagleTrainingModel._no_split_modules)
    assert wrap_targets == {"PeagleFusedLayer", "PeagleVanillaLayer"}
    for layer_cls in (PeagleFusedLayer, PeagleVanillaLayer):
        assert layer_cls.__name__ in wrap_targets
        assert "forward" in vars(layer_cls), f"{layer_cls.__name__} must define forward for FSDP wrapping"
        assert not hasattr(layer_cls, "forward_peagle")


def test_peagle_training_model_wraps_the_draft() -> None:
    pytest.importorskip("torch")
    from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel
    from verl_speco.models.peagle import LlamaForCausalLMPeagle

    config = _tiny_peagle_config()
    draft = LlamaForCausalLMPeagle(config)
    training_model = PEagleTrainingModel(draft, num_depths=4, down_sample_ratio=0.5, down_sample_ratio_min=0.1)

    assert training_model.draft_model is draft
    assert training_model.config is draft.config
    assert training_model.num_depths == 4
    # The wrapper owns no parameters of its own, so the exported draft is complete.
    assert {name for name, _ in training_model.named_parameters()} == {
        f"draft_model.{name}" for name, _ in draft.named_parameters()
    }


def test_peagle_compute_loss_calls_the_module_forward() -> None:
    """compute_loss must drive the draft through ``model(...)`` so FSDP2 unshards."""
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from verl_speco.backends.peagle_trainer_backend import PEagleTrainerBackend

    backend = PEagleTrainerBackend(
        OmegaConf.create({"rollout": {"drafter": {"training": {}}}, "model": {"path": "/tmp/none"}}),
        OmegaConf.create({}),
    )
    backend.target_model = lambda last_hidden: torch.zeros(*last_hidden.shape[:-1], 6)

    calls = []

    def _model(**kwargs):
        calls.append(kwargs)
        return torch.tensor(8.0), torch.tensor(4.0), torch.tensor(3.0)

    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "hidden_states": torch.zeros(1, 5, 8),
        "loss_mask": torch.ones(1, 5),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "last_hidden_states": torch.zeros(1, 5, 4),
        "seq_lengths": torch.tensor([5]),
    }

    out = backend.compute_loss(_model, batch, 0)

    assert len(calls) == 1
    assert set(calls[0]) == {"input_ids", "aux_hidden", "loss_mask", "attention_mask", "target_logits", "seq_lengths"}
    assert calls[0]["target_logits"].shape == (1, 5, 6)
    assert float(out["total_local_ploss"]) == 8.0
    assert float(out["local_num_tokens"]) == 4.0
    assert float(out["accuracy"]) == pytest.approx(0.75)


def _peagle_training_model_and_batch(batch_size: int, seq_len: int, seq_lengths):
    import torch

    from verl_speco.backends.peagle_trainer_backend import PEagleTrainingModel
    from verl_speco.models.peagle import LlamaForCausalLMPeagle

    config = _tiny_peagle_config()
    model = PEagleTrainingModel(
        LlamaForCausalLMPeagle(config), num_depths=2, down_sample_ratio=0.5
    )
    kwargs = dict(
        input_ids=torch.zeros(batch_size, seq_len, dtype=torch.long),
        aux_hidden=torch.zeros(batch_size, seq_len, config.target_hidden_size * 3),
        loss_mask=torch.ones(batch_size, seq_len),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        target_logits=torch.zeros(batch_size, seq_len, config.vocab_size),
        seq_lengths=seq_lengths,
    )
    return model, kwargs


def test_peagle_rejects_packed_lengths_for_a_multi_row_batch() -> None:
    """seq_lengths describes one packed sequence, so it cannot span rows.

    The forward loops over the batch but applies the whole seq_lengths tensor to
    every row, which would silently give row 1 row 0's document layout.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model, kwargs = _peagle_training_model_and_batch(
        batch_size=2, seq_len=8, seq_lengths=torch.tensor([4, 4])
    )

    with pytest.raises(ValueError, match="single packed sequence"):
        model(**kwargs)


def test_peagle_rejects_packed_lengths_that_do_not_cover_the_sequence() -> None:
    """A short seq_lengths silently turns the tail into unattendable padding."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model, kwargs = _peagle_training_model_and_batch(
        batch_size=1, seq_len=8, seq_lengths=torch.tensor([3, 2])
    )

    with pytest.raises(ValueError, match="sum to 5"):
        model(**kwargs)


def test_peagle_checkpoint_export_unwraps_the_training_model() -> None:
    pytest.importorskip("torch")
    from types import SimpleNamespace

    base_trainer_mod = pytest.importorskip("verl_speco.trainer.base_trainer")

    draft = SimpleNamespace(name="draft")
    trainer = object.__new__(base_trainer_mod.DrafterBaseTrainer)
    trainer.model = SimpleNamespace(draft_model=draft)

    export_model, strip_prefixes = trainer._get_pretrained_export_model()

    assert export_model is draft
    assert "draft_model." in strip_prefixes
