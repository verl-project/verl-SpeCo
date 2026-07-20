"""Contract tests for the Domino drafter backend.

CPU-light: they exercise the Domino projector modules, the lambda-base
curriculum, the algorithm routing (config, oldlogprob aux layers, vLLM
guardrail), and the block-drafter classification. The full training forward is
validated on GPU by ``ci/domino_gpu_smoke.py``.
"""

from __future__ import annotations

import pytest


def _tiny_domino_config():
    from verl_speco.models.domino import DominoConfig

    return DominoConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        num_target_layers=4,
        num_context_layers=2,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        target_layer_ids=[1, 3],
        mask_token_id=31,
        block_size=4,
        num_anchors=8,
        emb_dim=6,
        gru_hidden_dim=10,
        pure_draft_prefix_len=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )


def test_domino_model_builds_projector_head() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.models.domino import DominoDraftModel

    config = _tiny_domino_config()
    model = DominoDraftModel(config)

    assert model.projector_type == "domino"
    # GRU consumes token embeddings (hidden_size) -> gru_hidden_dim.
    assert model.prefix_gru.input_size == config.hidden_size
    assert model.prefix_gru.hidden_size == config.gru_hidden_dim
    # embed_proj: [hidden + gru_hidden] -> emb_dim -> vocab.
    assert model.embed_proj[0].in_features == config.hidden_size + config.gru_hidden_dim
    assert model.embed_proj[0].out_features == config.emb_dim
    assert model.embed_proj[-1].out_features == config.vocab_size
    # It still carries the DFlash backbone token embedding (used by the GRU).
    assert model.embed_tokens.num_embeddings == config.vocab_size


def test_domino_forward_computes_top5_accuracy() -> None:
    """top5_correct_count must actually be reduced, not left at its zero init.

    DFlash and DSpark both compute it; a Domino regression here silently reports
    top5_acc=0 forever (base_trainer derives top5_acc from this counter).
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from verl_speco.backends.domino_trainer_backend import DominoTrainingModel
    from verl_speco.models.domino import DominoDraftModel

    torch.manual_seed(0)
    config = _tiny_domino_config()
    model = DominoTrainingModel(
        draft_model=DominoDraftModel(config),
        block_size=config.block_size,
        num_anchors=config.num_anchors,
        pure_draft_prefix_len=config.pure_draft_prefix_len,
    )

    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len))
    hidden_states_list = [torch.randn(bsz, seq_len, config.target_hidden_size) for _ in config.target_layer_ids]
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    lm_head_weight = torch.randn(config.vocab_size, config.hidden_size)

    _, _, _, _, _, diagnostics = model(input_ids, hidden_states_list, loss_mask, lm_head_weight)

    top1 = float(diagnostics["top1_correct_count"])
    top5 = float(diagnostics["top5_correct_count"])
    quality = float(diagnostics["quality_token_count"])

    assert quality > 0
    # top-5 is a superset of top-1, and with vocab_size=32 sampling 5 candidates
    # over that many tokens must hit at least one target.
    assert top5 >= top1
    assert top5 > 0


def test_domino_lambda_base_schedule() -> None:
    # get_lambda_base is pure-python, but its module (domino_trainer_backend)
    # subclasses the torch-based DFlash backend at import time, so it cannot be
    # imported without torch. Skip under the torch-free CPU CI like the siblings.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from verl_speco.backends.domino_trainer_backend import get_lambda_base

    assert get_lambda_base(0, decay_steps=100, lambda_start=1.0) == pytest.approx(1.0)
    assert get_lambda_base(50, decay_steps=100, lambda_start=1.0) == pytest.approx(0.5)
    assert get_lambda_base(100, decay_steps=100, lambda_start=1.0) == pytest.approx(0.0)
    assert get_lambda_base(200, decay_steps=100, lambda_start=1.0) == pytest.approx(0.0)
    assert get_lambda_base(25, decay_steps=100, lambda_start=0.4) == pytest.approx(0.3)


def test_domino_backend_is_block_drafter_metadata() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf

    from verl_speco.backends.domino_trainer_backend import DominoTrainerBackend

    backend = DominoTrainerBackend(
        OmegaConf.create({"rollout": {"drafter": {"training": {}}}, "model": {"path": "/tmp/none"}}),
        OmegaConf.create({}),
    )
    assert backend.model_type == "domino"


def test_domino_config_from_file_routes_to_domino(tmp_path) -> None:
    pytest.importorskip("transformers")
    import json

    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.domino import DominoConfig

    config = _tiny_domino_config().to_dict()
    config["architectures"] = ["DominoDraftModel"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    loaded = AutoDraftModelConfig.from_file(str(tmp_path / "config.json"))
    assert isinstance(loaded, DominoConfig)
    assert loaded.architectures == ["DominoDraftModel"]
    assert loaded.projector_type == "domino"


def test_domino_uses_dflash_aux_layers() -> None:
    from verl_speco.integration.oldlogprob_layer_ids import resolve_oldlogprob_aux_layer_ids

    layer_ids = resolve_oldlogprob_aux_layer_ids(
        {"speculative_algorithm": "DOMINO", "training": {"domino_num_target_layers": 5}},
        target_num_hidden_layers=36,
    )
    # Routes down the DFlash multi-context-layer branch (not the EAGLE default triple).
    assert layer_ids is not None
    assert len(layer_ids) == 5


def test_domino_rejected_by_vllm_config_builder() -> None:
    from verl_speco.integration.vllm_runtime import _speculative_method_from_drafter

    with pytest.raises(ValueError, match="projector sub-mode"):
        _speculative_method_from_drafter({"speculative_algorithm": "DOMINO"})


def test_domino_rejected_by_sglang_config_builder() -> None:
    from verl_speco.integration.sglang_runtime import _server_args_overrides_from_drafter

    with pytest.raises(ValueError, match="projector sub-mode"):
        _server_args_overrides_from_drafter(
            {"enable": True, "speculative_algorithm": "DOMINO"},
            supported_fields={"speculative_algorithm"},
        )
