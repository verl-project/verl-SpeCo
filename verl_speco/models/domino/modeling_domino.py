from __future__ import annotations

import torch.nn as nn

from verl_speco.models.dflash import DFlashDraftModel

from .configuration_domino import DominoConfig


class DominoDraftModel(DFlashDraftModel):
    """DFlash block drafter plus the Domino causal correction head.

    Adds two modules on top of the DFlash backbone (both faithful to the
    AutoModel ``projector_type='domino'`` build in ``draft_qwen3.py``):

    - ``prefix_gru``: a single-layer GRU over the block's previous-token
      embeddings, producing a causal state for each block position.
    - ``embed_proj``: a low-rank MLP over ``[backbone hidden | GRU state]`` that
      emits a full-vocabulary logit delta added to the parallel base logits.

    The head is applied by the training wrapper (``DominoTrainingModel``), mirroring
    how DSpark keeps the Markov head callable but applies the bias in the trainer.
    """

    config_class = DominoConfig

    def __init__(self, config: DominoConfig):
        super().__init__(config)
        self.projector_type = str(getattr(config, "projector_type", "domino"))
        self.pure_draft_prefix_len = int(getattr(config, "pure_draft_prefix_len", 1))
        self.shift_label = bool(getattr(config, "shift_label", True))
        self.emb_dim = int(getattr(config, "emb_dim", 256))
        self.gru_hidden_dim = int(getattr(config, "gru_hidden_dim", 1024))

        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_size + self.gru_hidden_dim, self.emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.emb_dim, config.vocab_size, bias=False),
        )


__all__ = ["DominoConfig", "DominoDraftModel"]
