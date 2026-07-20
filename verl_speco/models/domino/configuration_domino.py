from __future__ import annotations

import json
import os

from verl_speco.models.dflash import DFlashConfig


class DominoConfig(DFlashConfig):
    """Configuration for the Domino draft model.

    Domino uses the same target-hidden-state context backbone as DFlash and adds
    a causal correction head (a prefix GRU plus a low-rank ``embed_proj`` that
    produces a per-position logit delta). Enabled via ``projector_type='domino'``.
    """

    model_type = "domino"

    def __init__(
        self,
        *args,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_decay_gamma: float = 7.0,
        projector_type: str = "domino",
        emb_dim: int = 256,
        gru_hidden_dim: int = 1024,
        pure_draft_prefix_len: int = 1,
        shift_label: bool = True,
        lambda_base_start: float = 1.0,
        lambda_base_decay_steps: int = 2000,
        **kwargs,
    ):
        architectures = kwargs.pop("architectures", None)
        super().__init__(*args, **kwargs)
        self.architectures = architectures or ["DominoDraftModel"]
        self.block_size = int(block_size)
        self.num_anchors = int(num_anchors)
        self.loss_decay_gamma = float(loss_decay_gamma)
        self.projector_type = str(projector_type)
        self.emb_dim = int(emb_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.pure_draft_prefix_len = int(pure_draft_prefix_len)
        self.shift_label = bool(shift_label)
        self.lambda_base_start = float(lambda_base_start)
        self.lambda_base_decay_steps = int(lambda_base_decay_steps)

    @classmethod
    def from_domino_pretrained(cls, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["model_type"] = cls.model_type
        config["architectures"] = ["DominoDraftModel"]
        config.setdefault("projector_type", "domino")
        return cls.from_dict(config)
