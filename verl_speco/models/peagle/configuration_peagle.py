"""Configuration for the P-EAGLE (parallel-drafting) draft model."""

from transformers import LlamaConfig


class PeagleConfig(LlamaConfig):
    """Llama-style parallel-drafting EAGLE draft config.

    Extra fields on top of ``LlamaConfig`` (all serialized to config.json so the
    checkpoint reloads / serves unchanged):
        num_draft_layers: number of draft decoder layers (layer 0 is the fused
            ``[embed, hidden]`` layer; deeper layers are vanilla H-dim blocks).
        target_hidden_size: hidden size of the frozen target (aux feature width).
        num_aux_hidden_states: number of target aux layers concatenated into the
            collected feature and fused by ``fc``.
        draft_vocab_size: reduced draft vocabulary (t2d/d2t); defaults to vocab.
        num_depths: number of parallel prediction depths ``K``.
        down_sample_ratio / down_sample_ratio_min: COD geometric decay + floor.
        mask_token_id: token id used for masked (depth>=1) draft slots.
        fc_norm / norm_output: EAGLE-3.1-style toggles.
    """

    model_type = "llama_peagle"

    def __init__(
        self,
        num_draft_layers: int = 4,
        target_hidden_size: int | None = None,
        num_aux_hidden_states: int = 3,
        draft_vocab_size: int | None = None,
        num_depths: int = 8,
        down_sample_ratio: float = 0.7,
        down_sample_ratio_min: float = 0.2,
        mask_token_id: int | None = None,
        fc_norm: bool = False,
        norm_output: bool = False,
        parallel_drafting: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_draft_layers = int(num_draft_layers)
        self.target_hidden_size = int(target_hidden_size) if target_hidden_size is not None else self.hidden_size
        self.num_aux_hidden_states = int(num_aux_hidden_states)
        self.draft_vocab_size = int(draft_vocab_size) if draft_vocab_size is not None else self.vocab_size
        self.num_depths = int(num_depths)
        self.down_sample_ratio = float(down_sample_ratio)
        self.down_sample_ratio_min = float(down_sample_ratio_min)
        self.mask_token_id = int(mask_token_id) if mask_token_id is not None else self.vocab_size - 1
        self.fc_norm = bool(fc_norm)
        self.norm_output = bool(norm_output)
        self.parallel_drafting = bool(parallel_drafting)
