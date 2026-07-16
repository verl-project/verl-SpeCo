"""Configuration for the EAGLE-1 / EAGLE-2 dense draft model.

EAGLE-1 and EAGLE-2 share the same draft architecture and training objective;
they differ only in the speculative-decoding tree policy at inference time (see
``verl_speco.backends.eagle1_trainer_backend``). This config is a thin extension
of ``LlamaConfig`` that adds the draft-specific fields consumed by
``LlamaForCausalLMEagle1``.
"""

from transformers import LlamaConfig


class Eagle1Config(LlamaConfig):
    """Llama-style dense draft config for EAGLE-1 / EAGLE-2.

    Extra fields on top of ``LlamaConfig``:
        draft_num_hidden_layers: number of decoder layers in the draft (EAGLE
            uses a single layer by default).
        target_hidden_size: hidden size of the (frozen) target model, i.e. the
            width of the target feature fed into the draft ``fc`` fusion. Defaults
            to ``hidden_size`` when the draft and target share a width.
        num_aux_hidden_states: number of target hidden layers concatenated into
            the collected feature. EAGLE-1/2 fuse a single (last) layer, so this
            is fixed to 1 and is only kept for compatibility with the shared
            EAGLE data-collection plumbing.
    """

    model_type = "llama_eagle1"

    def __init__(
        self,
        draft_num_hidden_layers: int = 1,
        target_hidden_size: int | None = None,
        num_aux_hidden_states: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.draft_num_hidden_layers = int(draft_num_hidden_layers)
        self.target_hidden_size = int(target_hidden_size) if target_hidden_size is not None else self.hidden_size
        self.num_aux_hidden_states = int(num_aux_hidden_states)
