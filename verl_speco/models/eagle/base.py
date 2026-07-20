import glob
import json
import os
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel


def load_checkpoint(model_path: str, key: str) -> torch.Tensor:
    if not os.path.exists(model_path):
        # this is the case where model_path is a huggingface repository
        # we first need to locate its local cache
        model_path = snapshot_download(repo_id=model_path)

    # check if there is file ending with index.json
    glob_path = os.path.join(model_path, "*.index.json")
    index_json_path = glob.glob(glob_path)

    if len(index_json_path) == 0:
        # No index.json found, look for single model file
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                return f.get_tensor(key)

        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            return state_dict[key]

        raise FileNotFoundError(
            f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
        )
    if len(index_json_path) > 1:
        raise FileNotFoundError(f"Multiple index.json files found in {model_path}")
    index_json_path = index_json_path[0]

    with open(index_json_path, "r") as f:
        index_json = json.load(f)
    ckpt_file = index_json["weight_map"][key]

    if ckpt_file.endswith(".safetensors"):
        with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
            return f.get_tensor(key)
    else:
        state_dict = torch.load(os.path.join(model_path, ckpt_file))
        return state_dict[key]


class DraftModel(PreTrainedModel):
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input ids.
        """
        raise NotImplementedError("Subclasses must implement embed_input_ids")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the draft model.
        """
        raise NotImplementedError("Subclasses must implement compute_logits")

    def prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """
        Prepare the attention mask of the draft model.
        """

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        The backbone of the draft model.
        """
        raise NotImplementedError("Subclasses must implement backbone")

    def freeze_embedding(self) -> None:
        """
        Freeze the embeddings of the draft model so that they are not updated during training.
        """
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
            repository ID or a local directory path containing the model files.
        """
        emb_tokens = load_checkpoint(model_path, embedding_key)
        self.embed_tokens.weight.copy_(emb_tokens)


class Eagle3DraftModel(DraftModel):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """

    drafter_model_type = "LlamaForCausalLMEagle3"

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """
        raise NotImplementedError("Subclasses must implement project_hidden_states")

    def load_vocab_mapping(self, file_path: str) -> None:
        """
        Load the vocab buffers of the draft model.

        Args:
            file_path (str): The path to the vocab mapping file.
        """
        assert hasattr(self, "t2d") and hasattr(self, "d2t"), (
            "t2d and d2t buffersare not found in the draft model, please check your draft model implementation"
        )
        vocab_mapping = torch.load(file_path, map_location=self.t2d.device)
        t2d = vocab_mapping["t2d"].to(device=self.t2d.device, dtype=self.t2d.dtype)
        d2t = vocab_mapping["d2t"].to(device=self.d2t.device, dtype=self.d2t.dtype)
        if t2d.shape != self.t2d.shape:
            raise ValueError(
                f"Expected t2d shape {tuple(self.t2d.shape)}, got {tuple(t2d.shape)}"
            )
        if d2t.shape != self.d2t.shape:
            raise ValueError(
                f"Expected d2t shape {tuple(self.d2t.shape)}, got {tuple(d2t.shape)}"
            )
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)
        self.vocab_mapping_loaded = True
