import json
import os

from typing import Union
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    LlamaConfig,
    PretrainedConfig,
    modeling_utils,
)

from .dflash import DFlashConfig
from .eagle.llama_eagle import LlamaForCausalLMEagle3


_EAGLE3_ARCHITECTURE_ALIASES = {
    "LlamaForCausalLMEagle3",
    "Qwen3Eagle3Model",
}

_DSPARK_ARCHITECTURE_ALIASES = {
    "DSparkDraftModel",
    "Qwen3DSparkModel",
}

_DOMINO_ARCHITECTURE_ALIASES = {
    "DominoDraftModel",
    "Qwen3DominoModel",
}


def _normalize_int_list(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("["):
            value = json.loads(raw)
        else:
            value = [part.strip() for part in raw.split(",") if part.strip()]
    return [int(item) for item in list(value)]


def _first_present_eagle3_layer_ids(config: dict):
    eagle_config = config.get("eagle_config")
    candidates = []
    if isinstance(eagle_config, dict):
        candidates.extend(
            [
                eagle_config.get("target_hidden_layer_ids"),
                eagle_config.get("eagle_aux_hidden_state_layer_ids"),
            ]
        )
    candidates.extend(
        [
            config.get("target_hidden_layer_ids"),
            config.get("eagle_aux_hidden_state_layer_ids"),
            config.get("target_layer_ids"),
        ]
    )
    for candidate in candidates:
        layer_ids = _normalize_int_list(candidate)
        if layer_ids is not None:
            return layer_ids
    return None


def _normalize_eagle3_config_dict(config: dict) -> dict:
    layer_ids = _first_present_eagle3_layer_ids(config)
    if layer_ids is not None:
        configured_count = config.get("num_aux_hidden_states")
        if configured_count is not None and int(configured_count) != len(layer_ids):
            raise ValueError(
                "EAGLE3 num_aux_hidden_states does not match layer ids: "
                f"{configured_count} != {len(layer_ids)}"
            )
        config["num_aux_hidden_states"] = len(layer_ids)
        config["eagle_aux_hidden_state_layer_ids"] = layer_ids
        eagle_config = config.get("eagle_config")
        if not isinstance(eagle_config, dict):
            eagle_config = {}
            config["eagle_config"] = eagle_config
        eagle_config.setdefault("eagle_aux_hidden_state_layer_ids", layer_ids)
        eagle_config.setdefault("target_hidden_layer_ids", layer_ids)
    elif "num_aux_hidden_states" in config:
        config["num_aux_hidden_states"] = int(config["num_aux_hidden_states"])

    config["architectures"] = ["LlamaForCausalLMEagle3"]
    return config


class AutoDraftModel(AutoModelForCausalLMBase):
    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        """
        This class method takes a configuration object and create its model based on the
        _model_mapping class variable.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        # get the model class from the
        _model_cls = cls._model_mapping[type(config)]
        model = _model_cls(config, **config_kwargs)

        # Convert model to specified dtype if provided
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoEagle3DraftModel(AutoDraftModel):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
    }


class AutoDraftModelConfig:
    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "Qwen3Eagle3Model": LlamaConfig,
        "DFlashDraftModel": DFlashConfig,
    }

    @classmethod
    def from_file(cls, config_path: str):
        """
        This class method takes a configuration file path and create its configuration object based on the
        _config_mapping class variable.

        Args:
            config_path (str): A path to a configuration file.

        Returns:
            A configuration object.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        if "tie_word_embeddings" in config:
            print("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        # check for architectures
        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if (
            architecture not in cls._config_mapping
            and architecture not in _DSPARK_ARCHITECTURE_ALIASES
            and architecture not in _DOMINO_ARCHITECTURE_ALIASES
        ):
            raise ValueError(f"Architecture {architecture} not supported")

        config_class = cls._config_mapping.get(architecture)
        if architecture == "DFlashDraftModel":
            config["model_type"] = DFlashConfig.model_type
            config["architectures"] = ["DFlashDraftModel"]
        elif architecture in _DSPARK_ARCHITECTURE_ALIASES:
            from .dspark import DSparkConfig

            return DSparkConfig.from_dspark_dict(config)
        elif architecture in _DOMINO_ARCHITECTURE_ALIASES:
            from .domino import DominoConfig

            config_class = DominoConfig
            config["model_type"] = DominoConfig.model_type
            config["architectures"] = ["DominoDraftModel"]
            config.setdefault("projector_type", "domino")
        elif architecture in _EAGLE3_ARCHITECTURE_ALIASES:
            config = _normalize_eagle3_config_dict(config)

        if config_class is None:
            raise ValueError(f"Architecture {architecture} not supported")
        return config_class.from_dict(config)
