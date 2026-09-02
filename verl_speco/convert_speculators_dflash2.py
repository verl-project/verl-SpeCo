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
"""Convert a speculators-format DFlash2 drafter into the z-lab checkpoint layout.

Public DFlash2 drafters trained with the ``speculators`` library (for example
``mgoin/Qwen3-4B-speculator.dflash2``) ship a ``config.json`` that nests the
transformer hyperparameters under ``transformer_layer_config`` and describes the
target through ``speculators_config``. Both this overlay's trainer
(``DFlash2Config.from_dflash2_pretrained``) and vLLM releases up to 0.28.0 read
the released z-lab layout instead: a flat Qwen3-style config with the DFlash2
knobs under ``dflash_config``. The weights use the same parameter names in both
layouts, so only ``config.json`` is rewritten; the safetensors are copied as-is.

Run::

    python -m verl_speco.convert_speculators_dflash2 \
        --input /path/to/speculators-dflash2 \
        --target /path/to/target-model \
        --output /path/to/dflash2-drafter
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from typing import Any

# Transformer hyperparameters copied verbatim from ``transformer_layer_config``.
_TRANSFORMER_KEYS = (
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "rope_parameters",
    "vocab_size",
)
# Sliding-window attention knobs. The trainer's DFlash backbone attends over the
# full anchored context, so the converted drafter defaults to full attention to
# keep training and rollout on the same model.
_SLIDING_WINDOW_KEYS = (
    "layer_types",
    "max_window_layers",
    "sliding_window",
    "use_sliding_window",
)
_DFLASH2_KEYS = (
    "block_size",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)
_WEIGHT_PATTERNS = ("*.safetensors", "*.safetensors.index.json")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return loaded


def _target_layer_ids(speculators_config: dict[str, Any]) -> list[int]:
    """The DFlash context layers, as decoder-layer indices.

    speculators records ``aux_hidden_state_layer_ids`` as indices into
    ``output_hidden_states`` (entry 0 is the embedding output), while the z-lab
    layout's ``target_layer_ids`` count decoder layers, hence the ``- 1``; this
    matches the ``eagle_aux_hidden_state_layer_ids = target_layer_ids + 1`` alias
    vLLM applies in the other direction.
    """
    if "target_layer_ids" in speculators_config:
        return [int(layer_id) for layer_id in speculators_config["target_layer_ids"]]
    aux_layer_ids = speculators_config.get("aux_hidden_state_layer_ids")
    if not aux_layer_ids:
        raise ValueError(
            "speculators DFlash2 config has neither target_layer_ids nor "
            "aux_hidden_state_layer_ids"
        )
    layer_ids = [int(layer_id) - 1 for layer_id in aux_layer_ids]
    if min(layer_ids) < 0:
        raise ValueError(
            f"aux_hidden_state_layer_ids={aux_layer_ids!r} must be >= 1 (hidden_states indices)"
        )
    return layer_ids


def convert_speculators_dflash2_config(
    speculators_config: dict[str, Any],
    target_config: dict[str, Any],
    *,
    keep_sliding_window: bool = False,
) -> dict[str, Any]:
    """Build the z-lab layout ``config.json`` for a speculators DFlash2 drafter."""
    model_type = str(speculators_config.get("speculators_model_type") or "").lower()
    algorithm = str(
        (speculators_config.get("speculators_config") or {}).get("algorithm") or ""
    ).lower()
    if "dflash2" not in (model_type, algorithm):
        raise ValueError(
            "not a speculators DFlash2 config: speculators_model_type="
            f"{model_type!r} algorithm={algorithm!r}"
        )
    transformer = speculators_config.get("transformer_layer_config")
    if not isinstance(transformer, dict):
        raise ValueError("speculators DFlash2 config lacks transformer_layer_config")

    target_text = target_config.get("text_config") or target_config
    target_num_hidden_layers = int(target_text["num_hidden_layers"])
    target_layer_ids = _target_layer_ids(speculators_config)
    if max(target_layer_ids) >= target_num_hidden_layers:
        raise ValueError(
            f"target_layer_ids={target_layer_ids} exceed the target's "
            f"num_hidden_layers={target_num_hidden_layers}"
        )

    converted: dict[str, Any] = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": str(transformer.get("model_type") or "qwen3"),
        "is_causal": False,
        "tie_word_embeddings": False,
        "use_cache": True,
        "dtype": str(speculators_config.get("dtype") or "bfloat16"),
    }
    for key in _TRANSFORMER_KEYS:
        if transformer.get(key) is not None:
            converted[key] = transformer[key]
    rope_parameters = transformer.get("rope_parameters")
    # transformers < 5 reads rope_theta at the top level.
    if (
        "rope_theta" not in converted
        and isinstance(rope_parameters, dict)
        and rope_parameters.get("rope_theta") is not None
    ):
        converted["rope_theta"] = rope_parameters["rope_theta"]
    if keep_sliding_window:
        for key in _SLIDING_WINDOW_KEYS:
            if key in transformer:
                converted[key] = transformer[key]
    else:
        converted["use_sliding_window"] = False
        converted["sliding_window"] = None
        converted["layer_types"] = ["full_attention"] * int(
            converted["num_hidden_layers"]
        )
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = transformer.get(key)
        if value is None:
            value = target_text.get(key)
        converted[key] = value

    draft_vocab_size = speculators_config.get("draft_vocab_size")
    if draft_vocab_size is not None and int(draft_vocab_size) != int(
        converted["vocab_size"]
    ):
        raise ValueError(
            "DFlash2 drafts share the target vocabulary; "
            f"draft_vocab_size={draft_vocab_size} != vocab_size={converted['vocab_size']}"
        )

    mask_token_id = speculators_config.get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("speculators DFlash2 config lacks mask_token_id")
    dflash_config = {
        key: speculators_config[key]
        for key in _DFLASH2_KEYS
        if speculators_config.get(key) is not None
    }
    missing = sorted(set(_DFLASH2_KEYS) - set(dflash_config))
    if missing:
        raise ValueError(f"speculators DFlash2 config lacks {missing}")
    dflash_config["mask_token_id"] = int(mask_token_id)
    dflash_config["target_layer_ids"] = target_layer_ids
    converted["dflash_config"] = dflash_config
    converted["num_target_layers"] = target_num_hidden_layers
    return converted


def convert_speculators_dflash2_checkpoint(
    input_dir: str,
    target_dir: str,
    output_dir: str,
    *,
    keep_sliding_window: bool = False,
    link_weights: bool = False,
) -> dict[str, Any]:
    """Write the converted drafter to ``output_dir`` and return its config."""
    speculators_config = _load_json(os.path.join(input_dir, "config.json"))
    target_config = _load_json(os.path.join(target_dir, "config.json"))
    converted = convert_speculators_dflash2_config(
        speculators_config, target_config, keep_sliding_window=keep_sliding_window
    )

    weight_files = sorted(
        path
        for pattern in _WEIGHT_PATTERNS
        for path in glob.glob(os.path.join(input_dir, pattern))
    )
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights under {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, sort_keys=True)
        f.write("\n")
    for source in weight_files:
        destination = os.path.join(output_dir, os.path.basename(source))
        if os.path.lexists(destination):
            os.remove(destination)
        if link_weights:
            os.symlink(os.path.abspath(source), destination)
        else:
            shutil.copy2(source, destination)
    return converted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input", required=True, help="speculators-format DFlash2 drafter directory"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="target model directory (its config.json supplies the layer count and special tokens)",
    )
    parser.add_argument("--output", required=True, help="converted drafter directory")
    parser.add_argument(
        "--keep-sliding-window",
        action="store_true",
        help="keep the speculators sliding-window attention settings instead of full attention",
    )
    parser.add_argument(
        "--link-weights",
        action="store_true",
        help="symlink the safetensors into the output instead of copying them",
    )
    args = parser.parse_args(argv)
    converted = convert_speculators_dflash2_checkpoint(
        args.input,
        args.target,
        args.output,
        keep_sliding_window=args.keep_sliding_window,
        link_weights=args.link_weights,
    )
    print(
        f"wrote {args.output}: architectures={converted['architectures']} "
        f"dflash_config={json.dumps(converted['dflash_config'], sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
