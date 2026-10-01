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
"""Export a frozen full-vocabulary P-EAGLE checkpoint for vLLM 0.29."""

import argparse
import json
from pathlib import Path
import shutil


def convert_checkpoint(
    source: Path, destination: Path, target_layer_ids: list[int]
) -> None:
    config = json.loads((source / "config.json").read_text())
    if config["model_type"] != "llama_peagle":
        raise ValueError("Expected a SpeCo llama_peagle checkpoint")
    if config["draft_vocab_size"] != config["vocab_size"]:
        raise ValueError("Reduced-vocabulary serving is not yet validated")
    if (
        len(target_layer_ids) != config["num_aux_hidden_states"]
        or target_layer_ids != sorted(set(target_layer_ids))
        or min(target_layer_ids) < 0
    ):
        raise ValueError("Target layer IDs must match the ordered auxiliary features")
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError("Expected safetensors checkpoint weights")

    # vLLM's EAGLE3 class implements the same fused-first-layer architecture.
    # The proposer must additionally enable parallel_drafting explicitly.
    config.update(model_type="llama", architectures=["Eagle3LlamaForCausalLM"])
    config["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
    destination.mkdir(parents=True, exist_ok=False)
    for path in weights + sorted(source.glob("*.safetensors.index.json")):
        shutil.copy2(path, destination / path.name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination, args.target_layer_ids)


if __name__ == "__main__":
    main()
