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
import json

import pytest

from verl_speco.convert_peagle_vllm import convert_checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama_peagle",
                "vocab_size": 256,
                "draft_vocab_size": 256,
                "num_aux_hidden_states": 3,
            }
        )
    )
    (source / "model.safetensors").write_bytes(b"unchanged weights")
    return source, tmp_path / "destination"


def test_preserves_checkpoint_bytes(checkpoint):
    source, destination = checkpoint
    convert_checkpoint(source, destination, [0, 1, 2])
    assert (destination / "model.safetensors").read_bytes() == (
        source / "model.safetensors"
    ).read_bytes()
    assert (
        json.loads((source / "config.json").read_text())["model_type"] == "llama_peagle"
    )
    config = json.loads((destination / "config.json").read_text())
    assert config["architectures"] == ["Eagle3LlamaForCausalLM"]
    assert config["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [0, 1, 2]


@pytest.mark.parametrize("layers", [[0, 1], [0, 1, 1], [2, 1, 0], [-1, 1, 2]])
def test_rejects_wrong_auxiliary_layers(checkpoint, layers):
    source, destination = checkpoint
    with pytest.raises(ValueError, match="Target layer IDs"):
        convert_checkpoint(source, destination, layers)
    assert not destination.exists()


@pytest.mark.parametrize(
    "key,value", [("model_type", "llama"), ("draft_vocab_size", 128)]
)
def test_rejects_unvalidated_checkpoint(checkpoint, key, value):
    source, destination = checkpoint
    config = json.loads((source / "config.json").read_text())
    config[key] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        convert_checkpoint(source, destination, [0, 1, 2])
    assert not destination.exists()
