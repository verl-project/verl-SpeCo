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
import importlib
import json

import pytest

torch = pytest.importorskip("torch")

draft_dataset = importlib.import_module("verl_speco.trainer.draft_dataset")
feature_store = importlib.import_module("verl_speco.trainer.feature_store")
DraftFeatureDataLoader = draft_dataset.DraftFeatureDataLoader
DraftFeatureDataLoaderConfig = draft_dataset.DraftFeatureDataLoaderConfig
DraftFeatureSample = feature_store.DraftFeatureSample
DraftReplaySample = feature_store.DraftReplaySample
JsonlTokenReplayFeatureStore = feature_store.JsonlTokenReplayFeatureStore
TokenReplayFeatureStore = feature_store.TokenReplayFeatureStore
TorchShardFeatureStore = feature_store.TorchShardFeatureStore
VllmSafetensorsFeatureStore = feature_store.VllmSafetensorsFeatureStore
build_feature_store_from_config = feature_store.build_feature_store_from_config


def _sample(index: int = 0):
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long) + index
    loss_mask = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    hidden_states = torch.randn(4, 8, dtype=torch.float32)
    last_hidden_states = torch.randn(4, 4, dtype=torch.float32)
    return DraftFeatureSample(
        algorithm="EAGLE3",
        input_ids=input_ids,
        loss_mask=loss_mask,
        hidden_states=hidden_states,
        last_hidden_states=last_hidden_states,
        metadata={
            "source": "unit",
            "global_step": index,
            "hidden_states_layout": "eagle3_aux_plus_last",
            "sequence_length": 4,
            "loss_tokens": 2,
        },
    )


def _sample_without_step(index: int = 0):
    sample = _sample(index)
    sample.metadata.pop("global_step", None)
    sample.metadata.pop("step", None)
    return sample


def test_torch_shard_feature_store_roundtrip(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=2)
    store.write_many([_sample(0), _sample(1)])
    store.close()

    reader = TorchShardFeatureStore(tmp_path, read_only=True)
    keys = list(reader.iter_keys(shuffle=False))
    assert len(keys) == 2
    loaded = reader.read(keys[0])
    assert loaded.algorithm == "EAGLE3"
    assert torch.equal(loaded.input_ids, torch.tensor([1, 2, 3, 4]))
    assert loaded.metadata["hidden_states_layout"] == "eagle3_aux_plus_last"
    assert reader.get_metadata()["num_samples"] == 2


def test_token_replay_feature_store_roundtrip(tmp_path):
    sample = DraftReplaySample(
        algorithm="DSPARK",
        input_ids=torch.arange(12, dtype=torch.long),
        loss_mask=torch.ones(12, dtype=torch.float32),
        attention_mask=torch.ones(12, dtype=torch.bool),
        position_ids=torch.arange(12, dtype=torch.long),
        feature_positions=torch.arange(4, 10, dtype=torch.long),
        draft_position_ids=torch.arange(5, 11, dtype=torch.long),
        metadata={"target_model_path": "/target", "global_step": 3},
    )
    store = TokenReplayFeatureStore(tmp_path, max_samples_per_shard=1)
    store.write_many([sample])
    store.close()

    reader = TokenReplayFeatureStore(tmp_path, read_only=True)
    loaded = reader.read(next(reader.iter_keys()))

    assert loaded.algorithm == "DSPARK"
    assert loaded.input_ids.dtype == torch.int32
    assert loaded.attention_mask.dtype == torch.bool
    assert torch.equal(loaded.feature_positions, torch.arange(4, 10, dtype=torch.int32))
    assert "hidden_states" not in loaded.to_dict()
    assert reader.get_metadata()["format"] == "token_replay"


def test_token_replay_rejects_non_contiguous_feature_positions():
    sample = DraftReplaySample(
        input_ids=torch.arange(8),
        loss_mask=torch.ones(8),
        attention_mask=torch.ones(8, dtype=torch.bool),
        position_ids=torch.arange(8),
        feature_positions=torch.tensor([2, 4]),
        draft_position_ids=torch.tensor([3, 5]),
    )

    with pytest.raises(ValueError, match="contiguous"):
        sample.validate(strict=True)


def test_jsonl_token_replay_reads_input_ids_and_loss_mask(tmp_path):
    path = tmp_path / "samples.jsonl"
    row = {
        "id": "sample-0",
        "input_ids": list(range(10)),
        "loss_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        "text": "ignored for replay",
        "metadata": {"source": "unit"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    store = JsonlTokenReplayFeatureStore(path, read_only=True, max_seq_len=4)
    keys = list(store.iter_keys(shuffle=False))
    loaded = store.read(keys[0])

    assert keys == ["samples.jsonl:0"]
    assert loaded.algorithm == "EAGLE3"
    assert torch.equal(loaded.input_ids, torch.arange(10))
    assert torch.equal(loaded.loss_mask, torch.tensor(row["loss_mask"], dtype=torch.float32))
    assert torch.equal(loaded.attention_mask, torch.ones(10, dtype=torch.bool))
    assert torch.equal(loaded.position_ids, torch.arange(10))
    assert torch.equal(loaded.feature_positions, torch.arange(3, 7))
    assert torch.equal(loaded.draft_position_ids, torch.arange(4, 8))
    assert loaded.metadata["source"] == "jsonl_token_replay"
    assert loaded.metadata["id"] == "sample-0"
    assert store.get_metadata()["format"] == "jsonl_token_replay"
    assert store.get_metadata()["num_samples"] == 1


def test_build_feature_store_from_config_supports_jsonl_token_replay(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}) + "\n",
        encoding="utf-8",
    )

    store = build_feature_store_from_config(
        {
            "type": "jsonl_token_replay",
            "path": path,
            "max_seq_len": 8,
        },
        read_only=True,
    )
    loaded = store.read(next(store.iter_keys(shuffle=False)))

    assert isinstance(loaded, DraftReplaySample)
    assert torch.equal(loaded.feature_positions, torch.arange(0, 3))


def test_jsonl_token_replay_reads_conversations_with_chat_template(tmp_path):
    class FakeTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ):
            assert tokenize is True
            token_ids = []
            for message in messages:
                role_id = {"user": 10, "assistant": 20, "system": 30}[message["role"]]
                token_ids.extend([role_id, len(message["content"])])
            if add_generation_prompt:
                token_ids.append(20)
            return token_ids

    path = tmp_path / "samples.jsonl"
    row = {
        "id": "conv-0",
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "assistant", "value": "answer"},
        ],
        "algorithm": "DSPARK",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    store = JsonlTokenReplayFeatureStore(
        path,
        read_only=True,
        max_seq_len=8,
        tokenizer_path="/target",
    )
    store._tokenizer = FakeTokenizer()
    loaded = store.read(next(store.iter_keys(shuffle=False)))

    assert loaded.algorithm == "DSPARK"
    assert torch.equal(loaded.input_ids, torch.tensor([10, 8, 20, 6]))
    assert torch.equal(loaded.loss_mask, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(loaded.feature_positions, torch.arange(2, 4))
    assert torch.equal(loaded.draft_position_ids, torch.arange(3, 5))
    assert loaded.metadata["source"] == "jsonl_conversations"
    assert loaded.metadata["id"] == "conv-0"


def test_jsonl_token_replay_accepts_tensor_chat_template_output(tmp_path):
    class TensorTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ):
            token_ids = []
            for message in messages:
                token_ids.extend([1 if message["role"] == "user" else 2, 3])
            if add_generation_prompt:
                token_ids.append(2)
            return torch.tensor([token_ids], dtype=torch.long)

    path = tmp_path / "samples.jsonl"
    row = {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "assistant", "value": "answer"},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    store = JsonlTokenReplayFeatureStore(
        path,
        read_only=True,
        tokenizer_path="/target",
    )
    store._tokenizer = TensorTokenizer()
    loaded = store.read(next(store.iter_keys(shuffle=False)))

    assert torch.equal(loaded.input_ids, torch.tensor([1, 3, 2, 3]))
    assert torch.equal(loaded.loss_mask, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_jsonl_token_replay_accepts_batch_encoding_chat_template_output(tmp_path):
    class FakeBatchEncoding:
        def __init__(self, input_ids):
            self.data = {"input_ids": input_ids}

    class BatchEncodingTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ):
            token_ids = []
            for message in messages:
                token_ids.extend([1 if message["role"] == "user" else 2, 3])
            if add_generation_prompt:
                token_ids.append(2)
            return FakeBatchEncoding([token_ids])

    path = tmp_path / "samples.jsonl"
    row = {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "assistant", "value": "answer"},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    store = JsonlTokenReplayFeatureStore(
        path,
        read_only=True,
        tokenizer_path="/target",
    )
    store._tokenizer = BatchEncodingTokenizer()
    loaded = store.read(next(store.iter_keys(shuffle=False)))

    assert torch.equal(loaded.input_ids, torch.tensor([1, 3, 2, 3]))
    assert torch.equal(loaded.loss_mask, torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_vllm_safetensors_feature_store_records_manifest_path_and_roundtrips(tmp_path):
    pytest.importorskip("safetensors.torch")
    hidden_positions = torch.arange(160, dtype=torch.long)
    sample = DraftFeatureSample(
        algorithm="DSPARK",
        input_ids=torch.arange(160, dtype=torch.long),
        loss_mask=torch.ones(160, dtype=torch.float32),
        hidden_states=torch.randn(160, 16, dtype=torch.float32),
        position_ids=torch.arange(10, 170, dtype=torch.long),
        metadata={
            "source": "token_replay_vllm_file",
            "global_step": 7,
            "hidden_states_layout": "dflash_aux_plus_last",
            "hidden_positions": hidden_positions,
            "feature_start": 10,
            "feature_end": 170,
        },
    )
    store = VllmSafetensorsFeatureStore(tmp_path)

    keys = store.write_many([sample])
    store.close()

    manifest_lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    entry = json.loads(manifest_lines[0])
    assert entry["path"].endswith(".safetensors")
    assert (tmp_path / entry["path"]).exists()
    assert entry["sample"]["metadata"]["hidden_positions"] == {
        "__tensor__": True,
        "dtype": "torch.int64",
        "shape": [160],
    }

    reader = VllmSafetensorsFeatureStore(tmp_path, read_only=True)
    loaded = reader.read(keys[0])

    assert loaded.algorithm == "DSPARK"
    assert torch.equal(loaded.input_ids, sample.input_ids)
    assert torch.equal(loaded.position_ids, sample.position_ids)
    assert torch.equal(loaded.hidden_states, sample.hidden_states)
    assert torch.equal(loaded.metadata["hidden_positions"], hidden_positions)
    assert reader.get_metadata()["format"] == "vllm_safetensors"
    assert reader.get_metadata()["num_samples"] == 1


def test_build_feature_store_from_config_supports_vllm_safetensors(tmp_path):
    pytest.importorskip("safetensors.torch")
    writer = build_feature_store_from_config(
        {"type": "vllm_safetensors", "path": tmp_path}
    )
    writer.write_many([_sample(0)])
    writer.close()

    reader = build_feature_store_from_config(
        {"type": "vllm_safetensors", "path": tmp_path}, read_only=True
    )
    loaded = reader.read(next(reader.iter_keys(shuffle=False)))

    assert isinstance(loaded, DraftFeatureSample)
    assert torch.equal(loaded.input_ids, torch.tensor([1, 2, 3, 4]))


def test_feature_sample_normalizes_singleton_position_ids():
    sample = DraftFeatureSample(
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        loss_mask=torch.tensor([0, 1, 1, 0], dtype=torch.float32),
        hidden_states=torch.randn(4, 8, dtype=torch.float32),
        position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    )

    sample.validate(strict=True)

    assert sample.position_ids.shape == (4,)


def test_feature_sample_rejects_position_id_length_mismatch():
    sample = DraftFeatureSample(
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        loss_mask=torch.tensor([0, 1, 1, 0], dtype=torch.float32),
        hidden_states=torch.randn(4, 8, dtype=torch.float32),
        position_ids=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="input_ids/position_ids length mismatch"):
        sample.validate(strict=True)


def test_feature_sample_restores_online_alignment_metadata():
    sample = DraftFeatureSample(
        input_ids=torch.arange(127, dtype=torch.long),
        loss_mask=torch.ones(127, dtype=torch.float32),
        hidden_states=torch.randn(127, 12288, dtype=torch.float32),
        target_logprobs=torch.zeros(126, 128, 2, dtype=torch.float32),
        position_ids=torch.arange(140, 267, dtype=torch.long),
        metadata={
            "global_step": 1,
            "hidden_states_layout": "eagle3_aux_plus_last",
            "full_sequence_length": 1164,
            "feature_start": 139,
            "feature_end": 266,
            "hidden_position_start": 139,
            "hidden_position_end": 266,
            "hidden_positions": torch.arange(139, 266, dtype=torch.long),
            "target_logprobs_position_start": 140,
            "target_logprobs_position_end": 266,
        },
    )

    item = sample.to_training_item()

    assert item["_verl_feature_start"] == 139
    assert item["_verl_feature_end"] == 266
    assert item["_verl_target_position_start"] == 140
    assert item["_verl_target_position_end"] == 266
    assert item["_verl_target_tensor_position_start"] == 140
    assert item["_verl_target_tensor_position_end"] == 266
    assert item["_verl_target_start"] == 0
    assert item["_verl_target_end"] == 126
    assert torch.equal(
        item["_verl_hidden_positions"], torch.arange(139, 266, dtype=torch.long)
    )


def test_draft_feature_dataloader_slices_keys_by_rank(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=4)
    store.write_many([_sample(i) for i in range(4)])
    store.close()

    rank0 = DraftFeatureDataLoader(
        TorchShardFeatureStore(tmp_path, read_only=True),
        DraftFeatureDataLoaderConfig(
            batch_size=8, rank=0, world_size=2, shuffle=False, repeat=False
        ),
    )
    rank1 = DraftFeatureDataLoader(
        TorchShardFeatureStore(tmp_path, read_only=True),
        DraftFeatureDataLoaderConfig(
            batch_size=8, rank=1, world_size=2, shuffle=False, repeat=False
        ),
    )

    rank0_ids = [int(sample.input_ids[0].item()) for batch in rank0 for sample in batch]
    rank1_ids = [int(sample.input_ids[0].item()) for batch in rank1 for sample in batch]
    assert rank0_ids == [1, 3]
    assert rank1_ids == [2, 4]


def test_draft_feature_dataloader_balances_uneven_distributed_shards(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=5)
    store.write_many([_sample(i) for i in range(5)])
    store.close()

    rank_batches = []
    for rank in range(2):
        loader = DraftFeatureDataLoader(
            TorchShardFeatureStore(tmp_path, read_only=True),
            DraftFeatureDataLoaderConfig(
                batch_size=1, rank=rank, world_size=2, shuffle=False, repeat=False
            ),
        )
        rank_batches.append(list(loader))

    assert [len(batches) for batches in rank_batches] == [2, 2]


def test_draft_feature_dataloader_filters_samples_by_step_window(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=8)
    store.write_many([_sample(i) for i in range(6)])
    store.close()

    loader = DraftFeatureDataLoader(
        TorchShardFeatureStore(tmp_path, read_only=True),
        DraftFeatureDataLoaderConfig(
            batch_size=8,
            shuffle=False,
            repeat=False,
            min_sample_step=2,
            max_sample_step=4,
        ),
    )

    steps = [
        int(sample.metadata["global_step"]) for batch in loader for sample in batch
    ]
    assert steps == [2, 3, 4]


def test_draft_feature_dataloader_keeps_samples_without_step_metadata(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=8)
    store.write_many([_sample(1), _sample_without_step(10), _sample(5)])
    store.close()

    loader = DraftFeatureDataLoader(
        TorchShardFeatureStore(tmp_path, read_only=True),
        DraftFeatureDataLoaderConfig(
            batch_size=8,
            shuffle=False,
            repeat=False,
            min_sample_step=2,
            max_sample_step=4,
        ),
    )

    input_starts = [
        int(sample.input_ids[0].item()) for batch in loader for sample in batch
    ]
    assert input_starts == [11]


def test_draft_feature_dataloader_filters_before_distributed_slicing(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=8)
    store.write_many([_sample(i) for i in range(6)])
    store.close()

    rank_batches = []
    for rank in range(2):
        loader = DraftFeatureDataLoader(
            TorchShardFeatureStore(tmp_path, read_only=True),
            DraftFeatureDataLoaderConfig(
                batch_size=8,
                rank=rank,
                world_size=2,
                shuffle=False,
                repeat=False,
                min_sample_step=1,
                max_sample_step=4,
            ),
        )
        rank_batches.append(
            [
                int(sample.metadata["global_step"])
                for batch in loader
                for sample in batch
            ]
        )

    assert rank_batches == [[1, 3], [2, 4]]


def test_draft_feature_dataloader_rejects_rank_out_of_range(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=4)

    with pytest.raises(ValueError, match="Invalid rank/world_size configuration"):
        DraftFeatureDataLoader(
            store,
            DraftFeatureDataLoaderConfig(batch_size=1, rank=2, world_size=2),
        )


def test_draft_feature_dataloader_rejects_non_positive_world_size(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=4)

    with pytest.raises(ValueError, match="Invalid world_size"):
        DraftFeatureDataLoader(
            store,
            DraftFeatureDataLoaderConfig(batch_size=1, rank=0, world_size=0),
        )


def test_flush_interval_zero_relies_on_shard_capacity(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=4)
    store.write_many([_sample(0), _sample(1)])

    assert store.flush_on_step(global_step=1, interval_steps=0) == []
    assert list(tmp_path.glob("shard_*.pt")) == []

    store.write_many([_sample(2), _sample(3)])
    assert len(list(tmp_path.glob("shard_*.pt"))) == 1
    assert store.get_metadata()["num_samples"] == 4


def test_flush_interval_one_flushes_once_per_step(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=32)
    store.write_many([_sample(index) for index in range(16)])

    keys = store.flush_on_step(global_step=1, interval_steps=1)

    assert len(keys) == 16
    assert len(list(tmp_path.glob("shard_*.pt"))) == 1
    assert store.get_metadata()["num_samples"] == 16


def test_flush_interval_n_only_flushes_matching_steps(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=32)
    store.write_many([_sample(0)])

    assert store.flush_on_step(global_step=1, interval_steps=2) == []
    assert list(tmp_path.glob("shard_*.pt")) == []
    assert len(store.flush_on_step(global_step=2, interval_steps=2)) == 1
    assert len(list(tmp_path.glob("shard_*.pt"))) == 1
