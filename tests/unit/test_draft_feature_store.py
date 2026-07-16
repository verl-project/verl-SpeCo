import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.draft_dataset import DraftFeatureDataLoader, DraftFeatureDataLoaderConfig
from verl_speco.trainer.feature_store import DraftFeatureSample, TorchShardFeatureStore


def _sample(index: int = 0) -> DraftFeatureSample:
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
    assert torch.equal(item["_verl_hidden_positions"], torch.arange(139, 266, dtype=torch.long))


def test_draft_feature_dataloader_slices_keys_by_rank(tmp_path):
    store = TorchShardFeatureStore(tmp_path, max_samples_per_shard=4)
    store.write_many([_sample(i) for i in range(4)])
    store.close()

    rank0 = DraftFeatureDataLoader(
        TorchShardFeatureStore(tmp_path, read_only=True),
        DraftFeatureDataLoaderConfig(batch_size=8, rank=0, world_size=2, shuffle=False, repeat=False),
    )
    rank1 = DraftFeatureDataLoader(
        TorchShardFeatureStore(tmp_path, read_only=True),
        DraftFeatureDataLoaderConfig(batch_size=8, rank=1, world_size=2, shuffle=False, repeat=False),
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
            DraftFeatureDataLoaderConfig(batch_size=1, rank=rank, world_size=2, shuffle=False, repeat=False),
        )
        rank_batches.append(list(loader))

    assert [len(batches) for batches in rank_batches] == [2, 2]


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
