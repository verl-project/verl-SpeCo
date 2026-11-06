# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""Saved raw hidden references reuse the online Consumer without deleting data."""

import asyncio
import copy
import json
from queue import Queue

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.producer.hidden_states_store import (
    RawHiddenDatasetWriter,
    RawHiddenSample,
    load_raw_hidden_manifest,
)
from verl_speco.standalone_tq_producer import run_producer
from verl_speco.standalone_tq_training_launcher import (
    _stable_path_identity,
    build_pipeline_commands,
    resolve_pipeline_config,
)
from verl_speco.trainer.feature_store import build_feature_store_from_config
from verl_speco.trainer.standalone_data import ReadyEntry, build_assignments
from verl_speco.trainer.standalone_resume import (
    build_input_fingerprint,
    save_standalone_resume,
)
from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples


@pytest.fixture
def dataset(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({"num_hidden_layers": 3}))
    root = tmp_path / "hidden"
    writer = RawHiddenDatasetWriter(root)
    contract = {
        "algorithm": "DSPARK",
        "target_layer_ids": [0],
        "vllm_aux_hidden_state_layer_ids": [1],
        "hidden_states_layout": "dflash_aux_plus_last",
        "dtype": "torch.bfloat16",
        "target_model_id": str(target),
        "target_model_revision": _stable_path_identity("target", str(target)),
        "tokenizer_fingerprint": _stable_path_identity("tokenizer", str(target)),
        "use_logits": False,
        "target_config_fingerprint": None,
        "source": "standalone_tq_producer",
        "require_full_alignment": True,
        "target_num_hidden_layers": 3,
    }
    request = {
        "input_ids": [10, 11, 12, 13],
        "loss_mask": [0.0, 1.0, 1.0, 1.0],
        "position_ids": [0, 1, 2, 3],
        "feature_positions": [0, 1, 2, 3],
        "draft_position_ids": [1, 2, 3, 4],
        "vllm_prompt_token_ids": [10, 11, 12, 13],
        "source_metadata": {},
    }
    for index in range(3):
        writer.write_many(
            [
                RawHiddenSample(
                    {
                        "token_ids": torch.tensor([10, 11, 12, 13]),
                        "hidden_states": (
                            torch.arange(32).reshape(4, 2, 4) + 100 * index
                        ).float(),
                    },
                    f"row-{index}",
                    index,
                    copy.deepcopy(request),
                    copy.deepcopy(contract),
                )
            ]
        )
    return root, target


def pipeline(dataset, extra=()):
    root, target = dataset
    args = [
        f"data.train_files={root}",
        f"actor_rollout_ref.model.path={target}",
        "speco.draft_training.data_source=saved_hidden",
        "actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK",
        "actor_rollout_ref.rollout.drafter.training.max_steps=2",
        "actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=2",
        *extra,
    ]
    # Invalid ambient services must be irrelevant to saved files.
    env = {
        "SPECO_VLLM_ENDPOINTS": "bad endpoint",
        "SPECO_VLLM_HIDDEN_STATES_STORE": "mooncake",
        "SPECO_TQ_STORAGE_BACKEND": "MooncakeStore",
    }
    config = resolve_pipeline_config(args, environ=env)
    commands = build_pipeline_commands(config, args, ray_address="local:6379", env=env)
    producer = OmegaConf.from_dotlist(
        [item.lstrip("+") for item in commands.producer_overrides]
    )
    consumer = OmegaConf.from_dotlist(
        [item.lstrip("+") for item in commands.consumer_overrides]
    )
    return commands, producer, consumer


async def collect(config, batch_size=2, before=None):
    entries = []

    async def publish(items):
        entries.extend(items)

    stats = await run_producer(
        config,
        on_references=publish,
        reference_batch_size=batch_size,
        before_request=before,
    )
    return stats, entries


def test_directory_launch_uses_manifest_layers_and_no_online_services(
    dataset, monkeypatch
):
    from verl_speco import standalone_ray_runtime as runtime
    from verl_speco import standalone_tq_training_launcher as launcher

    reads = []
    load_manifest = launcher.load_raw_hidden_manifest

    def count_read(path):
        reads.append(path)
        return load_manifest(path)

    monkeypatch.setattr(launcher, "load_raw_hidden_manifest", count_read)

    commands, producer, consumer = pipeline(dataset)
    assert len(reads) == 1
    assert commands.data_source == "saved_hidden"
    assert (
        commands.vllm is None and commands.owner == [] and commands.vllm_endpoints == ()
    )
    assert producer.speco.draft_training.data_source == "saved_hidden"
    training = consumer.actor_rollout_ref.rollout.drafter.training
    assert training.feature_store.type == "ray_reference"
    assert training.dspark_target_layer_ids == [0]
    assert training.dspark_num_target_layers == 1
    assert training.target_feature_replay.hidden_states_store.backend == "file"
    monkeypatch.setattr(runtime, "run_ray_actors", lambda *args, **kwargs: 17)

    def unexpected(*args, **kwargs):
        raise AssertionError("Offline launch must not probe or spawn services")

    monkeypatch.setattr(runtime, "validate_tq_backend", unexpected)
    monkeypatch.setattr(runtime, "validate_hidden_states_store", unexpected)
    assert (
        runtime.run_ray_pipeline(
            commands,
            ray_module=object(),
            ray_address="local:6379",
            popen=unexpected,
            endpoint_ready=unexpected,
        )
        == 17
    )


def test_producer_repeats_unique_occurrences_and_consumer_preserves_files(
    dataset, monkeypatch
):
    import verl_speco.standalone_tq_producer as producer_module
    from verl_speco.producer import hidden_states_store

    _commands, producer, consumer = pipeline(dataset)

    def no_connector_lock(*args, **kwargs):
        raise AssertionError("Permanent offline files must not query connector locks")

    monkeypatch.setattr(hidden_states_store, "_wait_for_lock", no_connector_lock)

    def unexpected(*args, **kwargs):
        raise AssertionError("Offline Producer must not initialize online resources")

    monkeypatch.setattr(producer_module, "validate_producer_config", unexpected)
    monkeypatch.setattr(producer_module, "_load_tokenizer", unexpected)
    monkeypatch.setattr(producer_module, "VllmFeatureClientPool", unexpected)
    stats, wire = asyncio.run(collect(producer))
    assert stats.published_count == 4
    assert [item["tag"]["sample_id"] for item in wire] == [
        "row-0",
        "row-1",
        "row-2",
        "row-0",
    ]
    assert [item["tag"]["sequence_no"] for item in wire] == list(range(4))
    assert len({item["key"] for item in wire}) == 4
    training = consumer.actor_rollout_ref.rollout.drafter.training
    store = build_feature_store_from_config(
        training.feature_store,
        read_only=True,
        transfer_queue_cfg=training.transfer_queue,
        hidden_states_store_cfg=training.target_feature_replay.hidden_states_store,
        reference_lifetime="persistent",
    )
    assert store.config["reference_lifetime"] == "persistent"
    store.connect()
    entries = [ReadyEntry(item["key"], item["tag"]) for item in wire]
    rank_entries = build_assignments(entries, batch_size=2, world_size=2)
    first = store.prepare_batch(rank_entries[0], pin_memory=False)
    assert first["last_hidden_state_norm"] == "raw_target_residual"
    assert torch.equal(
        first["target_last_hidden_states"][0].float(),
        torch.arange(32).reshape(4, 2, 4)[:, -1].float(),
    )
    from verl_speco.trainer.reference_feature_store import ReferenceFeatureStore

    spawned_store = ReferenceFeatureStore(store.config)
    assert not spawned_store.release_hidden_states_after_use
    for reason in ("trained", "tail", "shutdown", "discarded"):
        assert cleanup_standalone_samples(store, entries, reason=reason) == []
    assert len(load_raw_hidden_manifest(dataset[0])) == 3
    again = store.prepare_batch([entries[-1]], pin_memory=False)
    assert torch.equal(first["hidden_states"][0], again["hidden_states"][0])
    store.close()


def test_pause_resume_and_actor_completion_reuse_existing_callbacks(dataset):
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor

    _, config, _ = pipeline(dataset)

    async def run():
        events = Queue()
        actor = StandaloneProducerActor(config, events=events, reference_batch_size=2)
        await actor.pause()
        await actor.start()
        await asyncio.sleep(0.03)
        assert events.empty()
        assert not actor._task.done()
        await actor.resume()
        result = await asyncio.wait_for(actor.wait(), timeout=5)
        assert result["published_count"] == 4
        kinds = []
        while not events.empty():
            kinds.append(events.get()["kind"])
        assert kinds.count("samples_ready") == 2
        assert "producer_done" in kinds

    asyncio.run(run())


def test_shuffled_resume_replays_only_unconsumed_occurrences(dataset, tmp_path):
    _, config, _ = pipeline(dataset)
    producer = config.speco.standalone_tq_producer
    producer.update(shuffle=True, shuffle_seed=7, max_samples=8)
    _, first = asyncio.run(collect(config))
    expected = [
        f"row-{i}"
        for epoch in range(3)
        for i in np.random.default_rng(7 + epoch).permutation(3)
    ][:8]
    assert [item["tag"]["sample_id"] for item in first] == expected
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(
        checkpoint,
        [0, 2, 4],
        optimizer_step=1,
        input_path=dataset[0],
        producer_shuffle={"shuffle": True, "shuffle_seed": 7},
    )
    producer.resume_checkpoint_path = str(checkpoint)
    producer.max_samples = 5
    _, resumed = asyncio.run(collect(config))
    assert [item["tag"]["sequence_no"] for item in resumed] == [1, 3, 5, 6, 7]
    assert [item["tag"]["sample_id"] for item in resumed] == [
        expected[i] for i in [1, 3, 5, 6, 7]
    ]
    producer.shuffle_seed = 8
    with pytest.raises(ValueError, match="shuffle"):
        asyncio.run(collect(config))


def test_directory_resume_detects_manifest_mutation(dataset, tmp_path):
    _, config, _ = pipeline(dataset)
    checkpoint = tmp_path / "checkpoint"
    save_standalone_resume(checkpoint, [0], optimizer_step=1, input_path=dataset[0])
    config.speco.standalone_tq_producer.resume_checkpoint_path = str(checkpoint)
    before = build_input_fingerprint(dataset[0])
    manifest = dataset[0] / "manifest.jsonl"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("row-0", "row-X"), encoding="utf-8")
    assert before != build_input_fingerprint(dataset[0])
    with pytest.raises(ValueError, match="input file changed"):
        asyncio.run(collect(config))


@pytest.mark.parametrize(
    "extra",
    [
        ["speco.draft_training.data_transport=tq"],
        ["actor_rollout_ref.rollout.drafter.training.use_logits=true"],
        ["actor_rollout_ref.rollout.drafter.training.gradient_accumulation_steps=2"],
        ["actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.0"],
        ["actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=2"],
        ["speco.standalone_tq_producer.vllm_aux_hidden_state_layer_ids=[2]"],
    ],
)
def test_incompatible_training_contract_fails_before_publication(dataset, extra):
    with pytest.raises(ValueError):
        pipeline(dataset, extra)


def test_raw_format_and_directory_are_required(dataset, tmp_path):
    root, target = dataset
    args = [
        f"data.train_files={root / 'manifest.jsonl'}",
        f"actor_rollout_ref.model.path={target}",
        "speco.draft_training.data_source=saved_hidden",
    ]
    with pytest.raises(ValueError, match="directory"):
        resolve_pipeline_config(args)
    (root / "metadata.json").write_text(
        json.dumps({"format": "vllm_safetensors", "schema_version": 1})
    )
    args[0] = f"data.train_files={root}"
    with pytest.raises(ValueError, match="normalized"):
        resolve_pipeline_config(args)


@pytest.mark.parametrize(
    "algorithm,layout,target_ids,extra",
    [
        (
            "DFLASH",
            "dflash_aux",
            [0],
            ["actor_rollout_ref.rollout.drafter.training.dflash_block_size=2"],
        ),
        ("EAGLE3", "eagle3_aux_plus_last", [1], []),
    ],
)
def test_other_supported_algorithms_publish_existing_reference_format(
    dataset, algorithm, layout, target_ids, extra
):
    root, _ = dataset
    manifest = root / "manifest.jsonl"
    entries = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    for entry in entries:
        entry["feature_contract"].update(
            algorithm=algorithm,
            hidden_states_layout=layout,
            target_layer_ids=target_ids,
        )
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    _, producer, _ = pipeline(
        dataset,
        [
            f"actor_rollout_ref.rollout.drafter.speculative_algorithm={algorithm}",
            *extra,
        ],
    )
    stats, entries = asyncio.run(collect(producer))
    assert stats.published_count == 4
    assert all(entry["tag"]["payload_kind"] == "vllm_reference" for entry in entries)


def test_training_block_size_cannot_silently_drop_saved_supervision(dataset):
    _, producer, _ = pipeline(dataset)
    # Keep a valid contract but shorten the saved supervision to a single token.
    root, _ = dataset
    manifest = root / "manifest.jsonl"
    entries = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    for entry in entries:
        entry["request"]["loss_mask"] = [0.0, 0.0, 0.0, 1.0]
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sufficient"):
        asyncio.run(collect(producer))


@pytest.mark.parametrize("raw_capture", [False, True])
def test_saved_filter_preserves_original_occurrence_numbers(dataset, raw_capture):
    root, _ = dataset
    manifest = root / "manifest.jsonl"
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]
    if raw_capture:
        for entry in entries:
            stored = entry["feature_contract"]
            entry["feature_contract"] = {
                name: stored[name] for name in (
                    "target_model_id", "target_model_revision", "tokenizer_fingerprint",
                    "target_num_hidden_layers", "vllm_aux_hidden_state_layer_ids",
                )
            }
            entry["feature_contract"].update(contract_type="raw_capture", include_final_hidden=True)
    entries[1]["request"]["loss_mask"] = [0.0, 0.0, 0.0, 1.0]
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    _, producer, _ = pipeline(dataset)
    stats, references = asyncio.run(collect(producer))
    assert stats.published_count == 4
    assert [entry["tag"]["sequence_no"] for entry in references] == [0, 2, 3, 5]


def test_saved_identity_uses_model_files_and_keeps_legacy_identity(tmp_path):
    from verl_speco.standalone_tq_training_launcher import _saved_hidden_identity

    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        path.mkdir()
        (path / "config.json").write_text('{"num_hidden_layers": 3}')
        (path / "tokenizer.json").write_text('{"vocab": {"hello": 1}}')
    for kind in ("target", "tokenizer"):
        assert _saved_hidden_identity(kind, str(first)) == _saved_hidden_identity(kind, str(second))
        legacy = _stable_path_identity(kind, str(first))
        assert _saved_hidden_identity(kind, str(first), legacy) == legacy
    (second / "tokenizer.json").write_text('{"vocab": {"hello": 2}}')
    assert _saved_hidden_identity("tokenizer", str(first)) != _saved_hidden_identity("tokenizer", str(second))


@pytest.mark.parametrize(
    "algorithm,extra,expected_layout,expected_dtype",
    [
        ("DSPARK", [], "dflash_aux_plus_last", "torch.bfloat16"),
        (
            "DSPARK",
            ["actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0"],
            "dflash_aux",
            "torch.bfloat16",
        ),
        (
            "DSPARK",
            ["speco.standalone_tq_producer.hidden_dtype=float32"],
            "dflash_aux_plus_last",
            "torch.float32",
        ),
        (
            "DFLASH",
            ["actor_rollout_ref.rollout.drafter.training.dflash_block_size=2"],
            "dflash_aux",
            "torch.bfloat16",
        ),
        ("EAGLE3", [], "eagle3_aux_plus_last", "torch.bfloat16"),
    ],
)
def test_capture_dataset_binds_algorithm_layout_and_dtype_only_at_training(
    dataset, algorithm, extra, expected_layout, expected_dtype
):
    root, _ = dataset
    manifest = root / "manifest.jsonl"
    entries = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    for entry in entries:
        previous = entry["feature_contract"]
        entry["feature_contract"] = {
            key: previous[key]
            for key in (
                "target_model_id",
                "target_model_revision",
                "tokenizer_fingerprint",
                "target_num_hidden_layers",
                "vllm_aux_hidden_state_layer_ids",
            )
        }
        entry["feature_contract"].update(
            contract_type="raw_capture",
            include_final_hidden=True,
            raw_hidden_dtype="torch.float32",
            source="standalone_data_generate",
        )
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    _, producer, _ = pipeline(
        dataset,
        [
            f"actor_rollout_ref.rollout.drafter.speculative_algorithm={algorithm}",
            *extra,
        ],
    )
    stats, references = asyncio.run(collect(producer))
    assert stats.published_count == 4
    bound = references[0]["tag"]["feature_contract"]
    assert bound["algorithm"] == algorithm
    assert bound["hidden_states_layout"] == expected_layout
    assert bound["dtype"] == expected_dtype
    assert len(load_raw_hidden_manifest(root)) == 3


def test_capture_window_is_selected_by_training_and_leaves_full_file_untouched(dataset):
    root, _ = dataset
    manifest = root / "manifest.jsonl"
    entries = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    for entry in entries:
        previous = entry["feature_contract"]
        entry["feature_contract"] = {
            key: previous[key]
            for key in (
                "target_model_id",
                "target_model_revision",
                "tokenizer_fingerprint",
                "target_num_hidden_layers",
                "vllm_aux_hidden_state_layer_ids",
            )
        }
        entry["feature_contract"].update(
            contract_type="raw_capture", include_final_hidden=True
        )
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    _, producer, _ = pipeline(
        dataset, ["speco.standalone_tq_producer.max_feature_length=2"]
    )
    _, references = asyncio.run(collect(producer))
    assert references[0]["tag"]["request"]["feature_positions"] == [0, 1, 2]
    assert load_raw_hidden_manifest(root)[0]["request"]["feature_positions"] == [
        0,
        1,
        2,
        3,
    ]


def test_capture_without_final_layer_fails_when_training_needs_it():
    from verl_speco.producer.hidden_states_store import bind_raw_hidden_contract

    captured = {
        "contract_type": "raw_capture",
        "target_model_id": "m",
        "target_model_revision": "r",
        "tokenizer_fingerprint": "t",
        "vllm_aux_hidden_state_layer_ids": [1],
        "target_num_hidden_layers": 3,
        "include_final_hidden": False,
    }
    expected = {
        key: captured[key]
        for key in (
            "target_model_id",
            "target_model_revision",
            "tokenizer_fingerprint",
            "vllm_aux_hidden_state_layer_ids",
        )
    }
    expected["hidden_states_layout"] = "dflash_aux_plus_last"
    with pytest.raises(ValueError, match="final hidden"):
        bind_raw_hidden_contract(captured, expected)
