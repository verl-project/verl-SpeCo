# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""The retained standalone workflows and explicit rejection of retired paths."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl_speco.standalone_tq_training_launcher import (
    build_pipeline_commands,
    resolve_pipeline_config,
)
from verl_speco.trainer.standalone_data import validate_standalone_training_source


def _args():
    return [
        "data.train_files=/data/train.jsonl",
        "actor_rollout_ref.model.path=/models/target",
        "actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK",
        "actor_rollout_ref.rollout.drafter.training.max_steps=1",
    ]


@pytest.mark.parametrize("source", ["torch_shard", "vllm_safetensors", "safetensors"])
def test_complete_offline_features_remain_supported(source):
    validate_standalone_training_source({"feature_store": {"type": source}})


@pytest.mark.parametrize("source", ["token_replay", "jsonl_token_replay", "jsonl"])
def test_token_only_training_is_rejected_with_generation_migration(source):
    with pytest.raises(ValueError, match="vllm_hidden_states_generate"):
        validate_standalone_training_source({"feature_store": {"type": source}})


@pytest.mark.parametrize("source", ["tq", "ray_reference"])
def test_streaming_training_requires_ray_scheduler(source):
    training = {"feature_store": {"type": source}}
    with pytest.raises(ValueError, match="Ray Producer/Consumer"):
        validate_standalone_training_source(training)
    validate_standalone_training_source(training, scheduled_commands=object())


def test_thread_pipeline_is_rejected():
    with pytest.raises(ValueError, match="target_feature_pipeline was removed"):
        validate_standalone_training_source(
            {"target_feature_pipeline": {"enabled": True}}
        )


def test_tq_loader_cannot_poll_without_scheduler():
    from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader

    with pytest.raises(ValueError, match="scheduler-assigned"):
        TQFeatureDataLoader(object(), batch_size=1, rank=0, world_size=1)


@pytest.mark.parametrize(
    "override",
    [
        "speco.draft_training.runtime_backend=subprocess",
        "speco.draft_training.data_transport=unknown",
    ],
)
def test_retired_or_unknown_launch_modes_fail_before_starting_services(override):
    args = [*_args(), override]
    with pytest.raises(ValueError):
        build_pipeline_commands(
            resolve_pipeline_config(args, environ={}),
            args,
            ray_address="unused",
            env={},
        )


def test_ray_tq_launch_starts_owner_and_runs_producer_consumer_as_actors(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from verl_speco import standalone_ray_runtime as runtime

    args = [*_args(), "speco.draft_training.data_transport=tq"]
    commands = build_pipeline_commands(
        resolve_pipeline_config(args, environ={}),
        args,
        ray_address="127.0.0.1:6379",
        env={},
    )
    prefix = "actor_rollout_ref.rollout.drafter.training"
    assert f"{prefix}.feature_store.type=tq" in commands.consumer_overrides
    assert f"{prefix}.feature_store.type=tq" in commands.producer_overrides
    assert f"{prefix}.transfer_queue.enable=true" in commands.consumer_overrides
    assert not hasattr(commands, "producer") and not hasattr(commands, "consumer")
    started, actor_runs, probes = [], [], []
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: None,
        wait=lambda **kwargs: None,
    )

    def popen(command, *, env):
        assert command == commands.owner
        assert env["RAY_ADDRESS"] == "127.0.0.1:6379"
        Path(env["SPECO_TQ_OWNER_READY_FILE"]).touch()
        started.append("owner")
        return process

    monkeypatch.setattr(runtime, "validate_tq_backend", lambda env: probes.append("tq"))
    monkeypatch.setattr(
        runtime,
        "run_ray_actors",
        lambda commands, **kwargs: actor_runs.append(commands) or 0,
    )
    assert (
        runtime.run_ray_pipeline(
            commands,
            ray_module=object(),
            ray_address="127.0.0.1:6379",
            environ={},
            popen=popen,
            endpoint_ready=lambda endpoint: True,
        )
        == 0
    )
    assert started == ["owner"] and actor_runs == [commands] and probes == ["tq"]


def test_offline_feature_generation_still_writes_a_trainable_store(
    tmp_path, monkeypatch
):
    from verl_speco import vllm_hidden_states_generate as generation
    from verl_speco.trainer.feature_store import (
        DraftFeatureSample,
        DraftReplaySample,
        TokenReplayFeatureStore,
        VllmSafetensorsFeatureStore,
    )

    input_path, output_path = tmp_path / "tokens", tmp_path / "features"
    sample = DraftReplaySample(
        input_ids=torch.arange(4),
        loss_mask=torch.ones(4),
        attention_mask=torch.ones(4),
        position_ids=torch.arange(4),
        feature_positions=torch.arange(4),
        draft_position_ids=torch.arange(1, 5),
    )
    store = TokenReplayFeatureStore(input_path)
    store.write_many([sample])
    store.close()
    closed = []

    class Generator:
        def __init__(self, config, **kwargs):
            assert (
                config.actor_rollout_ref.rollout.drafter.training.target_feature_replay.backend
                == "vllm_file"
            )

        def materialize(self, samples):
            return [
                DraftFeatureSample(
                    input_ids=s.input_ids,
                    loss_mask=s.loss_mask,
                    hidden_states=torch.ones(4, 8),
                    position_ids=s.draft_position_ids,
                )
                for s in samples
            ]

        def close(self):
            closed.append(True)

        def metrics(self):
            return {}

    monkeypatch.setattr(generation, "TargetFeatureReplayer", Generator)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "drafter": {
                        "training": {
                            "feature_store": {
                                "type": "token_replay",
                                "path": str(input_path),
                            },
                            "target_feature_replay": {
                                "offline_generation": {
                                    "input_path": str(input_path),
                                    "output_path": str(output_path),
                                }
                            },
                        }
                    }
                }
            }
        }
    )
    result = generation.generate_vllm_safetensors_features(config)
    assert result["written_samples"] == 1 and closed == [True]
    reader = VllmSafetensorsFeatureStore(output_path, read_only=True)
    loaded = reader.read(next(reader.iter_keys()))
    torch.testing.assert_close(loaded.hidden_states, torch.ones(4, 8))
    reader.close()
