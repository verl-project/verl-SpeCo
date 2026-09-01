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
from __future__ import annotations

import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from verl_speco.trainer.draft_training_loop import (  # noqa: E402
    _build_backend,
    _clear_tq_batch_across_ranks,
    _connect_tq_store_across_ranks,
    _contains_replay_samples,
    _is_out_of_memory_error,
    _next_batch_across_ranks,
    _rewrite_standalone_block_runtime_config,
    _save_standalone_checkpoint,
    _should_log_batch_progress,
)
from verl_speco.trainer.feature_store import DraftReplaySample  # noqa: E402


class _FakeTrainer:
    def __init__(self):
        self.checkpoint_dir = "/tmp/draft"
        self._pending_full_checkpoint_future = None
        self.future = Future()
        self.calls = 0

    def _save_checkpoint_async(self, step: int):
        self.calls += 1
        self.step = step
        self._pending_full_checkpoint_future = self.future
        return self.future


class _FakeTQLoader:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.clear_calls: list[list[str] | None] = []

    def clear_completed_batch(self, keys):
        self.clear_calls.append(keys)
        if self.error is not None:
            raise self.error


class _FakeTQStore:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1
        if self.error is not None:
            raise self.error


def test_tq_completed_batch_is_cleared_once_on_rank_zero() -> None:
    loader = _FakeTQLoader()
    _clear_tq_batch_across_ranks(
        loader,
        ["k0", "k1"],
        rank=0,
        device=torch.device("cpu"),
    )
    assert loader.clear_calls == [["k0", "k1"]]


def test_tq_clear_failure_is_reported_and_not_retried() -> None:
    loader = _FakeTQLoader(RuntimeError("clear failed"))
    with pytest.raises(RuntimeError, match="failed to clear"):
        _clear_tq_batch_across_ranks(
            loader,
            ["k0", "k1"],
            rank=0,
            device=torch.device("cpu"),
        )
    assert loader.clear_calls == [["k0", "k1"]]


def test_tq_store_connection_failure_is_reported() -> None:
    store = _FakeTQStore(RuntimeError("connect failed"))
    with pytest.raises(RuntimeError, match="failed to connect"):
        _connect_tq_store_across_ranks(
            store,
            rank=0,
            device=torch.device("cpu"),
        )
    assert store.connect_calls == 1


def _export_trainer(model_type: str, model_path=None):
    """Minimal trainer stand-in for the standalone checkpoint export helpers."""
    return SimpleNamespace(
        backend=SimpleNamespace(model_type=model_type),
        config=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=model_path))
        ),
    )


def _standalone_config(algorithm: str):
    return OmegaConf.create(
        {
            "model": {"path": "/does/not/exist"},
            "rollout": {
                "drafter": {"speculative_algorithm": algorithm, "training": {}}
            },
        }
    )


@pytest.mark.parametrize(
    ("attempted_batches", "expected"),
    [
        (1, True),
        (2, True),
        (3, True),
        (4, False),
        (99, False),
        (100, True),
        (101, False),
    ],
)
def test_should_log_standalone_batch_progress(attempted_batches, expected):
    assert _should_log_batch_progress(attempted_batches) is expected


def test_is_out_of_memory_error_matches_npu_oom_message():
    error = RuntimeError("NPU out of memory. Tried to allocate 258.00 MiB")

    assert _is_out_of_memory_error(error)
    assert not _is_out_of_memory_error(RuntimeError("bad batch"))


def test_contains_replay_samples_detects_draft_replay_sample():
    sample = DraftReplaySample(
        input_ids=torch.arange(4),
        loss_mask=torch.ones(4),
        attention_mask=torch.ones(4, dtype=torch.bool),
        position_ids=torch.arange(4),
        feature_positions=torch.arange(1, 3),
        draft_position_ids=torch.arange(2, 4),
    )

    assert _contains_replay_samples([sample])
    assert not _contains_replay_samples([{"input_ids": [1, 2]}])


@pytest.mark.parametrize(
    ("algorithm", "expected_backend", "expected_model_type"),
    [
        ("EAGLE3", "Eagle3TrainerBackend", "eagle3"),
        ("EAGLE1", "Eagle1TrainerBackend", "eagle3"),
        ("eagle2", "Eagle1TrainerBackend", "eagle3"),
        ("DFLASH", "DFlashTrainerBackend", "dflash"),
        ("DSPARK", "DSparkTrainerBackend", "dspark"),
        ("DOMINO", "DominoTrainerBackend", "domino"),
        ("PEAGLE", "PEagleTrainerBackend", "peagle"),
    ],
)
def test_standalone_backend_covers_every_online_algorithm(algorithm, expected_backend, expected_model_type):
    backend = _build_backend(_standalone_config(algorithm))

    assert type(backend).__name__ == expected_backend
    assert backend.model_type == expected_model_type


def test_standalone_backend_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported drafter algorithm"):
        _build_backend(_standalone_config("NOT_AN_ALGORITHM"))


def test_standalone_checkpoint_schedules_without_waiting():
    trainer = _FakeTrainer()

    result = _save_standalone_checkpoint(trainer, 5)

    assert result["saved"] is True
    assert result["reason"] == "scheduled"
    assert trainer.calls == 1
    assert trainer._pending_full_checkpoint_future is trainer.future


def test_standalone_checkpoint_waits_when_requested():
    trainer = _FakeTrainer()
    trainer.future.set_result(None)

    result = _save_standalone_checkpoint(trainer, 5, wait=True)

    assert result["saved"] is True
    assert result["reason"] == "saved"
    assert trainer._pending_full_checkpoint_future is None


def test_standalone_checkpoint_skips_when_previous_save_is_running():
    trainer = SimpleNamespace(
        checkpoint_dir="/tmp/draft", _pending_full_checkpoint_future=Future()
    )

    result = _save_standalone_checkpoint(trainer, 5)

    assert result["saved"] is False
    assert result["reason"] == "previous_save_running"


def test_public_checkpoint_path_rewrites_dspark_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepSeekDSparkModel"],
                "target_layer_ids": [1, 9, 17],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dspark",
                "architectures": ["DSparkDraftModel"],
                "target_layer_ids": [1, 9, 17],
                "markov_head_type": "vanilla",
            }
        ),
        encoding="utf-8",
    )

    class _PublicCheckpointTrainer:
        backend = SimpleNamespace(model_type="dspark")
        config = SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        )

        @staticmethod
        def save_checkpoint(step: int, wait: bool):
            assert step == 5
            assert wait is True
            return {"saved": True, "reason": "saved", "path": str(checkpoint_dir)}

    result = _save_standalone_checkpoint(_PublicCheckpointTrainer(), 5, wait=True)

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert result["saved"] is True
    assert runtime_config["model_type"] == "deepseek_v3"
    assert runtime_config["architectures"] == ["DeepSeekDSparkModel"]
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert (checkpoint_dir / "speco_training_config.json").exists()


def test_standalone_checkpoint_rewrites_runtime_config_after_save(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {"model_type": "deepseek_v3", "architectures": ["DeepSeekDSparkModel"]}
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    events = []

    class _CheckpointTrainer:
        backend = SimpleNamespace(model_type="dspark")
        config = SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        )

        @staticmethod
        def save_checkpoint(step: int, wait: bool):
            assert step == 5
            assert wait is True
            events.append("save")
            return {"saved": True, "reason": "saved", "path": str(checkpoint_dir)}

    result = _save_standalone_checkpoint(_CheckpointTrainer(), 5, wait=True)

    assert result["saved"] is True
    assert events == ["save"]


def test_standalone_dspark_checkpoint_preserves_source_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    source_config = {
        "model_type": "deepseek_v3",
        "architectures": ["DeepSeekDSparkModel"],
        "target_layer_ids": [1, 9, 17],
    }
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    training_config = {
        "model_type": "dspark",
        "architectures": ["DSparkDraftModel"],
        "target_layer_ids": [1, 9, 17],
        "mask_token_id": 151669,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "block_size": 7,
        "num_context_layers": 3,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(training_config), encoding="utf-8")
    trainer = _export_trainer("dspark", str(source_dir))

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "deepseek_v3"
    assert runtime_config["architectures"] == ["DeepSeekDSparkModel"]
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert runtime_config["dflash_config"]["target_layer_ids"] == [1, 9, 17]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18]
    assert saved_training_config == training_config


def test_standalone_dspark_checkpoint_rewrites_generic_qwen3_architecture(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["DSparkDraftModel"],
                "markov_head_type": "vanilla",
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}), encoding="utf-8"
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dspark",
                "architectures": ["DSparkDraftModel"],
                "markov_head_type": "vanilla",
            }
        ),
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(source_dir))
            ),
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["DSparkDraftModel"]
    assert runtime_config["speco_training_model_type"] == "dspark"


def test_standalone_domino_checkpoint_exports_dflash_projector_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_domino"
    source_dir.mkdir()
    source_config = {
        "model_type": "qwen3",
        "architectures": ["DominoDraftModel"],
    }
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    training_config = {
        "model_type": "domino",
        "architectures": ["DominoDraftModel"],
        "target_layer_ids": [2, 10, 18],
        "mask_token_id": 151669,
        "num_context_layers": 3,
        "block_size": 16,
        "num_anchors": 512,
        "projector_type": "domino",
        "emb_dim": 256,
        "gru_hidden_dim": 1024,
        "pure_draft_prefix_len": 1,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(training_config), encoding="utf-8")
    trainer = _export_trainer("domino", str(source_dir))

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    saved_training_config = json.loads((checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8"))
    dflash_config = runtime_config["dflash_config"]
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["speco_training_model_type"] == "domino"
    # Engines serve Domino through the DFlash method and switch on projector_type.
    assert dflash_config["projector_type"] == "domino"
    assert dflash_config["emb_dim"] == 256
    assert dflash_config["gru_hidden_dim"] == 1024
    assert dflash_config["pure_draft_prefix_len"] == 1
    assert dflash_config["block_size"] == 16
    assert dflash_config["target_layer_ids"] == [2, 10, 18]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [3, 11, 19]
    assert saved_training_config == training_config


def test_standalone_domino_checkpoint_defaults_projector_type(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "domino", "architectures": ["DominoDraftModel"]}),
        encoding="utf-8",
    )
    trainer = _export_trainer("domino", None)

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert runtime_config["dflash_config"]["projector_type"] == "domino"


def test_standalone_dflash_checkpoint_preserves_source_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dflash"
    source_dir.mkdir()
    source_config = {
        "model_type": "qwen3",
        "architectures": ["DFlashForCausalLM"],
    }
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    training_config = {
        "model_type": "dflash",
        "architectures": ["DFlashDraftModel"],
        "target_layer_ids": [2, 10, 18],
        "mask_token_id": 151669,
        "num_context_layers": 3,
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(training_config), encoding="utf-8")
    trainer = _export_trainer("dflash", str(source_dir))

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["DFlashForCausalLM"]
    assert runtime_config["dflash_config"]["target_layer_ids"] == [2, 10, 18]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [3, 11, 19]
    assert saved_training_config == training_config


def test_standalone_block_checkpoint_uses_target_model_type_without_source_config(
    tmp_path,
):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    missing_source_dir = tmp_path / "missing_source_dspark"
    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "head_dim": 128,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )
    training_config = {
        "model_type": "dspark",
        "architectures": ["DSparkDraftModel"],
        "target_layer_ids": [1, 9, 17],
        "markov_head_type": "vanilla",
        "head_dim": 80,
        "rope_theta": 10000.0,
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(missing_source_dir))
            ),
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "dspark"
    assert runtime_config["architectures"] == ["DSparkDraftModel"]
    assert runtime_config["speco_training_model_type"] == "dspark"
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert runtime_config["head_dim"] == 80
    assert runtime_config["rope_theta"] == 10000.0
    assert saved_training_config == training_config


def test_standalone_eagle3_checkpoint_exports_vllm_llama_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    missing_source_dir = tmp_path / "missing_source_eagle3"
    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 4096,
                "head_dim": 128,
                "rope_theta": 1000000,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )
    training_config = {
        "model_type": "qwen3",
        "architectures": ["LlamaForCausalLMEagle3"],
        "num_hidden_layers": 1,
        "hidden_size": 4096,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="eagle3"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(missing_source_dir))
            ),
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["LlamaForCausalLMEagle3"]
    assert runtime_config["num_hidden_layers"] == 1
    assert runtime_config["tie_word_embeddings"] is False
    assert not (checkpoint_dir / "speco_training_config.json").exists()


def test_next_batch_across_ranks_returns_local_batch_without_distributed():
    batch = [object()]

    assert _next_batch_across_ranks(
        iter([batch]), rank=0, device=torch.device("cpu")
    ) is batch


def test_next_batch_across_ranks_returns_none_when_source_is_exhausted():
    assert (
        _next_batch_across_ranks(
            iter(()), rank=0, device=torch.device("cpu")
        )
        is None
    )


def test_next_batch_across_ranks_preserves_local_producer_error():
    def broken_source():
        raise ValueError("producer failed")
        yield []

    with pytest.raises(RuntimeError, match="failed on rank=0") as exc_info:
        _next_batch_across_ranks(
            iter(broken_source()), rank=0, device=torch.device("cpu")
        )

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_next_batch_across_ranks_stops_for_remote_rank_failure(monkeypatch):
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.get_world_size", lambda: 2
    )

    def fake_all_reduce(state, op):
        del op
        state[0] = 1

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.all_reduce", fake_all_reduce
    )

    with pytest.raises(RuntimeError, match="failed on another rank"):
        _next_batch_across_ranks(
            iter([[object()]]), rank=1, device=torch.device("cpu")
        )
