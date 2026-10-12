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
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from verl_speco.trainer.base_trainer import DrafterBaseTrainer  # noqa: E402
from verl_speco.trainer.draft_training_loop import (  # noqa: E402
    _assert_standalone_layer_migration,
    _build_backend,
    _build_standalone_tracking,
    _clear_tq_batch_across_ranks,
    _connect_tq_store_across_ranks,
    _contains_replay_samples,
    _finalize_standalone_checkpoint,
    _finish_standalone_tracking,
    _is_out_of_memory_error,
    _log_standalone_tracking_metrics,
    _next_batch_across_ranks,
    _raise_standalone_export_error,
    _rewrite_standalone_block_runtime_config,
    _save_standalone_checkpoint,
    _select_standalone_tracking_metrics,
    _should_log_batch_progress,
    _sync_standalone_export_error,
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


class _TransientClearTQLoader(_FakeTQLoader):
    def __init__(self, failures: int):
        super().__init__()
        self.failures_remaining = failures

    def clear_completed_batch(self, keys):
        self.clear_calls.append(keys)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("clear failed")


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


def test_tq_clear_retries_transient_failure(monkeypatch) -> None:
    loader = _TransientClearTQLoader(failures=2)
    delays = []
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.time.sleep", delays.append
    )

    _clear_tq_batch_across_ranks(
        loader,
        ["k0", "k1"],
        rank=0,
        device=torch.device("cpu"),
    )

    assert loader.clear_calls == [["k0", "k1"]] * 3
    assert delays == [0.5, 1.0]


def test_tq_clear_failure_is_reported_after_three_attempts(monkeypatch) -> None:
    loader = _FakeTQLoader(RuntimeError("clear failed"))
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.time.sleep", lambda _: None
    )
    with pytest.raises(RuntimeError, match="failed to clear"):
        _clear_tq_batch_across_ranks(
            loader,
            ["k0", "k1"],
            rank=0,
            device=torch.device("cpu"),
        )
    assert loader.clear_calls == [["k0", "k1"]] * 3


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


def test_standalone_dspark_tracking_keeps_only_core_metrics():
    metrics = {
        "dspark/loss": 0.7, "dspark/ce_loss": 2.5, "dspark/l1_loss": 1.0,
        "dspark/top1_acc": 0.6, "dspark/mean_acceptance_length": 2.0,
        "train/lr": 1e-6, "perf/step_time": 10.0,
        "perf/consumer_wait_time": 3.0, "perf/train_time": 5.0,
        "train/ploss_0": 2.5, "dspark/valid_token_count": 42.0,
        "perf/tq_get_time": 1.0,
    }
    selected = _select_standalone_tracking_metrics(metrics)
    assert len(selected) == 10
    assert selected["dspark/loss"] == 0.7
    assert selected["perf/other_time"] == 2.0
    assert "train/ploss_0" not in selected
    assert "perf/tq_get_time" not in selected
    assert "perf/other_time" not in metrics


def test_standalone_dspark_tracking_skips_missing_and_nonfinite_metrics():
    selected = _select_standalone_tracking_metrics({
        "dspark/ce_loss": float("nan"), "dspark/loss": 1.0,
        "perf/train_time": float("inf"),
    })
    assert selected == {"dspark/loss": 1.0}
    assert _select_standalone_tracking_metrics({"dflash/accuracy": 0.5}) == {}


@pytest.mark.parametrize("prefix", ["eagle3", "dflash", "dspark", "dflash2"])
def test_standalone_global_loss_and_selector_aggregation(prefix):
    trainer = SimpleNamespace(
        _training_metric_sums={
            f"{prefix}/loss": 6.0,
            f"{prefix}/selector_loss_sum": 12.0,
            f"{prefix}/selector_weight_count": 30.0,
            f"{prefix}/selector_correct_count": 9.0,
            f"{prefix}/selector_token_count": 15.0,
        },
        _training_metric_steps=2, optimizer_steps_total=10, optimizer=None,
        _block_drafter_metric_prefix=lambda: prefix,
        _block_drafter_config_value=lambda name, default: 0,
    )
    metrics = DrafterBaseTrainer.get_training_metrics(trainer)
    assert metrics[f"{prefix}/loss"] == 3.0
    assert metrics[f"{prefix}/selector_loss"] == 0.4
    assert metrics[f"{prefix}/selector_accuracy"] == 0.6


@pytest.mark.parametrize("prefix", ["eagle3", "dflash", "dspark", "dflash2"])
def test_standalone_core_metrics_for_each_backend(prefix):
    metrics = {
        f"{prefix}/loss": 1.0, f"{prefix}/top1_acc": 0.5,
        f"{prefix}/mean_acceptance_length": 2.0,
        "train/simulated_acc_len": 1.8, "train/lr": 1e-6,
        "perf/step_time": 10.0, "perf/consumer_wait_time": 3.0,
        "perf/train_time": 5.0, "train/ploss_0": 1.0,
        f"{prefix}/count_per_position/0": 100.0,
        "dflash2/selector_loss": 0.4, "dflash2/selector_accuracy": 0.6,
    }
    selected = _select_standalone_tracking_metrics(metrics)
    assert selected[f"{prefix}/loss"] == 1.0
    assert selected["perf/other_time"] == 2.0
    assert "train/ploss_0" not in selected
    assert ("train/simulated_acc_len" in selected) == (prefix == "eagle3")
    assert ("dflash2/selector_loss" in selected) == (prefix == "dflash2")


def test_standalone_tracking_enables_requested_backends_on_rank_zero(monkeypatch):
    import verl.utils.tracking as tracking_module

    created = []

    class _FakeTracking:
        supported_backend = ("wandb", "tensorboard", "console", "swanlab")

        def __init__(self, project_name, experiment_name, default_backend, config=None):
            created.append((project_name, experiment_name, list(default_backend)))

    monkeypatch.setattr(tracking_module, "Tracking", _FakeTracking)
    config = OmegaConf.create(
        {
            "trainer": {
                "logger": ["console", "tensorboard", "wandb", "bogus", "tensorboard"],
                "project_name": "proj",
                "experiment_name": "exp",
            }
        }
    )

    tracker = _build_standalone_tracking(config, rank=0)
    assert len(tracker) == 2
    assert all(isinstance(item, _FakeTracking) for item in tracker)
    # one tracker per supported backend; console/unsupported filtered, duplicates dropped.
    assert created == [("proj", "exp", ["tensorboard"]), ("proj", "exp", ["wandb"])]
    assert _build_standalone_tracking(config, rank=1) == []
    config.trainer.logger = ["console", "swanlab"]
    assert len(_build_standalone_tracking(config, rank=0)) == 1
    assert created[-1] == ("proj", "exp", ["swanlab"])

    console_only = OmegaConf.create({"trainer": {"logger": ["console"]}})
    assert _build_standalone_tracking(console_only, rank=0) == []
    unsupported_only = OmegaConf.create({"trainer": {"logger": ["bogus"]}})
    assert _build_standalone_tracking(unsupported_only, rank=0) == []


def test_standalone_tracking_isolates_a_failing_backend(monkeypatch):
    import verl.utils.tracking as tracking_module

    class _FlakyTracking:
        supported_backend = ("wandb", "tensorboard", "console")

        def __init__(self, project_name, experiment_name, default_backend, config=None):
            if list(default_backend) == ["wandb"]:
                raise RuntimeError("wandb unavailable")

    monkeypatch.setattr(tracking_module, "Tracking", _FlakyTracking)
    config = OmegaConf.create({"trainer": {"logger": ["tensorboard", "wandb"]}})

    trackers = _build_standalone_tracking(config, rank=0)
    # tensorboard still initializes even though wandb raised.
    assert len(trackers) == 1


def test_finish_standalone_tracking_swallows_backend_errors():
    class _Boom:
        def finish(self):
            raise RuntimeError("boom")

    class _Ok:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    ok = _Ok()
    _finish_standalone_tracking([ok, _Boom()], rank=0)
    assert ok.finished


def test_log_standalone_tracking_metrics_swallows_backend_errors():
    class _Boom:
        def log(self, **kwargs):
            raise RuntimeError("boom")

    _log_standalone_tracking_metrics([_Boom()], {"loss": 1.0}, step=1)


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


def test_standalone_base_writer_defers_completion_marker(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "draft_step_5"

    class _ExportModel:
        @staticmethod
        def save_pretrained(path, *, state_dict, **kwargs):
            del state_dict, kwargs
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "pytorch_model.bin").write_bytes(b"weights")

    trainer = SimpleNamespace(
        _pending_full_checkpoint_future=None,
        _full_checkpoint_executor=None,
        rank=0,
        config=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=None))
        ),
        optimizer_steps_total=5,
        training_steps=5,
        lr_scheduler=None,
        optimizer=None,
        _is_checkpoint_leader=lambda: True,
        _get_pretrained_export_model=lambda: (_ExportModel(), None),
        _get_pretrained_export_state_dict=lambda: {"weight": torch.ones(1)},
        _infer_pretrained_save_kwargs=lambda: {},
        _clear_existing_pretrained_weight_files=lambda path: None,
        _copy_drafter_auxiliary_files=lambda path: None,
        _atomic_json_dump=DrafterBaseTrainer._atomic_json_dump,
    )
    monkeypatch.setattr(
        "verl_speco.trainer.base_trainer.release_checkpoint_host_memory",
        lambda *args, **kwargs: {
            "elapsed_sec": 0.0,
            "files_advised": 0,
            "files_failed": 0,
        },
    )

    future = DrafterBaseTrainer._save_pretrained_checkpoint_async(
        trainer,
        str(checkpoint_dir),
        5,
        {"format": "torch_distributed_checkpoint"},
        defer_completion=True,
    )
    future.result()
    trainer._full_checkpoint_executor.shutdown(wait=True)

    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["complete"] is False


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
        def save_checkpoint(step: int, wait: bool, *, defer_completion: bool):
            assert step == 5
            assert wait is True
            assert defer_completion is True
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
        def save_checkpoint(step: int, wait: bool, *, defer_completion: bool):
            assert step == 5
            assert wait is True
            assert defer_completion is True
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
        "target_layer_ids": [0, 8, 16],
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
    assert runtime_config["target_layer_ids"] == [0, 8, 16]
    assert runtime_config["dflash_config"]["target_layer_ids"] == [0, 8, 16]
    assert runtime_config["dspark_config"]["target_layer_ids"] == [0, 8, 16]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [1, 9, 17]
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
        "target_layer_ids": [1, 9, 17],
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
    assert runtime_config["target_layer_ids"] == [1, 9, 17]
    assert dflash_config["target_layer_ids"] == [1, 9, 17]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18]
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
        "target_layer_ids": [1, 9, 17],
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
    assert runtime_config["target_layer_ids"] == [1, 9, 17]
    assert runtime_config["dflash_config"]["target_layer_ids"] == [1, 9, 17]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18]
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
        "target_layer_ids": [0, 8, 16],
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


def test_standalone_dflash_checkpoint_rejects_negative_decoder_layer_id(tmp_path):
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
        "target_layer_ids": [-1, 9, 17],
        "mask_token_id": 151669,
        "num_context_layers": 3,
    }
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(training_config), encoding="utf-8")
    trainer = _export_trainer("dflash", str(source_dir))

    with pytest.raises(ValueError, match="decoder layer id must be non-negative"):
        _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    # Validate before writing either the runtime config or a training-config copy.
    assert json.loads(config_path.read_text(encoding="utf-8")) == training_config
    assert not (checkpoint_dir / "speco_training_config.json").exists()


def _migration_trainer(model_type: str, model_path, target_layer_ids=None):
    """Trainer stand-in carrying the launcher-provided decoder layer IDs."""
    training = (
        {}
        if target_layer_ids is None
        else {f"{model_type}_target_layer_ids": target_layer_ids}
    )
    return SimpleNamespace(
        backend=SimpleNamespace(model_type=model_type),
        config=SimpleNamespace(
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=model_path, training=training)
            )
        ),
    )


def _write_drafter_config(directory, config) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def test_standalone_layer_migration_accepts_consistent_source_config(tmp_path) -> None:
    matching = _write_drafter_config(
        tmp_path / "matching", {"aux_hidden_state_layer_ids": [2, 10, 18]}
    )
    # Launcher decoder IDs [1, 9, 17] imply vLLM output IDs [2, 10, 18].
    _assert_standalone_layer_migration(
        _migration_trainer("dspark", matching, [1, 9, 17]), "dspark"
    )

    # A source config that declares no layer IDs cannot conflict either.
    silent = _write_drafter_config(tmp_path / "silent", {"model_type": "dspark"})
    _assert_standalone_layer_migration(
        _migration_trainer("dspark", silent, [1, 9, 17]), "dspark"
    )

    # No launcher IDs and no source config are both no-ops.
    _assert_standalone_layer_migration(_migration_trainer("dspark", matching), "dspark")
    _assert_standalone_layer_migration(
        _migration_trainer("dspark", None, [1, 9, 17]), "dspark"
    )


def test_standalone_layer_migration_rejects_mismatched_vllm_ids(tmp_path) -> None:
    source = _write_drafter_config(
        tmp_path / "flipped", {"eagle_aux_hidden_state_layer_ids": [1, 9, 17]}
    )

    with pytest.raises(ValueError, match="migration mismatch"):
        _assert_standalone_layer_migration(
            _migration_trainer("dspark", source, [1, 9, 17]), "dspark"
        )


def test_standalone_layer_migration_rejects_mismatched_decoder_ids(tmp_path) -> None:
    source = _write_drafter_config(
        tmp_path / "decoder", {"dflash_config": {"target_layer_ids": [0, 8, 16]}}
    )

    with pytest.raises(ValueError, match="Cannot safely migrate"):
        _assert_standalone_layer_migration(
            _migration_trainer("dspark", source, [1, 9, 17]), "dspark"
        )


def test_standalone_checkpoint_export_error_surfaces_in_main_thread(
    monkeypatch, tmp_path
) -> None:
    trainer = _migration_trainer("dspark", None)
    completed = Future()
    completed.set_result({"saved": True, "path": str(tmp_path)})

    def fail_export(*args, **kwargs):
        raise ValueError("layer-ID migration mismatch")

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop._rewrite_standalone_block_runtime_config",
        fail_export,
    )

    # The writer-thread callback must not raise; it records the failure instead.
    _finalize_standalone_checkpoint(
        trainer, str(tmp_path / "draft_step_5"), completed, step=5
    )

    assert isinstance(trainer._standalone_export_error, ValueError)
    with pytest.raises(RuntimeError, match="checkpoint export failed") as exc_info:
        _raise_standalone_export_error(trainer)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_standalone_checkpoint_writer_error_surfaces_in_main_thread(
    monkeypatch, tmp_path
) -> None:
    trainer = _migration_trainer("dspark", None)
    completed = Future()
    writer_error = OSError("disk full")
    completed.set_exception(writer_error)
    rewrite_called = False

    def unexpected_rewrite(*args, **kwargs):
        nonlocal rewrite_called
        rewrite_called = True

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop._rewrite_standalone_block_runtime_config",
        unexpected_rewrite,
    )

    _finalize_standalone_checkpoint(
        trainer, str(tmp_path / "draft_step_5"), completed, step=5
    )

    assert rewrite_called is False
    assert trainer._standalone_export_error is writer_error
    with pytest.raises(RuntimeError, match="checkpoint export failed") as exc_info:
        _raise_standalone_export_error(trainer)
    assert exc_info.value.__cause__ is writer_error


def test_standalone_checkpoint_finalize_controls_completion_marker(
    monkeypatch, tmp_path
) -> None:
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    metadata_path = checkpoint_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"step": 5, "complete": True}), encoding="utf-8"
    )
    completed = Future()
    completed.set_result(None)
    states = []

    def observe_incomplete(*args, **kwargs):
        del args, kwargs
        states.append(json.loads(metadata_path.read_text(encoding="utf-8"))["complete"])

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop._rewrite_standalone_block_runtime_config",
        observe_incomplete,
    )

    _finalize_standalone_checkpoint(
        SimpleNamespace(), str(checkpoint_dir), completed, step=5
    )

    assert states == [False]
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["complete"] is True


def test_standalone_checkpoint_finalize_failure_stays_incomplete(
    monkeypatch, tmp_path
) -> None:
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    metadata_path = checkpoint_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"step": 5, "complete": True}), encoding="utf-8"
    )
    trainer = SimpleNamespace()
    completed = Future()
    completed.set_result(None)

    def fail_rewrite(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop._rewrite_standalone_block_runtime_config",
        fail_rewrite,
    )

    _finalize_standalone_checkpoint(trainer, str(checkpoint_dir), completed, step=5)

    assert json.loads(metadata_path.read_text(encoding="utf-8"))["complete"] is False
    assert isinstance(trainer._standalone_export_error, OSError)


def test_standalone_runtime_config_write_error_propagates(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark"}), encoding="utf-8"
    )
    trainer = _migration_trainer("dspark", None)

    def fail_atomic_write(payload, output_path):
        del payload, output_path
        raise OSError("disk full")

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.DrafterBaseTrainer._atomic_json_dump",
        fail_atomic_write,
    )

    with pytest.raises(OSError, match="disk full"):
        _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))


def test_standalone_runtime_config_missing_after_writer_is_an_error(tmp_path) -> None:
    trainer = _migration_trainer("dspark", None)

    with pytest.raises(FileNotFoundError, match="missing"):
        _rewrite_standalone_block_runtime_config(trainer, str(tmp_path))


def test_standalone_checkpoint_export_error_stops_remote_ranks(monkeypatch) -> None:
    trainer = SimpleNamespace()
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.get_world_size", lambda: 2
    )

    def fake_all_reduce(failed, op):
        del op
        failed[0] = 1

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.all_reduce", fake_all_reduce
    )

    with pytest.raises(RuntimeError, match="failed on another rank"):
        _sync_standalone_export_error(trainer, torch.device("cpu"))


def test_standalone_checkpoint_export_error_can_be_deferred_until_teardown(
    monkeypatch,
) -> None:
    trainer = SimpleNamespace()
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.get_world_size", lambda: 2
    )

    def fake_all_reduce(failed, op):
        del op
        failed[0] = 1

    monkeypatch.setattr(
        "verl_speco.trainer.draft_training_loop.dist.all_reduce", fake_all_reduce
    )

    assert _sync_standalone_export_error(
        trainer, torch.device("cpu"), raise_on_error=False
    )
