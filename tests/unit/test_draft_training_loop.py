from __future__ import annotations

import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.draft_training_loop import (  # noqa: E402
    _rewrite_standalone_block_runtime_config,
    _save_standalone_checkpoint,
    _torch_load_cpu,
)


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
    trainer = SimpleNamespace(checkpoint_dir="/tmp/draft", _pending_full_checkpoint_future=Future())

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

    runtime_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert result["saved"] is True
    assert runtime_config["model_type"] == "deepseek_v3"
    assert runtime_config["architectures"] == ["DeepSeekDSparkModel"]
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert (checkpoint_dir / "speco_training_config.json").exists()


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
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    saved_training_config = json.loads((checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8"))
    assert runtime_config["model_type"] == "deepseek_v3"
    assert runtime_config["architectures"] == ["DeepSeekDSparkModel"]
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert runtime_config["dflash_config"]["target_layer_ids"] == [1, 9, 17]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18]
    assert saved_training_config == training_config


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
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dflash"),
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    saved_training_config = json.loads((checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8"))
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["DFlashForCausalLM"]
    assert runtime_config["dflash_config"]["target_layer_ids"] == [2, 10, 18]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [3, 11, 19]
    assert saved_training_config == training_config


def test_standalone_block_checkpoint_appends_source_lm_head_weight(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["DSparkForCausalLM"]}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    lm_head = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    safetensors_torch.save_file({"lm_head.weight": lm_head}, str(source_dir / "model.safetensors"))
    safetensors_torch.save_file({"fc.weight": torch.ones(2, 2)}, str(checkpoint_dir / "model.safetensors"))
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    exported_state = safetensors_torch.load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    assert torch.equal(exported_state["lm_head.weight"], lm_head)
    assert torch.equal(exported_state["fc.weight"], torch.ones(2, 2))


def test_standalone_block_checkpoint_appends_lm_head_to_sharded_safetensors_index(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["DSparkForCausalLM"]}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    lm_head = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    fc_weight = torch.ones(2, 2)
    safetensors_torch.save_file({"lm_head.weight": lm_head}, str(source_dir / "model.safetensors"))
    safetensors_torch.save_file({"fc.weight": fc_weight}, str(checkpoint_dir / "model-00001-of-00001.safetensors"))
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": fc_weight.numel() * fc_weight.element_size()},
                "weight_map": {"fc.weight": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    index_data = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index_data["weight_map"]["lm_head.weight"] == "model-lm-head.safetensors"
    assert index_data["metadata"]["total_size"] == (
        fc_weight.numel() * fc_weight.element_size() + lm_head.numel() * lm_head.element_size()
    )
    added_state = safetensors_torch.load_file(str(checkpoint_dir / "model-lm-head.safetensors"), device="cpu")
    assert torch.equal(added_state["lm_head.weight"], lm_head)


def test_torch_load_cpu_falls_back_without_weights_only(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "pytorch_model.bin"
    expected = {"lm_head.weight": torch.ones(2, 2)}
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("weights_only is unsupported")
        assert path == str(checkpoint_path)
        return expected

    monkeypatch.setattr(torch, "load", fake_load)

    assert _torch_load_cpu(str(checkpoint_path)) is expected
    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ]
