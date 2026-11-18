# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for DP ownership, real feature contracts and automatic phase gating."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_speco import probe_deepseek_v4_training_cycle as cycle
from verl_speco.integration.deepseek_v4_server_probe import probe_drafter_config
from verl_speco.trainer.deepseek_v4_contract import DeepSeekV4Contract


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 2,
                "hc_mult": 2,
                "vocab_size": 64,
                "num_hidden_layers": 3,
                "dspark_target_layer_ids": [0, 1],
                "dspark_block_size": 2,
            }
        )
    )
    return path


def options(model, output):
    return {"model": str(model), "feature_store": str(output), "max_model_len": 64}


def test_capture_is_opt_in_and_does_not_enable_resident_training(model, tmp_path):
    assert not probe_drafter_config({})["training"][
        "collect_hidden_states_from_rollout"
    ]
    with pytest.raises(ValueError, match="removed"):
        probe_drafter_config(options(model, tmp_path))


def test_prompt_excludes_recorded_answer_and_rejects_invalid_mask(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    path.write_text(json.dumps({"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1]}))
    assert cycle.prompt_from_trajectory(
        path, vocab_size=8, max_model_len=32, response_tokens=16
    ) == [1, 2]
    with pytest.raises(ValueError, match="budget"):
        cycle.prompt_from_trajectory(
            path, vocab_size=8, max_model_len=16, response_tokens=16
        )
    path.write_text(json.dumps({"input_ids": [1, 2], "loss_mask": [1, -100]}))
    with pytest.raises(ValueError, match="SpeCo loss_mask"):
        cycle.prompt_from_trajectory(
            path, vocab_size=8, max_model_len=32, response_tokens=16
        )


def test_plan_trains_fresh_captured_store_then_publishes_only_new_checkpoint(tmp_path):
    args = SimpleNamespace(
        model="/model",
        trajectories="/trajectory.jsonl",
        output=str(tmp_path),
        tensor_parallel_size=8,
        data_parallel_size=2,
        max_model_len=4096,
        response_tokens=64,
    )
    plan = cycle.cycle_plan(args)
    assert [s["name"] for s in plan["stages"]] == [
        "capture",
        "preflight",
        "train",
        "update",
    ]
    assert plan["environment"]["CARD_COUNT"] == "16"
    assert plan["environment"]["MAX_STEPS"] == "1"
    assert plan["environment"]["FEATURE_STORE"].endswith(
        "capture/features/target_step_00000000"
    )
    assert "draft_engine.target_runtime_version=0" in plan["stages"][2]["command"]
    update = plan["stages"][3]["command"]
    assert update[update.index("--checkpoint") + 1] == str(
        tmp_path / "train/checkpoints/step_00000001"
    )
    assert update[update.index("--trajectories") + 1] == str(tmp_path / "prompt.jsonl")


def test_failed_phase_never_launches_training_or_publish(tmp_path, monkeypatch):
    args = SimpleNamespace(
        model="/model",
        trajectories="/trajectory.jsonl",
        output=str(tmp_path / "run"),
        tensor_parallel_size=8,
        data_parallel_size=2,
        max_model_len=4096,
        response_tokens=64,
    )
    stages = []

    def fail(stage, *args):
        stages.append(stage["name"])
        raise RuntimeError("capture failed")

    monkeypatch.setattr(cycle, "run_stage", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        cycle.run_cycle(args, [1, 2], object())
    assert stages == ["capture"]
    root = Path(args.output)
    assert json.loads((root / "failure.json").read_text())["stage"] == "capture"
    assert not (root / "cycle_report.json").exists()
    with pytest.raises(FileExistsError):
        cycle.run_cycle(args, [1, 2], object())


@pytest.mark.parametrize(
    "delta,nonfinite,passed", [(0.1, 0, True), (0, 0, False), (0.1, 1, False)]
)
def test_training_requires_measured_update_and_finite_backward(
    tmp_path, delta, nonfinite, passed
):
    log = tmp_path / "train.log"
    metrics = {"train_loss": 0.2, "optimizer_lr": 1e-5, "optimizer_probe_delta": delta}
    coverage = {"attention": {"missing": 0, "nonfinite": nonfinite}}
    log.write_text(
        f"DSpark step=1 metrics={metrics}\nDSpark first-backward gradient coverage: {coverage}\n"
    )
    if passed:
        assert cycle.training_evidence(log)["metrics"] == metrics
    else:
        with pytest.raises(RuntimeError):
            cycle.training_evidence(log)


@pytest.mark.parametrize("committed", [True, False])
def test_cycle_accepts_only_complete_fresh_training_and_update_reports(
    tmp_path, monkeypatch, committed
):
    root = tmp_path / "cycle"
    args = SimpleNamespace(
        model="/model",
        trajectories="/trajectory.jsonl",
        output=str(root),
        tensor_parallel_size=8,
        data_parallel_size=2,
        max_model_len=4096,
        response_tokens=64,
    )
    executed = []

    def stage(stage, plan, output):
        executed.append(stage["name"])
        if stage["name"] == "train":
            path = root / "train"
            path.mkdir()
            (path / "training_report.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "hardware_training_validated": True,
                        "progress": {"global_step": 1},
                        "source_samples": 2,
                        "target_runtime_version": 0,
                        "tensor_parallel_size": 8,
                        "data_parallel_size": 2,
                        "checkpoint": str(root / "train/checkpoints/step_00000001"),
                    }
                )
            )
        if stage["name"] == "update":
            path = root / "update/report"
            path.mkdir(parents=True)
            report = dict.fromkeys(
                [
                    "ipc_validated",
                    "server_version_commit_validated",
                    "http_health_validated",
                    "admission_blocked_during_update",
                    "target_tokens_unchanged",
                ],
                True,
            )
            report.update(target_version=0, drafter_revision=1 if committed else 0)
            (path / "server_update_report.json").write_text(json.dumps(report))

    monkeypatch.setattr(cycle, "run_stage", stage)
    monkeypatch.setattr(cycle, "validate_capture", lambda *a: {})
    monkeypatch.setattr(
        cycle, "training_evidence", lambda *a: {"measured_update": True}
    )
    if committed:
        cycle.run_cycle(args, [1, 2], object())
        report = json.loads((root / "cycle_report.json").read_text())
        assert report["status"] == "passed"
        assert report["resident_online_trainer_validated"] is False
        assert report["rl_integration_validated"] is False
        assert executed == ["capture", "preflight", "train", "update"]
    else:
        with pytest.raises(RuntimeError, match="phase report differs"):
            cycle.run_cycle(args, [1, 2], object())
        assert not (root / "cycle_report.json").exists()


def test_server_actor_installs_compat_before_upstream_import(monkeypatch):
    import builtins
    import sys
    from types import SimpleNamespace

    import pytest

    from verl_speco.integration import verl_npu_vllm_compat as compat
    from verl_speco.integration.deepseek_v4_server_probe import (
        create_server_actor_class,
    )

    installed = []
    monkeypatch.setattr(
        compat, "install_verl_npu_vllm_import_compat", lambda: installed.append(True)
    )
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace())
    original = builtins.__import__

    class ReachedUpstream(Exception):
        pass

    def importing(name, *args, **kwargs):
        if name == "verl.workers.rollout.vllm_rollout":
            assert installed == [True]
            raise ReachedUpstream
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ReachedUpstream):
        create_server_actor_class()


@pytest.mark.parametrize("length,prompt_length", [(600, 536), (800, 200), (100, 36)])
def test_resident_window_keeps_absolute_positions_and_speco_mask(
    model, tmp_path, length, prompt_length
):
    from verl_speco.trainer.deepseek_v4_contract import EXTENSION, from_engine_tensors
    from verl_speco.trainer.deepseek_v4_resident_probe import collect_probe_window
    from verl_speco.trainer.feature_store import TorchShardFeatureStore
    from verl_speco.trainer.vllm_ascend_online import VllmAscendOnlineMixin

    contract = DeepSeekV4Contract.from_model(model)
    ids = torch.arange(length) % contract.vocab_size
    mask = (torch.arange(length) >= prompt_length).float()
    hidden = torch.arange(length).float().unsqueeze(1).repeat(1, 4)
    head = hidden + 1000
    sample = from_engine_tensors(
        {
            "input_ids": ids,
            "positions": torch.arange(length),
            "hidden_states": hidden,
            "target_head_hidden_states": head,
        },
        mask,
        contract,
        sample_id="window",
    )
    sample.metadata["runtime_drafter_revision"] = 0
    store = TorchShardFeatureStore(
        tmp_path / "window_store",
        metadata={"source": "verl_rollout", "runtime_target_version": 0},
    )
    store.write_many([sample])
    store.close()

    class Session:
        collect_online_data = VllmAscendOnlineMixin.collect_online_data
        dp_group_world_size = 1
        current_rl_step = 0

        def _mark_buffer_changed(self):
            pass

    session = Session()
    session.contract = contract
    session.collected_data = []
    session.config = SimpleNamespace(
        rollout=SimpleNamespace(drafter=SimpleNamespace(training={"max_seq_len": 512}))
    )
    source = collect_probe_window(
        session, {"store": tmp_path / "window_store"}, dp_rank=0, expected_revision=0
    )
    assert source == {"source": "recorded_rollout_store", "target_version": 0}
    actual = session.collected_data[0]["feature"]
    start = max(0, length - 512)
    torch.testing.assert_close(actual.input_ids, ids[start:])
    torch.testing.assert_close(actual.position_ids, torch.arange(start, length))
    torch.testing.assert_close(actual.loss_mask, mask[start:])
    torch.testing.assert_close(actual.hidden_states, hidden[start:])
    torch.testing.assert_close(
        actual.metadata[EXTENSION]["target_head_hidden_states"], head[start:]
    )
