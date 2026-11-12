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


def test_persistent_lease_checkpoint_provider_is_bound_after_trainer_construction(monkeypatch):
    import asyncio
    import sys
    from verl_speco.trainer import draft_training_loop as loop

    config = OmegaConf.create({
        "speco": {"draft_training": {"scheduler": {"per_rank_reference_ack": True}},
                  "standalone_tq_producer": {"reference_lease_journal_path": "leases.sqlite"}},
        "actor_rollout_ref": {"rollout": {"drafter": {
            "speculative_algorithm": "DSPARK", "training": {
                "feature_store": {"type": "ray_reference"}, "mode": "offline"}}}}})
    snapshot = {"snapshot": "producer-owned"}
    producer = SimpleNamespace(reference_lease_snapshot=SimpleNamespace(remote=lambda: snapshot))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda ref, timeout: ref))
    monkeypatch.setattr(loop, "_init_distributed", lambda: (0, 0, 1))
    for name in ("_verify_live_trainer_topology", "_verify_rollback_checkpoint_across_ranks",
                 "_disable_standalone_sequence_parallel", "_apply_standalone_fsdp_shard_default",
                 "_configure_device", "_build_training_device_mesh"):
        monkeypatch.setattr(loop, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(loop, "_build_backend", lambda _: SimpleNamespace())
    class ProviderBound(Exception):
        pass
    class Trainer:
        def __init__(self, **kwargs):
            self.constructed = True
        def __setattr__(self, name, value):
            if name == "_standalone_lease_snapshot_provider":
                assert self.constructed and value() is snapshot
                raise ProviderBound()
            object.__setattr__(self, name, value)
    monkeypatch.setattr(loop, "DrafterBaseTrainer", Trainer)
    with pytest.raises(ProviderBound):
        asyncio.run(loop._run_standalone_draft_training_async(
            config, scheduled_commands=object(), training_events=object(), producer_handle=producer))


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
        "verl_speco.trainer.tq_retry.time.sleep", delays.append
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
        "verl_speco.trainer.tq_retry.time.sleep", lambda _: None
    )
    with pytest.raises(RuntimeError, match="failed to clear"):
        _clear_tq_batch_across_ranks(
            loader,
            ["k0", "k1"],
            rank=0,
            device=torch.device("cpu"),
        )
    assert loader.clear_calls == [["k0", "k1"]] * 3


def test_tq_completion_retry_does_not_repeat_successful_clear(monkeypatch) -> None:
    loader = _FakeTQLoader()
    events = []
    monkeypatch.setattr("verl_speco.trainer.tq_retry.time.sleep", lambda _: None)

    def put(event):
        events.append(event)
        if len(events) == 1:
            raise ConnectionError("temporary event failure")

    event = {"kind": "training_completed", "plan_id": 1}
    _clear_tq_batch_across_ranks(
        loader, ["k0"], rank=0, device=torch.device("cpu"),
        training_events=SimpleNamespace(put=put), completion_event=event,
    )
    assert loader.clear_calls == [["k0"]]
    assert events == [event, event]


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


def test_prefetch_tracking_subtracts_foreground_wait_not_background_wait():
    selected = _select_standalone_tracking_metrics({
        "dspark/loss": 0.5, "perf/step_time": 0.9,
        "perf/loader_plan_wait_time": 10.0, "perf/batch_ready_wait_time": 0.03,
        "perf/train_time": 0.65,
    })
    assert selected["perf/batch_ready_wait_time"] == 0.03
    assert selected["perf/other_time"] == pytest.approx(0.22)


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
        "perf/loader_plan_wait_time": 30.0, "perf/batch_ready_wait_time": 3.0,
        "perf/train_time": 5.0,
        "train/ploss_0": 2.5, "dspark/valid_token_count": 42.0,
        "perf/loader_tq_get_time": 1.0,
    }
    selected = _select_standalone_tracking_metrics(metrics)
    assert len(selected) == 12
    assert selected["dspark/loss"] == 0.7
    assert selected["perf/other_time"] == 2.0
    assert "train/ploss_0" not in selected
    assert selected["perf/loader_tq_get_time"] == 1.0
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
        _training_sample_draws=6,
        _training_unique_sample_ids={1, 2, 3},
        _block_drafter_metric_prefix=lambda: prefix,
        _block_drafter_config_value=lambda name, default: 0,
    )
    metrics = DrafterBaseTrainer.get_training_metrics(trainer)
    assert metrics[f"{prefix}/loss"] == 3.0
    assert metrics[f"{prefix}/selector_loss"] == 0.4
    assert metrics[f"{prefix}/selector_accuracy"] == 0.6
    assert metrics["drafter/train_sample_draws"] == 6.0
    assert metrics["drafter/train_unique_samples"] == 3.0


@pytest.mark.parametrize("prefix", ["eagle3", "dflash", "dspark", "dflash2"])
def test_standalone_core_metrics_for_each_backend(prefix):
    metrics = {
        f"{prefix}/loss": 1.0, f"{prefix}/top1_acc": 0.5,
        f"{prefix}/mean_acceptance_length": 2.0,
        "train/simulated_acc_len": 1.8, "train/lr": 1e-6,
        "perf/step_time": 10.0, "perf/batch_ready_wait_time": 3.0,
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
        _speco_checkpoint_buffer_state=lambda: None,
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


def test_standalone_checkpoint_wait_timeout_keeps_running_future():
    trainer = _FakeTrainer()
    trainer._standalone_checkpoint_timeout = 0.02
    with pytest.raises(TimeoutError):
        _save_standalone_checkpoint(trainer, 5, wait=True)
    assert trainer._pending_full_checkpoint_future is trainer.future
    assert not trainer.future.cancelled() and not trainer.future.done()


def test_base_checkpoint_wait_timeout_keeps_running_future(monkeypatch, tmp_path):
    import verl_speco.trainer.base_trainer as base
    future = Future()
    trainer = SimpleNamespace(
        checkpoint_dir=str(tmp_path), rollout_dp_rank=0, rank=0,
        _pending_full_checkpoint_future=None, _standalone_checkpoint_timeout=0.02,
        _is_checkpoint_leader=lambda: True,
        _sync_checkpoint_phase_error=lambda error, phase: None,
        model=torch.nn.Linear(1, 1), training_device_mesh=None,
        _save_optimizer_checkpoint=lambda path: {"format": "test"},
    )
    def save_async(step, manifest, *, defer_completion):
        trainer._pending_full_checkpoint_future = future
        return future
    trainer._save_checkpoint_async = save_async
    monkeypatch.setattr(base, "device_name", "cpu")
    with pytest.raises(TimeoutError):
        DrafterBaseTrainer.save_checkpoint(trainer, 5, wait=True, defer_completion=True)
    assert trainer._pending_full_checkpoint_future is future
    assert not future.cancelled() and not future.done()


def test_base_previous_checkpoint_wait_timeout_keeps_running_future(tmp_path):
    future = Future()
    def sync(error, phase):
        assert phase == "previous save wait"
        raise error
    trainer = SimpleNamespace(checkpoint_dir=str(tmp_path), rollout_dp_rank=0, rank=0,
                              _pending_full_checkpoint_future=future, _standalone_checkpoint_timeout=0.02,
                              _is_checkpoint_leader=lambda: True, _sync_checkpoint_phase_error=sync)
    with pytest.raises(TimeoutError):
        DrafterBaseTrainer.save_checkpoint(trainer, 6, wait=True, defer_completion=True)
    assert trainer._pending_full_checkpoint_future is future
    assert not future.cancelled() and not future.done()


def test_checkpoint_error_sync_waits_for_finalization_callback(monkeypatch):
    import threading
    from verl_speco.trainer import draft_training_loop as loop

    started, release, checked = threading.Event(), threading.Event(), threading.Event()
    trainer, future = SimpleNamespace(), Future()
    def finalize(trainer, path, completed, **kwargs):
        started.set()
        assert release.wait(5)
        trainer._standalone_export_error = RuntimeError("finalization failed")
    monkeypatch.setattr(loop, "_finalize_standalone_checkpoint", finalize)
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: False)
    loop._attach_standalone_finalizer(trainer, future, "/unused")
    writer = threading.Thread(target=lambda: future.set_result(None))
    result = []
    checker = threading.Thread(target=lambda: (
        result.append(loop._sync_standalone_export_error(trainer, torch.device("cpu"), raise_on_error=False)),
        checked.set(),
    ))
    writer.start()
    try:
        assert started.wait(5)
        checker.start()
        assert not checked.wait(0.2)
        release.set()
        checker.join(timeout=5)
        assert checked.is_set() and result == [True]
    finally:
        release.set()
        writer.join(timeout=5)
        if checker.ident is not None:
            checker.join(timeout=5)


@pytest.mark.parametrize("keep,delete,exclusive,valid", [
    (0, False, False, True), (2, False, False, True), (2, True, True, True),
    (2, True, False, False), (0, True, True, False), (1, False, True, False),
    (2, True, 1, False), (True, False, True, False),
])
def test_checkpoint_retention_requires_exclusive_ownership_for_deletion(keep, delete, exclusive, valid):
    from verl_speco.trainer.draft_training_loop import _valid_checkpoint_retention
    assert _valid_checkpoint_retention(keep, delete, exclusive) is valid


def test_checkpoint_retention_inheritance_requires_exclusive_ownership():
    from verl_speco.trainer.draft_training_loop import _valid_checkpoint_retention
    assert _valid_checkpoint_retention(2, False, True, True)
    assert not _valid_checkpoint_retention(2, False, False, True)
    assert not _valid_checkpoint_retention(0, False, True, True)
    assert not _valid_checkpoint_retention(2, False, True, 1)


@pytest.mark.parametrize("host,frame,valid", [(300, 100, True), (299, 100, False),
                                             (0, 0, False), (True, 1, False), (300, -1, False)])
def test_node_window_host_reservation(host, frame, valid):
    from verl_speco.trainer.draft_training_loop import _validate_node_window_host_budget
    if valid:
        _validate_node_window_host_budget(host, frame)
    else:
        with pytest.raises(ValueError, match="three-frame"):
            _validate_node_window_host_budget(host, frame)


@pytest.mark.parametrize("failed", [False, True])
def test_owned_node_dispatch_preserves_reference_until_training_finishes(monkeypatch, failed):
    import asyncio
    from verl_speco.trainer import draft_training_loop as loop
    entries = [SimpleNamespace(tag={"sample": 1})]
    groups = {0: entries, 1: entries}
    batch = SimpleNamespace(file_entries=entries, node_file_entries=groups, plan_id=3,
                            receipt_token="receipt", is_padding_rank=False)
    store = SimpleNamespace(run_id="run", reference_contract=lambda values: "contract")
    trainer = SimpleNamespace(_get_hidden_state_clip_value=lambda: 1000.0)
    observed = []
    async def execute(*args, **kwargs):
        observed.append(kwargs["node_context"])
        if failed:
            raise RuntimeError("injected node failure")
        return True
    monkeypatch.setattr(loop, "_owned_reference_training_step", execute)
    monkeypatch.setattr(loop, "_ack_reference_materialization_across_ranks", lambda *args, **kwargs: observed.append("ack"))
    call = loop._execute_owned_reference_batch(trainer, store, batch, 0, rank=0, device=torch.device("cpu"),
                                               events=object(), transport_config={}, node_ranks_per_host=2)
    if failed:
        with pytest.raises(RuntimeError, match="injected node failure"):
            asyncio.run(call)
        assert len(observed) == 1
    else:
        assert asyncio.run(call) and observed[-1] == "ack"
    assert observed[0] == {"rank": 0, "ranks_per_host": 2, "entries": groups, "run_id": "run",
                           "plan_id": 3, "receipt_token": "receipt"}


def _retention_settings_gloo_worker(rank, port, mode, results):
    from datetime import timedelta
    import torch.distributed as dist
    from verl_speco.trainer.draft_training_loop import _validate_checkpoint_retention_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        settings = (2, bool(rank) if mode == "mismatch" else True,
                    False if mode == "invalid" and rank == 1 else True)
        try:
            _validate_checkpoint_retention_across_ranks(settings, rank=rank, device=torch.device("cpu"))
            results.put((rank, "ok"))
        except ValueError as exc:
            results.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("mode", ["same", "mismatch", "invalid"])
def test_retention_settings_agree_on_all_ranks(mode):
    import multiprocessing
    import socket
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [ctx.Process(target=_retention_settings_gloo_worker, args=(rank, port, mode, results))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        rows = [results.get(timeout=30) for _ in processes]
        assert {rank for rank, _ in rows} == {0, 1}
        assert all((result == "ok") if mode == "same" else ("must agree" in result) for _, result in rows)
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()


def test_token_metrics_window_aggregates_steps_and_preserves_selected_metrics(monkeypatch):
    from verl_speco.trainer import standalone_tracking as tracking

    monkeypatch.setattr(tracking.time, "perf_counter", lambda: 20.0)
    window = tracking.StandaloneTokenWindow(2)
    counts = {"source_tokens": 10, "window_tokens": 8, "supervised_tokens": 5,
              "padding_tokens": 2, "samples": 1}
    kwargs = dict(max_steps=10, start_time=10.0, device=torch.device("cpu"), world_size=1)
    assert window.update(counts, successful_steps=1, optimizer_step=1, **kwargs) == {}
    window.record_checkpoint(2.0)
    metrics = window.update(counts, successful_steps=2, optimizer_step=2, **kwargs)
    assert metrics["tokens/source"] == 20 and metrics["tokens/source_per_second"] == 2
    assert metrics["tokens/source_per_second_excluding_checkpoints"] == 2.5
    assert metrics["perf/checkpoint_blocking_time"] == 2.0
    assert metrics["tokens/supervised"] == 10 and metrics["tokens/padding"] == 4
    selected = _select_standalone_tracking_metrics({"dspark/loss": 0.2, **metrics})
    assert selected["tokens/source"] == 20 and selected["tokens/padding"] == 4
    assert selected["tokens/source_per_second_excluding_checkpoints"] == 2.5
    assert selected["perf/checkpoint_blocking_time"] == 2.0
    assert window.update(None, successful_steps=3, optimizer_step=3, **kwargs) == {}
    assert window.update(counts, successful_steps=4, optimizer_step=4, **kwargs) == {}


def test_reference_parts_stage_once_without_mutating_the_input():
    import asyncio
    from verl_speco.trainer.draft_training_loop import _reference_training_parts_across_ranks
    calls = []
    batch = {"hidden_source": "standalone_vllm_reference", "input_ids": torch.ones(1)}
    async def train(value, step, **kwargs):
        calls.append((value.get("_speco_reference_staged"), kwargs["accumulation_index"]))
        return True
    trainer = SimpleNamespace(_stage_standalone_cpu_batch=lambda value: value, training_step_from_batch=train,
                              backend=SimpleNamespace(reference_parts_additive=True))
    assert asyncio.run(_reference_training_parts_across_ranks(trainer, [batch], 1, device=torch.device("cpu")))
    assert calls == [(True, 0)] and "_speco_reference_staged" not in batch


def test_token_metrics_track_window_work_only_when_complete(monkeypatch):
    from verl_speco.trainer import standalone_tracking as tracking
    monkeypatch.setattr(tracking.time, "perf_counter", lambda: 20.0)
    window = tracking.StandaloneTokenWindow(2)
    counts = {"source_tokens": 10, "window_tokens": 8, "supervised_tokens": 5,
              "padding_tokens": 2, "samples": 1, "window_train_input_positions": 17,
              "window_statistics_input_positions": 17}
    kwargs = dict(max_steps=4, start_time=10.0, device=torch.device("cpu"), world_size=1)
    assert window.update(counts, successful_steps=1, optimizer_step=1, **kwargs) == {}
    metrics = window.update(counts, successful_steps=2, optimizer_step=2, **kwargs)
    assert metrics["tokens/window_train_input_positions"] == 34
    assert metrics["tokens/window_statistics_input_positions"] == 34
    assert tracking._select_standalone_tracking_metrics({"dspark/loss": 1, **metrics})[
        "tokens/window_train_input_positions"] == 34
    run = window.finish(device=torch.device("cpu"), world_size=1)
    assert run["tokens/run_window_statistics_input_positions"] == 34
    missing = dict(counts)
    del missing["window_train_input_positions"], missing["window_statistics_input_positions"]
    window.update(counts, successful_steps=3, optimizer_step=3, **kwargs)
    metrics = window.update(missing, successful_steps=4, optimizer_step=4, **kwargs)
    assert "tokens/window_train_input_positions" not in metrics
    assert "tokens/run_window_train_input_positions" not in window.finish(device=torch.device("cpu"), world_size=1)


@pytest.mark.parametrize("key,value", [("loss_mode", "invalid_ce"), ("loss_mode", "unsupported"),
                                      ("per_position_loss_weight", "unsupported"), ("ce_target_source", "unsupported")])
def test_reference_part_loss_recipe_rejects_unverified_modes(key, value):
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    values = dict(loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay", ce_target_source="labels",
                  l1_loss_alpha=0.0, lk_loss_alpha=0.0, confidence_head_alpha=0.0,
                  loss_decay_gamma=7.0, block_size=2)
    trainer = SimpleNamespace(backend=SimpleNamespace(model_type="dspark"),
                              model=SimpleNamespace(module=SimpleNamespace(**values)))
    _validate_reference_part_loss_recipe(trainer)
    setattr(trainer.model.module, key, value)
    with pytest.raises(ValueError, match="full-vocabulary fixed-weight CE"):
        _validate_reference_part_loss_recipe(trainer)


@pytest.mark.parametrize("alpha,l1", [(-0.01, 0.0), (1.01, 0.0), (True, 0.0), (float("nan"), 0.0)])
def test_reference_dpace_rejects_unsupported_normalization(alpha, l1):
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    model = SimpleNamespace(loss_mode="full_vocab", per_position_loss_weight="dpace", ce_target_source="labels",
                            l1_loss_alpha=l1, lk_loss_alpha=0.0, confidence_head_alpha=0.0,
                            loss_decay_gamma=7.0, block_size=2, dpace_alpha=alpha)
    trainer = SimpleNamespace(backend=SimpleNamespace(model_type="dspark"), model=model)
    with pytest.raises(ValueError, match="Reference D-PACE"):
        _validate_reference_part_loss_recipe(trainer)


def test_reference_restricted_ce_rejects_target_argmax():
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    model = SimpleNamespace(loss_mode="restricted_ce", per_position_loss_weight="fixed_exp_decay",
                            ce_target_source="target_argmax", l1_chunk_size=0, l1_loss_alpha=0.0,
                            lk_loss_alpha=0.0, confidence_head_alpha=0.0, loss_decay_gamma=7.0, block_size=2)
    trainer = SimpleNamespace(backend=SimpleNamespace(model_type="dspark"), model=model)
    with pytest.raises(ValueError, match="restricted CE with labels"):
        _validate_reference_part_loss_recipe(trainer)


def test_reference_part_loss_rejects_tiny_position_weights():
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    model = SimpleNamespace(loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay",
                            ce_target_source="labels", l1_loss_alpha=0.0, lk_loss_alpha=0.0,
                            confidence_head_alpha=0.0, loss_decay_gamma=0.01, block_size=2)
    with pytest.raises(ValueError, match="normalization floor"):
        _validate_reference_part_loss_recipe(SimpleNamespace(backend=SimpleNamespace(model_type="dspark"), model=model))


def test_reference_part_loss_accepts_unchunked_target_argmax():
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    model = SimpleNamespace(loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay",
                            ce_target_source="target_argmax", l1_chunk_size=0,
                            l1_loss_alpha=0.0, lk_loss_alpha=0.0, confidence_head_alpha=0.0,
                            loss_decay_gamma=7.0, block_size=2)
    trainer = SimpleNamespace(backend=SimpleNamespace(model_type="dspark"), model=model)
    assert _validate_reference_part_loss_recipe(trainer) is model
    model.l1_chunk_size = 1
    with pytest.raises(ValueError, match="full-vocabulary fixed-weight CE"):
        _validate_reference_part_loss_recipe(trainer)


def test_reference_part_loss_requires_deterministic_adaptive_lk():
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    model = SimpleNamespace(loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay",
                            ce_target_source="labels", l1_loss_alpha=0.9, lk_loss_alpha=0.1,
                            confidence_head_alpha=0.0, lk_loss_type="adaptive_hybrid",
                            lk_hybrid_eta=3.0, loss_decay_gamma=7.0, block_size=2, modules=lambda: [])
    trainer = SimpleNamespace(backend=SimpleNamespace(model_type="dspark"), model=model)
    assert _validate_reference_part_loss_recipe(trainer) is model
    model.modules = lambda: [torch.nn.Dropout(p=0.1)]
    with pytest.raises(ValueError, match="deterministic model dropout"):
        _validate_reference_part_loss_recipe(trainer)
    model.modules = lambda: [torch.nn.BatchNorm1d(2)]
    with pytest.raises(ValueError, match="deterministic model dropout and state"):
        _validate_reference_part_loss_recipe(trainer)
    model.modules = lambda: [SimpleNamespace(attention_dropout=0.1)]
    with pytest.raises(ValueError, match="deterministic model dropout and state"):
        _validate_reference_part_loss_recipe(trainer)
    model.modules = lambda: []
    model.lk_hybrid_eta = float("nan")
    with pytest.raises(ValueError, match="coefficient"):
        _validate_reference_part_loss_recipe(trainer)


def test_reference_confidence_requires_enabled_head():
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    model = SimpleNamespace(loss_mode="full_vocab", per_position_loss_weight="fixed_exp_decay",
                            ce_target_source="labels", l1_loss_alpha=0.0, lk_loss_alpha=0.0,
                            confidence_head_alpha=0.1, loss_decay_gamma=7.0, block_size=2)
    with pytest.raises(ValueError, match="enabled confidence head"):
        _validate_reference_part_loss_recipe(SimpleNamespace(backend=SimpleNamespace(model_type="dspark"), model=model))


def test_reference_part_loss_rejects_unverified_backend():
    from verl_speco.trainer.draft_training_loop import _validate_reference_part_loss_recipe
    with pytest.raises(ValueError, match="not validated"):
        _validate_reference_part_loss_recipe(SimpleNamespace(backend=SimpleNamespace(model_type="eagle3")))


def test_token_metrics_do_not_divide_by_zero_checkpoint_window(monkeypatch):
    from verl_speco.trainer import standalone_tracking as tracking
    monkeypatch.setattr(tracking.time, "perf_counter", lambda: 10.0)
    window = tracking.StandaloneTokenWindow(1)
    window.record_checkpoint(1.0)
    counts = {"source_tokens": 5, "window_tokens": 5, "supervised_tokens": 3,
              "padding_tokens": 0, "samples": 1}
    metrics = window.update(counts, successful_steps=1, optimizer_step=1, max_steps=1,
                            start_time=10.0, device=torch.device("cpu"), world_size=1)
    assert "tokens/source_per_second" not in metrics
    assert "tokens/source_per_second_excluding_checkpoints" not in metrics
    assert metrics["perf/checkpoint_blocking_time"] == 1.0


def test_run_token_metrics_include_final_checkpoint_and_incomplete_counts(monkeypatch):
    from verl_speco.trainer import standalone_tracking as tracking
    clock = [20.0]
    monkeypatch.setattr(tracking.time, "perf_counter", lambda: clock[0])
    window = tracking.StandaloneTokenWindow(1)
    counts = {"source_tokens": 10, "window_tokens": 8, "supervised_tokens": 5,
              "padding_tokens": 2, "samples": 1}
    kwargs = dict(max_steps=2, start_time=10.0, device=torch.device("cpu"), world_size=1)
    window.update(counts, successful_steps=1, optimizer_step=1, **kwargs)
    clock[0] = 25.0
    window.update(counts, successful_steps=2, optimizer_step=2, **kwargs)
    window.record_checkpoint(5.0)
    clock[0] = 30.0
    metrics = window.finish(device=torch.device("cpu"), world_size=1)
    assert metrics["tokens/run_source"] == 20
    assert metrics["perf/run_training_time"] == 20
    assert metrics["perf/run_checkpoint_blocking_time"] == 5
    assert metrics["tokens/run_source_per_second"] == 1
    assert metrics["tokens/run_source_per_second_excluding_checkpoints"] == pytest.approx(20 / 15)
    incomplete = tracking.StandaloneTokenWindow(1)
    incomplete.update(None, successful_steps=1, optimizer_step=1, **kwargs)
    assert "tokens/run_source" not in incomplete.finish(device=torch.device("cpu"), world_size=1)



def test_checkpoint_remains_incomplete_if_input_changes_during_training(tmp_path, monkeypatch):
    from verl_speco.trainer import draft_training_loop as loop
    from verl_speco.trainer.standalone_resume import build_input_fingerprint, input_content_identity
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"first\n")
    identity = input_content_identity(build_input_fingerprint(source, include_content_hash=True))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text('{"complete": true}', encoding="utf-8")
    monkeypatch.setattr(loop, "_rewrite_standalone_block_runtime_config", lambda *args: None)
    source.write_bytes(b"other\n")
    trainer = SimpleNamespace(_standalone_input_identity=identity)
    with pytest.raises(ValueError, match="changed since training began"):
        loop._finalize_standalone_checkpoint_files(trainer, str(checkpoint),
            consumed_snapshot=torch.tensor([0]), step=1, input_path=str(source))
    assert json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["complete"] is False
    assert not (checkpoint / "standalone_resume.json").exists()
    with pytest.raises(ValueError, match="without input path and progress"):
        loop._finalize_standalone_checkpoint_files(trainer, str(checkpoint),
            consumed_snapshot=torch.tensor([0]), step=1, input_path=None)
    assert json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["complete"] is False



@pytest.mark.parametrize("storage_format", ["tensor", "gzip"])
def test_standalone_finalization_seals_checkpoint_before_retention(tmp_path, storage_format):
    import torch.distributed.checkpoint as dcp
    from verl_speco.trainer.checkpoint import verify_standalone_checkpoint_integrity, write_standalone_optimizer_layout
    from verl_speco.trainer.draft_training_loop import _finalize_standalone_checkpoint_files

    checkpoint = tmp_path / "draft_step_1"
    optimizer = checkpoint / "optimizer"
    optimizer.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "pytorch_model.bin").write_bytes(b"weights")
    dcp_metadata = dcp.save({"state": torch.tensor([1])}, checkpoint_id=optimizer, no_dist=True)
    write_standalone_optimizer_layout(optimizer, dcp_metadata)
    (optimizer / "trainer_state.pt").write_bytes(b"state")
    (checkpoint / "metadata.json").write_text(json.dumps({
        "step": 1, "complete": False,
        "optimizer": {"format": "torch_distributed_checkpoint", "path": "optimizer",
                      "trainer_state_file": "trainer_state.pt"},
    }), encoding="utf-8")
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="eagle3"),
        _standalone_retention_keep_last=2, _standalone_retention_delete=False,
        _standalone_retention_run_id="a" * 32, _standalone_retention_protected=(),
        _standalone_resume_format=storage_format,
    )
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    import hashlib
    lease_snapshot = {"version": 1, "incarnation": "producer", "drained": True}
    lease_snapshot["sha256"] = hashlib.sha256(json.dumps(lease_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    trainer._standalone_lease_snapshot_provider = lambda: lease_snapshot
    _finalize_standalone_checkpoint_files(
        trainer, str(checkpoint), consumed_snapshot=torch.tensor([1]), step=1, input_path=str(source),
    )
    from verl_speco.trainer.standalone_resume import load_standalone_resume
    consumed, resume = load_standalone_resume(checkpoint, input_path=source)
    assert consumed == {1} and resume["schema_version"] == 3 and resume["storage_format"] == storage_format
    assert verify_standalone_checkpoint_integrity(checkpoint)
    assert resume["reference_lease_snapshot_sha256"] == lease_snapshot["sha256"]
    assert (checkpoint / "reference_leases.json").is_file()
    assert json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))["complete"] is True


def test_checkpoint_retention_deletion_rechecks_directory_lease(tmp_path, monkeypatch):
    from verl_speco.trainer import checkpoint, draft_training_loop as loop
    calls = []
    monkeypatch.setattr(loop, "_set_standalone_checkpoint_complete", lambda *args, **kwargs: None)
    monkeypatch.setattr(loop, "_rewrite_standalone_block_runtime_config", lambda *args: None)
    monkeypatch.setattr(loop, "_save_resume_sidecar", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpoint, "write_standalone_integrity_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(checkpoint, "retire_standalone_checkpoints", lambda *args, **kwargs: calls.append(kwargs) or [])
    trainer = SimpleNamespace(_standalone_retention_keep_last=2, _standalone_retention_delete=True,
                              _standalone_retention_run_id="a" * 32, _standalone_retention_protected=())
    path = str(tmp_path / "draft_step_3")
    def finalize():
        loop._finalize_standalone_checkpoint_files(trainer, path, consumed_snapshot=None, step=3, input_path=None)
    finalize()
    assert not calls
    trainer._standalone_checkpoint_directory_lease = SimpleNamespace(
        directory=tmp_path, verify=lambda: (_ for _ in ()).throw(RuntimeError("marker missing")))
    finalize()
    assert not calls
    trainer._standalone_checkpoint_directory_lease.verify = lambda: None
    finalize()
    assert len(calls) == 1 and calls[0]["delete"] is True


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


def test_rollback_checkpoint_agreement_rejects_divergent_or_failed_rank(monkeypatch, tmp_path):
    import verl_speco.trainer.checkpoint as checkpoint
    import verl_speco.trainer.draft_training_loop as loop

    (tmp_path / "standalone_integrity.json").write_text("manifest", encoding="utf-8")
    monkeypatch.setattr(checkpoint, "verify_standalone_checkpoint_integrity", lambda path: True)
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: True)
    def gather_matching(out, local):
        out[:] = [local, local]
    monkeypatch.setattr(loop.dist, "all_gather_object", gather_matching)
    loop._verify_rollback_checkpoint_across_ranks({"checkpoint_auto_rollback": True}, tmp_path, 2)

    def gather_other(out, local):
        out[:] = [local, ("different", None)]
    monkeypatch.setattr(loop.dist, "all_gather_object", gather_other)
    with pytest.raises(ValueError, match="differs or is invalid"):
        loop._verify_rollback_checkpoint_across_ranks({"checkpoint_auto_rollback": True}, tmp_path, 2)

    monkeypatch.setattr(checkpoint, "verify_standalone_checkpoint_integrity", lambda path: False)
    monkeypatch.setattr(loop.dist, "all_gather_object", gather_matching)
    with pytest.raises(ValueError, match="integrity verification failed"):
        loop._verify_rollback_checkpoint_across_ranks({"checkpoint_auto_rollback": True}, tmp_path, 2)


def test_step_rng_is_repeatable_across_restart_and_distinct_between_ranks():
    import random
    import numpy as np
    from verl_speco.trainer.draft_training_loop import _seed_standalone_step
    def draw(rank, step):
        _seed_standalone_step(17, step, rank)
        return random.random(), np.random.random(), torch.rand(4).tolist()
    first = draw(0, 9)
    draw(1, 10)
    assert draw(0, 9) == first
    assert draw(1, 9) != first
    assert draw(0, 10) != first


def test_deterministic_rng_resume_binds_device_type(caplog):
    from verl_speco.trainer.draft_training_loop import _standalone_rng_identity, _validate_resume_training_plan
    assert _standalone_rng_identity(False, torch.device("cpu")) == {}
    for kind in ("cpu", "cuda", "npu"):
        assert _standalone_rng_identity(True, SimpleNamespace(type=kind)) == {"rng_device_type": kind}
    with pytest.raises(ValueError, match="must be a boolean"):
        _standalone_rng_identity(1, torch.device("cpu"))
    with pytest.raises(ValueError, match="supported"):
        _standalone_rng_identity(True, SimpleNamespace(type="meta"))
    saved = {"world_size": 2, "deterministic_step_rng": True, "rng_device_type": "cpu"}
    metadata = {"optimizer_step": 4, "training_plan": saved}
    _validate_resume_training_plan(metadata, 4, dict(saved))
    with pytest.raises(ValueError, match="training plan changed"):
        _validate_resume_training_plan(metadata, 4, {**saved, "rng_device_type": "cuda"})
    legacy = {key: value for key, value in saved.items() if key != "rng_device_type"}
    _validate_resume_training_plan({"optimizer_step": 4, "training_plan": legacy}, 4, saved)
    assert "cannot be verified" in caplog.text


def test_step_rng_seeds_npu_with_the_same_rank_step_seed(monkeypatch):
    from verl_speco.trainer.draft_training_loop import _seed_standalone_step
    seeds = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: seeds.append(("torch", seed)))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: True,
                        manual_seed_all=lambda seed: seeds.append(("npu", seed))), raising=False)
    _seed_standalone_step(17, 3, 1)
    assert len(seeds) == 2 and seeds[0][0] == "torch" and seeds[1] == ("npu", seeds[0][1])


def test_resume_rank_consensus_reports_failed_or_different_progress(monkeypatch):
    import verl_speco.trainer.draft_training_loop as loop
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loop, "load_standalone_resume", lambda *a, **kw: ({1, 2}, {"optimizer_step": 3}))
    def agree(out, local):
        out[:] = [local, local]
    monkeypatch.setattr(loop.dist, "all_gather_object", agree)
    assert loop._load_resume_across_ranks("path", None, 3, {"world_size": 2}, 2)[0] == {1, 2}
    def mismatch(out, local):
        out[:] = [local, {**local, "progress": "different"}]
    monkeypatch.setattr(loop.dist, "all_gather_object", mismatch)
    with pytest.raises(ValueError, match="differs or failed across ranks"):
        loop._load_resume_across_ranks("path", None, 3, {"world_size": 2}, 2)
    def fail(*args, **kwargs):
        raise OSError("missing checkpoint")
    monkeypatch.setattr(loop, "load_standalone_resume", fail)
    monkeypatch.setattr(loop.dist, "all_gather_object", agree)
    with pytest.raises(ValueError, match="missing checkpoint"):
        loop._load_resume_across_ranks("path", None, 3, {"world_size": 2}, 2)


def test_new_save_waits_for_previous_finalizer_and_surfaces_error():
    import threading
    import time
    trainer = _FakeTrainer()
    trainer._standalone_finalization_event = threading.Event()
    result = []
    def save():
        try:
            _save_standalone_checkpoint(trainer, 2)
        except Exception as exc:
            result.append(exc)
    thread = threading.Thread(target=save)
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive() and trainer.calls == 0
    trainer._standalone_export_error = RuntimeError("sidecar failed")
    trainer._standalone_finalization_event.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and trainer.calls == 0
    assert len(result) == 1 and "checkpoint" in str(result[0]).lower()


def test_checkpoint_finalization_timeout_stops_new_save_and_retains_state(monkeypatch):
    import threading
    import verl_speco.trainer.draft_training_loop as loop
    trainer = _FakeTrainer()
    trainer._standalone_finalization_event = threading.Event()
    trainer._standalone_checkpoint_timeout = 0.02
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="checkpoint export failed or timed out") as error:
        loop._sync_standalone_export_error(trainer, torch.device("cpu"))
    assert isinstance(error.value.__cause__, TimeoutError)
    assert not trainer._standalone_finalization_event.is_set() and trainer.calls == 0
    with pytest.raises(TimeoutError, match="remains unresolved"):
        _save_standalone_checkpoint(trainer, 2)
    assert trainer.calls == 0


@pytest.mark.parametrize("mapped", [False, True])
def test_detached_reference_batch_survives_source_reuse_and_checks_budget(tmp_path, mapped):
    from dataclasses import replace
    from verl_speco.trainer.draft_training_loop import _detach_reference_batch
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    source = torch.arange(8, dtype=torch.float32)
    if mapped:
        path = tmp_path / "source.pt"
        torch.save(source, path)
        source = torch.load(path, weights_only=True, mmap=True)
    batch = TQLocalBatch(["sample"], [], ["sample"], prepared_batch={"input_ids": source})
    detached = _detach_reference_batch(batch, {"consumer_prefetch_batch_bytes": 64})
    source.zero_()
    assert torch.equal(detached.prepared_batch["input_ids"], torch.arange(8, dtype=torch.float32))
    assert detached.global_keys == batch.global_keys
    with pytest.raises(ValueError, match="batch byte budget"):
        _detach_reference_batch(batch, {"consumer_prefetch_batch_bytes": 63})
    nested = replace(batch, prepared_batch={"nested": [torch.ones(16)]})
    with pytest.raises(ValueError, match="batch byte budget"):
        _detach_reference_batch(nested, {"consumer_prefetch_batch_bytes": 64})


@pytest.mark.parametrize("local_exhausted", [False, True])
def test_reference_source_rejects_inconsistent_rank_exhaustion(monkeypatch, local_exhausted):
    from verl_speco.trainer.tq_sample_source import TQLocalBatch

    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.is_initialized", lambda: True)
    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.get_world_size", lambda: 2)
    def reduce(state, op):
        state[1] = 1
        state[2] = 1
    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.all_reduce", reduce)
    batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)})
    with pytest.raises(RuntimeError, match="exhausted inconsistently"):
        _next_batch_across_ranks(iter([] if local_exhausted else [batch]), rank=0,
                                 device=torch.device("cpu"), require_prepared=True)


def test_reference_source_rejects_mismatched_dispatch_identity(monkeypatch):
    from verl_speco.trainer.tq_sample_source import TQLocalBatch

    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.is_initialized", lambda: True)
    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.get_world_size", lambda: 2)
    def reduce(state, op):
        state[3] += 1
    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.all_reduce", reduce)
    batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)},
                         plan_id=2, receipt_token="dispatch")
    with pytest.raises(RuntimeError, match="dispatch identity differs"):
        _next_batch_across_ranks(iter([batch]), rank=0, device=torch.device("cpu"), require_prepared=True)


def test_reference_source_accepts_only_matching_deferred_assignments(monkeypatch):
    from dataclasses import replace
    from verl_speco.trainer.tq_feature_store import ReadyEntry
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    monkeypatch.setattr("verl_speco.trainer.draft_training_loop.dist.is_initialized", lambda: False)
    entry = ReadyEntry("key", {"key": "metadata"})
    batch = TQLocalBatch(["key"], [], ["key"], file_entries=[entry], plan_id=2,
                         receipt_token="dispatch", assignment_digest="a" * 64)
    kwargs = dict(rank=0, device=torch.device("cpu"), require_reference_metadata=True)
    assert _next_batch_across_ranks(iter([batch]), **kwargs) is batch
    padding = replace(batch, local_keys=[], is_padding_rank=True)
    assert _next_batch_across_ranks(iter([padding]), **kwargs) is padding
    for invalid in (replace(batch, file_entries=[]), replace(batch, local_keys=["other"]),
                    replace(batch, cleanup_after_load=True), replace(batch, assignment_digest=None),
                    replace(batch, prepared_batch={"input_ids": torch.ones(1)})):
        with pytest.raises(RuntimeError, match="target-feature producer failed"):
            _next_batch_across_ranks(iter([invalid]), **kwargs)


def test_reference_batch_rejects_conflicting_source_identity(monkeypatch):
    import verl_speco.trainer.draft_training_loop as loop
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: False)
    expected = {"kind": "file", "sha256": "a" * 64, "size_bytes": 4}
    other = {"kind": "file", "sha256": "b" * 64, "size_bytes": 4}
    batch = TQLocalBatch([], [], [], prepared_batch={"input_ids": torch.ones(1)},
                         plan_id=1, receipt_token="token", input_identity=other)
    with pytest.raises(RuntimeError, match="target-feature producer failed") as error:
        loop._next_batch_across_ranks(iter([batch]), rank=0, device=torch.device("cpu"),
            require_prepared=True, expected_input_identity=expected)
    assert "input content differs" in str(error.value.__cause__)


def test_checkpoint_agreement_blocks_publication_when_another_rank_fails(monkeypatch):
    import verl_speco.trainer.draft_training_loop as loop

    trainer = _FakeTrainer()
    trainer.future.set_result(None)
    finalized = []
    monkeypatch.setattr(loop, "_finalize_standalone_checkpoint_files", lambda *a, **kw: finalized.append(True))
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loop.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(loop.dist, "get_rank", lambda: 0)
    def fail_remote(state, op):
        state[0] += 1
    monkeypatch.setattr(loop.dist, "all_reduce", fail_remote)
    with pytest.raises(RuntimeError, match="save failed on at least one rank"):
        loop._save_standalone_checkpoint_across_ranks(trainer, 5, device=torch.device("cpu"), wait=True)
    assert not finalized


def test_checkpoint_agreement_rejects_missing_export_writer(monkeypatch):
    import verl_speco.trainer.draft_training_loop as loop

    trainer = SimpleNamespace(save_checkpoint=lambda *a, **kw: {
        "saved": True, "path": "unused", "reason": "optimizer_shard_saved"})
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="exactly one export writer"):
        loop._save_standalone_checkpoint_across_ranks(trainer, 5, device=torch.device("cpu"))


def test_native_reader_requires_safe_scheduled_lifecycle():
    from verl_speco.trainer.draft_training_loop import _validate_native_reader
    valid = {"per_rank_ack": True, "node_reference_owner": False, "owned_windows": False,
             "tq_cfg": {"consumer_batch_workers": 0, "expected_feature": {"payload_kind": "vllm_reference"}},
             "producer_cfg": {"max_published_references": 1}}
    _validate_native_reader(**valid)
    _validate_native_reader(**{**valid, "producer_cfg": {"publication_after_parent_commit": True}})
    for change in ({"per_rank_ack": False}, {"node_reference_owner": True}, {"owned_windows": True},
                   {"tq_cfg": {"consumer_batch_workers": 1, "expected_feature": {"payload_kind": "vllm_reference"}}},
                   {"tq_cfg": {"consumer_batch_workers": 0}}, {"producer_cfg": {}}):
        with pytest.raises(ValueError, match="Native reader requires"):
            _validate_native_reader(**{**valid, **change})


def test_live_topology_rejects_native_reader_configuration_mismatch(monkeypatch):
    from copy import deepcopy
    from omegaconf import OmegaConf
    import verl_speco.trainer.draft_training_loop as loop
    config = OmegaConf.create({"speco": {"draft_training": {"nnodes": 1, "nproc_per_node": 2,
                                "trainer_node_ips": None, "scheduler": {"per_rank_reference_ack": True}},
                                "standalone_tq_producer": {"max_published_references": 2}},
                               "actor_rollout_ref": {"rollout": {"drafter": {"training": {
                                   "feature_store": {"type": "ray_reference"},
                                   "transfer_queue": {"run_id": "run"},
                                   "target_feature_replay": {"hidden_states_store": {
                                       "backend": "native", "native": {"generation": 1}}}}}}}})
    def gather(output, local):
        output[0] = deepcopy(local)
        output[1] = deepcopy(local)
        output[1]["rank"] = 1
        output[1]["plan"]["native_reader"]["settings"]["generation"] = 2
    monkeypatch.setattr(loop.dist, "all_gather_object", gather)
    with pytest.raises(ValueError, match="disagree"):
        loop._verify_live_trainer_topology(config, rank=0, world_size=2, scheduled=True,
                                           events_available=True, producer_available=True)


def test_reference_device_budget_checks_before_staging_and_training():
    import asyncio
    from verl_speco.trainer.draft_training_loop import _reference_training_step_across_ranks
    calls = []
    async def train(batch, step, **kwargs):
        calls.append("train")
        return True
    trainer = SimpleNamespace(
        _stage_standalone_cpu_batch=lambda batch: (calls.append("stage") or dict(batch)),
        training_step_from_batch=train,
    )
    batch = {"hidden_source": "standalone_vllm_reference", "hidden_states": torch.zeros(8)}
    with pytest.raises(RuntimeError, match="device byte budget") as error:
        asyncio.run(_reference_training_step_across_ranks(
            trainer, batch, 0, device=torch.device("cpu"), max_device_bytes=31))
    assert isinstance(error.value.__cause__, ValueError) and not calls
    assert asyncio.run(_reference_training_step_across_ranks(
        trainer, batch, 0, device=torch.device("cpu"), max_device_bytes=32))
    assert calls == ["stage", "train"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "1024"])
def test_reference_device_budget_rejects_invalid_values(value):
    from verl_speco.trainer.draft_training_loop import _reference_device_budget
    with pytest.raises(ValueError, match="non-negative integer"):
        _reference_device_budget({"consumer_device_batch_bytes": value}, "ray_reference")
    with pytest.raises(ValueError, match="requires ray_reference"):
        _reference_device_budget({"consumer_device_batch_bytes": 1024}, "tq")


def test_device_prefetch_requires_two_batch_budgets_and_supported_backend():
    from verl_speco.trainer.draft_training_loop import _reference_device_prefetch_budget
    assert _reference_device_prefetch_budget({}, "ray_reference", "cpu", 0) == 0
    assert _reference_device_prefetch_budget({"consumer_device_prefetch_bytes": 2048, "consumer_prefetch_batches": 1}, "ray_reference", "cuda", 1024) == 2048
    for config, store, device, limit in [
        ({"consumer_device_prefetch_bytes": -1}, "ray_reference", "cuda", 1024),
        ({"consumer_device_prefetch_bytes": 2048}, "ray_reference", "cpu", 1024),
        ({"consumer_device_prefetch_bytes": 2048}, "tq", "cuda", 1024),
        ({"consumer_device_prefetch_bytes": 1024}, "ray_reference", "npu", 1024),
        ({"consumer_device_prefetch_bytes": 2048}, "ray_reference", "npu", 0),
        ({"consumer_device_prefetch_bytes": 2048}, "ray_reference", "npu", 1024),
        ({"consumer_device_prefetch_bytes": 2048, "consumer_prefetch_batches": 1,
          "consumer_prefetch_host_bytes": 100, "consumer_prefetch_batch_bytes": 60}, "ray_reference", "npu", 1024),
    ]:
        with pytest.raises(ValueError):
            _reference_device_prefetch_budget(config, store, device, limit)


def test_reference_device_budget_counts_each_aliased_tensor_copy():
    from verl_speco.trainer.draft_training_loop import _reference_device_transfer_bytes
    value = torch.arange(8, dtype=torch.float32)
    assert _reference_device_transfer_bytes({"a": value[:4], "b": value[4:]}) == 64


def test_reference_step_stages_once_and_preserves_backend_hidden_source():
    import asyncio
    from verl_speco.trainer.draft_training_loop import _reference_training_step_across_ranks

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    stages = []
    def stage(batch):
        stages.append(True)
        return dict(batch)
    async def train(batch, step):
        assert batch["hidden_source"] == "standalone_vllm_reference"
        assert "_speco_reference_staged" not in batch
        return True
    trainer._stage_standalone_cpu_batch = stage
    trainer._training_step_on_batch = train
    assert asyncio.run(_reference_training_step_across_ranks(
        trainer, {"hidden_source": "standalone_vllm_reference"}, 0, device=torch.device("cpu")))
    assert stages == [True]


def _device_budget_gloo_worker(rank, port, oversized, results):
    import asyncio
    import datetime
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from verl_speco.trainer.draft_training_loop import _reference_training_step_across_ranks
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=25))
    calls = []
    async def train(batch, step, **kwargs):
        calls.append("train")
        return True
    trainer = SimpleNamespace(_stage_standalone_cpu_batch=lambda batch: (calls.append("stage") or dict(batch)),
                              training_step_from_batch=train)
    batch = {"hidden_source": "standalone_vllm_reference",
             "hidden_states": torch.zeros(64 if oversized and rank == 1 else 4)}
    try:
        try:
            asyncio.run(_reference_training_step_across_ranks(
                trainer, batch, 0, device=torch.device("cpu"), max_device_bytes=128))
            results.put((rank, "ok", calls))
        except RuntimeError as exc:
            results.put((rank, str(exc), calls))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("oversized", [False, True])
def test_reference_device_budget_two_gloo_ranks(oversized):
    import multiprocessing
    import socket
    import torch.distributed as dist
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [ctx.Process(target=_device_budget_gloo_worker, args=(rank, port, oversized, results))
                 for rank in range(2)]
    for process in processes:
        process.start()
    try:
        rows = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=20)
        assert all(process.exitcode == 0 for process in processes)
        assert {row[0] for row in rows} == {0, 1}
        if oversized:
            assert all("device byte budget" in row[1] and not row[2] for row in rows)
        else:
            assert all(row[1] == "ok" and row[2] == ["stage", "train"] for row in rows)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
