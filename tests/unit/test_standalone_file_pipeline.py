# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
from dataclasses import asdict
from functools import partial
from queue import Queue

import pytest
import torch
from safetensors.torch import save_file

from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples
from verl_speco.trainer.target_feature_replay import FeatureContract
from verl_speco.trainer.tq_feature_store import ReferenceFeatureStore, ReadyEntry, TQFeatureStore
from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader, TQPrefetchIterator
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag, make_sample_key,
)


def _entry(tmp_path):
    path = tmp_path / "hidden.safetensors"
    save_file({"token_ids": torch.arange(4), "hidden_states": torch.ones(4, 2, 2)}, path)
    contract = FeatureContract(
        algorithm="DFLASH", target_layer_ids=[0, 1], vllm_aux_hidden_state_layer_ids=[1, 2],
        hidden_states_layout="dflash_aux", dtype=torch.bfloat16, target_model_id="target",
        target_model_revision="rev", tokenizer_fingerprint="tokens", require_full_alignment=True,
        target_num_hidden_layers=4,
    )
    values = asdict(contract)
    values["dtype"] = str(contract.dtype)
    meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", "sample", 0)
    tag = make_reference_tag(meta, path=str(path), feature_contract=values, request={
        "input_ids": [0, 1, 2, 3], "vllm_prompt_token_ids": [0, 1, 2, 3],
        "loss_mask": [0, 1, 1, 1], "position_ids": [0, 1, 2, 3],
        "feature_positions": [0, 1, 2, 3], "draft_position_ids": [0, 1, 2, 3],
        "source_metadata": {},
    })
    return ReadyEntry(make_sample_key(meta), tag), path


@pytest.mark.parametrize("delete_fails", [2, 99])
def test_read_retry_finishes_before_independent_cleanup_retry(tmp_path, monkeypatch, delete_fails):
    import verl_speco.trainer.reference_batch_prefetch as worker
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    entry, path = _entry(tmp_path)
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"},
                                       "expected_feature": {"algorithm": "DFLASH"}})
    store.connect()
    original_prepare = store.prepare_batch
    original_release = FileHiddenStatesStore.release
    reads, deletes = [], []

    def prepare(entries, **options):
        reads.append(1)
        assert path.exists(), "Read retries must retain all source files"
        if len(reads) == 1:
            raise TimeoutError("temporary file read timeout")
        return original_prepare(entries, **options)

    def release(self, reference):
        deletes.append(1)
        assert len(reads) == 2
        if len(deletes) <= delete_fails:
            raise OSError("temporary unlink error")
        return original_release(self, reference)

    monkeypatch.setattr(store, "prepare_batch", prepare)
    monkeypatch.setattr(FileHiddenStatesStore, "release", release)
    monkeypatch.setattr(worker, "_STORE", store)
    monkeypatch.setattr(worker, "_OPTIONS", {"max_window": 4})
    monkeypatch.setattr(worker, "_ATTEMPTS", 2)
    batch, _ = worker._prepare_batch([entry])
    assert batch["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert len(reads) == 2 and len(deletes) == 3
    assert path.exists() == (delete_fails > 2)


def test_partial_batch_read_failure_does_not_delete_successful_sample(tmp_path, monkeypatch):
    import verl_speco.trainer.reference_batch_prefetch as worker

    entry, path = _entry(tmp_path)
    missing = ReadyEntry(entry.key, dict(entry.tag, hidden_states_path=str(tmp_path / "missing")))
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"},
                                       "expected_feature": {"algorithm": "DFLASH"}})
    store.connect()
    monkeypatch.setattr(worker, "_STORE", store)
    monkeypatch.setattr(worker, "_OPTIONS", {"max_window": 4})
    monkeypatch.setattr(worker, "_ATTEMPTS", 1)
    with pytest.raises(FileNotFoundError):
        worker._prepare_batch([entry, missing])
    assert path.exists()


def _process_source(tmp_path, monkeypatch, *, missing=False, locked=False):
    from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator

    commands = Queue()
    paths = []
    for index in range(2):
        folder = tmp_path / str(index)
        folder.mkdir()
        entry, path = _entry(folder)
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", f"sample-{index}", index)
        tag = dict(entry.tag, sample_id=meta.sample_id, sequence_no=index)
        entry = ReadyEntry(make_sample_key(meta), tag)
        commands.put({"kind": "batch", "plan_id": index, "global_keys": [entry.key],
                      "global_sequence_nos": [index],
                      "assignments": [[{"key": entry.key, "tag": entry.tag}]]})
        paths.append(path)
    commands.put({"kind": "stop"})
    if missing:
        paths[0].unlink()
    lock_handle = None
    if locked:
        lock_path = paths[0].with_suffix(".safetensors.lock")
        try:
            import fcntl
        except ImportError:
            lock_path.touch()  # Windows uses the existence-based fallback.
        else:
            lock_handle = lock_path.open("wb")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

    store = ReferenceFeatureStore({
        "run_id": "run", "ray": {"address": "unused"},
        "expected_feature": {"algorithm": "DFLASH", "target_model_path": "target"},
    })
    loader = TQFeatureDataLoader(
        store, batch_size=1, rank=0, world_size=1, scheduled_commands=commands,
        prepare_batch=partial(store.prepare_batch, max_window=4), io_retry_max_attempts=1,
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    source = ReferenceBatchPrefetchIterator(loader, 4, workers=2,
                                      prepare_options={"max_window": 4}, close_timeout=30)
    source._test_lock = lock_handle
    return source, loader, paths


def test_process_workers_prepare_different_batches_and_keep_fifo(tmp_path, monkeypatch):
    import time

    source, loader, paths = _process_source(tmp_path, monkeypatch, locked=True)
    lock = paths[0].with_suffix(".safetensors.lock")
    try:
        # Batch 0 cannot finish while locked. Batch 1 must already finish in
        # another real spawn process; a serial batch pipeline cannot pass this.
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            with source._condition:
                finished_second = any(batch.plan_id == 1 and result.ready()
                                      for batch, result in source._pending)
            if finished_second:
                break
            time.sleep(0.02)
        assert finished_second, source._error
        assert lock.exists() and paths[0].exists() and not paths[1].exists()
        if source._test_lock is not None:
            source._test_lock.close()
        lock.unlink()
        batches = [next(source), next(source)]
        assert [batch.plan_id for batch in batches] == [0, 1]
        for batch in batches:
            assert batch.prepared_batch["input_ids"].tolist() == [[0, 1, 2, 3]]
            assert batch.prepared_batch["hidden_states"].device.type == "cpu"
            assert batch.batch_prepare_seconds > 0
        with pytest.raises(StopIteration):
            next(source)
        assert not any(path.exists() for path in paths)
        for batch in batches:
            assert batch.cleanup_after_load
            loader.clear_completed_batch(batch.global_keys, cleanup_after_load=True)
        assert not any(path.exists() for path in paths)
    finally:
        if source._test_lock is not None:
            source._test_lock.close()
        lock.unlink(missing_ok=True)
        source.close()
    assert not source._thread.is_alive() and not source._collector.is_alive()
    assert not source._pin_thread.is_alive()


def test_process_worker_error_retains_pending_metadata(tmp_path, monkeypatch):
    source, loader, paths = _process_source(tmp_path, monkeypatch, missing=True)
    try:
        with pytest.raises(FileNotFoundError):
            next(source)
        assert len(loader._cleanup_entries) == 2
    finally:
        source.close()


def test_process_prefetch_close_cancels_locked_untrained_batches(tmp_path, monkeypatch):
    source, _, paths = _process_source(tmp_path, monkeypatch, locked=True)
    try:
        source.close()
        assert paths[0].exists()
        assert paths[0].with_suffix(".safetensors.lock").exists()
        assert not source._collector.is_alive() and not source._thread.is_alive()
    finally:
        if source._test_lock is not None:
            source._test_lock.close()


@pytest.mark.parametrize("fail_pin", [False, True])
def test_process_prefetch_rank_control_order_and_failure_agreement(tmp_path, monkeypatch, fail_pin):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator

    barrier = threading.Barrier(2, timeout=30)
    slots = [None, None]

    def exchange(rank, kind, value):
        slots[rank] = (kind, value)
        barrier.wait()
        assert slots[0][0] == slots[1][0], "Rank control collectives diverged"
        result = (slots[0][1] if kind == "broadcast" else
                  [max(a, b) for a, b in zip(slots[0][1], slots[1][1])])
        barrier.wait()
        return result

    class Loader(TQFeatureDataLoader):
        def _broadcast_command(self, command):
            return exchange(self.rank, "broadcast", command)

    class Source(ReferenceBatchPrefetchIterator):
        def _sync(self, values):
            return exchange(self.loader.rank, "reduce", values)

    def pin(tensor):
        if fail_pin and threading.current_thread().name == "file-pin-0":
            raise RuntimeError("injected process-pipeline pin failure")
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    assignments, paths = [], []
    for rank in range(2):
        folder = tmp_path / str(rank)
        folder.mkdir()
        entry, path = _entry(folder)
        meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", f"sample-{rank}", rank)
        assignments.append([{"key": make_sample_key(meta),
                             "tag": dict(entry.tag, sample_id=meta.sample_id, sequence_no=rank)}])
        paths.append(path)
    commands = Queue()
    commands.put({"kind": "batch", "plan_id": 0, "assignments": assignments,
                  "global_keys": [items[0]["key"] for items in assignments],
                  "global_sequence_nos": [0, 1]})
    commands.put({"kind": "stop"})
    sources = []
    for rank in range(2):
        store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"},
                                           "expected_feature": {"algorithm": "DFLASH"}})
        loader = Loader(store, batch_size=1, rank=rank, world_size=2,
                        scheduled_commands=commands, prepare_batch=store.prepare_batch)
        sources.append(Source(loader, 2, workers=1, close_timeout=30))
    main_barrier = threading.Barrier(2, timeout=30)
    failed = [False, False]
    trained = []

    def consume(rank):
        try:
            try:
                batch = next(sources[rank])
                assert batch.plan_id == 0
            except RuntimeError:
                failed[rank] = True
            main_barrier.wait()  # Model the existing pre-forward rank agreement.
            if not any(failed):
                trained.append(rank)
        finally:
            sources[rank].close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(consume, range(2)))
    assert trained == [] if fail_pin else sorted(trained) == [0, 1]
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("independent_pin", [False, True])
def test_scheduler_assignment_drives_file_prefetch_and_cleanup(tmp_path, monkeypatch, independent_pin):
    entry, path = _entry(tmp_path)
    store = TQFeatureStore({
        "run_id": "run", "ray": {"address": "unused"},
        "expected_feature": {"algorithm": "DFLASH", "target_model_path": "target"},
    })
    store._connected = True
    store.connect = lambda: None
    store.owner_ready = lambda: True
    cleared = []
    store.clear_many = lambda keys: cleared.extend(keys)
    store.get_many = lambda entries: pytest.fail("File references must not fetch TQ tensor fields")
    commands = Queue()
    commands.put({"kind": "batch", "plan_id": 7, "global_keys": [entry.key],
                  "global_sequence_nos": [0], "assignments": [[{"key": entry.key, "tag": entry.tag}]]})
    commands.put({"kind": "stop"})
    loader = TQFeatureDataLoader(
        store, batch_size=1, rank=0, world_size=1, scheduled_commands=commands,
        prepare_batch=partial(store.prepare_batch, pin_memory=False),
    )
    if independent_pin:
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    prefetched = TQPrefetchIterator(loader, 2, pin_memory=independent_pin, prepare_workers=2)
    batch = next(prefetched)
    assert batch.plan_id == 7 and batch.local_samples == []
    assert batch.prepared_batch["hidden_source"] == "standalone_vllm_reference"
    assert batch.prepared_batch["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert path.exists() and cleared == []
    loader.clear_completed_batch(batch.global_keys)
    assert cleared == [entry.key] and not path.exists()
    with pytest.raises(StopIteration):
        next(prefetched)
    prefetched.close()


def test_tq_clear_failure_preserves_file(tmp_path):
    entry, path = _entry(tmp_path)
    class Store:
        def clear_many(self, keys):
            raise ConnectionError("TQ down")
    with pytest.raises(ConnectionError):
        cleanup_standalone_samples(Store(), [entry], attempts=1)
    assert path.exists()


def test_file_retry_never_republishes_or_clears_tq_twice(tmp_path, monkeypatch):
    entry, path = _entry(tmp_path)
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
    releases = []
    original = FileHiddenStatesStore.release
    def release(self, reference):
        releases.append(reference)
        if len(releases) == 1:
            raise OSError("temporary filesystem error")
        original(self, reference)
    monkeypatch.setattr(FileHiddenStatesStore, "release", release)
    cleared = []
    from types import SimpleNamespace
    store = SimpleNamespace(clear_many=lambda keys: cleared.append(keys))
    assert cleanup_standalone_samples(store, [entry]) == []
    assert len(cleared) == 1 and len(releases) == 2 and not path.exists()
    cleanup_standalone_samples(store, [entry])
    assert not path.exists()


def test_untrained_prefetched_files_survive_shutdown(tmp_path):
    entry, path = _entry(tmp_path)
    # Preparing CPU tensors does not confer permission to delete source data.
    store = TQFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store._connected = True
    store.prepare_batch([entry], pin_memory=False)
    assert path.exists()


def test_ray_reference_adapter_prepares_and_cleans_without_any_tq_calls(tmp_path, monkeypatch):
    import verl_speco.trainer.tq_feature_store as stores

    def forbidden(*args, **kwargs):
        pytest.fail("File adapter accessed TransferQueue")

    for name in ("configure_transfer_queue", "connect_ray_cluster", "connect_transfer_queue_client",
                 "list_samples", "get_samples", "clear_samples", "close_transfer_queue_client"):
        monkeypatch.setattr(stores, name, forbidden)
    entry, path = _entry(tmp_path)
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.connect()
    assert store.owner_ready()
    batch = store.prepare_batch([entry], pin_memory=False)
    assert batch["input_ids"].tolist() == [[0, 1, 2, 3]]
    assert path.exists()
    cleanup_standalone_samples(store, [entry])
    assert not path.exists()
    store.close()


@pytest.mark.parametrize("training_fails", [False, True])
def test_driver_receives_metadata_once_before_producer_completion(tmp_path, monkeypatch, training_fails):
    from copy import deepcopy
    from types import SimpleNamespace

    from test_standalone_ray_runtime import (
        _commands, _runtime_config, _FakeRay, _ProducerHandle, _RecordingQueue, _WorkerGroup,
    )
    from verl_speco.standalone_ray_runtime import StandaloneRayTrainer

    config = deepcopy(_runtime_config())
    training = config.actor_rollout_ref.rollout.drafter.training
    # The fixture's get closures refer to the original; replace with local access.
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 1
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    entry, path = _entry(tmp_path)
    batch = {"kind": "samples_ready", "batch_id": 0,
             "entries": [{"key": entry.key, "tag": entry.tag}]}
    commands = _RecordingQueue()
    events = _RecordingQueue(events=[
        batch, dict(batch),  # Ambiguous send retry must not retrain the sample.
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "producer_done"},
    ])
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.connect()
    sent_commands = []
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original_weight = model.weight.detach().clone()

    class Ray(_FakeRay):
        def wait(self, refs, num_returns, timeout=0):
            # Actor return can be observed before its final queued events.
            producer = next((ref for ref in refs if ref.name == "producer"), None)
            if producer is not None:
                return [producer], []
            if commands.items and commands.items[0]["kind"] == "batch":
                sent_commands.append(commands.items[0])
                loader = TQFeatureDataLoader(
                    store, batch_size=1, rank=0, world_size=1, scheduled_commands=commands,
                    prepare_batch=partial(store.prepare_batch, pin_memory=False),
                )
                local = next(iter(loader))
                data = local.prepared_batch
                assert path.exists(), "Prefetch must not delete training input"
                if training_fails:
                    raise RuntimeError("injected training failure")
                prediction = model(data["hidden_states"].float()).squeeze(-1)
                loss = ((prediction - data["input_ids"].float()).square() * data["loss_mask"]).mean()
                loss.backward()
                optimizer.step()
                assert not torch.equal(model.weight, original_weight)
                assert path.exists(), "Source must survive until optimizer success"
                loader.clear_completed_batch(local.global_keys)
                events.put({"kind": "training_completed", "plan_id": local.plan_id,
                            "keys": local.global_keys, "successful": True})
                events.put({"kind": "training_metrics", "step": 1, "metrics": {"dspark/loss": loss.item()}})
            if commands.items and commands.items[-1]["kind"] == "stop":
                if events.items:
                    return [], refs
                return refs[:1], refs[1:]
            return [], refs

        def get(self, ref):
            if ref.name.startswith("consumer"):
                assert not events.items, "Consumer cannot exit until its final event is drained"
            return super().get(ref)

    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=Ray(_ProducerHandle()))
    trainer.producer = _ProducerHandle()
    trainer.consumer_wg = _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    # Use the real adapter: any accidental list_ready() raises.
    trainer.feature_store_factory = lambda cfg: store
    if training_fails:
        with pytest.raises(RuntimeError, match="injected training failure"):
            trainer.run()
        assert not path.exists()
        return
    assert trainer.run() == 0
    submitted = sent_commands
    assert len(submitted) == 1 and submitted[0]["global_keys"] == [entry.key]
    assert submitted[0]["assignments"][0][0]["tag"] == entry.tag
    assert not path.exists()
