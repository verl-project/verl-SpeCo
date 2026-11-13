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
    if delete_fails > 2:
        with pytest.raises(RuntimeError, match="Failed to release"):
            worker._prepare_batch([entry])
        assert path.exists()
    else:
        batch, _ = worker._prepare_batch([entry])
        assert batch["input_ids"].tolist() == [[0, 1, 2, 3]]
        assert not path.exists()
    assert len(reads) == 2 and len(deletes) == 3


@pytest.mark.parametrize("streaming", [False, True])
def test_partial_batch_read_failure_does_not_delete_successful_sample(tmp_path, monkeypatch, streaming):
    import verl_speco.trainer.reference_batch_prefetch as worker

    entry, path = _entry(tmp_path)
    missing = ReadyEntry(entry.key, dict(entry.tag, hidden_states_path=str(tmp_path / "missing")))
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"},
                                       "expected_feature": {"algorithm": "DFLASH"}})
    store.connect()
    monkeypatch.setattr(worker, "_STORE", store)
    monkeypatch.setattr(worker, "_OPTIONS", {"max_window": 4, "streaming": streaming})
    monkeypatch.setattr(worker, "_ATTEMPTS", 1)
    with pytest.raises(FileNotFoundError):
        worker._prepare_batch([entry, missing])
    assert path.exists()


def _process_source(tmp_path, monkeypatch, *, missing=False, locked=False, defer_cleanup=False, streaming=False, max_host_bytes=0, max_batch_host_bytes=0, use_threads=False):
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
        prepare_batch=partial(store.prepare_batch, max_window=2 if streaming else 4, streaming=streaming,
                              max_output_bytes=max_batch_host_bytes), io_retry_max_attempts=1,
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    if use_threads:
        from verl_speco.trainer.tq_sample_source import TQPrefetchIterator
        source = TQPrefetchIterator(loader, 4, pin_memory=True, prepare_workers=2, close_timeout=30,
                                    max_host_bytes=max_host_bytes, max_batch_host_bytes=max_batch_host_bytes)
    else:
        source = ReferenceBatchPrefetchIterator(loader, 4, workers=2,
                                          prepare_options={"max_window": 2 if streaming else 4, "streaming": streaming,
                                                           "max_output_bytes": max_batch_host_bytes}, close_timeout=30,
                                          defer_cleanup=defer_cleanup, max_host_bytes=max_host_bytes,
                                          max_batch_host_bytes=max_batch_host_bytes)
    source._test_lock = lock_handle
    return source, loader, paths


@pytest.mark.parametrize("use_threads", [False, True])
def test_process_prefetch_host_budget_bounds_depth_and_preserves_order(tmp_path, monkeypatch, use_threads):
    source, loader, paths = _process_source(tmp_path, monkeypatch, defer_cleanup=True,
                                            max_host_bytes=8192, max_batch_host_bytes=4096, use_threads=use_threads)
    try:
        assert source.depth == 2
        first, second = next(source), next(source)
        assert first.plan_id == 0 and second.plan_id == 1
        assert all(path.exists() for path in paths)
        loader.clear_completed_batch(first.global_keys)
        loader.clear_completed_batch(second.global_keys)
    finally:
        source.close()


@pytest.mark.parametrize("use_threads", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_process_prefetch_oversize_batch_retains_reference(tmp_path, monkeypatch, use_threads, streaming):
    source, _, paths = _process_source(tmp_path, monkeypatch, defer_cleanup=True,
                                        max_host_bytes=1, max_batch_host_bytes=1, use_threads=use_threads,
                                        streaming=streaming)
    try:
        with pytest.raises(RuntimeError if use_threads else ValueError) as error:
            next(source)
        detail = error.value.__cause__ if use_threads else error.value
        assert isinstance(detail, ValueError) and "consumer_prefetch_batch_bytes" in str(detail)
        assert ("Planned reference batch" in str(detail)) == streaming
        assert all(path.exists() for path in paths)
    finally:
        source.close()


@pytest.mark.parametrize("total,per_batch", [(4096, 0), (0, 4096), (4095, 4096), (-1, 1), (True, 1)])
def test_process_prefetch_rejects_invalid_host_reservations(total, per_batch):
    from verl_speco.trainer.reference_batch_prefetch import ReferenceBatchPrefetchIterator
    with pytest.raises(ValueError, match="Prefetch host"):
        ReferenceBatchPrefetchIterator(None, 2, max_host_bytes=total, max_batch_host_bytes=per_batch)


def test_prefetch_host_estimate_counts_shared_storage_and_pin_copies():
    from verl_speco.trainer.reference_batch_prefetch import _prepared_host_bytes
    storage = torch.arange(8, dtype=torch.float32)
    # The two views retain one original storage but each is pinned separately.
    assert _prepared_host_bytes({"left": storage[:4], "right": storage[4:]}) == 96


@pytest.mark.parametrize("streaming", [False, True])
def test_deferred_prefetch_retains_payload_until_batch_completion(tmp_path, monkeypatch, streaming):
    source, loader, paths = _process_source(tmp_path, monkeypatch, defer_cleanup=True, streaming=streaming)
    try:
        first = next(source)
        second = next(source)
        assert all(path.exists() for path in paths)
        assert first.prepared_batch["input_ids"].shape[1] == (2 if streaming else 4)
        assert not first.cleanup_after_load and not second.cleanup_after_load
        loader.clear_completed_batch(first.global_keys)
        assert not paths[0].exists() and paths[1].exists()
        loader.clear_completed_batch(second.global_keys)
        assert not paths[1].exists()
    finally:
        source.close()


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
            for key in ("loader_worker_result_wait_time", "loader_ipc_collect_time",
                        "loader_pin_queue_time", "loader_pin_time", "loader_ready_queue_time"):
                assert batch.delivery_times[key] >= 0
            assert not any(key.startswith("_") for key in batch.delivery_times)
            assert "_speco_worker_finished_at" not in batch.prepared_batch
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
@pytest.mark.parametrize("streaming", [False, True])
def test_scheduler_assignment_drives_file_prefetch_and_cleanup(tmp_path, monkeypatch, independent_pin, streaming):
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
        prepare_batch=partial(store.prepare_batch, pin_memory=False, streaming=streaming),
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


def test_cleanup_rejects_outside_shared_root_before_clearing(tmp_path):
    from types import SimpleNamespace
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
    entry, path = _entry(tmp_path)
    root = tmp_path / "shared"
    root.mkdir()
    cleared = []
    store = SimpleNamespace(hidden_states_store=FileHiddenStatesStore(str(root)),
                            clear_many=lambda keys: cleared.extend(keys))
    with pytest.raises(ValueError, match="outside the configured shared file root"):
        cleanup_standalone_samples(store, [entry], attempts=1)
    assert path.exists() and not cleared


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


def test_reference_contract_checks_identity_without_reading_payload(tmp_path):
    from copy import deepcopy
    entry, path = _entry(tmp_path)
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.connect()
    path.unlink()
    assert store.reference_contract([entry]).algorithm == "DFLASH"
    with pytest.raises(ValueError, match="identity mismatch"):
        store.reference_contract([ReadyEntry("wrong-key", entry.tag)])
    changed = deepcopy(entry.tag)
    changed["feature_contract"]["target_model_revision"] = "other"
    with pytest.raises(ValueError, match="contract mismatch"):
        store.reference_contract([entry, ReadyEntry(entry.key, changed)])
    store.close()


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


@pytest.mark.parametrize("credit_limit", [0, 1, 2, 3])
@pytest.mark.parametrize("incarnation_conflict", ["none", "sample", "done"])
@pytest.mark.parametrize("training_fails,stale_receipt,fragmented", [
    (False, False, False), (True, False, False), (False, True, False), (False, False, True),
])
def test_driver_receives_metadata_once_before_producer_completion(tmp_path, monkeypatch, training_fails, stale_receipt, fragmented, credit_limit, incarnation_conflict):
    from copy import deepcopy
    from types import SimpleNamespace

    from test_standalone_ray_runtime import (
        _commands, _runtime_config, _FakeRay, _ProducerHandle, _RecordingQueue, _WorkerGroup,
    )
    from verl_speco.standalone_ray_runtime import StandaloneRayTrainer

    if incarnation_conflict != "none" and (not credit_limit or training_fails or stale_receipt or fragmented):
        pytest.skip("Incarnation conflict needs one successful credit-enabled case")
    config = deepcopy(_runtime_config())
    config.speco.standalone_tq_producer.max_published_references = int(credit_limit == 1)
    config.speco.standalone_tq_producer.max_reference_execution_slots = int(credit_limit == 2)
    config.speco.standalone_tq_producer.max_published_file_tensor_bytes = int(credit_limit == 3)
    runtime = config.speco.draft_training
    if fragmented:
        runtime.scheduler = {"reference_command_frame_bytes": 1024}
        runtime.get = lambda key, default=None: getattr(runtime, key, default)
    training = config.actor_rollout_ref.rollout.drafter.training
    # The fixture's get closures refer to the original; replace with local access.
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 1
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    entry, path = _entry(tmp_path)
    if fragmented:
        entry.tag["request"]["source_metadata"]["padding"] = "x" * 4000
    batch = {"kind": "samples_ready", "batch_id": 0,
             "entries": [{"key": entry.key, "tag": entry.tag}]}
    if credit_limit:
        batch["producer_incarnation"] = "f" * 32
    repeated_batch = dict(batch)
    if incarnation_conflict == "sample":
        batch["producer_incarnation"] = "e" * 32
    commands = _RecordingQueue()
    events = _RecordingQueue(events=[
        batch, repeated_batch,  # Ambiguous send retry must not retrain the sample.
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "producer_done", **({"producer_incarnation": ("e" if incarnation_conflict == "done" else "f") * 32} if credit_limit else {})},
    ])
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.connect()
    sent_commands = []
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original_weight = model.weight.detach().clone()

    class Ray(_FakeRay):
        def wait(self, refs, num_returns, timeout=0):
            if refs[0].name == "stop":
                return refs[:1], refs[1:]
            # Actor return can be observed before its final queued events.
            producer = next((ref for ref in refs if ref.name == "producer"), None)
            if producer is not None:
                return [producer], []
            if commands.items and commands.items[0]["kind"] in {"batch", "batch_fragment"}:
                sent_commands.append(commands.items[0])
                loader = TQFeatureDataLoader(
                    store, batch_size=1, rank=0, world_size=1, scheduled_commands=commands,
                    reference_command_frame_bytes=1024 if fragmented else 4 * 1024 * 1024,
                    prepare_batch=partial(store.prepare_batch, pin_memory=False),
                )
                local = next(iter(loader))
                data = local.prepared_batch
                events.put({"kind": "training_materialized", "plan_id": local.plan_id,
                            "receipt_token": local.receipt_token})
                assert path.exists(), "Prefetch must not delete training input"
                if training_fails:
                    raise RuntimeError("injected training failure")
                prediction = model(data["hidden_states"].float()).squeeze(-1)
                loss = ((prediction - data["input_ids"].float()).square() * data["loss_mask"]).mean()
                loss.backward()
                optimizer.step()
                assert not torch.equal(model.weight, original_weight)
                assert path.exists(), "Source must survive until optimizer success"
                events.put({"kind": "training_applied", "plan_id": local.plan_id,
                            "receipt_token": local.receipt_token})
                loader.clear_completed_batch(local.global_keys)
                events.put({"kind": "training_completed", "plan_id": local.plan_id,
                            "receipt_token": "stale-driver" if stale_receipt else local.receipt_token,
                            "keys": local.global_keys, "successful": True})
                events.put({"kind": "training_metrics", "step": 1, "metrics": {"dspark/loss": loss.item()}})
            if commands.items and commands.items[-1]["kind"] == "stop":
                if events.items:
                    return [], refs
                return refs[:1], refs[1:]
            return [], refs

        def get(self, ref, timeout=None):
            if ref.name == "start" and credit_limit:
                return {"started": True, "producer_incarnation": "f" * 32}
            if ref.name.startswith("consumer"):
                assert not events.items, "Consumer cannot exit until its final event is drained"
            return super().get(ref)

    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    tracking = []
    monkeypatch.setattr("verl_speco.standalone_ray_runtime._log_standalone_tracking_metrics",
                        lambda trackers, metrics, **kwargs: tracking.append(dict(metrics)))
    trainer = StandaloneRayTrainer(_commands(), ray_module=Ray(_ProducerHandle()))
    trainer.producer = _ProducerHandle()
    credit_releases = []
    registered = []
    class Register:
        def remote(self, incarnation, plan_id, token, keys):
            assert incarnation == "f" * 32
            assert path.exists(), "Dispatch must be registered before training"
            registered.append((plan_id, token, keys))
            from test_standalone_ray_runtime import _Ref
            return _Ref("register")
    trainer.producer.register_dispatch = Register()
    class Release:
        def remote(self, incarnation, plan_id, token, keys):
            assert incarnation == "f" * 32
            assert not path.exists(), "Credits can only return after successful cleanup"
            assert (plan_id, token, keys) in registered
            credit_releases.append((plan_id, token, keys))
            from test_standalone_ray_runtime import _Ref
            return _Ref("release")
    trainer.producer.release_references = Release()
    trainer.consumer_wg = _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    # Use the real adapter: any accidental list_ready() raises.
    trainer.feature_store_factory = lambda cfg: store
    if incarnation_conflict != "none":
        with pytest.raises(ValueError, match="stale producer incarnation"):
            trainer.run()
        if incarnation_conflict == "sample":
            assert path.exists()
        assert not credit_releases
        return
    if training_fails:
        with pytest.raises(RuntimeError, match="injected training failure"):
            trainer.run()
        # A failed rank may still have an unfenced reader; preserve its input.
        assert path.exists()
        return
    if stale_receipt:
        with pytest.raises(RuntimeError, match="FIFO plan"):
            trainer.run()
        assert not any(item.get("kind") == "stop" for item in commands.items)
        return
    assert trainer.run() == 0
    assert len(credit_releases) == bool(credit_limit)
    if credit_limit:
        assert credit_releases[0][0] == 0 and credit_releases[0][2] == [entry.key]
    assert len(tracking) == 1
    assert tracking[0]["scheduler/completed_samples"] == 1.0
    assert tracking[0]["scheduler/pending_plans"] == 0.0
    assert tracking[0]["scheduler/completed_samples_per_second"] > 0
    submitted = sent_commands
    assert len(submitted) == 1
    if fragmented:
        assert submitted[0]["kind"] == "batch_fragment"
    else:
        assert submitted[0]["global_keys"] == [entry.key]
        assert submitted[0]["assignments"][0][0]["tag"] == entry.tag
    assert not path.exists()


@pytest.mark.parametrize("matching", [False, True, None])
def test_driver_waits_for_matching_input_identity_before_dispatch(tmp_path, monkeypatch, matching):
    from copy import deepcopy
    from types import SimpleNamespace
    from test_standalone_ray_runtime import _commands, _runtime_config, _FakeRay, _ProducerHandle, _RecordingQueue, _WorkerGroup
    from verl_speco.standalone_ray_runtime import StandaloneRayTrainer
    config = deepcopy(_runtime_config())
    config.speco.standalone_tq_producer.input_path = "bound-input"
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 1
    training.transfer_queue = {"consumer_prefetch_batches": 0}
    source_identity = {"kind": "file", "sha256": "a" * 64, "size_bytes": 4}
    consumer_identity = dict(source_identity)
    if matching is False:
        consumer_identity["sha256"] = "b" * 64
    entry, path = _entry(tmp_path)
    commands = _RecordingQueue()
    def receive(event):
        if event["kind"] == "consumer_ready":
            assert not [item for item in commands.items if item["kind"] == "batch"]
    events = _RecordingQueue(events=[
        {"kind": "samples_ready", "batch_id": 0, "entries": [{"key": entry.key, "tag": entry.tag}]},
        {"kind": "consumer_ready", "initial_optimizer_step": 0,
         **({"input_identity_unavailable": True} if matching is None else {"input_identity": consumer_identity})},
        {"kind": "consumer_failed", "rank": 0, "error": "stop test"},
    ], on_event=receive)
    class Ray(_FakeRay):
        def get(self, ref, timeout=None):
            if ref.name == "start":
                return {"started": True, "input_identity": source_identity}
            return super().get(ref)
        def wait(self, refs, **kwargs):
            if refs[0].name == "stop":
                return refs[:1], refs[1:]
            return [], refs
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    trainer = StandaloneRayTrainer(_commands(), ray_module=Ray(_ProducerHandle()), feature_store_factory=lambda _: store)
    trainer.producer, trainer.consumer_wg = trainer.ray.producer, _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    with pytest.raises(ValueError if matching is False else RuntimeError,
                       match="input content differs" if matching is False else "stop test"):
        trainer.run()
    batches = [item for item in commands.items if item["kind"] == "batch"]
    assert bool(batches) == (matching is not False)
    if batches:
        assert batches[0]["input_identity"] == source_identity
    assert path.exists()


@pytest.mark.parametrize("backend", ["file", "mooncake"])
def test_cleanup_discards_invalid_training_values_using_valid_reference(tmp_path, backend):
    from types import SimpleNamespace
    from verl_speco.transport.drafter_sample_protocol import validate_reference_tag

    entry, path = _entry(tmp_path)
    entry.tag["request"]["loss_mask"][1] = float("nan")
    entry.tag["request"]["feature_positions"][-1] = 99
    with pytest.raises(ValueError):
        validate_reference_tag(entry.tag)
    if backend == "file":
        store = SimpleNamespace(uses_transfer_queue=False)
        assert cleanup_standalone_samples(store, [entry]) == []
        assert not path.exists()
    else:
        entry.tag["backend"] = "mooncake"
        entry.tag.pop("hidden_states_path")
        entry.tag["handle"] = "discarded-handle"
        released = []
        store = SimpleNamespace(
            uses_transfer_queue=False,
            hidden_states_store=SimpleNamespace(backend="mooncake", release=released.append),
        )
        assert cleanup_standalone_samples(store, [entry]) == []
        assert released == ["discarded-handle"]


@pytest.mark.parametrize("failure", ["ambiguous", "backend_mismatch"])
def test_cleanup_checks_reference_before_deleting(tmp_path, failure):
    from types import SimpleNamespace

    entry, path = _entry(tmp_path)
    released = []
    store = SimpleNamespace(
        uses_transfer_queue=False,
        hidden_states_store=SimpleNamespace(backend="file", release=released.append),
    )
    if failure == "ambiguous":
        entry.tag["handle"] = "wrong"
    else:
        store.hidden_states_store.backend = "mooncake"
    with pytest.raises(ValueError):
        cleanup_standalone_samples(store, [entry])
    assert released == [] and path.exists()


def test_failed_reference_release_is_retried_without_deleting_successes_twice(tmp_path):
    from types import SimpleNamespace
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first, _ = _entry(first_dir)
    second, _ = _entry(second_dir)
    # Distinct sample identities share one batch, with one transient release error.
    second_meta = SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", "other", 1)
    second_tag = dict(second.tag)
    second_tag.update(second_meta.to_dict())
    second = ReadyEntry(make_sample_key(second_meta), second_tag)
    for entry, handle in ((first, "h0"), (second, "h1")):
        entry.tag["backend"] = "mooncake"
        entry.tag.pop("hidden_states_path")
        entry.tag["handle"] = handle
    calls = []
    def release(handle):
        calls.append(handle)
        if handle == "h1" and calls.count("h1") == 1:
            raise OSError("transient release failure")
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.hidden_states_store = SimpleNamespace(backend="mooncake", release=release)
    loader = TQFeatureDataLoader(store, batch_size=2, rank=0, world_size=1,
                                 scheduled_commands=Queue(), prepare_batch=lambda entries: {})
    loader._cleanup_entries.update({first.key: first, second.key: second})
    keys = [first.key, second.key]
    with pytest.raises(RuntimeError, match="Failed to release 1 completed references"):
        loader.clear_completed_batch(keys)
    assert first.key not in loader._cleanup_entries and second.key in loader._cleanup_entries
    loader.clear_completed_batch(keys)
    assert calls == ["h0", "h1", "h1"]
    assert not loader._cleanup_entries and not loader._released_cleanup_keys


def test_exhausted_release_failure_never_sends_training_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import verl_speco.trainer.draft_training_loop as loop
    entry, _ = _entry(tmp_path)
    calls = []
    def release(reference):
        calls.append(reference)
        raise OSError("store unavailable")
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.hidden_states_store = SimpleNamespace(backend="file", release=release)
    loader = TQFeatureDataLoader(store, batch_size=1, rank=0, world_size=1,
                                 scheduled_commands=Queue(), prepare_batch=lambda entries: {})
    loader._cleanup_entries[entry.key] = entry
    events = Queue()
    monkeypatch.setattr(loop, "_TQ_CLEAR_RETRY_BACKOFF_SECONDS", 0)
    with pytest.raises(RuntimeError, match="Failed to release"):
        loop._clear_tq_batch_across_ranks(
            loader, [entry.key], rank=0, device=torch.device("cpu"),
            training_events=events, completion_event={"kind": "training_completed"},
            synchronize=False,
        )
    assert events.empty()
    assert entry.key in loader._cleanup_entries
    assert len(calls) == loop._TQ_CLEAR_MAX_ATTEMPTS


@pytest.mark.parametrize("algorithm", ["DFLASH", "EAGLE3"])
def test_padding_rank_reads_template_without_credit_and_waits_for_shared_cleanup(tmp_path, algorithm):
    from functools import partial
    from verl_speco.trainer.draft_training_loop import _zero_standalone_padding_batch
    entry, path = _entry(tmp_path)
    if algorithm == "EAGLE3":
        save_file({"token_ids": torch.arange(4), "hidden_states": torch.ones(4, 3, 2)}, path)
        entry.tag["feature_contract"]["algorithm"] = algorithm
        entry.tag["feature_contract"]["hidden_states_layout"] = "eagle3_aux_plus_last"
    wire = {"key": entry.key, "tag": entry.tag}
    command = {"kind": "batch", "plan_id": 0, "global_keys": [entry.key],
               "global_sequence_nos": [0], "assignments": [[wire], []],
               "padding_ranks": [1], "padding_reference": wire}
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"},
                                   "expected_feature": {"algorithm": algorithm}})
    prepare = partial(store.prepare_batch, max_window=4, pin_memory=False)
    padding = TQFeatureDataLoader(store, batch_size=2, rank=1, world_size=2,
                                  scheduled_commands=Queue(), prepare_batch=prepare)
    padding._broadcast_command = lambda _: command
    local = next(iter(padding))
    assert local.is_padding_rank and local.has_padding_ranks and local.local_keys == []
    assert path.exists() and local.prepared_batch["loss_mask"].sum() > 0
    masked = _zero_standalone_padding_batch(local.prepared_batch)
    assert masked["loss_mask"].sum() == 0
    if "label_mask" in masked:
        assert masked["label_mask"].sum() == 0
    counts = masked["_speco_token_counts"]
    assert counts["padding_tokens"] == masked["input_ids"].numel()
    assert all(value == 0 for key, value in counts.items() if key != "padding_tokens")
    assert masked["_speco_sample_count"] == 0 and local.prepared_batch["loss_mask"].sum() > 0
    real_queue = Queue()
    real_queue.put(command)
    real = TQFeatureDataLoader(store, batch_size=2, rank=0, world_size=2,
                               scheduled_commands=real_queue, prepare_batch=prepare)
    real._broadcast_command = lambda value: value
    real_batch = next(iter(real))
    assert not real_batch.is_padding_rank and real_batch.has_padding_ranks
    assert real_batch.local_keys == real_batch.global_keys == [entry.key]
    assert path.exists()
    real.clear_completed_batch(real_batch.global_keys)
    assert not path.exists()


@pytest.mark.parametrize("fault", ["duplicate", "missing", "foreign", "sequence", "missing_sequences", "group"])
def test_reference_assignments_match_global_manifest_before_fetch(tmp_path, fault):
    from copy import deepcopy
    entry, path = _entry(tmp_path)
    wire = {"key": entry.key, "tag": entry.tag}
    command = {"kind": "batch", "plan_id": 0, "global_keys": [entry.key],
               "global_sequence_nos": [0], "assignments": [[wire], []]}
    command = deepcopy(command)
    if fault == "duplicate":
        command["assignments"][1].append(deepcopy(wire))
    elif fault == "missing":
        command["assignments"][0].clear()
    elif fault == "foreign":
        command["assignments"][0][0]["key"] = "foreign"
    elif fault == "sequence":
        command["global_sequence_nos"] = [1]
    elif fault == "missing_sequences":
        del command["global_sequence_nos"]
    else:
        command["assignments"][1] = None
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    loader = TQFeatureDataLoader(store, batch_size=2, rank=1, world_size=2,
                                 scheduled_commands=Queue(), prepare_batch=lambda entries: pytest.fail("unexpected fetch"))
    loader._broadcast_command = lambda _: command
    with pytest.raises(RuntimeError, match="assignment"):
        next(iter(loader))
    assert path.exists()


def test_node_owner_receives_local_assignments_including_padding_template(tmp_path):
    entry, path = _entry(tmp_path)
    wire = {"key": entry.key, "tag": entry.tag}
    command = {"kind": "batch", "plan_id": 0, "global_keys": [entry.key],
               "global_sequence_nos": [0], "assignments": [[wire], []],
               "padding_ranks": [1], "padding_reference": wire}
    queue = Queue()
    queue.put(command)
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    loader = TQFeatureDataLoader(store, batch_size=2, rank=0, world_size=2,
                                 scheduled_commands=queue, prepare_batch=lambda entries: {},
                                 node_ranks_per_host=2)
    loader._broadcast_command = lambda value: value
    batch = next(loader.iter_assignments())
    assert {rank: [item.key for item in entries] for rank, entries in batch.node_file_entries.items()} == {
        0: [entry.key], 1: [entry.key]}
    assert path.exists()


def _gloo_padding_pipeline_worker(rank, port, command, proceed, ready, invalid_mask, algorithm):
    import asyncio
    from contextlib import nullcontext
    from datetime import timedelta
    from types import SimpleNamespace
    import torch.distributed as dist
    import verl_speco.trainer.base_trainer as trainer_module
    from omegaconf import OmegaConf
    from verl_speco.backends.eagle3_trainer_backend import Eagle3TrainerBackend, _Eagle3VllmTargetHead
    from verl_speco.models.auto import AutoDraftModelConfig
    from verl_speco.models.eagle.llama_eagle import LlamaForCausalLMEagle3
    from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend, DFlashTrainingModel
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
    from verl_speco.models.dflash import DFlashConfig, DFlashDraftModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.draft_training_loop import (
        _ack_reference_materialization_across_ranks, _clear_tq_batch_across_ranks,
        _next_batch_across_ranks, _reference_training_step_across_ranks,
    )

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        trainer_module.device_name = "cpu"
        torch.manual_seed(21)
        common = dict(hidden_size=2, intermediate_size=4, num_hidden_layers=1,
                      num_attention_heads=1, num_key_value_heads=1, head_dim=2,
                      vocab_size=16, num_target_layers=4, num_context_layers=2,
                      target_hidden_size=2, target_num_hidden_layers=4,
                      target_layer_ids=[0, 1], mask_token_id=15)
        if algorithm == "DSPARK":
            cfg = DSparkConfig(**common, markov_rank=2, markov_head_type="gated")
            model = DSparkTrainingModel(draft_model=DSparkDraftModel(cfg), block_size=2,
                                        num_anchors=2, l1_loss_alpha=0.0)
            backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
        elif algorithm == "DFLASH":
            cfg = DFlashConfig(**common)
            model = DFlashTrainingModel(draft_model=DFlashDraftModel(cfg), block_size=2, num_anchors=2)
            backend = DFlashTrainerBackend.__new__(DFlashTrainerBackend)
        else:
            eagle_config = {"architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama",
                            "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 1,
                            "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 16,
                            "target_hidden_size": 2, "target_layer_ids": [0, 1], "num_aux_hidden_states": 2,
                            "max_position_embeddings": 128, "pad_token_id": 0, "pretraining_tp": 1,
                            "rms_norm_eps": 1e-6, "hidden_act": "silu", "tie_word_embeddings": False}
            cfg = AutoDraftModelConfig._config_mapping["LlamaForCausalLMEagle3"].from_dict(eagle_config)
            model = LlamaForCausalLMEagle3(cfg)
            backend = Eagle3TrainerBackend(OmegaConf.create({"rollout": {"drafter": {"training": {
                "mode": "offline", "feature_store": {"type": "ray_reference"}, "use_logits": False,
                "ttt_length": 1, "transfer_queue": {"expected_feature": {"payload_kind": "vllm_reference"}}}}},
                "model": {"path": "/unused"}}), OmegaConf.create({}))
            backend.target_model = _Eagle3VllmTargetHead(
                SimpleNamespace(fc=torch.nn.Linear(2, 16, bias=False)), torch.nn.LayerNorm(2))
        model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
        if algorithm != "EAGLE3":
            backend.target_lm_head = SimpleNamespace(fc=torch.nn.Linear(2, 16, bias=False))
        store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"},
                                       "expected_feature": {"algorithm": algorithm}})
        command_queue = Queue()
        if rank == 0:
            for message in command if isinstance(command, list) else [command]:
                command_queue.put(message)
        loader = TQFeatureDataLoader(store, batch_size=2, rank=rank, world_size=2,
                                     scheduled_commands=command_queue,
                                     reference_command_frame_bytes=1024 if isinstance(command, list) else 4 * 1024 * 1024,
                                     prepare_batch=partial(store.prepare_batch, max_window=4, pin_memory=False,
                                                           streaming=algorithm == "DSPARK" and not invalid_mask))
        source = iter(loader)
        if invalid_mask and rank == 1:
            from dataclasses import replace
            original_source = source
            def invalid_source():
                item = next(original_source)
                yield replace(item, prepared_batch={key: value for key, value in item.prepared_batch.items()
                                                    if key != "loss_mask"})
            source = invalid_source()
        try:
            local = _next_batch_across_ranks(source, rank=rank, device=torch.device("cpu"), require_prepared=True,
                expected_input_identity={"kind": "file", "sha256": "a" * 64, "size_bytes": 4})
        except RuntimeError:
            if not invalid_mask:
                raise
            ready.put((rank, "fetch_error"))
            return
        if invalid_mask:
            raise AssertionError("Both ranks must stop when a padding batch is invalid")
        assert local.receipt_token == (command[0] if isinstance(command, list) else command)["receipt_token"]
        assert local.input_identity == {"kind": "file", "sha256": "a" * 64, "size_bytes": 4}
        events = Queue()
        _ack_reference_materialization_across_ranks(
            local, rank=rank, device=torch.device("cpu"), events=events,
            config={"io_retry_max_attempts": 1, "io_timeout_seconds": 1},
        )
        if rank == 0:
            assert events.get_nowait() == {"kind": "training_materialized", "plan_id": 0,
                                           "receipt_token": "test-dispatch"}
        batch = local.prepared_batch
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.model, trainer.backend = model, backend
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        trainer.lr_scheduler = None
        trainer.training_steps = trainer.optimizer_steps_total = 0
        trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
        trainer.combine_standalone_metrics = True
        trainer._standalone_validate_grad_graph = True
        trainer._ulysses_group_context = nullcontext
        trainer.record_training_timing = lambda *args: None
        trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
        trainer._is_checkpoint_leader = lambda: False
        def reduce_statistics(values):
            dist.all_reduce(values)
            return values, 2
        trainer._reduce_loss_statistics = reduce_statistics
        trainer._stage_standalone_cpu_batch = lambda value: dict(value)
        async def run_step(value, step, *, raise_on_error):
            assert raise_on_error
            return await trainer._training_step_on_batch(value, step)
        trainer.training_step_from_batch = run_step
        assert asyncio.run(_reference_training_step_across_ranks(trainer, batch, 0, device=torch.device("cpu")))
        ready.put((rank, local.is_padding_rank, int(batch["loss_mask"].sum().item())))
        if rank == 1:
            assert proceed.wait(15)
        _clear_tq_batch_across_ranks(loader, local.global_keys, rank=rank,
            device=torch.device("cpu"), synchronize=False, synchronize_before_cleanup=True,
            training_events=events, completion_event={"kind": "training_completed", "plan_id": local.plan_id,
                "receipt_token": local.receipt_token, "keys": local.global_keys, "successful": True})
        if rank == 0:
            assert events.get_nowait()["kind"] == "training_applied"
            assert events.get_nowait()["kind"] == "training_completed"
        ready.put((rank, "cleared"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("invalid_mask", [False, True])
@pytest.mark.parametrize("algorithm", ["DSPARK", "DFLASH", "EAGLE3"])
def test_padding_tail_real_two_rank_loader_training_and_release(tmp_path, invalid_mask, algorithm, monkeypatch):
    import multiprocessing
    import socket
    entry, path = _entry(tmp_path)
    entry.tag["feature_contract"]["algorithm"] = algorithm
    if algorithm == "EAGLE3":
        monkeypatch.setenv("SPECO_EAGLE3_DISABLE_TORCH_COMPILE", "1")
        save_file({"token_ids": torch.arange(4), "hidden_states": torch.ones(4, 3, 2)}, path)
        entry.tag["feature_contract"]["hidden_states_layout"] = "eagle3_aux_plus_last"
    wire = {"key": entry.key, "tag": entry.tag}
    command = {"kind": "batch", "plan_id": 0, "receipt_token": "test-dispatch", "global_keys": [entry.key],
               "input_identity": {"kind": "file", "sha256": "a" * 64, "size_bytes": 4},
               "global_sequence_nos": [0], "assignments": [[wire], []],
               "padding_ranks": [1], "padding_reference": wire}
    if algorithm == "DSPARK" and not invalid_mask:
        from verl_speco.standalone_ray_runtime import _reference_command_frames
        entry.tag["request"]["source_metadata"]["padding"] = "x" * 4000
        command = list(_reference_command_frames(command, max_bytes=65536, frame_bytes=1024))
        assert len(command) > 1
    context = multiprocessing.get_context("spawn")
    ready, proceed = context.Queue(), context.Event()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [context.Process(target=_gloo_padding_pipeline_worker,
                                 args=(rank, port, command, proceed, ready, invalid_mask, algorithm)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        rows = {ready.get(timeout=45), ready.get(timeout=45)}
        if invalid_mask:
            assert rows == {(0, "fetch_error"), (1, "fetch_error")}
            assert path.exists()
        else:
            assert rows == {(0, False, 2 if algorithm == "EAGLE3" else 3), (1, True, 0)}
            assert path.exists()  # The other rank has not entered the release barrier.
            proceed.set()
        for process in processes:
            process.join(timeout=40)
        assert all(process.exitcode == 0 for process in processes)
        if not invalid_mask:
            assert {ready.get(timeout=5), ready.get(timeout=5)} == {(0, "cleared"), (1, "cleared")}
            assert not path.exists()
    finally:
        proceed.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        ready.close()


@pytest.mark.parametrize("fault", ["early_cleanup", "foreign_template", "unsupported_algorithm", "duplicate_rank"])
def test_padding_command_rejects_unsafe_reference_and_cleanup(tmp_path, fault):
    from copy import deepcopy
    entry, path = _entry(tmp_path)
    wire = {"key": entry.key, "tag": entry.tag}
    command = {"kind": "batch", "plan_id": 0, "global_keys": [entry.key],
               "global_sequence_nos": [0], "assignments": [[wire], []],
               "padding_ranks": [1], "padding_reference": wire}
    command = deepcopy(command)
    if fault == "foreign_template":
        command["padding_reference"]["key"] = "foreign"
    elif fault == "unsupported_algorithm":
        command["assignments"][0][0]["tag"]["feature_contract"]["algorithm"] = "UNKNOWN"
    elif fault == "duplicate_rank":
        command["padding_ranks"] = [1, 1]
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    loader = TQFeatureDataLoader(store, batch_size=2, rank=1, world_size=2,
                                 scheduled_commands=Queue(), prepare_batch=lambda entries: {},
                                 cleanup_after_load=fault == "early_cleanup")
    loader._broadcast_command = lambda _: command
    with pytest.raises(RuntimeError, match="padding|assignment"):
        next(iter(loader))
    assert path.exists()


@pytest.mark.parametrize("oversized", ["event", "command"])
def test_reference_control_limit_rejects_large_metadata_without_deleting(tmp_path, monkeypatch, oversized):
    from copy import deepcopy
    from types import SimpleNamespace
    from test_standalone_ray_runtime import (
        _commands, _runtime_config, _FakeRay, _ProducerHandle, _RecordingQueue, _WorkerGroup,
    )
    from verl_speco.standalone_ray_runtime import StandaloneRayTrainer

    config = deepcopy(_runtime_config())
    runtime = config.speco.draft_training
    runtime.scheduler = {"reference_command_max_bytes": 1024 if oversized == "command" else 268435456}
    runtime.get = lambda key, default=None: getattr(runtime, key, default)
    config.speco.standalone_tq_producer.reference_event_max_bytes = 1024 if oversized == "event" else 16777216
    training = config.actor_rollout_ref.rollout.drafter.training
    training.get = lambda key, default=None: getattr(training, key, default)
    training.feature_store = SimpleNamespace(type="ray_reference")
    training.batch_size_per_gpu = 1
    training.transfer_queue = {"consumer_prefetch_batches": 1}
    entry, path = _entry(tmp_path)
    entry.tag["request"]["source_metadata"]["padding"] = "x" * 3000
    events = _RecordingQueue(events=[
        {"kind": "consumer_ready", "initial_optimizer_step": 0},
        {"kind": "samples_ready", "batch_id": 0, "entries": [{"key": entry.key, "tag": entry.tag}]},
    ])
    commands = _RecordingQueue()
    class Ray(_FakeRay):
        def wait(self, refs, **kwargs):
            if refs and refs[0].name == "stop":
                return refs[:1], refs[1:]
            return [], list(refs)
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    monkeypatch.setattr("verl_speco.standalone_ray_runtime.compose_runtime_config", lambda _: config)
    trainer = StandaloneRayTrainer(_commands(), ray_module=Ray(_ProducerHandle()))
    trainer.producer, trainer.consumer_wg = _ProducerHandle(), _WorkerGroup()
    trainer.commands_queue, trainer.events_queue = commands, events
    trainer.feature_store_factory = lambda _: store
    with pytest.raises(ValueError, match="control metadata exceeds"):
        trainer.run()
    assert path.exists()
    assert not any(command.get("kind") == "batch" for command in commands.items)
