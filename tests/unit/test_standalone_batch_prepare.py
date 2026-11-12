# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import ast
import logging
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from verl_speco.trainer.standalone_batch_prepare import prepare_standalone_cpu_batch
from verl_speco.trainer.target_feature_replay import (
    FeatureContract, HiddenStateAlignmentError, feature_from_vllm_payload,
)
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION, SampleMetadata, make_reference_tag,
)


def test_local_frame_codec_binds_rank_and_tensor_metadata():
    import json
    from verl_speco.trainer.reference_shared_frame import encode_local_frame, decode_local_frame
    binding = {"run_id": "run", "generation": 1, "batch_id": 3, "rank": 2, "receipt_token": "receipt"}
    tensors = {"ids": torch.arange(6).reshape(2, 3), "hidden": torch.ones(2, 3, dtype=torch.bfloat16),
               "hidden_source": "standalone_vllm_reference", "_speco_pad_size": 0,
               "_speco_token_counts": {"source_tokens": 6, "supervised_tokens": 4}}
    payload, descriptor = encode_local_frame(tensors, binding=binding, max_bytes=4096)
    restored = decode_local_frame(payload, descriptor, binding=binding, max_bytes=4096)
    assert all(torch.equal(restored[name], value) if isinstance(value, torch.Tensor) else restored[name] == value
               for name, value in tensors.items())
    with pytest.raises(ValueError, match="identity"):
        decode_local_frame(payload, descriptor, binding={**binding, "rank": 1}, max_bytes=4096)
    with pytest.raises(ValueError, match="identity"):
        decode_local_frame(payload, descriptor, binding={**binding, "receipt_token": "old"}, max_bytes=4096)
    with pytest.raises(ValueError, match="checksum"):
        decode_local_frame(payload[:-1] + bytes([payload[-1] ^ 1]), descriptor, binding=binding, max_bytes=4096)
    manifest = json.loads(descriptor)
    manifest["tensors"]["ids"]["shape"] = [6]
    with pytest.raises(ValueError, match="checksum"):
        decode_local_frame(payload, json.dumps(manifest).encode(), binding=binding, max_bytes=4096)
    manifest = json.loads(descriptor)
    manifest["metadata"]["_speco_token_counts"]["source_tokens"] = 99
    with pytest.raises(ValueError, match="checksum"):
        decode_local_frame(payload, json.dumps(manifest).encode(), binding=binding, max_bytes=4096)
    with pytest.raises(ValueError, match="byte limit"):
        encode_local_frame(tensors, binding=binding, max_bytes=4)
    with pytest.raises(ValueError, match="metadata"):
        encode_local_frame({**tensors, "invalid": float("nan")}, binding=binding, max_bytes=4096)
    with pytest.raises(ValueError, match="metadata"):
        encode_local_frame({**tensors, "large": "x" * 65537}, binding=binding, max_bytes=4096)


def test_local_frame_mapped_tensors_survive_reader_context(tmp_path):
    from verl_speco.trainer.reference_shared_frame import _map_tensors
    path = tmp_path / "frame.safetensors"
    save_file({"ids": torch.arange(8)}, path)
    mapped = _map_tensors(str(path))
    assert torch.equal(mapped["ids"], torch.arange(8))


def test_local_frame_verifies_bytes_before_mapping(tmp_path, monkeypatch):
    import os
    from verl_speco.trainer import reference_shared_frame as frame
    binding = {"run_id": "run", "generation": 0, "batch_id": 1, "rank": 1, "receipt_token": "token"}
    payload, descriptor = frame.encode_local_frame({"ids": torch.arange(8), "label": "sample"},
                                                    binding=binding, max_bytes=4096)
    path = tmp_path / "frame.safetensors"
    path.write_bytes(payload)
    fd = os.open(path, os.O_RDONLY)
    def pread(handle, size, offset):
        os.lseek(handle, offset, os.SEEK_SET)
        return os.read(handle, size)
    original_map = frame._map_tensors
    calls = []
    monkeypatch.setattr(frame, "os", SimpleNamespace(fstat=os.fstat, pread=pread))
    monkeypatch.setattr(frame, "_map_tensors", lambda _: (calls.append(True), original_map(str(path)))[1])
    try:
        result = frame._read_mapped_frame(fd, descriptor, binding=binding, max_bytes=4096)
        assert result["label"] == "sample" and torch.equal(result["ids"], torch.arange(8))
        assert calls == [True]
        del result
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with pytest.raises(ValueError, match="checksum"):
            frame._read_mapped_frame(fd, descriptor, binding=binding, max_bytes=4096)
        assert calls == [True]
    finally:
        os.close(fd)


def test_local_frame_deadline_during_checksum_never_maps(tmp_path, monkeypatch):
    import os
    from verl_speco.trainer import reference_shared_frame as frame
    binding = {"run_id": "run", "generation": 0, "batch_id": 1, "rank": 1, "receipt_token": "token"}
    payload, descriptor = frame.encode_local_frame({"ids": torch.arange(8)}, binding=binding, max_bytes=4096)
    path = tmp_path / "frame.safetensors"
    path.write_bytes(payload)
    fd = os.open(path, os.O_RDONLY)
    now = [0.0]
    def pread(handle, size, offset):
        os.lseek(handle, offset, os.SEEK_SET)
        data = os.read(handle, size)
        now[0] = 2.0
        return data
    monkeypatch.setattr(frame, "os", SimpleNamespace(fstat=os.fstat, pread=pread))
    monkeypatch.setattr(frame, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(frame, "_map_tensors", lambda _: pytest.fail("expired frame must not be mapped"))
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            frame._read_mapped_frame(fd, descriptor, binding=binding, max_bytes=4096, deadline=1.0)
    finally:
        os.close(fd)


def test_local_frame_timeout_closes_received_fd(monkeypatch):
    import array
    import stat
    import sys
    from verl_speco.trainer import reference_shared_frame as frame
    closed = []
    fcntl = SimpleNamespace(F_SEAL_WRITE=1, F_SEAL_GROW=2, F_SEAL_SHRINK=4, F_SEAL_SEAL=8,
                            F_GET_SEALS=1, fcntl=lambda fd, command: 15)
    monkeypatch.setitem(sys.modules, "fcntl", fcntl)
    monkeypatch.setattr(frame, "socket", SimpleNamespace(SOL_SOCKET=1, SCM_RIGHTS=2, MSG_TRUNC=4, MSG_CTRUNC=8,
                                                          CMSG_SPACE=lambda value: value))
    monkeypatch.setattr(frame, "os", SimpleNamespace(fstat=lambda fd: SimpleNamespace(st_size=32, st_mode=stat.S_IFREG,
                                                                                         st_nlink=0), close=closed.append))
    monkeypatch.setattr(frame, "_verify_local_socket", lambda connection: None)
    monkeypatch.setattr(frame, "_read_mapped_frame", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("expired")))
    connection = SimpleNamespace(recvmsg=lambda *args: (b"descriptor", [(1, 2, array.array("i", [7]).tobytes())], 0, None))
    with pytest.raises(TimeoutError, match="expired"):
        frame.receive_local_frame(connection, binding={}, max_bytes=4096, deadline=1)
    assert closed == [7]


@pytest.mark.skipif(not hasattr(__import__("os"), "memfd_create"), reason="Linux memfd is unavailable")
def test_local_frame_passes_one_sealed_fd_and_closes_it():
    import os
    import socket
    from verl_speco.trainer.reference_shared_frame import (create_local_frame, encode_local_frame,
                                                            receive_local_frame, send_local_frame)
    binding = {"run_id": "run", "generation": 1, "batch_id": 0, "rank": 1, "receipt_token": "receipt"}
    payload, descriptor = encode_local_frame({"ids": torch.arange(4)}, binding=binding, max_bytes=4096)
    fd = create_local_frame(payload, max_bytes=4096)
    try:
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with sender, receiver:
            send_local_frame(sender, fd, descriptor)
            mapped = receive_local_frame(receiver, binding=binding, max_bytes=4096)
            os.close(fd)
            fd = None
            assert torch.equal(mapped["ids"], torch.arange(4))
    finally:
        if fd is not None:
            os.close(fd)


@pytest.mark.skipif(not hasattr(__import__("os"), "memfd_create"), reason="Linux memfd is unavailable")
def test_node_owner_materializes_only_each_requested_rank():
    import multiprocessing
    from verl_speco.trainer.reference_shared_frame import receive_node_batch
    binding = {"run_id": "run", "generation": 0, "batch_id": 2, "rank": 0, "receipt_token": "receipt"}
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    owner = context.Process(target=_node_owner_test_worker, args=(binding, results))
    owner.start()
    try:
        remote, _ = receive_node_batch(binding={**binding, "rank": 1}, owner_rank=0,
                                       max_bytes=4096, timeout=15)
        local, calls = results.get(timeout=15)
        owner.join(timeout=10)
        assert owner.exitcode == 0
        assert remote["ids"].tolist() == [7] and local == [5]
        assert calls == [7, 5]
    finally:
        if owner.is_alive():
            owner.terminate()
        owner.join(timeout=5)
        results.close()


def _node_owner_test_worker(binding, results):
    from verl_speco.trainer.reference_shared_frame import serve_node_batches
    calls = []
    def prepare(entries):
        calls.append(entries[0])
        return {"ids": torch.tensor(entries), "hidden_source": "reference"}, 0.0
    local, _ = serve_node_batches({0: [5], 1: [7]}, binding=binding,
                                  prepare=prepare, max_bytes=4096, timeout=15)
    results.put((local["ids"].tolist(), calls))


@pytest.mark.parametrize("fault", [None, "duplicate", "foreign", "prepare", "send", "slow_prepare", "slow_create", "slow_send", "slow_owner"])
def test_node_owner_routes_and_stops_on_failure_without_releasing_inputs(monkeypatch, fault):
    import json
    from verl_speco.trainer import reference_shared_frame as frame
    binding = {"run_id": "run", "generation": 0, "batch_id": 2, "rank": 0, "receipt_token": "receipt"}
    ranks = [2, 2 if fault == "duplicate" else 1]
    prepared, transmitted, closed, payloads = [], {}, [], {}
    now = [0.0]
    monkeypatch.setattr(frame, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0]))
    class Connection:
        def __init__(self, rank): self.rank = rank
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): pass
        def recv(self, size):
            token = "foreign" if fault == "foreign" and self.rank == 1 else "receipt"
            return json.dumps({**binding, "rank": self.rank, "receipt_token": token}).encode()
    class Listener:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def listen(self, size): pass
        def settimeout(self, value): pass
        def accept(self): return Connection(ranks.pop(0)), None
    monkeypatch.setattr(frame, "socket", SimpleNamespace(AF_UNIX=1, SOCK_SEQPACKET=2, socket=lambda *a: Listener()))
    monkeypatch.setattr(frame, "os", SimpleNamespace(memfd_create=True, close=closed.append))
    monkeypatch.setattr(frame, "_verify_local_socket", lambda connection: None)
    def create(payload, *, max_bytes):
        fd = 10 + len(payloads)
        payloads[fd] = payload
        if fault == "slow_create" and fd == 11:
            now[0] = 11.0
        return fd
    def send(connection, fd, descriptor):
        if fault == "send" and connection.rank == 1:
            raise RuntimeError("injected send failure")
        transmitted[connection.rank] = frame.decode_local_frame(
            payloads[fd], descriptor, binding={**binding, "rank": connection.rank}, max_bytes=4096)["ids"].tolist()
        if fault == "slow_send" and connection.rank == 1:
            now[0] = 11.0
    def prepare(entries):
        prepared.append(entries[0])
        if fault == "prepare" and entries[0] == 1:
            raise RuntimeError("injected read failure")
        if (fault == "slow_prepare" and entries[0] == 1) or (fault == "slow_owner" and entries[0] == 0):
            now[0] = 11.0
        return {"ids": torch.tensor(entries), "hidden_source": "reference"}, 0.0
    monkeypatch.setattr(frame, "create_local_frame", create)
    monkeypatch.setattr(frame, "send_local_frame", send)
    groups = {0: [0], 1: [1], 2: [2]}
    if fault is None:
        own, _ = frame.serve_node_batches(groups, binding=binding, prepare=prepare, max_bytes=4096, timeout=10)
        assert own["ids"].tolist() == [0]
        assert transmitted == {2: [2], 1: [1]} and prepared == [2, 1, 0]
    else:
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            frame.serve_node_batches(groups, binding=binding, prepare=prepare, max_bytes=4096, timeout=10)
        if fault == "slow_owner":
            assert prepared == [2, 1, 0] and transmitted == {2: [2], 1: [1]}
        elif fault == "slow_send":
            assert prepared == [2, 1] and transmitted == {2: [2], 1: [1]}
        else:
            assert 0 not in prepared and transmitted == {2: [2]}
    assert set(closed) == set(payloads)
    assert groups == {0: [0], 1: [1], 2: [2]}


def test_node_receiver_bounds_connect_and_closes_failed_socket(monkeypatch):
    from verl_speco.trainer import reference_shared_frame as frame
    binding = {"run_id": "run", "generation": 1, "batch_id": 2, "rank": 1, "receipt_token": "receipt"}
    calls = []
    class Connection:
        def settimeout(self, value): calls.append(("timeout", value))
        def connect(self, address):
            calls.append(("connect", address))
            raise TimeoutError("blocked connect")
        def close(self): calls.append(("close",))
    monkeypatch.setattr(frame, "time", SimpleNamespace(monotonic=lambda: 5.0))
    monkeypatch.setattr(frame, "os", SimpleNamespace(memfd_create=True))
    monkeypatch.setattr(frame, "socket", SimpleNamespace(AF_UNIX=1, SOCK_SEQPACKET=2, socket=lambda *a: Connection()))
    with pytest.raises(TimeoutError, match="blocked connect"):
        frame.receive_node_batch(binding=binding, owner_rank=0, max_bytes=4096, timeout=3)
    assert calls[0] == ("timeout", 3.0) and calls[1][0] == "connect" and calls[2] == ("close",)


def test_node_receiver_does_not_wait_again_after_slow_request(monkeypatch):
    from verl_speco.trainer import reference_shared_frame as frame
    binding = {"run_id": "run", "generation": 1, "batch_id": 2, "rank": 1, "receipt_token": "receipt"}
    now, closed = [0.0], []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): closed.append(True)
        def settimeout(self, value): pass
        def connect(self, address): pass
        def send(self, request):
            now[0] = 11.0
            return len(request)
    monkeypatch.setattr(frame, "time", SimpleNamespace(monotonic=lambda: now[0], perf_counter=lambda: now[0]))
    monkeypatch.setattr(frame, "os", SimpleNamespace(memfd_create=True))
    monkeypatch.setattr(frame, "socket", SimpleNamespace(AF_UNIX=1, SOCK_SEQPACKET=2, socket=lambda *a: Connection()))
    monkeypatch.setattr(frame, "receive_local_frame", lambda *args, **kwargs: pytest.fail("expired request cannot receive"))
    with pytest.raises(TimeoutError, match="deadline"):
        frame.receive_node_batch(binding=binding, owner_rank=0, max_bytes=4096, timeout=10)
    assert closed == [True]


@pytest.mark.parametrize("timeout,max_bytes", [(True, 4096), ("1", 4096), (1, 0), (1, True)])
def test_node_owner_rejects_invalid_deadline_or_budget(timeout, max_bytes):
    from verl_speco.trainer.reference_shared_frame import serve_node_batches
    binding = {"run_id": "run", "generation": 1, "batch_id": 2, "rank": 0, "receipt_token": "receipt"}
    with pytest.raises(ValueError, match="plan"):
        serve_node_batches({0: [1]}, binding=binding, prepare=lambda _: pytest.fail("unexpected read"),
                           max_bytes=max_bytes, timeout=timeout)


def _contract(algorithm="DSPARK", *, reuse_final=False):
    return FeatureContract(
        algorithm=algorithm, target_layer_ids=[0, 1],
        vllm_aux_hidden_state_layer_ids=[1, 4] if reuse_final else [1, 2],
        target_num_hidden_layers=4,
        hidden_states_layout=(
            "eagle3_aux_plus_last" if algorithm == "EAGLE3" else
            "dflash_aux" if algorithm == "DFLASH" else "dflash_aux_plus_last"
        ),
        dtype=torch.bfloat16, target_model_id="target", target_model_revision="rev",
        tokenizer_fingerprint="tokens", require_full_alignment=True,
    )


def _sample(tmp_path, contract, *, length=5, index=0, positions=None, hidden_offset=0):
    path = tmp_path / f"sample-{index}.safetensors"
    layers = 2 if contract.hidden_states_layout == "dflash_aux" or contract.vllm_aux_hidden_state_layer_ids == [1, 4] else 3
    hidden = torch.arange(length * layers * 2).reshape(length, layers, 2).float()
    save_file({"token_ids": torch.arange(length), "hidden_states": hidden[hidden_offset:]}, path)
    values = asdict(contract)
    values["dtype"] = str(contract.dtype)
    positions = list(range(length)) if positions is None else positions
    tag = make_reference_tag(
        SampleMetadata(PROTOCOL_SCHEMA_VERSION, "run", str(index), index), path=str(path),
        request={
            "input_ids": list(range(length)), "loss_mask": [0] + [1] * (length - 1),
            "position_ids": list(range(length)), "feature_positions": positions,
            "draft_position_ids": [p + 1 for p in positions],
            "vllm_prompt_token_ids": list(range(length)), "source_metadata": {},
        }, feature_contract=values,
    )
    return tag, hidden


@pytest.mark.parametrize("algorithm,packing", [("DSPARK", False), ("DSPARK", True),
                                                 ("DFLASH", False), ("DFLASH", True), ("EAGLE3", False)])
def test_node_frame_roundtrips_real_prepared_batches(tmp_path, algorithm, packing):
    from verl_speco.trainer.reference_shared_frame import encode_local_frame, decode_local_frame
    contract = _contract(algorithm)
    tags = [_sample(tmp_path, contract, length=length, index=index)[0]
            for index, length in enumerate((5, 3))]
    batch = prepare_standalone_cpu_batch(tags, contract, pin_memory=False, streaming=True,
                                         packing=packing, block_size=4)
    binding = {"run_id": "run", "generation": 0, "batch_id": 2, "rank": 1, "receipt_token": "token"}
    payload, descriptor = encode_local_frame(batch, binding=binding, max_bytes=1024 * 1024)
    result = decode_local_frame(payload, descriptor, binding=binding, max_bytes=1024 * 1024)
    assert set(result) == set(batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(result[key], value)
        else:
            assert result[key] == value


def test_owned_window_uses_configured_shared_file_root(tmp_path):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
    from verl_speco.trainer.standalone_batch_prepare import scan_owned_anchor_inputs, prepare_owned_anchor_window
    root = tmp_path / "shared"
    root.mkdir()
    contract = _contract("DSPARK")
    tag, _ = _sample(tmp_path, contract, length=8)
    store = FileHiddenStatesStore(str(root))
    with pytest.raises(ValueError, match="outside the configured shared file root"):
        scan_owned_anchor_inputs(tag, contract, hidden_states_store=store)
    with pytest.raises(ValueError, match="outside the configured shared file root"):
        prepare_owned_anchor_window(tag, contract, start=0, end=6, anchor=2, block_size=2,
                                    windows=[3], hidden_states_store=store)


@pytest.mark.parametrize("windows", [[3], [3, None]])
def test_owned_reference_window_reads_exact_rows_and_retains_source(tmp_path, monkeypatch, windows):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
    from verl_speco.trainer.standalone_batch_prepare import prepare_owned_anchor_window
    contract = _contract("DSPARK")
    tag, _ = _sample(tmp_path, contract, length=12)
    full = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    calls = []
    original = FileHiddenStatesStore.load_window
    def read(store, reference, **kwargs):
        calls.append(kwargs["selected_positions"])
        return original(store, reference, **kwargs)
    monkeypatch.setattr(FileHiddenStatesStore, "load_window", read)
    start, end, anchor = (0 if None in windows else 3), 9, 6
    part = prepare_owned_anchor_window(tag, contract, start=start, end=end, anchor=anchor,
                                       block_size=2, windows=windows)
    assert calls == [list(range(start, end))]
    for key in ("input_ids", "hidden_states", "loss_mask", "target_last_hidden_states", "label_ids", "label_mask"):
        assert torch.equal(part[key], full[key][:, start:end])
    assert part["preselected_anchor_positions"].tolist() == [[anchor - start]]
    assert part["preselected_anchor_keep_mask"].tolist() == [[True]]
    assert Path(tag["hidden_states_path"]).exists()
    with pytest.raises(ValueError, match="context or labels"):
        prepare_owned_anchor_window(tag, contract, start=start + 1, end=end, anchor=anchor,
                                    block_size=2, windows=windows)
    with pytest.raises(ValueError, match="batch_bytes"):
        prepare_owned_anchor_window(tag, contract, start=start, end=end, anchor=anchor,
                                    block_size=2, windows=windows, max_output_bytes=1)
    assert Path(tag["hidden_states_path"]).exists()


def test_owned_reference_window_respects_cropped_source(tmp_path):
    from verl_speco.trainer.standalone_batch_prepare import prepare_owned_anchor_window, scan_owned_anchor_inputs
    contract = _contract("DSPARK")
    tag, _ = _sample(tmp_path, contract, length=12)
    tag["request"]["loss_mask"] = [0] * 7 + [1] * 5
    scanned = scan_owned_anchor_inputs(tag, contract, max_window=5)
    assert scanned["feature_window"] == (5, 10)
    full = prepare_standalone_cpu_batch([tag], contract, pin_memory=False, max_window=5, streaming=True)
    part = prepare_owned_anchor_window(tag, contract, start=5, end=10, anchor=7,
                                       block_size=2, windows=[3], source_window=scanned["feature_window"])
    for key in ("input_ids", "hidden_states", "loss_mask", "target_last_hidden_states"):
        assert torch.equal(part[key], full[key])
    with pytest.raises(ValueError, match="context or labels"):
        prepare_owned_anchor_window(tag, contract, start=4, end=10, anchor=7,
                                    block_size=2, windows=[3], source_window=(5, 10))


def test_owned_reference_window_matches_training_loss_and_gradient(tmp_path):
    from copy import deepcopy
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.trainer.standalone_batch_prepare import prepare_owned_anchor_window
    torch.manual_seed(19)
    contract = _contract("DSPARK")
    tag, _ = _sample(tmp_path, contract, length=12)
    full = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    part = prepare_owned_anchor_window(tag, contract, start=3, end=9, anchor=6,
                                       block_size=2, windows=[3])
    config = DSparkConfig(hidden_size=2, intermediate_size=8, num_hidden_layers=1,
                          num_attention_heads=1, num_key_value_heads=1, head_dim=2,
                          vocab_size=32, num_target_layers=4, num_context_layers=2,
                          target_hidden_size=2, target_num_hidden_layers=4, target_layer_ids=[0, 1],
                          mask_token_id=31, markov_rank=2, markov_head_type="vanilla",
                          sliding_window=3, use_sliding_window=True, layer_types=["sliding_attention"])
    model = DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                                num_anchors=3, l1_loss_alpha=0.0,
                                verifier_norm=torch.nn.LayerNorm(2)).eval()
    local = deepcopy(model)
    head = torch.randn(32, 2)
    def run(module, batch, anchor):
        return module(input_ids=batch["input_ids"], hidden_states_list=list(batch["hidden_states"].split(2, dim=-1)),
                      loss_mask=batch["loss_mask"], lm_head_weight=head,
                      document_ids=torch.zeros_like(batch["input_ids"]),
                      label_ids=batch["label_ids"], label_mask=batch["label_mask"],
                      preselected_anchor_positions=torch.tensor([[anchor]]),
                      preselected_anchor_keep_mask=torch.tensor([[True]]))
    original = run(model, full, 6)
    cropped = run(local, part, 3)
    torch.testing.assert_close(original[0], cropped[0], rtol=2e-5, atol=2e-6)
    original[0].backward()
    cropped[0].backward()
    for a, b in zip(model.parameters(), local.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)


    model.zero_grad(set_to_none=True)
    local.zero_grad(set_to_none=True)
    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=head))
    backend.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
        "mode": "offline", "feature_store": {"type": "ray_reference"},
        "transfer_queue": {"expected_feature": {"payload_kind": "vllm_reference"}},
    })))
    full["document_ids"] = torch.zeros_like(full["input_ids"])
    full["preselected_anchor_positions"] = torch.tensor([[6]])
    full["preselected_anchor_keep_mask"] = torch.tensor([[True]])
    original_loss = backend.compute_loss(model, full, 0)
    local_loss = backend.compute_loss(local, part, 0)
    torch.testing.assert_close(original_loss["total_local_ploss"], local_loss["total_local_ploss"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original_loss["local_num_tokens"], local_loss["local_num_tokens"])
    original_loss["total_local_ploss"].backward()
    local_loss["total_local_ploss"].backward()
    for a, b in zip(model.parameters(), local.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)
    model.zero_grad(set_to_none=True)
    local.zero_grad(set_to_none=True)
    full["preselected_anchor_positions"] = torch.tensor([[3, 6, 9]])
    full["preselected_anchor_keep_mask"] = torch.tensor([[True, True, True]])
    total = backend.compute_loss(model, full, 0)
    parts = [prepare_owned_anchor_window(tag, contract, start=anchor - 3, end=anchor + 3,
                                         anchor=anchor, block_size=2, windows=[3])
             for anchor in (3, 6, 9)]
    results = [backend.compute_loss(local, item, 0) for item in parts]
    part_loss = sum(item["total_local_ploss"] for item in results)
    part_tokens = sum(item["local_num_tokens"] for item in results)
    torch.testing.assert_close(total["total_local_ploss"], part_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(total["local_num_tokens"], part_tokens)
    total["total_local_ploss"].backward()
    part_loss.backward()
    for a, b in zip(model.parameters(), local.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("max_window", [None, 5])
@pytest.mark.parametrize("positions", [[0, 2, 3, 5, 7, 8, 9], [9, 2, 2, 7, 3, 8, 5]])
def test_owned_anchor_scan_matches_baseline_sanitized_mask(tmp_path, dtype, max_window, positions):
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.standalone_batch_prepare import scan_owned_anchor_inputs
    contract = replace(_contract("DSPARK"), dtype=dtype)
    tag, hidden = _sample(tmp_path, contract, length=10, positions=positions)
    hidden[2, 0, 0] = float("nan")
    hidden[5, 1, 0] = 1e6
    hidden[7, 2, 0] = float("nan")  # Raw final target is sanitized later, not before sampling.
    save_file({"token_ids": torch.arange(10), "hidden_states": hidden}, tag["hidden_states_path"])
    full = prepare_standalone_cpu_batch([tag], contract, pin_memory=False, streaming=True, max_window=max_window)
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
        "mode": "offline", "feature_store": {"type": "ray_reference"}, "hidden_state_clip_value": 1000.0,
    })))
    trainer.backend, trainer.runtime_device, trainer.rank = SimpleNamespace(model_type="dspark"), torch.device("cpu"), 0
    staged = trainer._stage_standalone_cpu_batch(full)
    scanned = scan_owned_anchor_inputs(tag, contract, max_window=max_window, clip_value=1000.0, chunk_rows=2)
    for key in ("input_ids", "loss_mask", "label_mask", "attention_mask"):
        assert torch.equal(scanned[key], staged[key])
    start, end = scanned["feature_window"]
    assert scanned["input_ids"].tolist() == [[positions[i] for i in range(start, end)]]
    assert Path(tag["hidden_states_path"]).exists()


@pytest.mark.parametrize("empty_second", [False, True])
def test_owned_anchor_sampling_preserves_padded_batch_rng_and_windows(tmp_path, empty_second):
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainingModel
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.standalone_batch_prepare import sample_owned_anchor_windows, prepare_owned_anchor_window
    contract = _contract("DSPARK")
    tags = [_sample(tmp_path, contract, length=length, index=index)[0]
            for index, length in enumerate((12, 7))]
    tags[0]["request"]["loss_mask"] = [0] * 7 + [0.25] * 5
    if empty_second:
        tags[1]["request"]["loss_mask"] = [0] * 7
    else:
        tags[1]["request"]["loss_mask"][-1] = 1e-100  # Becomes zero in the baseline float32 mask.
    model = SimpleNamespace(num_anchors=4, block_size=2,
                            _cached_arange=lambda _key, size, device: torch.arange(size, device=device))
    model._sample_anchor_positions = DSparkTrainingModel._sample_anchor_positions.__get__(model)
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
        "mode": "offline", "feature_store": {"type": "ray_reference"}, "hidden_state_clip_value": 1000.0,
    })))
    trainer.backend, trainer.runtime_device, trainer.rank = SimpleNamespace(model_type="dspark"), torch.device("cpu"), 0
    full = trainer._stage_standalone_cpu_batch(prepare_standalone_cpu_batch(
        tags, contract, pin_memory=False, streaming=True, max_window=5))
    torch.manual_seed(42)
    expected, expected_keep = model._sample_anchor_positions(full["label_mask"].size(1), full["label_mask"], torch.device("cpu"))
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    counts, vocabulary = {}, {"size": 32}
    anchors, keep, plans = sample_owned_anchor_windows(tags, contract, model, device=torch.device("cpu"),
                                                        windows=[3], max_window=5, clip_value=1000.0, chunk_rows=2,
                                                        token_counts=counts, vocabulary=vocabulary)
    assert torch.equal(anchors, expected) and torch.equal(keep, expected_keep)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert counts == full["_speco_token_counts"]
    assert torch.equal(vocabulary["ids"], torch.unique(full["input_ids"], sorted=True))
    assert len(plans) == int(keep.sum())
    for plan in plans:
        row, slot = plan["sample_index"], plan["anchor_slot"]
        assert plan["anchor"] == int(anchors[row, slot]) + plan["source_window"][0]
        part = prepare_owned_anchor_window(tags[row], contract, start=plan["start"], end=plan["end"],
                                           anchor=plan["anchor"], block_size=2, windows=[3],
                                           source_window=plan["source_window"])
        source_start = plan["source_window"][0]
        a, b = plan["start"] - source_start, plan["end"] - source_start
        assert torch.equal(part["input_ids"][0], full["input_ids"][row, a:b])
        assert torch.equal(part["hidden_states"][0], full["hidden_states"][row, a:b])


@pytest.mark.parametrize("max_packed_len", [0, 12])
@pytest.mark.parametrize("packing_block_size", [2, 3])
def test_owned_anchor_sampling_matches_packed_rows(tmp_path, max_packed_len, packing_block_size):
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainingModel
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.standalone_batch_prepare import sample_owned_anchor_windows, prepare_owned_anchor_window
    contract = _contract("DSPARK")
    tags = [_sample(tmp_path, contract, length=length, index=index)[0]
            for index, length in enumerate((5, 3, 4, 7))]
    full = prepare_standalone_cpu_batch(tags, contract, pin_memory=False, streaming=True, packing=True,
                                         block_size=packing_block_size, max_packed_len=max_packed_len)
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
        "mode": "offline", "feature_store": {"type": "ray_reference"},
    })))
    trainer.backend, trainer.runtime_device, trainer.rank = SimpleNamespace(model_type="dspark"), torch.device("cpu"), 0
    staged = trainer._stage_standalone_cpu_batch(full)
    model = SimpleNamespace(num_anchors=6, block_size=2,
                            _cached_arange=lambda _key, size, device: torch.arange(size, device=device))
    model._sample_anchor_positions = DSparkTrainingModel._sample_anchor_positions.__get__(model)
    torch.manual_seed(42)
    expected, expected_keep = model._sample_anchor_positions(staged["input_ids"].size(1), staged["loss_mask"],
                                                              torch.device("cpu"), document_ids=staged["document_ids"])
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    counts, vocabulary = {}, {"size": 32}
    anchors, keep, plans = sample_owned_anchor_windows(tags, contract, model, device=torch.device("cpu"), windows=[3],
                                                        packing=True, packing_block_size=packing_block_size, max_packed_len=max_packed_len,
                                                        token_counts=counts, vocabulary=vocabulary)
    assert torch.equal(anchors, expected) and torch.equal(keep, expected_keep)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert counts == full["_speco_token_counts"] and len(plans) == int(keep.sum())
    assert torch.equal(vocabulary["ids"], torch.unique(full["input_ids"], sorted=True))
    for plan in plans:
        part = prepare_owned_anchor_window(tags[plan["sample_index"]], contract, start=plan["start"], end=plan["end"],
                                           anchor=plan["anchor"], block_size=2, windows=[3],
                                           source_window=plan["source_window"])
        packed_anchor = int(anchors[plan["batch_row"], plan["anchor_slot"]])
        local_anchor = int(part["preselected_anchor_positions"][0, 0])
        assert int(part["input_ids"][0, local_anchor]) == int(staged["input_ids"][plan["batch_row"], packed_anchor])
        assert torch.equal(part["hidden_states"][0, local_anchor], staged["hidden_states"][plan["batch_row"], packed_anchor])


def test_sampled_owned_windows_match_padded_batch_gradients(tmp_path):
    from copy import deepcopy
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.trainer.standalone_batch_prepare import sample_owned_anchor_windows, prepare_owned_anchor_window
    contract = _contract("DSPARK")
    tags = [_sample(tmp_path, contract, length=length, index=index)[0]
            for index, length in enumerate((12, 8))]
    full = prepare_standalone_cpu_batch(tags, contract, pin_memory=False, streaming=True)
    config = DSparkConfig(hidden_size=2, intermediate_size=8, num_hidden_layers=1,
                          num_attention_heads=1, num_key_value_heads=1, head_dim=2,
                          vocab_size=32, num_target_layers=4, num_context_layers=2,
                          target_hidden_size=2, target_num_hidden_layers=4, target_layer_ids=[0, 1],
                          mask_token_id=31, markov_rank=2, markov_head_type="vanilla",
                          sliding_window=3, use_sliding_window=True, layer_types=["sliding_attention"])
    torch.manual_seed(17)
    model = DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                                num_anchors=3, l1_loss_alpha=0.0,
                                verifier_norm=torch.nn.LayerNorm(2)).eval()
    local = deepcopy(model)
    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=torch.randn(32, 2)))
    backend.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
        "mode": "offline", "feature_store": {"type": "ray_reference"},
        "transfer_queue": {"expected_feature": {"payload_kind": "vllm_reference"}},
    })))
    torch.manual_seed(29)
    anchors, keep, plans = sample_owned_anchor_windows(tags, contract, model, device=torch.device("cpu"), windows=[3])
    full["preselected_anchor_positions"], full["preselected_anchor_keep_mask"] = anchors, keep
    expected = backend.compute_loss(model, full, 0)
    parts = [prepare_owned_anchor_window(tags[p["sample_index"]], contract, start=p["start"], end=p["end"],
                                        anchor=p["anchor"], block_size=2, windows=[3], source_window=p["source_window"])
             for p in plans]
    results = [backend.compute_loss(local, part, 0) for part in parts]
    actual_loss = sum(result["total_local_ploss"] for result in results)
    torch.testing.assert_close(expected["total_local_ploss"], actual_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(expected["local_num_tokens"], sum(result["local_num_tokens"] for result in results))
    expected["total_local_ploss"].backward()
    actual_loss.backward()
    for a, b in zip(model.parameters(), local.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6)
    from verl_speco.trainer.standalone_batch_prepare import prepare_empty_owned_window
    empty = prepare_empty_owned_window(tags[0], contract)
    zero = backend.compute_loss(local, empty, 0)
    assert zero["local_num_tokens"].item() == 0
    assert zero["total_local_ploss"].item() == 0 and zero["total_local_ploss"].requires_grad
    zero["total_local_ploss"].backward()
    import asyncio
    from verl_speco.trainer.draft_training_loop import _owned_reference_training_step
    seen = []
    async def train(batch, step, **kwargs):
        result = backend.compute_loss(local, batch, 0)
        seen.append((result["local_num_tokens"].item(), kwargs["accumulation_index"], kwargs["accumulation_steps"]))
        result["total_local_ploss"].backward()
        return True
    trainer = SimpleNamespace(model=local, backend=SimpleNamespace(model_type="dspark"),
                              optimizer=torch.optim.SGD(local.parameters(), lr=0.01),
                              _stage_standalone_cpu_batch=lambda batch: batch, training_step_from_batch=train)
    torch.manual_seed(29)
    assert asyncio.run(_owned_reference_training_step(trainer, tags, contract, 0, device=torch.device("cpu"), windows=[3]))
    assert len(seen) == len(plans) and sum(item[0] for item in seen) == pytest.approx(expected["local_num_tokens"].item())
    assert [item[1] for item in seen] == list(range(len(seen)))
    assert all(item[2] == len(seen) for item in seen)
    seen.clear()
    padding_counts = {}
    assert asyncio.run(_owned_reference_training_step(trainer, tags, contract, 0, device=torch.device("cpu"),
                                                       windows=[3], padding_rank=True, token_counts=padding_counts))
    assert seen == [(0.0, 0, 1)]
    assert padding_counts["padding_tokens"] == full["input_ids"].numel()
    assert padding_counts["window_train_input_positions"] == empty["input_ids"].numel()
    assert all(value == 0 for key, value in padding_counts.items()
               if key not in {"padding_tokens", "window_train_input_positions"})
    assert all(Path(tag["hidden_states_path"]).exists() for tag in tags)
    from queue import Queue
    from verl_speco.trainer.draft_training_loop import _execute_owned_reference_batch, _next_batch_across_ranks
    from verl_speco.trainer.tq_feature_store import ReadyEntry, ReferenceFeatureStore
    from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader
    from verl_speco.transport.drafter_sample_protocol import make_sample_key
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.connect()
    entries = [ReadyEntry(make_sample_key(SampleMetadata.from_dict(tag)), tag) for tag in tags]
    commands, events = Queue(), Queue()
    loader = TQFeatureDataLoader(store, batch_size=2, rank=0, world_size=1,
                                 scheduled_commands=commands, prepare_batch=store.prepare_batch)
    command = loader._batch_command(entries)
    command.update(plan_id=0, receipt_token="receipt")
    commands.put(command)
    assigned = _next_batch_across_ranks(iter(loader.iter_assignments()), rank=0, device=torch.device("cpu"),
                                         require_reference_metadata=True)
    trainer._get_hidden_state_clip_value = lambda: 10000.0
    seen.clear()
    assert events.empty() and all(Path(tag["hidden_states_path"]).exists() for tag in tags)
    async def fail_training(*args, **kwargs):
        raise RuntimeError("injected failure")
    trainer.training_step_from_batch = fail_training
    with pytest.raises(RuntimeError, match="Reference part training failed"):
        asyncio.run(_execute_owned_reference_batch(trainer, store, assigned, 0, rank=0,
                                                   device=torch.device("cpu"), events=events, transport_config={}))
    assert events.empty() and all(Path(tag["hidden_states_path"]).exists() for tag in tags)
    trainer.training_step_from_batch = train
    assert asyncio.run(_execute_owned_reference_batch(trainer, store, assigned, 0, rank=0,
                                                      device=torch.device("cpu"), events=events, transport_config={}))
    assert len(seen) == len(plans)
    assert events.get_nowait()["kind"] == "training_materialized"
    assert all(Path(tag["hidden_states_path"]).exists() for tag in tags)
    loader.clear_completed_batch(assigned.global_keys)
    assert all(not Path(tag["hidden_states_path"]).exists() for tag in tags)
    store.close()


@pytest.mark.parametrize("ce_target_source", ["labels", "target_argmax"])
@pytest.mark.parametrize("l1_alpha", [0.0, 0.9])
@pytest.mark.parametrize("lk_alpha,lk_type,confidence_alpha,confidence_mode", [
    (0.0, "alpha", 0.0, "greedy"), (0.3, "alpha", 0.0, "greedy"),
    (0.3, "adaptive_hybrid", 0.0, "greedy"), (0.0, "alpha", 0.4, "greedy"),
    (0.3, "alpha", 0.4, "probabilistic"), (0.3, "adaptive_hybrid", 0.4, "greedy"),
    (0.3, "adaptive_hybrid", 0.4, "probabilistic"), (0.0, "alpha", 0.4, "all_invalid"),
])
@pytest.mark.parametrize("packing", [False, True])
def test_owned_file_training_matches_full_batch_optimizer(tmp_path, monkeypatch, ce_target_source, l1_alpha, lk_alpha,
                                                     lk_type, confidence_alpha, confidence_mode, packing,
                                                     position_weight="fixed_exp_decay", loss_mode="full_vocab", negatives=9,
                                                     dpace_alpha=0.5, extreme_dpace=False):
    import asyncio
    from contextlib import nullcontext
    from copy import deepcopy
    import verl_speco.trainer.base_trainer as trainer_module
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from queue import Queue
    from verl_speco.trainer.draft_training_loop import (_execute_owned_reference_batch,
        _reference_training_parts_across_ranks, _reference_training_step_across_ranks)
    from verl_speco.trainer.tq_feature_store import ReadyEntry, ReferenceFeatureStore
    from verl_speco.trainer.tq_sample_source import TQLocalBatch
    from verl_speco.transport.drafter_sample_protocol import make_sample_key
    if extreme_dpace:
        import torch.nn.functional as functional
        original_nll = functional.nll_loss
        def extreme_nll(log_probs, targets, *args, **kwargs):
            result = original_nll(log_probs, targets, *args, **kwargs)
            return result + ((targets.remainder(2) == 0).to(result.dtype) * 200.0
                             if extreme_dpace == "zero" else 20.0)
        monkeypatch.setattr(functional, "nll_loss", extreme_nll)
    monkeypatch.setattr(trainer_module, "device_name", "cpu")
    if confidence_alpha and confidence_mode in {"probabilistic", "all_invalid"}:
        predict = DSparkDraftModel.predict_confidence
        def with_invalid_rows(self, hidden, *, prev_token_ids):
            logits = predict(self, hidden, prev_token_ids=prev_token_ids)
            invalid = torch.ones_like(prev_token_ids, dtype=torch.bool) if confidence_mode == "all_invalid" else prev_token_ids.remainder(2) == 0
            return torch.where(invalid, torch.full_like(logits, float("inf")), logits)
        monkeypatch.setattr(DSparkDraftModel, "predict_confidence", with_invalid_rows)
    contract = _contract("DSPARK")
    tags = [_sample(tmp_path, contract, length=length, index=index)[0]
            for index, length in enumerate((12, 8))]
    from safetensors.torch import load_file
    payload = load_file(tags[0]["hidden_states_path"])
    payload["hidden_states"][5, 0, 0] = float("nan")
    if ce_target_source == "target_argmax":
        payload["hidden_states"][3, 2, 0] = float("nan")
    save_file(payload, tags[0]["hidden_states_path"])
    full = prepare_standalone_cpu_batch(tags, contract, pin_memory=False, streaming=True, packing=packing, block_size=2)
    model_config = DSparkConfig(hidden_size=2, intermediate_size=8, num_hidden_layers=1,
                                num_attention_heads=1, num_key_value_heads=1, head_dim=2,
                                vocab_size=32, num_target_layers=4, num_context_layers=2,
                                target_hidden_size=2, target_num_hidden_layers=4, target_layer_ids=[0, 1],
                                mask_token_id=31, markov_rank=2, markov_head_type="vanilla",
                                enable_confidence_head=confidence_alpha > 0, confidence_head_with_markov=True,
                                sliding_window=3, use_sliding_window=True, layer_types=["sliding_attention"])
    torch.manual_seed(55)
    model = DSparkTrainingModel(draft_model=DSparkDraftModel(model_config), block_size=2,
                                num_anchors=3, l1_loss_alpha=l1_alpha, lk_loss_alpha=lk_alpha, ce_target_source=ce_target_source,
                                per_position_loss_weight=position_weight,
                                dpace_alpha=dpace_alpha,
                                loss_mode=loss_mode,
                                sampled_ce_negatives=negatives,
                                lk_loss_type=lk_type, confidence_head_alpha=confidence_alpha,
                                confidence_target_mode="probabilistic" if confidence_mode == "all_invalid" else confidence_mode,
                                l1_chunk_size=2 if l1_alpha and ce_target_source == "labels" else 0,
                                verifier_norm=torch.nn.LayerNorm(2))
    local = deepcopy(model)
    backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
    backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=torch.randn(32, 2)))
    backend.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
        "mode": "offline", "feature_store": {"type": "ray_reference"},
        "transfer_queue": {"expected_feature": {"payload_kind": "vllm_reference"}},
    })))
    observed_denominators = []
    if extreme_dpace:
        compute = backend.compute_loss
        def observe(*args, **kwargs):
            result = compute(*args, **kwargs)
            diagnostic = result["diagnostics"]
            observed_denominators.append((float(diagnostic["ce_weighted_token_count"]),
                                          float(diagnostic["l1_weighted_token_count"]),
                                          float(result["total_local_ploss"].detach())))
            return result
        backend.compute_loss = observe
    def make_trainer(current):
        trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
        trainer.model, trainer.backend, trainer.config = current, backend, backend.config
        trainer.runtime_device, trainer.rank = torch.device("cpu"), 0
        trainer.optimizer = torch.optim.SGD(current.parameters(), lr=0.01)
        trainer.lr_scheduler = None
        trainer.optimizer_steps_total = trainer.training_steps = 0
        trainer._standalone_validate_grad_graph = True
        trainer.combine_standalone_metrics = False
        trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
        trainer.record_training_timing = lambda *args: None
        trainer._ulysses_group_context = nullcontext
        trainer._record_dflash_training_metrics = lambda losses: {}
        trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
        trainer._is_checkpoint_leader = lambda: False
        trainer._reduce_loss_statistics = lambda values: (values, 1)
        return trainer
    baseline, owned = make_trainer(model), make_trainer(local)
    if confidence_alpha and confidence_mode in {"probabilistic", "all_invalid"}:
        torch.manual_seed(38)
        with torch.no_grad(), torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            diagnostics = backend.compute_loss(model, baseline._stage_standalone_cpu_batch(full), 0)["diagnostics"]
        if confidence_mode == "all_invalid":
            assert diagnostics["confidence_weighted_token_count"] == 0 < diagnostics["ce_weighted_token_count"]
        else:
            assert 0 < diagnostics["confidence_weighted_token_count"] < diagnostics["ce_weighted_token_count"]
    torch.manual_seed(38)
    assert asyncio.run(_reference_training_step_across_ranks(baseline, full, 0, device=torch.device("cpu")))
    baseline_rng = torch.random.get_rng_state().clone()
    store = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}})
    store.connect()
    entries = [ReadyEntry(make_sample_key(SampleMetadata.from_dict(tag)), tag) for tag in tags]
    assignment = TQLocalBatch([entry.key for entry in entries], [], [entry.key for entry in entries],
                              file_entries=entries, plan_id=0, receipt_token="receipt")
    events, token_counts = Queue(), {}
    torch.manual_seed(38)
    assert asyncio.run(_execute_owned_reference_batch(owned, store, assignment, 0, rank=0,
                                                      device=torch.device("cpu"), events=events,
                                                      transport_config={}, token_counts=token_counts,
                                                      packing=packing, packing_block_size=2))
    assert baseline.optimizer_steps_total == owned.optimizer_steps_total == 1
    assert {key: token_counts[key] for key in full["_speco_token_counts"]} == full["_speco_token_counts"]
    assert token_counts["window_train_input_positions"] > 0
    assert token_counts["window_statistics_input_positions"] == (
        token_counts["window_train_input_positions"]
        * (int(bool(confidence_alpha or lk_type == "adaptive_hybrid" or position_weight == "dpace"))
           + int(loss_mode == "sampled_ce" and negatives > 0)))
    if loss_mode == "sampled_ce":
        assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    assert events.get_nowait()["kind"] == "training_materialized"
    for a, b in zip(model.parameters(), local.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0.0002, atol=0.00005)
    if extreme_dpace:
        n_parts = (len(observed_denominators) - 1) // 2
        baseline_sum = observed_denominators[0][2]
        part_sum = sum(item[2] for item in observed_denominators[1 + n_parts:])
        assert part_sum == pytest.approx(baseline_sum, rel=0.003, abs=1e-12)
    if l1_alpha or lk_alpha or confidence_alpha:
        invalid = dict(full)
        invalid.pop("target_last_hidden_states")
        invalid["preselected_anchor_positions"] = torch.zeros((full["input_ids"].size(0), 1), dtype=torch.long)
        invalid["preselected_anchor_keep_mask"] = torch.zeros((full["input_ids"].size(0), 1), dtype=torch.bool)
        with pytest.raises(RuntimeError, match="materialization or staging failed"):
            asyncio.run(_reference_training_parts_across_ranks(owned, [invalid], 1, device=torch.device("cpu")))
        assert owned.optimizer_steps_total == 1
    assert all(Path(tag["hidden_states_path"]).exists() for tag in tags)
    if extreme_dpace == "zero":
        assert any(ce == 0 and auxiliary > 0 for ce, auxiliary, _ in observed_denominators)
        assert any(ce > 0 for ce, _, _ in observed_denominators)
    if extreme_dpace == "tiny":
        assert 0 < observed_denominators[0][0] < 1e-6
    store.close()


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("ce_target_source", ["labels", "target_argmax"])
@pytest.mark.parametrize("l1,lk,lk_type,confidence,mode", [
    (0.0, 0.0, "alpha", 0.0, "greedy"), (0.9, 0.0, "alpha", 0.0, "greedy"),
    (0.0, 0.3, "alpha", 0.0, "greedy"), (0.9, 0.3, "adaptive_hybrid", 0.4, "probabilistic"),
    (0.0, 0.0, "alpha", 0.4, "all_invalid"),
])
def test_owned_dpace_matches_full_batch_optimizer(tmp_path, monkeypatch, packing, ce_target_source,
                                                  l1, lk, lk_type, confidence, mode):
    test_owned_file_training_matches_full_batch_optimizer(
        tmp_path, monkeypatch, ce_target_source, l1, lk, lk_type, confidence, mode, packing,
        position_weight="dpace")


@pytest.mark.parametrize("packing", [False, True])
def test_owned_dpace_zero_alpha_preserves_auxiliary_loss_for_zero_ce_part(tmp_path, monkeypatch, packing):
    test_owned_file_training_matches_full_batch_optimizer(
        tmp_path, monkeypatch, "labels", 0.9, 0.3, "alpha", 0.0, "greedy", packing,
        position_weight="dpace", dpace_alpha=0.0, extreme_dpace="zero")


def test_owned_dpace_tiny_total_ce_weight_matches_full_batch_numerator(tmp_path, monkeypatch):
    test_owned_file_training_matches_full_batch_optimizer(
        tmp_path, monkeypatch, "labels", 0.9, 0.3, "alpha", 0.0, "greedy", False,
        position_weight="dpace", dpace_alpha=0.0, extreme_dpace="tiny")


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("position_weight,l1,lk,lk_type,confidence,mode", [
    ("fixed_exp_decay", 0.0, 0.0, "alpha", 0.0, "greedy"),
    ("fixed_exp_decay", 0.9, 0.3, "adaptive_hybrid", 0.4, "probabilistic"),
    ("dpace", 0.9, 0.3, "adaptive_hybrid", 0.4, "probabilistic"),
])
def test_owned_restricted_ce_matches_full_batch_optimizer(tmp_path, monkeypatch, packing, position_weight,
                                                           l1, lk, lk_type, confidence, mode):
    test_owned_file_training_matches_full_batch_optimizer(
        tmp_path, monkeypatch, "labels", l1, lk, lk_type, confidence, mode, packing,
        position_weight=position_weight, loss_mode="restricted_ce")


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("position_weight,l1,lk,lk_type,confidence,mode", [
    ("fixed_exp_decay", 0.0, 0.0, "alpha", 0.0, "greedy"),
    ("fixed_exp_decay", 0.9, 0.3, "adaptive_hybrid", 0.4, "probabilistic"),
    ("dpace", 0.9, 0.3, "adaptive_hybrid", 0.4, "probabilistic"),
])
def test_owned_sampled_ce_matches_full_batch_optimizer(tmp_path, monkeypatch, packing, position_weight,
                                                        l1, lk, lk_type, confidence, mode):
    test_owned_file_training_matches_full_batch_optimizer(
        tmp_path, monkeypatch, "labels", l1, lk, lk_type, confidence, mode, packing,
        position_weight=position_weight, loss_mode="sampled_ce")


def test_owned_sampled_ce_with_no_negatives_preserves_rng(tmp_path, monkeypatch):
    test_owned_file_training_matches_full_batch_optimizer(
        tmp_path, monkeypatch, "labels", 0.0, 0.0, "alpha", 0.0, "greedy", False,
        loss_mode="sampled_ce", negatives=0)


def _owned_file_step_gloo_worker(rank, port, directory, fail, output):
    import asyncio
    import datetime
    from contextlib import nullcontext
    from copy import deepcopy
    import torch.distributed as dist
    import verl_speco.trainer.base_trainer as trainer_module
    from verl_speco.backends.dspark_trainer_backend import DSparkTrainerBackend, DSparkTrainingModel
    from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
    from verl_speco.trainer.base_trainer import DrafterBaseTrainer
    from verl_speco.trainer.draft_training_loop import (_owned_reference_training_step,
        _reference_training_step_across_ranks, _zero_standalone_padding_batch)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        if fail == "node_dpace_zero":
            import torch.nn.functional as functional
            original_nll = functional.nll_loss
            def extreme_nll(log_probs, targets, *args, **kwargs):
                result = original_nll(log_probs, targets, *args, **kwargs)
                return result + (targets.remainder(2) == 0).to(result.dtype) * 200.0
            functional.nll_loss = extreme_nll
        path = Path(directory) / str(rank)
        path.mkdir()
        contract = _contract("DSPARK")
        tag, _ = _sample(path, contract, length=8)
        tags = [tag]
        if fail == "sampled_invalid" and rank == 1:
            from safetensors.torch import load_file
            payload = load_file(tag["hidden_states_path"])
            payload["hidden_states"][:, -1, :] = float("nan")
            save_file(payload, tag["hidden_states_path"])
        if fail in {"packed", "node_packed"} and rank == 0:
            tags.append(_sample(path, contract, length=6, index=1)[0])
        full = prepare_standalone_cpu_batch(tags, contract, pin_memory=False, streaming=True,
                                             packing=fail in {"packed", "node_packed"}, block_size=2)
        if rank == 1 and fail != "sampled_invalid":
            full = _zero_standalone_padding_batch(full)
        config = DSparkConfig(hidden_size=2, intermediate_size=8, num_hidden_layers=1,
                              num_attention_heads=1, num_key_value_heads=1, head_dim=2,
                              vocab_size=32, num_target_layers=4, num_context_layers=2,
                              target_hidden_size=2, target_num_hidden_layers=4, target_layer_ids=[0, 1],
                              mask_token_id=31, markov_rank=2, markov_head_type="vanilla",
                              sliding_window=3, use_sliding_window=True, layer_types=["sliding_attention"])
        torch.manual_seed(731)
        model = DSparkTrainingModel(draft_model=DSparkDraftModel(config), block_size=2,
                                    num_anchors=3, l1_loss_alpha=0.9 if fail in {"l1", "packed", "adaptive", "sampled_aux", "node_aux", "node_sampled_aux", "node_dpace_zero"} else 0.0,
                                    lk_loss_alpha=0.3 if fail in {"l1", "packed", "adaptive", "sampled_aux", "node_aux", "node_sampled_aux", "node_dpace_zero"} else 0.0,
                                    lk_loss_type="adaptive_hybrid" if fail in {"adaptive", "sampled_aux", "node_aux", "node_sampled_aux", "node_dpace_zero"} else "alpha",
                                    per_position_loss_weight="dpace" if fail == "node_dpace_zero" else "fixed_exp_decay",
                                    dpace_alpha=0.0 if fail == "node_dpace_zero" else 0.5,
                                    loss_mode="restricted_ce" if fail == "node_restricted" else "sampled_ce" if fail in {"sampled", "sampled_aux", "sampled_changed", "sampled_invalid", "sampled_mismatch", "node_sampled", "node_sampled_aux"} else "full_vocab",
                                    sampled_ce_negatives=0 if fail == "sampled_mismatch" and rank == 1 else 9,
                                    verifier_norm=torch.nn.LayerNorm(2))
        baseline = deepcopy(model)
        backend = DSparkTrainerBackend.__new__(DSparkTrainerBackend)
        backend.target_lm_head = SimpleNamespace(fc=SimpleNamespace(weight=torch.randn(32, 2)))
        backend.config = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={
            "mode": "offline", "feature_store": {"type": "ray_reference"},
            "transfer_queue": {"expected_feature": {"payload_kind": "vllm_reference"}},
        })))
        trainer_module.device_name = "cpu"
        def make_trainer(raw):
            trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
            trainer.model = torch.nn.parallel.DistributedDataParallel(raw, find_unused_parameters=True)
            trainer.backend, trainer.config = backend, backend.config
            trainer.runtime_device, trainer.rank = torch.device("cpu"), rank
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
            trainer.lr_scheduler = None
            trainer.optimizer_steps_total = trainer.training_steps = 0
            trainer._standalone_validate_grad_graph = True
            trainer.combine_standalone_metrics = False
            trainer._training_metric_sums, trainer._training_metric_steps = {}, 0
            trainer.record_training_timing = lambda *args: None
            trainer._ulysses_group_context = nullcontext
            trainer._record_dflash_training_metrics = lambda losses: {}
            trainer._clip_training_grad_norm = lambda: torch.tensor(0.0)
            trainer._is_checkpoint_leader = lambda: False
            def reduce_statistics(values):
                dist.all_reduce(values)
                return values, 2
            trainer._reduce_loss_statistics = reduce_statistics
            return trainer
        trainer, reference = make_trainer(model), make_trainer(baseline)
        node_context = None
        if isinstance(fail, str) and fail.startswith("node_"):
            import verl_speco.trainer.reference_shared_frame as shared
            from verl_speco.trainer.tq_feature_store import ReadyEntry
            from verl_speco.trainer.tq_sample_source import TQLocalBatch
            from verl_speco.trainer.draft_training_loop import _next_batch_across_ranks
            node_group = dist.new_group(backend="gloo")
            all_tags = [None, None]
            dist.all_gather_object(all_tags, tags)
            entries = ({peer: [ReadyEntry(f"{peer}-{index}", tag) for index, tag in enumerate(peer_tags)]
                        for peer, peer_tags in enumerate(all_tags)} if rank == 0 else None)
            if fail == "node_invalid_assignment" and rank == 0:
                del entries[1]
            node_context = {"rank": rank, "ranks_per_host": 2, "entries": entries,
                            "run_id": "run", "plan_id": 0,
                            "receipt_token": "wrong" if fail == "node_bad_identity" and rank == 1 else "receipt"}
            if fail not in {"node_invalid_assignment", "node_bad_identity"}:
                local = [ReadyEntry(f"{rank}-{index}", tag) for index, tag in enumerate(tags)]
                assignment = TQLocalBatch([entry.key for entry in local] if rank == 0 else [], [], [],
                                          file_entries=local, node_file_entries=entries, is_padding_rank=rank == 1,
                                          plan_id=0, receipt_token="receipt", assignment_digest="a" * 64)
                assignment = _next_batch_across_ranks(iter([assignment]), rank=rank, device=torch.device("cpu"),
                                                      require_reference_metadata=True, node_ranks_per_host=2)
                node_context["entries"] = assignment.node_file_entries
            def serve(groups, *, binding, prepare, max_bytes, timeout):
                if fail == "node_failure":
                    dist.send_object_list([{"injected_error": True}], dst=1, group=node_group)
                    raise OSError("injected owner read failure")
                remote, _ = prepare(groups[1])
                dist.send_object_list([remote], dst=1, group=node_group)
                return prepare(groups[0])
            def receive(*, binding, owner_rank, max_bytes, timeout):
                value = [None]
                dist.recv_object_list(value, src=owner_rank, group=node_group)
                if value[0].get("injected_error"):
                    raise OSError("owner failed to prepare window")
                return value[0], 0.0
            shared.serve_node_batches, shared.receive_node_batch = serve, receive
        if fail not in {"scan", "materialize", "sampled_mismatch"}:
            torch.manual_seed(38)
            asyncio.run(_reference_training_step_across_ranks(reference, full, 0, device=torch.device("cpu")))
            baseline_rng = torch.random.get_rng_state().clone()
        if fail == "scan" and rank == 1:
            Path(tag["hidden_states_path"]).unlink()
        elif fail == "materialize" and rank == 1:
            import verl_speco.trainer.standalone_batch_prepare as prepare
            original_sample = prepare.sample_owned_anchor_windows
            def sample_then_remove(*args, **kwargs):
                result = original_sample(*args, **kwargs)
                Path(tag["hidden_states_path"]).unlink()
                return result
            prepare.sample_owned_anchor_windows = sample_then_remove
        elif fail == "sampled_changed" and rank == 0:
            import verl_speco.trainer.standalone_batch_prepare as prepare
            original_sample, original_prepare = prepare.sample_owned_anchor_windows, prepare.prepare_owned_anchor_window
            state = {"parts": 0, "reads": 0}
            def remember_parts(*args, **kwargs):
                result = original_sample(*args, **kwargs)
                state["parts"] = len(result[2])
                return result
            def change_after_probe(*args, **kwargs):
                result = original_prepare(*args, **kwargs)
                state["reads"] += 1
                if state["reads"] > state["parts"]:
                    result["hidden_states"] = result["hidden_states"].clone()
                    result["hidden_states"][0, 0, 0] += 1
                return result
            prepare.sample_owned_anchor_windows, prepare.prepare_owned_anchor_window = remember_parts, change_after_probe
        seen = []
        train = trainer.training_step_from_batch
        async def record_train(batch, step, **kwargs):
            seen.append(float(batch["loss_mask"].sum()))
            return await train(batch, step, **kwargs)
        trainer.training_step_from_batch = record_train
        try:
            torch.manual_seed(38)
            asyncio.run(_owned_reference_training_step(trainer, tags, contract, 0, device=torch.device("cpu"),
                                                        windows=[3], padding_rank=rank == 1 and fail != "sampled_invalid",
                                                        packing=fail in {"packed", "node_packed"}, packing_block_size=2,
                                                        node_context=node_context, max_host_bytes=1024 * 1024 if node_context else 0))
            if fail in {"sampled", "sampled_aux", "sampled_invalid", "node_sampled", "node_sampled_aux"}:
                assert torch.equal(torch.random.get_rng_state(), baseline_rng)
            difference = max(float((a - b).abs().max()) for a, b in zip(model.parameters(), baseline.parameters(), strict=True))
            output.put((rank, "ok", seen, Path(tag["hidden_states_path"]).exists(), trainer.optimizer_steps_total, difference))
        except RuntimeError as exc:
            output.put((rank, str(exc), seen, Path(tag["hidden_states_path"]).exists(), trainer.optimizer_steps_total, None))
    finally:
        dist.destroy_process_group()


def _collect_gloo_workers(target, arguments, *, count, timeout, join_timeout):
    import multiprocessing
    import socket
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    ctx = multiprocessing.get_context("spawn")
    output = ctx.Queue()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes = [ctx.Process(target=target, args=arguments(rank, port, output)) for rank in range(count)]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        rows = sorted(output.get(timeout=timeout) for _ in processes)
        for process in processes:
            process.join(timeout=join_timeout)
        assert all(process.exitcode == 0 for process in processes)
        return rows
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        output.close()


@pytest.mark.parametrize("fail", [None, "l1", "packed", "adaptive", "scan", "materialize", "sampled", "sampled_aux", "sampled_changed", "sampled_invalid", "sampled_mismatch", "node_basic", "node_packed", "node_sampled", "node_failure", "node_invalid_assignment", "node_aux", "node_restricted", "node_sampled_aux", "node_bad_identity", "node_dpace_zero"])
def test_owned_file_step_two_gloo_ranks(tmp_path, fail):
    rows = _collect_gloo_workers(_owned_file_step_gloo_worker,
        lambda rank, port, output: (rank, port, str(tmp_path), fail, output), count=2, timeout=90, join_timeout=20)
    if fail in {"scan", "materialize", "sampled_changed", "sampled_mismatch", "node_failure", "node_invalid_assignment", "node_bad_identity"}:
        expected_error = ("planning failed" if fail == "scan" else "Node-owned window plan is invalid"
                          if fail == "node_invalid_assignment" else "Sampled vocabulary probe differs"
                          if fail == "sampled_mismatch" else "Node-owned window assignments differ"
                          if fail == "node_bad_identity" else "materialization or staging failed")
        assert all(expected_error in row[1] and not row[2] and row[4] == 0 for row in rows)
        assert rows[0][3]
    else:
        assert all(row[1] == "ok" and row[3] for row in rows)
        assert len(rows[0][2]) == len(rows[1][2]) > 1
        assert sum(rows[0][2]) > 0
        assert (sum(rows[1][2]) > 0) if fail == "sampled_invalid" else (sum(rows[1][2]) == 0)
        assert all(row[4] == 1 and row[5] < 0.0002 for row in rows)


def _node_window_four_rank_worker(rank, port, output):
    import datetime
    import torch.distributed as dist
    import verl_speco.trainer.reference_shared_frame as shared
    import verl_speco.trainer.standalone_batch_prepare as prepare
    from verl_speco.trainer.draft_training_loop import _node_owned_window_parts
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=35))
    try:
        owner = rank // 2 * 2
        entries = ({peer: [SimpleNamespace(tag={"rank": peer})] for peer in range(owner, owner + 2)}
                   if rank == owner else None)
        context = {"rank": rank, "ranks_per_host": 2, "entries": entries,
                   "run_id": "run", "plan_id": 7, "receipt_token": "receipt"}
        plan = {"sample_index": 0, "start": 0, "end": 4, "anchor": rank + 1, "source_window": [0, 4]}
        plans = [plan] if rank % 2 else [plan, {**plan, "anchor": rank + 10}]
        prepare.prepare_owned_anchor_window = lambda tag, contract, **kw: {"rank": tag["rank"], "anchor": kw["anchor"]}
        prepare.prepare_empty_owned_window = lambda tag, contract, **kw: {"rank": tag["rank"], "anchor": -1}
        sent = []
        def serve(groups, *, binding, prepare, max_bytes, timeout):
            peer = rank + 1
            sent.append(binding["receipt_token"])
            value, _ = prepare(groups[peer])
            dist.send_object_list([({**binding, "rank": peer}, value)], dst=peer)
            return prepare(groups[rank])
        def receive(*, binding, owner_rank, max_bytes, timeout):
            value = [None]
            dist.recv_object_list(value, src=owner_rank)
            assert value[0][0] == binding
            return value[0][1], 0.0
        shared.serve_node_batches, shared.receive_node_batch = serve, receive
        parts = _node_owned_window_parts(plans, 2, context, "contract", windows=[3], block_size=2,
                                         read_timeout=30.0, max_host_bytes=4096, device=torch.device("cpu"))
        values = []
        for index in (0, 1, 0, 1):
            values.append(parts[index]())
            dist.barrier()
        output.put((rank, values, sent))
    finally:
        dist.destroy_process_group()


def test_node_window_routing_isolated_across_two_node_groups():
    rows = _collect_gloo_workers(_node_window_four_rank_worker,
        lambda rank, port, output: (rank, port, output), count=4, timeout=60, join_timeout=10)
    for rank, values, sent in rows:
        expected = [rank + 1, rank + 10 if rank % 2 == 0 else -1] * 2
        assert [value["anchor"] for value in values] == expected
        assert all(value["rank"] == rank for value in values)
        assert len(set(sent)) == (4 if rank % 2 == 0 else 0)


@pytest.mark.parametrize("mode", ["ok", "error", "timeout"])
def test_native_producer_uses_source_owned_control_service(tmp_path, monkeypatch, mode):
    import asyncio
    import json
    from queue import Queue
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    from verl_speco.transport.drafter_sample_protocol import make_sample_key
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport, NativeSourceControlService

    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0
    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="source", metadata_server="metadata",
                                    device="cpu", max_bytes=4 * 1024**2, max_slots=1,
                                    chunk_bytes=2 * 1024**2, engine_factory=Engine)
    descriptor = source.publish({"ids": torch.arange(2)}, lease_id="sample")
    file_tag, _ = _sample(tmp_path, _contract("DSPARK"), length=8)
    tag = make_reference_tag(SampleMetadata.from_dict(file_tag), backend="native",
                             handle=json.dumps(descriptor, sort_keys=True, separators=(",", ":")),
                             request=file_tag["request"], feature_contract=file_tag["feature_contract"])
    key = make_sample_key(SampleMetadata.from_dict(tag))
    requests, responses = Queue(maxsize=1), Queue(maxsize=1)
    options = {"request_queue": requests, "response_queue": responses, "channel_id": "a" * 32, "timeout": 2.0}
    service = NativeSourceControlService(source, **options)
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {"max_published_references": 1}}},
                                    Queue(), reference_batch_size=1, native_transport=options)
    original_ack = source.acknowledge
    async def run():
        await actor._on_references([{"key": key, "tag": tag}])
        await actor.register_dispatch(actor._incarnation, 0, "receipt", [key], 1)
        if mode == "error":
            monkeypatch.setattr(source, "acknowledge", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
            with pytest.raises(RuntimeError, match="control operation failed"):
                await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
            assert actor._reference_aborted and key in actor._active_references and "sample" in source._slots
            with pytest.raises(RuntimeError, match="retain source"):
                service.close()
            original_ack(descriptor, reader=0, plan_id=0, receipt_token="receipt")
            source.release(descriptor, plan_id=0, receipt_token="receipt")
        elif mode == "timeout":
            from threading import Event
            proceed = Event()
            def delayed_ack(*args, **kwargs):
                assert proceed.wait(5)
                return original_ack(*args, **kwargs)
            monkeypatch.setattr(source, "acknowledge", delayed_ack)
            actor._native_transport.timeout = 0.05
            with pytest.raises(TimeoutError, match="deadline"):
                await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
            assert actor._reference_aborted and key in actor._active_references and "sample" in source._slots
            proceed.set()
            service.close()
            source.release(descriptor, plan_id=0, receipt_token="receipt")
        else:
            await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
            await actor.release_references(actor._incarnation, 0, "receipt", [key])
            assert not actor._active_references and not source._slots
            service.close()
    asyncio.run(run())
    source.close()


@pytest.mark.parametrize("failure", [None, "bind", "ack", "release", "discard", "discard_failure"])
def test_native_source_follows_producer_dispatch_and_rank_ack(tmp_path, monkeypatch, failure):
    import asyncio
    import json
    import threading
    from queue import Queue
    from verl_speco.producer.vllm_feature_client import VllmResponse
    from verl_speco.standalone_ray_runtime import StandaloneProducerActor
    from verl_speco.standalone_tq_producer import _reference_tag
    from verl_speco.transport.drafter_sample_protocol import make_sample_key
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport

    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, *args): return 0
        def unregister_memory(self, *args): return 0

    source = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                    metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                    max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    other_source = NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                          metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                          max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    contract = _contract("DSPARK")
    file_tag, _ = _sample(tmp_path, contract, length=8)
    descriptor = source.publish({"hidden": torch.ones(2, 2)}, lease_id="sample")
    request = SimpleNamespace(sample_id=file_tag["sample_id"], sequence_no=file_tag["sequence_no"],
                              prompt_token_ids=file_tag["request"]["vllm_prompt_token_ids"],
                              source_metadata=file_tag["request"]["source_metadata"],
                              **{name: torch.tensor(file_tag["request"][name]) for name in
                                 ("input_ids", "loss_mask", "position_ids", "feature_positions", "draft_position_ids")})
    response = VllmResponse(None, "http://vllm:8000/v1", store=SimpleNamespace(backend="native"),
                            handle=json.dumps(descriptor, sort_keys=True, separators=(",", ":")))
    tag = _reference_tag(request, response, contract, "run", {"schema_version": PROTOCOL_SCHEMA_VERSION})
    assert tag["backend"] == "native"
    key = make_sample_key(SampleMetadata.from_dict(tag))
    actor = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
        "max_published_references": 1}}}, Queue(), reference_batch_size=1, native_transport=source)
    original_ack = source.acknowledge
    original_release = source.release
    original_bind = source.bind_readers

    async def run():
        unconfigured = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_published_references": 1}}}, Queue(), reference_batch_size=1)
        with pytest.raises(ValueError, match="in-process source"):
            await unconfigured._on_references([{"key": key, "tag": tag}])
        foreign = StandaloneProducerActor({"speco": {"standalone_tq_producer": {
            "max_published_references": 1}}}, Queue(), reference_batch_size=1, native_transport=other_source)
        with pytest.raises(ValueError, match="not an active source"):
            await foreign._on_references([{"key": key, "tag": tag}])
        with pytest.raises(ValueError, match="duplicate native"):
            await actor._on_references([{"key": key, "tag": tag}, {"key": "alias", "tag": tag}])
        assert not actor._active_references and "sample" in source._slots
        if failure is None:
            entered, proceed = threading.Event(), threading.Event()
            original_validate = source.validate_published
            def delayed_validate(value):
                original_validate(value)
                entered.set()
                assert proceed.wait(10)
            monkeypatch.setattr(source, "validate_published", delayed_validate)
            publication = asyncio.create_task(actor._on_references([{"key": key, "tag": tag}]))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                old_handle, tag["handle"] = tag["handle"], "changed"
            finally:
                proceed.set()
            await publication
            assert actor._events.get_nowait()["entries"][0]["tag"]["handle"] == old_handle
            tag["handle"] = old_handle
        else:
            await actor._on_references([{"key": key, "tag": tag}])
        if failure in {"discard", "discard_failure"}:
            await actor.stop()
            if failure == "discard_failure":
                original_discard = source.discard_unbound
                monkeypatch.setattr(source, "discard_unbound", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
                with pytest.raises(RuntimeError, match="failed"):
                    await actor.discard_unassigned_references(actor._incarnation, "discard", [key])
                assert actor._reference_aborted and key in actor._active_references and "sample" in source._slots
                original_discard(descriptor)
                return
            await actor.discard_unassigned_references(actor._incarnation, "discard", [key])
            await actor.discard_unassigned_references(actor._incarnation, "discard", [key])
            assert not source._slots and not actor._active_references and not actor._native_references
            return
        with pytest.raises(ValueError, match="per-rank"):
            await actor.register_dispatch(actor._incarnation, 0, "receipt", [key])
        if failure == "bind":
            monkeypatch.setattr(source, "bind_readers", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
            with pytest.raises(RuntimeError, match="failed"):
                await actor.register_dispatch(actor._incarnation, 0, "receipt", [key], 2)
            assert actor._reference_aborted and key in actor._active_references and "sample" in source._slots
            original_bind(descriptor, readers=[0, 1], plan_id=0, receipt_token="receipt")
            for rank in (0, 1):
                original_ack(descriptor, reader=rank, plan_id=0, receipt_token="receipt")
            original_release(descriptor, plan_id=0, receipt_token="receipt")
            return
        await actor.register_dispatch(actor._incarnation, 0, "receipt", [key], 2)
        await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 0)
        with pytest.raises(ValueError, match="every rank"):
            await actor.release_references(actor._incarnation, 0, "receipt", [key])
        assert "sample" in source._slots and key in actor._active_references
        if failure == "ack":
            monkeypatch.setattr(source, "acknowledge", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
            with pytest.raises(RuntimeError, match="failed"):
                await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 1)
            assert actor._reference_aborted and key in actor._active_references and "sample" in source._slots
            monkeypatch.setattr(source, "acknowledge", original_ack)
            original_ack(descriptor, reader=1, plan_id=0, receipt_token="receipt")
            source.release(descriptor, plan_id=0, receipt_token="receipt")
        else:
            await actor.acknowledge_reference_rank(actor._incarnation, 0, "receipt", 1)
            if failure == "release":
                monkeypatch.setattr(source, "release", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")))
                with pytest.raises(RuntimeError, match="failed"):
                    await actor.release_references(actor._incarnation, 0, "receipt", [key])
                assert actor._reference_aborted and key in actor._active_references and "sample" in source._slots
                original_release(descriptor, plan_id=0, receipt_token="receipt")
            else:
                await actor.release_references(actor._incarnation, 0, "receipt", [key])
                await actor.release_references(actor._incarnation, 0, "receipt", [key])
                assert not source._slots and not actor._active_references and not actor._native_references
    asyncio.run(run())
    source.close()
    other_source.close()


def test_native_reference_prepares_batch_and_preserves_source_until_ack(tmp_path):
    import ctypes
    import json
    from safetensors.torch import load_file
    from verl_speco.producer.hidden_states_store import NativeHiddenStatesStore
    from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples
    from verl_speco.trainer.tq_feature_store import ReadyEntry, ReferenceFeatureStore
    from verl_speco.transport.drafter_sample_protocol import make_sample_key
    from verl_speco.transport.native_feature_transport import NativeFeatureTransport
    registrations = {}
    class Engine:
        def initialize(self, *args): return 0
        def register_memory(self, address, size): registrations[address] = size; return 0
        def unregister_memory(self, address): del registrations[address]; return 0
        def transfer_sync_read(self, segment, local, remote, size):
            assert all(any(base <= address and address + size <= base + capacity
                           for base, capacity in registrations.items()) for address in (local, remote))
            ctypes.memmove(local, remote, size)
            return 0
    def transport():
        return NativeFeatureTransport(run_id="run", generation=1, endpoint="localhost:1234",
                                      metadata_server="metadata", device="cpu", max_bytes=4 * 1024**2,
                                      max_slots=1, chunk_bytes=2 * 1024**2, engine_factory=Engine)
    contract = _contract("DSPARK")
    file_tag, _ = _sample(tmp_path, contract, length=8)
    expected = prepare_standalone_cpu_batch([file_tag], contract, pin_memory=False, streaming=True)
    source, receiver = transport(), transport()
    descriptor = source.publish(load_file(file_tag["hidden_states_path"]), lease_id="sample")
    binding = {"plan_id": 3, "receipt_token": "receipt"}
    source.bind_readers(descriptor, readers=[0], **binding)
    handle = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    native_tag = make_reference_tag(SampleMetadata.from_dict(file_tag), backend="native", handle=handle,
                                    request=file_tag["request"], feature_contract=file_tag["feature_contract"])
    from verl_speco.transport.drafter_sample_protocol import validate_reference_envelope
    with pytest.raises(ValueError, match="scope"):
        validate_reference_envelope({**native_tag, "handle": json.dumps({**descriptor, "run_id": "other"})})
    with pytest.raises(ValueError, match="Malformed"):
        validate_reference_envelope({**native_tag, "handle": "x" * (64 * 1024 + 1)})
    with pytest.raises(ValueError, match="canonical"):
        validate_reference_envelope({**native_tag, "handle": json.dumps(descriptor)})
    with pytest.raises(ValueError, match="fingerprint"):
        validate_reference_envelope({**native_tag, "handle": json.dumps({**descriptor, "sha256": "0" * 64},
                                                                        sort_keys=True, separators=(",", ":"))})
    store = NativeHiddenStatesStore(receiver, lambda value: source.release(value, **binding))
    adapter = ReferenceFeatureStore({"run_id": "run", "ray": {"address": "unused"}}, hidden_states_store=store)
    adapter.connect()
    entry = ReadyEntry(make_sample_key(SampleMetadata.from_dict(native_tag)), native_tag)
    actual = adapter.prepare_batch([entry], pin_memory=False)
    for key, value in expected.items():
        assert torch.equal(actual[key], value) if isinstance(value, torch.Tensor) else actual[key] == value
    from verl_speco.standalone_ray_runtime import _accept_reference_entries, _release_reference_entries
    seen_keys, seen_references = {}, {}
    accepted = _accept_reference_entries([{"key": entry.key, "tag": native_tag}], run_id="run",
                                         seen_keys=seen_keys, seen_references=seen_references)
    other = {**native_tag, "sample_id": "other", "sequence_no": 1}
    with pytest.raises(ValueError, match="belongs to multiple samples"):
        _accept_reference_entries([{"key": make_sample_key(SampleMetadata.from_dict(other)), "tag": other}],
                                  run_id="run", seen_keys=seen_keys, seen_references=seen_references)
    assert cleanup_standalone_samples(adapter, [entry], attempts=1) == [handle]
    assert len(registrations) == 1
    source.acknowledge(descriptor, reader=0, **binding)
    assert cleanup_standalone_samples(adapter, [entry], attempts=1) == []
    _release_reference_entries(accepted, seen_references)
    assert not seen_references
    assert not registrations
    adapter.close()
    source.close()


def test_dspark_padded_batch_retains_raw_final_and_files(tmp_path):
    contract = _contract()
    a, ah = _sample(tmp_path, contract)
    b, bh = _sample(tmp_path, contract, length=3, index=1)
    state = torch.random.get_rng_state().clone()
    batch = prepare_standalone_cpu_batch([a, b], contract, pin_memory=False)
    assert torch.equal(state, torch.random.get_rng_state())
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert batch["_speco_token_counts"] == {
        "source_tokens": 8, "window_tokens": 8, "supervised_tokens": 6,
        "padding_tokens": 2, "samples": 2,
    }
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]]
    torch.testing.assert_close(batch["hidden_states"][0], ah[:, :2].flatten(1).bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][0], ah[:, 2].bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][1, :3], bh[:, 2].bfloat16())
    assert torch.equal(batch["label_mask"], batch["loss_mask"])
    assert batch["last_hidden_state_norm"] == "raw_target_residual"
    assert all(t.device.type == "cpu" for t in batch.values() if isinstance(t, torch.Tensor))
    assert Path(a["hidden_states_path"]).exists() and Path(b["hidden_states_path"]).exists()


def test_eagle3_shift_and_final_layer_reuse(tmp_path):
    contract = _contract("EAGLE3", reuse_final=True)
    tag, hidden = _sample(tmp_path, contract)
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    assert batch["input_ids"].tolist() == [[1, 2, 3]]
    assert batch["loss_mask"].tolist() == [[1, 1, 1]]
    assert batch["position_ids"].tolist() == [[1, 2, 3]]
    torch.testing.assert_close(batch["last_hidden_states"][0], hidden[1:4, 1].bfloat16())
    torch.testing.assert_close(batch["hidden_states"][0], hidden[:3].flatten(1).bfloat16())


def test_dflash_window_and_suffix_alignment(tmp_path):
    contract = _contract("DFLASH")
    tag, hidden = _sample(tmp_path, contract, length=8, positions=[3, 4, 5, 6, 7], hidden_offset=3)
    tag["request"]["loss_mask"] = [0, 0, 0, 0, 0, 0, 1, 1]
    batch = prepare_standalone_cpu_batch([tag], contract, max_window=3, pin_memory=False)
    assert batch["input_ids"].tolist() == [[5, 6, 7]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1]]
    torch.testing.assert_close(batch["hidden_states"][0], hidden[5:8].flatten(1).bfloat16())
    assert "last_hidden_state_norm" not in batch


def test_dflash_aux_plus_last_reference_batch_keeps_raw_final(tmp_path):
    contract = replace(_contract("DFLASH"), hidden_states_layout="dflash_aux_plus_last")
    tag, hidden = _sample(tmp_path, contract)
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    torch.testing.assert_close(batch["hidden_states"][0], hidden[:, :2].flatten(1).bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][0], hidden[:, 2].bfloat16())
    assert batch["last_hidden_state_norm"] == "raw_target_residual"
    assert batch["hidden_source"] == "standalone_vllm_reference"


def _old_pack(chunks, block_size, limit, *, preserve=False, next_labels=False):
    # Execute the unchanged production packer without importing distributed/model
    # dependencies. This checks parity against actual legacy code, not a copy.
    source = Path(__file__).parents[2] / "verl_speco/trainer/base_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DrafterBaseTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pack_block_drafter_chunks")
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch, "logger": logging.getLogger(__name__),
                 "cast": __import__("typing").cast}
    exec(compile(module, str(source), "exec", flags=__import__('__future__').annotations.compiler_flag), namespace)
    cfg = SimpleNamespace(rollout=SimpleNamespace(drafter=SimpleNamespace(training={"packing": {"max_packed_len": limit}})))
    trainer = SimpleNamespace(
        config=cfg, _block_drafter_config_value=lambda *args: block_size,
        _block_drafter_metric_prefix=lambda: "dspark",
        _standalone_preserve_packed_samples=preserve,
    )
    keys = ["input_ids", "loss_mask", "hidden_states", "position_ids", "target_last_hidden_states"]
    extra = ([[c["input_ids"] + 1 for c in chunks], [c["loss_mask"] for c in chunks]]
             if next_labels else [])
    return namespace[method.name](trainer, *[[c[k] for c in chunks] for k in keys], *extra)


@pytest.mark.parametrize("limit", [0, 16])
def test_document_packing_matches_existing_packer(tmp_path, limit):
    contract = _contract()
    a, _ = _sample(tmp_path, contract, length=5)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    chunks = []
    for tag in [a, b]:
        single = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
        chunks.append({k: v.squeeze(0) for k, v in single.items() if isinstance(v, torch.Tensor)})
    old = _old_pack(chunks, 4, limit)
    batch = prepare_standalone_cpu_batch(
        [a, b], contract, packing=True, block_size=4, max_packed_len=limit, pin_memory=False,
    )
    for key, expected in zip(
        ["input_ids", "loss_mask", "hidden_states", "position_ids", "attention_mask", "document_ids",
         "target_last_hidden_states"], old[:7],
    ):
        torch.testing.assert_close(batch[key], expected)
    assert batch["document_ids"][0, 5:8].tolist() == [-1, -1, -1]


def test_document_packing_uses_multiple_rows_without_dropping_samples(tmp_path):
    contract = _contract()
    a, _ = _sample(tmp_path, contract, length=5)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    batch = prepare_standalone_cpu_batch(
        [a, b], contract, packing=True, block_size=4, max_packed_len=8, pin_memory=False,
    )
    assert batch["input_ids"].shape == (2, 8)
    assert batch["_speco_token_counts"]["padding_tokens"] == 8
    assert batch["document_ids"][0].tolist() == [0, 0, 0, 0, 0, -1, -1, -1]
    assert batch["document_ids"][1].tolist() == [0, 0, 0, -1, -1, -1, -1, -1]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert batch["input_ids"][1, :3].tolist() == [0, 1, 2]
    chunks = []
    for tag in [a, b]:
        single = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
        chunks.append({k: v.squeeze(0) for k, v in single.items() if isinstance(v, torch.Tensor)})
    packed = _old_pack(chunks, 4, 8, preserve=True, next_labels=True)
    assert packed[-1] == 2
    torch.testing.assert_close(packed[0], batch["input_ids"])
    torch.testing.assert_close(packed[5], batch["document_ids"])
    assert packed[7][1, :3].tolist() == [1, 2, 3]
    assert packed[8][1, 3:].sum() == 0
    from verl_speco.backends.dflash_trainer_backend import _document_boundary_validity

    indices = torch.arange(7).expand(2, -1)
    kept = _document_boundary_validity(torch.ones_like(indices, dtype=torch.bool),
        indices=indices, document_ids=batch["document_ids"], block_size=2, seq_len=8, label_shift=1)
    assert kept[0, 2] and not kept[0, 3]
    assert kept[1, 0] and not kept[1, 1]
    with pytest.raises(ValueError, match="exceeds max_packed_len"):
        prepare_standalone_cpu_batch([a], contract, packing=True, block_size=4,
                                     max_packed_len=4, pin_memory=False)


def test_pin_happens_after_collation(tmp_path, monkeypatch):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    shapes = []

    def pin(tensor):
        assert tensor.device.type == "cpu"
        shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    batch = prepare_standalone_cpu_batch([tag], contract)
    assert len(shapes) == sum(isinstance(v, torch.Tensor) for v in batch.values())
    assert (1, 5, 4) in shapes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real pinned allocation requires CUDA runtime")
def test_real_pinned_batch(tmp_path):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    batch = prepare_standalone_cpu_batch([tag], contract)
    assert all(t.is_pinned() and t.device.type == "cpu" for t in batch.values() if torch.is_tensor(t))


def test_invalid_contract_rejected_before_file_read(tmp_path):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    Path(tag["hidden_states_path"]).unlink()
    with pytest.raises(ValueError, match="contract mismatch"):
        prepare_standalone_cpu_batch([tag], replace(contract, target_model_id="other"), pin_memory=False)


def test_file_prefetch_leaves_nonfinite_handling_to_training(tmp_path, monkeypatch):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    hidden[0, 0, 0] = float("nan")
    tag["hidden_states_path"] = str(tmp_path / "nonfinite.safetensors")
    save_file({"token_ids": torch.arange(5), "hidden_states": hidden}, tag["hidden_states_path"])
    monkeypatch.setattr(torch, "isfinite", lambda *args, **kwargs: pytest.fail("CPU full hidden scan"))
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    assert torch.isnan(batch["hidden_states"][0, 0, 0])
    assert Path(tag["hidden_states_path"]).exists()


def test_legacy_conversion_still_rejects_nonfinite_hidden(tmp_path):
    contract = _contract()
    _, hidden = _sample(tmp_path, contract)
    hidden[0, 0, 0] = float("nan")
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.arange(5),
        draft_position_ids=torch.arange(1, 6), source_metadata={},
    )
    with pytest.raises(HiddenStateAlignmentError, match="NaN/Inf"):
        feature_from_vllm_payload({"token_ids": torch.arange(5), "hidden_states": hidden},
                                 request, contract, normalize_final_hidden=False)


def test_cpu_workers_prepare_assigned_samples_in_parallel_and_keep_order(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    contract = _contract()
    a, _ = _sample(tmp_path, contract, length=5)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    expected = prepare_standalone_cpu_batch([a, b], contract, pin_memory=False)
    barrier = Barrier(2)
    original = FileHiddenStatesStore.load
    calls = []

    def load(store, path, *, timeout):
        assert not torch.is_grad_enabled()
        calls.append(path)
        barrier.wait(timeout=3)  # Sequential execution cannot pass this barrier.
        return original(store, path, timeout=timeout)

    monkeypatch.setattr(FileHiddenStatesStore, "load", load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = prepare_standalone_cpu_batch([a, b], contract, pin_memory=False, executor=executor)
    assert len(calls) == 2
    for key, tensor in expected.items():
        if torch.is_tensor(tensor):
            torch.testing.assert_close(actual[key], tensor)


def test_raw_conversion_preserves_old_normalized_default(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.arange(5),
        draft_position_ids=torch.arange(1, 6), source_metadata={},
    )
    payload = {"token_ids": torch.arange(5), "hidden_states": hidden}
    norm = torch.nn.LayerNorm(2).bfloat16()
    old = feature_from_vllm_payload(payload, request, contract, final_norm=norm)
    raw = feature_from_vllm_payload(payload, request, contract, normalize_final_hidden=False)
    assert old.metadata["last_hidden_state_norm"] == "target_final_norm"
    assert raw.metadata["last_hidden_state_norm"] == "raw_target_residual"
    torch.testing.assert_close(old.hidden_states[:, :4], raw.hidden_states[:, :4])
    torch.testing.assert_close(old.hidden_states[:, 4:], norm(raw.hidden_states[:, 4:]))
    with pytest.raises(ValueError, match="require the target final norm"):
        feature_from_vllm_payload(payload, request, contract)
    # Preserve both auxiliary values and raw targets for training sanitation.
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    torch.testing.assert_close(batch["hidden_states"][0], hidden[:, :2].flatten(1).bfloat16())
    torch.testing.assert_close(batch["target_last_hidden_states"][0], hidden[:, 2].bfloat16())


def test_token_mismatch_and_insufficient_rows_preserve_files(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    save_file({"token_ids": torch.arange(5) + 1, "hidden_states": hidden}, tag["hidden_states_path"])
    with pytest.raises(HiddenStateAlignmentError, match="token_ids"):
        prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    save_file({"token_ids": torch.arange(5), "hidden_states": hidden[2:]}, tag["hidden_states_path"])
    with pytest.raises(HiddenStateAlignmentError, match="complete feature window"):
        prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    assert Path(tag["hidden_states_path"]).exists()


@pytest.mark.parametrize("algorithm,packing", [
    ("EAGLE3", False), ("DSPARK", False), ("DSPARK", True),
    ("DFLASH", False), ("DFLASH", True),
])
def test_contiguous_file_batch_avoids_hidden_gather_and_cat(tmp_path, monkeypatch, algorithm, packing):
    contract = _contract(algorithm)
    a, _ = _sample(tmp_path, contract)
    b, _ = _sample(tmp_path, contract, length=3, index=1)
    original = torch.Tensor.index_select

    def select(tensor, dimension, indices):
        assert tensor.dim() != 3, "Contiguous hidden rows must use a view"
        return original(tensor, dimension, indices)

    monkeypatch.setattr(torch.Tensor, "index_select", select)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("Intermediate cat in file preparation"))
    batch = prepare_standalone_cpu_batch(
        [a, b], contract, packing=packing, block_size=4, pin_memory=False,
    )
    assert batch["hidden_states"].dtype == torch.bfloat16
    assert batch["hidden_states"].shape[-1] == 4
    assert batch["hidden_states"].is_contiguous()


@pytest.mark.parametrize("algorithm", ["EAGLE3", "DSPARK", "DFLASH"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_split_features_preserve_legacy_values_and_sparse_order(tmp_path, algorithm, dtype):
    contract = replace(_contract(algorithm), dtype=dtype)
    tag, hidden = _sample(tmp_path, contract, positions=[0, 2, 4])
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.tensor([0, 2, 4]),
        draft_position_ids=torch.tensor([1, 3, 5]), source_metadata={},
    )
    payload = {"token_ids": torch.arange(5), "hidden_states": hidden}
    legacy = feature_from_vllm_payload(payload, request, contract, normalize_final_hidden=False)
    split = feature_from_vllm_payload(payload, request, contract,
                                     normalize_final_hidden=False, split_hidden_states=True)
    torch.testing.assert_close(split.input_ids, legacy.input_ids)
    torch.testing.assert_close(split.loss_mask, legacy.loss_mask)
    torch.testing.assert_close(split.hidden_states.flatten(1).bfloat16(), legacy.hidden_states[:, :4].bfloat16())
    if algorithm != "DFLASH":
        torch.testing.assert_close(split.last_hidden_states.bfloat16(), legacy.hidden_states[:, 4:].bfloat16())
    batch = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    expected = legacy.hidden_states[:, :4].bfloat16()
    if algorithm == "EAGLE3":
        expected = expected[:1]
    torch.testing.assert_close(batch["hidden_states"][0], expected)


def test_split_contiguous_features_share_source_storage(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract)
    request = SimpleNamespace(
        input_ids=torch.arange(5), prompt_token_ids=list(range(5)),
        loss_mask=torch.tensor([0, 1, 1, 1, 1]), feature_positions=torch.arange(5),
        draft_position_ids=torch.arange(1, 6), source_metadata={},
    )
    split = feature_from_vllm_payload(
        {"token_ids": torch.arange(5), "hidden_states": hidden}, request, contract,
        normalize_final_hidden=False, split_hidden_states=True,
    )
    pointer = hidden.untyped_storage().data_ptr()
    assert split.hidden_states.untyped_storage().data_ptr() == pointer
    assert split.last_hidden_states.untyped_storage().data_ptr() == pointer


@pytest.mark.parametrize("algorithm,packing", [("DSPARK", False), ("DSPARK", True), ("DFLASH", False), ("EAGLE3", False)])
def test_mooncake_reference_reuses_cpu_prepare_and_cleanup(tmp_path, monkeypatch, algorithm, packing):
    from verl_speco.producer import hidden_states_store as stores
    from verl_speco.trainer.feature_store import build_feature_store_from_config
    from verl_speco.trainer.standalone_sample_cleanup import cleanup_standalone_samples
    from verl_speco.trainer.tq_feature_store import ReadyEntry
    from verl_speco.transport.drafter_sample_protocol import SampleMetadata, make_sample_key

    contract = _contract(algorithm)
    tag, hidden = _sample(tmp_path, contract)
    expected = prepare_standalone_cpu_batch([tag], contract, packing=packing, block_size=2, pin_memory=False)
    tag.pop("hidden_states_path")
    tag.update(backend="mooncake", handle="hs-1", transfer_params={"handle": "hs-1", "version": 1})

    class Store:
        backend = "mooncake"
        def __init__(self):
            self.loads = 0
            self.deletes = 0
            self.closed = False
        def load(self, reference, *, timeout=120.0):
            # Loads receive the full transfer manifest; releases keep the
            # bare handle, matching the hs_connectors store contract.
            assert reference == {"handle": "hs-1", "version": 1}
            self.loads += 1
            return {"hidden_states": hidden, "token_ids": torch.arange(5)}, 0
        def release(self, handle):
            assert handle == "hs-1"
            self.deletes += 1
            if self.deletes == 1:
                raise RuntimeError("transient delete failure")
        def close(self):
            self.closed = True

    store = Store()
    monkeypatch.setattr(stores, "build_hidden_states_store", lambda config: store)
    config = {"run_id": "run", "ray": {"address": "local"}, "expected_feature": {}}
    adapter = build_feature_store_from_config(
        {"type": "ray_reference"}, read_only=True, transfer_queue_cfg=config,
        hidden_states_store_cfg={"backend": "mooncake", "mooncake": {"master_server_address": "host:50051"}},
    )
    assert adapter.config["hidden_states_store"]["backend"] == "mooncake"
    adapter.connect()
    entry = ReadyEntry(make_sample_key(SampleMetadata.from_dict(tag)), tag)
    actual = adapter.prepare_batch([entry], packing=packing, block_size=2, pin_memory=False)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value
    assert store.loads == 1 and store.deletes == 0
    assert cleanup_standalone_samples(adapter, [entry], attempts=2) == []
    assert store.loads == 1 and store.deletes == 2
    adapter.close()
    assert store.closed


def test_mooncake_batch_reads_share_one_timeout_budget(tmp_path, monkeypatch):
    import verl_speco.trainer.standalone_batch_prepare as prepare
    contract = _contract()
    tags = [_sample(tmp_path, contract, index=i)[0] for i in range(2)]
    hidden = torch.zeros(5, 3, 2)
    for i, tag in enumerate(tags):
        tag.pop("hidden_states_path")
        tag.update(backend="mooncake", handle=f"hs-{i}")
    now = [100.0]
    timeouts = []

    class Store:
        backend = "mooncake"
        def load(self, handle, *, timeout):
            timeouts.append(timeout)
            now[0] += 4
            return {"hidden_states": hidden, "token_ids": torch.arange(5)}, 0

    monkeypatch.setattr(prepare, "time", SimpleNamespace(monotonic=lambda: now[0]))
    prepare.prepare_standalone_cpu_batch(tags, contract, hidden_states_store=Store(), read_timeout=10, pin_memory=False)
    assert timeouts == [10, 6]
    now[0] = 100
    timeouts.clear()
    with pytest.raises(TimeoutError, match="batch read deadline"):
        prepare.prepare_standalone_cpu_batch(tags, contract, hidden_states_store=Store(), read_timeout=3, pin_memory=False)
    assert timeouts == [3]


@pytest.mark.parametrize("streaming", [False, True])
def test_file_batch_reads_share_one_timeout_budget(tmp_path, monkeypatch, streaming):
    import verl_speco.trainer.standalone_batch_prepare as prepare
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    contract = _contract()
    tags = [_sample(tmp_path, contract, length=5, index=i)[0] for i in range(2)]
    now, timeouts = [100.0], []
    method = "load_window" if streaming else "load"
    original = getattr(FileHiddenStatesStore, method)

    def load(self, *args, timeout, **kwargs):
        timeouts.append(timeout)
        now[0] += 4
        return original(self, *args, timeout=timeout, **kwargs)

    monkeypatch.setattr(prepare, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(FileHiddenStatesStore, method, load)
    options = dict(streaming=streaming, max_window=3, pin_memory=False)
    prepare.prepare_standalone_cpu_batch(tags, contract, read_timeout=10, **options)
    assert timeouts == [10, 6]
    now[0], timeouts[:] = 100, []
    with pytest.raises(TimeoutError, match="batch read deadline"):
        prepare.prepare_standalone_cpu_batch(tags, contract, read_timeout=3, **options)
    assert timeouts == [3]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "5"])
def test_reference_batch_read_timeout_must_be_positive_and_finite(tmp_path, timeout):
    contract = _contract()
    tag, _ = _sample(tmp_path, contract)
    with pytest.raises(ValueError, match="read_timeout"):
        prepare_standalone_cpu_batch([tag], contract, read_timeout=timeout, pin_memory=False)


@pytest.mark.parametrize("streaming", [False, True])
def test_file_lock_wait_obeys_batch_deadline(tmp_path, streaming):
    import os
    import time
    from pathlib import Path

    contract = _contract()
    tag, _ = _sample(tmp_path, contract, length=5)
    lock = Path(tag["hidden_states_path"] + ".lock")
    lock.touch()
    with lock.open("rb") as held:
        if os.name == "posix":
            import fcntl
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="hidden-state lock"):
            prepare_standalone_cpu_batch([tag], contract, streaming=streaming,
                                         max_window=3, read_timeout=0.05, pin_memory=False)
        assert time.monotonic() - started < 2
    assert Path(tag["hidden_states_path"]).exists()


@pytest.mark.parametrize("field,value", [
    ("input_ids", [False, 1, 2, 3, 4]),
    ("position_ids", [0.5, 1, 2, 3, 4]),
    ("loss_mask", [0, float("inf"), 1, 1, 1]),
    ("feature_positions", [0, 1, 2, 3, 99]),
    ("loss_mask", [1]),
])
def test_invalid_reference_rejected_before_any_hidden_read(tmp_path, field, value):
    contract = _contract()
    valid, _ = _sample(tmp_path, contract)
    invalid, _ = _sample(tmp_path, contract, index=1)
    invalid["request"][field] = value

    class UnreadStore:
        backend = "file"

        def load(self, *args, **kwargs):
            pytest.fail("all references must be validated before reading hidden data")

    with pytest.raises(ValueError):
        prepare_standalone_cpu_batch(
            [valid, invalid], contract, pin_memory=False, hidden_states_store=UnreadStore(),
        )


@pytest.mark.parametrize("algorithm,packing,limit,max_window", [
    ("DSPARK", False, 0, None), ("DSPARK", True, 0, None),
    ("DSPARK", True, 8, None), ("DFLASH", False, 0, 3),
    ("DFLASH", True, 8, 3), ("EAGLE3", False, 0, None),
])
def test_streamed_reference_batch_matches_existing_path(tmp_path, algorithm, packing, limit, max_window):
    contract = _contract(algorithm)
    tags = [_sample(tmp_path, contract, length=length, index=index)[0]
            for index, length in enumerate((5, 3, 6))]
    options = dict(packing=packing, block_size=4, max_packed_len=limit,
                   max_window=max_window, pin_memory=False)
    expected = prepare_standalone_cpu_batch(tags, contract, **options)
    actual = prepare_standalone_cpu_batch(tags, contract, streaming=True, **options)
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], equal_nan=True)
        else:
            assert actual[key] == expected[key]


def test_streamed_reference_batch_releases_each_intermediate(tmp_path, monkeypatch):
    import weakref
    import verl_speco.trainer.standalone_batch_prepare as prepare

    contract = _contract("DSPARK")
    tags = [_sample(tmp_path, contract, length=4, index=i)[0] for i in range(3)]
    original = prepare._prepare_chunk
    previous = []
    def tracked(*args, **kwargs):
        if previous:
            assert previous[-1]() is None
        result = original(*args, **kwargs)
        previous.append(weakref.ref(result["hidden_states"]))
        return result
    monkeypatch.setattr(prepare, "_prepare_chunk", tracked)
    batch = prepare_standalone_cpu_batch(tags, contract, streaming=True, pin_memory=False)
    assert batch["_speco_token_counts"]["samples"] == 3
    assert all(reference() is None for reference in previous)


def test_streamed_reference_rejects_sample_over_packing_limit(tmp_path):
    contract = _contract("DSPARK")
    tag, _ = _sample(tmp_path, contract, length=5)
    with pytest.raises(ValueError, match="exceeds max_packed_len"):
        prepare_standalone_cpu_batch([tag], contract, streaming=True, packing=True,
                                     block_size=4, max_packed_len=4, pin_memory=False)


@pytest.mark.parametrize("algorithm", ["DSPARK", "DFLASH"])
@pytest.mark.parametrize("positions,hidden_offset", [
    (list(range(12)), 0), ([2, 4, 6, 8, 10, 11], 2),
])
def test_streamed_file_window_reads_only_required_rows(tmp_path, monkeypatch, algorithm, positions, hidden_offset):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    contract = _contract(algorithm)
    tag, _ = _sample(tmp_path, contract, length=12, positions=positions, hidden_offset=hidden_offset)
    options = dict(max_window=3, pin_memory=False)
    expected = prepare_standalone_cpu_batch([tag], contract, **options)
    observed = []
    original = FileHiddenStatesStore.load_window
    def read_window(self, *args, **kwargs):
        payload, offset = original(self, *args, **kwargs)
        observed.append((payload["hidden_states"].size(0), offset))
        return payload, offset
    monkeypatch.setattr(FileHiddenStatesStore, "load_window", read_window)
    monkeypatch.setattr(FileHiddenStatesStore, "load", lambda *a, **kw: pytest.fail("whole tensor read"))
    actual = prepare_standalone_cpu_batch([tag], contract, streaming=True, **options)
    assert observed and observed[0][0] < 12 - hidden_offset
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize("algorithm,max_window", [("DSPARK", 4), ("DFLASH", 4), ("DSPARK", None), ("EAGLE3", None)])
def test_streamed_sparse_window_materializes_only_selected_rows(tmp_path, monkeypatch, algorithm, max_window):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    contract = _contract(algorithm)
    positions = [3, 12, 21, 30, 39, 48]
    tag, _ = _sample(tmp_path, contract, length=50, positions=positions, hidden_offset=3)
    options = dict(max_window=max_window, pin_memory=False)
    expected = prepare_standalone_cpu_batch([tag], contract, **options)
    observed = []
    original = FileHiddenStatesStore.load_window

    def read_window(self, *args, **kwargs):
        payload, rows = original(self, *args, **kwargs)
        observed.append((payload["hidden_states"].shape[0], rows))
        return payload, rows

    monkeypatch.setattr(FileHiddenStatesStore, "load_window", read_window)
    actual = prepare_standalone_cpu_batch([tag], contract, streaming=True, **options)
    selected = positions[:max_window] if max_window else positions
    assert observed == [(len(selected), selected)]
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value


def test_sparse_hidden_rows_must_match_request_positions(tmp_path):
    contract = _contract()
    tag, hidden = _sample(tmp_path, contract, length=5, positions=[0, 2, 4])
    request = dict(tag["request"])
    request["prompt_token_ids"] = request.pop("vllm_prompt_token_ids")
    for key in ("input_ids", "position_ids", "feature_positions", "draft_position_ids", "loss_mask"):
        request[key] = torch.tensor(request[key])
    payload = {"token_ids": torch.arange(5), "hidden_states": hidden[[0, 2, 4]]}
    with pytest.raises(HiddenStateAlignmentError, match="Explicit hidden rows"):
        feature_from_vllm_payload(payload, SimpleNamespace(**request), contract,
                                  normalize_final_hidden=False, split_hidden_states=True,
                                  hidden_row_positions=[0, 1, 4])


def test_fragmented_file_window_bounds_number_of_reads(tmp_path, monkeypatch):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore

    contract = _contract("DFLASH")
    positions = list(range(0, 4100, 4))
    tag, _ = _sample(tmp_path, contract, length=4100, positions=positions)
    expected = prepare_standalone_cpu_batch([tag], contract, pin_memory=False)
    reads = []
    original = FileHiddenStatesStore.load_window

    def observe(self, *args, **kwargs):
        payload, index = original(self, *args, **kwargs)
        reads.append(index)
        return payload, index

    monkeypatch.setattr(FileHiddenStatesStore, "load_window", observe)
    actual = prepare_standalone_cpu_batch([tag], contract, streaming=True, pin_memory=False)
    assert reads == [0]  # More than 1024 runs falls back to one contiguous read.
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("algorithm,packing,reuse_final", [
    ("DSPARK", False, False), ("DSPARK", True, False),
    ("DSPARK", True, True), ("DFLASH", False, False),
    ("DFLASH", True, False), ("EAGLE3", False, False),
])
def test_streaming_file_batch_budget_is_checked_before_materialization(tmp_path, monkeypatch, algorithm, packing, reuse_final):
    from verl_speco.producer.hidden_states_store import FileHiddenStatesStore
    from verl_speco.trainer.tq_sample_source import _prepared_host_bytes

    contract = _contract(algorithm, reuse_final=reuse_final)
    tags = [_sample(tmp_path, contract, length=length, index=i)[0] for i, length in enumerate([5, 3])]
    options = dict(streaming=True, packing=packing, block_size=4, max_packed_len=8, pin_memory=False)
    expected = prepare_standalone_cpu_batch(tags, contract, **options)
    budget = _prepared_host_bytes(expected)
    original = FileHiddenStatesStore.load_window
    called = []
    def load_window(self, *args, **kwargs):
        called.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(FileHiddenStatesStore, "load_window", load_window)
    with pytest.raises(ValueError, match="Planned reference batch exceeds"):
        prepare_standalone_cpu_batch(tags, contract, max_output_bytes=budget - 1, **options)
    assert not called
    actual = prepare_standalone_cpu_batch(tags, contract, max_output_bytes=budget, **options)
    assert len(called) == len(tags)
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value


def test_streamed_window_checks_full_feature_coverage(tmp_path):
    from verl_speco.trainer.target_feature_replay import HiddenStateAlignmentError

    contract = _contract("DFLASH")
    tag, _ = _sample(tmp_path, contract, length=10, positions=list(range(10)), hidden_offset=5)
    with pytest.raises(HiddenStateAlignmentError, match="complete feature window"):
        prepare_standalone_cpu_batch([tag], contract, streaming=True, max_window=2, pin_memory=False)


@pytest.mark.parametrize("algorithm", ["DSPARK", "DFLASH"])
def test_streamed_window_matches_full_read_for_varied_masks_and_offsets(tmp_path, algorithm):
    import random

    rng = random.Random(17)
    contract = _contract(algorithm)
    for index in range(12):
        size = rng.randrange(7, 18)
        offset = rng.randrange(0, 3)
        positions = sorted(rng.sample(range(offset, size), k=rng.randrange(4, size - offset + 1)))
        tag, _ = _sample(tmp_path, contract, length=size, index=index, positions=positions, hidden_offset=offset)
        tag["request"]["loss_mask"] = [rng.randrange(2) for _ in range(size)]
        window = rng.randrange(1, len(positions))
        options = dict(max_window=window, pin_memory=False)
        expected = prepare_standalone_cpu_batch([tag], contract, **options)
        actual = prepare_standalone_cpu_batch([tag], contract, streaming=True, **options)
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]
